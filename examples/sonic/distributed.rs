//! One-process-per-GPU synchronization for the SONIC trainer.
//!
//! NCCL owns the large, device-resident gradient collectives. A deliberately
//! small TCP control plane bootstraps NCCL's unique ID and reduces the handful
//! of CPU statistics PPO needs to keep identical on every rank.

use crate::gpu_ppo::PpoSynchronizer;
use anyhow::{Context, Result, bail, ensure};
use std::env;
use vortx::tensor::Tensor;
use zealot_rl::ppo::PendingNorm;

#[cfg(feature = "nccl")]
use cudarc::nccl::{Comm, Id, ReduceOp};
#[cfg(not(feature = "nccl"))]
use khal::backend::GpuBackend;
#[cfg(feature = "nccl")]
use khal::backend::{GpuBackend, GpuBuffer};

#[derive(Clone, Debug)]
pub struct DistributedConfig {
    rank: usize,
    local_rank: usize,
    world_size: usize,
    master_addr: String,
    master_port: u16,
}

impl DistributedConfig {
    pub fn from_env() -> Result<Self> {
        let world_size = parse_env("WORLD_SIZE")?.unwrap_or(1);
        let rank = parse_env("RANK")?.unwrap_or(0);
        let local_rank = parse_env("LOCAL_RANK")?.unwrap_or(rank);
        let master_addr = env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".into());
        let master_port = parse_env("MASTER_PORT")?.unwrap_or(29_500);
        ensure!(world_size > 0, "WORLD_SIZE must be positive");
        ensure!(
            rank < world_size,
            "RANK={rank} must be smaller than WORLD_SIZE={world_size}"
        );
        #[cfg(not(feature = "nccl"))]
        if world_size > 1 {
            bail!(
                "WORLD_SIZE={world_size} requests distributed training; rebuild sonic_train_gpu \
                 with --features \"gpu biped_gpu cuda_backend nccl\""
            );
        }
        Ok(Self {
            rank,
            local_rank,
            world_size,
            master_addr,
            master_port,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn is_primary(&self) -> bool {
        self.rank == 0
    }

    /// Make khal's fixed `Cuda::new(0)` select this process's local GPU.
    ///
    /// Job schedulers sometimes expose one device per process and sometimes
    /// expose a comma-separated node-wide list. Preserve the former; select
    /// LOCAL_RANK's entry from the latter; otherwise use LOCAL_RANK directly.
    pub fn configure_cuda_visibility(&self) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }
        #[cfg(feature = "nccl")]
        if !unsafe { cudarc::nccl::sys::is_culib_present() } {
            bail!(
                "libnccl was not found; install NCCL 2.22+ or add its lib directory to \
                 LD_LIBRARY_PATH"
            );
        }
        let selected = match env::var("CUDA_VISIBLE_DEVICES") {
            Ok(visible) => {
                let devices: Vec<_> = visible
                    .split(',')
                    .map(str::trim)
                    .filter(|device| !device.is_empty())
                    .collect();
                ensure!(
                    !devices.is_empty(),
                    "CUDA_VISIBLE_DEVICES is set but contains no devices"
                );
                if devices.len() == 1 {
                    devices[0].to_owned()
                } else {
                    devices
                        .get(self.local_rank)
                        .with_context(|| {
                            format!(
                                "LOCAL_RANK={} has no entry in CUDA_VISIBLE_DEVICES={visible:?}",
                                self.local_rank
                            )
                        })?
                        .to_string()
                }
            }
            Err(_) => self.local_rank.to_string(),
        };

        // Safety: this runs at the very beginning of main, before pollster,
        // rayon, CUDA, or any other worker thread is started.
        unsafe {
            env::set_var("CUDA_VISIBLE_DEVICES", &selected);
            env::set_var("KHAL_BACKEND", "cuda");
        }
        Ok(())
    }
}

fn parse_env<T>(name: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .map_err(|error| anyhow::anyhow!("parse {name}={value:?}: {error}"))
        })
        .transpose()
}

pub struct Distributed {
    config: DistributedConfig,
    #[cfg(feature = "nccl")]
    nccl: Option<NcclState>,
}

#[cfg(feature = "nccl")]
struct NcclState {
    communicator: Comm,
    control: ControlPlane,
}

impl Distributed {
    pub fn initialize(config: DistributedConfig, backend: &GpuBackend) -> Result<Self> {
        if config.world_size == 1 {
            return Ok(Self {
                config,
                #[cfg(feature = "nccl")]
                nccl: None,
            });
        }

        #[cfg(not(feature = "nccl"))]
        {
            let _ = backend;
            bail!(
                "WORLD_SIZE={} requests distributed training; rebuild sonic_train_gpu with \
                 --features \"gpu biped_gpu cuda_backend nccl\"",
                config.world_size
            );
        }

        #[cfg(feature = "nccl")]
        {
            let cuda = match backend {
                GpuBackend::Cuda(cuda) => cuda,
                _ => bail!("distributed SONIC training requires khal's native CUDA backend"),
            };
            // cudarc dynamically loads NCCL, so builds do not require an NCCL
            // SDK. Surface a useful error before its loader would panic.
            if !unsafe { cudarc::nccl::sys::is_culib_present() } {
                bail!(
                    "libnccl was not found; install NCCL 2.22+ or add its lib directory to \
                     LD_LIBRARY_PATH"
                );
            }

            let root_id = if config.is_primary() {
                Some(
                    Id::new()
                        .map_err(|error| anyhow::anyhow!("create NCCL unique ID: {error:?}"))?,
                )
            } else {
                None
            };
            let root_bytes = root_id.map(id_bytes);
            let (control, bytes) = ControlPlane::bootstrap(&config, root_bytes)?;
            let id = Id::uninit(bytes.map(|byte| byte as std::ffi::c_char));
            let communicator =
                Comm::from_rank(cuda.stream().clone(), config.rank, config.world_size, id)
                    .map_err(|error| anyhow::anyhow!("initialize NCCL communicator: {error:?}"))?;
            Ok(Self {
                config,
                nccl: Some(NcclState {
                    communicator,
                    control,
                }),
            })
        }
    }

    pub fn rank(&self) -> usize {
        self.config.rank
    }

    pub fn world_size(&self) -> usize {
        self.config.world_size
    }

    pub fn is_primary(&self) -> bool {
        self.config.is_primary()
    }

    /// Merge per-rank pending Welford accumulators into the same global state.
    pub fn synchronize_pending_norm(
        &mut self,
        pending: &mut PendingNorm,
        dimension: usize,
    ) -> Result<()> {
        if self.world_size() == 1 {
            return Ok(());
        }
        let (mean, m2, count) = pending.state();
        ensure!(
            mean.is_empty() || mean.len() == dimension,
            "pending normalizer dimension mismatch: {} != {dimension}",
            mean.len()
        );
        let count = count as f64;
        let mut moments = vec![0.0f64; 1 + 2 * dimension];
        moments[0] = count;
        for index in 0..dimension {
            let mean = mean.get(index).copied().unwrap_or(0.0) as f64;
            let m2 = m2.get(index).copied().unwrap_or(0.0) as f64;
            moments[1 + index] = mean * count;
            moments[1 + dimension + index] = m2 + mean * mean * count;
        }
        self.all_reduce_sum_f64(&mut moments)?;
        let global_count = moments[0];
        ensure!(
            global_count > 0.0,
            "cannot synchronize an empty pending normalizer"
        );
        let mut global_mean = vec![0.0f32; dimension];
        let mut global_m2 = vec![0.0f32; dimension];
        for index in 0..dimension {
            let sum = moments[1 + index];
            let sum_squares = moments[1 + dimension + index];
            let mean = sum / global_count;
            global_mean[index] = mean as f32;
            global_m2[index] = (sum_squares - sum * mean).max(0.0) as f32;
        }
        *pending = PendingNorm::from_state(global_mean, global_m2, global_count as f32);
        Ok(())
    }

    pub fn sum_metrics(&mut self, metrics: &mut [f64]) -> Result<()> {
        self.all_reduce_sum_f64(metrics)
    }
}

impl PpoSynchronizer for Distributed {
    fn world_size(&self) -> usize {
        self.config.world_size
    }

    fn average_gradient(&mut self, gradient: &mut Tensor<f32>) -> Result<()> {
        if self.config.world_size == 1 {
            return Ok(());
        }

        #[cfg(not(feature = "nccl"))]
        {
            let _ = gradient;
            unreachable!("multi-rank initialization fails without the nccl feature");
        }

        #[cfg(feature = "nccl")]
        {
            let state = self
                .nccl
                .as_ref()
                .context("NCCL state missing for a multi-rank run")?;
            let buffer = match gradient.buffer_mut() {
                GpuBuffer::Cuda(buffer) => buffer,
                _ => bail!("NCCL gradient is not backed by a CUDA buffer"),
            };
            let elements = buffer.byte_len() as usize / std::mem::size_of::<f32>();
            if elements == 0 {
                return Ok(());
            }
            let bytes = buffer
                .inner_mut()
                .context("non-empty CUDA tensor has no allocation")?;
            // Safety: Tensor<f32> allocations are f32-aligned and byte_len is
            // exactly a multiple of size_of::<f32>(); the view cannot outlive
            // the exclusive borrow of the underlying CUDA allocation.
            let mut view = unsafe { bytes.transmute_mut::<f32>(elements) }
                .context("reinterpret CUDA gradient as f32")?;
            state
                .communicator
                .all_reduce_in_place(&mut view, &ReduceOp::Avg)
                .map_err(|error| anyhow::anyhow!("NCCL gradient all-reduce: {error:?}"))?;
            Ok(())
        }
    }

    fn all_reduce_sum_f64(&mut self, values: &mut [f64]) -> Result<()> {
        if self.config.world_size == 1 {
            return Ok(());
        }

        #[cfg(not(feature = "nccl"))]
        {
            let _ = values;
            unreachable!("multi-rank initialization fails without the nccl feature");
        }

        #[cfg(feature = "nccl")]
        self.nccl
            .as_mut()
            .context("NCCL state missing for a multi-rank run")?
            .control
            .all_reduce_sum(values)
    }
}

#[cfg(feature = "nccl")]
fn id_bytes(id: Id) -> [u8; 128] {
    id.internal().map(|byte| byte as u8)
}

#[cfg(feature = "nccl")]
struct ControlPlane {
    sequence: u64,
    role: ControlRole,
}

#[cfg(feature = "nccl")]
enum ControlRole {
    Root(Vec<(usize, std::net::TcpStream)>),
    Worker(std::net::TcpStream),
}

#[cfg(feature = "nccl")]
impl ControlPlane {
    const HANDSHAKE: [u8; 8] = *b"ZLNC0001";
    const COLLECTIVE: [u8; 8] = *b"ZCOL0001";

    fn bootstrap(
        config: &DistributedConfig,
        root_id: Option<[u8; 128]>,
    ) -> Result<(Self, [u8; 128])> {
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream, ToSocketAddrs};
        use std::time::{Duration, Instant};

        let address = format!("{}:{}", config.master_addr, config.master_port);
        if config.is_primary() {
            let id = root_id.context("rank 0 did not create an NCCL ID")?;
            let listener = TcpListener::bind(&address)
                .with_context(|| format!("bind distributed rendezvous at {address}"))?;
            let mut workers = Vec::with_capacity(config.world_size - 1);
            let mut seen = vec![false; config.world_size];
            seen[0] = true;
            while workers.len() + 1 < config.world_size {
                let (mut stream, peer) = listener.accept().context("accept distributed rank")?;
                stream.set_nodelay(true)?;
                let mut hello = [0u8; 24];
                stream
                    .read_exact(&mut hello)
                    .with_context(|| format!("read handshake from {peer}"))?;
                ensure!(
                    hello[..8] == Self::HANDSHAKE,
                    "invalid handshake from {peer}"
                );
                let rank = decode_u64(&hello[8..16]) as usize;
                let world_size = decode_u64(&hello[16..24]) as usize;
                ensure!(
                    world_size == config.world_size,
                    "rank {rank} reports WORLD_SIZE={world_size}, expected {}",
                    config.world_size
                );
                ensure!(
                    rank > 0 && rank < config.world_size,
                    "invalid worker RANK={rank}"
                );
                ensure!(!seen[rank], "duplicate connection from RANK={rank}");
                seen[rank] = true;
                stream.write_all(&Self::HANDSHAKE)?;
                stream.write_all(&id)?;
                stream.flush()?;
                workers.push((rank, stream));
            }
            workers.sort_unstable_by_key(|(rank, _)| *rank);
            Ok((
                Self {
                    sequence: 0,
                    role: ControlRole::Root(workers),
                },
                id,
            ))
        } else {
            ensure!(
                root_id.is_none(),
                "worker rank unexpectedly owns an NCCL ID"
            );
            let socket_addresses: Vec<_> = address
                .to_socket_addrs()
                .with_context(|| format!("resolve MASTER_ADDR at {address}"))?
                .collect();
            ensure!(
                !socket_addresses.is_empty(),
                "MASTER_ADDR resolved no addresses"
            );
            let deadline = Instant::now() + Duration::from_secs(120);
            let mut last_error = None;
            let mut connected = None;
            while Instant::now() < deadline && connected.is_none() {
                for socket_address in &socket_addresses {
                    match TcpStream::connect_timeout(socket_address, Duration::from_secs(2)) {
                        Ok(stream) => {
                            connected = Some(stream);
                            break;
                        }
                        Err(error) => last_error = Some(error),
                    }
                }
                if connected.is_none() {
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
            let mut stream = connected.with_context(|| {
                format!(
                    "connect to distributed rendezvous at {address}: {}",
                    last_error
                        .map(|error| error.to_string())
                        .unwrap_or_else(|| "timeout".into())
                )
            })?;
            stream.set_nodelay(true)?;
            let mut hello = Vec::with_capacity(24);
            hello.extend_from_slice(&Self::HANDSHAKE);
            hello.extend_from_slice(&(config.rank as u64).to_le_bytes());
            hello.extend_from_slice(&(config.world_size as u64).to_le_bytes());
            stream.write_all(&hello)?;
            stream.flush()?;
            let mut response = [0u8; 8];
            stream.read_exact(&mut response)?;
            ensure!(response == Self::HANDSHAKE, "invalid rendezvous response");
            let mut id = [0u8; 128];
            stream.read_exact(&mut id)?;
            Ok((
                Self {
                    sequence: 0,
                    role: ControlRole::Worker(stream),
                },
                id,
            ))
        }
    }

    fn all_reduce_sum(&mut self, values: &mut [f64]) -> Result<()> {
        use std::io::{Read, Write};

        let sequence = self.sequence;
        self.sequence += 1;
        let header = collective_header(sequence, values.len());
        match &mut self.role {
            ControlRole::Root(workers) => {
                for (rank, stream) in workers.iter_mut() {
                    let mut peer_header = [0u8; 24];
                    stream
                        .read_exact(&mut peer_header)
                        .with_context(|| format!("read collective {sequence} from rank {rank}"))?;
                    validate_collective_header(&peer_header, sequence, values.len())
                        .with_context(|| format!("rank {rank} collective mismatch"))?;
                    for value in values.iter_mut() {
                        let mut bytes = [0u8; 8];
                        stream.read_exact(&mut bytes)?;
                        *value += f64::from_le_bytes(bytes);
                    }
                }
                for (_, stream) in workers.iter_mut() {
                    stream.write_all(&header)?;
                    for value in values.iter() {
                        stream.write_all(&value.to_le_bytes())?;
                    }
                    stream.flush()?;
                }
            }
            ControlRole::Worker(stream) => {
                stream.write_all(&header)?;
                for value in values.iter() {
                    stream.write_all(&value.to_le_bytes())?;
                }
                stream.flush()?;
                let mut root_header = [0u8; 24];
                stream.read_exact(&mut root_header)?;
                validate_collective_header(&root_header, sequence, values.len())?;
                for value in values.iter_mut() {
                    let mut bytes = [0u8; 8];
                    stream.read_exact(&mut bytes)?;
                    *value = f64::from_le_bytes(bytes);
                }
            }
        }
        Ok(())
    }
}

#[cfg(feature = "nccl")]
fn collective_header(sequence: u64, len: usize) -> [u8; 24] {
    let mut header = [0u8; 24];
    header[..8].copy_from_slice(&ControlPlane::COLLECTIVE);
    header[8..16].copy_from_slice(&sequence.to_le_bytes());
    header[16..24].copy_from_slice(&(len as u64).to_le_bytes());
    header
}

#[cfg(feature = "nccl")]
fn validate_collective_header(header: &[u8; 24], sequence: u64, len: usize) -> Result<()> {
    ensure!(
        header[..8] == ControlPlane::COLLECTIVE,
        "invalid collective magic"
    );
    ensure!(
        decode_u64(&header[8..16]) == sequence,
        "collective sequence differs"
    );
    ensure!(
        decode_u64(&header[16..24]) == len as u64,
        "collective length differs"
    );
    Ok(())
}

#[cfg(feature = "nccl")]
fn decode_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("u64 field has eight bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_config_defaults() {
        // The process running the workspace tests normally has no distributed
        // launcher environment. Test the invariant without mutating global env.
        let config = DistributedConfig {
            rank: 0,
            local_rank: 0,
            world_size: 1,
            master_addr: "127.0.0.1".into(),
            master_port: 29_500,
        };
        assert!(config.is_primary());
        assert_eq!(config.world_size(), 1);
    }

    #[cfg(feature = "nccl")]
    #[test]
    fn control_plane_bootstraps_and_reduces() {
        let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = reservation.local_addr().unwrap().port();
        drop(reservation);
        let root = DistributedConfig {
            rank: 0,
            local_rank: 0,
            world_size: 2,
            master_addr: "127.0.0.1".into(),
            master_port: port,
        };
        let worker = DistributedConfig {
            rank: 1,
            local_rank: 1,
            ..root.clone()
        };
        let root_thread = std::thread::spawn(move || -> Result<_> {
            let (mut control, id) = ControlPlane::bootstrap(&root, Some([7; 128]))?;
            let mut values = [1.0, 3.0];
            control.all_reduce_sum(&mut values)?;
            Ok((id, values))
        });
        let (mut worker_control, worker_id) =
            ControlPlane::bootstrap(&worker, None).expect("worker bootstrap");
        let mut worker_values = [2.0, 4.0];
        worker_control
            .all_reduce_sum(&mut worker_values)
            .expect("worker all-reduce");
        let (root_id, root_values) = root_thread.join().unwrap().unwrap();
        assert_eq!(root_id, [7; 128]);
        assert_eq!(worker_id, root_id);
        assert_eq!(root_values, [3.0, 7.0]);
        assert_eq!(worker_values, root_values);
    }
}
