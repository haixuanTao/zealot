//! GPU-resident batched forward for the PPO actor/critic, on vortx.
//!
//! Stage A of the GPU-policy port: the rollout's per-env CPU forward loop
//! (`for e in 0..N { actor.mean(); critic.value() }`) is the bottleneck — at
//! biped scale (N=4096) it's ~180 us/env. This replaces it with one batched
//! GEMM-stack per net (GEMM -> bias -> ELU, linear output), running on the SAME
//! backend as the nexus physics. The `policy_forward_bench` example measured
//! ~32x for exactly this swap, output matching the CPU net to ~1e-7.
//!
//! Only the forward moves to GPU. Sampling, log-prob, the running normalizers,
//! and the PPO update stay on the CPU `ActorCritic` (Stage B would move the
//! update too). After each `ac.update()` the weights change, so call
//! [`GpuPolicy::sync_weights`] once per PPO iteration to re-upload them.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend};
use nalgebra::DMatrix;
use vortx::linalg::{Activation, Gemm, OpAssign, OpAssignVariant};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use zealot_env::robots::lerobot_bipedal::NUM_JOINTS;
use zealot_rl::ActorCritic;
use zealot_rl::net::Mlp;

/// Upload an nalgebra matrix to a GPU tensor (panics on allocation failure —
/// these are fixed-size, built once or once-per-iteration).
fn matrix(backend: &GpuBackend, m: &DMatrix<f32>, usage: BufferUsages) -> Tensor<f32> {
    Tensor::matrix_from_na(backend, m, usage).expect("matrix_from_na")
}

/// One net's GPU-resident parameters + activation buffers for a fixed batch `n`.
struct GpuNet {
    /// Per-layer weight `[out x in]`.
    w: Vec<Tensor<f32>>,
    /// Per-layer bias pre-broadcast to `[out x n]` (so the add is same-shape).
    b: Vec<Tensor<f32>>,
    /// Activation buffers: `a[0]` = input `[in x n]`, `a[l]` = layer-l output.
    a: Vec<Tensor<f32>>,
    dims: Vec<usize>,
    /// Batch width (number of envs / columns).
    n: usize,
}

impl GpuNet {
    fn new(backend: &GpuBackend, net: &Mlp, n: usize) -> Self {
        let dims = net.dims.clone();
        let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let mut a = Vec::with_capacity(net.w.len() + 1);
        for l in 0..=net.w.len() {
            // a[0] is overwritten in place each step via write_buffer (needs
            // COPY_DST); a[1..] are copied out / read back (needs COPY_SRC).
            let usage = if l == 0 {
                BufferUsages::STORAGE | BufferUsages::COPY_DST
            } else {
                rw
            };
            a.push(matrix(backend, &DMatrix::<f32>::zeros(dims[l], n), usage));
        }
        let mut me = Self {
            w: Vec::new(),
            b: Vec::new(),
            a,
            dims,
            n,
        };
        me.sync(backend, net);
        me
    }

    /// (Re)upload weights and biases from the CPU net. Call after each PPO update.
    fn sync(&mut self, backend: &GpuBackend, net: &Mlp) {
        let st = BufferUsages::STORAGE;
        let n = self.n;
        self.w.clear();
        self.b.clear();
        for l in 0..net.w.len() {
            let (out, inp) = (self.dims[l + 1], self.dims[l]);
            // net.w[l] is row-major [out x in]; matches mlp_forward's recipe.
            let wm = DMatrix::from_fn(out, inp, |r, c| net.w[l][r * inp + c]);
            self.w.push(matrix(backend, &wm, st));
            let bm = DMatrix::from_fn(out, n, |r, _| net.b[l][r]);
            self.b.push(matrix(backend, &bm, st));
        }
    }

    /// Overwrite the persistent `a[0]` input buffer in place. `x` is `[in x n]`;
    /// `matrix_from_na` stores row-major, so we write `x.transpose()`'s slice to
    /// match that layout — no fresh GPU allocation per rollout step.
    fn set_input(&mut self, backend: &GpuBackend, x: &DMatrix<f32>) {
        backend
            .write_buffer(self.a[0].buffer_mut(), 0, x.transpose().as_slice())
            .expect("write a[0]");
    }

    /// Encode GEMM -> bias -> ELU per hidden layer (linear output) into `enc`.
    fn encode(
        &mut self,
        backend: &GpuBackend,
        ops: &Ops,
        shapes: &mut TensorLayoutBuffers,
        enc: &mut <GpuBackend as Backend>::Encoder,
    ) -> anyhow::Result<()> {
        let layers = self.w.len();
        for l in 0..layers {
            let (left, right) = self.a.split_at_mut(l + 1);
            let a_in = &left[l];
            let a_out = &mut right[0];
            {
                let mut p = enc.begin_pass("gemm", None);
                ops.gemm
                    .dispatch_tiled(backend, shapes, &mut p, &mut *a_out, &self.w[l], a_in)?;
            }
            {
                let mut p = enc.begin_pass("bias", None);
                ops.op.launch(
                    backend,
                    shapes,
                    &mut p,
                    OpAssignVariant::Add,
                    &mut *a_out,
                    &self.b[l],
                )?;
            }
            if l < layers - 1 {
                let mut p = enc.begin_pass("elu", None);
                ops.act.elu(backend, shapes, &mut p, &mut *a_out)?;
            }
        }
        Ok(())
    }

    fn output(&self) -> &Tensor<f32> {
        self.a.last().unwrap()
    }
}

/// vortx op handles (cheap to hold, created once from the backend).
struct Ops {
    gemm: Gemm,
    op: OpAssign,
    act: Activation,
}

/// GPU-resident actor + critic, batched over a fixed number of envs.
pub struct GpuPolicy {
    actor: GpuNet,
    critic: GpuNet,
    ops: Ops,
    shapes: TensorLayoutBuffers,
    n: usize,
}

impl GpuPolicy {
    /// Build from a CPU `ActorCritic`, mirroring its weights onto `backend` and
    /// sizing the activation buffers for `n` envs.
    pub fn new(backend: &GpuBackend, ac: &ActorCritic, n: usize) -> anyhow::Result<Self> {
        Ok(Self {
            actor: GpuNet::new(backend, &ac.actor, n),
            critic: GpuNet::new(backend, &ac.critic, n),
            ops: Ops {
                gemm: Gemm::from_backend(backend)?,
                op: OpAssign::from_backend(backend)?,
                act: Activation::from_backend(backend)?,
            },
            shapes: TensorLayoutBuffers::new(backend),
            n,
        })
    }

    /// Re-upload weights from `ac` after a PPO update mutated them.
    pub fn sync_weights(&mut self, backend: &GpuBackend, ac: &ActorCritic) {
        self.actor.sync(backend, &ac.actor);
        self.critic.sync(backend, &ac.critic);
    }

    /// Batched forward for all `n` envs. `cur` / `cur_c` are the *raw* per-env
    /// policy / critic observations; normalization uses `ac`'s running stats
    /// (matching `ActorCritic::mean` / `value`). Returns `(means, values)` with
    /// one entry per env.
    pub async fn forward(
        &mut self,
        backend: &GpuBackend,
        ac: &ActorCritic,
        cur: &[Vec<f32>],
        cur_c: &[Vec<f32>],
    ) -> anyhow::Result<(Vec<[f32; NUM_JOINTS]>, Vec<f32>)> {
        let n = self.n;
        debug_assert_eq!(cur.len(), n);
        let (obs_dim, crit_dim) = (self.actor.dims[0], self.critic.dims[0]);

        // Normalize on CPU (cheap, O(n·dim)) then pack column-major-by-env.
        let obs_norm: Vec<Vec<f32>> = cur.iter().map(|o| ac.obs_norm.normalize(o)).collect();
        let crit_norm: Vec<Vec<f32>> = cur_c.iter().map(|o| ac.critic_norm.normalize(o)).collect();
        let obs_m = DMatrix::from_fn(obs_dim, n, |r, c| obs_norm[c][r]);
        let crit_m = DMatrix::from_fn(crit_dim, n, |r, c| crit_norm[c][r]);
        self.actor.set_input(backend, &obs_m);
        self.critic.set_input(backend, &crit_m);

        let mut enc = backend.begin_encoding();
        self.actor.encode(backend, &self.ops, &mut self.shapes, &mut enc)?;
        self.critic.encode(backend, &self.ops, &mut self.shapes, &mut enc)?;
        backend.submit(enc)?;
        // No explicit synchronize(): the slow_read_vec below copies-to-staging
        // (ordered after this submit) and maps, which drains the queue anyway.
        // An extra synchronize() here is just a redundant device poll.

        // Outputs are row-major [out x n] -> element (r, e) at index r*n + e.
        let a_out = backend.slow_read_vec(self.actor.output().buffer()).await?;
        let c_out = backend.slow_read_vec(self.critic.output().buffer()).await?;
        let mut means = vec![[0f32; NUM_JOINTS]; n];
        for e in 0..n {
            for r in 0..NUM_JOINTS {
                means[e][r] = a_out[r * n + e];
            }
        }
        let values: Vec<f32> = (0..n).map(|e| c_out[e]).collect();
        Ok((means, values))
    }
}
