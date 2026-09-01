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

use crate::cutile_gemm::{CutileGemm, EncCursor};
use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend};
use nalgebra::DMatrix;
use rayon::prelude::*;
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
        let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let mut a = Vec::with_capacity(net.w.len() + 1);
        for l in 0..=net.w.len() {
            a.push(matrix(backend, &DMatrix::<f32>::zeros(dims[l], n), rw));
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
        let st = BufferUsages::STORAGE | BufferUsages::COPY_DST;
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

    /// Upload the input matrix `[in x n]` into `a[0]` (in place: the buffer is
    /// allocated once in `new`, so its device pointer stays stable — required
    /// by the cuTile view cache, and avoids a per-step allocation).
    fn set_input(&mut self, backend: &GpuBackend, x: &DMatrix<f32>) {
        // Row-major flatten (vortx layout; DMatrix is column-major).
        let (rows, cols) = (x.nrows(), x.ncols());
        let mut flat = vec![0f32; rows * cols];
        for c in 0..cols {
            for r in 0..rows {
                flat[r * cols + c] = x[(r, c)];
            }
        }
        backend
            .write_buffer(self.a[0].buffer_mut(), 0, &flat)
            .expect("write policy input");
    }

    /// Encode GEMM -> bias -> ELU per hidden layer (linear output). With a
    /// cuTile adapter, each layer is ONE fused tf32 launch (gemm + bias
    /// broadcast + ELU) on khal's stream; the bias is read as column 0 of the
    /// pre-broadcast `[out x n]` tensor (row stride n).
    fn encode(
        &mut self,
        backend: &GpuBackend,
        ops: &Ops,
        shapes: &mut TensorLayoutBuffers,
        cur: &mut EncCursor,
        ct: Option<&CutileGemm>,
    ) -> anyhow::Result<()> {
        let layers = self.w.len();
        for l in 0..layers {
            let (left, right) = self.a.split_at_mut(l + 1);
            let a_in = &left[l];
            let a_out = &mut right[0];
            if let Some(ct) = ct {
                cur.flush();
                ct.gemm_bias_act(
                    a_out,
                    &self.w[l],
                    a_in,
                    self.dims[l + 1],
                    self.n,
                    self.dims[l],
                    &self.b[l],
                    self.n,
                    l < layers - 1,
                )?;
                continue;
            }
            {
                let mut p = cur.pass("gemm");
                ops.gemm
                    .dispatch_naive(backend, shapes, &mut p, &mut *a_out, &self.w[l], a_in)?;
            }
            {
                let mut p = cur.pass("bias");
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
                let mut p = cur.pass("elu");
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
/// Constant-per-iteration buffers the staging kernel binds: the normalizer's
/// per-feature affine and the mirror signed-permutation tables.
pub struct NormBufs {
    pub mean_o: Tensor<f32>,
    pub inv_o: Tensor<f32>,
    pub mean_c: Tensor<f32>,
    pub inv_c: Tensor<f32>,
    pub perm_o: Tensor<u32>,
    pub sign_o: Tensor<f32>,
    pub perm_c: Tensor<u32>,
    pub sign_c: Tensor<f32>,
}

pub struct GpuPolicy {
    actor: GpuNet,
    critic: GpuNet,
    ops: Ops,
    shapes: TensorLayoutBuffers,
    n: usize,
    /// cuTile tf32 fused-forward adapter (BIPED_CUTILE_GEMM=1); None = vortx.
    ct: Option<&'static CutileGemm>,
    /// Reusable row-major staging for the per-step policy inputs, so a rollout
    /// step allocates nothing.
    scratch_obs: Vec<f32>,
    scratch_cobs: Vec<f32>,
    /// Same, holding the UNNORMALIZED values destined for the raw batch.
    scratch_raw_obs: Vec<f32>,
    scratch_raw_cobs: Vec<f32>,
    /// Staging kernel + per-step params, used to derive the normalized policy
    /// inputs from the raw batch on device (so obs upload once, not twice).
    stage: vortx::linalg::Ppo,
    stage_p_obs: Option<Tensor<vortx::linalg::PpoStageParams>>,
    stage_p_cobs: Option<Tensor<vortx::linalg::PpoStageParams>>,
    /// Identity signed-permutation tables (the new stage kernel always
    /// applies `perm`; the per-step policy-input staging must not mirror).
    id_perm_o: Option<Tensor<u32>>,
    id_sign_o: Option<Tensor<f32>>,
    id_perm_c: Option<Tensor<u32>>,
    id_sign_c: Option<Tensor<f32>>,
    /// Normalizer affine + (unused here) mirror tables, refreshed per iteration.
    norm: Option<NormBufs>,
    /// Step-blocked `[T][dim][n]` raw rollout observations, retained on device
    /// so the PPO update can build its batch there instead of on the host.
    /// `None` until `init_raw_batch`.
    raw_obs: Option<Tensor<f32>>,
    raw_cobs: Option<Tensor<f32>>,
}

impl GpuPolicy {
    /// Build from a CPU `ActorCritic`, mirroring its weights onto `backend` and
    /// sizing the activation buffers for `n` envs.
    pub fn new(backend: &GpuBackend, ac: &ActorCritic, n: usize) -> anyhow::Result<Self> {
        Ok(Self {
            scratch_obs: vec![0.0; ac.actor.dims[0] * n],
            scratch_cobs: vec![0.0; ac.critic.dims[0] * n],
            scratch_raw_obs: vec![0.0; ac.actor.dims[0] * n],
            scratch_raw_cobs: vec![0.0; ac.critic.dims[0] * n],
            stage: vortx::linalg::Ppo::from_backend(backend)?,
            stage_p_obs: None,
            stage_p_cobs: None,
            id_perm_o: None,
            id_sign_o: None,
            id_perm_c: None,
            id_sign_c: None,
            norm: None,
            raw_obs: None,
            raw_cobs: None,
            actor: GpuNet::new(backend, &ac.actor, n),
            critic: GpuNet::new(backend, &ac.critic, n),
            ops: Ops {
                gemm: Gemm::from_backend(backend)?,
                op: OpAssign::from_backend(backend)?,
                act: Activation::from_backend(backend)?,
            },
            shapes: TensorLayoutBuffers::new(backend),
            n,
            ct: None,
        })
    }

    /// Route the per-layer forward through the cuTile tf32 fused kernels.
    pub fn set_cutile(&mut self, ct: Option<&'static CutileGemm>) {
        self.ct = ct;
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
        step: usize,
    ) -> anyhow::Result<(Vec<[f32; NUM_JOINTS]>, Vec<f32>)> {
        let n = self.n;
        debug_assert_eq!(cur.len(), n);
        let (obs_dim, crit_dim) = (self.actor.dims[0], self.critic.dims[0]);

        // Normalize straight into vortx's ROW-MAJOR input layout in one pass.
        //
        // The previous spelling — a `Vec` per env from `normalize`, then a
        // column-major `DMatrix`, then `set_input`'s row-major re-flatten —
        // walked ~n·dim floats three times and allocated twice, every rollout
        // step. The normalizer is a per-feature affine, so hoisting (mean, 1/σ)
        // lets one parallel pass over ROWS write the buffer directly.
        // Transpose the RAW observations into row-major and upload ONCE. The
        // normalized policy input is then derived on device by the same staging
        // kernel that builds the PPO batch — uploading both forms cost 258 MB
        // per iteration at N=8192 to send the same numbers twice.
        fn fill_raw(dst: &mut [f32], dim: usize, n: usize, src: &[Vec<f32>]) {
            dst[..dim * n]
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(r, row)| {
                    for (e, d) in row.iter_mut().enumerate() {
                        *d = src[e][r];
                    }
                });
        }
        let mut ro = std::mem::take(&mut self.scratch_raw_obs);
        let mut rc = std::mem::take(&mut self.scratch_raw_cobs);
        ro.resize(obs_dim * n, 0.0);
        rc.resize(crit_dim * n, 0.0);
        fill_raw(&mut ro, obs_dim, n, cur);
        fill_raw(&mut rc, crit_dim, n, cur_c);
        // Step-blocked buffer, so this is ONE contiguous write per step.
        if let Some(rb) = self.raw_obs.as_mut() {
            backend.write_buffer(
                rb.buffer_mut(),
                (step * obs_dim * n) as u64,
                &ro[..obs_dim * n],
            )?;
        }
        if let Some(rb) = self.raw_cobs.as_mut() {
            backend.write_buffer(
                rb.buffer_mut(),
                (step * crit_dim * n) as u64,
                &rc[..crit_dim * n],
            )?;
        }
        self.scratch_raw_obs = ro;
        self.scratch_raw_cobs = rc;
        self.forward_from_raw(backend, ac, step).await
    }

    /// Device-obs forward: the raw observations are already ON DEVICE
    /// (`BIPED_GPU_OBS` — the env's `GpuObserve` assembled them). Copy the
    /// `[dim x n]` tensors into raw-arena slot `step` (device-to-device, no
    /// host round trip) and run the same staged forward as `forward`.
    pub async fn forward_dev(
        &mut self,
        backend: &GpuBackend,
        ac: &ActorCritic,
        obs_dev: &Tensor<f32>,
        cobs_dev: &Tensor<f32>,
        step: usize,
    ) -> anyhow::Result<(Vec<[f32; NUM_JOINTS]>, Vec<f32>)> {
        use khal::backend::Encoder as _;
        let n = self.n;
        let (obs_dim, crit_dim) = (self.actor.dims[0], self.critic.dims[0]);
        if std::env::var("BIPED_DEV_OBS_DBG").is_ok() {
            eprintln!(
                "[dev_obs] obs_dev.len={} cobs_dev.len={} raw_obs.len={} raw_cobs.len={} obs_dim={obs_dim} crit_dim={crit_dim} n={n} step={step}",
                obs_dev.len(),
                cobs_dev.len(),
                self.raw_obs.as_ref().unwrap().len(),
                self.raw_cobs.as_ref().unwrap().len(),
            );
        }
        {
            let mut enc = backend.begin_encoding();
            enc.copy_buffer_to_buffer::<f32>(
                &obs_dev.buffer(),
                0,
                &mut self.raw_obs.as_mut().expect("init_raw_batch").buffer_mut(),
                step * obs_dim * n,
                obs_dim * n,
            )?;
            enc.copy_buffer_to_buffer::<f32>(
                &cobs_dev.buffer(),
                0,
                &mut self.raw_cobs.as_mut().expect("init_raw_batch").buffer_mut(),
                step * crit_dim * n,
                crit_dim * n,
            )?;
            backend.submit(enc)?;
        }
        self.forward_from_raw(backend, ac, step).await
    }

    /// Stage slot `step` of the raw arena into the policy inputs, run both
    /// nets and read back `(means, values)` — shared tail of `forward` /
    /// `forward_dev`.
    async fn forward_from_raw(
        &mut self,
        backend: &GpuBackend,
        ac: &ActorCritic,
        step: usize,
    ) -> anyhow::Result<(Vec<[f32; NUM_JOINTS]>, Vec<f32>)> {
        let n = self.n;
        let (obs_dim, crit_dim) = (self.actor.dims[0], self.critic.dims[0]);
        let _ = ac;

        // Params for the single-step (`step_select`) form of the staging
        // kernel: columns are the n envs of rollout step `step_sel - 1`.
        let mk_p = |dim: usize, step_sel: usize| vortx::linalg::PpoStageParams {
            dim: dim as u32,
            n: n as u32,
            steps: 0,
            total_cols: n as u32,
            col_offset: 0,
            step_select: step_sel as u32,
            pad1: 0,
            pad2: 0,
        };
        let pu = BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST;
        if self.stage_p_obs.is_none() {
            self.stage_p_obs = Some(Tensor::scalar(backend, mk_p(obs_dim, 1), pu)?);
            self.stage_p_cobs = Some(Tensor::scalar(backend, mk_p(crit_dim, 1), pu)?);
        }
        backend.write_buffer(
            self.stage_p_obs.as_mut().unwrap().buffer_mut(),
            0,
            &[mk_p(obs_dim, step + 1)],
        )?;
        backend.write_buffer(
            self.stage_p_cobs.as_mut().unwrap().buffer_mut(),
            0,
            &[mk_p(crit_dim, step + 1)],
        )?;

        let mut cur = EncCursor::new(backend);
        {
            let nb = self.norm.as_ref().expect("GpuPolicy::set_norm not called");
            let (stage, po, pc) = (
                &self.stage,
                self.stage_p_obs.as_ref().unwrap(),
                self.stage_p_cobs.as_ref().unwrap(),
            );
            let (raw_o, raw_c) = (
                self.raw_obs.as_ref().expect("init_raw_batch"),
                self.raw_cobs.as_ref().expect("init_raw_batch"),
            );
            let mut pass = cur.pass("policy_input_normalize");
            stage.stage_batch(
                &mut pass,
                po,
                raw_o,
                &nb.mean_o,
                &nb.inv_o,
                self.id_perm_o.as_ref().unwrap(),
                self.id_sign_o.as_ref().unwrap(),
                &mut self.actor.a[0],
                n as u32,
                obs_dim as u32,
            )?;
            stage.stage_batch(
                &mut pass,
                pc,
                raw_c,
                &nb.mean_c,
                &nb.inv_c,
                self.id_perm_c.as_ref().unwrap(),
                self.id_sign_c.as_ref().unwrap(),
                &mut self.critic.a[0],
                n as u32,
                crit_dim as u32,
            )?;
        }
        self.actor
            .encode(backend, &self.ops, &mut self.shapes, &mut cur, self.ct)?;
        self.critic
            .encode(backend, &self.ops, &mut self.shapes, &mut cur, self.ct)?;
        cur.flush();
        backend.synchronize()?;

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

    /// The per-iteration normalizer/mirror buffers, for the batch build.
    pub fn norm_bufs(&self) -> &NormBufs {
        self.norm.as_ref().expect("set_norm not called")
    }

    /// Per-iteration normalizer affine + mirror tables shared by the policy
    /// input normalize and the PPO batch build.
    pub fn set_norm(&mut self, bufs: NormBufs) {
        self.norm = Some(bufs);
    }

    /// Allocate the step-blocked raw-observation batch buffers (`[T][dim][n]`).
    pub fn init_raw_batch(&mut self, backend: &GpuBackend, horizon: usize) -> anyhow::Result<()> {
        let n = self.n;
        let (od, cd) = (self.actor.dims[0], self.critic.dims[0]);
        let u = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        self.raw_obs = Some(Tensor::vector_uninit(
            backend,
            (horizon * od * n) as u32,
            u,
        )?);
        self.raw_cobs = Some(Tensor::vector_uninit(
            backend,
            (horizon * cd * n) as u32,
            u,
        )?);
        let sb = BufferUsages::STORAGE;
        self.id_perm_o = Some(Tensor::vector(
            backend,
            &(0..od as u32).collect::<Vec<u32>>(),
            sb,
        )?);
        self.id_sign_o = Some(Tensor::vector(backend, &vec![1.0f32; od], sb)?);
        self.id_perm_c = Some(Tensor::vector(
            backend,
            &(0..cd as u32).collect::<Vec<u32>>(),
            sb,
        )?);
        self.id_sign_c = Some(Tensor::vector(backend, &vec![1.0f32; cd], sb)?);
        Ok(())
    }

    /// The retained raw rollout observations (`[T][obs_dim][n]`).
    pub fn raw_obs(&self) -> &Tensor<f32> {
        self.raw_obs.as_ref().expect("init_raw_batch not called")
    }

    /// The retained raw critic observations (`[T][critic_dim][n]`).
    pub fn raw_cobs(&self) -> &Tensor<f32> {
        self.raw_cobs.as_ref().expect("init_raw_batch not called")
    }

    /// GPU-resident actor access: the input activation tensor (row-major
    /// [obs_dim × n]) for an external writer (e.g. the GPU obs-assembly
    /// kernel, which writes ALREADY-normalized values), …
    pub fn actor_input_mut(&mut self) -> &mut Tensor<f32> {
        &mut self.actor.a[0]
    }

    /// … the output activation (row-major [act_dim × n]) …
    pub fn actor_output(&self) -> &Tensor<f32> {
        self.actor.output()
    }

    /// … and an encode-only mean forward (no host copies, no sync — caller
    /// owns submission via the cursor).
    pub fn encode_actor(
        &mut self,
        backend: &GpuBackend,
        cur: &mut EncCursor,
    ) -> anyhow::Result<()> {
        self.actor
            .encode(backend, &self.ops, &mut self.shapes, cur, self.ct)?;
        Ok(())
    }
}
