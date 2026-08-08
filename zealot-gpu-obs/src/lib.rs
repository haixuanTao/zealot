//! Host bindings for the zealot GPU observation-assembly kernels
//! ([`zealot_obs_shaders`]): owns the per-env GPU state (prev-q / action ring
//! / episode counters, obs-history ring, commands, constants) and encodes the
//! two kernels around the policy GEMM each control step.
//!
//! The full resident control step is:
//! `gpu_assemble_obs` (physics buffers → normalized [225 × N] GEMM input) →
//! vortx policy GEMMs → `gpu_commit_actions` (actions → PD targets + ring
//! shift) → `scatter_motor_targets_gpu` → physics substeps. The CPU never
//! reads observations back.

pub use zealot_obs_shaders as shaders;

use include_dir::{Dir, include_dir};
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

use shaders::{
    C_DEFAULT, C_HI, C_LEN, C_LINK, C_LO, C_MEAN, C_STD, FRAME, FRAME_GYRO, FRAME_NO_GYRO, HIST,
    N_ACT,
    STATE_STRIDE,
};

/// Embedded SPIR-V shader directory (resolved by `#[derive(Shader)]`).
pub static SPIRV_DIR: Dir<'static> = include_dir!("$OUT_DIR/shaders-spirv");

/// Host binding for the GPU reward terms.
#[derive(Shader)]
struct RewardShader {
    terms: shaders::GpuRewardActionTerms,
}

/// GPU reward terms + persistent staging.
///
/// PARTIAL PORT BY DESIGN: the kernel owns only the action-smoothness terms so
/// far; the caller still computes the rest on host and the totals are
/// unaffected. A term moves across only once `BIPED_VERIFY_REWARD=1` shows it
/// matching the host value.
pub struct GpuRewardTerms {
    shader: RewardShader,
    params: vortx::tensor::Tensor<shaders::RewardActionParams>,
    last: vortx::tensor::Tensor<f32>,
    prev: vortx::tensor::Tensor<f32>,
    prev2: vortx::tensor::Tensor<f32>,
    hip_mask: vortx::tensor::Tensor<f32>,
    out: vortx::tensor::Tensor<f32>,
    n: usize,
    j: usize,
}


impl GpuRewardTerms {
    /// Allocate for `n` envs and `j` actuated joints. `hip` are the joint
    /// indices `action_rate_hipz_hipx` sums over.
    pub fn new(
        backend: &GpuBackend,
        n: usize,
        j: usize,
        hip: &[usize],
    ) -> Result<Self, khal::backend::GpuBackendError> {
        use khal::BufferUsages as U;
        let st = U::STORAGE | U::COPY_DST;
        let mut mask = vec![0.0f32; j];
        for &i in hip {
            if i < j {
                mask[i] = 1.0;
            }
        }
        Ok(Self {
            shader: RewardShader::from_backend(backend)?,
            params: vortx::tensor::Tensor::scalar(
                backend,
                shaders::RewardActionParams {
                    n_envs: n as u32,
                    num_joints: j as u32,
                    dt: 0.0,
                    w_action_rate: 0.0,
                    w_action_rate_hip: 0.0,
                    w_action_rate_rate: 0.0,
                    pad0: 0,
                    pad1: 0,
                },
                U::UNIFORM | U::STORAGE | U::COPY_DST,
            )?,
            last: vortx::tensor::Tensor::vector_uninit(backend, (j * n) as u32, st)?,
            prev: vortx::tensor::Tensor::vector_uninit(backend, (j * n) as u32, st)?,
            prev2: vortx::tensor::Tensor::vector_uninit(backend, (j * n) as u32, st)?,
            hip_mask: vortx::tensor::Tensor::vector(backend, &mask, st)?,
            out: vortx::tensor::Tensor::vector_uninit(backend, (3 * n) as u32, st | U::COPY_SRC)?,
            n,
            j,
        })
    }

    /// Upload the three action vectors (each row-major `[j x n]`), dispatch,
    /// and read back the `[3 x n]` term values.
    #[allow(clippy::too_many_arguments)]
    pub async fn compute(
        &mut self,
        backend: &GpuBackend,
        last: &[f32],
        prev: &[f32],
        prev2: &[f32],
        dt: f32,
        w_rate: f32,
        w_hip: f32,
        w_rate_rate: f32,
    ) -> Result<Vec<f32>, khal::backend::GpuBackendError> {
        use khal::backend::{Backend, Encoder};
        backend.write_buffer(self.params.buffer_mut(), 0, &[shaders::RewardActionParams {
            n_envs: self.n as u32,
            num_joints: self.j as u32,
            dt,
            w_action_rate: w_rate,
            w_action_rate_hip: w_hip,
            w_action_rate_rate: w_rate_rate,
            pad0: 0,
            pad1: 0,
        }])?;
        backend.write_buffer(self.last.buffer_mut(), 0, last)?;
        backend.write_buffer(self.prev.buffer_mut(), 0, prev)?;
        backend.write_buffer(self.prev2.buffer_mut(), 0, prev2)?;
        let mut enc = backend.begin_encoding();
        {
            let mut pass = enc.begin_pass("[reward] action terms", None);
            self.shader.terms.call(
                &mut pass,
                self.n as u32,
                &self.params,
                &self.last,
                &self.prev,
                &self.prev2,
                &self.hip_mask,
                &mut self.out,
            )?;
        }
        backend.submit(enc)?;
        backend.slow_read_vec(self.out.buffer()).await
    }
}

#[derive(Shader)]
struct ObsBundle {
    assemble: shaders::GpuAssembleObs,
    commit: shaders::GpuCommitActions,
}

/// Static config for [`GpuObs::new`].
pub struct GpuObsConfig {
    /// Child-link id of each policy joint (joint angles read from the link's
    /// ws joint-rotation quat).
    pub link_ids: [u32; N_ACT],
    /// Home pose (action origin).
    pub default_pos: [f32; N_ACT],
    /// PD target clamp.
    pub target_lo: [f32; N_ACT],
    pub target_hi: [f32; N_ACT],
    /// Welford normalizer over the stacked obs (`HIST * frame` long — its
    /// length is what sets [`GpuObs::frame`]).
    pub norm_mean: Vec<f32>,
    pub norm_std: Vec<f32>,
    /// Control dt (s) and action scale. The gait-clock period is not a knob:
    /// the kernel derives it from the command, mirroring the training env.
    pub dt: f32,
    pub action_scale: f32,
}

/// GPU-resident observation/actions state for `n` envs.
pub struct GpuObs {
    bundle: ObsBundle,
    state: Tensor<f32>,
    hist: Tensor<f32>,
    cmd: Tensor<f32>,
    consts: Tensor<f32>,
    /// Policy-target output of `gpu_commit_actions`, row-major [12 × N] —
    /// feed to `scatter_motor_targets_gpu`.
    pub targets: Tensor<f32>,
    u_n: Tensor<u32>,
    u_dt: Tensor<f32>,
    u_frame: Tensor<u32>,
    u_scale: Tensor<f32>,
    n: usize,
}

impl GpuObs {
    pub fn new(
        backend: &GpuBackend,
        n: usize,
        cfg: &GpuObsConfig,
    ) -> Result<Self, GpuBackendError> {
        // The checkpoint's normalizer length picks the obs width: 45 for
        // pre-gyro policies (v21 and earlier), 48 with the gyro (v24 on),
        // 53 with the step cue (v28 on — the kernel zero-fills the cue).
        assert_eq!(cfg.norm_mean.len(), cfg.norm_std.len());
        let frame = cfg.norm_mean.len() / HIST;
        assert!(
            (frame == FRAME || frame == FRAME_GYRO || frame == FRAME_NO_GYRO)
                && frame * HIST == cfg.norm_mean.len(),
            "unsupported obs layout: {} = {HIST} x {frame} (expected a frame of {FRAME_NO_GYRO}, {FRAME_GYRO} or {FRAME})",
            cfg.norm_mean.len(),
        );
        let mut consts = vec![0.0f32; C_LEN];
        for j in 0..N_ACT {
            consts[C_LINK + j] = cfg.link_ids[j] as f32;
            consts[C_DEFAULT + j] = cfg.default_pos[j];
            consts[C_LO + j] = cfg.target_lo[j];
            consts[C_HI + j] = cfg.target_hi[j];
        }
        // Packed at `frame` stride from the fixed region starts — the kernel
        // reads them the same way, and C_STD stays put as `frame` varies.
        consts[C_MEAN..C_MEAN + HIST * frame].copy_from_slice(&cfg.norm_mean);
        consts[C_STD..C_STD + HIST * frame].copy_from_slice(&cfg.norm_std);

        let st = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let uu = BufferUsages::STORAGE | BufferUsages::UNIFORM;
        Ok(Self {
            bundle: ObsBundle::from_backend(backend)?,
            state: Tensor::vector(backend, &vec![0.0f32; STATE_STRIDE * n], st)?,
            hist: Tensor::vector(backend, &vec![0.0f32; HIST * FRAME * n], st)?,
            cmd: Tensor::vector(backend, &vec![0.0f32; 4 * n], st)?,
            consts: Tensor::vector(backend, &consts, st)?,
            targets: {
                // Prefill with the home pose so the very first physics step
                // (before the first policy commit) holds the spawn stance.
                let mut t0 = vec![0.0f32; N_ACT * n];
                for j in 0..N_ACT {
                    for e in 0..n {
                        t0[j * n + e] = cfg.default_pos[j];
                    }
                }
                Tensor::vector(backend, &t0, st)?
            },
            u_n: Tensor::scalar(backend, n as u32, uu)?,
            u_dt: Tensor::scalar(backend, cfg.dt, uu)?,
            u_frame: Tensor::scalar(backend, frame as u32, uu)?,
            u_scale: Tensor::scalar(backend, cfg.action_scale, uu)?,
            n,
        })
    }

    /// Pin env `e`'s velocity command (vx, vy, yaw, 0).
    pub fn set_cmd(
        &mut self,
        backend: &GpuBackend,
        e: usize,
        cmd: [f32; 3],
    ) -> Result<(), GpuBackendError> {
        for (c, v) in [cmd[0], cmd[1], cmd[2], 0.0].into_iter().enumerate() {
            backend.write_buffer(self.cmd.buffer_mut(), (c * self.n + e) as u64, &[v])?;
        }
        Ok(())
    }

    /// Reset env `e`'s controller state (episode counter → 0 triggers
    /// reset-replicate history + zeroed lag terms on the next assemble).
    pub fn reset_env(&mut self, backend: &GpuBackend, e: usize) -> Result<(), GpuBackendError> {
        let zeros = [0.0f32; STATE_STRIDE];
        for i in 0..STATE_STRIDE {
            backend.write_buffer(
                self.state.buffer_mut(),
                (i * self.n + e) as u64,
                &zeros[i..=i],
            )?;
        }
        Ok(())
    }

    /// Encode obs assembly into `enc`: physics buffers → `gemm_input`
    /// (row-major [225 × N], the policy's a0).
    pub fn encode_assemble(
        &mut self,
        enc: &mut <GpuBackend as Backend>::Encoder,
        links_workspace: &Tensor<glamx::Vec4>,
        gemm_input: &mut Tensor<f32>,
    ) -> Result<(), GpuBackendError> {
        let groups = (self.n as u32).div_ceil(64);
        let mut pass = enc.begin_pass("zealot_assemble_obs", None);
        self.bundle.assemble.call(
            &mut pass,
            [groups, 1, 1],
            links_workspace,
            &mut self.state,
            &mut self.hist,
            &self.cmd,
            &self.consts,
            gemm_input,
            &self.u_n,
            &self.u_dt,
            &self.u_frame,
        )?;
        Ok(())
    }

    /// Encode the post-policy commit into `enc`: `actions` [12 × N] → PD
    /// `self.targets` + action-ring shift + episode-counter advance.
    pub fn encode_commit(
        &mut self,
        enc: &mut <GpuBackend as Backend>::Encoder,
        actions: &Tensor<f32>,
    ) -> Result<(), GpuBackendError> {
        let groups = (self.n as u32).div_ceil(64);
        let mut pass = enc.begin_pass("zealot_commit_actions", None);
        self.bundle.commit.call(
            &mut pass,
            [groups, 1, 1],
            actions,
            &mut self.state,
            &self.consts,
            &mut self.targets,
            &self.u_n,
            &self.u_scale,
        )?;
        Ok(())
    }
}
