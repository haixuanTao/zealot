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





/// Host binding for the feet-state kernel.
#[derive(Shader)]
struct FeetStateShader {
    feet: shaders::GpuFeetState,
}

/// Per-foot state on device. The per-env history it depends on (air time, last
/// touchdown foot, previous force, previous foot poses) is still supplied by
/// the caller — porting the maths first, the state migration after.
pub struct GpuFeetState {
    shader: FeetStateShader,
    params: Tensor<shaders::FeetStateParams>,
    foot_links: Tensor<u32>,
    foot_fwd: Tensor<f32>,
    sole_local: Tensor<f32>,
    prev_foot_pos: Tensor<f32>,
    ground_h: Tensor<f32>,
    sensed_force: Tensor<f32>,
    prev_force: Tensor<f32>,
    air_time: Tensor<f32>,
    last_td: Tensor<f32>,
    have_prev: Tensor<u32>,
    have_prev_force: Tensor<u32>,
    out: Tensor<f32>,
    n: usize,
}

/// Per-step inputs for [`GpuFeetState::compute`].
pub struct FeetInputs<'a> {
    pub sole_local: &'a [f32],
    pub prev_foot_pos: &'a [f32],
    pub ground_h: &'a [f32],
    pub sensed_force: &'a [f32],
    pub prev_force: &'a [f32],
    pub air_time: &'a [f32],
    pub last_td: &'a [f32],
    pub have_prev: &'a [u32],
    pub have_prev_force: &'a [u32],
}

impl GpuFeetState {
    pub fn new(
        backend: &GpuBackend,
        n: usize,
        n_feet: usize,
        foot_links: &[u32],
        foot_fwd: &[f32],
        params: shaders::FeetStateParams,
    ) -> Result<Self, GpuBackendError> {
        let st = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let f = |len: usize| Tensor::vector(backend, &vec![0.0f32; len], st);
        Ok(Self {
            shader: FeetStateShader::from_backend(backend)?,
            params: Tensor::scalar(
                backend,
                params,
                BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )?,
            foot_links: Tensor::vector(backend, foot_links, st)?,
            foot_fwd: Tensor::vector(backend, foot_fwd, st)?,
            sole_local: f(n_feet * 3 * n)?,
            prev_foot_pos: f(n_feet * 3 * n)?,
            ground_h: f(n_feet * n)?,
            sensed_force: f(n_feet * n)?,
            prev_force: f(n_feet * n)?,
            air_time: f(n_feet * n)?,
            last_td: f(n)?,
            have_prev: Tensor::vector(backend, &vec![0u32; n], st)?,
            have_prev_force: Tensor::vector(backend, &vec![0u32; n], st)?,
            out: Tensor::vector_uninit(backend, (26 * n) as u32, st | BufferUsages::COPY_SRC)?,
            n,
        })
    }

    /// Returns `[26 x n]`: per foot i at offset i*11 — contact, first_contact,
    /// air_time, height, planar_speed, tilt, yaw_rel_base, x, y, vz,
    /// force_rate; then alt_step at 22+i and the new air time at 24+i.
    pub async fn compute(
        &mut self,
        backend: &GpuBackend,
        body_poses: &Tensor<glamx::Pose3>,
        inputs: FeetInputs<'_>,
    ) -> Result<Vec<f32>, GpuBackendError> {
        backend.write_buffer(self.sole_local.buffer_mut(), 0, inputs.sole_local)?;
        backend.write_buffer(self.prev_foot_pos.buffer_mut(), 0, inputs.prev_foot_pos)?;
        backend.write_buffer(self.ground_h.buffer_mut(), 0, inputs.ground_h)?;
        backend.write_buffer(self.sensed_force.buffer_mut(), 0, inputs.sensed_force)?;
        backend.write_buffer(self.prev_force.buffer_mut(), 0, inputs.prev_force)?;
        backend.write_buffer(self.air_time.buffer_mut(), 0, inputs.air_time)?;
        backend.write_buffer(self.last_td.buffer_mut(), 0, inputs.last_td)?;
        backend.write_buffer(self.have_prev.buffer_mut(), 0, inputs.have_prev)?;
        backend.write_buffer(self.have_prev_force.buffer_mut(), 0, inputs.have_prev_force)?;
        let mut enc = backend.begin_encoding();
        {
            let mut pass = enc.begin_pass("[reward] feet state", None);
            self.shader.feet.call(
                &mut pass,
                self.n as u32,
                &self.params,
                body_poses,
                &self.foot_links,
                &self.foot_fwd,
                &self.sole_local,
                &self.prev_foot_pos,
                &self.ground_h,
                &self.sensed_force,
                &self.prev_force,
                &self.air_time,
                &self.last_td,
                &self.have_prev,
                &self.have_prev_force,
                &mut self.out,
            )?;
        }
        backend.submit(enc)?;
        backend.slow_read_vec(self.out.buffer()).await
    }
}

/// Host binding for the base-state reward terms.
#[derive(Shader)]
struct RewardBaseShader {
    terms: shaders::GpuRewardBaseTerms,
}

/// track_lin_vel / track_ang_vel / upright / base_height / body_ang_vel /
/// lin_vel_z, from the GPU base state.
pub struct GpuRewardBaseTerms {
    shader: RewardBaseShader,
    params: Tensor<shaders::RewardBaseParams>,
    base: Tensor<f32>,
    cmd: Tensor<f32>,
    cue: Tensor<f32>,
    out: Tensor<f32>,
    n: usize,
}

impl GpuRewardBaseTerms {
    pub fn new(backend: &GpuBackend, n: usize) -> Result<Self, GpuBackendError> {
        let st = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        Ok(Self {
            shader: RewardBaseShader::from_backend(backend)?,
            params: Tensor::scalar(
                backend,
                shaders::RewardBaseParams {
                    n_envs: n as u32,
                    dt: 0.0,
                    w_track_lin: 0.0,
                    w_forward_progress: 0.0,
                    w_track_ang: 0.0,
                    w_upright: 0.0,
                    w_base_height: 0.0,
                    w_body_ang_vel: 0.0,
                    w_lin_vel_z: 0.0,
                    std_lin: 1.0,
                    std_ang: 1.0,
                    std_base_h: 1.0,
                    std_upright: 1.0,
                    step_std_base_h: 1.0,
                    step_std_upright: 1.0,
                    step_relax_dist: 0.0,
                    h_target_stand: 0.0,
                    h_target_walk: 0.0,
                    pad0: 0,
                    pad1: 0,
                },
                BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )?,
            base: Tensor::vector_uninit(backend, (13 * n) as u32, st)?,
            cmd: Tensor::vector(backend, &vec![0.0f32; 4 * n], st)?,
            cue: Tensor::vector(backend, &vec![0.0f32; 4 * n], st)?,
            out: Tensor::vector_uninit(backend, (6 * n) as u32, st | BufferUsages::COPY_SRC)?,
            n,
        })
    }

    /// `base` is the `[13 x n]` block from `GpuBaseState`; returns `[6 x n]`.
    pub async fn compute(
        &mut self,
        backend: &GpuBackend,
        params: shaders::RewardBaseParams,
        base: &[f32],
        cmd: &[f32],
        cue: &[f32],
    ) -> Result<Vec<f32>, GpuBackendError> {
        backend.write_buffer(self.params.buffer_mut(), 0, &[params])?;
        backend.write_buffer(self.base.buffer_mut(), 0, base)?;
        backend.write_buffer(self.cmd.buffer_mut(), 0, cmd)?;
        backend.write_buffer(self.cue.buffer_mut(), 0, cue)?;
        let mut enc = backend.begin_encoding();
        {
            let mut pass = enc.begin_pass("[reward] base terms", None);
            self.shader.terms.call(
                &mut pass,
                self.n as u32,
                &self.params,
                &self.base,
                &self.cmd,
                &self.cue,
                &mut self.out,
            )?;
        }
        backend.submit(enc)?;
        backend.slow_read_vec(self.out.buffer()).await
    }
}

/// Host binding for the base-state kernel.
#[derive(Shader)]
struct BaseStateShader {
    base: shaders::GpuBaseState,
}

/// Base pose, world velocities and terrain-relative height on device.
pub struct GpuBaseState {
    shader: BaseStateShader,
    params: Tensor<shaders::BaseStateParams>,
    have_prev: Tensor<u32>,
    ground_h: Tensor<f32>,
    prev_pose: Tensor<f32>,
    out: Tensor<f32>,
    n: usize,
}

impl GpuBaseState {
    pub fn new(
        backend: &GpuBackend,
        n: usize,
        bodies_per_env: u32,
        torso_link: u32,
        control_dt: f32,
    ) -> Result<Self, GpuBackendError> {
        let st = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        Ok(Self {
            shader: BaseStateShader::from_backend(backend)?,
            params: Tensor::scalar(
                backend,
                shaders::BaseStateParams { n_envs: n as u32, bodies_per_env, torso_link, control_dt },
                BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )?,
            have_prev: Tensor::vector(backend, &vec![0u32; n], st)?,
            ground_h: Tensor::vector(backend, &vec![0.0f32; n], st)?,
            prev_pose: Tensor::vector(backend, &vec![0.0f32; 7 * n], st)?,
            out: Tensor::vector_uninit(backend, (13 * n) as u32, st | BufferUsages::COPY_SRC)?,
            n,
        })
    }

    /// Returns `[13 x n]`: quat xyzw, lin vel, ang vel, height, xy.
    pub async fn compute(
        &mut self,
        backend: &GpuBackend,
        body_poses: &Tensor<glamx::Pose3>,
        have_prev: &[u32],
        ground_h: &[f32],
    ) -> Result<Vec<f32>, GpuBackendError> {
        backend.write_buffer(self.have_prev.buffer_mut(), 0, have_prev)?;
        backend.write_buffer(self.ground_h.buffer_mut(), 0, ground_h)?;
        let mut enc = backend.begin_encoding();
        {
            let mut pass = enc.begin_pass("[reward] base state", None);
            self.shader.base.call(
                &mut pass,
                self.n as u32,
                &self.params,
                body_poses,
                &self.have_prev,
                &self.ground_h,
                &mut self.prev_pose,
                &mut self.out,
            )?;
        }
        backend.submit(enc)?;
        backend.slow_read_vec(self.out.buffer()).await
    }
}

/// Host binding for the joint-state kernel.
#[derive(Shader)]
struct JointStateShader {
    joints: shaders::GpuJointState,
}

/// GPU joint angles + finite-difference velocities.
///
/// Mirrors the TRAINER host's `read_state_from_poses` (parent-child relative
/// rotation with the rest quat removed), not the demo obs kernel's workspace
/// read — see `gpu_joint_state`.
pub struct GpuJointState {
    shader: JointStateShader,
    params: vortx::tensor::Tensor<shaders::JointStateParams>,
    parent_ids: vortx::tensor::Tensor<u32>,
    child_ids: vortx::tensor::Tensor<u32>,
    rest_quats: vortx::tensor::Tensor<glamx::Vec4>,
    have_prev: vortx::tensor::Tensor<u32>,
    prev_q: vortx::tensor::Tensor<f32>,
    q: vortx::tensor::Tensor<f32>,
    qd: vortx::tensor::Tensor<f32>,
    n: usize,
    j: usize,
}

impl GpuJointState {
    /// `rest` is one xyzw quaternion per joint; `parents`/`children` are link
    /// indices into the env-major `body_poses` buffer.
    pub fn new(
        backend: &GpuBackend,
        n: usize,
        j: usize,
        parents: &[u32],
        children: &[u32],
        rest: &[glamx::Vec4],
        bodies_per_env: u32,
        control_dt: f32,
    ) -> Result<Self, khal::backend::GpuBackendError> {
        use khal::BufferUsages as U;
        let st = U::STORAGE | U::COPY_DST;
        Ok(Self {
            shader: JointStateShader::from_backend(backend)?,
            params: vortx::tensor::Tensor::scalar(
                backend,
                shaders::JointStateParams {
                    n_envs: n as u32,
                    num_joints: j as u32,
                    bodies_per_env,
                    control_dt,
                    pad0: 0,
                },
                U::UNIFORM | U::STORAGE | U::COPY_DST,
            )?,
            parent_ids: vortx::tensor::Tensor::vector(backend, parents, st)?,
            child_ids: vortx::tensor::Tensor::vector(backend, children, st)?,
            rest_quats: vortx::tensor::Tensor::vector(backend, rest, st)?,
            have_prev: vortx::tensor::Tensor::vector(backend, &vec![0u32; n], st)?,
            prev_q: vortx::tensor::Tensor::vector(backend, &vec![0.0f32; j * n], st)?,
            q: vortx::tensor::Tensor::vector_uninit(backend, (j * n) as u32, st | U::COPY_SRC)?,
            qd: vortx::tensor::Tensor::vector_uninit(backend, (j * n) as u32, st | U::COPY_SRC)?,
            n,
            j,
        })
    }

    /// The device-resident joint angles, row-major `[j x n]`.
    pub fn q_buffer(&self) -> &Tensor<f32> {
        &self.q
    }

    /// The device-resident joint velocities, row-major `[j x n]`.
    pub fn qd_buffer(&self) -> &Tensor<f32> {
        &self.qd
    }

    /// Dispatch against the live `body_poses` buffer, then
    /// read back `(q, qd)` — both row-major `[j x n]`.
    pub async fn compute(
        &mut self,
        backend: &GpuBackend,
        body_poses: &vortx::tensor::Tensor<glamx::Pose3>,
        have_prev: &[u32],
    ) -> Result<(Vec<f32>, Vec<f32>), khal::backend::GpuBackendError> {
        use khal::backend::{Backend, Encoder};
        backend.write_buffer(self.have_prev.buffer_mut(), 0, have_prev)?;
        let mut enc = backend.begin_encoding();
        {
            let mut pass = enc.begin_pass("[reward] joint state", None);
            self.shader.joints.call(
                &mut pass,
                self.n as u32,
                &self.params,
                body_poses,
                &self.parent_ids,
                &self.child_ids,
                &self.rest_quats,
                &self.have_prev,
                &mut self.prev_q,
                &mut self.q,
                &mut self.qd,
            )?;
        }
        backend.submit(enc)?;
        let q = backend.slow_read_vec(self.q.buffer()).await?;
        let qd = backend.slow_read_vec(self.qd.buffer()).await?;
        Ok((q, qd))
    }
}



/// Host binding for the torque / power reward terms.
#[derive(Shader)]
struct RewardTorqueShader {
    terms: shaders::GpuRewardTorqueTerms,
}

/// torque_leg / torque_ankle / power from the GPU joint state + PD targets.
pub struct GpuRewardTorqueTerms {
    shader: RewardTorqueShader,
    params: Tensor<shaders::RewardTorqueParams>,
    q_target: Tensor<f32>,
    kp: Tensor<f32>,
    kd: Tensor<f32>,
    effort: Tensor<f32>,
    w_leg: Tensor<f32>,
    w_ankle: Tensor<f32>,
    w_knee: Tensor<f32>,
    out: Tensor<f32>,
    n: usize,
    j: usize,
}

impl GpuRewardTorqueTerms {
    /// `w_leg`/`w_ankle`/`w_knee` are per-joint, with the host's joint-name
    /// classification already resolved into them.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: &GpuBackend,
        n: usize,
        j: usize,
        kp: &[f32],
        kd: &[f32],
        effort: &[f32],
        w_leg: &[f32],
        w_ankle: &[f32],
        w_knee: &[f32],
    ) -> Result<Self, GpuBackendError> {
        let st = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        Ok(Self {
            shader: RewardTorqueShader::from_backend(backend)?,
            params: Tensor::scalar(
                backend,
                shaders::RewardTorqueParams {
                    n_envs: n as u32,
                    num_joints: j as u32,
                    dt: 0.0,
                    torque_w: 0.0,
                    ankle_torque_w: 0.0,
                    power_w: 0.0,
                    pad0: 0,
                    pad1: 0,
                },
                BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )?,
            q_target: Tensor::vector_uninit(backend, (j * n) as u32, st)?,
            kp: Tensor::vector(backend, kp, st)?,
            kd: Tensor::vector(backend, kd, st)?,
            effort: Tensor::vector(backend, effort, st)?,
            w_leg: Tensor::vector(backend, w_leg, st)?,
            w_ankle: Tensor::vector(backend, w_ankle, st)?,
            w_knee: Tensor::vector(backend, w_knee, st)?,
            out: Tensor::vector_uninit(backend, (3 * n) as u32, st | BufferUsages::COPY_SRC)?,
            n,
            j,
        })
    }

    /// `q_target` is the row-major `[j x n]` PD target the env already stages.
    #[allow(clippy::too_many_arguments)]
    pub async fn compute(
        &mut self,
        backend: &GpuBackend,
        joints: &GpuJointState,
        q_target: &[f32],
        dt: f32,
        torque_w: f32,
        ankle_torque_w: f32,
        power_w: f32,
    ) -> Result<Vec<f32>, GpuBackendError> {
        backend.write_buffer(self.params.buffer_mut(), 0, &[shaders::RewardTorqueParams {
            n_envs: self.n as u32,
            num_joints: self.j as u32,
            dt,
            torque_w,
            ankle_torque_w,
            power_w,
            pad0: 0,
            pad1: 0,
        }])?;
        backend.write_buffer(self.q_target.buffer_mut(), 0, q_target)?;
        let mut enc = backend.begin_encoding();
        {
            let mut pass = enc.begin_pass("[reward] torque terms", None);
            self.shader.terms.call(
                &mut pass,
                self.n as u32,
                &self.params,
                joints.q_buffer(),
                joints.qd_buffer(),
                &self.q_target,
                &self.kp,
                &self.kd,
                &self.effort,
                &self.w_leg,
                &self.w_ankle,
                &self.w_knee,
                &mut self.out,
            )?;
        }
        backend.submit(enc)?;
        backend.slow_read_vec(self.out.buffer()).await
    }
}

/// Host binding for the joint-only reward terms.
#[derive(Shader)]
struct RewardJointShader {
    terms: shaders::GpuRewardJointTerms,
}

/// pose / dof_pos_limits / dof_vel, computed from the GPU joint state.
pub struct GpuRewardJointTerms {
    shader: RewardJointShader,
    params: Tensor<shaders::RewardJointParams>,
    default_pos: Tensor<f32>,
    soft_lo: Tensor<f32>,
    soft_hi: Tensor<f32>,
    hip_mask: Tensor<f32>,
    mirror_idx: Tensor<u32>,
    mirror_sign: Tensor<f32>,
    cmd_yaw: Tensor<f32>,
    out: Tensor<f32>,
    n: usize,
    j: usize,
}

impl GpuRewardJointTerms {
    /// `soft_lo`/`soft_hi` must already carry the host's 0.9 band factor.
    pub fn new(
        backend: &GpuBackend,
        n: usize,
        j: usize,
        default_pos: &[f32],
        soft_lo: &[f32],
        soft_hi: &[f32],
        hip: &[usize],
        mirror_idx: &[u32],
        mirror_sign: &[f32],
    ) -> Result<Self, GpuBackendError> {
        let st = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let mut mask = vec![0.0f32; j];
        for &i in hip {
            if i < j {
                mask[i] = 1.0;
            }
        }
        Ok(Self {
            shader: RewardJointShader::from_backend(backend)?,
            params: Tensor::scalar(
                backend,
                shaders::RewardJointParams {
                    n_envs: n as u32,
                    num_joints: j as u32,
                    dt: 0.0,
                    w_pose: 0.0,
                    w_dof_limits: 0.0,
                    w_dof_vel: 0.0,
                    w_bilateral: 0.0,
                    sym_yaw_gate: 0.0,
                },
                BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )?,
            default_pos: Tensor::vector(backend, default_pos, st)?,
            soft_lo: Tensor::vector(backend, soft_lo, st)?,
            soft_hi: Tensor::vector(backend, soft_hi, st)?,
            hip_mask: Tensor::vector(backend, &mask, st)?,
            mirror_idx: Tensor::vector(backend, mirror_idx, st)?,
            mirror_sign: Tensor::vector(backend, mirror_sign, st)?,
            cmd_yaw: Tensor::vector(backend, &vec![0.0f32; n], st)?,
            out: Tensor::vector_uninit(backend, (4 * n) as u32, st | BufferUsages::COPY_SRC)?,
            n,
            j,
        })
    }

    /// Dispatch against the GPU joint state; returns `[3 x n]`.
    pub async fn compute(
        &mut self,
        backend: &GpuBackend,
        joints: &GpuJointState,
        dt: f32,
        w_pose: f32,
        w_dof_limits: f32,
        w_dof_vel: f32,
        w_bilateral: f32,
        sym_yaw_gate: f32,
        cmd_yaw: &[f32],
    ) -> Result<Vec<f32>, GpuBackendError> {
        backend.write_buffer(self.params.buffer_mut(), 0, &[shaders::RewardJointParams {
            n_envs: self.n as u32,
            num_joints: self.j as u32,
            dt,
            w_pose,
            w_dof_limits,
            w_dof_vel,
            w_bilateral,
            sym_yaw_gate,
        }])?;
        backend.write_buffer(self.cmd_yaw.buffer_mut(), 0, cmd_yaw)?;
        let mut enc = backend.begin_encoding();
        {
            let mut pass = enc.begin_pass("[reward] joint terms", None);
            self.shader.terms.call(
                &mut pass,
                self.n as u32,
                &self.params,
                joints.q_buffer(),
                joints.qd_buffer(),
                &self.default_pos,
                &self.soft_lo,
                &self.soft_hi,
                &self.hip_mask,
                &self.mirror_idx,
                &self.mirror_sign,
                &self.cmd_yaw,
                &mut self.out,
            )?;
        }
        backend.submit(enc)?;
        backend.slow_read_vec(self.out.buffer()).await
    }
}

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
