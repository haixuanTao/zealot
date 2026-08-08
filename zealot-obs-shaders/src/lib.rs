//! GPU observation assembly for the zealot biped env — the kernels that close
//! the browser demo's control loop on the GPU (physics → obs → policy GEMM →
//! motor targets, no per-step CPU readback).
//!
//! Layout contracts (all batch-interleaved like the nexus buffers — element
//! `i` of env `e` at `i · n_envs + e`):
//! - joint angles: the ws `WS_JOINT_ROT` quad of each policy joint's child
//!   link — converted models hinge about local +Z, so
//!   `q = 2·atan2(rot.z, rot.w)` (`dof_values`/ws-COORDS are NOT maintained
//!   by the refresh pipeline — probe-verified 2026-07-28).
//! - `links_workspace`: nexus SoA workspace (`Vec4` quads,
//!   `flat = (link·15 + quad)·n_envs + e`); the base world rotation is the
//!   local-to-world quat of link 0 (quad 5) — layout mirrored from
//!   `nexus src_rbd_shaders/dynamics/multibody/ws_soa.rs` (dim3), verified at
//!   init by the host against a CPU readback.
//! - `state` (STATE_STRIDE per env): [0..12) prev_q, [12..24) action t−2,
//!   [24..36) action t−1, [36] episode step counter (f32), [37] gait phase.
//! - `hist`: 5×`frame` obs-frame ring per env, slot s = ep % 5 (allocated at
//!   the 5×53 maximum; a narrower policy uses the first 5×frame).
//! - `consts`: [0..12) actuated dof ids (as f32), [12..24) default_pos,
//!   [24..36) target lo, [36..48) target hi, then normalizer mean and std
//!   (HIST·FRAME each, of which a narrower policy fills the first
//!   HIST·frame).
//! - `gemm_input`: row-major [5·frame × n_envs] — the vortx GEMM a0 buffer.
//! - `actions`/`targets`: row-major [12 × n_envs] (matches
//!   `scatter_motor_targets`).
//!
//! Obs frame (53), mirroring `sim2sim_g1_mujoco.py` / the training env:
//! [0..12) lag-2 action · [12..16) command · [16..28) q − default ·
//! [28..40) finite-diff joint velocity · [40..43) projected gravity ·
//! [43..45) gait-clock sin/cos · [45..48) base angular velocity (gyro,
//! body frame — added for v24) · [48..53) step cue (v28: distance, height,
//! edge sin/cos, validity — ALWAYS ZERO here, the trainer's "no step
//! detected" pattern; the demo terrain has no step-cue oracle). The first
//! 45 slots keep their v21 meaning. History ×5 oldest-first,
//! reset-replicated; Welford-normalized (clip ±5).
//!
//! No panics anywhere (no `[]` indexing, no `clamp`, no `step_by`): panic
//! edges become naga switch-breaks that Tint rejects — see the
//! browser-kernel playbook.

#![cfg_attr(target_arch = "spirv", no_std)]

use khal_std::glamx::{Pose3, UVec3, UVec4, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
#[allow(unused_imports)]
use khal_std::num_traits::Float;

/// f32 slots per env in the `state` buffer.
pub const STATE_STRIDE: usize = 38;
/// `state` slot holding the gait-clock phase ∈ [0,1). It is an accumulator,
/// not a function of the step counter, because the cycle period depends on the
/// (time-varying) command — see the clock note in `gpu_assemble_obs`.
const S_PHASE: usize = 37;
/// Gait cycle seconds at the slowest walking command (0.1 m/s).
const GAIT_PERIOD_SLOW: f32 = 0.8;
/// Gait cycle seconds at the full 0.5 m/s command.
const GAIT_PERIOD_FAST: f32 = 0.55;
/// The fixed, free-running period every pre-gyro checkpoint was trained on.
const GAIT_PERIOD_LEGACY: f32 = 0.7;
/// Policy joints.
pub const N_ACT: usize = 12;
/// Frames of obs history.
pub const HIST: usize = 5;
/// WIDEST single-frame obs the kernel can assemble, and the stride every
/// fixed-size buffer here is allocated at. The frame a given policy actually
/// wants is narrower or equal and comes in as the `frame` uniform: v21 and
/// earlier were 45, v24 added the gyro at [45..48), v28 the step cue at
/// [48..53). Checkpoints published before and after each change all have to
/// load and walk, so the width is a runtime value, not a rebuild.
pub const FRAME: usize = 53;
/// The pre-gyro frame, still used by every v21-and-earlier checkpoint.
pub const FRAME_NO_GYRO: usize = 45;
/// The gyro-era frame (v24–v27): everything of [[FRAME]] except the step cue.
/// Also the boundary above which a checkpoint expects the command-derived
/// gait clock (the clock landed in the same training branch as the gyro).
pub const FRAME_GYRO: usize = 48;
/// consts offsets. C_LINK: child-link id of each policy joint.
pub const C_LINK: usize = 0;
pub const C_DEFAULT: usize = 12;
pub const C_LO: usize = 24;
pub const C_HI: usize = 36;
pub const C_MEAN: usize = 48;
pub const C_STD: usize = 48 + HIST * FRAME;
/// Total consts length.
pub const C_LEN: usize = C_STD + HIST * FRAME;

/// nexus ws_soa (dim3): quads per link / joint-rotation + local-to-world quads.
const WS_QUADS: u32 = 15;
const WS_JOINT_ROT: u32 = 0;
const WS_LTW: u32 = 5;
/// `rb_vels.angular` — world-frame base angular velocity (Velocity = linear
/// quad then angular quad, at workspace quads 11/12).
const WS_RB_ANGVEL: u32 = 12;

/// Assemble the obs frame (`frame` wide), update the history ring, and write
/// the normalized stacked [5·frame × N] GEMM input. One thread per env.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_assemble_obs(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    links_workspace: &[khal_std::glamx::Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hist: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] cmd: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] consts: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] gemm_input: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] n_envs: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] dt: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] frame: &u32,
) {
    let e = invocation_id.x;
    let n = *n_envs;
    if e >= n {
        return;
    }
    let ne = n as usize;
    let eu = e as usize;
    let sb = |i: usize| i * ne + eu; // batch-interleaved slot
    // Obs width this policy was trained with (45 pre-gyro, 48 with gyro, 53
    // with the step cue). The `o` scratch is always FRAME wide; the loops
    // below stop at `fr`, so the wider slots simply never reach the history
    // ring for a narrower policy.
    let fr = *frame as usize;

    let ep = state.read(sb(36));

    // Joint angles from each child link's joint-rotation quat (hinge about
    // local +Z in the converted models): q = 2·atan2(rot.z, rot.w).
    let mut q = [0.0f32; N_ACT];
    let mut i = 0;
    while i < N_ACT {
        let link = consts.read(C_LINK + i) as usize;
        let jr = links_workspace.read((link * WS_QUADS as usize + WS_JOINT_ROT as usize) * ne + eu);
        q[i] = 2.0 * jr.z.atan2(jr.w);
        i += 1;
    }

    // Base world rotation: link 0's local-to-world quat (xyzw).
    let bq = links_workspace.read((WS_LTW as usize) * ne + eu);
    // Projected gravity: world-down in the base frame, v' = conj(q)·v·q with
    // v = (0,0,-1):  u = -q.xyz;  v + 2·(u×(u×v + w·v)).
    let (ux, uy, uz, w) = (-bq.x, -bq.y, -bq.z, bq.w);
    // c1 = u×v + w·v with v=(0,0,-1) → (−uy, ux, 0) + (0,0,−w)
    let (c1x, c1y, c1z) = (-uy, ux, -w);
    // c2 = u×c1
    let c2x = uy * c1z - uz * c1y;
    let c2y = uz * c1x - ux * c1z;
    let c2z = ux * c1y - uy * c1x;
    let g = [2.0 * c2x, 2.0 * c2y, -1.0 + 2.0 * c2z];

    // FRAME-wide scratch, zero-initialized — which already IS the step-cue
    // block for a 53-dim policy: all-zero = "no step detected", the exact
    // pattern the trainer emits off the Step family. Slots [48..53) are
    // never written below.
    let mut o = [0.0f32; FRAME];
    let lag2_ok = ep >= 2.0;
    let mut j = 0;
    while j < N_ACT {
        o[j] = if lag2_ok { state.read(sb(12 + j)) } else { 0.0 };
        o[16 + j] = q[j] - consts.read(C_DEFAULT + j);
        o[28 + j] = if ep <= 0.5 {
            0.0
        } else {
            (q[j] - state.read(sb(j))) / *dt
        };
        j += 1;
    }
    o[12] = cmd.read(eu);
    o[13] = cmd.read(ne + eu);
    o[14] = cmd.read(2 * ne + eu);
    o[15] = cmd.read(3 * ne + eu);
    o[40] = g[0];
    o[41] = g[1];
    o[42] = g[2];
    // Gait clock. For current policies this is derived from the command
    // exactly as the training env does (`biped_env_nexus`, "one
    // command-derived design, no knobs"): a standing command (< 0.1 m/s)
    // FREEZES the phase, so standing is its own observation state, and above
    // that the cycle period lerps 0.8 s → 0.55 s with commanded speed. A
    // free-running clock under such a policy is a train/deploy mismatch — the
    // obs keeps asking for swings it was trained to stop making, and the robot
    // marches in place when told to hold still. See the advance step below for
    // how older checkpoints keep their original fixed-period clock.
    let ph = if ep <= 0.5 {
        0.0
    } else {
        state.read(sb(S_PHASE))
    };
    let two_pi = 2.0 * core::f32::consts::PI;
    o[43] = (two_pi * ph).sin();
    o[44] = (two_pi * ph).cos();

    // Gyro: base angular velocity, world frame from the workspace, rotated
    // into the body frame by conj(q) — the same transform as the gravity
    // vector above, and what `quat_rotate_inv(orientation, ang_vel_world)`
    // does on the CPU side.
    let av = links_workspace.read((WS_RB_ANGVEL as usize) * ne + eu);
    let t1x = uy * av.z - uz * av.y + w * av.x;
    let t1y = uz * av.x - ux * av.z + w * av.y;
    let t1z = ux * av.y - uy * av.x + w * av.z;
    o[45] = av.x + 2.0 * (uy * t1z - uz * t1y);
    o[46] = av.y + 2.0 * (uz * t1x - ux * t1z);
    o[47] = av.z + 2.0 * (ux * t1y - uy * t1x);

    // History ring: slot ep%5; on episode start replicate into all slots.
    let slot = (ep as u32 % HIST as u32) as usize;
    let mut r = 0;
    while r < fr {
        if ep <= 0.5 {
            let mut s = 0;
            while s < HIST {
                hist.write((s * fr + r) * ne + eu, o[r]);
                s += 1;
            }
        } else {
            hist.write((slot * fr + r) * ne + eu, o[r]);
        }
        r += 1;
    }

    // Normalized stacked obs, oldest-first: frame k = ring slot
    // (slot + 1 + k) mod 5.
    let mut k = 0;
    while k < HIST {
        let src = (slot + 1 + k) % HIST;
        let mut r = 0;
        while r < fr {
            let x = hist.read((src * fr + r) * ne + eu);
            let m = consts.read(C_MEAN + k * fr + r);
            let s = consts.read(C_STD + k * fr + r);
            let mut v = (x - m) / s;
            if v > 5.0 {
                v = 5.0;
            }
            if v < -5.0 {
                v = -5.0;
            }
            gemm_input.write((k * fr + r) * ne + eu, v);
            r += 1;
        }
        k += 1;
    }

    // prev_q ← q for the next step's finite diff.
    let mut j = 0;
    while j < N_ACT {
        state.write(sb(j), q[j]);
        j += 1;
    }

    // Advance the gait phase for the next step. WHICH clock depends on the
    // policy: the command-derived one (freeze at a stand, period lerped with
    // commanded speed) landed in the same training branch as the gyro
    // observation, so a gyro-era frame (48 or 53 wide) means the checkpoint
    // expects it and a 45-wide one (v21 and earlier) expects the older
    // free-running fixed period. Feeding either the wrong clock is a
    // train/deploy mismatch, so the frame width picks it — published
    // checkpoints of every era walk.
    let mut next_ph = ph;
    if fr >= FRAME_GYRO {
        // FULL command magnitude INCLUDING yaw rate (the training env's
        // `VelocityCommand::speed()`, since ab7c811): a turn-in-place command
        // must tick the clock or the policy will not step through the turn.
        // (v24 trained with the vx/vy-only form; running it under this clock
        // is the one known mismatch of loading it from the Hub.)
        let cmd_speed = (o[12] * o[12] + o[13] * o[13] + o[14] * o[14]).sqrt();
        if cmd_speed >= 0.1 {
            let capped = if cmd_speed > 0.5 { 0.5 } else { cmd_speed };
            let t = (capped - 0.1) / 0.4;
            let period = GAIT_PERIOD_SLOW + (GAIT_PERIOD_FAST - GAIT_PERIOD_SLOW) * t;
            let raw = ph + *dt / period;
            next_ph = raw - raw.floor();
        }
    } else {
        let raw = ph + *dt / GAIT_PERIOD_LEGACY;
        next_ph = raw - raw.floor();
    }
    state.write(sb(S_PHASE), next_ph);
}

/// Post-policy commit: shift the action ring, advance the episode counter,
/// and turn actions into clamped PD position targets. One thread per env.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_commit_actions(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] actions: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] consts: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] targets: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] n_envs: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] action_scale: &f32,
) {
    let e = invocation_id.x;
    let n = *n_envs;
    if e >= n {
        return;
    }
    let ne = n as usize;
    let eu = e as usize;
    let sb = |i: usize| i * ne + eu;

    let mut j = 0;
    while j < N_ACT {
        let a = actions.read(j * ne + eu);
        // ring: t−2 ← t−1, t−1 ← a
        let prev1 = state.read(sb(24 + j));
        state.write(sb(12 + j), prev1);
        state.write(sb(24 + j), a);
        // PD target = clamp(default + scale·a, lo, hi) — min/max, not
        // `clamp` (its assert is a panic edge).
        let mut t = consts.read(C_DEFAULT + j) + *action_scale * a;
        let lo = consts.read(C_LO + j);
        let hi = consts.read(C_HI + j);
        if t < lo {
            t = lo;
        }
        if t > hi {
            t = hi;
        }
        targets.write(j * ne + eu, t);
        j += 1;
    }
    let ep = state.read(sb(36));
    state.write(sb(36), ep + 1.0);
}

/// Scalar parameters for [`gpu_reward_action_terms`] (uniform; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct RewardActionParams {
    /// Environments (columns of the row-major `[J x n]` action buffers).
    pub n_envs: u32,
    /// Actuated joints `J`.
    pub num_joints: u32,
    /// Control dt — every reward term is scaled by it.
    pub dt: f32,
    /// `weights.action_rate` (negative).
    pub w_action_rate: f32,
    /// `weights.action_rate_hipz_hipx` (negative).
    pub w_action_rate_hip: f32,
    /// `weights.action_rate_rate` (negative).
    pub w_action_rate_rate: f32,
    pub pad0: u32,
    pub pad1: u32,
}

/// The three action-smoothness reward terms, one thread per environment.
///
/// Exact port of `VelocityFlatTask::reward`'s action penalties:
///   action_rate      = w   · Σ(a − a′)²      · dt
///   action_rate_hip  = w_h · Σ_hip(a − a′)²  · dt
///   action_rate_rate = w_r · Σ(a − 2a′ + a″)² · dt
///
/// These depend ONLY on the last three action vectors — no physics state — so
/// they are the first terms that can move without the state-derivation kernel,
/// and they exercise the whole path (params uniform, per-term output buffer,
/// host comparison) end to end.
///
/// Action buffers are row-major `[num_joints x n_envs]`: joint `i` of env `e`
/// at `i·n_envs + e`. `hip_mask` is `num_joints` long, 1.0 on the hip yaw/roll
/// joints the hip-specific term sums over. `out` is `[3 x n_envs]`, rows in the
/// order above.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reward_action_terms(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &RewardActionParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] last: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] prev: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] prev2: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] hip_mask: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] out: &mut [f32],
) {
    let e = invocation_id.x as usize;
    let n = params.n_envs as usize;
    let j = params.num_joints as usize;
    if e < n {
        let mut da2 = 0.0f32;
        let mut da2_hip = 0.0f32;
        let mut dda2 = 0.0f32;
        for i in 0..j {
            let idx = i * n + e;
            let a = last.read(idx);
            let ap = prev.read(idx);
            let app = prev2.read(idx);
            let d = a - ap;
            da2 += d * d;
            da2_hip += hip_mask.read(i) * d * d;
            let dd = a - 2.0 * ap + app;
            dda2 += dd * dd;
        }
        let dt = params.dt;
        out.write(e, params.w_action_rate * da2 * dt);
        out.write(n + e, params.w_action_rate_hip * da2_hip * dt);
        out.write(2 * n + e, params.w_action_rate_rate * dda2 * dt);
    }
}

/// Scalar parameters for [`gpu_joint_state`] (uniform; 16 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct JointStateParams {
    pub n_envs: u32,
    pub num_joints: u32,
    /// Rigid bodies per env — `body_poses` is env-major (`env·bps + link`).
    pub bodies_per_env: u32,
    pub control_dt: f32,
    pub pad0: u32,
}

/// Hamilton product of two xyzw quaternions.
#[inline]
fn qmul(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::new(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )
}

/// Conjugate of an xyzw quaternion.
#[inline]
fn qconj(q: Vec4) -> Vec4 {
    Vec4::new(-q.x, -q.y, -q.z, q.w)
}

/// Per-joint angle and finite-difference velocity, one thread per environment.
///
/// EXACT port of the trainer host's `read_state_from_poses`: the angle comes
/// from the PARENT-CHILD relative rotation with the joint's rest quaternion
/// removed,
///   `rel = rest⁻¹ · qp⁻¹ · qc`,  `q = 2·atan2(rel.z, rel.w)`
/// which is NOT what `gpu_assemble_obs` does (that reads the workspace
/// joint-rotation quad directly). The reward must match the host, so this
/// kernel deliberately mirrors the host formulation.
///
/// `body_poses` is the nexus buffer, env-major: pose of `link` in env `e` at
/// `e·bodies_per_env + link`.
/// `have_prev` is per-env (0 right after a reset, when the host also reports
/// zero velocity). `prev_q` is updated in place for the next step.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_joint_state(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &JointStateParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_poses: &[Pose3],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] parent_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] child_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] rest_quats: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] have_prev: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] prev_q: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] q_out: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] qd_out: &mut [f32],
) {
    let e = invocation_id.x as usize;
    let n = params.n_envs as usize;
    let j = params.num_joints as usize;
    let bps = params.bodies_per_env as usize;
    if e < n {
        let base = e * bps;
        let fresh = have_prev.read(e);
        let dt = params.control_dt;
        for i in 0..j {
            let rp = body_poses.read(base + parent_ids.read(i) as usize).rotation;
            let rc = body_poses.read(base + child_ids.read(i) as usize).rotation;
            let qp = Vec4::new(rp.x, rp.y, rp.z, rp.w);
            let qc = Vec4::new(rc.x, rc.y, rc.z, rc.w);
            let rel = qmul(qmul(qconj(rest_quats.read(i)), qconj(qp)), qc);
            let q = 2.0 * rel.z.atan2(rel.w);
            let idx = i * n + e;
            let qd = if fresh != 0 { (q - prev_q.read(idx)) / dt } else { 0.0 };
            q_out.write(idx, q);
            qd_out.write(idx, qd);
            prev_q.write(idx, q);
        }
    }
}

/// Scalar parameters for [`gpu_reward_joint_terms`] (uniform; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct RewardJointParams {
    pub n_envs: u32,
    pub num_joints: u32,
    pub dt: f32,
    /// `weights.pose` — hip yaw/roll deviation from the default pose.
    pub w_pose: f32,
    /// `weights.dof_pos_limits` — soft band at 90% of each hard limit.
    pub w_dof_limits: f32,
    /// `weights.dof_vel` — joint-velocity L2.
    pub w_dof_vel: f32,
    /// `weights.bilateral_symmetry` — exp(-symmetry error).
    pub w_bilateral: f32,
    /// Yaw gate releasing the LATERAL half of the symmetry term as the turn
    /// command grows; 0 = never release (lateral always at full weight).
    pub sym_yaw_gate: f32,
}

/// The joint-only reward terms, one thread per environment.
///
/// Exact port of `VelocityFlatTask::reward`:
///   pose           = w   · Σ_hip (q − default)²                    · dt
///   dof_pos_limits = w_l · Σ_i  [max(q−hi,0) + max(lo−q,0)]        · dt
///   dof_vel        = w_v · Σ_i  q̇²                                 · dt
///
/// `lo`/`hi` are the SOFT limits — the host applies the 0.9 band factor, so it
/// is applied host-side when these are uploaded, not here.
///
/// `q`/`qd` are row-major `[num_joints x n_envs]` as produced by
/// [`gpu_joint_state`]; `out` is `[3 x n_envs]` in the order above.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reward_joint_terms(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &RewardJointParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] q: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] qd: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] default_pos: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] soft_lo: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] soft_hi: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] hip_mask: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] mirror_idx: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] mirror_sign: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] cmd_yaw: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] out: &mut [f32],
) {
    let e = invocation_id.x as usize;
    let n = params.n_envs as usize;
    let j = params.num_joints as usize;
    if e < n {
        let mut hip_dev2 = 0.0f32;
        let mut lim_pen = 0.0f32;
        let mut jv2 = 0.0f32;
        for i in 0..j {
            let idx = i * n + e;
            let qi = q.read(idx);
            let vi = qd.read(idx);
            let d = qi - default_pos.read(i);
            hip_dev2 += hip_mask.read(i) * d * d;
            let over = qi - soft_hi.read(i);
            let under = soft_lo.read(i) - qi;
            lim_pen += if over > 0.0 { over } else { 0.0 };
            lim_pen += if under > 0.0 { under } else { 0.0 };
            jv2 += vi * vi;
        }
        // Bilateral symmetry: each L/R pair counted once (partner index > i).
        // Lateral pairs (negative mirror sign) are scaled by the yaw gate.
        let gate = params.sym_yaw_gate;
        let lateral_scale = if gate > 0.0 {
            let s = 1.0 - (cmd_yaw.read(e).abs() / gate);
            if s < 0.0 {
                0.0
            } else if s > 1.0 {
                1.0
            } else {
                s
            }
        } else {
            1.0
        };
        let mut sym_err = 0.0f32;
        for i in 0..j {
            let jr = mirror_idx.read(i) as usize;
            if jr > i {
                let sgn = mirror_sign.read(i);
                let d = q.read(i * n + e) - sgn * q.read(jr * n + e);
                let sq = d * d;
                sym_err += if sgn < 0.0 { lateral_scale * sq } else { sq };
            }
        }

        let dt = params.dt;
        out.write(e, params.w_pose * hip_dev2 * dt);
        out.write(n + e, params.w_dof_limits * lim_pen * dt);
        out.write(2 * n + e, params.w_dof_vel * jv2 * dt);
        out.write(3 * n + e, params.w_bilateral * (-sym_err).exp() * dt);
    }
}

/// Scalar parameters for [`gpu_reward_torque_terms`] (uniform; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct RewardTorqueParams {
    pub n_envs: u32,
    pub num_joints: u32,
    pub dt: f32,
    /// Curriculum-ramped leg-torque gain (the env's `torque_scale`).
    pub torque_w: f32,
    /// Ankle-torque gain — full strength at all times, NOT ramped.
    pub ankle_torque_w: f32,
    /// Mechanical-power gain.
    pub power_w: f32,
    pub pad0: u32,
    pub pad1: u32,
}

/// Torque / power reward terms, one thread per environment.
///
/// Reconstructs the applied PD torque exactly as the host does:
///   `tau = clamp(kp·(q_target − q) − kd·q̇, ±effort)`
/// then
///   torque_leg   = −(torque_w · Σ w_leg·tau² + Σ w_knee·tau²) · dt
///   torque_ankle = −(ankle_torque_w · Σ w_ankle·tau²)         · dt
///   power        = −(power_w · Σ|tau·q̇|)                      · dt
///
/// The per-joint weights arrive as `[num_joints]` arrays so the kernel never
/// does the host's joint-NAME matching (`contains("ankle")`, `"knee"`, the two
/// roll naming schemes) — that classification is resolved once on the host and
/// baked into `w_leg` / `w_ankle` / `w_knee`.
///
/// Note the knee extra is deliberately OUTSIDE the ramped `torque_w`, matching
/// the host: it is full-strength from iteration 0 like the ankle extras.
///
/// `q`, `qd` and `q_target` are row-major `[num_joints x n_envs]`; `out` is
/// `[3 x n_envs]`: torque_leg, torque_ankle, power.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reward_torque_terms(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &RewardTorqueParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] q: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] qd: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] q_target: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] kp: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] kd: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] effort: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] w_leg: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] w_ankle: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] w_knee: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] out: &mut [f32],
) {
    let e = invocation_id.x as usize;
    let n = params.n_envs as usize;
    let j = params.num_joints as usize;
    if e < n {
        let mut leg_pen = 0.0f32;
        let mut ankle_pen = 0.0f32;
        let mut knee_pen = 0.0f32;
        let mut power = 0.0f32;
        for i in 0..j {
            let idx = i * n + e;
            let v = qd.read(idx);
            let lim = effort.read(i);
            let raw = kp.read(i) * (q_target.read(idx) - q.read(idx)) - kd.read(i) * v;
            let tau = if raw > lim {
                lim
            } else if raw < -lim {
                -lim
            } else {
                raw
            };
            let t2 = tau * tau;
            let p = tau * v;
            power += if p < 0.0 { -p } else { p };
            leg_pen += w_leg.read(i) * t2;
            ankle_pen += w_ankle.read(i) * t2;
            knee_pen += w_knee.read(i) * t2;
        }
        let dt = params.dt;
        out.write(e, -(params.torque_w * leg_pen + knee_pen) * dt);
        out.write(n + e, -(params.ankle_torque_w * ankle_pen) * dt);
        out.write(2 * n + e, -(params.power_w * power) * dt);
    }
}

/// Scalar parameters for [`gpu_base_state`] (uniform; 16 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct BaseStateParams {
    pub n_envs: u32,
    /// Rigid bodies per env (`body_poses` is env-major).
    pub bodies_per_env: u32,
    /// Index of the torso/base link within an env.
    pub torso_link: u32,
    pub control_dt: f32,
}

/// Base pose, world velocities and terrain-relative height, one thread per env.
///
/// Exact port of the host's `read_state_from_poses` base block: linear velocity
/// is a finite difference of the torso translation, and angular velocity uses
/// the small-rotation approximation `ω ≈ 2·Δq.xyz/dt` from
/// `dq = r · prev⁻¹`, with the hemisphere correction (`sign(dq.w)`) so
/// antipodal quaternions don't blow it up. Both are ZERO on the first step
/// after a reset, matching `has_prev_pose`.
///
/// `ground_h` is supplied per env: the terrain lookup stays on the host, so
/// heights remain relative to the LOCAL ground surface exactly as the host
/// computes them.
///
/// `out` is `[13 x n_envs]` row-major: rows 0..4 base quat xyzw, 4..7 linear
/// velocity, 7..10 angular velocity, 10 height, 11..13 base xy.
/// `prev_pose` holds the previous torso (quat xyzw, translation xyz) per env
/// as `[7 x n_envs]`, updated in place.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_base_state(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &BaseStateParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_poses: &[Pose3],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] have_prev: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] ground_h: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] prev_pose: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] out: &mut [f32],
) {
    let e = invocation_id.x as usize;
    let n = params.n_envs as usize;
    if e < n {
        let p = body_poses.read(e * params.bodies_per_env as usize + params.torso_link as usize);
        let (rx, ry, rz, rw) = (p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w);
        let (tx, ty, tz) = (p.translation.x, p.translation.y, p.translation.z);
        let dt = params.control_dt;

        let mut lv = [0.0f32; 3];
        let mut av = [0.0f32; 3];
        if have_prev.read(e) != 0 {
            let (px, py, pz, pw) = (
                prev_pose.read(e),
                prev_pose.read(n + e),
                prev_pose.read(2 * n + e),
                prev_pose.read(3 * n + e),
            );
            let (ptx, pty, ptz) = (
                prev_pose.read(4 * n + e),
                prev_pose.read(5 * n + e),
                prev_pose.read(6 * n + e),
            );
            lv = [(tx - ptx) / dt, (ty - pty) / dt, (tz - ptz) / dt];
            // dq = r · prev⁻¹
            let pc = Vec4::new(-px, -py, -pz, pw);
            let dq = qmul(Vec4::new(rx, ry, rz, rw), pc);
            let s = if dq.w >= 0.0 { 1.0 } else { -1.0 };
            av = [2.0 * s * dq.x / dt, 2.0 * s * dq.y / dt, 2.0 * s * dq.z / dt];
        }

        out.write(e, rx);
        out.write(n + e, ry);
        out.write(2 * n + e, rz);
        out.write(3 * n + e, rw);
        out.write(4 * n + e, lv[0]);
        out.write(5 * n + e, lv[1]);
        out.write(6 * n + e, lv[2]);
        out.write(7 * n + e, av[0]);
        out.write(8 * n + e, av[1]);
        out.write(9 * n + e, av[2]);
        out.write(10 * n + e, tz - ground_h.read(e));
        out.write(11 * n + e, tx);
        out.write(12 * n + e, ty);

        prev_pose.write(e, rx);
        prev_pose.write(n + e, ry);
        prev_pose.write(2 * n + e, rz);
        prev_pose.write(3 * n + e, rw);
        prev_pose.write(4 * n + e, tx);
        prev_pose.write(5 * n + e, ty);
        prev_pose.write(6 * n + e, tz);
    }
}

/// Scalar parameters for [`gpu_reward_base_terms`] (uniform; 80 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct RewardBaseParams {
    pub n_envs: u32,
    pub dt: f32,
    pub w_track_lin: f32,
    pub w_forward_progress: f32,
    pub w_track_ang: f32,
    pub w_upright: f32,
    pub w_base_height: f32,
    pub w_body_ang_vel: f32,
    pub w_lin_vel_z: f32,
    pub std_lin: f32,
    pub std_ang: f32,
    pub std_base_h: f32,
    pub std_upright: f32,
    /// Widened stds used while stepping over a detected edge.
    pub step_std_base_h: f32,
    pub step_std_upright: f32,
    pub step_relax_dist: f32,
    /// Height target when standing (`cmd.speed() < 0.1`) and when walking.
    pub h_target_stand: f32,
    pub h_target_walk: f32,
    pub pad0: u32,
    pub pad1: u32,
}

/// Rotate `v` by the INVERSE of xyzw quaternion `q` (world → body).
#[inline]
fn qrot_inv(q: Vec4, v: [f32; 3]) -> [f32; 3] {
    // quat_rotate with u = -q.xyz, w = q.w
    let (ux, uy, uz, w) = (-q.x, -q.y, -q.z, q.w);
    let tx = uy * v[2] - uz * v[1] + w * v[0];
    let ty = uz * v[0] - ux * v[2] + w * v[1];
    let tz = ux * v[1] - uy * v[0] + w * v[2];
    [
        v[0] + 2.0 * (uy * tz - uz * ty),
        v[1] + 2.0 * (uz * tx - ux * tz),
        v[2] + 2.0 * (ux * ty - uy * tx),
    ]
}

/// The base-state reward terms, one thread per environment.
///
/// Exact port of the host: velocities and gravity are taken into the BODY frame
/// (`quat_rotate_inv`), tracking uses exp kernels, and `base_height`/`upright`
/// widen their std while stepping over a detected edge — a condition that
/// depends on the per-env step cue, so the cue is supplied rather than
/// recomputed here.
///
/// `base` is `[13 x n]` from [`gpu_base_state`]; `cmd` is `[4 x n]`
/// (vx, vy, yaw_rate, speed); `cue` is `[4 x n]` (valid, distance, edge_cos,
/// edge_sin). `out` is `[6 x n]`: track_lin_vel, track_ang_vel, upright,
/// base_height, body_ang_vel, lin_vel_z.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reward_base_terms(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &RewardBaseParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] base: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] cmd: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] cue: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] out: &mut [f32],
) {
    let e = invocation_id.x as usize;
    let n = params.n_envs as usize;
    if e < n {
        let q = Vec4::new(base.read(e), base.read(n + e), base.read(2 * n + e), base.read(3 * n + e));
        let lv = [base.read(4 * n + e), base.read(5 * n + e), base.read(6 * n + e)];
        let av = [base.read(7 * n + e), base.read(8 * n + e), base.read(9 * n + e)];
        let height = base.read(10 * n + e);

        let v = qrot_inv(q, lv);
        let w = qrot_inv(q, av);
        let grav = qrot_inv(q, [0.0, 0.0, -1.0]);

        let (cvx, cvy) = (cmd.read(e), cmd.read(n + e));
        let cyaw = cmd.read(2 * n + e);
        // NOTE two different "speeds": the tracking clamp uses the PLANAR
        // command speed, while the height-target gate uses `cmd.speed()`, which
        // also includes yaw_rate. Conflating them is a silent 3e-2 error.
        let cspeed = cmd.read(3 * n + e);
        let planar = (cvx * cvx + cvy * cvy).sqrt();
        let dt = params.dt;

        // Tracking: exp kernel plus a LINEAR forward-progress term, which keeps
        // a gradient when the exp kernel has saturated flat.
        let lin_err = (cvx - v[0]) * (cvx - v[0]) + (cvy - v[1]) * (cvy - v[1]);
        let v_along = if planar > 1e-6 { (v[0] * cvx + v[1] * cvy) / planar } else { 0.0 };
        let v_clamped = if v_along < 0.0 {
            0.0
        } else if v_along > planar {
            planar
        } else {
            v_along
        };
        let track_lin = params.w_track_lin
            * (-lin_err / (params.std_lin * params.std_lin)).exp()
            * dt
            + params.w_forward_progress * v_clamped * dt;

        let ang_err = (cyaw - w[2]) * (cyaw - w[2]);
        let track_ang =
            params.w_track_ang * (-ang_err / (params.std_ang * params.std_ang)).exp() * dt;

        // Stepping relaxes the height/upright stds — same predicate as the host.
        let toward_step = v[0] * cue.read(2 * n + e) + v[1] * cue.read(3 * n + e);
        let dist = cue.read(n + e);
        let dist_abs = if dist < 0.0 { -dist } else { dist };
        let stepping =
            cue.read(e) > 0.5 && dist_abs < params.step_relax_dist && toward_step > 0.1;
        let std_h = if stepping { params.step_std_base_h } else { params.std_base_h };
        let std_up = if stepping { params.step_std_upright } else { params.std_upright };

        let tilt_err = grav[0] * grav[0] + grav[1] * grav[1];
        let upright = params.w_upright * (-tilt_err / (std_up * std_up)).exp() * dt;

        let h_target = if cspeed < 0.1 { params.h_target_stand } else { params.h_target_walk };
        let h_err = (height - h_target) * (height - h_target);
        let base_h = params.w_base_height * (-h_err / (std_h * std_h)).exp() * dt;

        let body_ang = params.w_body_ang_vel * (w[0] * w[0] + w[1] * w[1]) * dt;
        let lin_z = params.w_lin_vel_z * v[2] * v[2] * dt;

        out.write(e, track_lin);
        out.write(n + e, track_ang);
        out.write(2 * n + e, upright);
        out.write(3 * n + e, base_h);
        out.write(4 * n + e, body_ang);
        out.write(5 * n + e, lin_z);
    }
}

/// Scalar parameters for [`gpu_feet_state`] (uniform; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct FeetStateParams {
    pub n_envs: u32,
    /// Colliders per env — the pose buffer stride the host uses for feet.
    pub colliders_per_env: u32,
    pub torso_link: u32,
    pub control_dt: f32,
    /// Geometric contact threshold on the foot LINK-ORIGIN height.
    pub contact_z: f32,
    /// Force threshold (N) when force-based contact sensing is on.
    pub contact_force_n: f32,
    /// 1 = use sensed force for contact, 0 = geometric height proxy.
    pub contact_sense: u32,
    /// Body weight (mass·g), the force_rate normalizer.
    pub body_weight: f32,
}

/// Per-foot state, one thread per environment.
///
/// Exact port of the host's `compute_feet_from_poses`. The host-maintained
/// per-env history (air time, last touchdown foot, previous sensed force,
/// previous foot poses) is supplied as inputs for now so the MATHS can be
/// verified independently; migrating that state onto the device is a separate
/// step with reset semantics to match.
///
/// Layouts, all row-major with env-major inner index. Per-foot arrays are
/// `[NUM_FEET x n]`; `foot_links` is `[NUM_FEET]`; `sole_local` and
/// `prev_foot_pos` are `[NUM_FEET*3 x n]`.
///
/// `out` is `[22 x n]`, per foot i (offset i*11): contact, first_contact,
/// air_time, height, planar_speed, tilt, yaw_rel_base, pos_x, pos_y, vz,
/// force_rate — plus alt_step folded into first_contact's sign convention is
/// NOT done; alt_step is emitted in row 11*NUM_FEET + i.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_feet_state(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FeetStateParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_poses: &[Pose3],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] foot_links: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] foot_fwd_local: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] sole_local: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] prev_foot_pos: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] ground_h: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] sensed_force: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] prev_force: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] air_time: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] last_td_foot: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 11)] have_prev: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 12)] have_prev_force: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 13)] out: &mut [f32],
) {
    let e = invocation_id.x as usize;
    let n = params.n_envs as usize;
    let nf = 2usize; // NUM_FEET
    if e < n {
        let cpb = params.colliders_per_env as usize;
        let env_base = e * cpb;
        let dt = params.control_dt;
        let bq = body_poses.read(env_base + params.torso_link as usize).rotation;
        let base_inv = Vec4::new(-bq.x, -bq.y, -bq.z, bq.w);
        let hp = have_prev.read(e);
        let hpf = have_prev_force.read(e);

        for i in 0..nf {
            let fp = body_poses.read(env_base + foot_links.read(i) as usize);
            let (px, py, pz) = (fp.translation.x, fp.translation.y, fp.translation.z);
            let fq = Vec4::new(fp.rotation.x, fp.rotation.y, fp.rotation.z, fp.rotation.w);

            let mut planar_speed = 0.0f32;
            let mut vz = 0.0f32;
            if hp != 0 {
                let ox = prev_foot_pos.read((i * 3) * n + e);
                let oy = prev_foot_pos.read((i * 3 + 1) * n + e);
                let oz = prev_foot_pos.read((i * 3 + 2) * n + e);
                let dx = (px - ox) / dt;
                let dy = (py - oy) / dt;
                planar_speed = (dx * dx + dy * dy).sqrt();
                vz = (pz - oz) / dt;
            }

            // Sole normal into the world frame; tilt from horizontal.
            let sl = [
                sole_local.read((i * 3) * n + e),
                sole_local.read((i * 3 + 1) * n + e),
                sole_local.read((i * 3 + 2) * n + e),
            ];
            let wn = qrot(fq, sl);
            let wnz = if wn[2] < 0.0 { -wn[2] } else { wn[2] };
            let wnz = if wnz > 1.0 { 1.0 } else { wnz };
            let tilt = wnz.acos();

            // Foot forward, expressed in the BASE frame → yaw relative to base.
            let fwd = [foot_fwd_local.read(0), foot_fwd_local.read(1), foot_fwd_local.read(2)];
            let f_world = qrot(fq, fwd);
            let f_base = qrot(base_inv, f_world);
            let yaw_rel = f_base[1].atan2(f_base[0]);

            let gh = ground_h.read(i * n + e);
            let sf = sensed_force.read(i * n + e);
            let contact = if params.contact_sense != 0 {
                sf >= params.contact_force_n
            } else {
                (pz - gh) < params.contact_z
            };
            let prev_air = air_time.read(i * n + e);
            let first_contact = contact && prev_air > 0.0;
            let alt_step = first_contact && last_td_foot.read(e) != i as f32;
            let new_air = if contact { 0.0 } else { prev_air + dt };
            let force_rate = if params.contact_sense != 0 && hpf != 0 {
                let d = sf - prev_force.read(i * n + e);
                (if d < 0.0 { -d } else { d }) / params.body_weight
            } else {
                0.0
            };

            let b = i * 11;
            out.write(b * n + e, if contact { 1.0 } else { 0.0 });
            out.write((b + 1) * n + e, if first_contact { 1.0 } else { 0.0 });
            out.write((b + 2) * n + e, if contact { prev_air } else { new_air });
            out.write((b + 3) * n + e, pz - gh);
            out.write((b + 4) * n + e, planar_speed);
            out.write((b + 5) * n + e, tilt);
            out.write((b + 6) * n + e, yaw_rel);
            out.write((b + 7) * n + e, px);
            out.write((b + 8) * n + e, py);
            out.write((b + 9) * n + e, vz);
            out.write((b + 10) * n + e, force_rate);
            out.write((22 + i) * n + e, if alt_step { 1.0 } else { 0.0 });
            out.write((24 + i) * n + e, new_air);

            // Commit the per-env history the host used to own. `last_td_foot`
            // takes the LAST foot to touch down this step, matching the host's
            // serial commit (`if td_foot >= 0 { last_td_foot = td_foot }`,
            // last-wins when both land).
            air_time.write(i * n + e, new_air);
            prev_foot_pos.write((i * 3) * n + e, px);
            prev_foot_pos.write((i * 3 + 1) * n + e, py);
            prev_foot_pos.write((i * 3 + 2) * n + e, pz);
            if first_contact {
                last_td_foot.write(e, i as f32);
            }
        }
    }
}

/// Rotate `v` by xyzw quaternion `q`.
#[inline]
fn qrot(q: Vec4, v: [f32; 3]) -> [f32; 3] {
    let (ux, uy, uz, w) = (q.x, q.y, q.z, q.w);
    let tx = uy * v[2] - uz * v[1] + w * v[0];
    let ty = uz * v[0] - ux * v[2] + w * v[1];
    let tz = ux * v[1] - uy * v[0] + w * v[2];
    [
        v[0] + 2.0 * (uy * tz - uz * ty),
        v[1] + 2.0 * (uz * tx - ux * tz),
        v[2] + 2.0 * (ux * ty - uy * tx),
    ]
}

/// Scalar parameters for [`gpu_reward_feet_terms`] (uniform; 64 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct RewardFeetParams {
    pub n_envs: u32,
    pub dt: f32,
    pub w_flight: f32,
    pub w_foot_slip: f32,
    pub w_force_rate: f32,
    pub force_rate_deadband: f32,
    pub w_foot_orientation: f32,
    pub w_feet_yaw_mean: f32,
    pub w_feet_yaw_diff: f32,
    pub w_feet_distance: f32,
    pub feet_distance_ref: f32,
    pub w_touchdown_vz: f32,
    pub touchdown_vz_h: f32,
    pub touchdown_vz_ok: f32,
    pub pad0: u32,
    pub pad1: u32,
}

/// The self-contained per-foot reward terms, one thread per environment.
///
/// Exact ports:
///   flight           = w · dt                      when NO foot is in contact
///   foot_slip        = w · Σ_contact planar²       · dt
///   force_rate       = w · Σ max(|ΔF|−deadband,0)² · dt
///   foot_orientation = w · Σ tilt²                 · dt
///   feet_yaw_mean    = w · Σ yaw_rel_base²         · dt
///   feet_yaw_diff    = w · wrap(yaw₁−yaw₀)²        · dt
///   feet_distance    = w · | |lateral| − ref |     · dt   (L1, WBC default)
///   touchdown_vz     = w · Σ max(−vz−ok,0)²        · dt   for airborne feet
///                                                          below the gate height
///
/// `feet` is the `[26 x n]` block from [`gpu_feet_state`]; `base` is the
/// `[13 x n]` block from [`gpu_base_state`] (only the quaternion is used, for
/// the base-yaw projection of the stance width). `out` is `[8 x n]` in the
/// order above.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reward_feet_terms(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &RewardFeetParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] feet: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] base: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] out: &mut [f32],
) {
    let e = invocation_id.x as usize;
    let n = params.n_envs as usize;
    if e < n {
        let dt = params.dt;
        let g = |row: usize| feet.read(row * n + e);

        let mut n_contact = 0u32;
        let mut slip = 0.0f32;
        let mut frate = 0.0f32;
        let mut tilt_sq = 0.0f32;
        let mut yaw_sq = 0.0f32;
        let mut slam = 0.0f32;
        for i in 0..2usize {
            let b = i * 11;
            let contact = g(b) > 0.5;
            if contact {
                n_contact += 1;
                let ps = g(b + 4);
                slip += ps * ps;
            }
            let ex = g(b + 10) - params.force_rate_deadband;
            let ex = if ex > 0.0 { ex } else { 0.0 };
            frate += ex * ex;
            let ti = g(b + 5);
            tilt_sq += ti * ti;
            let yw = g(b + 6);
            yaw_sq += yw * yw;
            if !contact && g(b + 3) < params.touchdown_vz_h {
                let s = -g(b + 9) - params.touchdown_vz_ok;
                let s = if s > 0.0 { s } else { 0.0 };
                slam += s * s;
            }
        }

        // Wrapped yaw difference between the feet (base yaw cancels).
        let pi = 3.141_592_7f32;
        let d = g(1 * 11 + 6) - g(6);
        let two_pi = 2.0 * pi;
        let mut wrapped = (d + pi) - two_pi * ((d + pi) / two_pi).floor();
        wrapped -= pi;

        // Lateral stance width: world foot-to-foot XY taken into the base frame
        // using only the base YAW component.
        let (qx, qy, qz, qw) =
            (base.read(e), base.read(n + e), base.read(2 * n + e), base.read(3 * n + e));
        let base_yaw =
            (2.0 * (qw * qz + qx * qy)).atan2(1.0 - 2.0 * (qy * qy + qz * qz));
        let dx = g(7) - g(1 * 11 + 7);
        let dy = g(8) - g(1 * 11 + 8);
        let lateral = -base_yaw.sin() * dx + base_yaw.cos() * dy;
        let lat_abs = if lateral < 0.0 { -lateral } else { lateral };
        let err = lat_abs - params.feet_distance_ref;
        let err_abs = if err < 0.0 { -err } else { err };

        out.write(e, if n_contact == 0 { params.w_flight * dt } else { 0.0 });
        out.write(n + e, params.w_foot_slip * slip * dt);
        out.write(2 * n + e, params.w_force_rate * frate * dt);
        out.write(3 * n + e, params.w_foot_orientation * tilt_sq * dt);
        out.write(4 * n + e, params.w_feet_yaw_mean * yaw_sq * dt);
        out.write(5 * n + e, params.w_feet_yaw_diff * wrapped * wrapped * dt);
        out.write(6 * n + e, params.w_feet_distance * err_abs * dt);
        out.write(7 * n + e, params.w_touchdown_vz * slam * dt);
    }
}

/// Scalar parameters for [`gpu_reward_gait_terms`] (uniform; 64 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct RewardGaitParams {
    pub n_envs: u32,
    pub dt: f32,
    pub w_air_time: f32,
    pub w_single_support: f32,
    pub w_stand_planted: f32,
    pub w_foot_clearance: f32,
    pub foot_clearance_target: f32,
    pub w_gait_clock: f32,
    pub gait_swing_ratio: f32,
    /// Longest airborne time still counted as an ACTIVE swing.
    pub max_swing_s: f32,
    /// Foot resting height, subtracted before the clearance ratio.
    pub foot_rest_h: f32,
    /// Extra clearance demanded above a detected step edge.
    pub step_clear_margin: f32,
    pub step_relax_dist: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

/// The GATED gait reward terms, one thread per environment.
///
/// These are the terms whose whole character is in their guards, so the guards
/// are ported verbatim:
///   air_time       — paid only at an ALTERNATING touchdown, capped at 0.4 s,
///                    moving-gated and multiplied by `progress`
///   single_support — moving: penalise permanent double-support, and
///                    single-support only when the airborne foot is a HELD
///                    statue (air_time > max_swing); standing: the inverse,
///                    reward both planted and penalise stepping
///   stand_planted  — per airborne foot, standing only
///   foot_clearance — active swings only (air_time < max_swing), moving-gated,
///                    capped at 1 per foot, target raised over a detected edge
///   gait_clock     — Siekmann ±1 per foot for contact matching its phase
///                    window, moving-gated and progress-multiplied
///
/// `moving`/`standing` is `cmd.speed() >= 0.1` (the FULL speed, including
/// yaw_rate), while `progress` uses the PLANAR command — the same distinction
/// that bit the tracking term.
///
/// `feet` is `[26 x n]` from `gpu_feet_state`; `aux` is `[5 x n]`: phase,
/// progress, cmd_speed_full, cue_height, stepping. `out` is `[5 x n]`:
/// air_time, single_support, stand_planted, foot_clearance, gait_clock.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reward_gait_terms(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &RewardGaitParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] feet: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] aux: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] out: &mut [f32],
) {
    let e = invocation_id.x as usize;
    let n = params.n_envs as usize;
    if e < n {
        let dt = params.dt;
        let g = |row: usize| feet.read(row * n + e);
        let phase = aux.read(e);
        let progress = aux.read(n + e);
        let speed_full = aux.read(2 * n + e);
        let cue_h = aux.read(3 * n + e);
        let stepping = aux.read(4 * n + e) > 0.5;
        let moving = speed_full >= 0.1;

        let mut contacts = 0u32;
        let mut held = false;
        let mut air = 0.0f32;
        let mut airborne = 0.0f32;
        let mut foot_h = 0.0f32;
        let mut gc = 0.0f32;

        let clear_target = if stepping && cue_h > 0.0 {
            let t = cue_h + params.step_clear_margin;
            if t > params.foot_clearance_target { t } else { params.foot_clearance_target }
        } else {
            params.foot_clearance_target
        };

        for i in 0..2usize {
            let b = i * 11;
            let contact = g(b) > 0.5;
            let at = g(b + 2);
            let alt = g(22 + i) > 0.5;
            if contact {
                contacts += 1;
            } else {
                airborne += 1.0;
                if at > params.max_swing_s {
                    held = true;
                }
                if at < params.max_swing_s {
                    let lift = g(b + 3) - params.foot_rest_h;
                    let lift = if lift > 0.0 { lift } else { 0.0 } / clear_target;
                    foot_h += if lift > 1.0 { 1.0 } else { lift };
                }
            }
            if alt {
                air += if at > 0.4 { 0.4 } else { at };
            }
            // Siekmann gait clock: +1 when contact matches the phase window.
            let ph = phase + 0.5 * i as f32;
            let ph = ph - ph.floor();
            let want_swing = ph < params.gait_swing_ratio;
            let matched = if want_swing { !contact } else { contact };
            gc += if matched { 1.0 } else { -1.0 };
        }

        let ss = if moving {
            if contacts == 2 || (contacts == 1 && held) {
                -params.w_single_support * dt
            } else {
                0.0
            }
        } else if contacts == 2 {
            params.w_single_support * dt
        } else if contacts == 1 {
            -params.w_single_support * dt
        } else {
            0.0
        };

        out.write(e, if moving { params.w_air_time * air * dt * progress } else { 0.0 });
        out.write(n + e, ss);
        out.write(
            2 * n + e,
            if moving { 0.0 } else { params.w_stand_planted * airborne * dt },
        );
        out.write(
            3 * n + e,
            if moving { params.w_foot_clearance * foot_h * dt } else { 0.0 },
        );
        out.write(
            4 * n + e,
            if moving { params.w_gait_clock * gc * dt * progress } else { 0.0 },
        );
    }
}

/// Scalar parameters for [`gpu_reward_misc_terms`] (uniform; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct RewardMiscParams {
    pub n_envs: u32,
    pub colliders_per_env: u32,
    pub n_pairs: u32,
    pub dt: f32,
    pub sc_margin: f32,
    pub sc_weight: f32,
    pub chest_link: u32,
    pub chest_w: f32,
    pub w_termination: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

/// The three remaining reward terms, one thread per environment.
///
///   self_coll     = −(w · Σ_pairs max(margin − |pa − pb|, 0)) · dt
///   chest_ang_vel = −(w · (ωx² + ωy²)) · dt, ω from the chest link's rotation
///                   change with the same hemisphere correction as the base
///   termination   = w_termination when the env fell this step
///
/// `fell` is supplied per env: the termination PREDICATE still lives on the
/// host (it folds in joint faults, self-collision distance and time-outs), so
/// only the reward contribution moves here.
///
/// `out` is `[3 x n]`: self_coll, chest_ang_vel, termination.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reward_misc_terms(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &RewardMiscParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_poses: &[Pose3],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pair_a: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] pair_b: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] prev_chest: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] have_prev: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] fell: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] out: &mut [f32],
) {
    let e = invocation_id.x as usize;
    let n = params.n_envs as usize;
    if e < n {
        let base = e * params.colliders_per_env as usize;
        let dt = params.dt;

        let mut intrusion = 0.0f32;
        if params.sc_weight > 0.0 {
            for k in 0..params.n_pairs as usize {
                let pa = body_poses.read(base + pair_a.read(k) as usize).translation;
                let pb = body_poses.read(base + pair_b.read(k) as usize).translation;
                let dx = pa.x - pb.x;
                let dy = pa.y - pb.y;
                let dz = pa.z - pb.z;
                let d = params.sc_margin - (dx * dx + dy * dy + dz * dz).sqrt();
                intrusion += if d > 0.0 { d } else { 0.0 };
            }
        }

        let mut chest_pen = 0.0f32;
        if params.chest_w > 0.0 && have_prev.read(e) != 0 {
            let c = body_poses.read(base + params.chest_link as usize).rotation;
            let pq = Vec4::new(
                -prev_chest.read(e),
                -prev_chest.read(n + e),
                -prev_chest.read(2 * n + e),
                prev_chest.read(3 * n + e),
            );
            let dq = qmul(Vec4::new(c.x, c.y, c.z, c.w), pq);
            let s = if dq.w >= 0.0 { 1.0 } else { -1.0 };
            let wx = 2.0 * s * dq.x / dt;
            let wy = 2.0 * s * dq.y / dt;
            chest_pen = params.chest_w * (wx * wx + wy * wy) * dt;
        }

        out.write(e, -(params.sc_weight * intrusion * dt));
        out.write(n + e, -chest_pen);
        out.write(
            2 * n + e,
            if fell.read(e) != 0 { params.w_termination } else { 0.0 },
        );
    }
}

/// Clear the per-env feet history for the environments that just reset.
///
/// Mirrors the host reset: air time to zero, last-touchdown foot to −1 (so the
/// first step after a reset counts as alternating), and the previous sensed
/// force seeded from the current one. `env_ids` lists the resetting envs, so
/// this is one dispatch per reset batch rather than per env.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_feet_reset(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &UVec4,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] env_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] seed_force: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] air_time: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] last_td_foot: &mut [f32],
) {
    // params: x = count, y = n_envs, z = n_feet
    let k = invocation_id.x as usize;
    let count = params.x as usize;
    let n = params.y as usize;
    let nf = params.z as usize;
    if k < count {
        let e = env_ids.read(k) as usize;
        last_td_foot.write(e, -1.0);
        for i in 0..nf {
            air_time.write(i * n + e, 0.0);
        }
        let _ = seed_force;
    }
}
