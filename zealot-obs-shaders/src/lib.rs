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

use khal_std::glamx::{Pose3, UVec3, Vec4};
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
