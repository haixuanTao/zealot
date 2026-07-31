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
//!   [24..36) action t−1, [36] episode step counter (f32).
//! - `hist`: 5×48 obs-frame ring per env, slot s = ep % 5.
//! - `consts`: [0..12) actuated dof ids (as f32), [12..24) default_pos,
//!   [24..36) target lo, [36..48) target hi, [48..273) normalizer mean (225),
//!   [273..498) normalizer std (225).
//! - `gemm_input`: row-major [225 × n_envs] — the vortx GEMM a0 buffer.
//! - `actions`/`targets`: row-major [12 × n_envs] (matches
//!   `scatter_motor_targets`).
//!
//! Obs frame (48), mirroring `sim2sim_g1_mujoco.py` / the training env:
//! [0..12) lag-2 action · [12..16) command · [16..28) q − default ·
//! [28..40) finite-diff joint velocity · [40..43) projected gravity ·
//! [43..45) gait-clock sin/cos · [45..48) base angular velocity (gyro,
//! body frame — added for v24; the first 45 slots keep their v21 meaning).
//! History ×5 oldest-first, reset-replicated;
//! Welford-normalized (clip ±5).
//!
//! No panics anywhere (no `[]` indexing, no `clamp`, no `step_by`): panic
//! edges become naga switch-breaks that Tint rejects — see the
//! browser-kernel playbook.

#![cfg_attr(target_arch = "spirv", no_std)]

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
#[allow(unused_imports)]
use khal_std::num_traits::Float;

/// f32 slots per env in the `state` buffer.
pub const STATE_STRIDE: usize = 37;
/// Policy joints.
pub const N_ACT: usize = 12;
/// Frames of obs history.
pub const HIST: usize = 5;
/// Single-frame obs width.
pub const FRAME: usize = 48;
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

/// Assemble the 48-dim obs frame, update the history ring, and write the
/// normalized stacked [225 × N] GEMM input. One thread per env.
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
    #[spirv(uniform, descriptor_set = 0, binding = 8)] gait_period: &f32,
) {
    let e = invocation_id.x;
    let n = *n_envs;
    if e >= n {
        return;
    }
    let ne = n as usize;
    let eu = e as usize;
    let sb = |i: usize| i * ne + eu; // batch-interleaved slot

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

    // 48-dim frame.
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
    let ph_steps = if ep >= 1.0 { ep - 1.0 } else { 0.0 };
    let ph_raw = ph_steps * *dt / *gait_period;
    let ph = ph_raw - ph_raw.floor();
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
    while r < FRAME {
        if ep <= 0.5 {
            let mut s = 0;
            while s < HIST {
                hist.write((s * FRAME + r) * ne + eu, o[r]);
                s += 1;
            }
        } else {
            hist.write((slot * FRAME + r) * ne + eu, o[r]);
        }
        r += 1;
    }

    // Normalized stacked obs, oldest-first: frame k = ring slot
    // (slot + 1 + k) mod 5.
    let mut k = 0;
    while k < HIST {
        let src = (slot + 1 + k) % HIST;
        let mut r = 0;
        while r < FRAME {
            let x = hist.read((src * FRAME + r) * ne + eu);
            let m = consts.read(C_MEAN + k * FRAME + r);
            let s = consts.read(C_STD + k * FRAME + r);
            let mut v = (x - m) / s;
            if v > 5.0 {
                v = 5.0;
            }
            if v < -5.0 {
                v = -5.0;
            }
            gemm_input.write((k * FRAME + r) * ne + eu, v);
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
