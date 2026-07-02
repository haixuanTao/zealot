//! GPU reward kernel (`gpu_reward`) verified against a line-by-line CPU
//! transcription of `VelocityFlatTask::reward` + `compute_feet_from_poses` +
//! `fell_over`.
//!
//! Run (WebGPU):     `cargo run --release --example reward_check --features "gpu biped_gpu"`
//! Run (native CUDA): `BIPED_CUDA=1 cargo run --release --example reward_check --features "gpu biped_gpu cuda_backend"`
//!
//! All 20 reward weights are randomised NON-ZERO (the deployed config zeroes
//! several), so every term + gating branch (standing/moving, contact/flight,
//! single-support, first-contact air-time, joint limits, symmetry) is exercised.
//! Synthetic poses put some feet below the contact threshold and some torsos
//! below the height floor / past the tilt limit so the fall-termination path
//! fires too.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use nexus3d::rbd::math::{Quat, Vec3};
use vortx::linalg::{Reward, RewardJointCfg, RewardParams};
use vortx::tensor::Tensor;

const N: usize = 4096;
const J: usize = 12;
const CPB: usize = 16;
const TORSO: usize = 0;
const FEET: [usize; 2] = [14, 15];
const NF: usize = 2;
const DT: f32 = 1.0 / 50.0;
const STRIDE: usize = 8;
const CONTACT_Z: f32 = 0.025;
const AIR_CAP: f32 = 0.4;
const STANDING_SPEED: f32 = 0.1;
const LIMIT_SCALE: f32 = 0.9;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }
    fn quat(&mut self) -> Quat {
        Quat::from_xyzw(
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
        )
        .normalize()
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn qrot(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let w = q[3];
    let t = cross(u, v);
    let t = [t[0] + w * v[0], t[1] + w * v[1], t[2] + w * v[2]];
    let tt = cross(u, t);
    [v[0] + 2.0 * tt[0], v[1] + 2.0 * tt[1], v[2] + 2.0 * tt[2]]
}
fn qrot_inv(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    qrot([-q[0], -q[1], -q[2], q[3]], v)
}
fn rot(buf: &[f32], e: usize, link: usize) -> Quat {
    let b = (e * CPB + link) * STRIDE;
    Quat::from_xyzw(buf[b], buf[b + 1], buf[b + 2], buf[b + 3])
}
fn tr(buf: &[f32], e: usize, link: usize) -> [f32; 3] {
    let b = (e * CPB + link) * STRIDE;
    [buf[b + 4], buf[b + 5], buf[b + 6]]
}
fn angle(buf: &[f32], e: usize, cfg: &RewardJointCfg) -> f32 {
    let qp = rot(buf, e, cfg.parent_link as usize);
    let qc = rot(buf, e, cfg.child_link as usize);
    let rest = Quat::from_xyzw(cfg.rest_quat[0], cfg.rest_quat[1], cfg.rest_quat[2], cfg.rest_quat[3]);
    let rel = rest.conjugate() * qp.conjugate() * qc;
    2.0 * rel.z.atan2(rel.w)
}

async fn make_backend() -> GpuBackend {
    #[cfg(feature = "cuda_backend")]
    {
        if std::env::var("BIPED_CUDA").as_deref() == Ok("1") {
            use khal::backend::Cuda;
            eprintln!("backend = native CUDA");
            return GpuBackend::Cuda(Cuda::new(0).expect("cuda"));
        }
    }
    eprintln!("backend = WebGPU");
    let limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 14,
        ..Default::default()
    };
    GpuBackend::WebGpu(WebGpu::new(wgpu::Features::default(), limits).await.expect("webgpu"))
}

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let mut rng = Lcg(0x5EED);
    let tilt_cos = 70_f32.to_radians().cos();
    let base_height_target = 0.62;
    let foot_clearance_target = 0.08;
    let feet_distance_ref = 0.2;
    let min_base_height = 0.4;
    let (std_lin, std_ang, std_up, std_bh, std_pose) = (0.3f32, 0.2f32, 0.17f32, 0.1f32, 1.0f32);

    // All weights non-zero so every term is exercised.
    let w_track_lin = rng.range(1.0, 5.0);
    let w_track_ang = rng.range(1.0, 5.0);
    let w_upright = rng.range(1.0, 5.0);
    let w_base_height = rng.range(1.0, 3.0);
    let w_pose = rng.range(0.5, 2.0);
    let w_bilateral = rng.range(0.5, 2.0);
    let w_action_rate = -rng.range(0.1, 0.3);
    let w_action_rate_hip = -rng.range(0.1, 0.3);
    let w_body_ang_vel = -rng.range(0.1, 0.3);
    let w_lin_vel_z = -rng.range(0.1, 0.3);
    let w_dof_pos_limits = -rng.range(0.2, 0.6);
    let w_dof_vel = -rng.range(1e-4, 5e-4);
    let w_termination = -rng.range(10.0, 30.0);
    let w_air_time = rng.range(0.5, 2.0);
    let w_flight = -rng.range(10.0, 25.0);
    let w_single_support = rng.range(0.5, 2.0);
    let w_foot_slip = -rng.range(0.02, 0.08);
    let w_foot_clearance = -rng.range(0.5, 2.0);
    let w_foot_orientation = -rng.range(0.2, 0.6);
    let w_feet_yaw_mean = -rng.range(1.0, 3.0);
    let w_feet_distance = -rng.range(0.05, 0.2);

    // --- per-joint config: pair joints (0,1)(2,3)… for symmetry; mark a few hips ---
    let mut cfg = Vec::with_capacity(J);
    for k in 0..J {
        let parent = (rng.range(1.0, CPB as f32) as usize).min(CPB - 1) as u32;
        let mut child = (rng.range(1.0, CPB as f32) as usize).min(CPB - 1) as u32;
        if child == parent {
            child = 1 + (child % (CPB as u32 - 1));
        }
        let q = rng.quat();
        let lo = rng.range(-1.5, -0.5);
        let hi = rng.range(0.5, 1.5);
        // left joints = even k, partner = k+1, sign alternates
        let is_left = k % 2 == 0;
        let sym_active = if is_left { 1 } else { 0 };
        let sym_partner = if is_left { (k + 1) as u32 } else { k as u32 };
        let sym_sign = if (k / 2) % 2 == 0 { 1.0 } else { -1.0 };
        let is_hip = if k < 4 { 1 } else { 0 };
        cfg.push(RewardJointCfg {
            parent_link: parent,
            child_link: child,
            default_pos: rng.range(-0.4, 0.4),
            pos_lo: lo,
            pos_hi: hi,
            sym_partner,
            sym_sign,
            sym_active,
            is_hip,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            rest_quat: [q.x, q.y, q.z, q.w],
        });
    }

    // --- poses / state ---
    let mut poses = vec![0f32; N * CPB * STRIDE];
    let mut prev = vec![0f32; N * CPB * STRIDE];
    for (which, buf) in [&mut poses, &mut prev].into_iter().enumerate() {
        for e in 0..N {
            for l in 0..CPB {
                let q = rng.quat();
                let b = (e * CPB + l) * STRIDE;
                buf[b] = q.x;
                buf[b + 1] = q.y;
                buf[b + 2] = q.z;
                buf[b + 3] = q.w;
                // torso height spans the fall floor; foot z spans the contact threshold.
                let z = if l == TORSO {
                    rng.range(0.30, 0.72)
                } else if l == FEET[0] || l == FEET[1] {
                    rng.range(-0.01, 0.15)
                } else {
                    rng.range(0.0, 0.6)
                };
                buf[b + 4] = rng.range(-0.3, 0.3);
                buf[b + 5] = rng.range(-0.3, 0.3);
                buf[b + 6] = z;
                let _ = which;
            }
        }
    }
    let mut cmd = vec![0f32; 3 * N];
    for e in 0..N {
        if e % 3 == 0 {
            // standing command (all zero)
        } else {
            cmd[e] = rng.range(-0.5, 0.5);
            cmd[N + e] = rng.range(-0.3, 0.3);
            cmd[2 * N + e] = rng.range(-0.2, 0.2);
        }
    }
    let action2: Vec<f32> = (0..2 * J * N).map(|_| rng.range(-1.0, 1.0)).collect();
    let air_in: Vec<f32> = (0..NF * N).map(|_| rng.range(0.0, 0.6)).collect();
    // per-env foot-local sole normals near +Z
    let mut sole = vec![0f32; NF * 3 * N];
    for e in 0..N {
        for i in 0..NF {
            let v = Vec3::new(rng.range(-0.2, 0.2), rng.range(-0.2, 0.2), rng.range(0.8, 1.0))
                .normalize();
            sole[(i * 3) * N + e] = v.x;
            sole[(i * 3 + 1) * N + e] = v.y;
            sole[(i * 3 + 2) * N + e] = v.z;
        }
    }
    let flags: Vec<u32> = (0..N).map(|e| (e % 4) as u32).collect();

    // ===================== CPU reference =====================
    let mut reward_c = vec![0f32; N];
    let mut fell_c = vec![0u32; N];
    let mut new_air_c = vec![0f32; NF * N];
    for e in 0..N {
        let fl = flags[e];
        let has_prev_pose = fl & 1 != 0;
        let has_prev_jp = fl & 2 != 0;
        let r = rot(&poses, e, TORSO);
        let rq = [r.x, r.y, r.z, r.w];
        let t = tr(&poses, e, TORSO);
        let (mut lin_w, mut ang_w) = ([0f32; 3], [0f32; 3]);
        if has_prev_pose {
            let pr = rot(&prev, e, TORSO);
            let pt = tr(&prev, e, TORSO);
            lin_w = [(t[0] - pt[0]) / DT, (t[1] - pt[1]) / DT, (t[2] - pt[2]) / DT];
            let dq = r * pr.conjugate();
            let s = if dq.w >= 0.0 { 1.0 } else { -1.0 };
            ang_w = [2.0 * s * dq.x / DT, 2.0 * s * dq.y / DT, 2.0 * s * dq.z / DT];
        }
        let v = qrot_inv(rq, lin_w);
        let w = qrot_inv(rq, ang_w);
        let grav = qrot_inv(rq, [0.0, 0.0, -1.0]);
        let height = t[2];
        let (cvx, cvy, cyaw) = (cmd[e], cmd[N + e], cmd[2 * N + e]);
        let speed = (cvx * cvx + cvy * cvy + cyaw * cyaw).sqrt();
        let standing = speed < STANDING_SPEED;
        let moving = !standing;

        let lin_err = (cvx - v[0]).powi(2) + (cvy - v[1]).powi(2);
        let track_lin = w_track_lin * (-lin_err / std_lin.powi(2) as f32).exp() * DT;
        let ang_err = (cyaw - w[2]).powi(2);
        let track_ang = w_track_ang * (-ang_err / std_ang.powi(2) as f32).exp() * DT;
        let tilt_err = grav[0].powi(2) + grav[1].powi(2);
        let upright = w_upright * (-tilt_err / (std_up as f32).powi(2)).exp() * DT;
        let h_err = (height - base_height_target).powi(2);
        let base_h = w_base_height * (-h_err / (std_bh as f32).powi(2)).exp() * DT;

        let mut pose_err = 0.0;
        let mut sym_err = 0.0;
        let mut lim_pen = 0.0;
        let mut jv2 = 0.0;
        let mut da2 = 0.0;
        let mut da2_hip = 0.0;
        for k in 0..J {
            let c = &cfg[k];
            let q = angle(&poses, e, c);
            pose_err += (q - c.default_pos).powi(2);
            if c.sym_active != 0 {
                let qr = angle(&poses, e, &cfg[c.sym_partner as usize]);
                sym_err += (q - c.sym_sign * qr).powi(2);
            }
            let hi = c.pos_hi * LIMIT_SCALE;
            let lo = c.pos_lo * LIMIT_SCALE;
            lim_pen += (q - hi).max(0.0) + (lo - q).max(0.0);
            if has_prev_jp {
                let qprev = angle(&prev, e, c);
                jv2 += ((q - qprev) / DT).powi(2);
            }
            let la = action2[k * N + e];
            let pa = action2[(J + k) * N + e];
            da2 += (la - pa).powi(2);
            if c.is_hip != 0 {
                da2_hip += (la - pa).powi(2);
            }
        }
        let pose = if standing {
            w_pose * (-pose_err / (std_pose as f32).powi(2)).exp() * DT
        } else {
            0.0
        };
        let bilateral = w_bilateral * (-sym_err).exp() * DT;
        let action_rate = w_action_rate * da2 * DT;
        let action_rate_hip = w_action_rate_hip * da2_hip * DT;
        let body_ang_vel = w_body_ang_vel * (w[0].powi(2) + w[1].powi(2)) * DT;
        let lin_vel_z = w_lin_vel_z * v[2].powi(2) * DT;
        let dof_pos_limits = w_dof_pos_limits * lim_pen * DT;
        let dof_vel = w_dof_vel * jv2 * DT;

        // feet
        let base_rot_inv = r.conjugate();
        let mut air_sum = 0.0;
        let mut all_air = true;
        let mut contacts = 0;
        let mut slip = 0.0;
        let mut clr = 0.0;
        let mut tilt_sq = 0.0;
        let mut yaw_sq = 0.0;
        let mut fx = [0.0f32; NF];
        let mut fy = [0.0f32; NF];
        for i in 0..NF {
            let link = FEET[i];
            let fpos = tr(&poses, e, link);
            let frot = rot(&poses, e, link);
            let planar_speed = if has_prev_pose {
                let pp = tr(&prev, e, link);
                (((fpos[0] - pp[0]) / DT).powi(2) + ((fpos[1] - pp[1]) / DT).powi(2)).sqrt()
            } else {
                0.0
            };
            let sole_v = Vec3::new(
                sole[(i * 3) * N + e],
                sole[(i * 3 + 1) * N + e],
                sole[(i * 3 + 2) * N + e],
            );
            let world_normal = frot * sole_v;
            let tilt = world_normal.z.abs().clamp(0.0, 1.0).acos();
            let fx_base = (base_rot_inv * frot) * Vec3::X;
            let yaw_rel = fx_base.y.atan2(fx_base.x);
            let contact = fpos[2] < CONTACT_Z;
            let prev_air = air_in[i * N + e];
            let first_contact = contact && prev_air > 0.0;
            let na = if contact { 0.0 } else { prev_air + DT };
            new_air_c[i * N + e] = na;
            let air_time = if contact { prev_air } else { na };
            if contact {
                contacts += 1;
                slip += planar_speed.powi(2);
                tilt_sq += tilt.powi(2);
                all_air = false;
            } else {
                clr += (fpos[2] - foot_clearance_target).powi(2) * planar_speed;
            }
            if first_contact {
                air_sum += air_time.min(AIR_CAP);
            }
            yaw_sq += yaw_rel.powi(2);
            fx[i] = fpos[0];
            fy[i] = fpos[1];
        }
        let air_time = if moving { w_air_time * air_sum * DT } else { 0.0 };
        let flight = if all_air { w_flight * DT } else { 0.0 };
        let single_support = if moving && contacts == 1 { w_single_support * DT } else { 0.0 };
        let foot_slip = w_foot_slip * slip * DT;
        let foot_clearance = w_foot_clearance * clr * DT;
        let foot_orientation = w_foot_orientation * tilt_sq * DT;
        let feet_yaw_mean = w_feet_yaw_mean * yaw_sq * DT;
        let dx = fx[0] - fx[1];
        let dy = fy[0] - fy[1];
        let base_yaw = (2.0 * (rq[3] * rq[2] + rq[0] * rq[1]))
            .atan2(1.0 - 2.0 * (rq[1] * rq[1] + rq[2] * rq[2]));
        let lateral = -base_yaw.sin() * dx + base_yaw.cos() * dy;
        let feet_distance = w_feet_distance * (lateral.abs() - feet_distance_ref).abs() * DT;

        let mut total = track_lin
            + track_ang
            + upright
            + base_h
            + pose
            + bilateral
            + action_rate
            + action_rate_hip
            + body_ang_vel
            + lin_vel_z
            + dof_pos_limits
            + dof_vel
            + air_time
            + flight
            + single_support
            + foot_slip
            + foot_clearance
            + foot_orientation
            + feet_yaw_mean
            + feet_distance;
        let upright_cos = qrot(rq, [0.0, 0.0, 1.0])[2];
        let fell = !height.is_finite() || height < min_base_height || upright_cos < tilt_cos;
        if fell {
            total += w_termination;
            fell_c[e] = 1;
        }
        reward_c[e] = total;
    }

    // ===================== GPU =====================
    let backend = make_backend().await;
    let op = Reward::from_backend(&backend)?;
    let st = BufferUsages::STORAGE;
    let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let params = Tensor::scalar(
        &backend,
        RewardParams {
            num_envs: N as u32,
            num_joints: J as u32,
            num_feet: NF as u32,
            colliders_per_batch: CPB as u32,
            torso_link: TORSO as u32,
            fwd: 0,
            lat: 1,
            up: 2,
            foot_link0: FEET[0] as u32,
            foot_link1: FEET[1] as u32,
            pad_u0: 0,
            pad_u1: 0,
            control_dt: DT,
            w_track_lin,
            w_track_ang,
            w_upright,
            w_base_height,
            base_height_target,
            w_pose,
            w_bilateral,
            w_action_rate,
            w_action_rate_hip,
            w_body_ang_vel,
            w_lin_vel_z,
            w_dof_pos_limits,
            w_dof_vel,
            w_termination,
            w_air_time,
            w_flight,
            w_single_support,
            w_foot_slip,
            w_foot_clearance,
            foot_clearance_target,
            w_foot_orientation,
            w_feet_yaw_mean,
            w_feet_distance,
            feet_distance_ref,
            std_lin_vel: std_lin,
            std_ang_vel: std_ang,
            std_upright: std_up,
            std_base_height: std_bh,
            std_pose,
            contact_z: CONTACT_Z,
            min_base_height,
            tilt_cos,
            standing_speed: STANDING_SPEED,
            air_cap: AIR_CAP,
            limit_scale: LIMIT_SCALE,
            pad_f0: 0.0,
            pad_f1: 0.0,
        },
        BufferUsages::UNIFORM,
    )?;
    let poses_t = Tensor::vector(&backend, &poses, st)?;
    let prev_t = Tensor::vector(&backend, &prev, st)?;
    let cfg_t = Tensor::vector(&backend, &cfg, st)?;
    let cmd_t = Tensor::vector(&backend, &cmd, st)?;
    let a2_t = Tensor::vector(&backend, &action2, st)?;
    let air_t = Tensor::vector(&backend, &air_in, st)?;
    let sole_t = Tensor::vector(&backend, &sole, st)?;
    let flags_t = Tensor::vector(&backend, &flags, st)?;
    let mut reward_t = Tensor::vector(&backend, &vec![0f32; N], rw)?;
    let mut fell_t = Tensor::vector(&backend, &vec![0u32; N], rw)?;
    let mut newair_t = Tensor::vector(&backend, &vec![0f32; NF * N], rw)?;

    let mut enc = backend.begin_encoding();
    {
        let mut p = enc.begin_pass("reward", None);
        op.evaluate(
            &mut p, &params, &poses_t, &prev_t, &cfg_t, &cmd_t, &a2_t, &air_t, &sole_t, &flags_t,
            &mut reward_t, &mut fell_t, &mut newair_t,
        )?;
    }
    backend.submit(enc)?;
    backend.synchronize()?;

    let reward_g = backend.slow_read_vec(reward_t.buffer()).await?;
    let fell_g: Vec<u32> = backend.slow_read_vec(fell_t.buffer()).await?;
    let newair_g = backend.slow_read_vec(newair_t.buffer()).await?;

    let mut e_rew = 0f32;
    let mut e_rel = 0f32;
    for e in 0..N {
        let d = (reward_g[e] - reward_c[e]).abs();
        e_rew = e_rew.max(d);
        e_rel = e_rel.max(d / (reward_c[e].abs().max(1.0)));
    }
    let fell_mismatch = (0..N).filter(|&e| fell_g[e] != fell_c[e]).count();
    let e_air = newair_g
        .iter()
        .zip(&new_air_c)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let n_fell: usize = fell_c.iter().map(|&x| x as usize).sum();
    let n_stand = (0..N).filter(|&e| e % 3 == 0).count();
    println!("reward check (N={N}, J={J}, feet={NF}; {n_fell} fell, {n_stand} standing)");
    println!("  reward   max|gpu-cpu|     = {e_rew:.3e}");
    println!("  reward   max rel          = {e_rel:.3e}");
    println!("  new_air  max|gpu-cpu|     = {e_air:.3e}");
    println!("  fell     mismatches       = {fell_mismatch}");
    anyhow::ensure!(fell_mismatch == 0, "fall-termination flag mismatch on {fell_mismatch} envs");
    anyhow::ensure!(e_air < 1e-6, "new_air diverged ({e_air:.3e})");
    anyhow::ensure!(e_rel < 1e-4, "reward diverged from CPU (rel {e_rel:.3e})");
    println!("OK — gpu_reward matches the CPU reward/feet/fell reference.");
    Ok(())
}
