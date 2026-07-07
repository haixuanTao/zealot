//! GPU observation kernel (`gpu_obs`) verified against an independent CPU
//! reference of `VelocityFlatTask::observe` / `observe_critic` +
//! `read_state_from_poses`.
//!
//! Run (WebGPU):     `cargo run --release --example obs_check --features "gpu biped_gpu"`
//! Run (native CUDA): `BIPED_CUDA=1 cargo run --release --example obs_check --features "gpu biped_gpu cuda_backend"`
//!
//! Synthetic poses + per-env state are fed to both paths; the GPU kernel must
//! reproduce the policy obs `[J·2 last/jpr ... ]` and the privileged critic obs
//! (incl. finite-diff base lin/ang velocity) bit-closely (quaternion / atan2 ULP
//! aside). The CPU reference uses glamx `Quat` for the joint-angle and ω math
//! (the same ops the real env uses) — an independent check of the kernel's
//! hand-rolled quaternion product.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend};
use nexus3d::rbd::math::Quat;
use vortx::linalg::{JointObsCfg, Obs, ObsParams};
use vortx::tensor::Tensor;

const N: usize = 2048; // envs
const J: usize = 12; // joints
const CPB: usize = 16; // colliders/poses per env
const TORSO: usize = 0;
const OBS_DIM: usize = 4 * J - 5; // 12+4+12+12+3 = 43 for J=12
const CRITIC_OBS_DIM: usize = OBS_DIM + 6;
const DT: f32 = 1.0 / 50.0;
const POSE_STRIDE: usize = 8;

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

// zealot-env math.rs, transcribed (array quaternion rotate).
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let w = q[3];
    let t = cross(u, v);
    let t = [t[0] + w * v[0], t[1] + w * v[1], t[2] + w * v[2]];
    let tt = cross(u, t);
    [v[0] + 2.0 * tt[0], v[1] + 2.0 * tt[1], v[2] + 2.0 * tt[2]]
}
fn quat_rotate_inv(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    quat_rotate([-q[0], -q[1], -q[2], q[3]], v)
}

fn rot(buf: &[f32], e: usize, link: usize) -> Quat {
    let b = (e * CPB + link) * POSE_STRIDE;
    Quat::from_xyzw(buf[b], buf[b + 1], buf[b + 2], buf[b + 3])
}
fn tr(buf: &[f32], e: usize, link: usize) -> [f32; 3] {
    let b = (e * CPB + link) * POSE_STRIDE;
    [buf[b + 4], buf[b + 5], buf[b + 6]]
}

async fn make_backend() -> GpuBackend {
    let limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 14,
        ..Default::default()
    };
    GpuBackend::auto(wgpu::Features::default(), limits)
        .await
        .expect("init GPU backend")
}

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let mut rng = Lcg(0xABCDEF);

    // --- static per-joint config ---
    let mut cfg = Vec::with_capacity(J);
    for _ in 0..J {
        let parent = (rng.range(1.0, CPB as f32) as usize).min(CPB - 1) as u32;
        let mut child = (rng.range(1.0, CPB as f32) as usize).min(CPB - 1) as u32;
        if child == parent {
            child = (child + 1) % CPB as u32;
            if child == 0 {
                child = 1;
            }
        }
        let q = rng.quat();
        cfg.push(JointObsCfg {
            parent_link: parent,
            child_link: child,
            default_pos: rng.range(-0.5, 0.5),
            pad: 0.0,
            rest_quat: [q.x, q.y, q.z, q.w],
        });
    }

    // --- synthetic poses / prev poses / per-env state ---
    let mut poses = vec![0f32; N * CPB * POSE_STRIDE];
    let mut prev = vec![0f32; N * CPB * POSE_STRIDE];
    for buf in [&mut poses, &mut prev] {
        for e in 0..N {
            for l in 0..CPB {
                let q = rng.quat();
                let b = (e * CPB + l) * POSE_STRIDE;
                buf[b] = q.x;
                buf[b + 1] = q.y;
                buf[b + 2] = q.z;
                buf[b + 3] = q.w;
                buf[b + 4] = rng.range(-1.0, 1.0);
                buf[b + 5] = rng.range(-1.0, 1.0);
                buf[b + 6] = rng.range(0.2, 1.0);
            }
        }
    }
    let cmd: Vec<f32> = (0..3 * N).map(|_| rng.range(-0.5, 0.5)).collect(); // [3 x n]
    let last_action: Vec<f32> = (0..J * N).map(|_| rng.range(-1.0, 1.0)).collect();
    let prev_jp: Vec<f32> = (0..J * N).map(|_| rng.range(-1.0, 1.0)).collect();
    // Mix of flag states across envs (0=cold, 1=pose only, 2=jp only, 3=both).
    let flags: Vec<u32> = (0..N).map(|e| (e % 4) as u32).collect();

    // ---- CPU reference ----
    let mut obs_c = vec![0f32; OBS_DIM * N];
    let mut critic_c = vec![0f32; CRITIC_OBS_DIM * N];
    let mut jp_c = vec![0f32; J * N];
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
            lin_w = [
                (t[0] - pt[0]) / DT,
                (t[1] - pt[1]) / DT,
                (t[2] - pt[2]) / DT,
            ];
            let dq = r * pr.conjugate();
            let s = if dq.w >= 0.0 { 1.0 } else { -1.0 };
            ang_w = [
                2.0 * s * dq.x / DT,
                2.0 * s * dq.y / DT,
                2.0 * s * dq.z / DT,
            ];
        }
        let grav = quat_rotate_inv(rq, [0.0, 0.0, -1.0]);

        for k in 0..J {
            obs_c[k * N + e] = last_action[k * N + e];
        }
        obs_c[J * N + e] = cmd[e];
        obs_c[(J + 1) * N + e] = cmd[N + e];
        obs_c[(J + 2) * N + e] = cmd[2 * N + e];
        obs_c[(J + 3) * N + e] = 0.0;
        for k in 0..J {
            let qp = rot(&poses, e, cfg[k].parent_link as usize);
            let qc = rot(&poses, e, cfg[k].child_link as usize);
            let rest = Quat::from_xyzw(
                cfg[k].rest_quat[0],
                cfg[k].rest_quat[1],
                cfg[k].rest_quat[2],
                cfg[k].rest_quat[3],
            );
            let rel = rest.conjugate() * qp.conjugate() * qc;
            let theta = 2.0 * rel.z.atan2(rel.w);
            jp_c[k * N + e] = theta;
            obs_c[(J + 4 + k) * N + e] = theta - cfg[k].default_pos;
            let jv = if has_prev_jp {
                (theta - prev_jp[k * N + e]) / DT
            } else {
                0.0
            };
            obs_c[(J + 4 + J + k) * N + e] = jv;
        }
        let bg = J + 4 + J + J;
        obs_c[bg * N + e] = grav[0];
        obs_c[(bg + 1) * N + e] = grav[1];
        obs_c[(bg + 2) * N + e] = grav[2];

        for d in 0..OBS_DIM {
            critic_c[d * N + e] = obs_c[d * N + e];
        }
        let vb = quat_rotate_inv(rq, lin_w);
        let wb = quat_rotate_inv(rq, ang_w);
        for i in 0..3 {
            critic_c[(OBS_DIM + i) * N + e] = vb[i];
            critic_c[(OBS_DIM + 3 + i) * N + e] = wb[i];
        }
    }

    // ---- GPU ----
    let backend = make_backend().await;
    let obs_op = Obs::from_backend(&backend)?;
    let st = BufferUsages::STORAGE;
    let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let params = Tensor::scalar(
        &backend,
        ObsParams {
            num_envs: N as u32,
            num_joints: J as u32,
            colliders_per_batch: CPB as u32,
            torso_link: TORSO as u32,
            obs_dim: OBS_DIM as u32,
            critic_obs_dim: CRITIC_OBS_DIM as u32,
            fwd: 0,
            lat: 1,
            up: 2,
            control_dt: DT,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
            pad4: 0,
            pad5: 0,
        },
        BufferUsages::UNIFORM,
    )?;
    let poses_t = Tensor::vector(&backend, &poses, st)?;
    let prev_t = Tensor::vector(&backend, &prev, st)?;
    let cfg_t = Tensor::vector(&backend, &cfg, st)?;
    let cmd_t = Tensor::vector(&backend, &cmd, st)?;
    let la_t = Tensor::vector(&backend, &last_action, st)?;
    let pjp_t = Tensor::vector(&backend, &prev_jp, st)?;
    let flags_t = Tensor::vector(&backend, &flags, st)?;
    let mut obs_t = Tensor::vector(&backend, &vec![0f32; OBS_DIM * N], rw)?;
    let mut critic_t = Tensor::vector(&backend, &vec![0f32; CRITIC_OBS_DIM * N], rw)?;
    let mut jp_t = Tensor::vector(&backend, &vec![0f32; J * N], rw)?;

    let mut enc = backend.begin_encoding();
    {
        let mut p = enc.begin_pass("obs", None);
        obs_op.assemble(
            &mut p,
            &params,
            &poses_t,
            &prev_t,
            &cfg_t,
            &cmd_t,
            &la_t,
            &pjp_t,
            &flags_t,
            &mut obs_t,
            &mut critic_t,
            &mut jp_t,
        )?;
    }
    backend.submit(enc)?;
    backend.synchronize()?;

    let obs_g = backend.slow_read_vec(obs_t.buffer()).await?;
    let critic_g = backend.slow_read_vec(critic_t.buffer()).await?;
    let jp_g = backend.slow_read_vec(jp_t.buffer()).await?;

    // Velocity rows are finite-diffs (≈ value/dt), so they amplify the GPU
    // atan2/quat ULP by 1/dt (=50 here); hold them to a looser bound and the
    // position-like rows tight. Velocity rows: obs joint_vel block
    // [J+4+J .. J+4+2J), critic base-vel rows [OBS_DIM .. OBS_DIM+6).
    let jvel_lo = J + 4 + J;
    let jvel_hi = J + 4 + 2 * J;
    let split = |g: &[f32], c: &[f32], dim: usize, vel_row: &dyn Fn(usize) -> bool| {
        let (mut e_static, mut e_vel) = (0f32, 0f32);
        for row in 0..dim {
            for e in 0..N {
                let d = (g[row * N + e] - c[row * N + e]).abs();
                if vel_row(row) {
                    e_vel = e_vel.max(d);
                } else {
                    e_static = e_static.max(d);
                }
            }
        }
        (e_static, e_vel)
    };
    let (e_obs_s, e_obs_v) = split(&obs_g, &obs_c, OBS_DIM, &|row| {
        row >= jvel_lo && row < jvel_hi
    });
    let (e_crit_s, e_crit_v) = split(&critic_g, &critic_c, CRITIC_OBS_DIM, &|row| {
        (row >= jvel_lo && row < jvel_hi) || row >= OBS_DIM
    });
    let e_jp = jp_g
        .iter()
        .zip(&jp_c)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);

    println!("obs check (N={N}, J={J}, OBS_DIM={OBS_DIM}, CRITIC={CRITIC_OBS_DIM})");
    println!("  joint_pos          max|gpu-cpu| = {e_jp:.3e}");
    println!("  obs (position rows)            = {e_obs_s:.3e}");
    println!("  obs (velocity rows, /dt)       = {e_obs_v:.3e}");
    println!("  critic (position rows)         = {e_crit_s:.3e}");
    println!("  critic (velocity rows, /dt)    = {e_crit_v:.3e}");
    let e_static = e_jp.max(e_obs_s).max(e_crit_s);
    let e_vel = e_obs_v.max(e_crit_v);
    anyhow::ensure!(
        e_static < 5e-5,
        "obs position rows diverged (static {e_static:.3e})"
    );
    anyhow::ensure!(
        e_vel < 3e-3,
        "obs velocity rows diverged beyond 1/dt-amplified ULP ({e_vel:.3e})"
    );
    println!(
        "OK — gpu_obs matches the CPU observe/observe_critic reference (velocity rows within 1/dt·ULP)."
    );
    Ok(())
}
