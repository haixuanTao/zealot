//! PPO swing-up + balance of the 2-DOF GPU pendulum.
//!
//! Harder task than the balance demos: each env starts ~hanging (pole down) and
//! must swing up and hold vertical. Reward = u.y (cos tilt): −1 hanging, +1 up,
//! so credit assignment is non-trivial → a value function + PPO earn their keep.
//!
//! Real RL: separate policy MLP (6→32→2 Gaussian mean) and value MLP (6→32→1),
//! GAE(λ) advantages, clipped surrogate over K epochs, hand-written backprop +
//! Adam. Sim is a black box; we differentiate only logπ and the value MSE. The
//! batched GPU env supplies many randomized rollouts per iteration.
//!
//! After training, rolls out the learned policy (deterministic, no exploration
//! noise) for one episode and writes the rod's pose per step to a CSV, so you can
//! watch the swing-up with `render_pend2dof.py`.
//!
//! Run: `cargo run --release --example pendulum_ppo --features pendulum -- [seed] [out.csv]`

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::dynamics::GpuSimParams;
use nexus3d::rbd::math::{Pose, Rotation as Quat};
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;

const ROD_LEN: f32 = 2.0;
const DT: f32 = 1.0 / 60.0;
const LINK_ID: u32 = 1;
const VMAX: f32 = 10.0;
const MOTOR_DAMPING: f32 = 40.0;
const MOTOR_MAX_FORCE: f32 = 14.0; // moderate: must work to get up, can hold at top

const NB: usize = 128;
const T: usize = 110;
const ITERS: usize = 45;

const NI: usize = 6; // [ux, uy, uz, u̇x, u̇y, u̇z]
const NH: usize = 32;
const NA: usize = 2; // [ωx, ωz]
const SIGMA: f32 = 0.8;
const GAMMA: f32 = 0.99;
const LAMBDA: f32 = 0.95;
const CLIP: f32 = 0.2;
const EPOCHS: usize = 5;
const LR_PI: f32 = 6e-3;
const LR_V: f32 = 1e-2;

const UPRIGHT_TOL: f32 = 25.0 * std::f32::consts::PI / 180.0;
// start ~hanging: tilt from up in [130°, 180°]
const TILT_MIN: f32 = 130.0 * std::f32::consts::PI / 180.0;
const TILT_MAX: f32 = 180.0 * std::f32::consts::PI / 180.0;

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
    fn gauss(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * self.unit()).cos()
    }
}

/// Runtime-sized 1-hidden-layer tanh MLP.
struct Mlp {
    ni: usize,
    nh: usize,
    no: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}
struct MlpGrad {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}
impl MlpGrad {
    fn zero(m: &Mlp) -> Self {
        MlpGrad {
            w1: vec![0.0; m.nh * m.ni],
            b1: vec![0.0; m.nh],
            w2: vec![0.0; m.no * m.nh],
            b2: vec![0.0; m.no],
        }
    }
}
impl Mlp {
    fn new(ni: usize, nh: usize, no: usize, out_scale: f32, rng: &mut Lcg) -> Self {
        let s1 = (1.0 / ni as f32).sqrt();
        let s2 = (1.0 / nh as f32).sqrt() * out_scale;
        Mlp {
            ni,
            nh,
            no,
            w1: (0..nh * ni).map(|_| rng.gauss() * s1).collect(),
            b1: vec![0.0; nh],
            w2: (0..no * nh).map(|_| rng.gauss() * s2).collect(),
            b2: vec![0.0; no],
        }
    }
    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![0.0f32; self.nh];
        for j in 0..self.nh {
            let mut z = self.b1[j];
            for i in 0..self.ni {
                z += self.w1[j * self.ni + i] * x[i];
            }
            h[j] = z.tanh();
        }
        let mut o = vec![0.0f32; self.no];
        for k in 0..self.no {
            let mut z = self.b2[k];
            for j in 0..self.nh {
                z += self.w2[k * self.nh + j] * h[j];
            }
            o[k] = z;
        }
        (o, h)
    }
    fn backward(&self, x: &[f32], h: &[f32], g_out: &[f32], g: &mut MlpGrad) {
        let mut dh = vec![0.0f32; self.nh];
        for k in 0..self.no {
            for j in 0..self.nh {
                g.w2[k * self.nh + j] += g_out[k] * h[j];
                dh[j] += g_out[k] * self.w2[k * self.nh + j];
            }
            g.b2[k] += g_out[k];
        }
        for j in 0..self.nh {
            let dz = dh[j] * (1.0 - h[j] * h[j]);
            for i in 0..self.ni {
                g.w1[j * self.ni + i] += dz * x[i];
            }
            g.b1[j] += dz;
        }
    }
}

struct Adam {
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: i32,
}
impl Adam {
    fn new(net: &Mlp) -> Self {
        let shapes = [net.w1.len(), net.b1.len(), net.w2.len(), net.b2.len()];
        Adam {
            m: shapes.iter().map(|&n| vec![0.0; n]).collect(),
            v: shapes.iter().map(|&n| vec![0.0; n]).collect(),
            t: 0,
        }
    }
    fn step(&mut self, net: &mut Mlp, g: &MlpGrad, lr: f32) {
        self.t += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let (bc1, bc2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        let mut go = |pi: usize, p: &mut [f32], gr: &[f32]| {
            for i in 0..p.len() {
                self.m[pi][i] = b1 * self.m[pi][i] + (1.0 - b1) * gr[i];
                self.v[pi][i] = b2 * self.v[pi][i] + (1.0 - b2) * gr[i] * gr[i];
                p[i] -= lr * (self.m[pi][i] / bc1) / ((self.v[pi][i] / bc2).sqrt() + eps);
            }
        };
        go(0, &mut net.w1, &g.w1);
        go(1, &mut net.b1, &g.b1);
        go(2, &mut net.w2, &g.w2);
        go(3, &mut net.b2, &g.b2);
    }
}

fn build_env(rng: &mut Lcg) -> (RigidBodySet, ColliderSet, MultibodyJointSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut mb = MultibodyJointSet::new();
    let root = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(ColliderBuilder::cuboid(0.1, 0.1, 0.1), root, &mut bodies);
    let az = rng.range(0.0, std::f32::consts::TAU);
    let tilt = rng.range(TILT_MIN, TILT_MAX);
    let u = Vec3::new(tilt.sin() * az.cos(), tilt.cos(), tilt.sin() * az.sin()).normalize();
    let rot = Quat::from_rotation_arc(Vec3::X, u);
    let rod = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(u * ROD_LEN)
            .rotation(rot.to_scaled_axis()),
    );
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(ROD_LEN * 0.5, 0.12, 0.12).density(1.0),
        rod,
        &mut bodies,
    );
    let mut joint = SphericalJointBuilder::new()
        .local_anchor1(Vec3::ZERO)
        .local_anchor2(Vec3::new(-ROD_LEN, 0.0, 0.0));
    for axis in [JointAxis::AngX, JointAxis::AngZ] {
        joint = joint
            .motor_velocity(axis, 0.0, MOTOR_DAMPING)
            .motor_max_force(axis, MOTOR_MAX_FORCE);
    }
    mb.insert(root, rod, joint.build(), true);
    (bodies, colliders, mb)
}

async fn webgpu_backend() -> KhalGpuBackend {
    let limits = wgpu::Limits {
        max_buffer_size: 1_200_000_000,
        max_storage_buffer_binding_size: 1_200_000_000,
        max_storage_buffers_per_shader_stage: 14,
        max_compute_workgroup_storage_size: 19_904,
        ..Default::default()
    };
    let mut bk = KhalGpuBackend::auto(wgpu::Features::default(), limits)
        .await
        .expect("backend");
    // force_buffer_copy_src is a WebGPU-only readback workaround; no-op on CUDA.
    #[allow(irrefutable_let_patterns)]
    if let KhalGpuBackend::WebGpu(w) = &mut bk {
        w.force_buffer_copy_src = true;
    }
    bk
}

async fn read_poses(gpu: &KhalGpuBackend, state: &GpuPhysicsState) -> Vec<Pose> {
    gpu.slow_read_vec(state.poses().buffer())
        .await
        .expect("poses")
}

fn logp_gauss(a: &[f32], mu: &[f32]) -> f32 {
    let inv = 1.0 / (SIGMA * SIGMA);
    let mut s = 0.0;
    for k in 0..NA {
        let d = a[k] - mu[k];
        s += -0.5 * d * d * inv - SIGMA.ln() - 0.5 * (std::f32::consts::TAU).ln();
    }
    s
}

/// Roll out the trained policy on one fresh (hanging) env, taking the mean
/// (noise-free) action each step, and write the rod's pose per step as a CSV in
/// the `step,x,y,z,qx,qy,qz,qw` format `render_pend2dof.py` expects.
async fn record_rollout(
    gpu: &KhalGpuBackend,
    pipeline: &GpuPhysicsPipeline,
    pi: &Mlp,
    seed: u64,
    steps: usize,
    out: &str,
) {
    let mut es = Lcg(seed | 1);
    let (b, c, m) = build_env(&mut es);
    let impulse = ImpulseJointSet::new();
    let mut sp = GpuSimParams::default();
    sp.dt = DT;
    sp.num_solver_iterations = 8;
    let envs = vec![(&b, &c, &impulse, &m, &sp)];
    let mut state = GpuPhysicsState::from_rapier(gpu, &envs);

    // collider index 1 within the (single) env is the rod (index 0 is the root).
    let rod_dir = |p: &[Pose]| {
        let t = p[1].translation;
        Vec3::new(t.x, t.y, t.z).normalize()
    };
    let mut rows = String::from("step,x,y,z,qx,qy,qz,qw\n");
    let push = |rows: &mut String, step: usize, p: &Pose| {
        let (t, q) = (p.translation, p.rotation);
        rows.push_str(&format!(
            "{step},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}\n",
            t.x, t.y, t.z, q.x, q.y, q.z, q.w
        ));
    };

    let p0 = read_poses(gpu, &state).await;
    push(&mut rows, 0, &p0[1]);
    let mut u = rod_dir(&p0);
    let mut prev = u;

    for step in 1..=steps {
        {
            let du = (u - prev) * (1.0 / DT);
            let x = [u.x, u.y, u.z, du.x, du.y, du.z];
            let (mu, _) = pi.forward(&x); // deterministic: mean action, no σ·ε
            let mm = state.multibodies_mut();
            let _ =
                mm.set_motor_velocity(gpu, 0, LINK_ID, JointAxis::AngX, mu[0].clamp(-VMAX, VMAX));
            let _ =
                mm.set_motor_velocity(gpu, 0, LINK_ID, JointAxis::AngZ, mu[1].clamp(-VMAX, VMAX));
        }
        let _ = pipeline.step(gpu, &mut state, None).await;
        gpu.synchronize().expect("sync");
        pipeline.auto_resize_buffers(gpu, &mut state).await;
        let p = read_poses(gpu, &state).await;
        push(&mut rows, step, &p[1]);
        prev = u;
        u = rod_dir(&p);
    }

    std::fs::write(out, &rows).expect("write csv");
    println!("\nrecorded {steps}-step rollout of the trained policy → {out}");
    println!("visualize: python3 render_pend2dof.py {out} /tmp/pendulum_ppo.mp4");
}

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let out_csv = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/pendulum_ppo.csv".to_string());
    pollster::block_on(async {
        let gpu = webgpu_backend().await;
        let pipeline = GpuPhysicsPipeline::from_backend(&gpu);
        let mut rng = Lcg(seed | 1);
        let mut pi = Mlp::new(NI, NH, NA, 0.1, &mut rng); // small init → gentle start
        let mut vf = Mlp::new(NI, NH, 1, 1.0, &mut rng);
        let mut opt_pi = Adam::new(&pi);
        let mut opt_v = Adam::new(&vf);

        println!(
            "PPO swing-up: policy {NI}-{NH}-{NA}, value {NI}-{NH}-1, σ={SIGMA}, clip={CLIP}, {EPOCHS} epochs/iter"
        );
        println!(
            "batch {NB} × {T} steps, {ITERS} iters; start hanging (tilt 130–180°), reward=cos(tilt)\n"
        );
        println!(
            "{:>4}  {:>9}  {:>9}  {:>9}",
            "iter", "mean rew", "upright", "final up"
        );

        let ns = NB * T;
        let mut obs = vec![[0.0f32; NI]; ns];
        let mut act = vec![[0.0f32; NA]; ns];
        let mut logp_old = vec![0.0f32; ns];
        let mut val = vec![0.0f32; ns];
        let mut rew = vec![0.0f32; ns];

        for it in 0..ITERS {
            let mut es = Lcg(seed.wrapping_add(it as u64 * 2654435761) | 1);
            let (mut bs, mut cs, mut ms) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..NB {
                let (b, c, m) = build_env(&mut es);
                bs.push(b);
                cs.push(c);
                ms.push(m);
            }
            let impulse = ImpulseJointSet::new();
            let mut sp = GpuSimParams::default();
            sp.dt = DT;
            sp.num_solver_iterations = 8;
            let envs: Vec<_> = (0..NB)
                .map(|i| (&bs[i], &cs[i], &impulse, &ms[i], &sp))
                .collect();
            let mut state = GpuPhysicsState::from_rapier(&gpu, &envs);
            let stride = state.num_colliders_per_batch() as usize;
            let pole = |p: &[Pose], e: usize| {
                let t = p[e * stride + 1].translation;
                Vec3::new(t.x, t.y, t.z).normalize()
            };

            let p0 = read_poses(&gpu, &state).await;
            let mut u: Vec<Vec3> = (0..NB).map(|e| pole(&p0, e)).collect();
            let mut prev = u.clone();
            let mut up_steps = 0u64;
            let mut final_up = 0u64;

            for t in 0..T {
                {
                    let mm = state.multibodies_mut();
                    for e in 0..NB {
                        let du = (u[e] - prev[e]) * (1.0 / DT);
                        let x = [u[e].x, u[e].y, u[e].z, du.x, du.y, du.z];
                        let (mu, _) = pi.forward(&x);
                        let (v, _) = vf.forward(&x);
                        let mut a = [0.0f32; NA];
                        for k in 0..NA {
                            a[k] = (mu[k] + SIGMA * rng.gauss()).clamp(-VMAX, VMAX);
                        }
                        let i = e * T + t;
                        obs[i] = x;
                        act[i] = a;
                        logp_old[i] = logp_gauss(&a, &mu);
                        val[i] = v[0];
                        let _ =
                            mm.set_motor_velocity(&gpu, e as u32, LINK_ID, JointAxis::AngX, a[0]);
                        let _ =
                            mm.set_motor_velocity(&gpu, e as u32, LINK_ID, JointAxis::AngZ, a[1]);
                    }
                }
                let _ = pipeline.step(&gpu, &mut state, None).await;
                gpu.synchronize().expect("sync");
                pipeline.auto_resize_buffers(&gpu, &mut state).await;
                let p = read_poses(&gpu, &state).await;
                prev.copy_from_slice(&u);
                for e in 0..NB {
                    u[e] = pole(&p, e);
                    let i = e * T + t;
                    let pen = 0.001 * (act[i][0] * act[i][0] + act[i][1] * act[i][1]);
                    rew[i] = u[e].y - pen;
                    let upr = u[e].y.clamp(-1.0, 1.0).acos() < UPRIGHT_TOL;
                    if upr {
                        up_steps += 1;
                    }
                    if t >= T - 20 && upr {
                        final_up += 1;
                    }
                }
            }

            // GAE(λ) + returns, bootstrap with V(last obs) per env
            let mut adv = vec![0.0f32; ns];
            let mut ret = vec![0.0f32; ns];
            for e in 0..NB {
                let du = (u[e] - prev[e]) * (1.0 / DT);
                let xl = [u[e].x, u[e].y, u[e].z, du.x, du.y, du.z];
                let mut next_v = vf.forward(&xl).0[0];
                let mut gae = 0.0;
                for t in (0..T).rev() {
                    let i = e * T + t;
                    let delta = rew[i] + GAMMA * next_v - val[i];
                    gae = delta + GAMMA * LAMBDA * gae;
                    adv[i] = gae;
                    ret[i] = gae + val[i];
                    next_v = val[i];
                }
            }
            // normalize advantages
            let (mut m, mut s) = (0.0f32, 0.0f32);
            for a in &adv {
                m += *a;
            }
            m /= ns as f32;
            for a in &adv {
                s += (*a - m) * (*a - m);
            }
            let sd = (s / ns as f32).sqrt().max(1e-4);
            for a in adv.iter_mut() {
                *a = (*a - m) / sd;
            }

            // PPO epochs (full-batch)
            let inv = 1.0 / (SIGMA * SIGMA);
            for _ in 0..EPOCHS {
                let mut gp = MlpGrad::zero(&pi);
                let mut gv = MlpGrad::zero(&vf);
                let scale = 1.0 / ns as f32;
                for i in 0..ns {
                    let (mu, h) = pi.forward(&obs[i]);
                    let lp = logp_gauss(&act[i], &mu);
                    let ratio = (lp - logp_old[i]).exp();
                    let a = adv[i];
                    // clipped surrogate: gradient zero in the clipped-binding region
                    let clipped =
                        (a >= 0.0 && ratio > 1.0 + CLIP) || (a < 0.0 && ratio < 1.0 - CLIP);
                    let mut g_mu = [0.0f32; NA];
                    if !clipped {
                        for k in 0..NA {
                            // dL/dμ = A·ratio·(a−μ)/σ²  (ascend ⇒ negate for minimize)
                            g_mu[k] = -(a * ratio * (act[i][k] - mu[k]) * inv) * scale;
                        }
                    }
                    pi.backward(&obs[i], &h, &g_mu, &mut gp);
                    // value MSE: dL/dout = 2(V−ret)
                    let (v, hv) = vf.forward(&obs[i]);
                    let gv_out = [2.0 * (v[0] - ret[i]) * scale];
                    vf.backward(&obs[i], &hv, &gv_out, &mut gv);
                }
                opt_pi.step(&mut pi, &gp, LR_PI);
                opt_v.step(&mut vf, &gv, LR_V);
            }

            let mean_rew = rew.iter().sum::<f32>() / ns as f32;
            println!(
                "{:>4}  {:>9.3}  {:>8.0}%  {:>8.0}%",
                it,
                mean_rew,
                up_steps as f32 / ns as f32 * 100.0,
                final_up as f32 / (NB * 20) as f32 * 100.0
            );
        }
        println!(
            "\nPPO done — policy {} + value {} weights; swing-up from hanging on the batched GPU env.",
            NH * NI + NH + NA * NH + NA,
            NH * NI + NH + NH + 1
        );

        // Record one deterministic episode of the trained policy for rendering.
        record_rollout(
            &gpu,
            &pipeline,
            &pi,
            seed.wrapping_add(12345),
            300,
            &out_csv,
        )
        .await;
    });
}
