//! Policy-gradient (REINFORCE) training of an MLP policy on the batched nexus
//! GPU pendulum env — a real neural net trained with real gradients.
//!
//! Policy: MLP  obs(4) -> 16 tanh -> mean(2), Gaussian actions a = μ + σ·ε.
//! The simulator is a black box; we only differentiate the policy's log-prob:
//!   ∇θ J = E[ A · ∇θ log π(a|s) ],   ∇μ log π = (a − μ)/σ².
//! Forward + backprop + Adam are hand-written (CPU; the net is tiny, the physics
//! stays on the GPU). Reward = uprightness (u.y = cos tilt); advantages use a
//! per-timestep baseline + normalization. "Reset" = fresh randomized batch/iter.
//!
//! Run: `cargo run --release --example pendulum_pg --features pendulum -- [seed]`

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::dynamics::GpuSimParams;
use nexus3d::rbd::math::{Pose, Rotation as Quat};
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;

const ROD_LEN: f32 = 2.0;
const DT: f32 = 1.0 / 60.0;
const LINK_ID: u32 = 1;
const VMAX: f32 = 8.0;
const MOTOR_DAMPING: f32 = 50.0;
const MOTOR_MAX_FORCE: f32 = 60.0;

const NB: usize = 256; // batched envs per iteration (gradient samples)
const T: usize = 70; // rollout horizon
const ITERS: usize = 40;

const NI: usize = 4; // obs: [ux, uz, u̇x, u̇z]
const NH: usize = 16; // hidden units
const NO: usize = 2; // action mean: [ωx, ωz]
const SIGMA: f32 = 1.2; // exploration std on the motor-velocity action
const GAMMA: f32 = 0.98;
const LR: f32 = 0.02;

const TILT_MIN: f32 = 12.0 * std::f32::consts::PI / 180.0;
const TILT_MAX: f32 = 40.0 * std::f32::consts::PI / 180.0;
const UPRIGHT_TOL: f32 = 25.0 * std::f32::consts::PI / 180.0;

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

/// MLP params (flat) + Adam moments.
struct Mlp {
    w1: Vec<f32>,
    b1: Vec<f32>, // NH×NI, NH
    w2: Vec<f32>,
    b2: Vec<f32>, // NO×NH, NO
}
struct Grad {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}
impl Grad {
    fn zero() -> Self {
        Grad {
            w1: vec![0.0; NH * NI],
            b1: vec![0.0; NH],
            w2: vec![0.0; NO * NH],
            b2: vec![0.0; NO],
        }
    }
}

impl Mlp {
    fn new(rng: &mut Lcg) -> Self {
        let s1 = (1.0 / NI as f32).sqrt();
        let s2 = (1.0 / NH as f32).sqrt();
        Mlp {
            w1: (0..NH * NI).map(|_| rng.gauss() * s1).collect(),
            b1: vec![0.0; NH],
            w2: (0..NO * NH).map(|_| rng.gauss() * s2 * 0.3).collect(), // small output → start gentle
            b2: vec![0.0; NO],
        }
    }
    /// forward; returns (mu[NO], h[NH]) with h = tanh activations (kept for backprop)
    fn forward(&self, x: &[f32; NI]) -> ([f32; NO], [f32; NH]) {
        let mut h = [0.0f32; NH];
        for j in 0..NH {
            let mut z = self.b1[j];
            for i in 0..NI {
                z += self.w1[j * NI + i] * x[i];
            }
            h[j] = z.tanh();
        }
        let mut mu = [0.0f32; NO];
        for k in 0..NO {
            let mut z = self.b2[k];
            for j in 0..NH {
                z += self.w2[k * NH + j] * h[j];
            }
            mu[k] = z;
        }
        (mu, h)
    }
    /// accumulate dLoss/dparams given dLoss/dmu (g_mu), recomputing the hidden layer.
    fn backward(&self, x: &[f32; NI], h: &[f32; NH], g_mu: &[f32; NO], g: &mut Grad) {
        let mut dh = [0.0f32; NH];
        for k in 0..NO {
            for j in 0..NH {
                g.w2[k * NH + j] += g_mu[k] * h[j];
                dh[j] += g_mu[k] * self.w2[k * NH + j];
            }
            g.b2[k] += g_mu[k];
        }
        for j in 0..NH {
            let dz = dh[j] * (1.0 - h[j] * h[j]); // tanh'
            for i in 0..NI {
                g.w1[j * NI + i] += dz * x[i];
            }
            g.b1[j] += dz;
        }
    }
}

/// Adam optimizer over the four param tensors.
struct Adam {
    mw1: Vec<f32>,
    vw1: Vec<f32>,
    mb1: Vec<f32>,
    vb1: Vec<f32>,
    mw2: Vec<f32>,
    vw2: Vec<f32>,
    mb2: Vec<f32>,
    vb2: Vec<f32>,
    t: i32,
}
impl Adam {
    fn new() -> Self {
        Adam {
            mw1: vec![0.0; NH * NI],
            vw1: vec![0.0; NH * NI],
            mb1: vec![0.0; NH],
            vb1: vec![0.0; NH],
            mw2: vec![0.0; NO * NH],
            vw2: vec![0.0; NO * NH],
            mb2: vec![0.0; NO],
            vb2: vec![0.0; NO],
            t: 0,
        }
    }
    fn step(&mut self, p: &mut Mlp, g: &Grad) {
        self.t += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(self.t);
        let bc2 = 1.0 - b2.powi(self.t);
        let upd = |param: &mut [f32], grad: &[f32], m: &mut [f32], v: &mut [f32]| {
            for i in 0..param.len() {
                m[i] = b1 * m[i] + (1.0 - b1) * grad[i];
                v[i] = b2 * v[i] + (1.0 - b2) * grad[i] * grad[i];
                let mh = m[i] / bc1;
                let vh = v[i] / bc2;
                param[i] -= LR * mh / (vh.sqrt() + eps);
            }
        };
        upd(&mut p.w1, &g.w1, &mut self.mw1, &mut self.vw1);
        upd(&mut p.b1, &g.b1, &mut self.mb1, &mut self.vb1);
        upd(&mut p.w2, &g.w2, &mut self.mw2, &mut self.vw2);
        upd(&mut p.b2, &g.b2, &mut self.mb2, &mut self.vb2);
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
    let mut w = WebGpu::new(wgpu::Features::default(), limits)
        .await
        .expect("webgpu");
    w.force_buffer_copy_src = true;
    KhalGpuBackend::WebGpu(w)
}

async fn read_poses(gpu: &KhalGpuBackend, state: &GpuPhysicsState) -> Vec<Pose> {
    gpu.slow_read_vec(state.poses().buffer())
        .await
        .expect("poses")
}

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    pollster::block_on(async {
        let gpu = webgpu_backend().await;
        let pipeline = GpuPhysicsPipeline::from_backend(&gpu);
        let mut rng = Lcg(seed | 1);
        let mut net = Mlp::new(&mut rng);
        let mut adam = Adam::new();

        println!(
            "REINFORCE: MLP {NI}-{NH}(tanh)-{NO}, σ={SIGMA}, lr={LR}, batch {NB}, {T}-step rollouts, {ITERS} iters\n"
        );
        println!("{:>4}  {:>10}  {:>12}", "iter", "mean rew", "upright");

        // rollout buffers, indexed (e*T + t)
        let mut obs = vec![[0.0f32; NI]; NB * T];
        let mut noise = vec![[0.0f32; NO]; NB * T]; // (a − μ) actually applied
        let mut rew = vec![0.0f32; NB * T];

        for it in 0..ITERS {
            // fresh randomized batch
            let mut env_seed = Lcg(seed.wrapping_add(it as u64 * 2654435761) | 1);
            let (mut bs, mut cs, mut ms) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..NB {
                let (b, c, m) = build_env(&mut env_seed);
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

            for t in 0..T {
                {
                    let mm = state.multibodies_mut();
                    for e in 0..NB {
                        let x = [
                            u[e].x,
                            u[e].z,
                            (u[e].x - prev[e].x) / DT,
                            (u[e].z - prev[e].z) / DT,
                        ];
                        let (mu, _h) = net.forward(&x);
                        let mut a = [0.0f32; NO];
                        let mut nz = [0.0f32; NO];
                        for k in 0..NO {
                            let raw = mu[k] + SIGMA * rng.gauss();
                            a[k] = raw.clamp(-VMAX, VMAX);
                            nz[k] = a[k] - mu[k]; // applied perturbation
                        }
                        obs[e * T + t] = x;
                        noise[e * T + t] = nz;
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
                    rew[e * T + t] = u[e].y; // reward = cos(tilt)
                    if u[e].y.clamp(-1.0, 1.0).acos() < UPRIGHT_TOL {
                        up_steps += 1;
                    }
                }
            }

            // returns-to-go, per-timestep baseline, normalized advantage
            let mut adv = vec![0.0f32; NB * T];
            for e in 0..NB {
                let mut g = 0.0;
                for t in (0..T).rev() {
                    g = rew[e * T + t] + GAMMA * g;
                    adv[e * T + t] = g;
                }
            }
            for t in 0..T {
                let mut mean = 0.0;
                for e in 0..NB {
                    mean += adv[e * T + t];
                }
                mean /= NB as f32;
                for e in 0..NB {
                    adv[e * T + t] -= mean; // subtract baseline b_t
                }
            }
            let (mut m, mut s) = (0.0f32, 0.0f32);
            for a in &adv {
                m += *a;
            }
            m /= adv.len() as f32;
            for a in &adv {
                s += (*a - m) * (*a - m);
            }
            let std = (s / adv.len() as f32).sqrt().max(1e-4);
            for a in adv.iter_mut() {
                *a = (*a - m) / std;
            }

            // policy-gradient: dLoss/dμ = −A · (a−μ)/σ²   (minimize ⇒ ascend J)
            let mut grad = Grad::zero();
            let inv = 1.0 / (SIGMA * SIGMA);
            let scale = 1.0 / (NB * T) as f32;
            for e in 0..NB {
                for t in 0..T {
                    let i = e * T + t;
                    let (_, h) = net.forward(&obs[i]);
                    let a = adv[i];
                    let g_mu = [
                        -(a * noise[i][0] * inv) * scale,
                        -(a * noise[i][1] * inv) * scale,
                    ];
                    net.backward(&obs[i], &h, &g_mu, &mut grad);
                }
            }
            adam.step(&mut net, &grad);

            let mean_rew = rew.iter().sum::<f32>() / (NB * T) as f32;
            let upright = up_steps as f32 / (NB * T) as f32;
            println!("{:>4}  {:>10.3}  {:>11.0}%", it, mean_rew, upright * 100.0);
        }
        println!(
            "\ntrained MLP policy ({} weights) — gradient-based, on the batched GPU env.",
            NH * NI + NH + NO * NH + NO
        );
    });
}
