//! Minimal RL-style training loop on the batched nexus GPU pendulum env.
//!
//! Learns a *linear* policy  action = W·obs  (obs = [u.x, u.z, u̇.x, u̇.z],
//! action = [ωx, ωz] motor velocities) that balances the 2-DOF ball-joint
//! pendulum — replacing the hand-tuned PD from `pendulum2dof`/`pendulum_batch`.
//!
//! Optimizer: CEM (cross-entropy method), gradient-free. The whole candidate
//! population is evaluated in ONE batched GPU rollout — env `e` is driven by
//! candidate `e / M`, so a generation costs `T` GPU steps regardless of POP.
//! Reward = mean uprightness (u.y = cos tilt). "Reset" = rebuild a fresh
//! randomized batch each generation. Reports the learning curve.
//!
//! Run: `cargo run --release --example pendulum_learn --features pendulum -- [seed]`

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

const OBS: usize = 4;
const ACT: usize = 2;
const NPARAM: usize = OBS * ACT; // 8 linear weights

const POP: usize = 32; // candidate policies per generation
const M: usize = 8; // envs averaged per candidate (fitness variance reduction)
const NB: usize = POP * M; // total batched envs = 256
const T: usize = 90; // rollout horizon (steps)
const GENS: usize = 14;
const ELITE: usize = 8;

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
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
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

fn policy(w: &[f32; NPARAM], obs: &[f32; OBS]) -> [f32; ACT] {
    let mut a = [0.0f32; ACT];
    for j in 0..ACT {
        let mut s = 0.0;
        for i in 0..OBS {
            s += w[j * OBS + i] * obs[i];
        }
        a[j] = s.clamp(-VMAX, VMAX);
    }
    a
}

/// Evaluate `pop` candidate policies in one batched rollout. Returns
/// (fitness[POP] = mean uprightness cos-tilt, upright_frac[POP]).
async fn evaluate(
    gpu: &KhalGpuBackend,
    pipeline: &GpuPhysicsPipeline,
    pop: &[[f32; NPARAM]; POP],
    gen_seed: u64,
) -> ([f32; POP], [f32; POP]) {
    let mut rng = Lcg(gen_seed | 1);
    let (mut bs, mut cs, mut ms) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..NB {
        let (b, c, m) = build_env(&mut rng);
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
    let mut state = GpuPhysicsState::from_rapier(gpu, &envs);
    let stride = state.num_colliders_per_batch() as usize;

    let pole = |p: &[Pose], e: usize| {
        let t = p[e * stride + 1].translation;
        Vec3::new(t.x, t.y, t.z).normalize()
    };
    let p = read_poses(gpu, &state).await;
    let mut u: Vec<Vec3> = (0..NB).map(|e| pole(&p, e)).collect();
    let mut prev = u.clone();
    let mut reward = vec![0.0f32; NB];
    let mut upsteps = vec![0u32; NB];

    for _ in 0..T {
        {
            let mm = state.multibodies_mut();
            for e in 0..NB {
                let obs = [
                    u[e].x,
                    u[e].z,
                    (u[e].x - prev[e].x) / DT,
                    (u[e].z - prev[e].z) / DT,
                ];
                let a = policy(&pop[e / M], &obs);
                let _ = mm.set_motor_velocity(gpu, e as u32, LINK_ID, JointAxis::AngX, a[0]);
                let _ = mm.set_motor_velocity(gpu, e as u32, LINK_ID, JointAxis::AngZ, a[1]);
            }
        }
        let _ = pipeline.step(gpu, &mut state, None).await;
        gpu.synchronize().expect("sync");
        pipeline.auto_resize_buffers(gpu, &mut state).await;
        let p = read_poses(gpu, &state).await;
        prev.copy_from_slice(&u);
        for e in 0..NB {
            u[e] = pole(&p, e);
            reward[e] += u[e].y; // cos(tilt): +1 upright, negative when fallen
            if u[e].y.clamp(-1.0, 1.0).acos() < UPRIGHT_TOL {
                upsteps[e] += 1;
            }
        }
    }

    let mut fit = [0.0f32; POP];
    let mut upf = [0.0f32; POP];
    for c in 0..POP {
        let mut r = 0.0;
        let mut up = 0.0;
        for k in 0..M {
            r += reward[c * M + k] / T as f32;
            up += upsteps[c * M + k] as f32 / T as f32;
        }
        fit[c] = r / M as f32;
        upf[c] = up / M as f32;
    }
    (fit, upf)
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
        let mut mean = [0.0f32; NPARAM]; // start from a do-nothing policy
        let mut std = [3.0f32; NPARAM]; // wide exploration

        println!(
            "CEM training: {POP} candidates × {M} envs (batch {NB}), {T}-step rollouts, {GENS} gens"
        );
        println!("policy: linear, {NPARAM} weights (obs=[ux,uz,u̇x,u̇z] → act=[ωx,ωz])\n");
        println!(
            "{:>3}  {:>10}  {:>10}  {:>12}",
            "gen", "best fit", "elite fit", "elite upright"
        );

        let mut best_mean = mean;
        for g in 0..GENS {
            // sample population ~ N(mean, std²)
            let mut pop = [[0.0f32; NPARAM]; POP];
            for c in 0..POP {
                for i in 0..NPARAM {
                    pop[c][i] = mean[i] + std[i] * rng.gauss();
                }
            }
            // keep one elite (the current mean) to avoid regressions
            pop[0] = mean;

            let (fit, upf) =
                evaluate(&gpu, &pipeline, &pop, seed.wrapping_add(g as u64 * 1009)).await;

            // rank by fitness, take elites
            let mut idx: Vec<usize> = (0..POP).collect();
            idx.sort_by(|&a, &b| fit[b].partial_cmp(&fit[a]).unwrap());
            let elite = &idx[..ELITE];

            // CEM update: mean/std of elite params
            let mut new_mean = [0.0f32; NPARAM];
            let mut new_std = [0.0f32; NPARAM];
            for &e in elite {
                for i in 0..NPARAM {
                    new_mean[i] += pop[e][i] / ELITE as f32;
                }
            }
            for &e in elite {
                for i in 0..NPARAM {
                    let d = pop[e][i] - new_mean[i];
                    new_std[i] += d * d / ELITE as f32;
                }
            }
            for i in 0..NPARAM {
                new_std[i] = new_std[i].sqrt().max(0.15); // variance floor
            }
            mean = new_mean;
            std = new_std;
            best_mean = pop[idx[0]];

            let elite_fit = elite.iter().map(|&e| fit[e]).sum::<f32>() / ELITE as f32;
            let elite_up = elite.iter().map(|&e| upf[e]).sum::<f32>() / ELITE as f32;
            println!(
                "{:>3}  {:>10.3}  {:>10.3}  {:>11.0}%",
                g,
                fit[idx[0]],
                elite_fit,
                elite_up * 100.0
            );
        }

        println!("\nlearned policy (best candidate), action = W·[ux,uz,u̇x,u̇z]:");
        println!(
            "  ωx = {:+.2}·ux {:+.2}·uz {:+.2}·u̇x {:+.2}·u̇z",
            best_mean[0], best_mean[1], best_mean[2], best_mean[3]
        );
        println!(
            "  ωz = {:+.2}·ux {:+.2}·uz {:+.2}·u̇x {:+.2}·u̇z",
            best_mean[4], best_mean[5], best_mean[6], best_mean[7]
        );
        println!("(hand-tuned PD for reference: ωx=-9·uz-1.2·u̇z,  ωz=+9·ux+1.2·u̇x)");
    });
}
