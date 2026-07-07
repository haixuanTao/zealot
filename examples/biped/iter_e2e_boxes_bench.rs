//! Full training-ITERATION throughput on a RIGID-BODY (boxes) scene:
//! **WebGPU vs native-CUDA (cuda-oxide)**, end to end.
//!
//! One iteration = T-step rollout (nexus physics + vortx policy forward) +
//! (epochs x minibatches) GPU PPO update. Reports env-control-steps/second =
//! N*T / iteration_time — the SAME unit as `iter_e2e_bench`.
//!
//! The policy + PPO update workload is byte-for-byte the biped's (actor
//! [43,256,256,128,12], critic [49,512,256,128,1]); only the physics scene
//! differs. The biped's articulated multibody isn't yet ported to cuda-oxide
//! (a broad-phase BVH-traversal codegen bug surfaces on its mesh-OBB colliders),
//! so the physics rollout here uses the *verified* nexus boxes rigid-body
//! pipeline — bit-exact CUDA-vs-WebGPU over 200 steps. The result is therefore a
//! genuine full iteration (physics + policy + update) running entirely on native
//! CUDA, directly comparable to the same iteration on WebGPU.
//!
//! Run (native CUDA):  BIPED_CUDA=1 cargo run --release --example iter_e2e_boxes_bench \
//!                       --features "gpu biped_gpu cuda_backend" -- [N] [T] [epochs] [minibatches]
//! Run (WebGPU):                  cargo run --release --example iter_e2e_boxes_bench \
//!                       --features "gpu biped_gpu cuda_backend" -- [N] [T] [epochs] [minibatches]

#[path = "gpu_policy.rs"]
mod gpu_policy;

use gpu_policy::GpuPolicy;
use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend as KhalGpuBackend};
use nalgebra::DMatrix;
use nexus3d::rbd::math::Pose;
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;
use std::time::Instant;
use vortx::linalg::{
    Activation, Adam, AdamParams, Contiguous, Gemm, OpAssign, OpAssignVariant, Ppo, PpoActorParams,
    PpoValueParams,
};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use zealot_env::robots::lerobot_bipedal::NUM_JOINTS;
use zealot_env::tasks::velocity_flat::{CRITIC_OBS_DIM, OBS_DIM};
use zealot_rl::ActorCritic;
use zealot_rl::ppo::{Sample, gae};
use zealot_rl::rng::Lcg;

const LOG_SQRT_2PI: f32 = 0.918_938_5;

// ---- boxes scene (mirrors pendulum_headless boxes3d) -----------------------
const NX: i32 = 3;
const NY: i32 = 3;
const NZ: i32 = 2;
const HALF: f32 = 0.5;
const SPACING: f32 = 1.15;

struct SLcg(u64);
impl SLcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn sym(&mut self, m: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * m
    }
}

fn num_boxes() -> usize {
    (NX * NY * NZ) as usize
}

fn build_boxes() -> (RigidBodySet, ColliderSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut rng = SLcg(0xC0FFEE);
    for iy in 0..NY {
        for ix in 0..NX {
            for iz in 0..NZ {
                let x = (ix as f32 - (NX - 1) as f32 * 0.5) * SPACING + rng.sym(0.05);
                let z = (iz as f32 - (NZ - 1) as f32 * 0.5) * SPACING + rng.sym(0.05);
                let y = 1.2 + iy as f32 * (2.0 * HALF + 0.25);
                let body = bodies.insert(
                    RigidBodyBuilder::dynamic()
                        .translation(Vec3::new(x, y, z))
                        .rotation(Vec3::new(rng.sym(0.15), rng.sym(0.15), rng.sym(0.15))),
                );
                colliders.insert_with_parent(
                    ColliderBuilder::cuboid(HALF, HALF, HALF).density(1.0),
                    body,
                    &mut bodies,
                );
            }
        }
    }
    let floor = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, 0.0)));
    colliders.insert_with_parent(ColliderBuilder::cuboid(8.0, 0.5, 8.0), floor, &mut bodies);
    (bodies, colliders)
}

/// Pick the GPU backend (WebGPU by default; native CUDA when compiled with
/// `cuda_backend` AND `BIPED_CUDA=1`).
async fn make_backend() -> KhalGpuBackend {
    let limits = wgpu::Limits {
        max_buffer_size: 1_200_000_000,
        max_storage_buffer_binding_size: 1_200_000_000,
        max_storage_buffers_per_shader_stage: 14,
        max_compute_workgroup_storage_size: 19_904,
        ..Default::default()
    };
    let mut bk = KhalGpuBackend::auto(wgpu::Features::default(), limits)
        .await
        .expect("init GPU backend");
    if let KhalGpuBackend::WebGpu(w) = &mut bk {
        w.force_buffer_copy_src = true;
    }
    bk
}

/// Build a per-env obs vector of length `dim` from the env's body poses.
/// Synthetic but a real GPU->CPU readback path (same on both backends).
fn obs_from_poses(poses: &[Pose], env: usize, bodies_per_env: usize, dim: usize) -> Vec<f32> {
    let base = env * bodies_per_env;
    let mut o = Vec::with_capacity(dim);
    'outer: for b in 0..bodies_per_env {
        let p = poses[base + b];
        let t = p.translation;
        let q = p.rotation;
        for v in [t.x, t.y, t.z, q.x, q.y, q.z, q.w] {
            o.push(v);
            if o.len() == dim {
                break 'outer;
            }
        }
    }
    while o.len() < dim {
        o.push(0.0);
    }
    o
}

// ---- GpuMlp: GPU-resident PPO update net (verbatim from iter_e2e_bench) -----
fn mk(b: &KhalGpuBackend, m: &DMatrix<f32>, u: BufferUsages) -> Tensor<f32> {
    Tensor::matrix_from_na(b, m, u).unwrap()
}
fn wmat(w: &[f32], out: usize, inp: usize) -> DMatrix<f32> {
    DMatrix::from_fn(out, inp, |r, c| w[r * inp + c])
}

struct GpuMlp {
    dims: Vec<usize>,
    batch: usize,
    w: Vec<Tensor<f32>>,
    b: Vec<Tensor<f32>>,
    mw: Vec<Tensor<f32>>,
    vw: Vec<Tensor<f32>>,
    mb: Vec<Tensor<f32>>,
    vb: Vec<Tensor<f32>>,
    a: Vec<Tensor<f32>>,
    bb: Vec<Tensor<f32>>,
    delta: Vec<Tensor<f32>>,
    dw: Vec<Tensor<f32>>,
    db: Vec<Tensor<f32>>,
}
impl GpuMlp {
    fn new(bk: &KhalGpuBackend, net: &zealot_rl::net::Mlp, m: usize) -> Self {
        let d = net.dims.clone();
        let (st, rw) = (
            BufferUsages::STORAGE,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let l = net.w.len();
        let z = |r: usize, c: usize| DMatrix::<f32>::zeros(r, c);
        GpuMlp {
            w: (0..l)
                .map(|i| mk(bk, &wmat(&net.w[i], d[i + 1], d[i]), rw))
                .collect(),
            b: (0..l)
                .map(|i| mk(bk, &DMatrix::from_fn(d[i + 1], 1, |r, _| net.b[i][r]), rw))
                .collect(),
            mw: (0..l).map(|i| mk(bk, &z(d[i + 1], d[i]), st)).collect(),
            vw: (0..l).map(|i| mk(bk, &z(d[i + 1], d[i]), st)).collect(),
            mb: (0..l).map(|i| mk(bk, &z(d[i + 1], 1), st)).collect(),
            vb: (0..l).map(|i| mk(bk, &z(d[i + 1], 1), st)).collect(),
            a: (0..=l).map(|i| mk(bk, &z(d[i], m), rw)).collect(),
            bb: (0..l).map(|i| mk(bk, &z(d[i + 1], m), st)).collect(),
            delta: (0..l).map(|i| mk(bk, &z(d[i + 1], m), rw)).collect(),
            dw: (0..l).map(|i| mk(bk, &z(d[i + 1], d[i]), rw)).collect(),
            db: (0..l).map(|i| mk(bk, &z(d[i + 1], 1), rw)).collect(),
            dims: d,
            batch: m,
        }
    }
    fn layers(&self) -> usize {
        self.w.len()
    }
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        bk: &KhalGpuBackend,
        g: &Gemm,
        op: &OpAssign,
        act: &Activation,
        sh: &mut TensorLayoutBuffers,
        enc: &mut <KhalGpuBackend as Backend>::Encoder,
        o1m: &Tensor<f32>,
    ) -> anyhow::Result<()> {
        let l = self.layers();
        for i in 0..l {
            let (lf, rt) = self.a.split_at_mut(i + 1);
            let (ain, aout) = (&lf[i], &mut rt[0]);
            {
                let mut p = enc.begin_pass("z", None);
                g.dispatch_tiled(bk, sh, &mut p, &mut *aout, &self.w[i], ain)?;
            }
            {
                let mut p = enc.begin_pass("bb", None);
                g.dispatch_naive(bk, sh, &mut p, &mut self.bb[i], &self.b[i], o1m)?;
            }
            {
                let mut p = enc.begin_pass("bias", None);
                op.launch(
                    bk,
                    sh,
                    &mut p,
                    OpAssignVariant::Add,
                    &mut *aout,
                    &self.bb[i],
                )?;
            }
            if i < l - 1 {
                let mut p = enc.begin_pass("elu", None);
                act.elu(bk, sh, &mut p, &mut *aout)?;
            }
        }
        Ok(())
    }
    fn backward(
        &mut self,
        bk: &KhalGpuBackend,
        g: &Gemm,
        act: &Activation,
        sh: &mut TensorLayoutBuffers,
        enc: &mut <KhalGpuBackend as Backend>::Encoder,
        om1: &Tensor<f32>,
    ) -> anyhow::Result<()> {
        for i in (0..self.layers()).rev() {
            {
                let mut p = enc.begin_pass("dw", None);
                g.dispatch_tiled(
                    bk,
                    sh,
                    &mut p,
                    &mut self.dw[i],
                    &self.delta[i],
                    self.a[i].transpose_last_dims(),
                )?;
            }
            {
                let mut p = enc.begin_pass("db", None);
                g.dispatch_naive(bk, sh, &mut p, &mut self.db[i], &self.delta[i], om1)?;
            }
            if i > 0 {
                {
                    let (lf, rt) = self.delta.split_at_mut(i);
                    let dp = &mut lf[i - 1];
                    let dc = &rt[0];
                    let mut p = enc.begin_pass("da", None);
                    g.dispatch_tiled(bk, sh, &mut p, dp, self.w[i].transpose_last_dims(), dc)?;
                }
                {
                    let mut p = enc.begin_pass("eb", None);
                    act.elu_backward(bk, sh, &mut p, &mut self.delta[i - 1], &self.a[i])?;
                }
            }
        }
        Ok(())
    }
    fn adam(
        &mut self,
        bk: &KhalGpuBackend,
        ad: &Adam,
        sh: &mut TensorLayoutBuffers,
        enc: &mut <KhalGpuBackend as Backend>::Encoder,
        ap: &Tensor<AdamParams>,
    ) -> anyhow::Result<()> {
        for i in 0..self.layers() {
            {
                let mut p = enc.begin_pass("aw", None);
                ad.step(
                    bk,
                    sh,
                    &mut p,
                    ap,
                    &mut self.w[i],
                    &self.dw[i],
                    &mut self.mw[i],
                    &mut self.vw[i],
                )?;
            }
            {
                let mut p = enc.begin_pass("ab", None);
                ad.step(
                    bk,
                    sh,
                    &mut p,
                    ap,
                    &mut self.b[i],
                    &self.db[i],
                    &mut self.mb[i],
                    &mut self.vb[i],
                )?;
            }
        }
        Ok(())
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let t_steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let epochs: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let minibatches: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let (gpu_iter_ms, backend_is_cuda) = pollster::block_on(async {
        let mut rng = Lcg::new(7);
        let (od, cd) = (OBS_DIM, CRITIC_OBS_DIM);
        let bodies_per_env = num_boxes() + 1; // dynamic boxes + floor

        // ---- backend + N batched boxes envs --------------------------------
        let bk = make_backend().await;
        println!("building {n} GPU boxes envs ({bodies_per_env} bodies each)...");
        let (bodies, colliders) = build_boxes();
        let ij = ImpulseJointSet::new();
        let mj = MultibodyJointSet::new();
        let sp = nexus3d::rbd::dynamics::GpuSimParams::default();
        let envs = vec![(&bodies, &colliders, &ij, &mj, &sp); n];
        let pipeline = GpuPhysicsPipeline::from_backend(&bk);
        let mut state = GpuPhysicsState::from_rapier(&bk, &envs);

        // ---- policy (rollout forward) + reference net ----------------------
        let mut ac = ActorCritic::new(
            &[od, 256, 256, 128, NUM_JOINTS],
            &[cd, 512, 256, 128, 1],
            1.0,
            1e-3,
            &mut rng,
        );
        let mut gpu = GpuPolicy::new(&bk, &ac, n).expect("gpu policy");

        let mut samp: Vec<Vec<Sample>> = (0..n).map(|_| Vec::with_capacity(t_steps)).collect();
        let (mut rs, mut vs, mut ds): (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<bool>>) = (
            (0..n).map(|_| vec![]).collect(),
            (0..n).map(|_| vec![]).collect(),
            (0..n).map(|_| vec![]).collect(),
        );

        let dbg = std::env::var_os("BOXES_BREAKDOWN").is_some();
        let (mut t_read, mut t_obs, mut t_fwd, mut t_samp, mut t_phys) =
            (0f64, 0f64, 0f64, 0f64, 0f64);
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
        let tg = Instant::now();
        // ===================== ROLLOUT =====================
        for _ in 0..t_steps {
            // obs <- GPU pose readback (real GPU->CPU sync, both backends)
            let t = Instant::now();
            let poses: Vec<Pose> = bk.slow_read_vec(state.poses().buffer()).await.unwrap();
            t_read += ms(t);
            let t = Instant::now();
            let gc: Vec<Vec<f32>> = (0..n)
                .map(|e| obs_from_poses(&poses, e, bodies_per_env, od))
                .collect();
            let gcc: Vec<Vec<f32>> = (0..n)
                .map(|e| obs_from_poses(&poses, e, bodies_per_env, cd))
                .collect();
            for e in 0..n {
                ac.record_obs(&gc[e], &gcc[e]);
            }
            t_obs += ms(t);
            // vortx batched policy forward
            let t = Instant::now();
            let (means, values) = gpu.forward(&bk, &ac, &gc, &gcc).await.unwrap();
            t_fwd += ms(t);
            let t = Instant::now();
            for e in 0..n {
                let mean = means[e].to_vec();
                let mut a = vec![0.0f32; NUM_JOINTS];
                for k in 0..NUM_JOINTS {
                    a[k] = mean[k] + ac.log_std[k].exp() * rng.gauss();
                }
                let lp = ac.logp(&a, &mean);
                samp[e].push(Sample {
                    obs: gc[e].clone(),
                    critic_obs: gcc[e].clone(),
                    action: a,
                    mean_old: mean,
                    logp_old: lp,
                    value_old: values[e],
                    adv: 0.0,
                    ret: 0.0,
                });
                vs[e].push(values[e]);
                // synthetic reward/done (physics is free boxes — no task)
                rs[e].push(0.0);
                ds[e].push(false);
            }
            t_samp += ms(t);
            // nexus physics step (batched, all N envs)
            let t = Instant::now();
            let _ = pipeline.step(&bk, &mut state, None).await;
            bk.synchronize().unwrap();
            t_phys += ms(t);
        }
        let t_roll = ms(tg);
        if dbg {
            eprintln!(
                "[breakdown] rollout {t_roll:.0}ms = read {t_read:.0} | obs-cpu {t_obs:.0} | fwd {t_fwd:.0} | sample-cpu {t_samp:.0} | physics {t_phys:.0}"
            );
        }
        let t_upd_start = Instant::now();

        // Physics correctness fingerprint (compare CUDA vs WebGPU): the summed
        // box height + finite count after the rollout. Matching across backends
        // proves the batched physics ran correctly (not just without crashing).
        if std::env::var_os("BOXES_FINGERPRINT").is_some() {
            let poses: Vec<Pose> = bk.slow_read_vec(state.poses().buffer()).await.unwrap();
            let mut sy = 0f64;
            let mut finite = 0usize;
            for p in &poses {
                if p.translation.y.is_finite() {
                    sy += p.translation.y as f64;
                    finite += 1;
                }
            }
            eprintln!(
                "[fingerprint] sum_y={sy:.5} finite={finite}/{}",
                poses.len()
            );
        }

        // ---- GAE + batch ---------------------------------------------------
        let mut batch: Vec<Sample> = Vec::with_capacity(n * t_steps);
        for e in 0..n {
            let lv = 0.0f32;
            let (adv, ret) = gae(&rs[e], &vs[e], &ds[e], lv, 0.99, 0.95);
            for t in 0..t_steps {
                samp[e][t].adv = adv[t];
                samp[e][t].ret = ret[t];
                batch.push(std::mem::take(&mut samp[e][t]));
            }
        }

        // ===================== GPU PPO UPDATE =====================
        // (verbatim from iter_e2e_bench's FULL GPU update)
        let total = batch.len();
        let mb = total / minibatches;
        let g = Gemm::from_backend(&bk).unwrap();
        let op = OpAssign::from_backend(&bk).unwrap();
        let act = Activation::from_backend(&bk).unwrap();
        let ad = Adam::from_backend(&bk).unwrap();
        let ppo = Ppo::from_backend(&bk).unwrap();
        let cont = Contiguous::from_backend(&bk).unwrap();
        let mut sh = TensorLayoutBuffers::new(&bk);
        let (st, rw) = (
            BufferUsages::STORAGE,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let mut a_net = GpuMlp::new(&bk, &ac.actor, mb);
        let mut c_net = GpuMlp::new(&bk, &ac.critic, mb);
        let ad_ = NUM_JOINTS;
        let on: Vec<Vec<f32>> = batch
            .iter()
            .map(|s| ac.obs_norm.normalize(&s.obs))
            .collect();
        let cn: Vec<Vec<f32>> = batch
            .iter()
            .map(|s| ac.critic_norm.normalize(&s.critic_obs))
            .collect();
        let f_obs = mk(&bk, &DMatrix::from_fn(od, total, |r, c| on[c][r]), st);
        let f_cobs = mk(&bk, &DMatrix::from_fn(cd, total, |r, c| cn[c][r]), st);
        let f_act = mk(
            &bk,
            &DMatrix::from_fn(ad_, total, |r, c| batch[c].action[r]),
            st,
        );
        let f_adv = mk(&bk, &DMatrix::from_fn(1, total, |_, c| batch[c].adv), st);
        let f_lpo = mk(
            &bk,
            &DMatrix::from_fn(1, total, |_, c| batch[c].logp_old),
            st,
        );
        let f_vo = mk(
            &bk,
            &DMatrix::from_fn(1, total, |_, c| batch[c].value_old),
            st,
        );
        let f_ret = mk(&bk, &DMatrix::from_fn(1, total, |_, c| batch[c].ret), st);
        let mut action_t = mk(&bk, &DMatrix::<f32>::zeros(ad_, mb), rw);
        let mut adv_t = mk(&bk, &DMatrix::<f32>::zeros(1, mb), rw);
        let mut lpo = mk(&bk, &DMatrix::<f32>::zeros(1, mb), rw);
        let mut vo = mk(&bk, &DMatrix::<f32>::zeros(1, mb), rw);
        let mut ret = mk(&bk, &DMatrix::<f32>::zeros(1, mb), rw);
        let mut lst = mk(&bk, &DMatrix::from_fn(ad_, 1, |r, _| ac.log_std[r]), rw);
        let o1m = mk(&bk, &DMatrix::<f32>::from_element(1, mb, 1.0), st);
        let om1 = mk(&bk, &DMatrix::<f32>::from_element(mb, 1, 1.0), st);
        let mut gls = mk(&bk, &DMatrix::<f32>::zeros(ad_, mb), rw);
        let mut dls = mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), rw);
        let (mut mls, mut vls) = (
            mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), st),
            mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), st),
        );
        let scale = 1.0 / mb as f32;
        let ap = Tensor::scalar(
            &bk,
            PpoActorParams {
                clip: 0.2,
                entropy_coef: 0.005,
                scale,
                log_sqrt_2pi: LOG_SQRT_2PI,
                action_dim: ad_ as u32,
                num_cols: mb as u32,
                pad0: 0,
                pad1: 0,
            },
            BufferUsages::UNIFORM,
        )
        .unwrap();
        let vp = Tensor::scalar(
            &bk,
            PpoValueParams {
                clip: 0.2,
                value_coef: 0.5,
                scale,
                num_cols: mb as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
                pad3: 0,
            },
            BufferUsages::UNIFORM,
        )
        .unwrap();
        let adp = Tensor::scalar(
            &bk,
            AdamParams {
                lr: 1e-3,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                bias_correction1: 0.1,
                bias_correction2: 0.001,
                pad0: 0.0,
                pad1: 0.0,
            },
            BufferUsages::UNIFORM,
        )
        .unwrap();
        let (la, lc) = (a_net.layers() - 1, c_net.layers() - 1);
        for _ in 0..epochs {
            let mut enc = bk.begin_encoding();
            for k in 0..minibatches {
                let off = (k * mb) as u32;
                let nb = mb as u32;
                {
                    let mut p = enc.begin_pass("g_obs", None);
                    cont.launch(
                        &bk,
                        &mut sh,
                        &mut p,
                        &mut a_net.a[0],
                        f_obs.columns(off, nb),
                        None,
                    )
                    .unwrap();
                }
                {
                    let mut p = enc.begin_pass("g_cobs", None);
                    cont.launch(
                        &bk,
                        &mut sh,
                        &mut p,
                        &mut c_net.a[0],
                        f_cobs.columns(off, nb),
                        None,
                    )
                    .unwrap();
                }
                {
                    let mut p = enc.begin_pass("g_act", None);
                    cont.launch(
                        &bk,
                        &mut sh,
                        &mut p,
                        &mut action_t,
                        f_act.columns(off, nb),
                        None,
                    )
                    .unwrap();
                }
                {
                    let mut p = enc.begin_pass("g_adv", None);
                    cont.launch(
                        &bk,
                        &mut sh,
                        &mut p,
                        &mut adv_t,
                        f_adv.columns(off, nb),
                        None,
                    )
                    .unwrap();
                }
                {
                    let mut p = enc.begin_pass("g_lpo", None);
                    cont.launch(&bk, &mut sh, &mut p, &mut lpo, f_lpo.columns(off, nb), None)
                        .unwrap();
                }
                {
                    let mut p = enc.begin_pass("g_vo", None);
                    cont.launch(&bk, &mut sh, &mut p, &mut vo, f_vo.columns(off, nb), None)
                        .unwrap();
                }
                {
                    let mut p = enc.begin_pass("g_ret", None);
                    cont.launch(&bk, &mut sh, &mut p, &mut ret, f_ret.columns(off, nb), None)
                        .unwrap();
                }
                a_net
                    .forward(&bk, &g, &op, &act, &mut sh, &mut enc, &o1m)
                    .unwrap();
                c_net
                    .forward(&bk, &g, &op, &act, &mut sh, &mut enc, &o1m)
                    .unwrap();
                {
                    let mut p = enc.begin_pass("ag", None);
                    ppo.actor_grad(
                        &mut p,
                        &ap,
                        &a_net.a[la + 1],
                        &action_t,
                        &lst,
                        &adv_t,
                        &lpo,
                        &mut a_net.delta[la],
                        &mut gls,
                    )
                    .unwrap();
                }
                {
                    let mut p = enc.begin_pass("vg", None);
                    ppo.value_grad(
                        &mut p,
                        &vp,
                        &c_net.a[lc + 1],
                        &vo,
                        &ret,
                        &mut c_net.delta[lc],
                    )
                    .unwrap();
                }
                a_net
                    .backward(&bk, &g, &act, &mut sh, &mut enc, &om1)
                    .unwrap();
                c_net
                    .backward(&bk, &g, &act, &mut sh, &mut enc, &om1)
                    .unwrap();
                {
                    let mut p = enc.begin_pass("dl", None);
                    g.dispatch_naive(&bk, &mut sh, &mut p, &mut dls, &gls, &om1)
                        .unwrap();
                }
                a_net.adam(&bk, &ad, &mut sh, &mut enc, &adp).unwrap();
                c_net.adam(&bk, &ad, &mut sh, &mut enc, &adp).unwrap();
                {
                    let mut p = enc.begin_pass("al", None);
                    ad.step(
                        &bk, &mut sh, &mut p, &adp, &mut lst, &dls, &mut mls, &mut vls,
                    )
                    .unwrap();
                }
            }
            bk.submit(enc).unwrap();
            bk.synchronize().unwrap();
        }
        if dbg {
            eprintln!(
                "[breakdown] update {:.0}ms ({epochs}e x {minibatches}mb)",
                ms(t_upd_start)
            );
        }
        (tg.elapsed().as_secs_f64() * 1e3, bk.is_cuda())
    });

    let ctrl = (n * t_steps) as f64;
    let gpu_eps = ctrl / (gpu_iter_ms / 1e3) / 1e3;
    let backend = if backend_is_cuda {
        "native CUDA (cuda-oxide)"
    } else {
        "WebGPU"
    };
    println!("\nfull iteration (boxes physics + vortx policy + GPU PPO update)");
    println!("  backend: {backend}");
    println!("  {n} envs, T={t_steps}, {epochs}e x {minibatches}mb");
    println!("  FULL GPU iteration : {gpu_iter_ms:9.1} ms = {gpu_eps:6.2} k env/s");
}
