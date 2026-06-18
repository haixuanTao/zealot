//! GPU-resident PPO trainer for the nexus biped — the real-training version of
//! the machinery `iter_e2e_bench` benchmarks one-shot. Both the rollout policy
//! forward (`GpuPolicy`) AND the PPO update (GpuMlp forward/backward/Adam +
//! vortx `Ppo` actor/value grads) run on the GPU; only action sampling, GAE, and
//! reward/obs (host) stay on the CPU. ~5x faster/iter than `biped_train_nexus`
//! (which runs policy + PPO on CPU).
//!
//! Net + hyperparameters match WBC-AGILE's T1 velocity policy (actor
//! [obs,256,256,128,12], critic [cobs,512,256,128,1], init_noise_std=1.0,
//! entropy 0.005, clip 0.2, velocity curriculum 0→1 over the first 40% of iters).
//!
//! Unlike the throughput bench, this is a correct multi-iteration optimizer:
//! the GPU nets + Adam moments PERSIST across iterations, Adam bias-correction
//! advances with a global step, and the updated weights are synced
//! GPU→CPU(ActorCritic)→GpuPolicy each iteration for the next rollout.
//!
//! Run:
//!   export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin
//!   export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$HOME/nexus_ptx/vortx_shaders.cubin
//!   BIPED_CUDA=1 cargo run --release --example biped_train_gpu \
//!       --features "gpu biped_gpu cuda_backend" -- [iters] [num_envs] [ckpt]

#[path = "biped_env.rs"]
mod biped_env;
#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "gpu_policy.rs"]
mod gpu_policy;

use biped_env_nexus::{default_mjcf_path, BipedNexusBatchEnv, REWARD_COMP_NAMES};
use gpu_policy::GpuPolicy;
use khal::backend::{Backend, Encoder, GpuBackend};
use khal::BufferUsages;
use khal::Shader;
use nalgebra::DMatrix;
use std::time::Instant;
use vortx::linalg::{
    Activation, Adam, AdamParams, Contiguous, Gemm, OpAssign, OpAssignVariant, Ppo, PpoActorParams,
    PpoValueParams,
};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use zealot_env::robots::lerobot_bipedal::NUM_JOINTS;
use zealot_rl::net::Mlp;
use zealot_rl::ppo::{gae, Sample};
use zealot_rl::rng::Lcg;
use zealot_rl::ActorCritic;

const LOG_SQRT_2PI: f32 = 0.918_938_5;
const T: usize = 24; // rollout horizon
const EPOCHS: usize = 5;
const MINIBATCHES: usize = 4;
const LR: f32 = 1e-3;
const CLIP: f32 = 0.2;
const ENTROPY: f32 = 0.01; // WBC lerobot entropy_coef (was 0.005)
const VALUE_COEF: f32 = 1.0; // WBC-AGILE / rsl_rl value_coef
const GAMMA: f32 = 0.99;
const LAM: f32 = 0.95;
// Adaptive-KL LR schedule (rsl_rl / WBC-AGILE): lr ÷1.5 when KL > 2·desired,
// ×1.5 when KL < desired/2, clamped to [LR_MIN, LR_MAX].
const DESIRED_KL: f32 = 0.01;
const LR_MIN: f32 = 1e-5;
const LR_MAX: f32 = 1e-2;

fn mk(b: &GpuBackend, m: &DMatrix<f32>, u: BufferUsages) -> Tensor<f32> {
    Tensor::matrix_from_na(b, m, u).unwrap()
}
fn wmat(w: &[f32], out: usize, inp: usize) -> DMatrix<f32> {
    DMatrix::from_fn(out, inp, |r, c| w[r * inp + c])
}
fn to_action(v: &[f32]) -> [f32; NUM_JOINTS] {
    let mut a = [0.0; NUM_JOINTS];
    a.copy_from_slice(&v[..NUM_JOINTS]);
    a
}

/// GPU MLP with persistent weights + Adam moments (copied from iter_e2e_bench,
/// plus `read_into` to write the trained weights back to a CPU `Mlp`).
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
    fn new(bk: &GpuBackend, net: &Mlp, m: usize) -> Self {
        let d = net.dims.clone();
        let (st, rw) = (BufferUsages::STORAGE, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let l = net.w.len();
        let z = |r: usize, c: usize| DMatrix::<f32>::zeros(r, c);
        GpuMlp {
            w: (0..l).map(|i| mk(bk, &wmat(&net.w[i], d[i + 1], d[i]), rw)).collect(),
            b: (0..l).map(|i| mk(bk, &DMatrix::from_fn(d[i + 1], 1, |r, _| net.b[i][r]), rw)).collect(),
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
    fn forward(&mut self, bk: &GpuBackend, g: &Gemm, op: &OpAssign, act: &Activation, sh: &mut TensorLayoutBuffers, enc: &mut <GpuBackend as Backend>::Encoder, o1m: &Tensor<f32>) -> anyhow::Result<()> {
        let l = self.layers();
        for i in 0..l {
            let (lf, rt) = self.a.split_at_mut(i + 1);
            let (ain, aout) = (&lf[i], &mut rt[0]);
            {
                let mut p = enc.begin_pass("z", None);
                g.dispatch_tiled(bk, sh, &mut p, &mut *aout, &self.w[i], ain)?;
            }
            { let mut p = enc.begin_pass("bb", None); g.dispatch_naive(bk, sh, &mut p, &mut self.bb[i], &self.b[i], o1m)?; }
            { let mut p = enc.begin_pass("bias", None); op.launch(bk, sh, &mut p, OpAssignVariant::Add, &mut *aout, &self.bb[i])?; }
            if i < l - 1 { let mut p = enc.begin_pass("elu", None); act.elu(bk, sh, &mut p, &mut *aout)?; }
        }
        Ok(())
    }
    fn backward(&mut self, bk: &GpuBackend, g: &Gemm, act: &Activation, sh: &mut TensorLayoutBuffers, enc: &mut <GpuBackend as Backend>::Encoder, om1: &Tensor<f32>) -> anyhow::Result<()> {
        for i in (0..self.layers()).rev() {
            { let mut p = enc.begin_pass("dw", None); g.dispatch_tiled(bk, sh, &mut p, &mut self.dw[i], &self.delta[i], self.a[i].transpose_last_dims())?; }
            { let mut p = enc.begin_pass("db", None); g.dispatch_naive(bk, sh, &mut p, &mut self.db[i], &self.delta[i], om1)?; }
            if i > 0 {
                { let (lf, rt) = self.delta.split_at_mut(i); let dp = &mut lf[i - 1]; let dc = &rt[0]; let mut p = enc.begin_pass("da", None); g.dispatch_tiled(bk, sh, &mut p, dp, self.w[i].transpose_last_dims(), dc)?; }
                { let mut p = enc.begin_pass("eb", None); act.elu_backward(bk, sh, &mut p, &mut self.delta[i - 1], &self.a[i])?; }
            }
        }
        Ok(())
    }
    fn adam(&mut self, bk: &GpuBackend, ad: &Adam, sh: &mut TensorLayoutBuffers, enc: &mut <GpuBackend as Backend>::Encoder, ap: &Tensor<AdamParams>) -> anyhow::Result<()> {
        for i in 0..self.layers() {
            { let mut p = enc.begin_pass("aw", None); ad.step(bk, sh, &mut p, ap, &mut self.w[i], &self.dw[i], &mut self.mw[i], &mut self.vw[i])?; }
            { let mut p = enc.begin_pass("ab", None); ad.step(bk, sh, &mut p, ap, &mut self.b[i], &self.db[i], &mut self.mb[i], &mut self.vb[i])?; }
        }
        Ok(())
    }
    /// Write the trained GPU weights back into a CPU `Mlp` (`w[l]` is row-major
    /// `[out x in]`, `b[l]` is `[out x 1]`).
    async fn read_into(&self, bk: &GpuBackend, net: &mut Mlp) {
        for l in 0..self.w.len() {
            let (out, inp) = (self.dims[l + 1], self.dims[l]);
            let w = bk.slow_read_vec(self.w[l].buffer()).await.unwrap();
            net.w[l].copy_from_slice(&w[..out * inp]);
            let b = bk.slow_read_vec(self.b[l].buffer()).await.unwrap();
            net.b[l].copy_from_slice(&b[..out]);
        }
    }
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let ckpt = std::env::args().nth(3).unwrap_or_else(|| "/tmp/biped_policy_gpu.safetensors".to_string());
    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");

    pollster::block_on(async {
        println!("building {n} GPU nexus envs...");
        let mut env = BipedNexusBatchEnv::new(&xml, n, 32, 0xC0FFEE).await;
        let (od, cd) = (env.obs_dim(), env.critic_obs_dim());
        let mut rng = Lcg::new(7);
        let mut ac = if !ckpt.is_empty() && std::path::Path::new(&ckpt).exists() {
            println!("resuming from {ckpt}...");
            ActorCritic::load(&ckpt).expect("load checkpoint")
        } else {
            ActorCritic::new(&[od, 256, 256, 128, NUM_JOINTS], &[cd, 512, 256, 128, 1], 1.0, 1e-3, &mut rng)
        };
        let bk = env.backend().clone();
        let mut gpu = GpuPolicy::new(&bk, &ac, n).expect("gpu policy");

        // Persistent GPU update state (weights + Adam moments survive all iters).
        let total = n * T;
        let mb = total / MINIBATCHES;
        let g = Gemm::from_backend(&bk).unwrap();
        let op = OpAssign::from_backend(&bk).unwrap();
        let act = Activation::from_backend(&bk).unwrap();
        let ad = Adam::from_backend(&bk).unwrap();
        let ppo = Ppo::from_backend(&bk).unwrap();
        let cont = Contiguous::from_backend(&bk).unwrap();
        let mut sh = TensorLayoutBuffers::new(&bk);
        let (st, rw) = (BufferUsages::STORAGE, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let mut a_net = GpuMlp::new(&bk, &ac.actor, mb);
        let mut c_net = GpuMlp::new(&bk, &ac.critic, mb);
        let ad_ = NUM_JOINTS;
        let mut lst = mk(&bk, &DMatrix::from_fn(ad_, 1, |r, _| ac.log_std[r]), rw);
        let (mut mls, mut vls) = (mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), st), mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), st));
        // Reused mb-sized scratch.
        let mut action_t = mk(&bk, &DMatrix::<f32>::zeros(ad_, mb), rw);
        let mut adv_t = mk(&bk, &DMatrix::<f32>::zeros(1, mb), rw);
        let mut lpo = mk(&bk, &DMatrix::<f32>::zeros(1, mb), rw);
        let mut vo = mk(&bk, &DMatrix::<f32>::zeros(1, mb), rw);
        let mut ret = mk(&bk, &DMatrix::<f32>::zeros(1, mb), rw);
        let o1m = mk(&bk, &DMatrix::<f32>::from_element(1, mb, 1.0), st);
        let om1 = mk(&bk, &DMatrix::<f32>::from_element(mb, 1, 1.0), st);
        let mut gls = mk(&bk, &DMatrix::<f32>::zeros(ad_, mb), rw);
        let mut dls = mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), rw);
        let scale_mb = 1.0 / mb as f32;
        let (la, lc) = (a_net.layers() - 1, c_net.layers() - 1);
        let mut gstep: u64 = 0;
        let mut lr = LR; // adaptive-KL LR, persists across iterations

        let (mut gc, mut gcc) = env.initial_obs().await;
        // Velocity-command curriculum: STAND-BEFORE-WALK. Hold the command at 0
        // (cscale=0 → all commands standing) for the first `stand_frac` of training
        // so the policy first learns to BALANCE, then ramp the command 0→1 over
        // `stand_frac`→`ramp_end`, full command after. v10 (and earlier) ramped
        // the command from iter 0, so it was asked to move before it could stand —
        // it never escaped the falling/ignore-command regime. Now that the motor
        // fix makes the zero pose stable, a dedicated standing phase is learnable.
        let stand_frac: f32 = std::env::var("BIPED_STAND_FRAC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.3);
        // Fraction of training by which the velocity command reaches full scale
        // (command ramps 0→1 over [stand_frac, ramp_end]). BIPED_RAMP_END lets a
        // WARM-STARTED run (resuming a competent standing policy with
        // BIPED_STAND_FRAC=0) reach walking speed quickly instead of over the
        // default 70% of training.
        let ramp_end: f32 = std::env::var("BIPED_RAMP_END")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.7);
        println!("\n{:>4}  {:>5}  {:>9}  {:>7}  {:>8}  {:>9}  {:>7}  {:>6}", "iter", "curr", "step_rew", "falls", "torso_z", "lr", "kl", "sec");

        // Torque-penalty curriculum target (full WBC weight = 1.0). Ramped 0→max
        // over iters 40%→90% so the effort penalty engages only AFTER the policy
        // can stand — at full strength from scratch it fights learning to stand.
        let torque_max = std::env::var("BIPED_TORQUE_MAX")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0);
        // Cap the command scale (BIPED_MAX_CSCALE, default 1.0). The sampler's full
        // range is ±0.5 m/s; capping at e.g. 0.4 → max ±0.2 m/s = a SLOW walk, so
        // the policy learns a deliberate low-cadence gait (step → stabilize → step)
        // instead of fast continuous tiny stepping. Slow + quasi-static also
        // transfers far better (no reliance on dynamic contact timing).
        let max_cscale: f32 = std::env::var("BIPED_MAX_CSCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        for it in 0..iters {
            let t_iter = Instant::now();
            let frac = it as f32 / iters as f32;
            let cscale = if frac < stand_frac {
                0.0
            } else {
                ((frac - stand_frac) / (ramp_end - stand_frac)).clamp(0.0, 1.0)
            } * max_cscale;
            env.set_command_scale(cscale);
            let tscale = ((it as f32 / iters as f32 - 0.4) / 0.5).clamp(0.0, 1.0) * torque_max;
            env.set_torque_scale(tscale);

            // ---------------- ROLLOUT (GPU policy forward, host sample) ----------------
            let mut samp: Vec<Vec<Sample>> = (0..n).map(|_| Vec::with_capacity(T)).collect();
            let (mut rs, mut vs, mut ds): (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<bool>>) =
                ((0..n).map(|_| vec![]).collect(), (0..n).map(|_| vec![]).collect(), (0..n).map(|_| vec![]).collect());
            let (mut total_reward, mut falls) = (0.0f32, 0u32);
            let t_roll = Instant::now();
            let mut reset_dur = std::time::Duration::ZERO;
            for _ in 0..T {
                for e in 0..n {
                    ac.record_obs(&gc[e], &gcc[e]);
                }
                let (means, values) = gpu.forward(&bk, &ac, &gc, &gcc).await.unwrap();
                let mut acts = Vec::with_capacity(n);
                for e in 0..n {
                    let mean = means[e];
                    let mut a = vec![0.0f32; NUM_JOINTS];
                    for k in 0..NUM_JOINTS {
                        a[k] = mean[k] + ac.log_std[k].exp() * rng.gauss();
                    }
                    let lp = ac.logp(&a, &mean[..]);
                    acts.push(to_action(&a));
                    samp[e].push(Sample {
                        obs: gc[e].clone(),
                        critic_obs: gcc[e].clone(),
                        action: a,
                        mean_old: mean.to_vec(),
                        logp_old: lp,
                        value_old: values[e],
                        adv: 0.0,
                        ret: 0.0,
                    });
                    vs[e].push(values[e]);
                }
                let outs = env.step(&acts).await;
                for e in 0..n {
                    total_reward += outs[e].reward;
                    if outs[e].fell {
                        falls += 1;
                    }
                    rs[e].push(outs[e].reward);
                    ds[e].push(outs[e].done);
                    if outs[e].done {
                        let tr = Instant::now();
                        let (o, c) = env.reset_env(e).await;
                        reset_dur += tr.elapsed();
                        gc[e] = o;
                        gcc[e] = c;
                    } else {
                        gc[e].clone_from(&outs[e].obs);
                        gcc[e].clone_from(&outs[e].critic_obs);
                    }
                }
            }

            let roll_s = t_roll.elapsed().as_secs_f64();
            // ---------------- GAE + batch ----------------
            let t_gae = Instant::now();
            let mut batch: Vec<Sample> = Vec::with_capacity(total);
            for e in 0..n {
                let lv = ac.value(&gcc[e]);
                let (adv, retn) = gae(&rs[e], &vs[e], &ds[e], lv, GAMMA, LAM);
                for t in 0..T {
                    samp[e][t].adv = adv[t];
                    samp[e][t].ret = retn[t];
                    batch.push(std::mem::take(&mut samp[e][t]));
                }
            }
            // Normalize advantages across the batch (mean 0, std 1) — this is what
            // CPU `ActorCritic::update` does; the GPU `Ppo::actor_grad` consumes raw
            // `adv`, so without this the PPO gradients are mis-scaled and the policy
            // plateaus instead of learning.
            let amean: f32 = batch.iter().map(|s| s.adv).sum::<f32>() / total as f32;
            let avar: f32 = batch.iter().map(|s| (s.adv - amean).powi(2)).sum::<f32>() / total as f32;
            let asd = avar.sqrt().max(1e-6);
            for s in batch.iter_mut() {
                s.adv = (s.adv - amean) / asd;
            }

            let gae_s = t_gae.elapsed().as_secs_f64();
            // ---------------- GPU PPO UPDATE (persistent nets, advancing Adam) -------
            let t_upd = Instant::now();
            let on: Vec<Vec<f32>> = batch.iter().map(|s| ac.obs_norm.normalize(&s.obs)).collect();
            let cn: Vec<Vec<f32>> = batch.iter().map(|s| ac.critic_norm.normalize(&s.critic_obs)).collect();
            let f_obs = mk(&bk, &DMatrix::from_fn(od, total, |r, c| on[c][r]), st);
            let f_cobs = mk(&bk, &DMatrix::from_fn(cd, total, |r, c| cn[c][r]), st);
            let f_act = mk(&bk, &DMatrix::from_fn(ad_, total, |r, c| batch[c].action[r]), st);
            let f_adv = mk(&bk, &DMatrix::from_fn(1, total, |_, c| batch[c].adv), st);
            let f_lpo = mk(&bk, &DMatrix::from_fn(1, total, |_, c| batch[c].logp_old), st);
            let f_vo = mk(&bk, &DMatrix::from_fn(1, total, |_, c| batch[c].value_old), st);
            let f_ret = mk(&bk, &DMatrix::from_fn(1, total, |_, c| batch[c].ret), st);
            let ap = Tensor::scalar(&bk, PpoActorParams { clip: CLIP, entropy_coef: ENTROPY, scale: scale_mb, log_sqrt_2pi: LOG_SQRT_2PI, action_dim: ad_ as u32, num_cols: mb as u32, pad0: 0, pad1: 0 }, BufferUsages::UNIFORM).unwrap();
            let vp = Tensor::scalar(&bk, PpoValueParams { clip: CLIP, value_coef: VALUE_COEF, scale: scale_mb, num_cols: mb as u32, pad0: 0, pad1: 0, pad2: 0, pad3: 0 }, BufferUsages::UNIFORM).unwrap();

            // Old-policy means for the LAST minibatch — drives the per-epoch KL
            // for the adaptive-KL LR schedule (mirrors CPU `minibatch_step`'s
            // `self.kl`, here at per-epoch rather than per-minibatch granularity).
            let last_off = (MINIBATCHES - 1) * mb;
            let mean_old_last: Vec<Vec<f32>> =
                (0..mb).map(|c| batch[last_off + c].mean_old.clone()).collect();
            let mut last_kl = 0.0f32;
            for _epoch in 0..EPOCHS {
                gstep += MINIBATCHES as u64;
                let bc1 = 1.0 - 0.9f32.powi(gstep.min(1 << 30) as i32);
                let bc2 = 1.0 - 0.999f32.powi(gstep.min(1 << 30) as i32);
                let adp = Tensor::scalar(&bk, AdamParams { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, bias_correction1: bc1, bias_correction2: bc2, pad0: 0.0, pad1: 0.0 }, BufferUsages::UNIFORM).unwrap();
                let mut enc = bk.begin_encoding();
                for k in 0..MINIBATCHES {
                    let off = (k * mb) as u32;
                    let nb = mb as u32;
                    { let mut p = enc.begin_pass("g_obs", None); cont.launch(&bk, &mut sh, &mut p, &mut a_net.a[0], f_obs.columns(off, nb), None).unwrap(); }
                    { let mut p = enc.begin_pass("g_cobs", None); cont.launch(&bk, &mut sh, &mut p, &mut c_net.a[0], f_cobs.columns(off, nb), None).unwrap(); }
                    { let mut p = enc.begin_pass("g_act", None); cont.launch(&bk, &mut sh, &mut p, &mut action_t, f_act.columns(off, nb), None).unwrap(); }
                    { let mut p = enc.begin_pass("g_adv", None); cont.launch(&bk, &mut sh, &mut p, &mut adv_t, f_adv.columns(off, nb), None).unwrap(); }
                    { let mut p = enc.begin_pass("g_lpo", None); cont.launch(&bk, &mut sh, &mut p, &mut lpo, f_lpo.columns(off, nb), None).unwrap(); }
                    { let mut p = enc.begin_pass("g_vo", None); cont.launch(&bk, &mut sh, &mut p, &mut vo, f_vo.columns(off, nb), None).unwrap(); }
                    { let mut p = enc.begin_pass("g_ret", None); cont.launch(&bk, &mut sh, &mut p, &mut ret, f_ret.columns(off, nb), None).unwrap(); }
                    a_net.forward(&bk, &g, &op, &act, &mut sh, &mut enc, &o1m).unwrap();
                    c_net.forward(&bk, &g, &op, &act, &mut sh, &mut enc, &o1m).unwrap();
                    { let mut p = enc.begin_pass("ag", None); ppo.actor_grad(&mut p, &ap, &a_net.a[la + 1], &action_t, &lst, &adv_t, &lpo, &mut a_net.delta[la], &mut gls).unwrap(); }
                    { let mut p = enc.begin_pass("vg", None); ppo.value_grad(&mut p, &vp, &c_net.a[lc + 1], &vo, &ret, &mut c_net.delta[lc]).unwrap(); }
                    a_net.backward(&bk, &g, &act, &mut sh, &mut enc, &om1).unwrap();
                    c_net.backward(&bk, &g, &act, &mut sh, &mut enc, &om1).unwrap();
                    { let mut p = enc.begin_pass("dl", None); g.dispatch_naive(&bk, &mut sh, &mut p, &mut dls, &gls, &om1).unwrap(); }
                    a_net.adam(&bk, &ad, &mut sh, &mut enc, &adp).unwrap();
                    c_net.adam(&bk, &ad, &mut sh, &mut enc, &adp).unwrap();
                    { let mut p = enc.begin_pass("al", None); ad.step(&bk, &mut sh, &mut p, &adp, &mut lst, &dls, &mut mls, &mut vls).unwrap(); }
                }
                bk.submit(enc).unwrap();
                bk.synchronize().unwrap();

                // Per-epoch KL (last minibatch) → adaptive-KL LR for the next epoch.
                let mn = bk.slow_read_vec(a_net.a[la + 1].buffer()).await.unwrap(); // [ad x mb]
                let ls = bk.slow_read_vec(lst.buffer()).await.unwrap(); // [ad]
                let mut kl = 0.0f32;
                for c in 0..mb {
                    for k in 0..ad_ {
                        let inv = (-ls[k]).exp();
                        let d = (mn[k * mb + c] - mean_old_last[c][k]) * inv;
                        kl += 0.5 * d * d;
                    }
                }
                kl /= mb as f32;
                last_kl = kl;
                if kl > DESIRED_KL * 2.0 {
                    lr = (lr / 1.5).max(LR_MIN);
                } else if kl > 0.0 && kl < DESIRED_KL / 2.0 {
                    lr = (lr * 1.5).min(LR_MAX);
                }
            }

            // ---------------- SYNC trained weights GPU → ac → GpuPolicy ----------------
            a_net.read_into(&bk, &mut ac.actor).await;
            c_net.read_into(&bk, &mut ac.critic).await;
            let ls = bk.slow_read_vec(lst.buffer()).await.unwrap();
            ac.log_std.copy_from_slice(&ls[..ad_]);
            // Floor the exploration std. The GPU Adam path that trains `log_std`
            // (the "al" pass) has NO clamp — left alone it collapses to ~ln(0.06),
            // exploration dies, and the policy locks into the limit-riding optimum
            // (the dead clamp in ppo.rs::step_log_std never runs on this path).
            // Re-floor to [ln 0.2, ln 1.0] each iter and push the clamped values
            // back into the GPU param buffer so the next update continues from them.
            const LOG_STD_MIN: f32 = -1.6; // std 0.20
            const LOG_STD_MAX: f32 = 0.0; // std 1.0
            let mut clamped = false;
            for v in ac.log_std.iter_mut() {
                let c = v.clamp(LOG_STD_MIN, LOG_STD_MAX);
                if c != *v {
                    *v = c;
                    clamped = true;
                }
            }
            if clamped {
                lst = mk(&bk, &DMatrix::from_fn(ad_, 1, |r, _| ac.log_std[r]), rw);
            }
            gpu.sync_weights(&bk, &ac);
            let upd_s = t_upd.elapsed().as_secs_f64();

            if it % 10 == 0 || it == iters - 1 {
                let zs = env.torso_heights().await;
                let torso = zs.iter().sum::<f32>() / n as f32;
                println!(
                    "{:>4}  {:>5.2}  {:>9.4}  {:>7}  {:>8.3}  {:>9.2e}  {:>7.4}  {:>6.1}",
                    it, cscale, total_reward / total as f32, falls, torso, lr, last_kl, t_iter.elapsed().as_secs_f64()
                );
                // [prof] coarse iteration split + rollout per-phase ms/step
                // (env.take_step_timings drains the StepTimings accumulator).
                let st = env.take_step_timings();
                let ns2ms = |x: u64| (x as f64) / (st.steps.max(1) as f64) / 1e6;
                println!(
                    "[prof] roll={:.2}s (reset={:.2}s) gae={:.2}s upd={:.2}s | per-step ms: pipe={:.1} gpuwait={:.1} readback={:.1} reward={:.1} stage={:.1} flush={:.1} commit={:.1}",
                    roll_s, reset_dur.as_secs_f64(), gae_s, upd_s,
                    ns2ms(st.pipeline_step_ns), ns2ms(st.gpu_wait_ns), ns2ms(st.readback_ns),
                    ns2ms(st.par_compute_ns), ns2ms(st.stage_motors_ns), ns2ms(st.flush_static_ns), ns2ms(st.serial_commit_ns),
                );
                // Structured per-component reward + termination-cause line for the
                // W&B sidecar (`wandb_logger.py` parses the `[rb]` prefix). Mean of
                // each reward term over the window since the last drain, plus
                // episode-termination counts split by cause.
                if let Some(rl) = env.take_reward_log() {
                    let mut s = format!("[rb] iter {it}");
                    for (name, v) in REWARD_COMP_NAMES.iter().zip(rl.comps.iter()) {
                        s.push_str(&format!(" {name}={v:.5}"));
                    }
                    s.push_str(&format!(
                        " term_illegal={} term_fell={} term_timeout={} samples={}",
                        rl.illegal, rl.fell, rl.timeout, rl.samples
                    ));
                    println!("{s}");
                }
            }
            if !ckpt.is_empty() && (it % 50 == 0 || it == iters - 1) {
                let _ = ac.save(&ckpt);
            }
        }
        if !ckpt.is_empty() {
            ac.save(&ckpt).expect("save");
            println!("saved → {ckpt}");
        }
    });
}
