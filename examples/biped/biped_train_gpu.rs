//! Train flat velocity tracking on the LeRobot bipedal — **fully GPU-resident**.
//!
//! Like `biped_train_nexus`, but the policy **forward** and the **PPO update**
//! both run on the GPU (vortx), not the CPU `ActorCritic`. The CPU net stays the
//! source of truth for checkpointing: each iter we GPU-forward the rollout
//! (`GpuPolicy`), run the GPU-resident PPO update (`GpuMlp` + ppo/adam kernels,
//! the same path benchmarked in `iter_e2e_bench`), then read the updated weights
//! back into `ac` and re-sync the rollout policy. This moves the ~95% of the
//! `biped_train_nexus` iteration that was CPU-bound (policy fwd 41% + update 54%)
//! onto the GPU.
//!
//! Run: `BIPED_CUDA=1 cargo run --release --example biped_train_gpu \
//!        --features "gpu biped_gpu cuda_backend" -- [iters] [num_envs]`

#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "gpu_policy.rs"]
mod gpu_policy;

use biped_env_nexus::{BipedNexusBatchEnv, StepOut, default_mjcf_path};
use gpu_policy::GpuPolicy;
use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend};
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
use zealot_rl::ppo::{Sample, gae};
use zealot_rl::rng::Lcg;
use zealot_rl::{ActorCritic, PpoConfig};

const T: usize = 32; // steps per env per iteration
const LOG_SQRT_2PI: f32 = 0.918_938_5;

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

// ── GPU MLP with forward/backward/adam (lifted from iter_e2e_bench) ─────────
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
            { let mut p = enc.begin_pass("z", None); g.dispatch_tiled(bk, sh, &mut p, &mut *aout, &self.w[i], ain)?; }
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
    /// Read the (possibly Adam-updated) GPU weights back into a CPU `Mlp`.
    /// `w[i]` is `[out x in]` row-major, matching `Mlp::w[i]`'s flat layout.
    async fn read_into(&self, bk: &GpuBackend, net: &mut Mlp) -> anyhow::Result<()> {
        for i in 0..self.layers() {
            net.w[i] = bk.slow_read_vec(self.w[i].buffer()).await?;
            net.b[i] = bk.slow_read_vec(self.b[i].buffer()).await?;
        }
        Ok(())
    }
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let num_envs: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let policy_path = std::env::args().nth(3).unwrap_or_else(|| "/tmp/biped_policy_gpu.safetensors".to_string());
    let epochs = 5usize;
    let minibatches = 16usize;

    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");
    println!("building {num_envs} GPU envs (32 DR templates)...");

    pollster::block_on(async {
        let mut env = BipedNexusBatchEnv::new(&xml, num_envs, 32, 0xC0FFEE).await;
        let bk = env.backend().clone();
        let (od, cd, ad_) = (env.obs_dim(), env.critic_obs_dim(), env.action_dim());
        println!("obs={od} critic_obs={cd} action={ad_}");

        let mut rng = Lcg::new(7);
        let mut ac = if !policy_path.is_empty() && std::path::Path::new(&policy_path).exists() {
            println!("resuming from {policy_path}...");
            ActorCritic::load(&policy_path).expect("load checkpoint")
        } else {
            ActorCritic::new(&[od, 256, 256, 128, ad_], &[cd, 512, 256, 128, 1], 1.0, 1e-3, &mut rng)
        };
        let cfg = PpoConfig { entropy_coef: 0.005, ..PpoConfig::default() };
        let mut gpu = GpuPolicy::new(&bk, &ac, num_envs).expect("gpu policy");

        // Persistent GPU update ops (built once).
        let g = Gemm::from_backend(&bk).unwrap();
        let op = OpAssign::from_backend(&bk).unwrap();
        let act = Activation::from_backend(&bk).unwrap();
        let adam = Adam::from_backend(&bk).unwrap();
        let ppo = Ppo::from_backend(&bk).unwrap();
        let cont = Contiguous::from_backend(&bk).unwrap();

        let total = num_envs * T;
        let mb = total / minibatches;
        assert_eq!(mb * minibatches, total, "num_envs*T must divide minibatches");

        let (mut cur, mut cur_c) = env.initial_obs().await;
        println!("\n{:>4}  {:>5}  {:>9}  {:>7}  {:>8}  {:>10}", "iter", "curr", "step_rew", "falls", "torso_z", "iter_ms");

        let warmup = (iters as f32 * 0.4).max(1.0);
        const CHECKPOINT_EVERY: usize = 50;

        for it in 0..iters {
            let t_iter = Instant::now();
            let scale = (it as f32 / warmup).min(1.0);
            env.set_command_scale(scale);

            let mut samples: Vec<Vec<Sample>> = (0..num_envs).map(|_| Vec::with_capacity(T)).collect();
            let mut rs: Vec<Vec<f32>> = (0..num_envs).map(|_| Vec::with_capacity(T)).collect();
            let mut vs: Vec<Vec<f32>> = (0..num_envs).map(|_| Vec::with_capacity(T)).collect();
            let mut ds: Vec<Vec<bool>> = (0..num_envs).map(|_| Vec::with_capacity(T)).collect();
            let (mut total_reward, mut falls) = (0.0f32, 0u32);
            let (mut t_roll, mut t_upd) = (0u128, 0u128);

            // ── Rollout: GPU forward + CPU sample + GPU physics ──
            let tr = Instant::now();
            for _ in 0..T {
                for e in 0..num_envs {
                    ac.record_obs(&cur[e], &cur_c[e]); // updates running normalizers (cheap)
                }
                // GPU batched forward (the 41% that was CPU).
                let (means, values) = gpu.forward(&bk, &ac, &cur, &cur_c).await.unwrap();
                let mut actions: Vec<[f32; NUM_JOINTS]> = Vec::with_capacity(num_envs);
                for e in 0..num_envs {
                    let mean = means[e];
                    let mut a = vec![0.0f32; NUM_JOINTS];
                    for k in 0..NUM_JOINTS {
                        a[k] = mean[k] + ac.log_std[k].exp() * rng.gauss();
                    }
                    let mean_v = mean.to_vec();
                    let lp = ac.logp(&a, &mean_v);
                    actions.push(to_action(&a));
                    samples[e].push(Sample {
                        obs: cur[e].clone(),
                        critic_obs: cur_c[e].clone(),
                        action: a,
                        mean_old: mean_v,
                        logp_old: lp,
                        value_old: values[e],
                        adv: 0.0,
                        ret: 0.0,
                    });
                    vs[e].push(values[e]);
                }
                let outs: Vec<StepOut> = env.step(&actions).await;
                for e in 0..num_envs {
                    let out = &outs[e];
                    total_reward += out.reward;
                    rs[e].push(out.reward);
                    ds[e].push(out.done);
                    if out.fell {
                        falls += 1;
                    }
                    if out.done {
                        let (o, c) = env.reset_env(e).await;
                        cur[e] = o;
                        cur_c[e] = c;
                    } else {
                        cur[e].clone_from(&out.obs);
                        cur_c[e].clone_from(&out.critic_obs);
                    }
                }
            }
            t_roll += tr.elapsed().as_nanos();

            // GAE + flat batch (CPU, cheap). Bootstrap value from the rollout net.
            let mut batch: Vec<Sample> = Vec::with_capacity(total);
            for e in 0..num_envs {
                let last_v = ac.value(&cur_c[e]);
                let (adv, ret) = gae(&rs[e], &vs[e], &ds[e], last_v, cfg.gamma, cfg.lam);
                for t in 0..T {
                    samples[e][t].adv = adv[t];
                    samples[e][t].ret = ret[t];
                    batch.push(std::mem::take(&mut samples[e][t]));
                }
            }

            // ── GPU-resident PPO update (the 54% that was CPU) ──
            let tu = Instant::now();
            {
                let (st, rw) = (BufferUsages::STORAGE, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
                let mut sh = TensorLayoutBuffers::new(&bk);
                let mut a_net = GpuMlp::new(&bk, &ac.actor, mb);
                let mut c_net = GpuMlp::new(&bk, &ac.critic, mb);
                let on: Vec<Vec<f32>> = batch.iter().map(|s| ac.obs_norm.normalize(&s.obs)).collect();
                let cn: Vec<Vec<f32>> = batch.iter().map(|s| ac.critic_norm.normalize(&s.critic_obs)).collect();
                let f_obs = mk(&bk, &DMatrix::from_fn(od, total, |r, c| on[c][r]), st);
                let f_cobs = mk(&bk, &DMatrix::from_fn(cd, total, |r, c| cn[c][r]), st);
                let f_act = mk(&bk, &DMatrix::from_fn(ad_, total, |r, c| batch[c].action[r]), st);
                let f_adv = mk(&bk, &DMatrix::from_fn(1, total, |_, c| batch[c].adv), st);
                let f_lpo = mk(&bk, &DMatrix::from_fn(1, total, |_, c| batch[c].logp_old), st);
                let f_vo = mk(&bk, &DMatrix::from_fn(1, total, |_, c| batch[c].value_old), st);
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
                let (mut mls, mut vls) = (mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), st), mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), st));
                let sc = 1.0 / mb as f32;
                let ap = Tensor::scalar(&bk, PpoActorParams { clip: 0.2, entropy_coef: 0.005, scale: sc, log_sqrt_2pi: LOG_SQRT_2PI, action_dim: ad_ as u32, num_cols: mb as u32, pad0: 0, pad1: 0 }, BufferUsages::UNIFORM).unwrap();
                let vp = Tensor::scalar(&bk, PpoValueParams { clip: 0.2, value_coef: 0.5, scale: sc, num_cols: mb as u32, pad0: 0, pad1: 0, pad2: 0, pad3: 0 }, BufferUsages::UNIFORM).unwrap();
                let adp = Tensor::scalar(&bk, AdamParams { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, bias_correction1: 0.1, bias_correction2: 0.001, pad0: 0.0, pad1: 0.0 }, BufferUsages::UNIFORM).unwrap();
                let (la, lc) = (a_net.layers() - 1, c_net.layers() - 1);
                macro_rules! run_minibatches {
                    ($enc:ident) => {{
                        for k in 0..minibatches {
                            let off = (k * mb) as u32;
                            let nb = mb as u32;
                            { let mut p = $enc.begin_pass("g_obs", None); cont.launch(&bk, &mut sh, &mut p, &mut a_net.a[0], f_obs.columns(off, nb), None).unwrap(); }
                            { let mut p = $enc.begin_pass("g_cobs", None); cont.launch(&bk, &mut sh, &mut p, &mut c_net.a[0], f_cobs.columns(off, nb), None).unwrap(); }
                            { let mut p = $enc.begin_pass("g_act", None); cont.launch(&bk, &mut sh, &mut p, &mut action_t, f_act.columns(off, nb), None).unwrap(); }
                            { let mut p = $enc.begin_pass("g_adv", None); cont.launch(&bk, &mut sh, &mut p, &mut adv_t, f_adv.columns(off, nb), None).unwrap(); }
                            { let mut p = $enc.begin_pass("g_lpo", None); cont.launch(&bk, &mut sh, &mut p, &mut lpo, f_lpo.columns(off, nb), None).unwrap(); }
                            { let mut p = $enc.begin_pass("g_vo", None); cont.launch(&bk, &mut sh, &mut p, &mut vo, f_vo.columns(off, nb), None).unwrap(); }
                            { let mut p = $enc.begin_pass("g_ret", None); cont.launch(&bk, &mut sh, &mut p, &mut ret, f_ret.columns(off, nb), None).unwrap(); }
                            a_net.forward(&bk, &g, &op, &act, &mut sh, &mut $enc, &o1m).unwrap();
                            c_net.forward(&bk, &g, &op, &act, &mut sh, &mut $enc, &o1m).unwrap();
                            { let mut p = $enc.begin_pass("ag", None); ppo.actor_grad(&mut p, &ap, &a_net.a[la + 1], &action_t, &lst, &adv_t, &lpo, &mut a_net.delta[la], &mut gls).unwrap(); }
                            { let mut p = $enc.begin_pass("vg", None); ppo.value_grad(&mut p, &vp, &c_net.a[lc + 1], &vo, &ret, &mut c_net.delta[lc]).unwrap(); }
                            a_net.backward(&bk, &g, &act, &mut sh, &mut $enc, &om1).unwrap();
                            c_net.backward(&bk, &g, &act, &mut sh, &mut $enc, &om1).unwrap();
                            { let mut p = $enc.begin_pass("dl", None); g.dispatch_naive(&bk, &mut sh, &mut p, &mut dls, &gls, &om1).unwrap(); }
                            a_net.adam(&bk, &adam, &mut sh, &mut $enc, &adp).unwrap();
                            c_net.adam(&bk, &adam, &mut sh, &mut $enc, &adp).unwrap();
                            { let mut p = $enc.begin_pass("al", None); adam.step(&bk, &mut sh, &mut p, &adp, &mut lst, &dls, &mut mls, &mut vls).unwrap(); }
                        }
                    }};
                }
                // Eager epochs (per-iter buffers => no graph replay).
                for _ in 0..epochs {
                    let mut enc = bk.begin_encoding();
                    run_minibatches!(enc);
                    bk.submit(enc).unwrap();
                    bk.synchronize().unwrap();
                }
                // Close the loop: GPU weights -> CPU ac -> re-sync rollout policy.
                a_net.read_into(&bk, &mut ac.actor).await.unwrap();
                c_net.read_into(&bk, &mut ac.critic).await.unwrap();
                ac.log_std = bk.slow_read_vec(lst.buffer()).await.unwrap();
            }
            gpu.sync_weights(&bk, &ac);
            t_upd += tu.elapsed().as_nanos();

            if it % 5 == 0 || it == iters - 1 {
                let zs = env.torso_heights().await;
                let torso = zs.iter().sum::<f32>() / num_envs as f32;
                println!(
                    "{:>4}  {:>5.2}  {:>9.4}  {:>7}  {:>8.3}  {:>10.0}  [roll {:.0} upd {:.0}]",
                    it, scale, total_reward / (num_envs * T) as f32, falls, torso,
                    t_iter.elapsed().as_secs_f64() * 1e3,
                    t_roll as f64 / 1e6, t_upd as f64 / 1e6,
                );
            }
            if !policy_path.is_empty() && it > 0 && (it % CHECKPOINT_EVERY == 0 || it == iters - 1) {
                ac.save(&policy_path).unwrap_or_else(|e| eprintln!("checkpoint failed: {e}"));
            }
        }
        if !policy_path.is_empty() {
            ac.save(&policy_path).expect("save policy");
            println!("saved final policy → {policy_path}");
        }
    });
}
