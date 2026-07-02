//! Full training-ITERATION throughput: FULL CPU vs FULL GPU, end to end.
//!   full CPU = rapier rollout (CPU MLP + rayon physics) + CPU PPO update
//!   full GPU = nexus rollout (vortx policy) + GPU PPO update
//! One iteration = T-step rollout + (epochs x minibatches) PPO update.
//! Reports env-control-steps/second = N*T / iteration_time, the same unit as the
//! rollout table and Isaac's total `Computation` (collection + learning).
//!
//! Run: `cargo run --release --example iter_e2e_bench --features "gpu biped_gpu" -- [num_envs] [T] [epochs] [minibatches]`

#[path = "biped_env.rs"]
mod biped_env;
#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "gpu_policy.rs"]
mod gpu_policy;

use biped_env::BipedEnv;
use biped_env_nexus::{BipedNexusBatchEnv, StepOut, default_mjcf_path};
use gpu_policy::GpuPolicy;
use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend};
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::time::Instant;
use vortx::linalg::{
    Activation, Adam, AdamParams, Contiguous, Gemm, OpAssign, OpAssignVariant, Ppo,
    PpoActorParams, PpoValueParams,
};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use zealot_env::robots::lerobot_bipedal::NUM_JOINTS;
use zealot_rl::net::Mlp;
use zealot_rl::ppo::{Sample, gae};
use zealot_rl::rng::Lcg;
use zealot_rl::{ActorCritic, PpoConfig};

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

struct GpuMlp {
    dims: Vec<usize>,
    batch: usize,
    w: Vec<Tensor<f32>>, b: Vec<Tensor<f32>>,
    mw: Vec<Tensor<f32>>, vw: Vec<Tensor<f32>>, mb: Vec<Tensor<f32>>, vb: Vec<Tensor<f32>>,
    a: Vec<Tensor<f32>>, bb: Vec<Tensor<f32>>, delta: Vec<Tensor<f32>>, dw: Vec<Tensor<f32>>, db: Vec<Tensor<f32>>,
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
    fn layers(&self) -> usize { self.w.len() }
    #[allow(clippy::too_many_arguments)]
    fn forward(&mut self, bk: &GpuBackend, g: &Gemm, op: &OpAssign, act: &Activation, sh: &mut TensorLayoutBuffers, enc: &mut <GpuBackend as Backend>::Encoder, o1m: &Tensor<f32>) -> anyhow::Result<()> {
        let l = self.layers();
        for i in 0..l {
            let (lf, rt) = self.a.split_at_mut(i + 1);
            let (ain, aout) = (&lf[i], &mut rt[0]);
            {
                let mut p = enc.begin_pass("z", None);
                // vec4 global loads only for the qualifying contiguous hidden GEMMs
                // (out%64, in%16, batch%64); input/output layers fall back to scalar.
                if self.dims[i + 1] % 64 == 0 && self.dims[i] % 16 == 0 && self.batch % 64 == 0 {
                    g.dispatch_tiled(bk, sh, &mut p, &mut *aout, &self.w[i], ain)?;
                } else {
                    g.dispatch_tiled(bk, sh, &mut p, &mut *aout, &self.w[i], ain)?;
                }
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
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let t_steps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let epochs: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let minibatches: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    let xml = std::fs::read_to_string(default_mjcf_path()).expect("mjcf");
    let cfg = PpoConfig { clip: 0.2, entropy_coef: 0.005, value_coef: 0.5, epochs, minibatches, adaptive_lr: false, max_grad_norm: 1e9, ..PpoConfig::default() };

    // ===================== FULL CPU ITERATION =====================
    println!("building {n} CPU rapier envs...");
    let mut rng = Lcg::new(7);
    let mut cpu_envs: Vec<BipedEnv> = (0..n).map(|e| BipedEnv::new(&xml, e as u64)).collect();
    let (od, cd) = (cpu_envs[0].obs_dim(), cpu_envs[0].critic_obs_dim());
    let mut ac = ActorCritic::new(&[od, 256, 256, 128, NUM_JOINTS], &[cd, 512, 256, 128, 1], 1.0, 1e-3, &mut rng);
    let (mut cur, mut cur_c): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for env in cpu_envs.iter_mut() { let (o, c) = env.reset_full(); cur.push(o); cur_c.push(c); }
    let mut samp: Vec<Vec<Sample>> = (0..n).map(|_| Vec::with_capacity(t_steps)).collect();
    let (mut rs, mut vs, mut ds): (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<bool>>) =
        ((0..n).map(|_| vec![]).collect(), (0..n).map(|_| vec![]).collect(), (0..n).map(|_| vec![]).collect());
    let tc = Instant::now();
    for _ in 0..t_steps {
        let mut acts = Vec::with_capacity(n);
        for e in 0..n {
            ac.record_obs(&cur[e], &cur_c[e]);
            let (a, lp, m) = ac.sample(&cur[e], &mut rng);
            let v = ac.value(&cur_c[e]);
            acts.push(to_action(&a));
            samp[e].push(Sample { obs: cur[e].clone(), critic_obs: cur_c[e].clone(), action: a, mean_old: m, logp_old: lp, value_old: v, adv: 0.0, ret: 0.0 });
            vs[e].push(v);
        }
        let outs: Vec<_> = cpu_envs.par_iter_mut().enumerate().map(|(e, env)| env.step(&acts[e])).collect();
        for e in 0..n { rs[e].push(outs[e].reward); ds[e].push(outs[e].done); cur[e].clone_from(&outs[e].obs); cur_c[e].clone_from(&outs[e].critic_obs); }
    }
    let mut batch: Vec<Sample> = Vec::with_capacity(n * t_steps);
    for e in 0..n {
        let lv = ac.value(&cur_c[e]);
        let (adv, ret) = gae(&rs[e], &vs[e], &ds[e], lv, 0.99, 0.95);
        for t in 0..t_steps { samp[e][t].adv = adv[t]; samp[e][t].ret = ret[t]; batch.push(std::mem::take(&mut samp[e][t])); }
    }
    ac.update(&mut batch, &cfg);
    let cpu_iter_ms = tc.elapsed().as_secs_f64() * 1e3;
    drop(cpu_envs);

    // ===================== FULL GPU ITERATION =====================
    let gpu_iter_ms = pollster::block_on(async {
        println!("building {n} GPU nexus envs...");
        let mut env = BipedNexusBatchEnv::new(&xml, n, 32, 0xC0FFEE).await;
        let mut ac2 = ActorCritic::new(&[od, 256, 256, 128, NUM_JOINTS], &[cd, 512, 256, 128, 1], 1.0, 1e-3, &mut rng);
        let mut gpu = GpuPolicy::new(env.backend(), &ac2, n).expect("gpu policy");
        let (mut gc, mut gcc) = env.initial_obs().await;
        let mut samp: Vec<Vec<Sample>> = (0..n).map(|_| Vec::with_capacity(t_steps)).collect();
        let (mut rs, mut vs, mut ds): (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<bool>>) =
            ((0..n).map(|_| vec![]).collect(), (0..n).map(|_| vec![]).collect(), (0..n).map(|_| vec![]).collect());
        let tg = Instant::now();
        // [IDLE-BREAKDOWN] phase timers: record/forward(+readback)/sample are
        // CPU-serial (GPU idle); env.step is mixed (its own buckets split it).
        let (mut t_rec, mut t_fwd, mut t_samp, mut t_step, mut t_commit) =
            (0u128, 0u128, 0u128, 0u128, 0u128);
        let _ = env.take_step_timings(); // reset env buckets before the timed rollout
        for _ in 0..t_steps {
            let t = Instant::now();
            for e in 0..n { ac2.record_obs(&gc[e], &gcc[e]); }
            t_rec += t.elapsed().as_nanos();
            let t = Instant::now();
            let (means, values) = gpu.forward(env.backend(), &ac2, &gc, &gcc).await.unwrap();
            t_fwd += t.elapsed().as_nanos();
            let t = Instant::now();
            let mut acts = Vec::with_capacity(n);
            for e in 0..n {
                let mean = means[e].to_vec();
                let mut a = vec![0.0f32; NUM_JOINTS];
                for k in 0..NUM_JOINTS { a[k] = mean[k] + ac2.log_std[k].exp() * rng.gauss(); }
                let lp = ac2.logp(&a, &mean);
                acts.push(to_action(&a));
                samp[e].push(Sample { obs: gc[e].clone(), critic_obs: gcc[e].clone(), action: a, mean_old: mean, logp_old: lp, value_old: values[e], adv: 0.0, ret: 0.0 });
                vs[e].push(values[e]);
            }
            t_samp += t.elapsed().as_nanos();
            let t = Instant::now();
            let outs: Vec<StepOut> = env.step(&acts).await;
            t_step += t.elapsed().as_nanos();
            let t = Instant::now();
            for e in 0..n { rs[e].push(outs[e].reward); ds[e].push(outs[e].done); gc[e].clone_from(&outs[e].obs); gcc[e].clone_from(&outs[e].critic_obs); }
            t_commit += t.elapsed().as_nanos();
        }
        if std::env::var("BIPED_IDLE_DBG").is_ok() {
            let et = env.take_step_timings();
            let ms = |x: u128| x as f64 / 1e6;
            eprintln!(
                "[idle] rollout {:.0}ms: record(cpu) {:.0} | forward(gpu+readback) {:.0} | sample(cpu) {:.0} | step {:.0} | commit(cpu) {:.0}",
                ms(t_rec + t_fwd + t_samp + t_step + t_commit),
                ms(t_rec), ms(t_fwd), ms(t_samp), ms(t_step), ms(t_commit));
            eprintln!(
                "[idle]   env.step breakdown: stage_motors {:.0} | flush {:.0} | physics_encode {:.0} | gpu_wait(compute) {:.0} | slurp(readback) {:.0} | par_obs+reward(cpu rayon) {:.0} | commit {:.0}",
                ms(et.stage_motors_ns as u128), ms(et.flush_static_ns as u128),
                ms(et.pipeline_step_ns as u128), ms(et.gpu_wait_ns as u128),
                ms(et.readback_ns as u128), ms(et.par_compute_ns as u128),
                ms(et.serial_commit_ns as u128));
        }
        let mut batch: Vec<Sample> = Vec::with_capacity(n * t_steps);
        for e in 0..n {
            let lv = ac2.value(&gcc[e]);
            let (adv, ret) = gae(&rs[e], &vs[e], &ds[e], lv, 0.99, 0.95);
            for t in 0..t_steps { samp[e][t].adv = adv[t]; samp[e][t].ret = ret[t]; batch.push(std::mem::take(&mut samp[e][t])); }
        }
        // GPU update over epochs x minibatches
        let tu = Instant::now();
        let total = batch.len();
        let mb = total / minibatches;
        let bk = env.backend().clone();
        let g = Gemm::from_backend(&bk).unwrap();
        let op = OpAssign::from_backend(&bk).unwrap();
        let act = Activation::from_backend(&bk).unwrap();
        let ad = Adam::from_backend(&bk).unwrap();
        let ppo = Ppo::from_backend(&bk).unwrap();
        let cont = Contiguous::from_backend(&bk).unwrap();
        let mut sh = TensorLayoutBuffers::new(&bk);
        let (st, rw) = (BufferUsages::STORAGE, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let mut a_net = GpuMlp::new(&bk, &ac2.actor, mb);
        let mut c_net = GpuMlp::new(&bk, &ac2.critic, mb);
        let ad_ = NUM_JOINTS;
        // GPU-RESIDENT: normalize + upload the FULL batch once; each minibatch is
        // a contiguous copy of a column slice (no per-minibatch CPU work).
        let on: Vec<Vec<f32>> = batch.iter().map(|s| ac2.obs_norm.normalize(&s.obs)).collect();
        let cn: Vec<Vec<f32>> = batch.iter().map(|s| ac2.critic_norm.normalize(&s.critic_obs)).collect();
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
        let mut lst = mk(&bk, &DMatrix::from_fn(ad_, 1, |r, _| ac2.log_std[r]), rw);
        let o1m = mk(&bk, &DMatrix::<f32>::from_element(1, mb, 1.0), st);
        let om1 = mk(&bk, &DMatrix::<f32>::from_element(mb, 1, 1.0), st);
        let mut gls = mk(&bk, &DMatrix::<f32>::zeros(ad_, mb), rw);
        let mut dls = mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), rw);
        let (mut mls, mut vls) = (mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), st), mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), st));
        let scale = 1.0 / mb as f32;
        let ap = Tensor::scalar(&bk, PpoActorParams { clip: 0.2, entropy_coef: 0.005, scale, log_sqrt_2pi: LOG_SQRT_2PI, action_dim: ad_ as u32, num_cols: mb as u32, pad0: 0, pad1: 0 }, BufferUsages::UNIFORM).unwrap();
        let vp = Tensor::scalar(&bk, PpoValueParams { clip: 0.2, value_coef: 0.5, scale, num_cols: mb as u32, pad0: 0, pad1: 0, pad2: 0, pad3: 0 }, BufferUsages::UNIFORM).unwrap();
        let adp = Tensor::scalar(&bk, AdamParams { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, bias_correction1: 0.1, bias_correction2: 0.001, pad0: 0.0, pad1: 0.0 }, BufferUsages::UNIFORM).unwrap();
        let (la, lc) = (a_net.layers() - 1, c_net.layers() - 1);
        // One encoder + one submit/sync PER EPOCH (not per minibatch); inter-pass
        // barriers keep the minibatch steps ordered (Adam writes w, next forward
        // reads it) while removing ~minibatches-1 host<->GPU sync stalls per epoch.
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
                    a_net.adam(&bk, &ad, &mut sh, &mut $enc, &adp).unwrap();
                    c_net.adam(&bk, &ad, &mut sh, &mut $enc, &adp).unwrap();
                    { let mut p = $enc.begin_pass("al", None); ad.step(&bk, &mut sh, &mut p, &adp, &mut lst, &dls, &mut mls, &mut vls).unwrap(); }
                }
            }};
        }
        // KHAL_CUDA_PROFILE syncs after every launch, which is illegal mid-capture
        // — fall through to the eager path so the per-kernel timer can run.
        let profiling = std::env::var_os("KHAL_CUDA_PROFILE").is_some();
        if let Some(cuda) = bk.as_cuda().filter(|_| !profiling) {
            // CUDA-graph path: capture ONE epoch's launch sequence (recorded, not
            // executed) into a graph, then replay it `epochs` times with one
            // cuGraphLaunch each — skipping all per-dispatch host encode. Every
            // epoch runs the identical kernel sequence (same minibatch offsets,
            // in-place weight updates), so the captured graph is correct to replay.
            // Warm the TensorLayoutBuffers shape cache with ONE eager epoch first
            // (this also executes epoch 0). The cache is idempotent, so the
            // subsequent captured epoch issues NO host->device shape uploads —
            // those pageable H2D copies during capture would make cuGraphLaunch
            // return INVALID_VALUE.
            {
                let mut enc = bk.begin_encoding();
                run_minibatches!(enc);
                bk.submit(enc).unwrap();
                bk.synchronize().unwrap();
            }
            let tcap = Instant::now();
            cuda.begin_capture().unwrap();
            let mut enc = bk.begin_encoding();
            run_minibatches!(enc);
            bk.submit(enc).unwrap();
            let graph = cuda.end_capture().unwrap();
            let cap_ms = tcap.elapsed().as_secs_f64() * 1e3;
            // epoch 0 ran eagerly above; replay the captured epoch for the rest.
            let trep = Instant::now();
            for _ in 1..epochs {
                graph.launch().unwrap();
            }
            bk.synchronize().unwrap();
            let rep_ms = trep.elapsed().as_secs_f64() * 1e3;
            let reps = (epochs - 1).max(1);
            eprintln!(
                "  [graph] capture+instantiate {cap_ms:.1} ms; replay {reps}e {rep_ms:.1} ms ({:.2} ms/epoch vs eager ~29)",
                rep_ms / reps as f64
            );
        } else {
            for _ in 0..epochs {
                let mut enc = bk.begin_encoding();
                run_minibatches!(enc);
                bk.submit(enc).unwrap();
                bk.synchronize().unwrap();
            }
        }
        let upd_ms = tu.elapsed().as_secs_f64() * 1e3;
        let gpu_ms = tg.elapsed().as_secs_f64() * 1e3;
        eprintln!("  [WebGPU update only] {upd_ms:9.1} ms");
        // Correctness guard: a_net.a[0] holds the LAST minibatch's gathered obs —
        // confirm the .columns()+Contiguous gather matches the CPU-normalized data.
        let a0 = bk.slow_read_vec(a_net.a[0].buffer()).await.unwrap();
        let last = (minibatches - 1) * mb;
        let mut gerr = 0f32;
        for c in 0..mb.min(96) {
            for r in 0..od {
                gerr = gerr.max((a0[r * mb + c] - on[last + c][r]).abs());
            }
        }
        if gerr > 1e-4 {
            println!("  WARNING: gather mismatch {gerr:.3e} — GPU update inputs are WRONG");
        } else {
            println!("  (gather verified, err {gerr:.3e})");
        }
        gpu_ms
    });

    // Per-kernel GPU timing ranking (KHAL_CUDA_PROFILE=1). No-op otherwise.
    #[cfg(feature = "cuda_backend")]
    khal::backend::cuda::dump_kernel_profile();

    let ctrl = (n * t_steps) as f64;
    let cpu_eps = ctrl / (cpu_iter_ms / 1e3) / 1e3;
    let gpu_eps = ctrl / (gpu_iter_ms / 1e3) / 1e3;
    println!("\nfull iteration — {n} envs, T={t_steps}, {epochs}e x {minibatches}mb");
    println!("  FULL CPU (rapier + CPU MLP + CPU update) : {cpu_iter_ms:9.1} ms = {cpu_eps:6.2} k env/s");
    println!("  FULL GPU (nexus + vortx + GPU update)    : {gpu_iter_ms:9.1} ms = {gpu_eps:6.2} k env/s");
    println!("  speedup                                  : {:.2}x", cpu_iter_ms / gpu_iter_ms);
}
