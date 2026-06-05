//! Benchmark: GPU PPO update vs CPU `ActorCritic::update` (rayon) at biped scale.
//!
//! Run: `cargo run --release --example gpu_update_bench --features "gpu biped_gpu" -- [minibatch] [reps]`
//!
//! The CPU PPO update is what caps training-iteration throughput (the rollout is
//! already on the GPU). This times one minibatch update — actor+critic forward,
//! PPO gradients, backward, Adam — CPU (multi-core rayon) vs GPU (vortx), at the
//! deployed net shapes (actor [43,256,256,128,12], critic [49,512,256,128,1]).
//! Same update path verified bit-exact in `gpu_update_check`.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use nalgebra::DMatrix;
use std::time::Instant;
use vortx::linalg::{
    Activation, Adam, AdamParams, Gemm, OpAssign, OpAssignVariant, Ppo, PpoActorParams,
    PpoValueParams,
};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use wgpu::{Features, Limits};
use zealot_rl::net::Mlp;
use zealot_rl::rng::Lcg;
use zealot_rl::{ActorCritic, PpoConfig, Sample};

const A_DIMS: [usize; 5] = [43, 256, 256, 128, 12];
const C_DIMS: [usize; 5] = [49, 512, 256, 128, 1];
const ACT: usize = 12;
const CLIP: f32 = 0.2;
const ENT: f32 = 0.005;
const VCOEF: f32 = 0.5;
const LR: f32 = 1e-3;
const LOG_SQRT_2PI: f32 = 0.918_938_5;

fn mk(b: &GpuBackend, m: &DMatrix<f32>, u: BufferUsages) -> Tensor<f32> {
    Tensor::matrix_from_na(b, m, u).unwrap()
}
fn wmat(w: &[f32], out: usize, inp: usize) -> DMatrix<f32> {
    DMatrix::from_fn(out, inp, |r, c| w[r * inp + c])
}

struct GpuMlp {
    dims: Vec<usize>,
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
        let dims = net.dims.clone();
        let st = BufferUsages::STORAGE;
        let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let l = net.w.len();
        let z = |r: usize, c: usize| DMatrix::<f32>::zeros(r, c);
        GpuMlp {
            w: (0..l)
                .map(|i| mk(bk, &wmat(&net.w[i], dims[i + 1], dims[i]), rw))
                .collect(),
            b: (0..l)
                .map(|i| {
                    mk(
                        bk,
                        &DMatrix::from_fn(dims[i + 1], 1, |r, _| net.b[i][r]),
                        rw,
                    )
                })
                .collect(),
            mw: (0..l)
                .map(|i| mk(bk, &z(dims[i + 1], dims[i]), st))
                .collect(),
            vw: (0..l)
                .map(|i| mk(bk, &z(dims[i + 1], dims[i]), st))
                .collect(),
            mb: (0..l).map(|i| mk(bk, &z(dims[i + 1], 1), st)).collect(),
            vb: (0..l).map(|i| mk(bk, &z(dims[i + 1], 1), st)).collect(),
            a: (0..=l).map(|i| mk(bk, &z(dims[i], m), rw)).collect(),
            bb: (0..l).map(|i| mk(bk, &z(dims[i + 1], m), st)).collect(),
            delta: (0..l).map(|i| mk(bk, &z(dims[i + 1], m), rw)).collect(),
            dw: (0..l)
                .map(|i| mk(bk, &z(dims[i + 1], dims[i]), rw))
                .collect(),
            db: (0..l).map(|i| mk(bk, &z(dims[i + 1], 1), rw)).collect(),
            dims,
        }
    }
    fn layers(&self) -> usize {
        self.w.len()
    }
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        bk: &GpuBackend,
        g: &Gemm,
        op: &OpAssign,
        act: &Activation,
        sh: &mut TensorLayoutBuffers,
        enc: &mut <GpuBackend as Backend>::Encoder,
        ones_1m: &Tensor<f32>,
    ) -> anyhow::Result<()> {
        let l = self.layers();
        for i in 0..l {
            let (left, right) = self.a.split_at_mut(i + 1);
            let (ain, aout) = (&left[i], &mut right[0]);
            {
                let mut p = enc.begin_pass("z", None);
                g.dispatch_naive(bk, sh, &mut p, &mut *aout, &self.w[i], ain)?;
            }
            {
                let mut p = enc.begin_pass("bb", None);
                g.dispatch_naive(bk, sh, &mut p, &mut self.bb[i], &self.b[i], ones_1m)?;
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
        bk: &GpuBackend,
        g: &Gemm,
        act: &Activation,
        sh: &mut TensorLayoutBuffers,
        enc: &mut <GpuBackend as Backend>::Encoder,
        ones_m1: &Tensor<f32>,
    ) -> anyhow::Result<()> {
        for i in (0..self.layers()).rev() {
            {
                let mut p = enc.begin_pass("dw", None);
                g.dispatch_naive(
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
                g.dispatch_naive(bk, sh, &mut p, &mut self.db[i], &self.delta[i], ones_m1)?;
            }
            if i > 0 {
                {
                    let (left, right) = self.delta.split_at_mut(i);
                    let dprev = &mut left[i - 1];
                    let dcur = &right[0];
                    let mut p = enc.begin_pass("da", None);
                    g.dispatch_naive(bk, sh, &mut p, dprev, self.w[i].transpose_last_dims(), dcur)?;
                }
                {
                    let mut p = enc.begin_pass("elub", None);
                    act.elu_backward(bk, sh, &mut p, &mut self.delta[i - 1], &self.a[i])?;
                }
            }
        }
        Ok(())
    }
    fn adam(
        &mut self,
        bk: &GpuBackend,
        ad: &Adam,
        sh: &mut TensorLayoutBuffers,
        enc: &mut <GpuBackend as Backend>::Encoder,
        params: &Tensor<AdamParams>,
    ) -> anyhow::Result<()> {
        for i in 0..self.layers() {
            {
                let mut p = enc.begin_pass("adw", None);
                ad.step(
                    bk,
                    sh,
                    &mut p,
                    params,
                    &mut self.w[i],
                    &self.dw[i],
                    &mut self.mw[i],
                    &mut self.vw[i],
                )?;
            }
            {
                let mut p = enc.begin_pass("adb", None);
                ad.step(
                    bk,
                    sh,
                    &mut p,
                    params,
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

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let mb: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let mut rng = Lcg::new(11);
    let mut ac = ActorCritic::new(&A_DIMS, &C_DIMS, 1.0, LR, &mut rng);

    let mut obs = vec![[0f32; 43]; mb];
    let mut cobs = vec![[0f32; 49]; mb];
    for e in 0..mb {
        for v in obs[e].iter_mut() {
            *v = rng.gauss();
        }
        for v in cobs[e].iter_mut() {
            *v = rng.gauss();
        }
        ac.record_obs(&obs[e], &cobs[e]);
    }
    let mut batch: Vec<Sample> = Vec::with_capacity(mb);
    for e in 0..mb {
        let (action, logp, mean) = ac.sample(&obs[e], &mut rng);
        let value = ac.value(&cobs[e]);
        batch.push(Sample {
            obs: obs[e].to_vec(),
            critic_obs: cobs[e].to_vec(),
            action,
            mean_old: mean,
            logp_old: logp,
            value_old: value,
            adv: rng.gauss() * 1.5,
            ret: value + rng.gauss(),
        });
    }
    let cfg = PpoConfig {
        clip: CLIP,
        entropy_coef: ENT,
        value_coef: VCOEF,
        epochs: 1,
        minibatches: 1,
        adaptive_lr: false,
        max_grad_norm: 1e9,
        ..PpoConfig::default()
    };

    // ---- CPU: time one minibatch update (rayon) ----
    let mut ac_cpu = ac.clone();
    ac_cpu.update(&mut batch.clone(), &cfg); // warmup
    let t0 = Instant::now();
    for _ in 0..reps {
        ac_cpu.update(&mut batch.clone(), &cfg);
    }
    let cpu = t0.elapsed().as_secs_f64() / reps as f64 * 1e3;

    // ---- GPU setup ----
    // Bump buffer limits so large minibatches (e.g. 32768) fit, like
    // pendulum_gpu_policy — default wgpu caps reject the big activation buffers.
    let limits = Limits {
        max_buffer_size: 4_000_000_000,
        max_storage_buffer_binding_size: 2_000_000_000,
        max_storage_buffers_per_shader_stage: 14,
        ..Limits::default()
    };
    let bk = GpuBackend::WebGpu(WebGpu::new(Features::default(), limits).await?);
    let g = Gemm::from_backend(&bk)?;
    let op = OpAssign::from_backend(&bk)?;
    let act = Activation::from_backend(&bk)?;
    let ad = Adam::from_backend(&bk)?;
    let ppo = Ppo::from_backend(&bk)?;
    let mut sh = TensorLayoutBuffers::new(&bk);
    let st = BufferUsages::STORAGE;
    let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let mut a_net = GpuMlp::new(&bk, &ac.actor, mb);
    let mut c_net = GpuMlp::new(&bk, &ac.critic, mb);
    let on: Vec<Vec<f32>> = obs.iter().map(|o| ac.obs_norm.normalize(o)).collect();
    let cn: Vec<Vec<f32>> = cobs.iter().map(|o| ac.critic_norm.normalize(o)).collect();
    a_net.a[0] = mk(&bk, &DMatrix::from_fn(43, mb, |r, c| on[c][r]), rw);
    c_net.a[0] = mk(&bk, &DMatrix::from_fn(49, mb, |r, c| cn[c][r]), rw);
    let action_t = mk(
        &bk,
        &DMatrix::from_fn(ACT, mb, |r, c| batch[c].action[r]),
        st,
    );
    let mut log_std_t = mk(&bk, &DMatrix::from_fn(ACT, 1, |r, _| ac.log_std[r]), rw);
    let adv_t = mk(&bk, &DMatrix::from_fn(1, mb, |_, c| batch[c].adv), st);
    let logp_old_t = mk(&bk, &DMatrix::from_fn(1, mb, |_, c| batch[c].logp_old), st);
    let value_old_t = mk(&bk, &DMatrix::from_fn(1, mb, |_, c| batch[c].value_old), st);
    let ret_t = mk(&bk, &DMatrix::from_fn(1, mb, |_, c| batch[c].ret), st);
    let ones_1m = mk(&bk, &DMatrix::<f32>::from_element(1, mb, 1.0), st);
    let ones_m1 = mk(&bk, &DMatrix::<f32>::from_element(mb, 1, 1.0), st);
    let mut g_logstd = mk(&bk, &DMatrix::<f32>::zeros(ACT, mb), rw);
    let mut dlog_std = mk(&bk, &DMatrix::<f32>::zeros(ACT, 1), rw);
    let (mut m_ls, mut v_ls) = (
        mk(&bk, &DMatrix::<f32>::zeros(ACT, 1), st),
        mk(&bk, &DMatrix::<f32>::zeros(ACT, 1), st),
    );
    let scale = 1.0 / mb as f32;
    let ap = Tensor::scalar(
        &bk,
        PpoActorParams {
            clip: CLIP,
            entropy_coef: ENT,
            scale,
            log_sqrt_2pi: LOG_SQRT_2PI,
            action_dim: ACT as u32,
            num_cols: mb as u32,
            pad0: 0,
            pad1: 0,
        },
        BufferUsages::UNIFORM,
    )?;
    let vp = Tensor::scalar(
        &bk,
        PpoValueParams {
            clip: CLIP,
            value_coef: VCOEF,
            scale,
            num_cols: mb as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        },
        BufferUsages::UNIFORM,
    )?;
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let adam_p = Tensor::scalar(
        &bk,
        AdamParams {
            lr: LR,
            beta1: b1,
            beta2: b2,
            eps,
            bias_correction1: 1.0 - b1,
            bias_correction2: 1.0 - b2,
            pad0: 0.0,
            pad1: 0.0,
        },
        BufferUsages::UNIFORM,
    )?;

    let la = a_net.layers() - 1;
    let lc = c_net.layers() - 1;
    let mut one_update = |a_net: &mut GpuMlp,
                          c_net: &mut GpuMlp,
                          log_std_t: &mut Tensor<f32>,
                          g_logstd: &mut Tensor<f32>,
                          dlog_std: &mut Tensor<f32>,
                          m_ls: &mut Tensor<f32>,
                          v_ls: &mut Tensor<f32>|
     -> anyhow::Result<()> {
        let mut enc = bk.begin_encoding();
        a_net.forward(&bk, &g, &op, &act, &mut sh, &mut enc, &ones_1m)?;
        c_net.forward(&bk, &g, &op, &act, &mut sh, &mut enc, &ones_1m)?;
        {
            let mut p = enc.begin_pass("ag", None);
            ppo.actor_grad(
                &mut p,
                &ap,
                &a_net.a[la + 1],
                &action_t,
                &*log_std_t,
                &adv_t,
                &logp_old_t,
                &mut a_net.delta[la],
                &mut *g_logstd,
            )?;
        }
        {
            let mut p = enc.begin_pass("vg", None);
            ppo.value_grad(
                &mut p,
                &vp,
                &c_net.a[lc + 1],
                &value_old_t,
                &ret_t,
                &mut c_net.delta[lc],
            )?;
        }
        a_net.backward(&bk, &g, &act, &mut sh, &mut enc, &ones_m1)?;
        c_net.backward(&bk, &g, &act, &mut sh, &mut enc, &ones_m1)?;
        {
            let mut p = enc.begin_pass("dls", None);
            g.dispatch_naive(&bk, &mut sh, &mut p, &mut *dlog_std, &*g_logstd, &ones_m1)?;
        }
        a_net.adam(&bk, &ad, &mut sh, &mut enc, &adam_p)?;
        c_net.adam(&bk, &ad, &mut sh, &mut enc, &adam_p)?;
        {
            let mut p = enc.begin_pass("als", None);
            ad.step(
                &bk,
                &mut sh,
                &mut p,
                &adam_p,
                &mut *log_std_t,
                &*dlog_std,
                &mut *m_ls,
                &mut *v_ls,
            )?;
        }
        bk.submit(enc)?;
        bk.synchronize()?;
        Ok(())
    };
    one_update(
        &mut a_net,
        &mut c_net,
        &mut log_std_t,
        &mut g_logstd,
        &mut dlog_std,
        &mut m_ls,
        &mut v_ls,
    )?; // warmup
    let t1 = Instant::now();
    for _ in 0..reps {
        one_update(
            &mut a_net,
            &mut c_net,
            &mut log_std_t,
            &mut g_logstd,
            &mut dlog_std,
            &mut m_ls,
            &mut v_ls,
        )?;
    }
    let gpu = t1.elapsed().as_secs_f64() / reps as f64 * 1e3;

    println!("PPO update — minibatch {mb}, actor {A_DIMS:?}, critic {C_DIMS:?}");
    println!("  CPU update (rayon) : {cpu:8.2} ms/minibatch-step");
    println!("  GPU update (vortx) : {gpu:8.2} ms/minibatch-step");
    println!("  speedup            : {:.1}x", cpu / gpu);
    Ok(())
}
