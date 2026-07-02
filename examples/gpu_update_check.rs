//! Stage B — full PPO update on the GPU, verified against `ActorCritic::update`.
//!
//! Run: `cargo run --release --example gpu_update_check --features "gpu biped_gpu"`
//!
//! Assembles the verified kernels (GEMM, ELU fwd/bwd, the PPO actor/value
//! gradient kernels, GpuAdam) into one minibatch PPO update — forward(cache) →
//! ppo_actor_grad → actor backward → ppo_value_grad → critic backward →
//! log_std grad reduce → Adam — and checks the resulting weights against the CPU
//! `ActorCritic::update` run on the same batch (1 epoch, 1 minibatch, no clip,
//! fixed lr). A match here means the GPU update path is correct end to end.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend};
use nalgebra::DMatrix;
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

const A_DIMS: [usize; 4] = [6, 16, 8, 3]; // actor: obs 6 -> ... -> act 3
const C_DIMS: [usize; 4] = [8, 16, 8, 1]; // critic: cobs 8 -> ... -> 1
const M: usize = 128; // minibatch (== batch, single minibatch)
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

/// One GPU-resident MLP with Adam state + fwd/bwd/step buffers, batch `M`.
struct GpuMlp {
    dims: Vec<usize>,
    w: Vec<Tensor<f32>>,
    b: Vec<Tensor<f32>>, // [out x 1]
    mw: Vec<Tensor<f32>>,
    vw: Vec<Tensor<f32>>,
    mb: Vec<Tensor<f32>>,
    vb: Vec<Tensor<f32>>,
    a: Vec<Tensor<f32>>,     // activations [dim x M]
    bb: Vec<Tensor<f32>>,    // bias broadcast [out x M]
    delta: Vec<Tensor<f32>>, // [out x M]
    dw: Vec<Tensor<f32>>,
    db: Vec<Tensor<f32>>,
}

impl GpuMlp {
    fn new(bk: &GpuBackend, net: &Mlp) -> Self {
        let dims = net.dims.clone();
        let st = BufferUsages::STORAGE;
        let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let l = net.w.len();
        let z = |r: usize, c: usize| DMatrix::<f32>::zeros(r, c);
        let mut s = GpuMlp {
            dims: dims.clone(),
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
            a: (0..=l).map(|i| mk(bk, &z(dims[i], M), rw)).collect(),
            bb: (0..l).map(|i| mk(bk, &z(dims[i + 1], M), st)).collect(),
            delta: (0..l).map(|i| mk(bk, &z(dims[i + 1], M), rw)).collect(),
            dw: (0..l)
                .map(|i| mk(bk, &z(dims[i + 1], dims[i]), rw))
                .collect(),
            db: (0..l).map(|i| mk(bk, &z(dims[i + 1], 1), rw)).collect(),
        };
        s.dims = dims;
        s
    }

    fn layers(&self) -> usize {
        self.w.len()
    }

    // forward: caches activations into self.a (a[0] must be set to the input first)
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
                g.dispatch_tiled(bk, sh, &mut p, &mut *aout, &self.w[i], ain)?;
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

    // backward: assumes delta[L-1] already holds dL/d(output); fills dw, db.
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
                g.dispatch_naive(bk, sh, &mut p, &mut self.db[i], &self.delta[i], ones_m1)?;
            }
            if i > 0 {
                {
                    let (left, right) = self.delta.split_at_mut(i);
                    let dprev = &mut left[i - 1];
                    let dcur = &right[0];
                    let mut p = enc.begin_pass("da", None);
                    g.dispatch_tiled(bk, sh, &mut p, dprev, self.w[i].transpose_last_dims(), dcur)?;
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
    let mut rng = Lcg::new(11);
    let mut ac = ActorCritic::new(&A_DIMS, &C_DIMS, 1.0, LR, &mut rng);

    // Build a batch; populate the normalizers so forward normalization is valid.
    let mut obs = vec![[0f32; 6]; M];
    let mut cobs = vec![[0f32; 8]; M];
    for e in 0..M {
        for v in obs[e].iter_mut() {
            *v = rng.gauss();
        }
        for v in cobs[e].iter_mut() {
            *v = rng.gauss();
        }
        ac.record_obs(&obs[e], &cobs[e]);
    }
    // Fill samples using the *current* policy (so logp_old/mean_old/value_old are consistent).
    let mut batch: Vec<Sample> = Vec::with_capacity(M);
    for e in 0..M {
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

    // ---- CPU reference: one update, single epoch/minibatch, no clip, fixed lr ----
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
    let mut ac_cpu = ac.clone();
    let mut batch_cpu = batch.clone();
    ac_cpu.update(&mut batch_cpu, &cfg);

    // The CPU `update` normalizes advantages in-place across the batch; replicate
    // that exact transform for the GPU inputs.
    let mean_adv: f32 = batch.iter().map(|s| s.adv).sum::<f32>() / M as f32;
    let var_adv: f32 = batch
        .iter()
        .map(|s| (s.adv - mean_adv).powi(2))
        .sum::<f32>()
        / M as f32;
    let sd_adv = var_adv.sqrt().max(1e-6);

    // ---- GPU update ----
    let bk = GpuBackend::auto(Features::default(), Limits::default()).await?;
    let g = Gemm::from_backend(&bk)?;
    let op = OpAssign::from_backend(&bk)?;
    let act = Activation::from_backend(&bk)?;
    let ad = Adam::from_backend(&bk)?;
    let ppo = Ppo::from_backend(&bk)?;
    let mut sh = TensorLayoutBuffers::new(&bk);
    let st = BufferUsages::STORAGE;
    let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

    let mut a_net = GpuMlp::new(&bk, &ac.actor);
    let mut c_net = GpuMlp::new(&bk, &ac.critic);

    // Inputs (normalized obs, matching ac's normalizers), row-major [dim x M].
    let on: Vec<Vec<f32>> = obs.iter().map(|o| ac.obs_norm.normalize(o)).collect();
    let cn: Vec<Vec<f32>> = cobs.iter().map(|o| ac.critic_norm.normalize(o)).collect();
    a_net.a[0] = mk(&bk, &DMatrix::from_fn(6, M, |r, c| on[c][r]), rw);
    c_net.a[0] = mk(&bk, &DMatrix::from_fn(8, M, |r, c| cn[c][r]), rw);

    let action_t = mk(&bk, &DMatrix::from_fn(3, M, |r, c| batch[c].action[r]), st);
    let mut log_std_t = mk(&bk, &DMatrix::from_fn(3, 1, |r, _| ac.log_std[r]), rw);
    let adv_t = mk(
        &bk,
        &DMatrix::from_fn(1, M, |_, c| (batch[c].adv - mean_adv) / sd_adv),
        st,
    );
    let logp_old_t = mk(&bk, &DMatrix::from_fn(1, M, |_, c| batch[c].logp_old), st);
    let value_old_t = mk(&bk, &DMatrix::from_fn(1, M, |_, c| batch[c].value_old), st);
    let ret_t = mk(&bk, &DMatrix::from_fn(1, M, |_, c| batch[c].ret), st);
    let ones_1m = mk(&bk, &DMatrix::<f32>::from_element(1, M, 1.0), st);
    let ones_m1 = mk(&bk, &DMatrix::<f32>::from_element(M, 1, 1.0), st);
    let mut g_logstd = mk(&bk, &DMatrix::<f32>::zeros(3, M), rw);
    let mut dlog_std = mk(&bk, &DMatrix::<f32>::zeros(3, 1), rw);
    let (mut m_ls, mut v_ls) = (
        mk(&bk, &DMatrix::<f32>::zeros(3, 1), st),
        mk(&bk, &DMatrix::<f32>::zeros(3, 1), st),
    );

    let scale = 1.0 / M as f32;
    let ap = Tensor::scalar(
        &bk,
        PpoActorParams {
            clip: CLIP,
            entropy_coef: ENT,
            scale,
            log_sqrt_2pi: LOG_SQRT_2PI,
            action_dim: 3,
            num_cols: M as u32,
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
            num_cols: M as u32,
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

    let mut enc = bk.begin_encoding();
    a_net.forward(&bk, &g, &op, &act, &mut sh, &mut enc, &ones_1m)?;
    c_net.forward(&bk, &g, &op, &act, &mut sh, &mut enc, &ones_1m)?;
    // PPO output grads -> write straight into each net's top delta buffer.
    let la = a_net.layers() - 1;
    {
        let mut p = enc.begin_pass("actor_grad", None);
        ppo.actor_grad(
            &mut p,
            &ap,
            &a_net.a[la + 1],
            &action_t,
            &log_std_t,
            &adv_t,
            &logp_old_t,
            &mut a_net.delta[la],
            &mut g_logstd,
        )?;
    }
    let lc = c_net.layers() - 1;
    {
        let mut p = enc.begin_pass("value_grad", None);
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
    // log_std grad = sum over columns of g_logstd
    {
        let mut p = enc.begin_pass("dlogstd", None);
        g.dispatch_naive(&bk, &mut sh, &mut p, &mut dlog_std, &g_logstd, &ones_m1)?;
    }
    // Adam steps
    a_net.adam(&bk, &ad, &mut sh, &mut enc, &adam_p)?;
    c_net.adam(&bk, &ad, &mut sh, &mut enc, &adam_p)?;
    {
        let mut p = enc.begin_pass("adam_logstd", None);
        ad.step(
            &bk,
            &mut sh,
            &mut p,
            &adam_p,
            &mut log_std_t,
            &dlog_std,
            &mut m_ls,
            &mut v_ls,
        )?;
    }
    bk.submit(enc)?;
    bk.synchronize()?;

    // ---- compare actor + critic weights ----
    let mut worst = 0f32;
    for (name, gpu, cpu) in [
        ("actor", &a_net, &ac_cpu.actor),
        ("critic", &c_net, &ac_cpu.critic),
    ] {
        for i in 0..gpu.layers() {
            let (out, inp) = (gpu.dims[i + 1], gpu.dims[i]);
            let wg = bk.slow_read_vec(gpu.w[i].buffer()).await?;
            let bg = bk.slow_read_vec(gpu.b[i].buffer()).await?;
            let mut ew = 0f32;
            for r in 0..out {
                for c in 0..inp {
                    ew = ew.max((wg[r * inp + c] - cpu.w[i][r * inp + c]).abs());
                }
            }
            let mut eb = 0f32;
            for r in 0..out {
                eb = eb.max((bg[r] - cpu.b[i][r]).abs());
            }
            println!("  {name} L{i}: dW err {ew:.3e}  db err {eb:.3e}");
            worst = worst.max(ew).max(eb);
        }
    }
    // log_std (separate Adam in the CPU; no clamp hit for one small step)
    let lsg = bk.slow_read_vec(log_std_t.buffer()).await?;
    let mut els = 0f32;
    for k in 0..3 {
        els = els.max((lsg[k] - ac_cpu.log_std[k]).abs());
    }
    println!("  log_std err {els:.3e}");
    worst = worst.max(els);
    println!("worst param error after one PPO update (gpu vs cpu) = {worst:.3e}");
    anyhow::ensure!(
        worst < 1e-4,
        "GPU PPO update diverged from CPU ActorCritic::update"
    );
    println!("OK — GPU PPO update matches CPU ActorCritic::update. Stage B core verified.");
    Ok(())
}
