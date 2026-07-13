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

use biped_env_nexus::{BipedNexusBatchEnv, REWARD_COMP_NAMES, default_mjcf_path};
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
use zealot_env::robots::{RobotSpec, NUM_JOINTS};
use zealot_rl::ActorCritic;
use zealot_rl::net::Mlp;
use zealot_rl::ppo::{Sample, gae};
use zealot_rl::rng::Lcg;

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
// LR floor. Was 1e-5, then 3e-4: KL used to persistently sit ~0.02–0.04 so the
// adaptive-KL controller pegged lr at the floor — at 1e-5 too low to learn
// (crouch-and-fall optimum), so it was raised to 3e-4. But 3e-4 was then too HIGH
// to brake: once the walk-command ramp destabilized the policy, lr couldn't drop
// enough and per-iter KL ran away to ~100. The real fix is the KL early-stop in
// the epoch loop (caps per-iter KL), which stops KL persistently sitting high —
// so the controller no longer pegs the floor and 1e-4 is safe (gives braking room
// without the crouch regression).
const LR_MIN: f32 = 1e-4;
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

// L/R mirror augmentation → symmetric policy (fixes the lopsided, veering gait).
// The joint mirror permutation/signs are per-robot (canonical joint orders
// differ): lateral families (roll/yaw) negate, sagittal (pitch/knee) keep. The
// mirror is an action-space isometry, so logp_old/adv are preserved when
// mean_old mirrors.
static ROBOT: std::sync::LazyLock<RobotSpec> = std::sync::LazyLock::new(RobotSpec::from_env);
static JMIRROR: std::sync::LazyLock<[usize; NUM_JOINTS]> =
    std::sync::LazyLock::new(|| ROBOT.mirror);
static JSIGN: std::sync::LazyLock<[f32; NUM_JOINTS]> =
    std::sync::LazyLock::new(|| ROBOT.mirror_sign);
fn jmirror(v: &[f32]) -> Vec<f32> {
    (0..NUM_JOINTS).map(|i| JSIGN[i] * v[JMIRROR[i]]).collect()
}
// obs(45): last_action[0:12], cmd[12:16]=(vx,vy,yaw,aux), joint_pos[16:28],
// joint_vel[28:40], proj_grav[40:43]=(fwd,lat,up), gait_phase[43:45]=(sin,cos).
fn mirror_obs(o: &[f32]) -> Vec<f32> {
    let mut m = o.to_vec();
    m[0..12].copy_from_slice(&jmirror(&o[0..12]));
    m[13] = -o[13];
    m[14] = -o[14];
    m[16..28].copy_from_slice(&jmirror(&o[16..28]));
    m[28..40].copy_from_slice(&jmirror(&o[28..40]));
    m[41] = -o[41];
    m[43] = -o[43];
    m[44] = -o[44];
    m
}
// critic(51) = obs(45) + base_lin_vel(3)[fwd,lat,up] + base_ang_vel(3)[roll,pitch,yaw].
fn mirror_critic(c: &[f32]) -> Vec<f32> {
    let mut m = mirror_obs(&c[0..45]);
    m.extend_from_slice(&c[45..]);
    m[46] = -c[46];
    m[48] = -c[48];
    m[50] = -c[50];
    m
}
fn mirror_sample(s: &Sample) -> Sample {
    Sample {
        obs: mirror_obs(&s.obs),
        critic_obs: mirror_critic(&s.critic_obs),
        action: jmirror(&s.action),
        mean_old: jmirror(&s.mean_old),
        logp_old: s.logp_old,
        value_old: s.value_old,
        adv: s.adv,
        ret: s.ret,
    }
}

// ---------------------------------------------------------------------------
// Symmetry method 4 — NET: an *architecturally* equivariant policy.
//
// A signed permutation P on R^d: `(P x)[i] = sign[i] * x[perm[i]]`, an
// involution (P·P = I). The four mirrors above are all signed permutations:
// `mirror_obs`/`mirror_critic`/`jmirror` ARE such P on the obs/critic/action
// spaces. A linear layer `y = W x` is equivariant under input rep `P_in` and
// output rep `P_out` iff `W = P_out · W · P_in`; the symmetric projection
// `W ← ½(W + P_out W P_in)` lands W in that subspace exactly. Chain it through
// every layer (re-projecting each iter so Adam can't drift it) and the whole
// net satisfies `π(mirror(s)) = mirror(π(s))` BY CONSTRUCTION — provably
// symmetric, zero inference cost, and (unlike DUP) no batch/KL distortion so it
// warm-starts cleanly. ELU commutes with the hidden reps ONLY if they are pure
// permutations (sign +1) — ELU(−x) ≠ −ELU(x) — so hidden layers use an
// adjacent-pair swap (no sign); the signed input/output reps sit at the linear
// boundaries (raw input, linear output) where no nonlinearity breaks them.
struct SPerm {
    perm: Vec<usize>,
    sign: Vec<f32>,
}
impl SPerm {
    fn identity(d: usize) -> Self {
        SPerm {
            perm: (0..d).collect(),
            sign: vec![1.0; d],
        }
    }
    /// Pure adjacent-pair swap 2k↔2k+1 (sign +1) — a permutation involution that
    /// commutes with elementwise ELU. `d` must be even (our hidden dims are).
    fn pair_swap(d: usize) -> Self {
        let mut perm: Vec<usize> = (0..d).collect();
        let mut k = 0;
        while k + 1 < d {
            perm[k] = k + 1;
            perm[k + 1] = k;
            k += 2;
        }
        SPerm {
            perm,
            sign: vec![1.0; d],
        }
    }
}
fn action_sperm() -> SPerm {
    SPerm {
        perm: JMIRROR.to_vec(),
        sign: JSIGN.to_vec(),
    }
}
// obs(45) signed perm — exactly the index/sign pattern of `mirror_obs`.
fn obs_sperm() -> SPerm {
    let mut perm: Vec<usize> = (0..45).collect();
    let mut sign = vec![1.0f32; 45];
    for i in 0..NUM_JOINTS {
        perm[i] = JMIRROR[i];
        sign[i] = JSIGN[i]; // last_action
        perm[16 + i] = 16 + JMIRROR[i];
        sign[16 + i] = JSIGN[i]; // joint_pos
        perm[28 + i] = 28 + JMIRROR[i];
        sign[28 + i] = JSIGN[i]; // joint_vel
    }
    sign[13] = -1.0; // cmd vy
    sign[14] = -1.0; // cmd yaw
    sign[41] = -1.0; // proj_grav lateral
    sign[43] = -1.0; // gait phase sin
    sign[44] = -1.0; // gait phase cos
    SPerm { perm, sign }
}
// critic(51) signed perm = obs(45) + [lin_vel(3), ang_vel(3)] per `mirror_critic`.
fn critic_sperm() -> SPerm {
    let mut sp = obs_sperm();
    sp.perm.extend(45..51);
    sp.sign.extend(std::iter::repeat(1.0).take(6));
    sp.sign[46] = -1.0; // lin_vel lateral
    sp.sign[48] = -1.0; // ang_vel roll
    sp.sign[50] = -1.0; // ang_vel yaw
    sp
}
/// Project every layer of `net` onto the equivariant subspace for the given
/// per-layer reps (`reps[l]` acts on layer-l activations; `reps[0]` = input rep,
/// `reps[L]` = output rep). Idempotent; called each iter after the GPU→CPU sync.
fn symmetrize_mlp(net: &mut Mlp, reps: &[SPerm]) {
    for l in 0..net.layers() {
        let (out, inp) = (net.dims[l + 1], net.dims[l]);
        let (ro, ri) = (&reps[l + 1], &reps[l]);
        let orig = net.w[l].clone();
        for o in 0..out {
            for i in 0..inp {
                let m = ro.sign[o] * ri.sign[i] * orig[ro.perm[o] * inp + ri.perm[i]];
                net.w[l][o * inp + i] = 0.5 * (orig[o * inp + i] + m);
            }
        }
        let ob = net.b[l].clone();
        for o in 0..out {
            net.b[l][o] = 0.5 * (ob[o] + ro.sign[o] * ob[ro.perm[o]]);
        }
    }
}
fn actor_reps(net: &Mlp) -> Vec<SPerm> {
    let mut r = vec![obs_sperm()];
    for &h in &net.dims[1..net.dims.len() - 1] {
        r.push(SPerm::pair_swap(h));
    }
    r.push(action_sperm());
    r
}
fn critic_reps(net: &Mlp) -> Vec<SPerm> {
    let mut r = vec![critic_sperm()];
    for &h in &net.dims[1..net.dims.len() - 1] {
        r.push(SPerm::pair_swap(h));
    }
    r.push(SPerm::identity(1)); // value is mirror-INVARIANT (trivial output rep)
    r
}
fn symmetrize_ac(ac: &mut ActorCritic) {
    let ar = actor_reps(&ac.actor);
    symmetrize_mlp(&mut ac.actor, &ar);
    let cr = critic_reps(&ac.critic);
    symmetrize_mlp(&mut ac.critic, &cr);
    // Also symmetrize the per-action exploration std so the action DISTRIBUTION
    // (not just the mean) is equivariant: std[i] = std[JMIRROR[i]]. log_std is
    // magnitude (sign-free), so just average the mirror pair. Without this the
    // mean is exactly symmetric but exploration is slightly lopsided.
    let orig = ac.log_std.clone();
    for i in 0..NUM_JOINTS {
        ac.log_std[i] = 0.5 * (orig[i] + orig[JMIRROR[i]]);
    }
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
        bk: &GpuBackend,
        g: &Gemm,
        op: &OpAssign,
        act: &Activation,
        sh: &mut TensorLayoutBuffers,
        enc: &mut <GpuBackend as Backend>::Encoder,
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
        bk: &GpuBackend,
        g: &Gemm,
        act: &Activation,
        sh: &mut TensorLayoutBuffers,
        enc: &mut <GpuBackend as Backend>::Encoder,
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
        bk: &GpuBackend,
        ad: &Adam,
        sh: &mut TensorLayoutBuffers,
        enc: &mut <GpuBackend as Backend>::Encoder,
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
    /// Overwrite the GPU weight/bias tensors from a CPU `Mlp` (reverse of
    /// `read_into`). Used by the NET symmetry method to push the re-projected
    /// (equivariant) weights back into the authoritative GPU training copy each
    /// iter, so the constraint actually guides training (the Adam moment tensors
    /// are untouched — the projection is a small correction). Recreating the
    /// tensors is fine: `forward`/`backward` only borrow `w`/`b` per pass.
    fn write_w(&mut self, bk: &GpuBackend, net: &Mlp) {
        let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        for l in 0..self.w.len() {
            let (out, inp) = (self.dims[l + 1], self.dims[l]);
            self.w[l] = mk(bk, &wmat(&net.w[l], out, inp), rw);
            self.b[l] = mk(bk, &DMatrix::from_fn(out, 1, |r, _| net.b[l][r]), rw);
        }
    }
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let ckpt = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "/tmp/biped_policy_gpu.safetensors".to_string());
    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");

    pollster::block_on(async {
        println!("building {n} GPU nexus envs...");
        let mut env = BipedNexusBatchEnv::new(&xml, n, 32, 0xC0FFEE).await;
        let (od, cd) = (env.obs_dim(), env.critic_obs_dim());
        let mut rng = Lcg::new(7);
        let resumed = !ckpt.is_empty() && std::path::Path::new(&ckpt).exists();
        let mut ac = if resumed {
            println!("resuming from {ckpt}...");
            ActorCritic::load(&ckpt).expect("load checkpoint")
        } else {
            ActorCritic::new(
                &[od, 256, 256, 128, NUM_JOINTS],
                &[cd, 512, 256, 128, 1],
                1.0,
                1e-3,
                &mut rng,
            )
        };
        // Symmetry method 4 — NET (BIPED_MIRROR_NET): project the actor+critic
        // weights onto the equivariant subspace so the policy is symmetric BY
        // CONSTRUCTION. Done here (before the GPU nets are built from `ac`) so the
        // rollout policy and the GpuMlp update copies all start equivariant, and
        // re-applied each iter after the GPU→CPU sync (below).
        let mirror_net = std::env::var("BIPED_MIRROR_NET").is_ok();
        if mirror_net {
            println!("equivariant NET ENABLED (weight symmetrization, provably symmetric)");
            symmetrize_ac(&mut ac);
        }
        let bk = env.backend().clone();
        let mut gpu = GpuPolicy::new(&bk, &ac, n).expect("gpu policy");

        // Persistent GPU update state (weights + Adam moments survive all iters).
        let total = n * T;
        // BIPED_MINIBATCHES overrides the minibatch COUNT (default MINIBATCHES=4)
        // — fewer, larger minibatches = fewer kernel launches per epoch at the
        // same total FLOPs (launch-gap vs compute-bound diagnostics / tuning).
        let minibatches: usize = std::env::var("BIPED_MINIBATCHES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(MINIBATCHES);
        let mb = total / minibatches;
        // Mirror augmentation (BIPED_MIRROR_AUG=1): append the L/R mirror of every
        // sample → symmetric policy. To keep the minibatch SIZE `mb` (and the
        // pre-sized GPU buffers) unchanged, we double the minibatch COUNT instead
        // (n_mb below), so a doubled batch just runs 2× minibatches at the same mb.
        let mirror_aug = std::env::var("BIPED_MIRROR_AUG").is_ok();
        if mirror_aug {
            println!("mirror augmentation ENABLED (symmetric policy)");
        }
        // Symmetry method 2 — LOSS (BIPED_MIRROR_LOSS=<weight>, 0=off): add an
        // auxiliary symmetry penalty ½·w·‖μ(s) − mirror(μ(mirror(s)))‖² to the
        // actor loss. Stop-gradient on the mirrored branch (the mirrored output
        // is a target), so it needs only ONE extra actor forward per minibatch
        // and no second backward — and, unlike DUP, doesn't touch the batch size
        // or the KL signal, so it warm-starts cleanly.
        let mirror_loss: f32 = std::env::var("BIPED_MIRROR_LOSS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        if mirror_loss > 0.0 {
            println!("mirror LOSS ENABLED (auxiliary symmetry penalty, weight={mirror_loss})");
        }
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
        let mut lst = mk(&bk, &DMatrix::from_fn(ad_, 1, |r, _| ac.log_std[r]), rw);
        let (mut mls, mut vls) = (
            mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), st),
            mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), st),
        );
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
        // LOSS scratch: a second actor net (weights refreshed each iter as the
        // stop-grad target source), the action mirror as a [ad×ad] signed-perm
        // matrix `pa` (so `pa·μ = jmirror(μ)` via GEMM), per-minibatch scratch
        // `tgt`/`res`, and a constant gradient-scale tensor `gw` = w/mb (matches
        // the PPO grad's own 1/mb averaging).
        let mut s_net = if mirror_loss > 0.0 {
            Some(GpuMlp::new(&bk, &ac.actor, mb))
        } else {
            None
        };
        let pa = mk(
            &bk,
            &DMatrix::from_fn(
                ad_,
                ad_,
                |o, j| if j == JMIRROR[o] { JSIGN[o] } else { 0.0 },
            ),
            st,
        );
        let mut tgt = mk(&bk, &DMatrix::<f32>::zeros(ad_, mb), rw);
        let mut res = mk(&bk, &DMatrix::<f32>::zeros(ad_, mb), rw);
        let gw = mk(
            &bk,
            &DMatrix::<f32>::from_element(ad_, mb, mirror_loss * scale_mb),
            st,
        );
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
        // Defaults are RESUME-AWARE: a warm-started run (checkpoint existed)
        // skips the stand phase and re-ramps the command over the first 20% —
        // re-running the full stand→walk schedule on a resumed policy wastes
        // most of the run and resets the command curriculum out from under a
        // policy that already walks (the historical "KL runaway on resume").
        // Env vars always win.
        let stand_frac: f32 = std::env::var("BIPED_STAND_FRAC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(if resumed { 0.0 } else { 0.3 });
        // Fraction of training by which the velocity command reaches full scale
        // (command ramps 0→1 over [stand_frac, ramp_end]).
        let ramp_end: f32 = std::env::var("BIPED_RAMP_END")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(if resumed { 0.2 } else { 0.7 });
        println!(
            "\n{:>4}  {:>5}  {:>9}  {:>7}  {:>8}  {:>9}  {:>7}  {:>6}",
            "iter", "curr", "step_rew", "falls", "torso_z", "lr", "kl", "sec"
        );

        // Torque-penalty curriculum target (full WBC weight = 1.0). Ramped 0→max
        // over iters 40%→90% so the effort penalty engages only AFTER the policy
        // can stand — at full strength from scratch it fights learning to stand.
        let torque_max = std::env::var("BIPED_TORQUE_MAX")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0);
        // Cap the command scale (BIPED_MAX_CSCALE, default 0.4). The sampler's full
        // range is ±0.5 m/s; the 0.4 cap → max ±0.2 m/s = a SLOW walk, so
        // the policy learns a deliberate low-cadence gait (step → stabilize → step)
        // instead of fast continuous tiny stepping that leans on ankle torque.
        // Slow + quasi-static also transfers far better (no reliance on dynamic
        // contact timing). Set 1.0 for the full ±0.5 m/s range.
        let max_cscale: f32 = std::env::var("BIPED_MAX_CSCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.4);
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
            let (mut rs, mut vs, mut ds): (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<bool>>) = (
                (0..n).map(|_| vec![]).collect(),
                (0..n).map(|_| vec![]).collect(),
                (0..n).map(|_| vec![]).collect(),
            );
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
                    // Time-limit bootstrapping (Pardo et al.): a TIMEOUT is a
                    // truncation, NOT a failure — the episode would have continued,
                    // so bootstrap the value of the final state (`r + γ·V(s_final)`)
                    // instead of treating it as terminal (value 0). A FALL stays a
                    // true termination (no bootstrap). Both still set `done` so GAE
                    // cuts the trajectory at the episode boundary (no bleed into the
                    // post-reset state). Without this, surviving to the 20 s cap was
                    // valued at 0 → the value fn under-valued long-stable-survival,
                    // biasing AGAINST the stability we want (worsens as the policy
                    // improves and more episodes reach timeout). `critic_obs` is the
                    // final (pre-reset) state, since env.step doesn't reset.
                    let r = if outs[e].done && !outs[e].fell {
                        outs[e].reward + GAMMA * ac.value(&outs[e].critic_obs)
                    } else {
                        outs[e].reward
                    };
                    rs[e].push(r);
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
            // Mirror augmentation: append the L/R mirror of every sample. `total`
            // doubles; minibatch size `mb` is unchanged, so `n_mb` (count) doubles.
            if mirror_aug {
                let mir: Vec<Sample> = batch.iter().map(mirror_sample).collect();
                batch.extend(mir);
            }
            let total = batch.len();
            let n_mb = total / mb;
            // Normalize advantages across the batch (mean 0, std 1) — this is what
            // CPU `ActorCritic::update` does; the GPU `Ppo::actor_grad` consumes raw
            // `adv`, so without this the PPO gradients are mis-scaled and the policy
            // plateaus instead of learning.
            let amean: f32 = batch.iter().map(|s| s.adv).sum::<f32>() / total as f32;
            let avar: f32 =
                batch.iter().map(|s| (s.adv - amean).powi(2)).sum::<f32>() / total as f32;
            let asd = avar.sqrt().max(1e-6);
            for s in batch.iter_mut() {
                s.adv = (s.adv - amean) / asd;
            }

            let gae_s = t_gae.elapsed().as_secs_f64();
            // ---------------- GPU PPO UPDATE (persistent nets, advancing Adam) -------
            let t_upd = Instant::now();
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
            // LOSS: normalized MIRRORED obs (normalize ∘ mirror — exact, not
            // mirror ∘ normalize), and refresh the stop-grad target net to the
            // current (iter-start) actor weights.
            let f_obs_mir = if mirror_loss > 0.0 {
                let onm: Vec<Vec<f32>> = batch
                    .iter()
                    .map(|s| ac.obs_norm.normalize(&mirror_obs(&s.obs)))
                    .collect();
                if let Some(sn) = s_net.as_mut() {
                    sn.write_w(&bk, &ac.actor);
                }
                Some(mk(&bk, &DMatrix::from_fn(od, total, |r, c| onm[c][r]), st))
            } else {
                None
            };
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
            let ap = Tensor::scalar(
                &bk,
                PpoActorParams {
                    clip: CLIP,
                    entropy_coef: ENTROPY,
                    scale: scale_mb,
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
                    clip: CLIP,
                    value_coef: VALUE_COEF,
                    scale: scale_mb,
                    num_cols: mb as u32,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                    pad3: 0,
                },
                BufferUsages::UNIFORM,
            )
            .unwrap();

            // Old-policy means for the LAST minibatch — drives the per-epoch KL
            // for the adaptive-KL LR schedule (mirrors CPU `minibatch_step`'s
            // `self.kl`, here at per-epoch rather than per-minibatch granularity).
            let last_off = (n_mb - 1) * mb;
            let mean_old_last: Vec<Vec<f32>> = (0..mb)
                .map(|c| batch[last_off + c].mean_old.clone())
                .collect();
            let mut last_kl = 0.0f32;
            // Update-phase profile: where the PPO update wall-clock goes
            // (encode = CPU command recording, exec = submit+synchronize i.e.
            // GPU execution incl. launch gaps, kl = the per-epoch readback).
            let (mut enc_s, mut exec_s, mut kl_s) = (0.0f64, 0.0f64, 0.0f64);
            for _epoch in 0..EPOCHS {
                gstep += n_mb as u64;
                let bc1 = 1.0 - 0.9f32.powi(gstep.min(1 << 30) as i32);
                let bc2 = 1.0 - 0.999f32.powi(gstep.min(1 << 30) as i32);
                let adp = Tensor::scalar(
                    &bk,
                    AdamParams {
                        lr,
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-8,
                        bias_correction1: bc1,
                        bias_correction2: bc2,
                        pad0: 0.0,
                        pad1: 0.0,
                    },
                    BufferUsages::UNIFORM,
                )
                .unwrap();
                let t_enc = Instant::now();
                let mut enc = bk.begin_encoding();
                for k in 0..n_mb {
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
                    // LOSS: add the symmetry-penalty gradient to the actor output
                    // delta BEFORE backward — `delta[la] += gw·(μ(s) − pa·μ(Ms))`,
                    // with `pa·μ(Ms)` the mirror of the (stop-grad) mirrored-obs
                    // forward. One extra forward + a signed-perm GEMM + 4 op-assigns.
                    if let Some(sn) = s_net.as_mut() {
                        let fom = f_obs_mir.as_ref().unwrap();
                        {
                            let mut p = enc.begin_pass("s_obs", None);
                            cont.launch(
                                &bk,
                                &mut sh,
                                &mut p,
                                &mut sn.a[0],
                                fom.columns(off, nb),
                                None,
                            )
                            .unwrap();
                        }
                        sn.forward(&bk, &g, &op, &act, &mut sh, &mut enc, &o1m)
                            .unwrap();
                        {
                            let mut p = enc.begin_pass("s_tgt", None);
                            g.dispatch_tiled(&bk, &mut sh, &mut p, &mut tgt, &pa, &sn.a[la + 1])
                                .unwrap();
                        }
                        {
                            let mut p = enc.begin_pass("s_cp", None);
                            op.launch(
                                &bk,
                                &mut sh,
                                &mut p,
                                OpAssignVariant::Copy,
                                &mut res,
                                &a_net.a[la + 1],
                            )
                            .unwrap();
                        }
                        {
                            let mut p = enc.begin_pass("s_sub", None);
                            op.launch(&bk, &mut sh, &mut p, OpAssignVariant::Sub, &mut res, &tgt)
                                .unwrap();
                        }
                        {
                            let mut p = enc.begin_pass("s_mul", None);
                            op.launch(&bk, &mut sh, &mut p, OpAssignVariant::Mul, &mut res, &gw)
                                .unwrap();
                        }
                        {
                            let mut p = enc.begin_pass("s_add", None);
                            op.launch(
                                &bk,
                                &mut sh,
                                &mut p,
                                OpAssignVariant::Add,
                                &mut a_net.delta[la],
                                &res,
                            )
                            .unwrap();
                        }
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
                enc_s += t_enc.elapsed().as_secs_f64();
                let t_exec = Instant::now();
                bk.submit(enc).unwrap();
                bk.synchronize().unwrap();
                exec_s += t_exec.elapsed().as_secs_f64();

                // Per-epoch KL (last minibatch) → adaptive-KL LR for the next epoch.
                let t_kl = Instant::now();
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
                kl_s += t_kl.elapsed().as_secs_f64();
                last_kl = kl;
                if kl > DESIRED_KL * 2.0 {
                    lr = (lr / 1.5).max(LR_MIN);
                } else if kl > 0.0 && kl < DESIRED_KL / 2.0 {
                    lr = (lr * 1.5).min(LR_MAX);
                }
                // KL early-stop (rsl_rl / WBC-AGILE): if this iteration's policy has
                // already drifted far past target, stop the remaining epochs so one
                // iteration can't run KL away. Without it the loop ran all EPOCHS
                // regardless, letting per-iter KL blow to ~100 during the walk-command
                // ramp (the policy thrashed instead of refining a gait). `kl` here is
                // current-vs-rollout policy, i.e. cumulative per-iter drift, so this
                // caps per-iter KL at ~5× target.
                if kl > DESIRED_KL * 5.0 {
                    break;
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
            // NET: re-project the freshly-trained weights onto the equivariant
            // subspace, then push them BACK into the GPU training nets (so next
            // iter's update starts equivariant) as well as the rollout policy.
            if mirror_net {
                symmetrize_ac(&mut ac);
                // Push the now-symmetrized log_std + weights back to the GPU update
                // copies so the next iter's update continues from the equivariant state.
                lst = mk(&bk, &DMatrix::from_fn(ad_, 1, |r, _| ac.log_std[r]), rw);
                a_net.write_w(&bk, &ac.actor);
                c_net.write_w(&bk, &ac.critic);
            }
            gpu.sync_weights(&bk, &ac);
            let upd_s = t_upd.elapsed().as_secs_f64();

            if it % 10 == 0 || it == iters - 1 {
                let zs = env.torso_heights().await;
                let torso = zs.iter().sum::<f32>() / n as f32;
                println!(
                    "{:>4}  {:>5.2}  {:>9.4}  {:>7}  {:>8.3}  {:>9.2e}  {:>7.4}  {:>6.1}",
                    it,
                    cscale,
                    total_reward / total as f32,
                    falls,
                    torso,
                    lr,
                    last_kl,
                    t_iter.elapsed().as_secs_f64()
                );
                // [prof] coarse iteration split + rollout per-phase ms/step
                // (env.take_step_timings drains the StepTimings accumulator).
                let st = env.take_step_timings();
                let ns2ms = |x: u64| (x as f64) / (st.steps.max(1) as f64) / 1e6;
                println!(
                    "[prof] roll={:.2}s (reset={:.2}s) gae={:.2}s upd={:.2}s | per-step ms: pipe={:.1} gpuwait={:.1} readback={:.1} reward={:.1} stage={:.1} flush={:.1} commit={:.1}",
                    roll_s,
                    reset_dur.as_secs_f64(),
                    gae_s,
                    upd_s,
                    ns2ms(st.pipeline_step_ns),
                    ns2ms(st.gpu_wait_ns),
                    ns2ms(st.readback_ns),
                    ns2ms(st.par_compute_ns),
                    ns2ms(st.stage_motors_ns),
                    ns2ms(st.flush_static_ns),
                    ns2ms(st.serial_commit_ns),
                );
                // Update-phase split: stage = batch normalize/transpose + H2D
                // upload (everything outside the epoch loop); encode = CPU
                // command recording; exec = submit+synchronize (GPU execution
                // incl. per-launch gaps — the CUDA-graph target); kl = per-epoch
                // KL readback for the adaptive-LR schedule.
                println!(
                    "[prof-upd] stage={:.2}s encode={:.2}s exec={:.2}s kl={:.2}s",
                    upd_s - enc_s - exec_s - kl_s,
                    enc_s,
                    exec_s,
                    kl_s,
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
