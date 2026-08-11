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

#[path = "../biped/biped_env.rs"]
mod biped_env;
#[path = "../biped/biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "../biped/cutile_gemm.rs"]
mod cutile_gemm;
#[path = "../biped/gpu_policy.rs"]
mod gpu_policy;

use biped_env_nexus::{BipedNexusBatchEnv, REWARD_COMP_NAMES, default_mjcf_path};
use cutile_gemm::{CutileGemm, EncCursor};
use gpu_policy::GpuPolicy;
use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend};
use nalgebra::DMatrix;
use std::time::Instant;
use vortx::linalg::{
    Activation, Adam, AdamParams, Contiguous, Gemm, OpAssign, OpAssignVariant, Ppo, PpoActorParams,
    PpoValueParams, Reduce, ReduceVariant,
};
use rayon::prelude::*;
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
const LR_MIN: f32 = 1e-5;
const LR_MAX: f32 = 1e-2;

/// Fused PPO batch staging: normalize (+ optional signed-perm mirror) + clamp
/// straight into the row-major `[dim × total]` upload buffer, rayon over row
/// blocks. Replaces the old per-sample `normalize` allocations plus TWO full
/// transposes (`DMatrix::from_fn` then `matrix_from_na`'s internal one) —
/// that chain was the dominant update-phase cost (~0.7 s/iter at N=4096).
fn stage_norm_flat<'a>(
    dim: usize,
    get: impl Fn(usize) -> &'a [f32] + Sync,
    total: usize,
    norm: &zealot_rl::ppo::Normalizer,
    sperm: Option<&SPerm>,
) -> Vec<f32> {
    let (mean, inv) = norm.affine();
    let mut flat = vec![0f32; dim * total];
    const RB: usize = 8; // row block: col reads stay hot, writes stride `total`
    flat.par_chunks_mut(RB * total)
        .enumerate()
        .for_each(|(bi, chunk)| {
            let r0 = bi * RB;
            let rows = chunk.len() / total;
            for c in 0..total {
                let src = get(c);
                for dr in 0..rows {
                    let r = r0 + dr;
                    // mirror ∘ raw first, then normalize — the transform
                    // `normalize(mirror_obs(x))` (exact, not mirror ∘ normalize).
                    let (j, s) = match sperm {
                        Some(p) => (p.perm[r], p.sign[r]),
                        None => (r, 1.0),
                    };
                    chunk[dr * total + c] =
                        ((src[j] * s - mean[r]) * inv[r]).clamp(-5.0, 5.0);
                }
            }
        });
    flat
}


/// Transpose per-env obs `[n][dim]` into the row-major staging `[dim][n]`,
/// rayon over env blocks: each env's vec is read SEQUENTIALLY (17 cachelines)
/// instead of the row-major gather's one pointer-chase per (d, e) — ~2x on
/// wide obs. Blocks own disjoint column ranges, so writes never share lines.
fn stage_transpose(dst: &mut [f32], src: &[Vec<f32>], dim: usize, n: usize) {
    const EB: usize = 32;
    // SAFETY-free parallelism: wrap dst in per-block raw parts via chunks of
    // the COLUMN index — rayon over env blocks writing strided columns.
    let dst_ptr = SendPtr(dst.as_mut_ptr());
    (0..n.div_ceil(EB)).into_par_iter().for_each(|b| {
        let p = dst_ptr; // move the Send wrapper into the closure
        let e0 = b * EB;
        let e1 = (e0 + EB).min(n);
        for e in e0..e1 {
            let row = &src[e];
            for d in 0..dim {
                // Each block writes columns [e0, e1) only — disjoint.
                unsafe { *p.0.add(d * n + e) = row[d] };
            }
        }
    });
}

/// Raw-pointer Send wrapper for the disjoint-column transpose writes.
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

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
/// BIPED_OBS_HISTORY frame count (1 = feature off) — must match the env's.
static OBS_H: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    // THE single source (zealot_env::knobs) — the env's frame stacking and
    // this network sizing can no longer disagree (v29 lesson).
    zealot_env::knobs::OBS_HISTORY.get().max(1)
});
fn jmirror(v: &[f32]) -> Vec<f32> {
    (0..NUM_JOINTS).map(|i| JSIGN[i] * v[JMIRROR[i]]).collect()
}
// obs frame(48): last_action[0:12], cmd[12:16]=(vx,vy,yaw,aux), joint_pos[16:28],
// joint_vel[28:40], proj_grav[40:43]=(fwd,lat,up), gait_phase[43:45]=(sin,cos),
// base_ang_vel[45:48]=(roll,pitch,yaw).
fn mirror_frame(o: &[f32]) -> Vec<f32> {
    let mut m = o.to_vec();
    m[0..12].copy_from_slice(&jmirror(&o[0..12]));
    m[13] = -o[13];
    m[14] = -o[14];
    m[16..28].copy_from_slice(&jmirror(&o[16..28]));
    m[28..40].copy_from_slice(&jmirror(&o[28..40]));
    m[41] = -o[41];
    m[43] = -o[43];
    m[44] = -o[44];
    // Angular velocity is a PSEUDOVECTOR: under an L/R (sagittal-plane)
    // mirror the roll and yaw components negate while pitch is preserved --
    // the same signs the critic's copy already uses.
    if o.len() > 45 {
        m[45] = -o[45];
        m[47] = -o[47];
    }
    // Step cue (48..53): distance, height and validity are mirror-invariant --
    // an edge 0.4 m ahead and 0.15 m up is the same fact for either chirality.
    // The edge ORIENTATION is not: a step angled to the left mirrors to one
    // angled to the right, so sin negates while cos is preserved, same as any
    // in-plane direction.
    if o.len() > 50 {
        m[50] = -o[50]; // edge_sin
    }
    m
}
// With BIPED_OBS_HISTORY the actor obs is H stacked 45-frames — the mirror is
// block-diagonal (each frame mirrors independently). H=1 = the plain frame.
fn mirror_obs(o: &[f32]) -> Vec<f32> {
    o.chunks(OBS_FRAME).flat_map(|f| mirror_frame(f)).collect()
}
// critic = obs frame(OBS_FRAME) + base_lin_vel(3)[fwd,lat,up] +
// base_ang_vel(3)[roll,pitch,yaw] — single-frame (no history).
fn mirror_critic(c: &[f32]) -> Vec<f32> {
    let mut m = mirror_frame(&c[0..OBS_FRAME]);
    m.extend_from_slice(&c[OBS_FRAME..]);
    // lin_vel: lateral negates. ang_vel: roll and yaw negate (pseudovector).
    m[OBS_FRAME + 1] = -c[OBS_FRAME + 1];
    m[OBS_FRAME + 3] = -c[OBS_FRAME + 3];
    m[OBS_FRAME + 5] = -c[OBS_FRAME + 5];
    // Clean step cue (+5 tail): edge_sin negates, like the actor copy's.
    if c.len() > OBS_FRAME + 8 {
        m[OBS_FRAME + 8] = -c[OBS_FRAME + 8];
    }
    m
}
const OBS_FRAME: usize = zealot_env::tasks::velocity_flat::OBS_DIM;
fn mirror_sample(s: &Sample) -> Sample {
    Sample {
        // Raw obs ride in samples only under BIPED_VERIFY_STAGE (the device
        // staging mirrors via the sperm tables); empty stays empty.
        obs: if s.obs.is_empty() {
            vec![]
        } else {
            mirror_obs(&s.obs)
        },
        critic_obs: if s.critic_obs.is_empty() {
            vec![]
        } else {
            mirror_critic(&s.critic_obs)
        },
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
// obs frame(OBS_FRAME) signed perm — exactly the index/sign pattern of
// `mirror_frame` (which the v28 53-dim frames extended past the old 45).
fn obs_frame_sperm() -> SPerm {
    let d = OBS_FRAME;
    let mut perm: Vec<usize> = (0..d).collect();
    let mut sign = vec![1.0f32; d];
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
    if d > 45 {
        sign[45] = -1.0; // base_ang_vel roll (pseudovector)
        sign[47] = -1.0; // base_ang_vel yaw
    }
    if d > 50 {
        sign[50] = -1.0; // step-cue edge_sin
    }
    SPerm { perm, sign }
}
// Full actor-input signed perm: the frame perm tiled H times (block-diagonal),
// matching `mirror_obs`.
fn obs_sperm() -> SPerm {
    let f = obs_frame_sperm();
    let h = *OBS_H;
    let mut perm = Vec::with_capacity(OBS_FRAME * h);
    let mut sign = Vec::with_capacity(OBS_FRAME * h);
    for b in 0..h {
        perm.extend(f.perm.iter().map(|&p| p + OBS_FRAME * b));
        sign.extend_from_slice(&f.sign);
    }
    SPerm { perm, sign }
}
// critic signed perm = obs frame(OBS_FRAME) + [lin_vel(3), ang_vel(3)]
// (+ optional clean step-cue tail) per `mirror_critic`.
fn critic_sperm(cd: usize) -> SPerm {
    let mut sp = obs_frame_sperm();
    sp.perm.extend(OBS_FRAME..cd);
    sp.sign.extend(std::iter::repeat(1.0).take(cd - OBS_FRAME));
    sp.sign[OBS_FRAME + 1] = -1.0; // lin_vel lateral
    sp.sign[OBS_FRAME + 3] = -1.0; // ang_vel roll
    sp.sign[OBS_FRAME + 5] = -1.0; // ang_vel yaw
    if cd > OBS_FRAME + 8 {
        sp.sign[OBS_FRAME + 8] = -1.0; // clean step-cue edge_sin
    }
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
    let mut r = vec![critic_sperm(net.dims[0])];
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
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
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
        cur: &mut EncCursor,
        ct: Option<&CutileGemm>,
        o1m: &Tensor<f32>,
    ) -> anyhow::Result<()> {
        let l = self.layers();
        for i in 0..l {
            let (lf, rt) = self.a.split_at_mut(i + 1);
            let (ain, aout) = (&lf[i], &mut rt[0]);
            // z = act(W[i]·a[i] + b[i]) — ONE fused tf32 cuTile launch when
            // enabled (gemm + bias broadcast + ELU epilogue); else the vortx
            // gemm / bias-GEMV / add / ELU pass chain.
            if let Some(ct) = ct {
                cur.flush(); // pending khal work must hit the stream first
                ct.gemm_bias_act(
                    aout,
                    &self.w[i],
                    ain,
                    self.dims[i + 1],
                    self.batch,
                    self.dims[i],
                    &self.b[i],
                    1,
                    i < l - 1,
                )?;
            } else {
                {
                    let mut p = cur.pass("z");
                    g.dispatch_tiled(bk, sh, &mut p, &mut *aout, &self.w[i], ain)?;
                }
                {
                    let mut p = cur.pass("bb");
                    g.dispatch_naive(bk, sh, &mut p, &mut self.bb[i], &self.b[i], o1m)?;
                }
                {
                    let mut p = cur.pass("bias");
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
                    let mut p = cur.pass("elu");
                    act.elu(bk, sh, &mut p, &mut *aout)?;
                }
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
        cur: &mut EncCursor,
        ct: Option<&CutileGemm>,
        om1: &Tensor<f32>,
    ) -> anyhow::Result<()> {
        for i in (0..self.layers()).rev() {
            // dW[i] = delta[i] · a[i]ᵀ — the deep-K (K = batch) wgrad, where
            // the cuTile split-K path matters most.
            if let Some(ct) = ct {
                cur.flush();
                ct.gemm(
                    &self.dw[i],
                    &self.delta[i],
                    false,
                    &self.a[i],
                    true,
                    self.dims[i + 1],
                    self.dims[i],
                    self.batch,
                )?;
            } else {
                let mut p = cur.pass("dw");
                g.dispatch_tiled(
                    bk,
                    sh,
                    &mut p,
                    &mut self.dw[i],
                    &self.delta[i],
                    self.a[i].transpose_last_dims(),
                )?;
            }
            // db = row-sums of delta (the vortx GEMV ran ~100x below bandwidth).
            if let Some(ct) = ct {
                cur.flush();
                ct.row_sum(&self.db[i], &self.delta[i], self.dims[i + 1], self.batch)?;
            } else {
                let mut p = cur.pass("db");
                g.dispatch_naive(bk, sh, &mut p, &mut self.db[i], &self.delta[i], om1)?;
            }
            if i > 0 {
                {
                    let (lf, rt) = self.delta.split_at_mut(i);
                    let dp = &mut lf[i - 1];
                    let dc = &rt[0];
                    // delta[i-1] = W[i]ᵀ · delta[i] (dgrad).
                    if let Some(ct) = ct {
                        cur.flush();
                        ct.gemm(
                            dp,
                            &self.w[i],
                            true,
                            dc,
                            false,
                            self.dims[i],
                            self.batch,
                            self.dims[i + 1],
                        )?;
                    } else {
                        let mut p = cur.pass("da");
                        g.dispatch_tiled(bk, sh, &mut p, dp, self.w[i].transpose_last_dims(), dc)?;
                    }
                }
                if let Some(ct) = ct {
                    cur.flush();
                    ct.elu_backward(&self.delta[i - 1], &self.a[i], self.dims[i], self.batch)?;
                } else {
                    let mut p = cur.pass("eb");
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
        cur: &mut EncCursor,
        ap: &Tensor<AdamParams>,
    ) -> anyhow::Result<()> {
        for i in 0..self.layers() {
            {
                let mut p = cur.pass("aw");
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
                let mut p = cur.pass("ab");
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
        let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
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
            let ac = ActorCritic::load(&ckpt).expect("load checkpoint");
            // BIPED_OBS_HISTORY changes the actor input dim — a stale checkpoint
            // must fail loudly here, not as a silent normalizer truncation.
            assert_eq!(
                ac.actor.dims[0], od,
                "checkpoint actor input dim {} != env obs dim {} — \
                 BIPED_OBS_HISTORY mismatch? delete {ckpt} or match the setting",
                ac.actor.dims[0], od
            );
            assert_eq!(
                ac.critic.dims[0], cd,
                "checkpoint critic input dim {} != env critic obs dim {} — delete {ckpt}",
                ac.critic.dims[0], cd
            );
            ac
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

        // cuTile tf32 tensor-core GEMMs for the update AND the rollout policy
        // forward (BIPED_CUTILE_GEMM=1; needs --features cutile). Self-tests
        // against a CPU reference at init; None → the unchanged vortx path.
        let ct: Option<&'static CutileGemm> = CutileGemm::init(&bk).await;
        gpu.set_cutile(ct);

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
        let mirror_aug = std::env::var("BIPED_MIRROR_AUG").map_or(true, |v| v != "0");
        if mirror_aug {
            println!("mirror augmentation ENABLED (symmetric policy)");
        }
        // Print the obs geometry so a config-echo comparison catches any
        // trainer/env disagreement (the v29 single-frame bug hid because
        // nothing echoed the dims).
        println!(
            "actor obs: {} = {} frames x {} dims (BIPED_OBS_HISTORY)",
            OBS_FRAME * *OBS_H,
            *OBS_H,
            OBS_FRAME
        );
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
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        // ---- Device PPO-batch staging ----
        // The raw obs cross PCIe ONCE, step-blocked, during the rollout; the
        // update builds its `[dim × total]` batch on device with
        // `gpu_ppo_stage_batch` (normalize + ±5 clamp + signed-perm mirror in
        // one dispatch per half) instead of re-normalizing, transposing and
        // uploading ~100 MB of host batch every iteration. Mirror arrives as
        // the explicit `obs_sperm`/`critic_sperm` tables, so it cannot drift
        // from `mirror_obs`/`mirror_critic`. Verify any change with
        // BIPED_VERIFY_STAGE=1 (element-wise device-vs-host compare).
        let total_cols = if mirror_aug { 2 * T * n } else { T * n };
        env.raw_obs_t =
            Some(Tensor::<f32>::vector_uninit(&bk, ((T + 1) * od * n) as u32, rw).unwrap());
        env.raw_cobs_t =
            Some(Tensor::<f32>::vector_uninit(&bk, ((T + 1) * cd * n) as u32, rw).unwrap());
        let mut f_obs_dev =
            Tensor::<f32>::matrix_uninit(&bk, od as u32, total_cols as u32, rw).unwrap();
        let mut f_cobs_dev =
            Tensor::<f32>::matrix_uninit(&bk, cd as u32, total_cols as u32, rw).unwrap();
        let mut f_obs_mir_dev = if mirror_loss > 0.0 {
            Some(Tensor::<f32>::matrix_uninit(&bk, od as u32, total_cols as u32, rw).unwrap())
        } else {
            None
        };
        let sp_o = obs_sperm();
        let sp_c = critic_sperm(cd);
        let to_u32 = |p: &[usize]| p.iter().map(|&x| x as u32).collect::<Vec<u32>>();
        let perm_id_o = Tensor::vector(&bk, &(0..od as u32).collect::<Vec<u32>>(), st).unwrap();
        let perm_mir_o = Tensor::vector(&bk, &to_u32(&sp_o.perm), st).unwrap();
        let sign_id_o = Tensor::vector(&bk, &vec![1.0f32; od], st).unwrap();
        let sign_mir_o = Tensor::vector(&bk, &sp_o.sign, st).unwrap();
        let perm_id_c = Tensor::vector(&bk, &(0..cd as u32).collect::<Vec<u32>>(), st).unwrap();
        let perm_mir_c = Tensor::vector(&bk, &to_u32(&sp_c.perm), st).unwrap();
        let sign_id_c = Tensor::vector(&bk, &vec![1.0f32; cd], st).unwrap();
        let sign_mir_c = Tensor::vector(&bk, &sp_c.sign, st).unwrap();
        let mut raw_stage = vec![0.0f32; od.max(cd) * n];
        let mut gauss_buf: Vec<f32> = Vec::new();
        let verify_stage = std::env::var("BIPED_VERIFY_STAGE").is_ok();
        // Per-step params for the device-side ROLLOUT staging (step_select
        // picks the raw-arena step; the policy input is [dim × n], stride n).
        // Constant across iterations — built once.
        let stage_prm = |dim: usize, cols: usize, off: u32, step_sel: u32| {
            Tensor::scalar(
                &bk,
                vortx::linalg::PpoStageParams {
                    dim: dim as u32,
                    n: n as u32,
                    steps: T as u32,
                    total_cols: cols as u32,
                    col_offset: off,
                    step_select: step_sel,
                    pad1: 0,
                    pad2: 0,
                },
                BufferUsages::UNIFORM,
            )
            .unwrap()
        };
        let prm_roll_o: Vec<Tensor<vortx::linalg::PpoStageParams>> =
            (0..=T).map(|t| stage_prm(od, n, 0, t as u32 + 1)).collect();
        let prm_roll_c: Vec<Tensor<vortx::linalg::PpoStageParams>> =
            (0..=T).map(|t| stage_prm(cd, n, 0, t as u32 + 1)).collect();
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
        // Best-checkpoint tracking. The adaptive-KL controller can oscillate a
        // CONVERGED policy off its peak late in training (reward drifts down,
        // terrain curriculum collapses), and the periodic `ckpt` save keeps only
        // the LATEST (possibly degraded) weights. Track a smoothed reward and
        // save the peak policy separately to `<ckpt>.best` — that's the one to
        // deploy; overtraining then can't cost us the good model.
        let mut rew_ema = 0.0f32;
        let mut best_ema = f32::NEG_INFINITY;
        // LR floor override (BIPED_LR_MIN). rsl_rl's adaptive-KL schedule brakes
        // down to 1e-5; our 1e-4 default was tuned for the shaped reward and is
        // too high a floor for spikier reward sets (KL runs away when the
        // controller wants to brake further and can't).
        let lr_min: f32 = std::env::var("BIPED_LR_MIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(LR_MIN);
        // BIPED_NORM_FREEZE=1: freeze the obs normalizers for each whole
        // iteration (rollout forwards, mean_old/logp_old, GAE, staged update
        // obs and the mirror map all share ONE transform; new stats merge at
        // the iteration boundary). Without it the update re-normalizes with
        // stats that drifted since the rollout forwards, corrupting the PPO
        // ratios — measured as a pre-update KL floor (BIPED_KL_PROBE) that
        // grows whenever obs statistics move fast.
        // Global grad-norm clip bound (BIPED_GRAD_CLIP, rsl_rl max_grad_norm
        // semantics; 0/absent = off, rsl_rl uses 1.0).
        let grad_clip: f32 = std::env::var("BIPED_GRAD_CLIP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        if grad_clip > 0.0 {
            println!("grad-norm clip ENABLED: global max_grad_norm = {grad_clip} (rsl_rl parity)");
        }
        // Grad-clip machinery: GPU-side squared norms (one SqNorm reduce per
        // grad tensor into a slot of `sqn`, single tiny readback per
        // minibatch) + per-shape scale tensors for the (rare) clip events.
        // Tensor storage is ROW-major: 64 floats per slot row = 256 bytes, so
        // every rows_mut(slot, 1) view starts at a 256-byte-aligned buffer
        // offset — WebGPU/Metal bind groups reject unaligned storage offsets
        // (Vulkan wants 256, Metal 32); the old 1x32 layout bound slot k at
        // offset 4k and panicked on any non-CUDA backend (grad clip was
        // CUDA-only until the defaults change).
        const SQN_STRIDE: usize = 64;
        let red = Reduce::from_backend(&bk).unwrap();
        let mut sqn = mk(&bk, &DMatrix::<f32>::zeros(32, SQN_STRIDE), rw);
        let (mut gc_scales, mut gc_lens): (Vec<Tensor<f32>>, Vec<usize>) = (vec![], vec![]);
        let mut gc_slots = 0usize;
        if grad_clip > 0.0 {
            for net in [&a_net, &c_net] {
                for i in 0..net.layers() {
                    let (r, c) = (net.dims[i + 1], net.dims[i]);
                    gc_scales.push(mk(&bk, &DMatrix::<f32>::zeros(r, c), st));
                    gc_lens.push(r * c);
                    gc_slots += 1;
                }
                for i in 0..net.layers() {
                    let r = net.dims[i + 1];
                    gc_scales.push(mk(&bk, &DMatrix::<f32>::zeros(r, 1), st));
                    gc_lens.push(r);
                    gc_slots += 1;
                }
            }
            gc_scales.push(mk(&bk, &DMatrix::<f32>::zeros(ad_, 1), st));
            gc_lens.push(ad_);
            gc_slots += 1;
        }
        let norm_freeze = std::env::var("BIPED_NORM_FREEZE").map_or(true, |v| v != "0");
        if norm_freeze {
            println!("obs-normalizer FREEZE enabled: per-iteration stats snapshot (exact PPO ratios)");
        }
        let mut pn_obs = zealot_rl::ppo::PendingNorm::default();
        let mut pn_cobs = zealot_rl::ppo::PendingNorm::default();
        // BIPED_GPU_OBS: obs live on device end-to-end — no host mirrors (gc/
        // gcc frozen at initial values, unused), no per-step slot uploads, no
        // host welford. Requires the frozen-normalizer schedule.
        let gpu_obs = env.gpu_obs_on();
        let gpu_obs_verify = gpu_obs && std::env::var("BIPED_GPU_OBS_VERIFY").is_ok();
        // Debug bisect: device obs pipeline still runs, but the forward reads
        // HOST-uploaded slots (the known-good path). Requires verify (live gc).
        let gpu_obs_hostfeed = gpu_obs_verify && std::env::var("BIPED_GPU_OBS_HOSTFEED").is_ok();
        let verify_stage = verify_stage && !gpu_obs;
        if gpu_obs {
            assert!(norm_freeze, "BIPED_GPU_OBS requires BIPED_NORM_FREEZE");
        }

        let (mut gc, mut gcc) = env.initial_obs().await;
        if gpu_obs {
            // One-time seed: arena slot 0 = the initial obs (host-assembled).
            // Every later slot is device-written; iteration boundaries reuse
            // slot T via `copy_bootstrap_to_slot0`.
            let mut tmp = vec![0.0f32; od.max(cd) * n];
            stage_transpose(&mut tmp[..od * n], &gc, od, n);
            bk.write_buffer(env.raw_obs_t.as_mut().unwrap().buffer_mut(), 0, &tmp[..od * n])
                .unwrap();
            env.seed_initial_stack(&tmp[..od * n]);
            stage_transpose(&mut tmp[..cd * n], &gcc, cd, n);
            bk.write_buffer(env.raw_cobs_t.as_mut().unwrap().buffer_mut(), 0, &tmp[..cd * n])
                .unwrap();
        }
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
        // BIPED_TERRAIN: AGILE-consistency — the terrain curriculum IS the
        // difficulty control (no stand phase, no command ramp; full ±0.5 m/s
        // from iter 0). Note: episodes under a STANDING command travel < 2 m
        // and count as curriculum failures (AGILE has the same coupling);
        // consider BIPED_STAND_PROB=0 for terrain runs.
        let terrain_on = zealot_env::knobs::TERRAIN.get();
        if terrain_on {
            println!(
                "terrain curriculum drives difficulty: command scale pinned to 1.0 \
                 (stand/ramp curricula bypassed)"
            );
        }
        for it in 0..iters {
            // BIPED_GPU_OBS: slot 0 of this rollout = last iteration's final
            // state (slot T). Deferred to HERE — after the update — because
            // the update's t=0 batch columns still read the OLD slot 0.
            if gpu_obs && it > 0 {
                env.copy_bootstrap_to_slot0();
            }
            let t_iter = Instant::now();
            let frac = it as f32 / iters as f32;
            let cscale = if terrain_on {
                1.0
            } else if frac < stand_frac {
                0.0
            } else {
                ((frac - stand_frac) / (ramp_end - stand_frac)).clamp(0.0, 1.0) * max_cscale
            };
            env.set_command_scale(cscale);
            let tscale = ((it as f32 / iters as f32 - 0.4) / 0.5).clamp(0.0, 1.0) * torque_max;
            env.set_torque_scale(tscale);
            // Annealed ACTOR cue dropout (asymmetric AC exploration schedule):
            // start near-blind (0.9) so the policy practices crossing steps
            // before it can learn to fear them -- measured trap: with the cue
            // visible from iter 0, "cue -> stop" appears within ~1k iters
            // because on step terrain the cue is a fall predictor, and a
            // policy that stops never collects the crossing experience that
            // would flip the value estimate. The CRITIC sees the clean cue
            // throughout, so blind crossings are still credited correctly.
            // Linear 0.9 -> 0.1 over the first 30% of the run, then the
            // deploy-level 0.1 (matching BIPED_STEP_CUE_DROPOUT's default).
            let cue_hi: f32 = std::env::var("BIPED_STEP_CUE_DROPOUT_INIT")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(0.9);
            let cue_lo: f32 = std::env::var("BIPED_STEP_CUE_DROPOUT")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(0.10);
            let cue_frac = (frac / 0.3).clamp(0.0, 1.0);
            env.set_step_cue_dropout(cue_hi + (cue_lo - cue_hi) * cue_frac);

            // ---------------- ROLLOUT (GPU policy forward, host sample) ----------------
            let mut samp: Vec<Vec<Sample>> = (0..n).map(|_| Vec::with_capacity(T)).collect();
            let (mut rs, mut vs, mut ds): (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<bool>>) = (
                (0..n).map(|_| vec![]).collect(),
                (0..n).map(|_| vec![]).collect(),
                (0..n).map(|_| vec![]).collect(),
            );
            // Frozen-normalizer affine for the device-side rollout staging
            // (stats as of iteration start — exactly what the host `normalize`
            // sees under BIPED_NORM_FREEZE).
            let (rmo, rio) = ac.obs_norm.affine();
            let (rmc, ric) = ac.critic_norm.affine();
            let roll_mean_o = Tensor::vector(&bk, &rmo, st).unwrap();
            let roll_inv_o = Tensor::vector(&bk, &rio, st).unwrap();
            let roll_mean_c = Tensor::vector(&bk, &rmc, st).unwrap();
            let roll_inv_c = Tensor::vector(&bk, &ric, st).unwrap();
            let (mut total_reward, mut falls) = (0.0f32, 0u32);
            let t_roll = Instant::now();
            let mut reset_dur = std::time::Duration::ZERO;
            // [prof-roll]: where the rollout wall-clock goes, per phase.
            let (mut pr_norm, mut pr_fwd, mut pr_samp, mut pr_step, mut pr_post) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for t_step in 0..T {
                let tp = Instant::now();
                if gpu_obs_hostfeed {
                    stage_transpose(&mut raw_stage[..od * n], &gc, od, n);
                    bk.write_buffer(
                        env.raw_obs_t.as_mut().unwrap().buffer_mut(),
                        (t_step * od * n) as u64,
                        &raw_stage[..od * n],
                    )
                    .unwrap();
                    stage_transpose(&mut raw_stage[..cd * n], &gcc, cd, n);
                    bk.write_buffer(
                        env.raw_cobs_t.as_mut().unwrap().buffer_mut(),
                        (t_step * cd * n) as u64,
                        &raw_stage[..cd * n],
                    )
                    .unwrap();
                }
                if !gpu_obs {
                    for e in 0..n {
                        if norm_freeze {
                            pn_obs.push(&gc[e]);
                            pn_cobs.push(&gcc[e]);
                        } else {
                            ac.record_obs(&gc[e], &gcc[e]);
                        }
                    }
                    // Stage this step's RAW obs into the device rollout arenas
                    // (step-blocked [T][dim][n]) — the update's staging kernel
                    // reads them in place; the batch never re-crosses PCIe.
                    // (BIPED_GPU_OBS: the env already wrote this slot at the
                    // previous step's finalize; welford runs there too.)
                    stage_transpose(&mut raw_stage[..od * n], &gc, od, n);
                    bk.write_buffer(
                        env.raw_obs_t.as_mut().unwrap().buffer_mut(),
                        (t_step * od * n) as u64,
                        &raw_stage[..od * n],
                    )
                    .unwrap();
                    stage_transpose(&mut raw_stage[..cd * n], &gcc, cd, n);
                    bk.write_buffer(
                        env.raw_cobs_t.as_mut().unwrap().buffer_mut(),
                        (t_step * cd * n) as u64,
                        &raw_stage[..cd * n],
                    )
                    .unwrap();
                }
                pr_norm += tp.elapsed().as_secs_f64();
                let tp = Instant::now();
                // Device-normalized forward: stage this step's raw-arena slice
                // straight into the net inputs (identity perm, frozen affine)
                // — no host normalize/transpose/upload. Falls back to the
                // host path when the normalizer isn't frozen (its stats then
                // evolve WITHIN the rollout, which a per-iter affine can't
                // represent).
                let (means, values) = if norm_freeze {
                    // Staging + both MLP forwards on ONE cursor → one submit.
                    let mut cur = EncCursor::new(&bk);
                    {
                        let mut pass = cur.pass("policy_stage");
                        ppo.stage_batch(&mut pass, &prm_roll_o[t_step], env.raw_obs_t.as_ref().unwrap(), &roll_mean_o, &roll_inv_o, &perm_id_o, &sign_id_o, gpu.actor_input_mut(), n as u32, od as u32).unwrap();
                        ppo.stage_batch(&mut pass, &prm_roll_c[t_step], env.raw_cobs_t.as_ref().unwrap(), &roll_mean_c, &roll_inv_c, &perm_id_c, &sign_id_c, gpu.critic_input_mut(), n as u32, cd as u32).unwrap();
                    }
                    gpu.encode_actor(&bk, &mut cur).unwrap();
                    gpu.encode_critic(&bk, &mut cur).unwrap();
                    cur.flush();
                    bk.synchronize().unwrap();
                    gpu.read_forward_outputs(&bk).await.unwrap()
                } else {
                    gpu.forward(&bk, &ac, &gc, &gcc).await.unwrap()
                };
                pr_fwd += tp.elapsed().as_secs_f64();
                let tp = Instant::now();
                // Phase A (serial, tight): every gauss draw in the EXACT
                // per-env order of the old loop — the shared-LCG sequence is
                // training-trajectory-affecting, so it must not change.
                // Phase B (parallel): the per-env math + allocations, which
                // dominated this phase's wall-clock.
                if gauss_buf.len() != n * NUM_JOINTS {
                    gauss_buf.resize(n * NUM_JOINTS, 0.0);
                }
                for g in gauss_buf.iter_mut() {
                    *g = rng.gauss();
                }
                let std_k: [f32; NUM_JOINTS] =
                    std::array::from_fn(|k| ac.log_std[k].exp());
                let mut acts = vec![[0.0f32; NUM_JOINTS]; n];
                acts.par_iter_mut()
                    .zip(samp.par_iter_mut())
                    .enumerate()
                    .for_each(|(e, (act, se))| {
                        let mean = means[e];
                        let mut a = vec![0.0f32; NUM_JOINTS];
                        for k in 0..NUM_JOINTS {
                            a[k] = mean[k] + std_k[k] * gauss_buf[e * NUM_JOINTS + k];
                        }
                        let lp = ac.logp(&a, &mean[..]);
                        *act = to_action(&a);
                        se.push(Sample {
                            // The staging kernel reads obs from the device
                            // arenas; host copies only when the parity check
                            // needs them.
                            obs: if verify_stage { gc[e].clone() } else { vec![] },
                            critic_obs: if verify_stage { gcc[e].clone() } else { vec![] },
                            action: a,
                            mean_old: mean.to_vec(),
                            logp_old: lp,
                            value_old: values[e],
                            adv: 0.0,
                            ret: 0.0,
                        });
                    });
                for e in 0..n {
                    vs[e].push(values[e]);
                }
                pr_samp += tp.elapsed().as_secs_f64();
                let tp = Instant::now();
                let mut outs = env.step(&acts).await;
                pr_step += tp.elapsed().as_secs_f64();
                let tp = Instant::now();
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
                    if (!gpu_obs || gpu_obs_verify) && !outs[e].done {
                        // MOVE, don't clone — `outs` is dropped at the end
                        // of the step, and the bootstrap read of
                        // `outs[e].critic_obs` already happened above.
                        gc[e] = std::mem::take(&mut outs[e].obs);
                        gcc[e] = std::mem::take(&mut outs[e].critic_obs);
                    }
                }
                // Batched episode resets: one GPU submit for every done env
                // (the sequential per-env path cost seconds/iter at
                // early-training fall rates).
                let to_reset: Vec<usize> = (0..n).filter(|&e| outs[e].done).collect();
                if !to_reset.is_empty() {
                    let tr = Instant::now();
                    let fresh = env.reset_envs(&to_reset).await;
                    if !gpu_obs || gpu_obs_verify {
                        for (&e, (o, c)) in to_reset.iter().zip(fresh) {
                            gc[e] = o;
                            gcc[e] = c;
                        }
                    }
                    reset_dur += tr.elapsed();
                }
                // BIPED_GPU_OBS: finish this step's obs into arena slot t+1
                // AFTER resets so the template-frame column scatter lands
                // before stacking (slot T at t=T-1 doubles as the bootstrap).
                if gpu_obs {
                    env.finalize_obs_slot(t_step + 1);
                }
                if gpu_obs_verify {
                    let (dobs, dcobs) = env.debug_read_obs().await;
                    let (mut wo, mut wc) = (0.0f32, 0.0f32);
                    let (mut wo_d, mut wc_d, mut wo_e) = (0usize, 0usize, 0usize);
                    for e in 0..n {
                        for d in 0..od {
                            let x = (dobs[d * n + e] - gc[e][d]).abs();
                            if x > wo {
                                wo = x;
                                wo_d = d;
                                wo_e = e;
                            }
                        }
                        for d in 0..cd {
                            let x = (dcobs[d * n + e] - gcc[e][d]).abs();
                            if x > wc {
                                wc = x;
                                wc_d = d;
                            }
                        }
                    }
                    // Per-history-frame breakdown for the worst env + values.
                    let f = od / 5;
                    let mut per_frame = [0.0f32; 5];
                    for d in 0..od {
                        let x = (dobs[d * n + wo_e] - gc[wo_e][d]).abs();
                        let fi = d / f;
                        per_frame[fi] = per_frame[fi].max(x);
                    }
                    eprintln!(
                        "[verify_gpu_obs] t={t_step} obs={wo:.2e} (d{wo_d}={}|{} e{wo_e} slot{} frames {:.1e}/{:.1e}/{:.1e}/{:.1e}/{:.1e}) cobs={wc:.2e} (d{wc_d}) resets={} was_reset={}",
                        gc[wo_e][wo_d],
                        dobs[wo_d * n + wo_e],
                        wo_d % f,
                        per_frame[0], per_frame[1], per_frame[2], per_frame[3], per_frame[4],
                        to_reset.len(),
                        to_reset.contains(&wo_e),
                    );
                }
                pr_post += tp.elapsed().as_secs_f64();
            }
            if std::env::var("BIPED_ROLL_PROF").is_ok() {
                println!(
                    "[prof-roll] norm={pr_norm:.2}s forward={pr_fwd:.2}s sample={pr_samp:.2}s step={pr_step:.2}s post={pr_post:.2}s"
                );
            }

            let roll_s = t_roll.elapsed().as_secs_f64();
            // ---------------- GAE + batch ----------------
            let t_gae = Instant::now();
            // Bootstrap values V(s_T) for ALL envs in ONE device forward:
            // stage the final post-rollout states into arena slot T and run
            // the prestaged forward (the actor half rides along, ~free).
            // Replaces 4096 per-env HOST critic forwards (~0.08 s/iter) —
            // and is MORE consistent, since every other value in the GAE
            // stream (`vs`) already came from this same GPU path.
            let boot_vals: Vec<f32> = if norm_freeze {
                if !gpu_obs {
                    // (BIPED_GPU_OBS: slot T was device-written at t=T-1.)
                    stage_transpose(&mut raw_stage[..od * n], &gc, od, n);
                    bk.write_buffer(env.raw_obs_t.as_mut().unwrap().buffer_mut(), (T * od * n) as u64, &raw_stage[..od * n])
                        .unwrap();
                    stage_transpose(&mut raw_stage[..cd * n], &gcc, cd, n);
                    bk.write_buffer(env.raw_cobs_t.as_mut().unwrap().buffer_mut(), (T * cd * n) as u64, &raw_stage[..cd * n])
                        .unwrap();
                }
                let mut cur = EncCursor::new(&bk);
                {
                    let mut pass = cur.pass("bootstrap_stage");
                    ppo.stage_batch(&mut pass, &prm_roll_o[T], env.raw_obs_t.as_ref().unwrap(), &roll_mean_o, &roll_inv_o, &perm_id_o, &sign_id_o, gpu.actor_input_mut(), n as u32, od as u32).unwrap();
                    ppo.stage_batch(&mut pass, &prm_roll_c[T], env.raw_cobs_t.as_ref().unwrap(), &roll_mean_c, &roll_inv_c, &perm_id_c, &sign_id_c, gpu.critic_input_mut(), n as u32, cd as u32).unwrap();
                }
                gpu.encode_actor(&bk, &mut cur).unwrap();
                gpu.encode_critic(&bk, &mut cur).unwrap();
                cur.flush();
                bk.synchronize().unwrap();
                gpu.read_forward_outputs(&bk).await.unwrap().1
            } else {
                (0..n).map(|e| ac.value(&gcc[e])).collect()
            };
            // Per-env GAE is independent across envs; run in parallel, then
            // flatten in env order so the batch is unchanged.
            samp.par_iter_mut().enumerate().for_each(|(e, se)| {
                let lv = boot_vals[e];
                let (adv, retn) = gae(&rs[e], &vs[e], &ds[e], lv, GAMMA, LAM);
                for t in 0..T {
                    se[t].adv = adv[t];
                    se[t].ret = retn[t];
                }
            });
            let mut batch: Vec<Sample> = Vec::with_capacity(total);
            for se in samp.iter_mut() {
                for t in 0..T {
                    batch.push(std::mem::take(&mut se[t]));
                }
            }
            // Mirror augmentation: add the L/R mirror of every sample. `total`
            // doubles; minibatch size `mb` is unchanged, so `n_mb` (count) doubles.
            // Mirrored copies go FIRST, originals LAST: the adaptive-KL LR
            // schedule measures drift on the LAST minibatch, and a mirrored
            // sample's `mean_old` is the mirror of the original's mean — for a
            // non-equivariant policy that differs from the network's actual
            // output on the mirrored obs, so measuring KL on mirrored samples
            // reads policy ASYMMETRY as update drift (verified with
            // BIPED_KL_PROBE: pre-update KL ~0.02 and growing on mirrored
            // tails, exactly 0 on originals). That artifact pinned lr at the
            // floor and early-stopped every epoch under fast-changing rewards.
            if mirror_aug {
                let mut mir: Vec<Sample> = batch.par_iter().map(mirror_sample).collect();
                std::mem::swap(&mut batch, &mut mir);
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
            // Device batch staging: normalize + clamp + mirror straight from
            // the step-blocked raw rollout arenas into the persistent
            // [dim × total] tensors — one dispatch per half. Batch layout is
            // [mirrored; originals] (mirror_aug) so the first half uses the
            // sperm tables and the second the identity; the mirror-loss
            // tensor is the same staging with the halves' tables swapped
            // (mirror ∘ mirror = id).
            {
                debug_assert_eq!(total, total_cols);
                let (mo, io) = ac.obs_norm.affine();
                let (mc, ic) = ac.critic_norm.affine();
                let mean_o = Tensor::vector(&bk, &mo, st).unwrap();
                let inv_o = Tensor::vector(&bk, &io, st).unwrap();
                let mean_c = Tensor::vector(&bk, &mc, st).unwrap();
                let inv_c = Tensor::vector(&bk, &ic, st).unwrap();
                let half = (T * n) as u32;
                let prm = |dim: usize, off: u32| {
                    Tensor::scalar(
                        &bk,
                        vortx::linalg::PpoStageParams {
                            dim: dim as u32,
                            n: n as u32,
                            steps: T as u32,
                            total_cols: total_cols as u32,
                            col_offset: off,
                            step_select: 0,
                            pad1: 0,
                            pad2: 0,
                        },
                        BufferUsages::UNIFORM,
                    )
                    .unwrap()
                };
                let mut enc = bk.begin_encoding();
                {
                    let mut pass = enc.begin_pass("ppo_stage_batch", None);
                    if mirror_aug {
                        ppo.stage_batch(&mut pass, &prm(od, 0), env.raw_obs_t.as_ref().unwrap(), &mean_o, &inv_o, &perm_mir_o, &sign_mir_o, &mut f_obs_dev, half, od as u32).unwrap();
                        ppo.stage_batch(&mut pass, &prm(od, half), env.raw_obs_t.as_ref().unwrap(), &mean_o, &inv_o, &perm_id_o, &sign_id_o, &mut f_obs_dev, half, od as u32).unwrap();
                        ppo.stage_batch(&mut pass, &prm(cd, 0), env.raw_cobs_t.as_ref().unwrap(), &mean_c, &inv_c, &perm_mir_c, &sign_mir_c, &mut f_cobs_dev, half, cd as u32).unwrap();
                        ppo.stage_batch(&mut pass, &prm(cd, half), env.raw_cobs_t.as_ref().unwrap(), &mean_c, &inv_c, &perm_id_c, &sign_id_c, &mut f_cobs_dev, half, cd as u32).unwrap();
                    } else {
                        ppo.stage_batch(&mut pass, &prm(od, 0), env.raw_obs_t.as_ref().unwrap(), &mean_o, &inv_o, &perm_id_o, &sign_id_o, &mut f_obs_dev, half, od as u32).unwrap();
                        ppo.stage_batch(&mut pass, &prm(cd, 0), env.raw_cobs_t.as_ref().unwrap(), &mean_c, &inv_c, &perm_id_c, &sign_id_c, &mut f_cobs_dev, half, cd as u32).unwrap();
                    }
                    if let Some(fom) = f_obs_mir_dev.as_mut() {
                        if mirror_aug {
                            ppo.stage_batch(&mut pass, &prm(od, 0), env.raw_obs_t.as_ref().unwrap(), &mean_o, &inv_o, &perm_id_o, &sign_id_o, &mut *fom, half, od as u32).unwrap();
                            ppo.stage_batch(&mut pass, &prm(od, half), env.raw_obs_t.as_ref().unwrap(), &mean_o, &inv_o, &perm_mir_o, &sign_mir_o, &mut *fom, half, od as u32).unwrap();
                        } else {
                            ppo.stage_batch(&mut pass, &prm(od, 0), env.raw_obs_t.as_ref().unwrap(), &mean_o, &inv_o, &perm_mir_o, &sign_mir_o, &mut *fom, half, od as u32).unwrap();
                        }
                    }
                }
                bk.submit(enc).unwrap();
            }
            let f_obs = &f_obs_dev;
            let f_cobs = &f_cobs_dev;
            // Opt-in parity check: the device staging must equal the host
            // formula element-for-element over both halves. (Fall counts are
            // far too weak a check for a column mapping.)
            if verify_stage {
                let dev = bk.slow_read_vec(f_obs.buffer()).await.unwrap();
                let host = stage_norm_flat(od, |c| &batch[c].obs, total, &ac.obs_norm, None);
                let md = dev.iter().zip(&host).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let devc = bk.slow_read_vec(f_cobs.buffer()).await.unwrap();
                let hostc =
                    stage_norm_flat(cd, |c| &batch[c].critic_obs, total, &ac.critic_norm, None);
                let mdc =
                    devc.iter().zip(&hostc).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                eprintln!("[verify_stage] obs maxdiff={md:.3e} critic maxdiff={mdc:.3e}");
            }
            // LOSS: refresh the stop-grad target net to the current
            // (iter-start) actor weights.
            let f_obs_mir = if mirror_loss > 0.0 {
                if let Some(sn) = s_net.as_mut() {
                    sn.write_w(&bk, &ac.actor);
                }
                f_obs_mir_dev.as_ref()
            } else {
                None
            };
            // Frozen-normalizer commit point: every consumer of the transform
            // this iteration (rollout forwards, GAE, staged obs, mirror map)
            // has now run — merge the iteration's pending stats for the next.
            if norm_freeze {
                if gpu_obs {
                    // Device Welford accumulated slots 1..T during finalize
                    // (host pushed 0..T-1 — a one-slot shift, statistically
                    // immaterial under the frozen-affine schedule).
                    let ((co, mo, m2o), (cc, mc, m2c)) = env.take_norm_moments().await;
                    let mut po = zealot_rl::ppo::PendingNorm::from_moments(co, mo, m2o);
                    let mut pc = zealot_rl::ppo::PendingNorm::from_moments(cc, mc, m2c);
                    ac.obs_norm.commit(&mut po);
                    ac.critic_norm.commit(&mut pc);
                } else {
                    ac.obs_norm.commit(&mut pn_obs);
                    ac.critic_norm.commit(&mut pn_cobs);
                }
            }
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
            // Rollout-policy log-std snapshot (σ_old), for the FULL Gaussian KL
            // below. `ac.log_std` (CPU) is the rollout value here — the GPU `lst`
            // buffer trains during the update and is synced back to `ac.log_std`
            // only after. The previous mean-only KL ignored σ entirely, so when
            // the policy's std drifted (the `al` pass trains log_std) it
            // UNDER-estimated the true divergence → the adaptive-LR controller
            // didn't brake enough → overshoot. rsl_rl uses the full analytic KL.
            let log_std_old: Vec<f32> = ac.log_std.clone();
            let mut last_kl = 0.0f32;
            // Update-phase profile: where the PPO update wall-clock goes
            // (encode = CPU command recording, exec = submit+synchronize i.e.
            // GPU execution incl. launch gaps, kl = the per-epoch readback).
            let (mut enc_s, mut exec_s, mut kl_s) = (0.0f64, 0.0f64, 0.0f64);
            // BIPED_KL_PROBE=1: measure KL of the last minibatch BEFORE any
            // update this iteration. Same weights + same states ⇒ must be ~0;
            // a nonzero floor here means the KL bookkeeping (mean_old vs the
            // staged obs) compares mismatched state/action pairs, i.e. the
            // adaptive-LR controller reacts to an artifact, not policy drift.
            let kl_probe = std::env::var("BIPED_KL_PROBE").is_ok_and(|v| v == "1");
            if kl_probe {
                let mut cur = EncCursor::new(&bk);
                {
                    let mut p = cur.pass("kp_obs");
                    cont.launch(
                        &bk,
                        &mut sh,
                        &mut p,
                        &mut a_net.a[0],
                        f_obs.columns(((n_mb - 1) * mb) as u32, mb as u32),
                        None,
                    )
                    .unwrap();
                }
                a_net
                    .forward(&bk, &g, &op, &act, &mut sh, &mut cur, ct, &o1m)
                    .unwrap();
                cur.flush();
                bk.synchronize().unwrap();
                let mn = bk.slow_read_vec(a_net.a[la + 1].buffer()).await.unwrap();
                let ls = bk.slow_read_vec(lst.buffer()).await.unwrap();
                let mut kl0 = 0.0f32;
                for c in 0..mb {
                    for k in 0..ad_ {
                        let ls_new = ls[k];
                        let ls_old = log_std_old[k];
                        let sig_old2 = (2.0 * ls_old).exp();
                        let inv_sig_new2 = (-2.0 * ls_new).exp();
                        let dmu = mn[k * mb + c] - mean_old_last[c][k];
                        kl0 += (ls_new - ls_old)
                            + 0.5 * (sig_old2 + dmu * dmu) * inv_sig_new2
                            - 0.5;
                    }
                }
                kl0 /= mb as f32;
                println!("[klprobe] pre-update kl = {kl0:.6} (must be ~0)");
            }
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
                let mut cur = EncCursor::new(&bk);
                for k in 0..n_mb {
                    let off = (k * mb) as u32;
                    let nb = mb as u32;
                    {
                        let mut p = cur.pass("g_obs");
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
                        let mut p = cur.pass("g_cobs");
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
                        let mut p = cur.pass("g_act");
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
                        let mut p = cur.pass("g_adv");
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
                        let mut p = cur.pass("g_lpo");
                        cont.launch(&bk, &mut sh, &mut p, &mut lpo, f_lpo.columns(off, nb), None)
                            .unwrap();
                    }
                    {
                        let mut p = cur.pass("g_vo");
                        cont.launch(&bk, &mut sh, &mut p, &mut vo, f_vo.columns(off, nb), None)
                            .unwrap();
                    }
                    {
                        let mut p = cur.pass("g_ret");
                        cont.launch(&bk, &mut sh, &mut p, &mut ret, f_ret.columns(off, nb), None)
                            .unwrap();
                    }
                    a_net
                        .forward(&bk, &g, &op, &act, &mut sh, &mut cur, ct, &o1m)
                        .unwrap();
                    c_net
                        .forward(&bk, &g, &op, &act, &mut sh, &mut cur, ct, &o1m)
                        .unwrap();
                    {
                        let mut p = cur.pass("ag");
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
                            let mut p = cur.pass("s_obs");
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
                        sn.forward(&bk, &g, &op, &act, &mut sh, &mut cur, ct, &o1m)
                            .unwrap();
                        {
                            let mut p = cur.pass("s_tgt");
                            g.dispatch_tiled(&bk, &mut sh, &mut p, &mut tgt, &pa, &sn.a[la + 1])
                                .unwrap();
                        }
                        {
                            let mut p = cur.pass("s_cp");
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
                            let mut p = cur.pass("s_sub");
                            op.launch(&bk, &mut sh, &mut p, OpAssignVariant::Sub, &mut res, &tgt)
                                .unwrap();
                        }
                        {
                            let mut p = cur.pass("s_mul");
                            op.launch(&bk, &mut sh, &mut p, OpAssignVariant::Mul, &mut res, &gw)
                                .unwrap();
                        }
                        {
                            let mut p = cur.pass("s_add");
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
                        let mut p = cur.pass("vg");
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
                        .backward(&bk, &g, &act, &mut sh, &mut cur, ct, &om1)
                        .unwrap();
                    c_net
                        .backward(&bk, &g, &act, &mut sh, &mut cur, ct, &om1)
                        .unwrap();
                    if let Some(ct) = ct {
                        cur.flush();
                        ct.row_sum(&dls, &gls, ad_, mb).unwrap();
                    } else {
                        let mut p = cur.pass("dl");
                        g.dispatch_naive(&bk, &mut sh, &mut p, &mut dls, &gls, &om1)
                            .unwrap();
                    }
                    // BIPED_GRAD_CLIP: rsl_rl-style global grad-norm clip
                    // (max_grad_norm, theirs 1.0) over actor+critic+log_std
                    // grads together, applied per minibatch BEFORE Adam (so
                    // moments accumulate clipped grads, like torch). Host
                    // roundtrip: flush+sync, read every grad buffer, compute
                    // one global norm, scale+write back only when clipping.
                    // Costs a per-minibatch sync — worth it: unclipped outlier
                    // minibatches spike per-epoch KL, and the adaptive-KL
                    // controller then pins lr ~10x below rsl_rl's operating
                    // point for hundreds of iterations.
                    if grad_clip > 0.0 {
                        // Squared norm of every grad tensor, GPU-side, into
                        // one slot each of `sqn`; a single 32-float readback
                        // then yields the global norm.
                        {
                            let mut p = cur.pass("gcn");
                            let mut slot = 0u32;
                            for net in [&a_net, &c_net] {
                                for t in net.dw.iter().chain(net.db.iter()) {
                                    red.launch(
                                        &bk,
                                        &mut sh,
                                        &mut p,
                                        ReduceVariant::SqNorm,
                                        t,
                                        sqn.rows_mut(slot, 1),
                                    )
                                    .unwrap();
                                    slot += 1;
                                }
                            }
                            red.launch(
                                &bk,
                                &mut sh,
                                &mut p,
                                ReduceVariant::SqNorm,
                                &dls,
                                sqn.rows_mut(slot, 1),
                            )
                            .unwrap();
                        }
                        cur.flush();
                        bk.synchronize().unwrap();
                        let sums = bk.slow_read_vec(sqn.buffer()).await.unwrap();
                        let norm = sums
                            .iter()
                            .step_by(SQN_STRIDE)
                            .take(gc_slots)
                            .sum::<f32>()
                            .sqrt();
                        if norm > grad_clip {
                            let sc = grad_clip / norm;
                            for (buf, len) in gc_scales.iter_mut().zip(gc_lens.iter()) {
                                bk.write_buffer(buf.buffer_mut(), 0, &vec![sc; *len])
                                    .unwrap();
                            }
                            let mut p = cur.pass("gcs");
                            let mut si = 0usize;
                            for net in [&mut a_net, &mut c_net] {
                                for i in 0..net.dw.len() {
                                    op.launch(
                                        &bk,
                                        &mut sh,
                                        &mut p,
                                        OpAssignVariant::Mul,
                                        &mut net.dw[i],
                                        &gc_scales[si],
                                    )
                                    .unwrap();
                                    si += 1;
                                }
                                for i in 0..net.db.len() {
                                    op.launch(
                                        &bk,
                                        &mut sh,
                                        &mut p,
                                        OpAssignVariant::Mul,
                                        &mut net.db[i],
                                        &gc_scales[si],
                                    )
                                    .unwrap();
                                    si += 1;
                                }
                            }
                            op.launch(
                                &bk,
                                &mut sh,
                                &mut p,
                                OpAssignVariant::Mul,
                                &mut dls,
                                &gc_scales[si],
                            )
                            .unwrap();
                        }
                    }
                    a_net.adam(&bk, &ad, &mut sh, &mut cur, &adp).unwrap();
                    c_net.adam(&bk, &ad, &mut sh, &mut cur, &adp).unwrap();
                    {
                        let mut p = cur.pass("al");
                        ad.step(
                            &bk, &mut sh, &mut p, &adp, &mut lst, &dls, &mut mls, &mut vls,
                        )
                        .unwrap();
                    }
                }
                enc_s += t_enc.elapsed().as_secs_f64();
                let t_exec = Instant::now();
                cur.flush();
                bk.synchronize().unwrap();
                exec_s += t_exec.elapsed().as_secs_f64();

                // Per-epoch KL (last minibatch) → adaptive-KL LR for the next epoch.
                let t_kl = Instant::now();
                let mn = bk.slow_read_vec(a_net.a[la + 1].buffer()).await.unwrap(); // [ad x mb]
                let ls = bk.slow_read_vec(lst.buffer()).await.unwrap(); // [ad]
                // Full analytic Gaussian KL(old‖new) per rsl_rl:
                //   Σ_k  log(σ_new/σ_old) + (σ_old² + (μ_old−μ_new)²)/(2σ_new²) − ½
                // (ls = log σ_new current; log_std_old = log σ_old rollout). The
                // old mean-only form dropped the σ terms, under-reading KL when the
                // std moved and letting the LR controller overshoot.
                let mut kl = 0.0f32;
                for c in 0..mb {
                    for k in 0..ad_ {
                        let ls_new = ls[k];
                        let ls_old = log_std_old[k];
                        let sig_old2 = (2.0 * ls_old).exp();
                        let inv_sig_new2 = (-2.0 * ls_new).exp();
                        let dmu = mn[k * mb + c] - mean_old_last[c][k];
                        kl += (ls_new - ls_old)
                            + 0.5 * (sig_old2 + dmu * dmu) * inv_sig_new2
                            - 0.5;
                    }
                }
                kl /= mb as f32;
                kl_s += t_kl.elapsed().as_secs_f64();
                last_kl = kl;
                if kl_probe {
                    println!("[klprobe] epoch {_epoch} cumulative kl = {kl:.6} (lr {lr:.2e})");
                }
                if kl > DESIRED_KL * 2.0 {
                    lr = (lr / 1.5).max(lr_min);
                } else if kl > 0.0 && kl < DESIRED_KL / 2.0 {
                    lr = (lr * 1.5).min(LR_MAX);
                }
                // KL early-stop (rsl_rl / WBC-AGILE): if this iteration's policy has
                // already drifted far past target, stop the remaining epochs so one
                // iteration can't run KL away. Without it the loop ran all EPOCHS
                // regardless, letting per-iter KL blow to ~100 during the walk-command
                // ramp (the policy thrashed instead of refining a gait). `kl` here is
                // current-vs-rollout policy, i.e. cumulative per-iter drift, so this
                // caps per-iter KL at ~5× target. (Tightening this to 1.5× was tried
                // and crippled learning ~40× — it sits inside the per-epoch KL
                // estimate's noise floor and trips constantly. The late-phase
                // degradation is instead handled by the cosine LR-ceiling decay, so
                // this early-phase safety stays loose.)
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
                    if let Some(lvl) = env.mean_terrain_level() {
                        s.push_str(&format!(" terrain_level={lvl:.3}"));
                    }
                    println!("{s}");
                }
            }
            if !ckpt.is_empty() && (it % 50 == 0 || it == iters - 1) {
                let _ = ac.save(&ckpt);
            }
            // Periodic ARCHIVE snapshots (`<ckpt>.iterN`, every 500 iters):
            // unlike the overwritten latest/best files these keep the whole
            // training trajectory on disk (~1.4 MB each), so any stage can be
            // re-rendered / recovered / diffed after the fact — the original
            // degraded run's iter-4000 peak was unrecoverable precisely
            // because only the (overwritten) latest existed.
            if !ckpt.is_empty() && it > 0 && it % 500 == 0 {
                let _ = ac.save(&format!("{ckpt}.iter{it}"));
            }
            // Best-checkpoint: EMA-smooth the mean step reward (α=0.02, ~50-iter
            // window) so we save on a sustained improvement, not a noisy spike;
            // warm up 200 iters before arming so early-training garbage never
            // wins. `<ckpt>.best` = the peak policy to deploy.
            let mean_step_rew = total_reward / total as f32;
            rew_ema = if it == 0 { mean_step_rew } else { 0.98 * rew_ema + 0.02 * mean_step_rew };
            if !ckpt.is_empty() && it >= 200 && rew_ema > best_ema {
                best_ema = rew_ema;
                if ac.save(&format!("{ckpt}.best")).is_ok() {
                    println!("[best] new peak reward-EMA {rew_ema:.4} @iter {it} → {ckpt}.best");
                }
            }
        }
        if !ckpt.is_empty() {
            ac.save(&ckpt).expect("save");
            println!("saved → {ckpt} (latest); best policy at {ckpt}.best (reward-EMA {best_ema:.4})");
        }
    });
}
