//! PPO with a diagonal-Gaussian actor-critic — the `rsl_rl` tier.
//!
//! Matches the deployed policy's training config: separate actor/critic MLPs
//! (ELU, e.g. `[512, 256, 128]`), a state-independent learnable `log_std`, GAE(λ)
//! advantages, the clipped surrogate over K epochs × M minibatches, an
//! adaptive-KL learning rate, an entropy bonus, clipped value loss, and global
//! grad-norm clipping.
//!
//! The actor sees the (noised, partial) policy observation; the critic sees the
//! privileged observation — the asymmetric setup AGILE uses. The update is a
//! direct, allocation-light port of the math in `pendulum_ppo`, generalized to
//! many action dims and a real value function.

use crate::net::{Adam, Mlp, MlpGrad};
use crate::rng::Lcg;
use rayon::prelude::*;

const LOG_SQRT_2PI: f32 = 0.918_938_5; // ½·ln(2π)

/// Static PPO hyperparameters (the learning rate lives on [`ActorCritic`] because
/// it adapts). Defaults are the deployed LeRobot policy's.
#[derive(Clone, Copy, Debug)]
pub struct PpoConfig {
    /// Discount factor.
    pub gamma: f32,
    /// GAE λ.
    pub lam: f32,
    /// Surrogate clip ε.
    pub clip: f32,
    /// Entropy bonus coefficient.
    pub entropy_coef: f32,
    /// Value-loss coefficient.
    pub value_coef: f32,
    /// Learning epochs per update.
    pub epochs: usize,
    /// Minibatches per epoch.
    pub minibatches: usize,
    /// Global grad-norm clip.
    pub max_grad_norm: f32,
    /// Target KL for the adaptive LR schedule.
    pub desired_kl: f32,
    /// LR clamp `(min, max)` for the adaptive schedule.
    pub lr_bounds: (f32, f32),
    /// Whether to adapt the LR from the KL each minibatch (rsl_rl "adaptive"
    /// schedule). When `false`, the LR is held fixed at its initial value.
    pub adaptive_lr: bool,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lam: 0.95,
            clip: 0.2,
            entropy_coef: 0.01,
            value_coef: 1.0,
            epochs: 5,
            minibatches: 4,
            max_grad_norm: 1.0,
            desired_kl: 0.01,
            lr_bounds: (1e-5, 1e-2),
            adaptive_lr: true,
        }
    }
}

/// One transition's training data (filled during rollout, consumed by
/// [`ActorCritic::update`]).
#[derive(Clone, Debug, Default)]
pub struct Sample {
    /// Policy observation (actor input).
    pub obs: Vec<f32>,
    /// Privileged observation (critic input).
    pub critic_obs: Vec<f32>,
    /// Action taken.
    pub action: Vec<f32>,
    /// Actor mean at sample time (for the exact KL).
    pub mean_old: Vec<f32>,
    /// log-prob of `action` under the sampling policy.
    pub logp_old: f32,
    /// Critic value estimate at sample time (for clipped value loss).
    pub value_old: f32,
    /// GAE advantage (filled post-rollout).
    pub adv: f32,
    /// Return target (filled post-rollout).
    pub ret: f32,
}

/// Diagnostics returned by an update.
#[derive(Clone, Copy, Debug, Default)]
pub struct PpoStats {
    /// Mean approximate KL over the last epoch.
    pub kl: f32,
    /// Mean clipped surrogate (policy objective, higher is better).
    pub surrogate: f32,
    /// Mean value loss.
    pub value_loss: f32,
    /// Mean policy entropy.
    pub entropy: f32,
    /// Learning rate after adaptation.
    pub lr: f32,
}

/// Running mean/variance normalizer (Welford), with a unit-variance prior so it
/// is the identity until enough samples accumulate. rsl_rl's empirical
/// observation normalization — applied to obs before the actor and to the
/// privileged obs before the critic.
#[derive(Clone, Debug)]
pub struct Normalizer {
    mean: Vec<f32>,
    m2: Vec<f32>,
    count: f32,
}

impl Normalizer {
    /// New normalizer for `dim`-vectors (starts as the identity transform).
    pub fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.0; dim],
            m2: vec![1.0; dim], // var = m2/count = 1 initially
            count: 1.0,
        }
    }

    /// Fold one observation into the running statistics.
    pub fn update(&mut self, x: &[f32]) {
        self.count += 1.0;
        for i in 0..x.len() {
            let d = x[i] - self.mean[i];
            self.mean[i] += d / self.count;
            self.m2[i] += d * (x[i] - self.mean[i]);
        }
    }

    /// Whiten `x` to ~zero-mean/unit-variance, clamped to ±5.
    pub fn normalize(&self, x: &[f32]) -> Vec<f32> {
        (0..x.len())
            .map(|i| {
                let var = (self.m2[i] / self.count).max(1e-8);
                ((x[i] - self.mean[i]) / var.sqrt()).clamp(-5.0, 5.0)
            })
            .collect()
    }
}

/// A Gaussian actor-critic policy.
#[derive(Clone, Debug)]
pub struct ActorCritic {
    /// Actor network: policy-obs → action mean.
    pub actor: Mlp,
    /// Critic network: privileged-obs → scalar value.
    pub critic: Mlp,
    /// State-independent log standard deviations (one per action dim).
    pub log_std: Vec<f32>,
    /// Running normalizer for the policy observation.
    pub obs_norm: Normalizer,
    /// Running normalizer for the privileged (critic) observation.
    pub critic_norm: Normalizer,
    opt_actor: Adam,
    opt_critic: Adam,
    // Adam state for log_std.
    m_logstd: Vec<f32>,
    v_logstd: Vec<f32>,
    t_logstd: i32,
    /// Current (adaptive) learning rate.
    pub lr: f32,
}

impl ActorCritic {
    /// Build from explicit layer stacks. `actor_dims[0]` is the policy-obs dim and
    /// `actor_dims[last]` the action dim; `critic_dims[0]` is the privileged-obs
    /// dim and `critic_dims[last]` must be 1.
    pub fn new(
        actor_dims: &[usize],
        critic_dims: &[usize],
        init_noise_std: f32,
        init_lr: f32,
        rng: &mut Lcg,
    ) -> Self {
        assert_eq!(
            *critic_dims.last().unwrap(),
            1,
            "critic must output a scalar"
        );
        let action_dim = *actor_dims.last().unwrap();
        // Small actor output init → gentle, near-default initial policy.
        let actor = Mlp::new(actor_dims, 0.01, rng);
        let critic = Mlp::new(critic_dims, 1.0, rng);
        let opt_actor = Adam::new(&actor);
        let opt_critic = Adam::new(&critic);
        Self {
            obs_norm: Normalizer::new(actor_dims[0]),
            critic_norm: Normalizer::new(critic_dims[0]),
            actor,
            critic,
            log_std: vec![init_noise_std.ln(); action_dim],
            opt_actor,
            opt_critic,
            m_logstd: vec![0.0; action_dim],
            v_logstd: vec![0.0; action_dim],
            t_logstd: 0,
            lr: init_lr,
        }
    }

    /// Action dimensionality.
    #[inline]
    pub fn action_dim(&self) -> usize {
        self.log_std.len()
    }

    /// Fold an observation pair into the running normalizers (call once per
    /// collected step, during rollout).
    pub fn record_obs(&mut self, obs: &[f32], critic_obs: &[f32]) {
        self.obs_norm.update(obs);
        self.critic_norm.update(critic_obs);
    }

    /// Actor mean for an observation (normalized internally).
    pub fn mean(&self, obs: &[f32]) -> Vec<f32> {
        self.actor
            .forward(&self.obs_norm.normalize(obs))
            .output()
            .to_vec()
    }

    /// Critic value for a privileged observation (normalized internally).
    pub fn value(&self, critic_obs: &[f32]) -> f32 {
        self.critic
            .forward(&self.critic_norm.normalize(critic_obs))
            .output()[0]
    }

    /// Diagonal-Gaussian log-prob of `action` given `mean`.
    pub fn logp(&self, action: &[f32], mean: &[f32]) -> f32 {
        let mut lp = 0.0;
        for k in 0..self.action_dim() {
            let std = self.log_std[k].exp();
            let d = (action[k] - mean[k]) / std;
            lp += -0.5 * d * d - self.log_std[k] - LOG_SQRT_2PI;
        }
        lp
    }

    /// Policy entropy (nats).
    pub fn entropy(&self) -> f32 {
        let c = 0.5 + LOG_SQRT_2PI; // ½·ln(2πe)
        self.log_std.iter().map(|ls| ls + c).sum()
    }

    /// Sample an action: `a = mean + std · ε`. Returns `(action, logp, mean)`.
    pub fn sample(&self, obs: &[f32], rng: &mut Lcg) -> (Vec<f32>, f32, Vec<f32>) {
        let mean = self.mean(obs);
        let mut action = vec![0.0; self.action_dim()];
        for k in 0..self.action_dim() {
            let std = self.log_std[k].exp();
            action[k] = mean[k] + std * rng.gauss();
        }
        let lp = self.logp(&action, &mean);
        (action, lp, mean)
    }

    /// Exact KL `D(old ‖ new)` for diagonal Gaussians that share `log_std`
    /// (so the std terms cancel): `½ Σ ((μ_new − μ_old)/σ)²`.
    fn kl(&self, mean_new: &[f32], mean_old: &[f32]) -> f32 {
        let mut kl = 0.0;
        for k in 0..self.action_dim() {
            let inv = (-self.log_std[k]).exp(); // 1/σ
            kl += 0.5 * ((mean_new[k] - mean_old[k]) * inv).powi(2);
        }
        kl
    }

    /// Run the PPO update over `batch` (advantages/returns already filled). The
    /// batch is normalized internally. Returns last-epoch diagnostics.
    pub fn update(&mut self, batch: &mut [Sample], cfg: &PpoConfig) -> PpoStats {
        let n = batch.len();
        assert!(n >= cfg.minibatches, "fewer samples than minibatches");

        // Normalize advantages across the batch.
        let mean: f32 = batch.iter().map(|s| s.adv).sum::<f32>() / n as f32;
        let var: f32 = batch.iter().map(|s| (s.adv - mean).powi(2)).sum::<f32>() / n as f32;
        let sd = var.sqrt().max(1e-6);
        for s in batch.iter_mut() {
            s.adv = (s.adv - mean) / sd;
        }

        let mb = n / cfg.minibatches;
        let mut stats = PpoStats {
            lr: self.lr,
            ..Default::default()
        };

        for _ in 0..cfg.epochs {
            // A fresh shuffle each epoch (Fisher–Yates with the same LCG).
            let mut order: Vec<usize> = (0..n).collect();
            let mut sh = Lcg::new(self.t_logstd as u64 + 1);
            for i in (1..n).rev() {
                let j = (sh.unit() * (i + 1) as f32) as usize;
                order.swap(i, j.min(i));
            }

            let mut epoch = PpoStats::default();
            for m in 0..cfg.minibatches {
                let slice = &order[m * mb..(m + 1) * mb];
                let s = self.minibatch_step(batch, slice, cfg);
                epoch.kl += s.kl;
                epoch.surrogate += s.surrogate;
                epoch.value_loss += s.value_loss;
                epoch.entropy += s.entropy;
            }
            let inv = 1.0 / cfg.minibatches as f32;
            stats = PpoStats {
                kl: epoch.kl * inv,
                surrogate: epoch.surrogate * inv,
                value_loss: epoch.value_loss * inv,
                entropy: epoch.entropy * inv,
                lr: self.lr,
            };
        }
        stats
    }

    /// One minibatch: accumulate gradients (data-parallel across cores), adapt LR
    /// from the KL, clip, and step.
    fn minibatch_step(&mut self, batch: &[Sample], idx: &[usize], cfg: &PpoConfig) -> PpoStats {
        let action_dim = self.action_dim();
        let scale = 1.0 / idx.len() as f32;

        // Per-job accumulator: (actor grad, critic grad, log_std grad, surrogate,
        // value_loss, kl). Folded over disjoint sample chunks in parallel — every
        // read of `self` (forward/backward/logp/kl) is immutable, so this is sound.
        type Acc = (MlpGrad, MlpGrad, Vec<f32>, f32, f32, f32);
        let init = || {
            (
                MlpGrad::zero(&self.actor),
                MlpGrad::zero(&self.critic),
                vec![0.0f32; action_dim],
                0.0f32,
                0.0f32,
                0.0f32,
            )
        };
        let (g_actor, g_critic, g_logstd, surr_sum, vloss_sum, kl_sum) = idx
            .par_iter()
            .fold(init, |mut acc, &i| {
                let s = &batch[i];
                // --- actor (normalized obs) ---
                let act = self.actor.forward(&self.obs_norm.normalize(&s.obs));
                let mean = act.output().to_vec();
                let logp = self.logp(&s.action, &mean);
                let ratio = (logp - s.logp_old).exp();
                let a = s.adv;
                let surr = (ratio * a).min(ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * a);
                acc.3 += surr * scale;

                let clipped =
                    (a >= 0.0 && ratio > 1.0 + cfg.clip) || (a < 0.0 && ratio < 1.0 - cfg.clip);
                let mut g_mean = vec![0.0f32; action_dim];
                for k in 0..action_dim {
                    let inv_var = (-2.0 * self.log_std[k]).exp(); // 1/σ²
                    if !clipped {
                        let d = s.action[k] - mean[k];
                        g_mean[k] = -(a * ratio * d * inv_var) * scale;
                        let dls = a * ratio * ((d * d) * inv_var - 1.0);
                        acc.2[k] += -dls * scale;
                    }
                    // Entropy bonus: −entropy_coef·entropy, d/d log_std = −entropy_coef.
                    acc.2[k] += -cfg.entropy_coef * scale;
                }
                self.actor.backward(&act, &g_mean, &mut acc.0);

                // --- critic (clipped value loss, normalized obs) ---
                let vact = self
                    .critic
                    .forward(&self.critic_norm.normalize(&s.critic_obs));
                let v = vact.output()[0];
                let v_clipped = s.value_old + (v - s.value_old).clamp(-cfg.clip, cfg.clip);
                let l_unclipped = (v - s.ret).powi(2);
                let l_clipped = (v_clipped - s.ret).powi(2);
                let (vloss, dv) = if l_clipped > l_unclipped {
                    (l_clipped, 2.0 * (v_clipped - s.ret))
                } else {
                    (l_unclipped, 2.0 * (v - s.ret))
                };
                acc.4 += vloss * scale;
                let gv = [cfg.value_coef * dv * scale];
                self.critic.backward(&vact, &gv, &mut acc.1);

                acc.5 += self.kl(&mean, &s.mean_old) * scale;
                acc
            })
            .reduce(init, |mut a: Acc, b: Acc| {
                a.0.add(&b.0);
                a.1.add(&b.1);
                for k in 0..action_dim {
                    a.2[k] += b.2[k];
                }
                (a.0, a.1, a.2, a.3 + b.3, a.4 + b.4, a.5 + b.5)
            });

        let mut g_actor = g_actor;
        let mut g_critic = g_critic;
        let g_logstd = g_logstd;
        let mut acc = PpoStats {
            surrogate: surr_sum,
            value_loss: vloss_sum,
            kl: kl_sum,
            ..Default::default()
        };
        acc.entropy = self.entropy();

        // Adaptive LR from the KL (rsl_rl rule), applied before stepping.
        if cfg.adaptive_lr {
            if acc.kl > cfg.desired_kl * 2.0 {
                self.lr = (self.lr / 1.5).max(cfg.lr_bounds.0);
            } else if acc.kl > 0.0 && acc.kl < cfg.desired_kl / 2.0 {
                self.lr = (self.lr * 1.5).min(cfg.lr_bounds.1);
            }
        }

        // Clip and step.
        g_actor.clip(cfg.max_grad_norm);
        g_critic.clip(cfg.max_grad_norm);
        self.opt_actor.step(&mut self.actor, &g_actor, self.lr);
        self.opt_critic.step(&mut self.critic, &g_critic, self.lr);
        self.step_log_std(&g_logstd);

        acc.lr = self.lr;
        acc
    }

    /// Adam step for the `log_std` parameter vector.
    fn step_log_std(&mut self, g: &[f32]) {
        self.t_logstd += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(self.t_logstd);
        let bc2 = 1.0 - b2.powi(self.t_logstd);
        for k in 0..self.action_dim() {
            self.m_logstd[k] = b1 * self.m_logstd[k] + (1.0 - b1) * g[k];
            self.v_logstd[k] = b2 * self.v_logstd[k] + (1.0 - b2) * g[k] * g[k];
            let step = self.lr * (self.m_logstd[k] / bc1) / ((self.v_logstd[k] / bc2).sqrt() + eps);
            // Keep std in a sane band so exploration neither vanishes nor explodes.
            // Floor ≈ ln(0.2): a std of 0.2 keeps enough exploration to escape
            // local optima (e.g. the sink), rather than collapsing to ~0.
            self.log_std[k] = (self.log_std[k] - step).clamp(-1.6f32, 1.0);
        }
    }
}

/// GAE(λ) advantages and returns for one trajectory (`rewards`, `values`,
/// `dones` are per-step; `last_value` bootstraps the final step). `dones[t]`
/// being true zeroes the bootstrap across that boundary.
pub fn gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    last_value: f32,
    gamma: f32,
    lam: f32,
) -> (Vec<f32>, Vec<f32>) {
    let t = rewards.len();
    let mut adv = vec![0.0f32; t];
    let mut ret = vec![0.0f32; t];
    let mut gae_acc = 0.0;
    let mut next_v = last_value;
    for i in (0..t).rev() {
        let nonterminal = if dones[i] { 0.0 } else { 1.0 };
        let delta = rewards[i] + gamma * next_v * nonterminal - values[i];
        gae_acc = delta + gamma * lam * nonterminal * gae_acc;
        adv[i] = gae_acc;
        ret[i] = gae_acc + values[i];
        next_v = values[i];
    }
    (adv, ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gae_matches_hand_computation() {
        // Two steps, no termination, γ=1, λ=1 → adv = discounted reward-to-go − V.
        let rewards = [1.0, 2.0];
        let values = [0.0, 0.0];
        let dones = [false, false];
        let (adv, ret) = gae(&rewards, &values, &dones, 0.0, 1.0, 1.0);
        // ret[1] = 2, ret[0] = 1 + 2 = 3.
        assert!((ret[1] - 2.0).abs() < 1e-6);
        assert!((ret[0] - 3.0).abs() < 1e-6);
        assert!((adv[0] - 3.0).abs() < 1e-6);
        assert!((adv[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn gae_respects_done_boundary() {
        let rewards = [1.0, 1.0];
        let values = [5.0, 5.0];
        let dones = [true, false]; // step 0 terminal → no bootstrap from step 1
        let (adv, _ret) = gae(&rewards, &values, &dones, 0.0, 0.99, 0.95);
        // adv[0] = r0 - V0 = 1 - 5 = -4 (bootstrap zeroed by done).
        assert!((adv[0] - (-4.0)).abs() < 1e-6);
    }

    /// A 1-step Gaussian bandit: reward = −‖action − target‖². PPO should pull the
    /// actor mean toward the target and raise mean reward.
    #[test]
    fn ppo_improves_on_bandit() {
        let mut rng = Lcg::new(2024);
        let obs_dim = 3;
        let act_dim = 2;
        let target = [0.7f32, -0.4];
        let mut ac = ActorCritic::new(
            &[obs_dim, 32, act_dim],
            &[obs_dim, 32, 1],
            0.6,
            3e-3,
            &mut rng,
        );
        let cfg = PpoConfig {
            entropy_coef: 0.0,
            minibatches: 4,
            epochs: 4,
            ..Default::default()
        };
        let obs = [0.5f32, -0.2, 0.1];

        let mean_reward = |ac: &ActorCritic| -> f32 {
            let m = ac.mean(&obs);
            -((m[0] - target[0]).powi(2) + (m[1] - target[1]).powi(2))
        };
        let r0 = mean_reward(&ac);

        for _ in 0..120 {
            let mut batch: Vec<Sample> = Vec::new();
            let n = 256;
            let mut rewards = Vec::new();
            let mut values = Vec::new();
            for _ in 0..n {
                let (action, logp, mean_old) = ac.sample(&obs, &mut rng);
                let r = -((action[0] - target[0]).powi(2) + (action[1] - target[1]).powi(2));
                let v = ac.value(&obs);
                rewards.push(r);
                values.push(v);
                batch.push(Sample {
                    obs: obs.to_vec(),
                    critic_obs: obs.to_vec(),
                    action,
                    mean_old,
                    logp_old: logp,
                    value_old: v,
                    adv: 0.0,
                    ret: 0.0,
                });
            }
            // Each sample is its own 1-step episode.
            for i in 0..n {
                let (a, ret) = gae(
                    &[rewards[i]],
                    &[values[i]],
                    &[true],
                    0.0,
                    cfg.gamma,
                    cfg.lam,
                );
                batch[i].adv = a[0];
                batch[i].ret = ret[0];
            }
            ac.update(&mut batch, &cfg);
        }

        let r1 = mean_reward(&ac);
        assert!(
            r1 > r0 + 0.2,
            "PPO did not improve bandit reward: {r0} -> {r1}"
        );
        let m = ac.mean(&obs);
        assert!((m[0] - target[0]).abs() < 0.25, "mean[0] off: {}", m[0]);
        assert!((m[1] - target[1]).abs() < 0.25, "mean[1] off: {}", m[1]);
    }
}
