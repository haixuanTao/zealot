//! SONIC-local GPU PPO updater.
//!
//! This is the dimension-generic vortx path used by the locomotion GPU trainer,
//! kept local so `biped_train_gpu` remains an independent, stable example.

use anyhow::Result;
use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, GpuBackend};
use nalgebra::DMatrix;
use rayon::prelude::*;
use vortx::linalg::{
    Activation, Adam, AdamParams, Contiguous, Gemm, OpAssign, OpAssignVariant, Ppo, PpoActorParams,
    PpoValueParams,
};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use zealot_rl::ActorCritic;
use zealot_rl::net::Mlp;
use zealot_rl::ppo::Sample;

use crate::cutile_gemm::EncCursor;

const LOG_SQRT_2PI: f32 = 0.918_938_5;

fn tensor(backend: &GpuBackend, matrix: &DMatrix<f32>, usage: BufferUsages) -> Tensor<f32> {
    Tensor::matrix_from_na(backend, matrix, usage).expect("upload GPU PPO tensor")
}

fn weight_matrix(weights: &[f32], output: usize, input: usize) -> DMatrix<f32> {
    DMatrix::from_fn(output, input, |r, c| weights[r * input + c])
}

struct GpuMlp {
    dims: Vec<usize>,
    weights: Vec<Tensor<f32>>,
    biases: Vec<Tensor<f32>>,
    weight_m: Vec<Tensor<f32>>,
    weight_v: Vec<Tensor<f32>>,
    bias_m: Vec<Tensor<f32>>,
    bias_v: Vec<Tensor<f32>>,
    activations: Vec<Tensor<f32>>,
    broadcast_biases: Vec<Tensor<f32>>,
    deltas: Vec<Tensor<f32>>,
    weight_grads: Vec<Tensor<f32>>,
    bias_grads: Vec<Tensor<f32>>,
}

impl GpuMlp {
    fn new(backend: &GpuBackend, net: &Mlp, batch: usize) -> Self {
        let dims = net.dims.clone();
        let layers = net.w.len();
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let read_write = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let zeros = |rows, cols| DMatrix::<f32>::zeros(rows, cols);
        Self {
            weights: (0..layers)
                .map(|i| {
                    tensor(
                        backend,
                        &weight_matrix(&net.w[i], dims[i + 1], dims[i]),
                        read_write,
                    )
                })
                .collect(),
            biases: (0..layers)
                .map(|i| {
                    tensor(
                        backend,
                        &DMatrix::from_fn(dims[i + 1], 1, |r, _| net.b[i][r]),
                        read_write,
                    )
                })
                .collect(),
            weight_m: (0..layers)
                .map(|i| tensor(backend, &zeros(dims[i + 1], dims[i]), storage))
                .collect(),
            weight_v: (0..layers)
                .map(|i| tensor(backend, &zeros(dims[i + 1], dims[i]), storage))
                .collect(),
            bias_m: (0..layers)
                .map(|i| tensor(backend, &zeros(dims[i + 1], 1), storage))
                .collect(),
            bias_v: (0..layers)
                .map(|i| tensor(backend, &zeros(dims[i + 1], 1), storage))
                .collect(),
            activations: (0..=layers)
                .map(|i| tensor(backend, &zeros(dims[i], batch), read_write))
                .collect(),
            broadcast_biases: (0..layers)
                .map(|i| tensor(backend, &zeros(dims[i + 1], batch), storage))
                .collect(),
            deltas: (0..layers)
                .map(|i| tensor(backend, &zeros(dims[i + 1], batch), read_write))
                .collect(),
            weight_grads: (0..layers)
                .map(|i| tensor(backend, &zeros(dims[i + 1], dims[i]), read_write))
                .collect(),
            bias_grads: (0..layers)
                .map(|i| tensor(backend, &zeros(dims[i + 1], 1), read_write))
                .collect(),
            dims,
        }
    }

    fn layers(&self) -> usize {
        self.weights.len()
    }

    fn forward(
        &mut self,
        backend: &GpuBackend,
        gemm: &Gemm,
        op: &OpAssign,
        activation: &Activation,
        shapes: &mut TensorLayoutBuffers,
        encoder: &mut EncCursor,
        ones_row: &Tensor<f32>,
    ) -> Result<()> {
        let layers = self.layers();
        for i in 0..layers {
            let (left, right) = self.activations.split_at_mut(i + 1);
            let (input, output) = (&left[i], &mut right[0]);
            {
                let mut pass = encoder.pass("sonic_ppo_forward");
                gemm.dispatch_tiled(
                    backend,
                    shapes,
                    &mut pass,
                    &mut *output,
                    &self.weights[i],
                    input,
                )?;
            }
            {
                let mut pass = encoder.pass("sonic_ppo_bias_broadcast");
                gemm.dispatch_naive(
                    backend,
                    shapes,
                    &mut pass,
                    &mut self.broadcast_biases[i],
                    &self.biases[i],
                    ones_row,
                )?;
            }
            {
                let mut pass = encoder.pass("sonic_ppo_bias");
                op.launch(
                    backend,
                    shapes,
                    &mut pass,
                    OpAssignVariant::Add,
                    &mut *output,
                    &self.broadcast_biases[i],
                )?;
            }
            if i + 1 < layers {
                let mut pass = encoder.pass("sonic_ppo_elu");
                activation.elu(backend, shapes, &mut pass, &mut *output)?;
            }
        }
        Ok(())
    }

    fn backward(
        &mut self,
        backend: &GpuBackend,
        gemm: &Gemm,
        activation: &Activation,
        shapes: &mut TensorLayoutBuffers,
        encoder: &mut EncCursor,
        ones_column: &Tensor<f32>,
    ) -> Result<()> {
        for i in (0..self.layers()).rev() {
            {
                let mut pass = encoder.pass("sonic_ppo_weight_grad");
                gemm.dispatch_tiled(
                    backend,
                    shapes,
                    &mut pass,
                    &mut self.weight_grads[i],
                    &self.deltas[i],
                    self.activations[i].transpose_last_dims(),
                )?;
            }
            {
                let mut pass = encoder.pass("sonic_ppo_bias_grad");
                gemm.dispatch_naive(
                    backend,
                    shapes,
                    &mut pass,
                    &mut self.bias_grads[i],
                    &self.deltas[i],
                    ones_column,
                )?;
            }
            if i > 0 {
                {
                    let (left, right) = self.deltas.split_at_mut(i);
                    let (previous, current) = (&mut left[i - 1], &right[0]);
                    let mut pass = encoder.pass("sonic_ppo_activation_grad");
                    gemm.dispatch_tiled(
                        backend,
                        shapes,
                        &mut pass,
                        previous,
                        self.weights[i].transpose_last_dims(),
                        current,
                    )?;
                }
                let mut pass = encoder.pass("sonic_ppo_elu_backward");
                activation.elu_backward(
                    backend,
                    shapes,
                    &mut pass,
                    &mut self.deltas[i - 1],
                    &self.activations[i],
                )?;
            }
        }
        Ok(())
    }

    fn adam(
        &mut self,
        backend: &GpuBackend,
        adam: &Adam,
        shapes: &mut TensorLayoutBuffers,
        encoder: &mut EncCursor,
        params: &Tensor<AdamParams>,
    ) -> Result<()> {
        for i in 0..self.layers() {
            {
                let mut pass = encoder.pass("sonic_ppo_adam_weight");
                adam.step(
                    backend,
                    shapes,
                    &mut pass,
                    params,
                    &mut self.weights[i],
                    &self.weight_grads[i],
                    &mut self.weight_m[i],
                    &mut self.weight_v[i],
                )?;
            }
            {
                let mut pass = encoder.pass("sonic_ppo_adam_bias");
                adam.step(
                    backend,
                    shapes,
                    &mut pass,
                    params,
                    &mut self.biases[i],
                    &self.bias_grads[i],
                    &mut self.bias_m[i],
                    &mut self.bias_v[i],
                )?;
            }
        }
        Ok(())
    }

    async fn read_into(&self, backend: &GpuBackend, net: &mut Mlp) -> Result<()> {
        for i in 0..self.layers() {
            let (output, input) = (self.dims[i + 1], self.dims[i]);
            let weights = backend.slow_read_vec(self.weights[i].buffer()).await?;
            net.w[i].copy_from_slice(&weights[..output * input]);
            let biases = backend.slow_read_vec(self.biases[i].buffer()).await?;
            net.b[i].copy_from_slice(&biases[..output]);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GpuPpoConfig {
    pub epochs: usize,
    pub minibatches: usize,
    pub clip: f32,
    pub entropy_coef: f32,
    pub value_coef: f32,
    pub desired_kl: f32,
    pub lr_min: f32,
    pub lr_max: f32,
    pub log_std_min: f32,
    pub log_std_max: f32,
}

impl Default for GpuPpoConfig {
    fn default() -> Self {
        Self {
            epochs: 5,
            minibatches: 4,
            clip: 0.2,
            entropy_coef: 0.005,
            value_coef: 1.0,
            desired_kl: 0.01,
            lr_min: 1e-5,
            lr_max: 1e-2,
            log_std_min: -2.3,
            log_std_max: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuPpoStats {
    pub kl: f32,
    pub lr: f32,
    pub seconds: f64,
}

pub struct GpuPpoUpdater {
    config: GpuPpoConfig,
    actor: GpuMlp,
    critic: GpuMlp,
    log_std: Tensor<f32>,
    log_std_m: Tensor<f32>,
    log_std_v: Tensor<f32>,
    action: Tensor<f32>,
    advantage: Tensor<f32>,
    logp_old: Tensor<f32>,
    value_old: Tensor<f32>,
    returns: Tensor<f32>,
    log_std_grad_per_sample: Tensor<f32>,
    log_std_grad: Tensor<f32>,
    ones_row: Tensor<f32>,
    ones_column: Tensor<f32>,
    gemm: Gemm,
    op: OpAssign,
    activation: Activation,
    adam: Adam,
    ppo: Ppo,
    contiguous: Contiguous,
    shapes: TensorLayoutBuffers,
    action_dim: usize,
    minibatch: usize,
    adam_step: u64,
    lr: f32,
}

impl GpuPpoUpdater {
    pub fn new(
        backend: &GpuBackend,
        actor_critic: &ActorCritic,
        total_samples: usize,
        config: GpuPpoConfig,
    ) -> Result<Self> {
        anyhow::ensure!(config.minibatches > 0, "minibatches must be positive");
        anyhow::ensure!(
            total_samples % config.minibatches == 0,
            "sample count must divide evenly into minibatches"
        );
        let minibatch = total_samples / config.minibatches;
        let action_dim = actor_critic.action_dim();
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let read_write = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        Ok(Self {
            config,
            actor: GpuMlp::new(backend, &actor_critic.actor, minibatch),
            critic: GpuMlp::new(backend, &actor_critic.critic, minibatch),
            log_std: tensor(
                backend,
                &DMatrix::from_fn(action_dim, 1, |r, _| actor_critic.log_std[r]),
                read_write,
            ),
            log_std_m: tensor(backend, &DMatrix::zeros(action_dim, 1), storage),
            log_std_v: tensor(backend, &DMatrix::zeros(action_dim, 1), storage),
            action: tensor(backend, &DMatrix::zeros(action_dim, minibatch), read_write),
            advantage: tensor(backend, &DMatrix::zeros(1, minibatch), read_write),
            logp_old: tensor(backend, &DMatrix::zeros(1, minibatch), read_write),
            value_old: tensor(backend, &DMatrix::zeros(1, minibatch), read_write),
            returns: tensor(backend, &DMatrix::zeros(1, minibatch), read_write),
            log_std_grad_per_sample: tensor(
                backend,
                &DMatrix::zeros(action_dim, minibatch),
                read_write,
            ),
            log_std_grad: tensor(backend, &DMatrix::zeros(action_dim, 1), read_write),
            ones_row: tensor(backend, &DMatrix::from_element(1, minibatch, 1.0), storage),
            ones_column: tensor(backend, &DMatrix::from_element(minibatch, 1, 1.0), storage),
            gemm: Gemm::from_backend(backend)?,
            op: OpAssign::from_backend(backend)?,
            activation: Activation::from_backend(backend)?,
            adam: Adam::from_backend(backend)?,
            ppo: Ppo::from_backend(backend)?,
            contiguous: Contiguous::from_backend(backend)?,
            shapes: TensorLayoutBuffers::new(backend),
            action_dim,
            minibatch,
            adam_step: 0,
            lr: actor_critic.lr,
        })
    }

    pub async fn update(
        &mut self,
        backend: &GpuBackend,
        actor_critic: &mut ActorCritic,
        batch: &mut [Sample],
    ) -> Result<GpuPpoStats> {
        let started = std::time::Instant::now();
        anyhow::ensure!(
            batch.len() % self.minibatch == 0,
            "batch length must be a whole number of updater minibatches"
        );
        let total = batch.len();
        let minibatches = total / self.minibatch;
        let actor_dim = actor_critic.actor.dims[0];
        let critic_dim = actor_critic.critic.dims[0];

        let advantage_mean = batch.iter().map(|s| s.adv).sum::<f32>() / total as f32;
        let advantage_var = batch
            .iter()
            .map(|s| (s.adv - advantage_mean).powi(2))
            .sum::<f32>()
            / total as f32;
        let advantage_std = advantage_var.sqrt().max(1e-6);
        for sample in batch.iter_mut() {
            sample.adv = (sample.adv - advantage_mean) / advantage_std;
        }

        let normalized_actor: Vec<Vec<f32>> = batch
            .par_iter()
            .map(|sample| actor_critic.obs_norm.normalize(&sample.obs))
            .collect();
        let normalized_critic: Vec<Vec<f32>> = batch
            .par_iter()
            .map(|sample| actor_critic.critic_norm.normalize(&sample.critic_obs))
            .collect();
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let actor_obs = tensor(
            backend,
            &DMatrix::from_fn(actor_dim, total, |r, c| normalized_actor[c][r]),
            storage,
        );
        let critic_obs = tensor(
            backend,
            &DMatrix::from_fn(critic_dim, total, |r, c| normalized_critic[c][r]),
            storage,
        );
        let actions = tensor(
            backend,
            &DMatrix::from_fn(self.action_dim, total, |r, c| batch[c].action[r]),
            storage,
        );
        let advantages = tensor(
            backend,
            &DMatrix::from_fn(1, total, |_, c| batch[c].adv),
            storage,
        );
        let old_logp = tensor(
            backend,
            &DMatrix::from_fn(1, total, |_, c| batch[c].logp_old),
            storage,
        );
        let old_values = tensor(
            backend,
            &DMatrix::from_fn(1, total, |_, c| batch[c].value_old),
            storage,
        );
        let return_targets = tensor(
            backend,
            &DMatrix::from_fn(1, total, |_, c| batch[c].ret),
            storage,
        );

        let actor_params = Tensor::scalar(
            backend,
            PpoActorParams {
                clip: self.config.clip,
                entropy_coef: self.config.entropy_coef,
                scale: 1.0 / self.minibatch as f32,
                log_sqrt_2pi: LOG_SQRT_2PI,
                action_dim: self.action_dim as u32,
                num_cols: self.minibatch as u32,
                pad0: 0,
                pad1: 0,
            },
            BufferUsages::UNIFORM,
        )?;
        let value_params = Tensor::scalar(
            backend,
            PpoValueParams {
                clip: self.config.clip,
                value_coef: self.config.value_coef,
                scale: 1.0 / self.minibatch as f32,
                num_cols: self.minibatch as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
                pad3: 0,
            },
            BufferUsages::UNIFORM,
        )?;
        let last_offset = (minibatches - 1) * self.minibatch;
        let old_last_means: Vec<Vec<f32>> = (0..self.minibatch)
            .map(|c| batch[last_offset + c].mean_old.clone())
            .collect();
        let actor_last = self.actor.layers() - 1;
        let critic_last = self.critic.layers() - 1;
        let mut last_kl = 0.0;

        for _ in 0..self.config.epochs {
            self.adam_step += minibatches as u64;
            let correction1 = 1.0 - 0.9f32.powi(self.adam_step.min(1 << 30) as i32);
            let correction2 = 1.0 - 0.999f32.powi(self.adam_step.min(1 << 30) as i32);
            let adam_params = Tensor::scalar(
                backend,
                AdamParams {
                    lr: self.lr,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    bias_correction1: correction1,
                    bias_correction2: correction2,
                    pad0: 0.0,
                    pad1: 0.0,
                },
                BufferUsages::UNIFORM,
            )?;
            let mut encoder = EncCursor::new(backend);
            for minibatch_index in 0..minibatches {
                let offset = (minibatch_index * self.minibatch) as u32;
                let width = self.minibatch as u32;
                for (label, destination, source) in [
                    (
                        "sonic_stage_actor_obs",
                        &mut self.actor.activations[0],
                        actor_obs.columns(offset, width),
                    ),
                    (
                        "sonic_stage_critic_obs",
                        &mut self.critic.activations[0],
                        critic_obs.columns(offset, width),
                    ),
                    (
                        "sonic_stage_action",
                        &mut self.action,
                        actions.columns(offset, width),
                    ),
                    (
                        "sonic_stage_advantage",
                        &mut self.advantage,
                        advantages.columns(offset, width),
                    ),
                    (
                        "sonic_stage_logp",
                        &mut self.logp_old,
                        old_logp.columns(offset, width),
                    ),
                    (
                        "sonic_stage_value",
                        &mut self.value_old,
                        old_values.columns(offset, width),
                    ),
                    (
                        "sonic_stage_return",
                        &mut self.returns,
                        return_targets.columns(offset, width),
                    ),
                ] {
                    let mut pass = encoder.pass(label);
                    self.contiguous.launch(
                        backend,
                        &mut self.shapes,
                        &mut pass,
                        destination,
                        source,
                        None,
                    )?;
                }
                self.actor.forward(
                    backend,
                    &self.gemm,
                    &self.op,
                    &self.activation,
                    &mut self.shapes,
                    &mut encoder,
                    &self.ones_row,
                )?;
                self.critic.forward(
                    backend,
                    &self.gemm,
                    &self.op,
                    &self.activation,
                    &mut self.shapes,
                    &mut encoder,
                    &self.ones_row,
                )?;
                {
                    let mut pass = encoder.pass("sonic_actor_gradient");
                    self.ppo.actor_grad(
                        &mut pass,
                        &actor_params,
                        &self.actor.activations[actor_last + 1],
                        &self.action,
                        &self.log_std,
                        &self.advantage,
                        &self.logp_old,
                        &mut self.actor.deltas[actor_last],
                        &mut self.log_std_grad_per_sample,
                    )?;
                }
                {
                    let mut pass = encoder.pass("sonic_value_gradient");
                    self.ppo.value_grad(
                        &mut pass,
                        &value_params,
                        &self.critic.activations[critic_last + 1],
                        &self.value_old,
                        &self.returns,
                        &mut self.critic.deltas[critic_last],
                    )?;
                }
                self.actor.backward(
                    backend,
                    &self.gemm,
                    &self.activation,
                    &mut self.shapes,
                    &mut encoder,
                    &self.ones_column,
                )?;
                self.critic.backward(
                    backend,
                    &self.gemm,
                    &self.activation,
                    &mut self.shapes,
                    &mut encoder,
                    &self.ones_column,
                )?;
                {
                    let mut pass = encoder.pass("sonic_log_std_reduce");
                    self.gemm.dispatch_naive(
                        backend,
                        &mut self.shapes,
                        &mut pass,
                        &mut self.log_std_grad,
                        &self.log_std_grad_per_sample,
                        &self.ones_column,
                    )?;
                }
                self.actor.adam(
                    backend,
                    &self.adam,
                    &mut self.shapes,
                    &mut encoder,
                    &adam_params,
                )?;
                self.critic.adam(
                    backend,
                    &self.adam,
                    &mut self.shapes,
                    &mut encoder,
                    &adam_params,
                )?;
                {
                    let mut pass = encoder.pass("sonic_log_std_adam");
                    self.adam.step(
                        backend,
                        &mut self.shapes,
                        &mut pass,
                        &adam_params,
                        &mut self.log_std,
                        &self.log_std_grad,
                        &mut self.log_std_m,
                        &mut self.log_std_v,
                    )?;
                }
            }
            encoder.flush();
            backend.synchronize()?;

            let current_means = backend
                .slow_read_vec(self.actor.activations[actor_last + 1].buffer())
                .await?;
            let log_std = backend.slow_read_vec(self.log_std.buffer()).await?;
            let mut kl = 0.0;
            for c in 0..self.minibatch {
                for k in 0..self.action_dim {
                    let inverse_std = (-log_std[k]).exp();
                    let difference = (current_means[k * self.minibatch + c] - old_last_means[c][k])
                        * inverse_std;
                    kl += 0.5 * difference * difference;
                }
            }
            kl /= self.minibatch as f32;
            last_kl = kl;
            if kl > self.config.desired_kl * 2.0 {
                self.lr = (self.lr / 1.5).max(self.config.lr_min);
            } else if kl > 0.0 && kl < self.config.desired_kl / 2.0 {
                self.lr = (self.lr * 1.5).min(self.config.lr_max);
            }
            if kl > self.config.desired_kl * 5.0 {
                break;
            }
        }

        self.actor
            .read_into(backend, &mut actor_critic.actor)
            .await?;
        self.critic
            .read_into(backend, &mut actor_critic.critic)
            .await?;
        let mut log_std = backend.slow_read_vec(self.log_std.buffer()).await?;
        let mut clamped = false;
        for value in &mut log_std[..self.action_dim] {
            let next = value.clamp(self.config.log_std_min, self.config.log_std_max);
            clamped |= next != *value;
            *value = next;
        }
        actor_critic
            .log_std
            .copy_from_slice(&log_std[..self.action_dim]);
        actor_critic.lr = self.lr;
        if clamped {
            let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
            self.log_std = tensor(
                backend,
                &DMatrix::from_fn(self.action_dim, 1, |r, _| log_std[r]),
                usage,
            );
        }
        Ok(GpuPpoStats {
            kl: last_kl,
            lr: self.lr,
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}
