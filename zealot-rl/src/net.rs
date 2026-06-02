//! A small multi-layer perceptron with hand-written backprop and Adam.
//!
//! Generalizes the 1-hidden-layer net in `pendulum_ppo` to the
//! `[512, 256, 128]` ELU actor/critic stacks AGILE/rsl_rl use. Hidden layers are
//! ELU; the output layer is linear. No autograd — gradients are accumulated
//! explicitly in [`MlpGrad`] and applied with [`Adam`]. This keeps zealot-rl
//! dependency-free and CPU-testable; a GPU/`burn` backend can replace it later
//! behind the same [`Policy`](crate::ppo) surface.

use crate::rng::Lcg;

/// ELU activation, `f(x) = x if x > 0 else exp(x) − 1`.
#[inline]
fn elu(x: f32) -> f32 {
    if x > 0.0 { x } else { x.exp() - 1.0 }
}

/// ELU derivative expressed from the *post*-activation value `a = elu(x)`:
/// `f'(x) = 1 if x > 0 else exp(x) = a + 1`. Valid because `a > 0 ⟺ x > 0`.
#[inline]
fn elu_grad_from_act(a: f32) -> f32 {
    if a > 0.0 { 1.0 } else { a + 1.0 }
}

/// A dense feed-forward network: ELU hidden layers, linear output.
#[derive(Clone, Debug)]
pub struct Mlp {
    /// Layer sizes, `[in, h1, .., hk, out]`.
    pub dims: Vec<usize>,
    /// Weight matrices, one per layer; `w[l]` is row-major `dims[l+1] × dims[l]`.
    pub w: Vec<Vec<f32>>,
    /// Bias vectors, one per layer; `b[l]` has length `dims[l+1]`.
    pub b: Vec<Vec<f32>>,
}

/// Gradient accumulator with the same shape as an [`Mlp`].
#[derive(Clone, Debug)]
pub struct MlpGrad {
    /// Weight gradients, matching [`Mlp::w`].
    pub w: Vec<Vec<f32>>,
    /// Bias gradients, matching [`Mlp::b`].
    pub b: Vec<Vec<f32>>,
}

/// Forward-pass activation cache, needed by [`Mlp::backward`].
pub struct Activations {
    /// Per-layer outputs, `a[0] = input`, `a[L] = network output`.
    a: Vec<Vec<f32>>,
}

impl Activations {
    /// The network output (last layer activations).
    pub fn output(&self) -> &[f32] {
        self.a.last().expect("at least one layer")
    }
}

impl Mlp {
    /// Initialize with He-ish scaling: `w ~ N(0, 1/fan_in)`, last layer scaled by
    /// `out_scale` (use a small value for a gentle initial policy).
    pub fn new(dims: &[usize], out_scale: f32, rng: &mut Lcg) -> Self {
        assert!(dims.len() >= 2, "need at least input and output dims");
        let layers = dims.len() - 1;
        let mut w = Vec::with_capacity(layers);
        let mut b = Vec::with_capacity(layers);
        for l in 0..layers {
            let (fan_in, fan_out) = (dims[l], dims[l + 1]);
            let last = l == layers - 1;
            let s = (1.0 / fan_in as f32).sqrt() * if last { out_scale } else { 1.0 };
            w.push((0..fan_out * fan_in).map(|_| rng.gauss() * s).collect());
            b.push(vec![0.0; fan_out]);
        }
        Self {
            dims: dims.to_vec(),
            w,
            b,
        }
    }

    /// Number of layers (weight matrices).
    #[inline]
    pub fn layers(&self) -> usize {
        self.w.len()
    }

    /// Forward pass, returning the activation cache (call [`Activations::output`]
    /// for the result).
    pub fn forward(&self, x: &[f32]) -> Activations {
        debug_assert_eq!(x.len(), self.dims[0]);
        let mut a: Vec<Vec<f32>> = Vec::with_capacity(self.layers() + 1);
        a.push(x.to_vec());
        for l in 0..self.layers() {
            let (fan_in, fan_out) = (self.dims[l], self.dims[l + 1]);
            let prev = &a[l];
            let mut out = vec![0.0f32; fan_out];
            let last = l == self.layers() - 1;
            for o in 0..fan_out {
                let mut z = self.b[l][o];
                let row = &self.w[l][o * fan_in..(o + 1) * fan_in];
                for i in 0..fan_in {
                    z += row[i] * prev[i];
                }
                out[o] = if last { z } else { elu(z) };
            }
            a.push(out);
        }
        Activations { a }
    }

    /// Backprop a gradient on the output (`g_out`, length `dims[last]`) through
    /// the cached activations, accumulating into `g`. Returns the gradient w.r.t.
    /// the input (used to chain a shared trunk, unused here but cheap).
    pub fn backward(&self, act: &Activations, g_out: &[f32], g: &mut MlpGrad) -> Vec<f32> {
        let mut delta = g_out.to_vec();
        for l in (0..self.layers()).rev() {
            let (fan_in, fan_out) = (self.dims[l], self.dims[l + 1]);
            let prev = &act.a[l];
            let cur = &act.a[l + 1];
            // For hidden layers, fold the ELU derivative into delta (output layer
            // is linear, so it passes through).
            if l != self.layers() - 1 {
                for o in 0..fan_out {
                    delta[o] *= elu_grad_from_act(cur[o]);
                }
            }
            let mut g_prev = vec![0.0f32; fan_in];
            for o in 0..fan_out {
                let d = delta[o];
                g.b[l][o] += d;
                let wrow = &self.w[l][o * fan_in..(o + 1) * fan_in];
                let grow = &mut g.w[l][o * fan_in..(o + 1) * fan_in];
                for i in 0..fan_in {
                    grow[i] += d * prev[i];
                    g_prev[i] += d * wrow[i];
                }
            }
            delta = g_prev;
        }
        delta
    }
}

impl MlpGrad {
    /// A zeroed gradient shaped like `net`.
    pub fn zero(net: &Mlp) -> Self {
        Self {
            w: net.w.iter().map(|m| vec![0.0; m.len()]).collect(),
            b: net.b.iter().map(|v| vec![0.0; v.len()]).collect(),
        }
    }

    /// Add another gradient into this one in place (for parallel reduction).
    pub fn add(&mut self, other: &MlpGrad) {
        for (m, o) in self.w.iter_mut().zip(&other.w) {
            for (x, y) in m.iter_mut().zip(o) {
                *x += *y;
            }
        }
        for (v, o) in self.b.iter_mut().zip(&other.b) {
            for (x, y) in v.iter_mut().zip(o) {
                *x += *y;
            }
        }
    }

    /// Scale all gradients in place (e.g. to average over a minibatch).
    pub fn scale(&mut self, s: f32) {
        for m in &mut self.w {
            for x in m {
                *x *= s;
            }
        }
        for v in &mut self.b {
            for x in v {
                *x *= s;
            }
        }
    }

    /// Global L2 norm of the gradient (for grad clipping).
    pub fn l2_norm(&self) -> f32 {
        let mut s = 0.0;
        for m in &self.w {
            for &x in m {
                s += x * x;
            }
        }
        for v in &self.b {
            for &x in v {
                s += x * x;
            }
        }
        s.sqrt()
    }

    /// Clip the gradient to a maximum global L2 norm.
    pub fn clip(&mut self, max_norm: f32) {
        let n = self.l2_norm();
        if n > max_norm && n > 0.0 {
            self.scale(max_norm / n);
        }
    }
}

/// Adam optimizer state for one [`Mlp`].
#[derive(Clone, Debug)]
pub struct Adam {
    mw: Vec<Vec<f32>>,
    vw: Vec<Vec<f32>>,
    mb: Vec<Vec<f32>>,
    vb: Vec<Vec<f32>>,
    t: i32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl Adam {
    /// Adam with the usual defaults (β₁=0.9, β₂=0.999, ε=1e-8).
    pub fn new(net: &Mlp) -> Self {
        Self {
            mw: net.w.iter().map(|m| vec![0.0; m.len()]).collect(),
            vw: net.w.iter().map(|m| vec![0.0; m.len()]).collect(),
            mb: net.b.iter().map(|v| vec![0.0; v.len()]).collect(),
            vb: net.b.iter().map(|v| vec![0.0; v.len()]).collect(),
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// One Adam step: `net ← net − lr · m̂ / (√v̂ + ε)`.
    pub fn step(&mut self, net: &mut Mlp, g: &MlpGrad, lr: f32) {
        self.t += 1;
        let (b1, b2) = (self.beta1, self.beta2);
        let bc1 = 1.0 - b1.powi(self.t);
        let bc2 = 1.0 - b2.powi(self.t);
        let eps = self.eps;
        let upd = |p: &mut [f32], gr: &[f32], m: &mut [f32], v: &mut [f32]| {
            for i in 0..p.len() {
                m[i] = b1 * m[i] + (1.0 - b1) * gr[i];
                v[i] = b2 * v[i] + (1.0 - b2) * gr[i] * gr[i];
                p[i] -= lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps);
            }
        };
        for l in 0..net.w.len() {
            upd(&mut net.w[l], &g.w[l], &mut self.mw[l], &mut self.vw[l]);
            upd(&mut net.b[l], &g.b[l], &mut self.mb[l], &mut self.vb[l]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elu_and_grad() {
        assert_eq!(elu(2.0), 2.0);
        assert!((elu(0.0)).abs() < 1e-9);
        assert!((elu(-1.0) - ((-1.0f32).exp() - 1.0)).abs() < 1e-6);
        assert_eq!(elu_grad_from_act(2.0), 1.0); // post-act > 0
        assert!((elu_grad_from_act(elu(-1.0)) - (-1.0f32).exp()).abs() < 1e-6);
    }

    /// Finite-difference gradient check on a scalar loss `½‖out − target‖²`.
    #[test]
    fn backward_matches_finite_difference() {
        let mut rng = Lcg::new(123);
        let dims = [4usize, 6, 5, 3];
        let mut net = Mlp::new(&dims, 1.0, &mut rng);
        let x = [0.3f32, -0.7, 0.1, 0.9];
        let target = [0.2f32, -0.4, 0.6];

        let loss = |net: &Mlp| -> f32 {
            let out = net.forward(&x);
            let o = out.output();
            0.5 * (0..3).map(|k| (o[k] - target[k]).powi(2)).sum::<f32>()
        };

        // Analytic gradient.
        let act = net.forward(&x);
        let o = act.output().to_vec();
        let g_out: Vec<f32> = (0..3).map(|k| o[k] - target[k]).collect();
        let mut g = MlpGrad::zero(&net);
        net.backward(&act, &g_out, &mut g);

        // Compare a sample of weights against central differences.
        let eps = 1e-3;
        for l in 0..net.layers() {
            for idx in [0usize, net.w[l].len() / 2, net.w[l].len() - 1] {
                let orig = net.w[l][idx];
                net.w[l][idx] = orig + eps;
                let lp = loss(&net);
                net.w[l][idx] = orig - eps;
                let lm = loss(&net);
                net.w[l][idx] = orig;
                let fd = (lp - lm) / (2.0 * eps);
                let an = g.w[l][idx];
                assert!(
                    (fd - an).abs() < 1e-2,
                    "layer {l} idx {idx}: fd={fd} analytic={an}"
                );
            }
        }
    }

    #[test]
    fn adam_minimizes_quadratic() {
        // Fit a 1-layer linear net to a constant target; loss must fall.
        let mut rng = Lcg::new(7);
        let mut net = Mlp::new(&[2, 1], 1.0, &mut rng);
        let mut opt = Adam::new(&net);
        let x = [1.0f32, -2.0];
        let target = 0.5f32;
        let loss = |net: &Mlp| (net.forward(&x).output()[0] - target).powi(2);
        let l0 = loss(&net);
        for _ in 0..500 {
            let act = net.forward(&x);
            let g_out = [2.0 * (act.output()[0] - target)];
            let mut g = MlpGrad::zero(&net);
            net.backward(&act, &g_out, &mut g);
            opt.step(&mut net, &g, 1e-2);
        }
        let l1 = loss(&net);
        assert!(l1 < l0 * 1e-3, "loss did not converge: {l0} -> {l1}");
    }

    #[test]
    fn grad_clip_caps_norm() {
        let mut rng = Lcg::new(1);
        let net = Mlp::new(&[3, 4, 2], 1.0, &mut rng);
        let mut g = MlpGrad::zero(&net);
        g.b[0][0] = 100.0;
        g.clip(1.0);
        assert!((g.l2_norm() - 1.0).abs() < 1e-4);
    }
}
