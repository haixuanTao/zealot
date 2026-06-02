//! A tiny deterministic LCG for weight init and exploration sampling.
//!
//! Duplicated (rather than shared) from `zealot-env`'s `rng` so the learning tier
//! stays independent of the env tier — they're a few lines and have no reason to
//! couple.

/// 64-bit linear congruential generator.
#[derive(Clone, Debug)]
pub struct Lcg(pub u64);

impl Lcg {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Standard normal via Box–Muller.
    #[inline]
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * self.unit()).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_unit() {
        let mut a = Lcg::new(9);
        let mut b = Lcg::new(9);
        for _ in 0..100 {
            assert_eq!(a.unit(), b.unit());
        }
    }
}
