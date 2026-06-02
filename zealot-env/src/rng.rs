//! A tiny, deterministic LCG — enough for command sampling and domain
//! randomization without pulling in `rand`. Same generator the pendulum examples
//! use, lifted here so the env layer can sample reproducibly from a seed.

/// 64-bit linear congruential generator.
#[derive(Clone, Debug)]
pub struct Lcg(pub u64);

impl Lcg {
    /// Seed the generator (any value; `0` is fine).
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

    /// Uniform in `[a, b)`.
    #[inline]
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }

    /// Standard normal via Box–Muller.
    #[inline]
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * self.unit()).cos()
    }

    /// `true` with probability `p`.
    #[inline]
    pub fn chance(&mut self, p: f32) -> bool {
        self.unit() < p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_in_range_and_deterministic() {
        let mut a = Lcg::new(42);
        let mut b = Lcg::new(42);
        for _ in 0..1000 {
            let x = a.unit();
            assert!((0.0..1.0).contains(&x));
            assert_eq!(x, b.unit());
        }
    }

    #[test]
    fn range_respects_bounds() {
        let mut r = Lcg::new(7);
        for _ in 0..1000 {
            let x = r.range(-0.8, 0.8);
            assert!((-0.8..0.8).contains(&x));
        }
    }
}
