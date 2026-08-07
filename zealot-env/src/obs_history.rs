//! Per-env observation-history stacking (`BIPED_OBS_HISTORY=H`).
//!
//! Matches Isaac Lab / WBC-AGILE observation-history semantics for an MLP
//! policy: the policy observation is the last `H` frames stacked
//! **oldest→newest**; on episode reset the current frame is replicated into
//! all `H` slots (a freshly reset env never sees another episode's frames or
//! zeros). The frame pushed each step is the final (noised) actor
//! observation, so the history records exactly what the policy saw.
//!
//! `H <= 1` (or the env var unset) disables the feature entirely — callers
//! hold an `Option<ObsHistory>` and skip stacking when `None`, keeping the
//! default path byte-identical.

/// Flat ring buffer of the last `h` observation frames for `n` envs.
pub struct ObsHistory {
    h: usize,
    dim: usize,
    /// `n * h * dim`, env-major; slot layout is a ring indexed by `head`.
    buf: Vec<f32>,
    /// Per-env index of the NEXT slot to write (== oldest stored frame).
    head: Vec<usize>,
}

/// Programmatic override for `BIPED_OBS_HISTORY` — for wasm demos, where env
/// vars can't be set (same pattern as `robots::set_robot_override`).
pub static OBS_HISTORY_OVERRIDE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

impl ObsHistory {
    /// Parse `BIPED_OBS_HISTORY` (or [`OBS_HISTORY_OVERRIDE`]); `None` unless
    /// it resolves to an `H > 1`.
    pub fn from_env(n: usize, dim: usize) -> Option<Self> {
        let h: usize = OBS_HISTORY_OVERRIDE.get().copied().unwrap_or_else(|| {
            std::env::var("BIPED_OBS_HISTORY")
                .ok()
                .and_then(|s| s.parse().ok())
                // 5 = the production default; MUST agree with the trainer's
                // OBS_H read (src/bin/biped_train_gpu.rs). This defaulted to
                // 1 while the trainer said 5 — v29's first 12k iters trained
                // single-frame because only the trainer copy was updated.
                .unwrap_or(5)
        });
        (h > 1).then(|| Self::new(n, h, dim))
    }

    pub fn new(n: usize, h: usize, dim: usize) -> Self {
        ObsHistory {
            h,
            dim,
            buf: vec![0.0; n * h * dim],
            head: vec![0; n],
        }
    }

    pub fn h(&self) -> usize {
        self.h
    }

    /// Stacked observation length (`h * dim`).
    pub fn stacked_dim(&self) -> usize {
        self.h * self.dim
    }

    fn env_base(&self, e: usize) -> usize {
        e * self.h * self.dim
    }

    /// Append `frame` as env `e`'s newest frame (overwrites the oldest).
    pub fn push(&mut self, e: usize, frame: &[f32]) {
        debug_assert_eq!(frame.len(), self.dim);
        let slot = self.env_base(e) + self.head[e] * self.dim;
        self.buf[slot..slot + self.dim].copy_from_slice(frame);
        self.head[e] = (self.head[e] + 1) % self.h;
    }

    /// Episode reset: replicate `frame` into all `h` slots of env `e`.
    pub fn reset(&mut self, e: usize, frame: &[f32]) {
        debug_assert_eq!(frame.len(), self.dim);
        let base = self.env_base(e);
        for s in 0..self.h {
            self.buf[base + s * self.dim..base + (s + 1) * self.dim].copy_from_slice(frame);
        }
        self.head[e] = 0;
    }

    /// Write env `e`'s frames into `out` oldest→newest (`out.len() == h*dim`).
    pub fn write_stacked(&self, e: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.stacked_dim());
        let base = self.env_base(e);
        // `head` points at the oldest frame (next write slot).
        for i in 0..self.h {
            let slot = (self.head[e] + i) % self.h;
            out[i * self.dim..(i + 1) * self.dim]
                .copy_from_slice(&self.buf[base + slot * self.dim..base + (slot + 1) * self.dim]);
        }
    }

    /// Convenience: push then return the freshly stacked observation.
    pub fn push_stacked(&mut self, e: usize, frame: &[f32]) -> Vec<f32> {
        self.push(e, frame);
        let mut out = vec![0.0; self.stacked_dim()];
        self.write_stacked(e, &mut out);
        out
    }

    /// Split into per-env mutable views (disjoint slices), so a batch of
    /// envs can push+stack CONCURRENTLY. The serial push_stacked loop
    /// measured ~12 ms/step at 4096 envs (H=5, dim=53) — all in one thread;
    /// the caller feeds these views to rayon instead.
    pub fn env_views(&mut self) -> Vec<EnvHistView<'_>> {
        let (h, dim) = (self.h, self.dim);
        self.buf
            .chunks_mut(h * dim)
            .zip(self.head.iter_mut())
            .map(|(buf, head)| EnvHistView { h, dim, buf, head })
            .collect()
    }

    /// Convenience: reset then return the stacked (replicated) observation.
    pub fn reset_stacked(&mut self, e: usize, frame: &[f32]) -> Vec<f32> {
        self.reset(e, frame);
        let mut out = vec![0.0; self.stacked_dim()];
        self.write_stacked(e, &mut out);
        out
    }
}

/// One env's history, borrowed disjointly out of [`ObsHistory::env_views`].
pub struct EnvHistView<'a> {
    h: usize,
    dim: usize,
    /// This env's `h * dim` ring slice.
    buf: &'a mut [f32],
    head: &'a mut usize,
}

impl EnvHistView<'_> {
    /// Push `*obs` (one frame) into the ring, then REPLACE it with the
    /// freshly stacked `h * dim` window (oldest→newest) — the batch-parallel
    /// equivalent of [`ObsHistory::push_stacked`], bit-identical output.
    pub fn push_stacked_replace(&mut self, obs: &mut Vec<f32>) {
        debug_assert_eq!(obs.len(), self.dim);
        let slot = *self.head * self.dim;
        self.buf[slot..slot + self.dim].copy_from_slice(obs);
        *self.head = (*self.head + 1) % self.h;
        let mut out = vec![0.0; self.h * self.dim];
        for i in 0..self.h {
            let s = (*self.head + i) % self.h;
            out[i * self.dim..(i + 1) * self.dim]
                .copy_from_slice(&self.buf[s * self.dim..(s + 1) * self.dim]);
        }
        *obs = out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_replicates_and_push_shifts_oldest_first() {
        let mut h = ObsHistory::new(2, 3, 2);
        h.reset(0, &[1.0, 1.0]);
        let mut out = vec![0.0; 6];
        h.write_stacked(0, &mut out);
        assert_eq!(out, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // Push frames 2, 3: history becomes [1, 2, 3] oldest→newest.
        assert_eq!(h.push_stacked(0, &[2.0, 2.0]), vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0]);
        assert_eq!(h.push_stacked(0, &[3.0, 3.0]), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        // Ring wrap: [2, 3, 4].
        assert_eq!(h.push_stacked(0, &[4.0, 4.0]), vec![2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);

        // Env 1 untouched by env 0 traffic.
        h.reset(1, &[9.0, 9.0]);
        let mut out1 = vec![0.0; 6];
        h.write_stacked(1, &mut out1);
        assert_eq!(out1, vec![9.0; 6]);

        // Reset mid-stream replicates again.
        assert_eq!(h.reset_stacked(0, &[5.0, 5.0]), vec![5.0; 6]);
    }

    #[test]
    fn env_views_match_serial_push_stacked_exactly() {
        // Same frame sequence through both paths → identical stacks & ring
        // state, including across the ring wrap.
        let mut serial = ObsHistory::new(3, 4, 2);
        let mut batch = ObsHistory::new(3, 4, 2);
        for e in 0..3 {
            serial.reset(e, &[e as f32; 2]);
            batch.reset(e, &[e as f32; 2]);
        }
        for step in 0..6 {
            let mut frames: Vec<Vec<f32>> = (0..3)
                .map(|e| vec![10.0 * e as f32 + step as f32; 2])
                .collect();
            let expected: Vec<Vec<f32>> = frames
                .iter()
                .enumerate()
                .map(|(e, f)| serial.push_stacked(e, f))
                .collect();
            for (view, obs) in batch.env_views().iter_mut().zip(frames.iter_mut()) {
                view.push_stacked_replace(obs);
            }
            assert_eq!(frames, expected, "diverged at step {step}");
        }
    }
}
