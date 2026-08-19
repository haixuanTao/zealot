//! Resume state that lives *outside* the policy checkpoint.
//!
//! [`ActorCritic::save`](crate::ActorCritic::save) persists what you need to
//! *run* a policy: weights, observation-normalizer statistics, `log_std`, and
//! the adaptive-KL learning rate. That is deliberately all it holds — the wasm
//! demos embed that file via `include_bytes!`, so it stays small.
//!
//! It is **not** enough to *continue training*. A resumed run also needs the
//! optimizer's memory (Adam's first/second moments plus the global step that
//! drives bias correction), the best-so-far reward EMA, and the terrain level
//! each env had climbed to. Dropping those means:
//!
//! - Adam restarts from zero moments at `t = 0`, so the bias-corrected first
//!   updates are effectively enormous and kick a converged policy off its
//!   optimum — the resumed run degrades immediately and may never recover.
//! - The best-checkpoint tracker re-arms from `-inf`, so the first EMA after
//!   warmup overwrites `<ckpt>.best` even when the *previous* run's best was
//!   better. The better policy is destroyed.
//! - Every env's terrain curriculum is redrawn from `U{0,1}`, discarding the
//!   difficulty the population had earned.
//!
//! This sidecar (`<ckpt>.train`) carries that training-only state. It sits next
//! to the checkpoint and is entirely optional: a missing or mismatched sidecar
//! just means a cold-start optimizer, which is exactly the old behavior.

use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

/// Adam's persistent state for one network, in the flat row-major layout the
/// GPU trainer reads back: `mw`/`vw` are `[out x in]` per layer, `mb`/`vb` are
/// `[out]` per layer.
#[derive(Clone, Debug, Default)]
pub struct MomentSet {
    /// Layer dimensions, e.g. `[obs, 256, 256, 128, 12]` — `dims.len() - 1`
    /// layers. Checked against the live net on load so a checkpoint from a
    /// different architecture falls back to a cold optimizer instead of
    /// silently loading garbage.
    pub dims: Vec<usize>,
    pub mw: Vec<Vec<f32>>,
    pub vw: Vec<Vec<f32>>,
    pub mb: Vec<Vec<f32>>,
    pub vb: Vec<Vec<f32>>,
}

impl MomentSet {
    fn layers(&self) -> usize {
        self.dims.len().saturating_sub(1)
    }

    /// Every per-layer vector has the length its `dims` entry implies.
    fn is_consistent(&self) -> bool {
        let l = self.layers();
        if l == 0 || self.mw.len() != l || self.vw.len() != l || self.mb.len() != l || self.vb.len() != l {
            return false;
        }
        (0..l).all(|i| {
            let (out, inp) = (self.dims[i + 1], self.dims[i]);
            self.mw[i].len() == out * inp
                && self.vw[i].len() == out * inp
                && self.mb[i].len() == out
                && self.vb[i].len() == out
        })
    }
}

/// Everything a resumed run needs that the policy checkpoint does not carry.
#[derive(Clone, Debug, Default)]
pub struct TrainState {
    pub actor: MomentSet,
    pub critic: MomentSet,
    /// Global Adam step — drives bias correction (`1 - beta^gstep`).
    pub gstep: u64,
    /// Iterations completed by the run that wrote this file (provenance and
    /// logging; the loop itself still counts from 0).
    pub iter_done: u64,
    /// Peak reward EMA seen so far. Restoring this is what stops a resumed run
    /// from overwriting a better `<ckpt>.best`.
    pub best_ema: f32,
    /// Live reward EMA, so the smoothing continues instead of restarting.
    pub rew_ema: f32,
    /// Per-env terrain curriculum `(level, successes, failures)`. Empty when
    /// the terrain curriculum is off.
    pub terrain: Vec<(u32, u32, u32)>,
}

/// Conventional sidecar path for a checkpoint: `<ckpt>.train`.
pub fn sidecar_path(ckpt: &str) -> String {
    format!("{ckpt}.train")
}

impl TrainState {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut owned: Vec<(String, Vec<u8>, Vec<usize>, Dtype)> = Vec::new();
        for (prefix, ms) in [("actor", &self.actor), ("critic", &self.critic)] {
            for l in 0..ms.layers() {
                let (out, inp) = (ms.dims[l + 1], ms.dims[l]);
                for (tag, v) in [("mw", &ms.mw[l]), ("vw", &ms.vw[l])] {
                    owned.push((format!("{prefix}.{tag}_{l}"), f32_vec_bytes(v), vec![out, inp], Dtype::F32));
                }
                for (tag, v) in [("mb", &ms.mb[l]), ("vb", &ms.vb[l])] {
                    owned.push((format!("{prefix}.{tag}_{l}"), f32_vec_bytes(v), vec![out], Dtype::F32));
                }
            }
            // Architecture, so load can reject a mismatched checkpoint.
            let d: Vec<f32> = ms.dims.iter().map(|&x| x as f32).collect();
            owned.push((format!("{prefix}.dims"), f32_vec_bytes(&d), vec![d.len()], Dtype::F32));
        }
        // Counters that must survive exactly — f64 so a long run cannot lose
        // integer precision the way f32 would past 2^24 steps.
        for (name, v) in [("gstep", self.gstep as f64), ("iter_done", self.iter_done as f64)] {
            owned.push((name.into(), v.to_le_bytes().to_vec(), vec![1], Dtype::F64));
        }
        for (name, v) in [("best_ema", self.best_ema), ("rew_ema", self.rew_ema)] {
            owned.push((name.into(), v.to_le_bytes().to_vec(), vec![1], Dtype::F32));
        }
        let terr: Vec<f32> = self
            .terrain
            .iter()
            .flat_map(|&(l, s, f)| [l as f32, s as f32, f as f32])
            .collect();
        owned.push(("terrain".into(), f32_vec_bytes(&terr), vec![self.terrain.len(), 3], Dtype::F32));

        let views: Vec<(String, TensorView)> = owned
            .iter()
            .map(|(n, b, sh, dt)| {
                TensorView::new(*dt, sh.clone(), b)
                    .map(|v| (n.clone(), v))
                    .map_err(io_err)
            })
            .collect::<std::io::Result<_>>()?;
        let bytes = safetensors::serialize(views, &None).map_err(io_err)?;
        // Write-then-rename so a crash mid-write cannot leave a torn sidecar
        // that would poison the next resume.
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes).map_err(io_err)?;
        let read_f32 = |name: &str| -> std::io::Result<Vec<f32>> {
            Ok(bytes_to_f32(st.tensor(name).map_err(io_err)?.data()))
        };
        let read_f64 = |name: &str| -> std::io::Result<f64> {
            let d = st.tensor(name).map_err(io_err)?;
            let b: [u8; 8] = d.data()[..8].try_into().map_err(io_err)?;
            Ok(f64::from_le_bytes(b))
        };
        let read_ms = |prefix: &str| -> std::io::Result<MomentSet> {
            let dims: Vec<usize> = read_f32(&format!("{prefix}.dims"))?
                .into_iter()
                .map(|x| x as usize)
                .collect();
            let l = dims.len().saturating_sub(1);
            let mut ms = MomentSet { dims, ..Default::default() };
            for i in 0..l {
                ms.mw.push(read_f32(&format!("{prefix}.mw_{i}"))?);
                ms.vw.push(read_f32(&format!("{prefix}.vw_{i}"))?);
                ms.mb.push(read_f32(&format!("{prefix}.mb_{i}"))?);
                ms.vb.push(read_f32(&format!("{prefix}.vb_{i}"))?);
            }
            if !ms.is_consistent() {
                return Err(io_err_msg(format!("{prefix}: moment shapes disagree with dims")));
            }
            Ok(ms)
        };
        let terrain = read_f32("terrain")
            .unwrap_or_default()
            .chunks_exact(3)
            .map(|c| (c[0] as u32, c[1] as u32, c[2] as u32))
            .collect();
        Ok(TrainState {
            actor: read_ms("actor")?,
            critic: read_ms("critic")?,
            gstep: read_f64("gstep")? as u64,
            iter_done: read_f64("iter_done")? as u64,
            best_ema: read_f32("best_ema")?[0],
            rew_ema: read_f32("rew_ema")?[0],
            terrain,
        })
    }

    /// True when both moment sets match the architecture about to be trained.
    /// The trainer uses this to decide between restoring and cold-starting.
    pub fn matches(&self, actor_dims: &[usize], critic_dims: &[usize]) -> bool {
        self.actor.dims == actor_dims && self.critic.dims == critic_dims
    }
}

fn f32_vec_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn io_err<E: std::fmt::Debug>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e:?}"))
}

fn io_err_msg(s: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> TrainState {
        let ms = |dims: Vec<usize>| {
            let l = dims.len() - 1;
            MomentSet {
                mw: (0..l).map(|i| vec![i as f32 + 0.5; dims[i + 1] * dims[i]]).collect(),
                vw: (0..l).map(|i| vec![i as f32 + 1.5; dims[i + 1] * dims[i]]).collect(),
                mb: (0..l).map(|i| vec![i as f32 + 2.5; dims[i + 1]]).collect(),
                vb: (0..l).map(|i| vec![i as f32 + 3.5; dims[i + 1]]).collect(),
                dims,
            }
        };
        TrainState {
            actor: ms(vec![7, 5, 3]),
            critic: ms(vec![4, 6, 1]),
            gstep: 1_234_567_890,
            iter_done: 49_999,
            best_ema: 0.0335,
            rew_ema: 0.0241,
            terrain: vec![(3, 1, 2), (0, 0, 9)],
        }
    }

    #[test]
    fn round_trips() {
        let p = std::env::temp_dir().join("zealot_trainstate_rt.train");
        let p = p.to_str().unwrap();
        let a = sample();
        a.save(p).unwrap();
        let b = TrainState::load(p).unwrap();
        assert_eq!(a.gstep, b.gstep);
        assert_eq!(a.iter_done, b.iter_done);
        assert_eq!(a.best_ema, b.best_ema);
        assert_eq!(a.rew_ema, b.rew_ema);
        assert_eq!(a.terrain, b.terrain);
        assert_eq!(a.actor.dims, b.actor.dims);
        assert_eq!(a.actor.mw, b.actor.mw);
        assert_eq!(a.actor.vb, b.actor.vb);
        assert_eq!(a.critic.vw, b.critic.vw);
        assert!(b.matches(&[7, 5, 3], &[4, 6, 1]));
        assert!(!b.matches(&[7, 5, 4], &[4, 6, 1]));
        let _ = std::fs::remove_file(p);
    }

    /// A large step count must survive exactly — f32 would round it.
    #[test]
    fn gstep_keeps_integer_precision() {
        let p = std::env::temp_dir().join("zealot_trainstate_gstep.train");
        let p = p.to_str().unwrap();
        let mut a = sample();
        a.gstep = (1u64 << 40) + 12_345;
        a.save(p).unwrap();
        assert_eq!(TrainState::load(p).unwrap().gstep, a.gstep);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn missing_sidecar_is_an_error_not_a_panic() {
        assert!(TrainState::load("/nonexistent/zealot/no.train").is_err());
    }
}
