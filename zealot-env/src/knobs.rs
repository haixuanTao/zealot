//! THE config — every runtime knob that more than one crate/site consumes is
//! declared here EXACTLY ONCE (env-var name + default + doc), parsed once,
//! cached for the process lifetime.
//!
//! Motivation: knobs used to be `std::env::var` reads scattered per call site,
//! each with its own copy of the default. When a default changed, every copy
//! had to be found — and when one was missed, the two sites silently
//! disagreed (v29 trained 16 h single-frame because the trainer's
//! `BIPED_OBS_HISTORY` default said 5 while the env's said 1).
//!
//! Rules:
//! - A knob read by MORE THAN ONE site MUST live here. Single-site knobs may
//!   stay local, but never duplicate a default.
//! - wasm/demo overrides: call [`Knob::set_override`] before first use (env
//!   vars don't exist in the browser). Overrides beat env; env beats default.

use std::str::FromStr;
use std::sync::OnceLock;

/// One runtime knob: env-var name, default, one-time parse + cache.
pub struct Knob<T: 'static> {
    name: &'static str,
    default: T,
    cell: OnceLock<T>,
    ovr: OnceLock<T>,
}

impl<T: Copy + FromStr> Knob<T> {
    pub const fn new(name: &'static str, default: T) -> Self {
        Self {
            name,
            default,
            cell: OnceLock::new(),
            ovr: OnceLock::new(),
        }
    }

    /// Resolve: override → env var → default. Cached on first call.
    pub fn get(&self) -> T {
        *self.cell.get_or_init(|| {
            if let Some(v) = self.ovr.get() {
                return *v;
            }
            std::env::var(self.name)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(self.default)
        })
    }

    /// Programmatic override (wasm demos). Must run before the first `get`.
    pub fn set_override(&self, v: T) {
        let _ = self.ovr.set(v);
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// Boolean flags: default ON, `<name>=0` disables (the production convention).
pub struct Flag {
    name: &'static str,
    default_on: bool,
    cell: OnceLock<bool>,
    ovr: OnceLock<bool>,
}

impl Flag {
    pub const fn new(name: &'static str, default_on: bool) -> Self {
        Self {
            name,
            default_on,
            cell: OnceLock::new(),
            ovr: OnceLock::new(),
        }
    }

    pub fn get(&self) -> bool {
        *self.cell.get_or_init(|| {
            if let Some(v) = self.ovr.get() {
                return *v;
            }
            match std::env::var(self.name).ok().as_deref() {
                Some("0") => false,
                Some(_) => true,
                None => self.default_on,
            }
        })
    }

    pub fn set_override(&self, v: bool) {
        let _ = self.ovr.set(v);
    }
}

// ---------------------------------------------------------------------------
// The multi-site knobs (each was read in ≥2 places with per-site defaults).
// Values = the production config (see scripts/train.sh — bare run == prod).
// ---------------------------------------------------------------------------

/// Actor observation frame stacking (1 = off). MUST be one value for the
/// env's ring buffer AND the trainer's network sizing.
pub static OBS_HISTORY: Knob<usize> = Knob::new("BIPED_OBS_HISTORY", 5);

/// Rough-terrain difficulty curriculum.
pub static TERRAIN: Flag = Flag::new("BIPED_TERRAIN", true);

/// Force-based foot-contact sensing (feeds contact obs + force_rate reward).
pub static CONTACT_SENSE: Flag = Flag::new("BIPED_CONTACT_SENSE", true);

/// Per-pair contact-manifold reduction (≤4 points; the terrain perf lever).
pub static CONTACT_REDUCE: Flag = Flag::new("BIPED_CONTACT_REDUCE", true);

/// Contact-buffer pre-size per batch.
pub static CONTACT_CAP: Knob<u32> = Knob::new("BIPED_CONTACT_CAP", 128);

/// Contact natural frequency (Hz) — pinned production value.
pub static CONTACT_NF: Knob<f32> = Knob::new("BIPED_CONTACT_NF", 240.0);

/// Contact damping ratio — pinned production value.
pub static CONTACT_DR: Knob<f32> = Knob::new("BIPED_CONTACT_DR", 1.0);

/// Physics decimation (substeps per control step at fixed control_dt=0.02).
pub static DECIMATION: Knob<u32> = Knob::new("BIPED_DECIMATION", 4);

/// Substep refresh mode: 0 = refresh dynamics + constraints ONCE per control
/// step (default), 1 = rebuild them every substep, 2 = light split cadence
/// (constraints per substep, M/LU per step).
///
/// This is the single biggest physics-cost dial in the stack. Measured on a
/// 5090 at N=4096 with the production config: mode 1 costs 114.6 ms of GPU per
/// control step vs 22.5 ms at mode 0 — 4.4 s/iter vs 2.1 s/iter, i.e. per-substep
/// refresh is ~5x the physics bill for the whole trainer.
///
/// The knob arrived with the nexus branch that added per-substep refresh and
/// defaulted to 1, which silently halved throughput relative to v28 (2.5 s/iter,
/// 62 ms/step) — that regression is what this default reverses.
///
/// NOTE: refresh cadence changes the physics slightly, so train and eval with
/// the SAME setting; a policy trained at mode 1 is not guaranteed to transfer to
/// mode 0 unchanged.
pub static SUBSTEP_REFRESH: Knob<u32> = Knob::new("NEXUS_SUBSTEP_REFRESH", 0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_resolve_without_env() {
        // The dangerous pair that diverged in v29: one source of truth now.
        assert_eq!(OBS_HISTORY.get(), 5);
        assert!(TERRAIN.get());
        assert_eq!(DECIMATION.get(), 4);
    }

    #[test]
    fn override_beats_default() {
        static K: Knob<u32> = Knob::new("ZEALOT_TEST_KNOB_XYZ", 7);
        K.set_override(9);
        assert_eq!(K.get(), 9);
    }
}
