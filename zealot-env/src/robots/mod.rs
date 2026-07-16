//! Robot asset specifications — the analogue of AGILE's `assets/robots/*.py`.
//!
//! A robot spec is **pure data**: the kinematic/actuation facts a task needs that
//! don't belong to any single MDP — joint names and ordering, default ("home")
//! pose, per-joint PD gains and effort limits, action scales, and where the
//! MJCF/URDF live. It is deliberately free of nexus / GPU types so it builds and
//! unit-tests without the `cargo-gpu` toolchain; the env loop (Isaac Lab
//! "manager" tier) consumes it once physics is wired.
//!
//! All supported robots are 12-DOF lower-body walkers (6 per leg), so the
//! observation/action layout is identical across them and [`RobotSpec`] is one
//! concrete struct rather than a trait. Select at runtime with `BIPED_ROBOT`
//! (see [`RobotSpec::from_env`]):
//!
//! - `lerobot` (default) — the LeRobot Humanoid bipedal platform.
//! - `g1` — Unitree G1, the official legs-only 12-DOF model (upper body fused
//!   into the pelvis), gains from unitree_rl_gym.
//! - `h2plus` — Unitree H2 Plus, legs-only model generated from the official
//!   URDF (no public RL config exists; gains are mass-scaled from the G1's).

pub mod lerobot_bipedal;
pub mod unitree_g1;
pub mod unitree_h2_plus;

pub use lerobot_bipedal::LeRobotBipedal;

/// Number of actuated leg DOFs (6 per leg) — shared by every supported robot.
pub const NUM_JOINTS: usize = 12;

/// Per-joint specification: gains, limits, action scale, and home pose.
///
/// One entry per actuated DOF. `kp`/`kd` are the position/velocity gains of the
/// joint's PD controller (stiffness / damping); the env applies
/// `τ = kp·(q_target − q) − kd·q̇`, saturated at `effort_limit`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointSpec {
    /// MJCF/URDF joint name (e.g. `"hipy_left"`, `"left_hip_pitch_joint"`).
    pub name: &'static str,
    /// Position gain (stiffness), N·m/rad.
    pub kp: f32,
    /// Velocity gain (damping), N·m·s/rad.
    pub kd: f32,
    /// Torque saturation, N·m.
    pub effort_limit: f32,
    /// Velocity limit, rad/s (soft-limit penalties reference this).
    pub vel_limit: f32,
    /// Action scale: `q_target = default_pos + scale · action`, action ∈ ~[-1, 1].
    pub action_scale: f32,
    /// Default ("home") joint position, rad.
    pub default_pos: f32,
    /// Hard joint position limits `(lower, upper)`, rad (from the model).
    pub pos_limit: (f32, f32),
    /// Rotor/reflected inertia (kg·m²), added to the joint's dof inertia — what
    /// makes stiff PD-controlled joints numerically stable in sim.
    pub armature: f32,
    /// Passive joint damping (N·m·s/rad). Folded into the motor's velocity gain
    /// since the nexus passive-damping buffer is otherwise a hardcoded 0.1
    /// default. The MJCF `damping` attr, when present, takes precedence.
    pub damping: f32,
    /// Coulomb joint friction (N·m) — a constant torque opposing motion,
    /// applied as `-frictionloss·sign(q̇)` via the nexus `dof_frictionloss`
    /// buffer (energy dissipation / stiction).
    pub frictionloss: f32,
}

/// A 12-DOF bipedal robot: joints in **canonical policy order** (the order in
/// which joint quantities appear in observations and actions) plus the
/// robot-level facts the env and task need. Pure `'static` data — every robot
/// is a `const fn` constructor.
#[derive(Clone, Copy, Debug)]
pub struct RobotSpec {
    /// Short identifier (`"lerobot"`, `"unitree_g1"`, `"unitree_h2_plus"`).
    pub name: &'static str,
    /// Per-joint specs, in canonical policy order.
    pub joints: [JointSpec; NUM_JOINTS],
    /// Root/base link name in the model.
    pub base_link: &'static str,
    /// Left / right foot link names (contact + foot-reward references). Must be
    /// the links carrying the sole `class="collision"` capsules in the MJCF.
    pub foot_links: [&'static str; 2],
    /// The foot's "forward" direction in the foot link's LOCAL frame (used by
    /// the foot-yaw-vs-base observation). +X for models whose foot frame is
    /// upright; the Unitree G1's foot frame is axis-normalized so its forward
    /// is +Z (see `tools/convert_unitree_biped.py`).
    pub foot_forward_local: [f32; 3],
    /// Foot-link-origin height (m, world) below which the foot counts as in
    /// ground contact. Robot-specific because the sole-to-link-origin offset
    /// differs (lerobot/G1 ≈ 0.035 m, H2 Plus ≈ 0.054 m). `BIPED_CONTACT_Z`
    /// overrides.
    pub foot_contact_z: f32,
    /// Base height in the DEFAULT (home) pose with the sole on the ground, m —
    /// the base-height reward target.
    pub base_height: f32,
    /// Straight-leg (all joints at 0) base height with the sole exactly on the
    /// ground, m — the spawn height (the multibody rest pose is q = 0).
    pub spawn_z: f32,
    /// Fall-termination floor on base height, m. Below this the episode ends —
    /// this is what stops the policy reward-hacking by sinking the
    /// (collider-less) torso through the ground while staying upright.
    pub min_base_height: f32,
    /// Total robot mass, kg (≈ sum of model link masses).
    pub total_mass: f32,
    /// MJCF path (zealot dialect — see `tools/convert_unitree_biped.py`),
    /// relative to `$HOME`.
    pub mjcf_rel_path: &'static str,
    /// Source URDF path, relative to `$HOME` (kept for the URDF-based smoke
    /// examples and provenance).
    pub urdf_rel_path: &'static str,
    /// Left↔right mirror permutation over canonical joint indices:
    /// `mirror[i]` is the contralateral counterpart of joint `i`.
    pub mirror: [usize; NUM_JOINTS],
    /// Mirror sign per joint: sagittal joints (pitch/knee) mirror equal (+1),
    /// lateral joints (roll/yaw) mirror opposite (−1). Used by the bilateral
    /// symmetry reward and the trainers' mirror-symmetry loss.
    pub mirror_sign: [f32; NUM_JOINTS],
    /// Canonical indices of the four hip yaw/roll DOFs (the jittery lateral
    /// hips), for the targeted action-rate penalty.
    pub hip_yawroll: [usize; 4],
    /// Link-name fragments identifying the leg links that have NO ground
    /// collider and must never legitimately touch the floor (WBC-AGILE-style
    /// `illegal_contact` termination). Ankle + foot links must NOT match.
    pub illegal_ground_fragments: &'static [&'static str],
    /// Left/right link pairs (foot, shin, thigh) for the leg-crossing
    /// self-collision guard: terminate if any pair gets closer than
    /// `BIPED_SELF_COLL_DIST`.
    pub self_collision_pairs: &'static [(&'static str, &'static str)],
    /// PD gains for NON-action joints the model may carry (e.g. the G1
    /// 29-DOF body's waist/arms, which the sim holds at the rest pose while
    /// the policy drives the legs): `(name_fragment, kp, kd, effort_limit)`,
    /// first matching fragment wins. Joints matching nothing fall back to the
    /// env's generic holding gains. Empty for legs-only models.
    pub held_joints: &'static [(&'static str, f32, f32, f32)],
}

impl RobotSpec {
    /// Select the robot from `BIPED_ROBOT` (default: `lerobot`).
    pub fn from_env() -> Self {
        let name = std::env::var("BIPED_ROBOT").unwrap_or_default();
        match name.as_str() {
            "" | "lerobot" => lerobot_bipedal::lerobot(),
            "g1" | "unitree_g1" => unitree_g1::unitree_g1(),
            "g1_agile" => unitree_g1::unitree_g1_agile(),
            "g1_29dof_agile" => unitree_g1::unitree_g1_29dof_agile(),
            "g1_29dof" | "g1_29" | "g1full" => unitree_g1::unitree_g1_29dof(),
            "h2plus" | "h2_plus" | "unitree_h2_plus" => unitree_h2_plus::unitree_h2_plus(),
            other => {
                panic!("unknown BIPED_ROBOT '{other}' (expected lerobot | g1 | g1_agile | g1_29dof | g1_29dof_agile | h2plus)")
            }
        }
    }

    /// Default joint positions in canonical order (the home pose targets).
    pub fn default_pose(&self) -> [f32; NUM_JOINTS] {
        std::array::from_fn(|i| self.joints[i].default_pos)
    }

    /// Per-joint action scales in canonical order.
    pub fn action_scales(&self) -> [f32; NUM_JOINTS] {
        std::array::from_fn(|i| self.joints[i].action_scale)
    }

    /// Absolute path to the zealot-dialect MJCF, resolved against `$HOME`.
    pub fn mjcf_path(&self) -> std::path::PathBuf {
        Self::home().join(self.mjcf_rel_path)
    }

    /// Absolute path to the source URDF, resolved against `$HOME`.
    pub fn urdf_path(&self) -> std::path::PathBuf {
        Self::home().join(self.urdf_rel_path)
    }

    fn home() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_invariants(r: &RobotSpec) {
        // Unique joint names.
        let mut seen = std::collections::HashSet::new();
        for j in &r.joints {
            assert!(seen.insert(j.name), "{}: duplicate joint {}", r.name, j.name);
        }
        // Mirror is a sign-consistent involution pairing distinct joints.
        for i in 0..NUM_JOINTS {
            let m = r.mirror[i];
            assert_ne!(m, i, "{}: joint {i} mirrors itself", r.name);
            assert_eq!(r.mirror[m], i, "{}: mirror not an involution at {i}", r.name);
            assert_eq!(
                r.mirror_sign[i], r.mirror_sign[m],
                "{}: mirror sign mismatch {i}<->{m}",
                r.name
            );
        }
        // hip_yawroll indices are valid and distinct.
        let mut hy = std::collections::HashSet::new();
        for &i in &r.hip_yawroll {
            assert!(i < NUM_JOINTS);
            assert!(hy.insert(i), "{}: duplicate hip_yawroll idx {i}", r.name);
        }
        // Default pose respects the joint limits; sane scalar facts.
        for j in &r.joints {
            assert!(j.pos_limit.0 <= j.default_pos && j.default_pos <= j.pos_limit.1);
            assert!(j.kp > 0.0 && j.kd > 0.0 && j.effort_limit > 0.0);
        }
        assert!(r.min_base_height < r.base_height);
        assert!(r.base_height <= r.spawn_z + 0.05); // home pose ≈ slightly crouched
        assert!(r.total_mass > 1.0);
    }

    #[test]
    fn all_robots_consistent() {
        check_invariants(&lerobot_bipedal::lerobot());
        check_invariants(&unitree_g1::unitree_g1());
        check_invariants(&unitree_g1::unitree_g1_29dof());
        check_invariants(&unitree_h2_plus::unitree_h2_plus());
    }
}
