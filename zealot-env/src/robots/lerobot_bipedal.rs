//! The **LeRobot Humanoid** bipedal platform (lower body, "no-arms") — the real
//! target robot from the lerobot team's `lerobot-humanoid-design`.
//!
//! This mirrors AGILE's `assets/robots/booster_t1.py`, but as pure-data Rust. The
//! numbers are taken from the working `Mjlab-Velocity-Flat-LeRobot-Humanoid-no-arms`
//! policy (the one already deployed on the physical robot via
//! `to_real_robot/`) so a policy trained here stays swap-compatible with that
//! deployment path:
//!
//! - **DOF / kinematics**: from the URDF
//!   `lerobot-humanoid-design/urdf/bipedal_plateform/urdf/robot.urdf` — 12
//!   revolute leg joints, all `axis = (0,0,1)` in their local frame, limits ±π,
//!   root link `torso_subassembly`, total mass ≈ 10.2 kg.
//! - **PD gains**: from that policy's `gain.md`.
//! - **Action scales**: from the policy's `actions.joint_pos.scale` (per joint).
//! - **Default pose**: all joints at 0 rad (the mjlab `init_state` is neutral).
//! - **Effort limits**: from the mjlab actuator config (the URDF's `effort=10` is a
//!   placeholder the trainer overrides).
//!
//! ## Joint ordering
//!
//! [`JOINT_NAMES`] is the **canonical policy order** for this crate: the order in
//! which joint quantities appear in observations and actions. It is provisional —
//! it will be reconciled in two places as later stages land:
//! 1. the nexus/rapier multibody DOF order produced from the URDF (the env loop
//!    builds a DOF↔policy-index map), and
//! 2. the mjlab/deployment order the `to_real_robot` adapter expects (for ONNX
//!    export / sim-to-real). The adapter already remaps orders, so this choice
//!    only fixes *our* internal convention, not the physics.

/// Number of actuated leg DOFs (6 per leg).
pub const NUM_JOINTS: usize = 12;

/// Per-joint specification: gains, limits, action scale, and home pose.
///
/// One entry per actuated DOF. `kp`/`kd` are the position/velocity gains of the
/// joint's PD controller (stiffness / damping); the env applies
/// `τ = kp·(q_target − q) − kd·q̇`, saturated at `effort_limit`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointSpec {
    /// URDF joint name (e.g. `"hipy_left"`).
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
    /// Hard joint position limits `(lower, upper)`, rad (from the URDF).
    pub pos_limit: (f32, f32),
    /// Rotor/reflected inertia (kg·m²), system-identified by WBC-AGILE
    /// (`config.yaml` `joint_armature`). Added to the joint's dof inertia — what
    /// makes stiff PD-controlled joints numerically stable in sim.
    pub armature: f32,
    /// Passive joint damping (N·m·s/rad), from the MJCF `damping`. The real
    /// joints are significantly damped (0.5–2.3); without it the sim joints slew
    /// far too fast (~50 rad/s). Folded into the motor's velocity gain since the
    /// nexus passive-damping buffer is otherwise a hardcoded 0.1 default.
    pub damping: f32,
    /// Coulomb joint friction (N·m), from the MJCF `frictionloss` — a constant
    /// torque opposing motion. Applied as `-frictionloss·sign(q̇)` via the nexus
    /// `dof_frictionloss` buffer (energy dissipation / stiction).
    pub frictionloss: f32,
}

const PI: f32 = std::f32::consts::PI;

/// Canonical policy joint order (see module docs). Alphabetical by URDF joint
/// name — the order the mjlab trainer resolves to with `preserve_order = false`,
/// chosen so eventual ONNX export lines up with the deployment adapter.
pub const JOINT_NAMES: [&str; NUM_JOINTS] = [
    "anklex_left",
    "anklex_right",
    "ankley_left",
    "ankley_right",
    "hipx_left",
    "hipx_right",
    "hipy_left",
    "hipy_right",
    "hipz_left",
    "hipz_right",
    "knee_left",
    "knee_right",
];

/// Per-joint-*family* gains/scales (the two legs are symmetric, so left and right
/// of the same family share values). Effort limits: hips & knees 88 N·m, ankles
/// 44 N·m. Action scales and PD gains are from the deployed policy's `gain.md` /
/// action config.
const fn family(name: &'static str) -> JointSpec {
    // (kp, kd, effort, action_scale, armature, pos_limit) per joint family.
    // Armature is WBC-AGILE's system-identified rotor inertia. `pos_limit` is the
    // real per-joint range from the MJCF (mjlab `range`, rad) — NOT the old ±π
    // placeholder, which let the ankle fold the foot into its own shin. The model
    // ranges are mildly L/R-asymmetric; we use the symmetric magnitude (enough to
    // stop the over-flex). The nexus env prefers an explicit MJCF `range` when the
    // model provides one (`MjBody::joint_range`) and falls back to this.
    // ...also the MJCF per-joint passive `damping` (N·m·s/rad) — real joints are
    // damped 0.5–2.3; the sim default (0.1) leaves them slewing at ~50 rad/s.
    // ...and the MJCF per-joint `frictionloss` (N·m, Coulomb) — last tuple slot.
    let (kp, kd, effort, scale, armature, lim, damping, frictionloss) = if starts_with(name, "hipz") {
        (30.0, 3.0, 88.0, 0.733, 0.0227, (-0.349, 0.349), 0.514, 1.351)
    } else if starts_with(name, "hipx") {
        (40.0, 3.0, 88.0, 0.55, 0.1333, (-0.349, 0.349), 0.738, 1.158)
    } else if starts_with(name, "hipy") {
        (60.0, 4.0, 88.0, 0.367, 0.1408, (-1.047, 1.047), 1.455, 1.312)
    } else if starts_with(name, "knee") {
        (60.0, 4.0, 88.0, 0.367, 0.1233, (-0.524, 0.524), 2.264, 0.998)
    } else if starts_with(name, "anklex") {
        (20.0, 1.5, 44.0, 0.55, 0.0299, (-0.175, 0.175), 0.214, 0.262) // ankle-roll
    } else {
        // ankley (ankle pitch) — the one that folds the foot into the shin.
        (20.0, 1.5, 44.0, 0.55, 0.0299, (-0.349, 0.349), 0.0286, 0.171)
    };
    JointSpec {
        name,
        kp,
        kd,
        effort_limit: effort,
        vel_limit: 10.0,
        action_scale: scale,
        default_pos: 0.0,
        pos_limit: lim,
        armature,
        damping,
        frictionloss,
    }
}

/// `const`-friendly `str::starts_with` for ASCII prefixes (so [`family`] can be
/// evaluated at compile time).
const fn starts_with(s: &str, prefix: &str) -> bool {
    let (s, p) = (s.as_bytes(), prefix.as_bytes());
    if p.len() > s.len() {
        return false;
    }
    let mut i = 0;
    while i < p.len() {
        if s[i] != p[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// The LeRobot bipedal platform spec.
#[derive(Clone, Copy, Debug)]
pub struct LeRobotBipedal {
    /// Per-joint specs, in [`JOINT_NAMES`] order.
    pub joints: [JointSpec; NUM_JOINTS],
    /// Root/base link name in the URDF.
    pub base_link: &'static str,
    /// Left / right foot link names (contact + foot-reward references).
    pub foot_links: [&'static str; 2],
    /// Nominal base height above ground in the home pose, metres. Used for the
    /// initial spawn pose and as a height-reward reference.
    pub base_height: f32,
    /// Total robot mass, kg (≈ sum of URDF link masses).
    pub total_mass: f32,
    /// URDF path, relative to `$HOME` (the asset lives outside this repo, in the
    /// sibling `lerobot-humanoid-design` checkout).
    pub urdf_rel_path: &'static str,
}

impl Default for LeRobotBipedal {
    fn default() -> Self {
        Self::new()
    }
}

impl LeRobotBipedal {
    /// Build the spec with the values from the deployed velocity policy.
    pub const fn new() -> Self {
        // `JOINT_NAMES` is const, so unroll the family lookup per index.
        let joints = [
            family(JOINT_NAMES[0]),
            family(JOINT_NAMES[1]),
            family(JOINT_NAMES[2]),
            family(JOINT_NAMES[3]),
            family(JOINT_NAMES[4]),
            family(JOINT_NAMES[5]),
            family(JOINT_NAMES[6]),
            family(JOINT_NAMES[7]),
            family(JOINT_NAMES[8]),
            family(JOINT_NAMES[9]),
            family(JOINT_NAMES[10]),
            family(JOINT_NAMES[11]),
        ];
        Self {
            joints,
            base_link: "torso_subassembly",
            foot_links: ["foot_left", "foot_right"],
            // Lower-body platform stands ~0.5 m at the torso mount; refined once
            // the URDF is dropped on the ground in the physics smoke test.
            base_height: 0.5,
            total_mass: 10.18,
            urdf_rel_path: "Documents/work/lerobot-humanoid-design/urdf/bipedal_plateform/urdf/robot.urdf",
        }
    }

    /// Default joint positions in [`JOINT_NAMES`] order (the home pose targets).
    pub fn default_pose(&self) -> [f32; NUM_JOINTS] {
        std::array::from_fn(|i| self.joints[i].default_pos)
    }

    /// Per-joint action scales in [`JOINT_NAMES`] order.
    pub fn action_scales(&self) -> [f32; NUM_JOINTS] {
        std::array::from_fn(|i| self.joints[i].action_scale)
    }

    /// Absolute path to the URDF, resolved against `$HOME`.
    pub fn urdf_path(&self) -> std::path::PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        std::path::Path::new(&home).join(self.urdf_rel_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_names_unique_and_complete() {
        let mut seen = std::collections::HashSet::new();
        for n in JOINT_NAMES {
            assert!(seen.insert(n), "duplicate joint name {n}");
        }
        assert_eq!(seen.len(), NUM_JOINTS);
    }

    #[test]
    fn gains_match_families() {
        let r = LeRobotBipedal::new();
        let by = |name: &str| r.joints.iter().find(|j| j.name == name).copied().unwrap();
        assert_eq!(by("hipz_left").kp, 30.0);
        assert_eq!(by("hipx_right").kp, 40.0);
        assert_eq!(by("hipy_left").kp, 60.0);
        assert_eq!(by("knee_right").kp, 60.0);
        assert_eq!(by("ankley_left").kp, 20.0);
        assert_eq!(by("anklex_right").effort_limit, 44.0);
        assert_eq!(by("knee_left").effort_limit, 88.0);
        // Action scales per family.
        assert_eq!(by("hipz_left").action_scale, 0.733);
        assert_eq!(by("hipy_right").action_scale, 0.367);
        assert_eq!(by("anklex_left").action_scale, 0.55);
    }

    #[test]
    fn default_pose_is_neutral() {
        let r = LeRobotBipedal::new();
        assert_eq!(r.default_pose(), [0.0; NUM_JOINTS]);
    }

    #[test]
    fn const_starts_with() {
        assert!(starts_with("hipz_left", "hipz"));
        assert!(!starts_with("hipx_left", "hipz"));
        assert!(!starts_with("hi", "hipz"));
    }
}
