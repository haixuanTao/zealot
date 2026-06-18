//! **Flat velocity tracking** for the LeRobot bipedal — the first milestone:
//! reproduce, in Rust on nexus, the deployed
//! `Mjlab-Velocity-Flat-LeRobot-Humanoid-no-arms` policy.
//!
//! The robot walks on flat ground tracking a commanded planar velocity
//! `(vx, vy, yaw_rate)`. This file is the *task config* (Isaac Lab tier): the
//! observation layout, the action→joint-target map, the reward terms with their
//! weights, the command sampler, and the termination rule — ported from the
//! mjlab config of that policy and AGILE's T1 velocity task.
//!
//! It is pure CPU code over [`RobotState`]; the env loop fills that struct from
//! nexus readback (`dof_values`/`dof_state` + base pose/velocity) and applies the
//! [`VelocityFlatTask::joint_targets`] output through the joint PD controllers.
//!
//! ## Frame convention (Z-up)
//!
//! The env runs Z-up, matching the MuJoCo model and the real robot: base axes
//! **forward = +X, lateral = +Y, up = +Z**; yaw is rotation about +Z. (Verified
//! against the spawned MJCF stance in `biped_mjcf`.)
//!
//! ## Reward terms
//!
//! Proprioceptive terms (velocity/yaw tracking, upright, posture, symmetry,
//! action-rate, joint penalties) and the **foot-contact** terms (feet air-time,
//! foot-slip, foot-clearance) are all live — the env fills [`FootObs`] from the
//! simulator's contacts. A few minor deployed terms remain unimplemented
//! (action-FFT smoothness band, angular-momentum, soft-landing) and
//! self-collision is N/A here (only the feet carry colliders).

use crate::math::{quat_rotate, quat_rotate_inv};
use crate::rng::Lcg;
use crate::robots::lerobot_bipedal::{LeRobotBipedal, NUM_JOINTS};

/// Body axis indices under the Z-up convention (see module docs).
pub const FWD: usize = 0;
/// Lateral axis index (Y).
pub const LAT: usize = 1;
/// Up axis index (Z).
pub const UP: usize = 2;

/// Observation vector length (policy group): `last_action(12) + command(4) +
/// joint_pos_rel(12) + joint_vel(12) + projected_gravity(3)`.
pub const OBS_DIM: usize = NUM_JOINTS + 4 + NUM_JOINTS + NUM_JOINTS + 3;
/// Action vector length: one position target per leg DOF.
pub const ACTION_DIM: usize = NUM_JOINTS;
/// Privileged (critic) observation length: policy obs plus base linear & angular
/// velocity in the body frame. Foot/contact privileged terms are deferred.
pub const CRITIC_OBS_DIM: usize = OBS_DIM + 3 + 3;

/// Base (root link) physics state, as read back from nexus each control step.
#[derive(Clone, Copy, Debug)]
pub struct BaseState {
    /// Orientation quaternion `(x, y, z, w)`, body→world.
    pub orientation: [f32; 4],
    /// Linear velocity in the world frame, m/s.
    pub lin_vel_world: [f32; 3],
    /// Angular velocity in the world frame, rad/s.
    pub ang_vel_world: [f32; 3],
    /// World height of the base, m (for height-based termination/reward).
    pub height: f32,
}

impl Default for BaseState {
    fn default() -> Self {
        Self {
            orientation: [0.0, 0.0, 0.0, 1.0],
            lin_vel_world: [0.0; 3],
            ang_vel_world: [0.0; 3],
            height: 0.5,
        }
    }
}

/// Full per-environment state the task reads. `last_action`/`prev_action` are the
/// policy outputs (pre-scale) from the last two control steps, kept here so the
/// action-rate rewards and the `actions` observation are self-contained.
/// Number of feet (contact bodies).
pub const NUM_FEET: usize = 2;

/// Per-foot state needed by the contact-shaped rewards. The env fills this from
/// the simulator's contacts; `air_time` is tracked across steps by the env.
#[derive(Clone, Copy, Debug)]
pub struct FootObs {
    /// Foot is touching the ground this step.
    pub contact: bool,
    /// Touchdown this step (was airborne last step, now in contact).
    pub first_contact: bool,
    /// Seconds the foot has been airborne (0 while in contact).
    pub air_time: f32,
    /// Foot world height, m.
    pub height: f32,
    /// Foot horizontal speed, m/s (for slip / clearance shaping).
    pub planar_speed: f32,
    /// Sole tilt from horizontal, rad (0 = sole flat on the ground). Used by the
    /// flat-foot reward so the robot plants its whole sole, not an edge/toe/heel.
    pub tilt: f32,
    /// Foot yaw RELATIVE to the base (rad; 0 = foot points the same direction as
    /// base). Computed by the env as `atan2(y, x)` of `q_base⁻¹·q_foot · X̂`. Used
    /// by `feet_yaw_mean`.
    pub yaw_rel_base: f32,
    /// Foot world horizontal position (m). The reward uses the difference between
    /// the two feet's positions, transformed into the base frame, to compute the
    /// lateral stance width.
    pub pos_xy: [f32; 2],
}

impl Default for FootObs {
    fn default() -> Self {
        // Grounded & still → zero foot-reward contribution.
        Self {
            contact: true,
            first_contact: false,
            air_time: 0.0,
            height: 0.0,
            planar_speed: 0.0,
            tilt: 0.0,
            yaw_rel_base: 0.0,
            pos_xy: [0.0, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RobotState {
    /// Base link state.
    pub base: BaseState,
    /// Joint positions, rad, in canonical joint order.
    pub joint_pos: [f32; NUM_JOINTS],
    /// Joint velocities, rad/s, in canonical joint order.
    pub joint_vel: [f32; NUM_JOINTS],
    /// Previous policy action (this step's `actions` observation).
    pub last_action: [f32; NUM_JOINTS],
    /// Action before that (for the action-rate-of-rate term).
    pub prev_action: [f32; NUM_JOINTS],
    /// Per-foot contact state (left, right).
    pub feet: [FootObs; NUM_FEET],
}

impl Default for RobotState {
    fn default() -> Self {
        Self {
            base: BaseState::default(),
            joint_pos: [0.0; NUM_JOINTS],
            joint_vel: [0.0; NUM_JOINTS],
            last_action: [0.0; NUM_JOINTS],
            prev_action: [0.0; NUM_JOINTS],
            feet: [FootObs::default(); NUM_FEET],
        }
    }
}

/// A commanded planar base velocity.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VelocityCommand {
    /// Forward velocity, m/s.
    pub vx: f32,
    /// Lateral velocity, m/s.
    pub vy: f32,
    /// Yaw rate, rad/s.
    pub yaw_rate: f32,
}

impl VelocityCommand {
    /// The 4-channel command observation `[vx, vy, yaw_rate, aux]`. The 4th
    /// channel is reserved (0.0) so the layout matches the deployed policy's
    /// 4-wide `twist` command until its exact meaning is reconciled.
    #[inline]
    pub fn obs(&self) -> [f32; 4] {
        [self.vx, self.vy, self.yaw_rate, 0.0]
    }

    /// Commanded planar speed (used for posture gating / zeroing).
    #[inline]
    pub fn speed(&self) -> f32 {
        (self.vx * self.vx + self.vy * self.vy + self.yaw_rate * self.yaw_rate).sqrt()
    }
}

/// Samples velocity commands like the mjlab/AGILE uniform velocity generator:
/// uniform ranges, a fraction of standing (zero) commands, and periodic
/// resampling. Ranges are the deployed policy's.
#[derive(Clone, Debug)]
pub struct CommandSampler {
    /// `(min, max)` forward velocity, m/s.
    pub lin_vel_x: (f32, f32),
    /// `(min, max)` lateral velocity, m/s.
    pub lin_vel_y: (f32, f32),
    /// `(min, max)` yaw rate, rad/s.
    pub ang_vel_z: (f32, f32),
    /// Probability a resample yields a standing (all-zero) command.
    pub standing_prob: f32,
    /// `(min, max)` resample interval, seconds.
    pub resample_s: (f32, f32),
}

impl Default for CommandSampler {
    fn default() -> Self {
        // From the deployed Mjlab-Velocity-Flat-LeRobot config.
        Self {
            // Reachable at our budget: 0.5 m/s is achievable, 0.8 is not. With an
            // unreachable max command the curriculum forces the policy into a
            // regime where tracking reward is uniformly tiny → it gives up.
            lin_vel_x: (-0.5, 0.5),
            lin_vel_y: (-0.3, 0.3),
            ang_vel_z: (-0.2, 0.2),
            standing_prob: 0.1,
            resample_s: (3.0, 8.0),
        }
    }
}

impl CommandSampler {
    /// Draw a fresh command.
    pub fn sample(&self, rng: &mut Lcg) -> VelocityCommand {
        if rng.chance(self.standing_prob) {
            return VelocityCommand::default();
        }
        VelocityCommand {
            vx: rng.range(self.lin_vel_x.0, self.lin_vel_x.1),
            vy: rng.range(self.lin_vel_y.0, self.lin_vel_y.1),
            yaw_rate: rng.range(self.ang_vel_z.0, self.ang_vel_z.1),
        }
    }

    /// Draw a resample interval in control steps, given the control dt.
    pub fn resample_steps(&self, rng: &mut Lcg, control_dt: f32) -> u32 {
        (rng.range(self.resample_s.0, self.resample_s.1) / control_dt).round() as u32
    }
}

/// Reward term weights (per-second; the task scales by the control dt). Defaults
/// are the deployed policy's. `deferred_*` weights are recorded for parity but
/// their terms aren't summed yet (they need contact readback).
#[derive(Clone, Copy, Debug)]
pub struct RewardWeights {
    /// Linear-velocity tracking (exp kernel), `std` below.
    pub track_lin_vel: f32,
    /// Angular (yaw) velocity tracking (exp kernel).
    pub track_ang_vel: f32,
    /// Upright / flat-orientation (exp kernel on tilt).
    pub upright: f32,
    /// Base-height tracking (exp kernel) — keeps the robot standing tall instead
    /// of crouching to trivially avoid falling.
    pub base_height: f32,
    /// Target base (torso) height, m (for `base_height`).
    pub base_height_target: f32,
    /// Hip yaw/roll deviation penalty gain (NEGATIVE). L2 penalty on hipz+hipx
    /// deviation from default, always-on — stops the policy from limit-riding the
    /// lateral hips into a splayed brace. (Was an unused full-posture reward.)
    pub pose: f32,
    /// Left/right symmetry (exp kernel on mirror error).
    pub bilateral_symmetry: f32,
    /// L2 penalty on action change.
    pub action_rate: f32,
    /// L2 penalty on action change for the hip yaw/roll DOFs only.
    pub action_rate_hipz_hipx: f32,
    /// L2 penalty on base roll/pitch angular velocity.
    pub body_ang_vel: f32,
    /// L2 penalty on vertical base velocity.
    pub lin_vel_z: f32,
    /// Soft joint-position-limit penalty.
    pub dof_pos_limits: f32,
    /// L2 penalty on joint velocity.
    pub dof_vel: f32,
    /// One-shot penalty applied by the env on a non-timeout termination.
    pub termination: f32,

    // --- foot-contact shaped (need per-foot contact from the sim) ---
    /// Feet air-time bonus on touchdown (shapes a stepping gait).
    pub air_time: f32,
    /// Penalty applied every step BOTH feet are off the ground. Forces a walking
    /// gait (always ≥1 foot planted) instead of hopping/bounding.
    pub flight: f32,
    /// Bonus applied (while moving) every step EXACTLY ONE foot is on the ground.
    /// Single-support is the defining phase of walking — rewarding it directly
    /// makes stepping beat both standing (double-support) and hopping (flight).
    pub single_support: f32,
    /// Foot-slip penalty (horizontal foot speed while in contact).
    pub foot_slip: f32,
    /// Foot-clearance penalty (swing-foot height vs target).
    pub foot_clearance: f32,
    /// Target swing-foot clearance height, m (for `foot_clearance`).
    pub foot_clearance_target: f32,
    /// Flat-foot penalty: penalizes sole tilt (rad²) of a foot IN CONTACT, so the
    /// robot plants its whole sole flat instead of balancing on a toe/heel/edge.
    pub foot_orientation: f32,
    /// WBC's `feet_yaw_mean_vs_base`: penalty on each foot's yaw (in base frame),
    /// summed over both feet. Strong because un-shaped foot yaw is the dominant
    /// posture artefact for under-constrained humanoid RL.
    pub feet_yaw_mean: f32,
    /// WBC's `feet_distance_from_ref` (lateral mode): penalises deviation of the
    /// lateral (body-Y) foot separation from `feet_distance_ref`.
    pub feet_distance: f32,
    /// Reference lateral foot separation, m (for `feet_distance`).
    pub feet_distance_ref: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        // Ported directly from WBC-AGILE's G1 locomotion `RewardsCfg` at
        // WBC-AGILE/agile/rl_env/tasks/locomotion/g1/velocity_history_env_cfg.py.
        // Where we share a term, the weight is matched verbatim. Where they have a
        // term we don't (torque-family — they expose torque obs, we use position
        // PD), we leave it out. Where we had a term they don't (`air_time`,
        // `single_support`, `foot_clearance`, `pose`), we zero it: WBC produces
        // stepping from "strong tracking + strong upright + jumping penalty"
        // without an explicit step bonus, and our additions were workarounds for
        // weights being too weak elsewhere.
        // ALIGNED to WBC-AGILE's *lerobot* velocity config (the one that actually
        // trained THIS robot), not the G1 config the old weights were ported from.
        // The G1 port over-penalized motion (ang_vel/lin_vel/action_rate 5x too
        // harsh) and used the wrong base-height target (0.62 vs the lerobot trunk
        // height 0.72 = spawn height), so the reward pushed the robot to crouch
        // into a fall and punished the very motion it needed to learn.
        // (Per-step terms are ×control_dt like Isaac Lab; `termination` is applied
        // once WITHOUT dt in the env, so -2.0 here ≈ WBC's -100·dt effective.)
        Self {
            track_lin_vel: 5.0,         // WBC 5.0
            track_ang_vel: 5.0,         // WBC 5.0
            upright: 5.0,               // ~WBC flat_orientation_l2 -5.0 (exp form here)
            base_height: 2.0,           // WBC 2.0
            base_height_target: 0.72,   // WBC DEFAULT_TRUNK_HEIGHT (was 0.62 — crouch bug)
            pose: -8.0, // hip yaw/roll deviation penalty (anti-limit-ride)
            bilateral_symmetry: 0.0,
            action_rate: -0.1,          // WBC -0.1 (was -0.25)
            action_rate_hipz_hipx: 0.0,
            body_ang_vel: -0.05,        // WBC ang_vel_xy -0.05 (was -0.25)
            lin_vel_z: -0.05,           // WBC -0.05 (was -0.25)
            dof_pos_limits: -0.5,       // WBC -0.1; strengthened to discourage limit-bracing
            dof_vel: -2e-4,             // WBC -2e-4 (was -1e-4)
            termination: -2.0,          // WBC is_terminated -100 ×dt(0.02) (was -25 one-shot)
            // Gait shaping — turned ON to drive a clean, TRANSFERABLE alternating
            // stride. The un-shaped reward let the policy track velocity with a
            // nexus-specific foot-shuffle (ankle-slam propulsion) that didn't
            // survive MuJoCo (sim2sim ratio 0.19 walking vs 1.00 standing). These
            // push toward real stepping: swing duration (air_time), exactly one
            // foot planted (single_support), no double-flight (flight), foot lifted
            // to a target clearance while swinging (foot_clearance), and — the key
            // anti-shuffle / anti-exploit term — a strong penalty on a planted foot
            // sliding (foot_slip, 50× the old WBC value).
            air_time: 1.5,              // reward completed swing time at touchdown
            flight: -1.0,               // penalize both feet airborne (no hopping)
            single_support: 1.0,        // bonus for the one-foot-down walking phase
            foot_slip: -0.5,            // STRONG: planted feet must not slide (was -0.01)
            foot_clearance: -1.0,       // lift the swing foot to the target height
            foot_clearance_target: 0.08,
            foot_orientation: -0.01,    // WBC feet_roll_l2 -0.01 (was -0.5)
            feet_yaw_mean: -0.4,        // WBC feet_yaw_mean_vs_base -0.4 (was -2.0)
            feet_distance: -0.02,       // WBC feet_distance_from_ref -0.02 (was -0.1)
            feet_distance_ref: 0.2,
        }
    }
}

/// Standard deviations of the exponential tracking kernels (`exp(-err²/std²)`),
/// from the deployed policy.
#[derive(Clone, Copy, Debug)]
pub struct RewardStds {
    /// Linear-velocity tracking std.
    pub lin_vel: f32,
    /// Angular-velocity tracking std.
    pub ang_vel: f32,
    /// Upright std (on horizontal projected-gravity components).
    pub upright: f32,
    /// Base-height std, m.
    pub base_height: f32,
    /// Posture std (on per-joint deviation, summed).
    pub pose: f32,
}

impl Default for RewardStds {
    fn default() -> Self {
        // WBC-AGILE values (`std=` on the matching RewTerm in the G1 config):
        //   track_lin_vel_xy_exp    std = 0.2
        //   track_ang_vel_z_exp     std = 0.2
        //   base_height_exp         std = 0.1
        //   flat_body_orientation   std = 10° = 0.1745 rad
        // WBC's std=0.2 is too tight at our reachable command range — at cmd=1.0
        // tracking reward is ~0 anywhere the policy can actually move, so there's
        // no useful gradient. Widen to 0.3 so a 0.3 m/s walk vs standing gives a
        // ~6× reward difference at the rollout's pinned cmd=0.4.
        Self {
            lin_vel: 0.3,
            ang_vel: 0.2,
            upright: 10_f32.to_radians(),
            base_height: 0.1,
            pose: 1.0, // unused (pose weight = 0) but kept for API stability
        }
    }
}

/// Per-term reward contributions for one step (already weighted and dt-scaled),
/// kept separate so training can log each term like rsl_rl's episode sums.
#[derive(Clone, Copy, Debug, Default)]
pub struct RewardBreakdown {
    /// Linear-velocity tracking contribution.
    pub track_lin_vel: f32,
    /// Angular-velocity tracking contribution.
    pub track_ang_vel: f32,
    /// Upright contribution.
    pub upright: f32,
    /// Base-height contribution.
    pub base_height: f32,
    /// Posture contribution.
    pub pose: f32,
    /// Symmetry contribution.
    pub bilateral_symmetry: f32,
    /// Action-rate penalty contribution.
    pub action_rate: f32,
    /// Hip-DOF action-rate penalty contribution.
    pub action_rate_hipz_hipx: f32,
    /// Base roll/pitch angular-velocity penalty contribution.
    pub body_ang_vel: f32,
    /// Vertical-velocity penalty contribution.
    pub lin_vel_z: f32,
    /// Joint-limit penalty contribution.
    pub dof_pos_limits: f32,
    /// Joint-velocity penalty contribution.
    pub dof_vel: f32,
    /// Feet air-time contribution.
    pub air_time: f32,
    /// Flight (both-feet-airborne) penalty contribution.
    pub flight: f32,
    /// Single-support (exactly-one-foot-down) bonus contribution.
    pub single_support: f32,
    /// Foot-slip penalty contribution.
    pub foot_slip: f32,
    /// Foot-clearance penalty contribution.
    pub foot_clearance: f32,
    /// Flat-foot (sole-tilt) penalty contribution.
    pub foot_orientation: f32,
    /// Foot-yaw-vs-base penalty contribution.
    pub feet_yaw_mean: f32,
    /// Lateral foot-distance penalty contribution.
    pub feet_distance: f32,
}

impl RewardBreakdown {
    /// Sum of all live terms — the scalar step reward (before any termination
    /// penalty, which the env applies separately).
    pub fn total(&self) -> f32 {
        self.track_lin_vel
            + self.track_ang_vel
            + self.upright
            + self.base_height
            + self.pose
            + self.bilateral_symmetry
            + self.action_rate
            + self.action_rate_hipz_hipx
            + self.body_ang_vel
            + self.lin_vel_z
            + self.dof_pos_limits
            + self.dof_vel
            + self.air_time
            + self.flight
            + self.single_support
            + self.foot_slip
            + self.foot_clearance
            + self.foot_orientation
            + self.feet_yaw_mean
            + self.feet_distance
    }
}

/// The flat velocity-tracking task.
#[derive(Clone, Debug)]
pub struct VelocityFlatTask {
    /// The robot spec (gains, default pose, limits, joint order).
    pub robot: LeRobotBipedal,
    /// Reward term weights.
    pub weights: RewardWeights,
    /// Tracking-kernel stds.
    pub stds: RewardStds,
    /// Physics timestep, s (200 Hz).
    pub sim_dt: f32,
    /// Control decimation (physics steps per control step).
    pub decimation: u32,
    /// Episode length, s.
    pub episode_s: f32,
    /// Termination tilt limit, rad (base up-axis vs world up).
    pub tilt_limit: f32,
    /// Termination floor on base height, m. Below this the episode ends — this is
    /// what stops the policy reward-hacking by sinking the (collider-less) torso
    /// through the ground while staying upright.
    pub min_base_height: f32,
    /// Indices of the hip yaw/roll DOFs (for `action_rate_hipz_hipx`).
    hip_yawroll_idx: [usize; 4],
}

impl Default for VelocityFlatTask {
    fn default() -> Self {
        Self::new()
    }
}

impl VelocityFlatTask {
    /// Build the task with the deployed policy's settings.
    pub fn new() -> Self {
        let robot = LeRobotBipedal::new();
        // Locate the hipz/hipx DOFs in canonical order for the targeted action-rate
        // penalty (these lateral-hip joints are the jittery ones).
        let mut hip_yawroll_idx = [0usize; 4];
        let mut k = 0;
        for (i, j) in robot.joints.iter().enumerate() {
            if j.name.starts_with("hipz") || j.name.starts_with("hipx") {
                hip_yawroll_idx[k] = i;
                k += 1;
            }
        }
        debug_assert_eq!(k, 4);
        // Reward-weight overrides for fast retuning without a rebuild. The
        // stand-still local optimum (policy collects upright + base_height +
        // free track_ang at zero command, ignores the velocity command) is the
        // walking blocker, so the key dials are the velocity-tracking weight vs
        // the standing magnets (upright / base_height). Set e.g.
        // `BIPED_W_TRACK_LIN=10 BIPED_W_UPRIGHT=3 BIPED_W_BASE_H=1.5` at launch.
        let mut weights = RewardWeights::default();
        let env_f32 = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<f32>().ok());
        if let Some(v) = env_f32("BIPED_W_TRACK_LIN") {
            weights.track_lin_vel = v;
        }
        if let Some(v) = env_f32("BIPED_W_TRACK_ANG") {
            weights.track_ang_vel = v;
        }
        if let Some(v) = env_f32("BIPED_W_UPRIGHT") {
            weights.upright = v;
        }
        if let Some(v) = env_f32("BIPED_W_BASE_H") {
            weights.base_height = v;
        }
        let mut stds = RewardStds::default();
        if let Some(v) = env_f32("BIPED_STD_LIN") {
            stds.lin_vel = v;
        }
        Self {
            robot,
            weights,
            stds,
            sim_dt: 1.0 / 200.0,
            decimation: 4,
            episode_s: 20.0,
            tilt_limit: 70.0_f32.to_radians(),
            min_base_height: 0.4,
            hip_yawroll_idx,
        }
    }

    /// Control timestep, s (`sim_dt · decimation`).
    #[inline]
    pub fn control_dt(&self) -> f32 {
        self.sim_dt * self.decimation as f32
    }

    /// Episode length in control steps.
    #[inline]
    pub fn max_steps(&self) -> u32 {
        (self.episode_s / self.control_dt()).round() as u32
    }

    /// Map a policy action to per-joint PD position targets:
    /// `q_target = default_pos + action_scale · action`.
    pub fn joint_targets(&self, action: &[f32; NUM_JOINTS]) -> [f32; NUM_JOINTS] {
        std::array::from_fn(|i| {
            let j = self.robot.joints[i];
            // Clamp the PD target to the joint's physical limit. Unbounded targets
            // let the policy command far past the stops (measured: hipx target
            // ~1.5 rad vs a ±0.35 rad limit), so the PD slams the joint into its
            // limit at near-saturated torque every step. That "limit-riding" pose
            // is a degenerate local optimum: it wastes torque (critically on the
            // fragile ankle) and gives the policy a flat, zero-gradient region to
            // get stuck in — every trained policy collapsed to it. Clamping keeps
            // the commanded pose physical and the PD error (hence torque) bounded,
            // while still allowing each joint its FULL range (the action scale,
            // not ±1, sets how far |action| must go to reach the limit).
            let (lo, hi) = j.pos_limit;
            (j.default_pos + j.action_scale * action[i]).clamp(lo, hi)
        })
    }

    /// Gravity direction in the base frame (`projected_gravity`). Upright ≈
    /// `(0, -1, 0)`; its horizontal components measure tilt.
    #[inline]
    pub fn projected_gravity(&self, base: &BaseState) -> [f32; 3] {
        let mut world_down = [0.0; 3];
        world_down[UP] = -1.0;
        quat_rotate_inv(base.orientation, world_down)
    }

    /// Base linear velocity in the body frame.
    #[inline]
    fn base_lin_vel_body(&self, base: &BaseState) -> [f32; 3] {
        quat_rotate_inv(base.orientation, base.lin_vel_world)
    }

    /// Base angular velocity in the body frame.
    #[inline]
    fn base_ang_vel_body(&self, base: &BaseState) -> [f32; 3] {
        quat_rotate_inv(base.orientation, base.ang_vel_world)
    }

    /// Cosine of the base tilt (body up-axis · world up). 1.0 = upright.
    #[inline]
    pub fn upright_cos(&self, base: &BaseState) -> f32 {
        let mut up = [0.0; 3];
        up[UP] = 1.0;
        quat_rotate(base.orientation, up)[UP]
    }

    /// Assemble the 43-dim policy observation into `obs`.
    ///
    /// Layout: `[last_action(12), command(4), joint_pos_rel(12), joint_vel(12),
    /// projected_gravity(3)]`. `joint_pos_rel = q − default_pos`.
    pub fn observe(&self, state: &RobotState, cmd: &VelocityCommand, obs: &mut [f32]) {
        debug_assert_eq!(obs.len(), OBS_DIM);
        let mut o = 0;
        let put = |obs: &mut [f32], o: &mut usize, v: f32| {
            obs[*o] = v;
            *o += 1;
        };
        for i in 0..NUM_JOINTS {
            put(obs, &mut o, state.last_action[i]);
        }
        for c in cmd.obs() {
            put(obs, &mut o, c);
        }
        for i in 0..NUM_JOINTS {
            put(
                obs,
                &mut o,
                state.joint_pos[i] - self.robot.joints[i].default_pos,
            );
        }
        for i in 0..NUM_JOINTS {
            put(obs, &mut o, state.joint_vel[i]);
        }
        for g in self.projected_gravity(&state.base) {
            put(obs, &mut o, g);
        }
        debug_assert_eq!(o, OBS_DIM);
    }

    /// Assemble the privileged critic observation: the policy obs followed by the
    /// (un-noised, normally unobservable) base linear & angular velocity.
    pub fn observe_critic(&self, state: &RobotState, cmd: &VelocityCommand, obs: &mut [f32]) {
        debug_assert_eq!(obs.len(), CRITIC_OBS_DIM);
        self.observe(state, cmd, &mut obs[..OBS_DIM]);
        let v = self.base_lin_vel_body(&state.base);
        let w = self.base_ang_vel_body(&state.base);
        obs[OBS_DIM..OBS_DIM + 3].copy_from_slice(&v);
        obs[OBS_DIM + 3..CRITIC_OBS_DIM].copy_from_slice(&w);
    }

    /// Compute the per-term reward for one control step.
    pub fn reward(&self, state: &RobotState, cmd: &VelocityCommand) -> RewardBreakdown {
        let dt = self.control_dt();
        let v = self.base_lin_vel_body(&state.base);
        let w = self.base_ang_vel_body(&state.base);
        let grav = self.projected_gravity(&state.base);

        // Tracking (exp kernels).
        let lin_err = (cmd.vx - v[FWD]).powi(2) + (cmd.vy - v[LAT]).powi(2);
        let track_lin_vel =
            self.weights.track_lin_vel * (-lin_err / self.stds.lin_vel.powi(2)).exp() * dt;

        let ang_err = (cmd.yaw_rate - w[UP]).powi(2);
        let track_ang_vel =
            self.weights.track_ang_vel * (-ang_err / self.stds.ang_vel.powi(2)).exp() * dt;

        // Upright: horizontal components of projected gravity → 0 when flat.
        let tilt_err = grav[FWD].powi(2) + grav[LAT].powi(2);
        let upright = self.weights.upright * (-tilt_err / self.stds.upright.powi(2)).exp() * dt;

        // Base height: stand tall (exp kernel around the target) so the policy
        // can't trivially crouch to avoid falling.
        let h_err = (state.base.height - self.weights.base_height_target).powi(2);
        let base_height =
            self.weights.base_height * (-h_err / self.stds.base_height.powi(2)).exp() * dt;

        // Hip yaw/roll deviation penalty (reuses the `pose` slot — the WBC port
        // left the full-posture reward at weight 0). The LATERAL hip DOFs (hipz
        // yaw, hipx roll) should stay near neutral whether standing OR walking
        // straight; without a penalty the policy braces by jamming them to their
        // ±20° limits (limit-riding — a degenerate, non-transferring stance).
        // L2 penalty (negative weight), ALWAYS-on so it also keeps the gait from
        // splaying. Targets ONLY hipz/hipx, leaving the sagittal walking DOFs
        // (hipy/knee/ankley) free. `weights.pose` is the (negative) penalty gain.
        let standing = cmd.speed() < 0.1;
        let mut hip_dev2 = 0.0;
        for &i in &self.hip_yawroll_idx {
            hip_dev2 += (state.joint_pos[i] - self.robot.joints[i].default_pos).powi(2);
        }
        let pose = self.weights.pose * hip_dev2 * dt;

        // Bilateral symmetry: sagittal joints (hipy/knee/ankley) mirror equal,
        // lateral joints (hipz/hipx/anklex) mirror opposite. Reward exp(-error).
        let sym_err = self.symmetry_error(&state.joint_pos);
        let bilateral_symmetry = self.weights.bilateral_symmetry * (-sym_err).exp() * dt;

        // Penalties (negative weights).
        let mut da2 = 0.0;
        for i in 0..NUM_JOINTS {
            da2 += (state.last_action[i] - state.prev_action[i]).powi(2);
        }
        let action_rate = self.weights.action_rate * da2 * dt;

        let mut da2_hip = 0.0;
        for &i in &self.hip_yawroll_idx {
            da2_hip += (state.last_action[i] - state.prev_action[i]).powi(2);
        }
        let action_rate_hipz_hipx = self.weights.action_rate_hipz_hipx * da2_hip * dt;

        let body_ang_vel = self.weights.body_ang_vel * (w[FWD].powi(2) + w[LAT].powi(2)) * dt;
        let lin_vel_z = self.weights.lin_vel_z * v[UP].powi(2) * dt;

        // Soft joint-position-limit penalty (soft band at 90% of the hard limit).
        let mut lim_pen = 0.0;
        for i in 0..NUM_JOINTS {
            let (lo, hi) = self.robot.joints[i].pos_limit;
            let (lo, hi) = (lo * 0.9, hi * 0.9);
            let q = state.joint_pos[i];
            lim_pen += (q - hi).max(0.0) + (lo - q).max(0.0);
        }
        let dof_pos_limits = self.weights.dof_pos_limits * lim_pen * dt;

        let mut jv2 = 0.0;
        for i in 0..NUM_JOINTS {
            jv2 += state.joint_vel[i].powi(2);
        }
        let dof_vel = self.weights.dof_vel * jv2 * dt;

        // --- foot-contact shaped terms ---
        let moving = !standing;
        // Forward-progress gate for the stepping BONUSES (air_time, single-support
        // bonus). Without it the policy farms those bonuses by stepping IN PLACE
        // (v≈0) — it overcooked into marching, abandoning forward tracking.
        // progress = clamp((v·cmd)/|cmd|², 0, 1): 1 when the base moves at the
        // commanded velocity, 0 when stationary/backward. So a stepping bonus is
        // only paid when the steps actually carry the robot toward the command.
        let cmd_sp2 = cmd.vx * cmd.vx + cmd.vy * cmd.vy;
        let progress = if cmd_sp2 > 1e-6 {
            ((v[FWD] * cmd.vx + v[LAT] * cmd.vy) / cmd_sp2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // Air time: reward the completed swing duration at touchdown (capped), so
        // any reasonable step is encouraged — only when commanded to move. (The old
        // `air_time − 0.5` form was negative for sub-0.5 s steps, i.e. it *punished*
        // this small robot's normal-cadence stepping.)
        let mut air = 0.0;
        for f in &state.feet {
            if f.first_contact {
                air += f.air_time.min(0.4);
            }
        }
        let air_time = if moving {
            self.weights.air_time * air * dt * progress
        } else {
            0.0
        };

        // Flight penalty: both feet off the ground = hopping/bounding, not walking.
        let flight = if state.feet.iter().all(|f| !f.contact) {
            self.weights.flight * dt
        } else {
            0.0
        };

        // Single-support shaping (while moving): exactly one foot down is the
        // walking phase. A BONUS for single-support alone wasn't enough — the
        // policy just forgoes it and waddles in permanent double-support (both
        // feet planted, shuffling). So while moving we now also PENALIZE
        // double-support by the same magnitude: staying on both feet is actively
        // costly, which forces the policy to pick a foot up and step. Flight
        // (zero contacts) is left to the `flight` term.
        let contacts = state.feet.iter().filter(|f| f.contact).count();
        let single_support = if moving {
            match contacts {
                // Reward single-support ONLY when making forward progress (gated)
                // so it can't be farmed by stepping in place; double-support is
                // penalized regardless (always costly while moving).
                1 => self.weights.single_support * dt * progress,
                2 => -self.weights.single_support * dt,
                _ => 0.0, // flight: see `flight`
            }
        } else {
            0.0
        };

        // Slip: penalize horizontal foot speed while the foot is in contact.
        let mut slip = 0.0;
        for f in &state.feet {
            if f.contact {
                slip += f.planar_speed.powi(2);
            }
        }
        let foot_slip = self.weights.foot_slip * slip * dt;

        // Clearance: penalize swing-foot height deviation from the target, scaled by
        // foot speed (encourages lifting the foot while swinging it forward).
        let mut clr = 0.0;
        for f in &state.feet {
            if !f.contact {
                clr += (f.height - self.weights.foot_clearance_target).powi(2) * f.planar_speed;
            }
        }
        let foot_clearance = self.weights.foot_clearance * clr * dt;

        // Flat foot: penalize the squared sole tilt of any foot in contact, so the
        // robot plants its whole sole rather than balancing on a toe/heel/edge.
        let mut tilt_sq = 0.0;
        for f in &state.feet {
            if f.contact {
                tilt_sq += f.tilt.powi(2);
            }
        }
        let foot_orientation = self.weights.foot_orientation * tilt_sq * dt;

        // WBC's `feet_yaw_mean_vs_base`: sum of squared yaw (in base frame) over
        // both feet — drives each foot to point in the base's forward direction.
        let mut yaw_sq = 0.0;
        for f in &state.feet {
            yaw_sq += f.yaw_rel_base.powi(2);
        }
        let feet_yaw_mean = self.weights.feet_yaw_mean * yaw_sq * dt;

        // WBC's `feet_distance_from_ref` (lateral mode): penalise the absolute
        // deviation of the lateral stance width from `feet_distance_ref`. We
        // transform the foot-to-foot world XY difference into the base frame
        // using only its yaw component (base assumed near-upright).
        let feet_distance = if NUM_FEET == 2 {
            let dx = state.feet[0].pos_xy[0] - state.feet[1].pos_xy[0];
            let dy = state.feet[0].pos_xy[1] - state.feet[1].pos_xy[1];
            // Project (dx, dy) world into base frame using base yaw.
            let q = &state.base.orientation; // (x, y, z, w)
            let base_yaw =
                (2.0 * (q[3] * q[2] + q[0] * q[1])).atan2(1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
            let cy = base_yaw.cos();
            let sy = base_yaw.sin();
            // Inverse-yaw rotation: world → base.
            let lateral = -sy * dx + cy * dy;
            let err = lateral.abs() - self.weights.feet_distance_ref;
            // L1 error like WBC's default `norm="l1"`.
            self.weights.feet_distance * err.abs() * dt
        } else {
            0.0
        };

        RewardBreakdown {
            track_lin_vel,
            track_ang_vel,
            upright,
            base_height,
            pose,
            bilateral_symmetry,
            action_rate,
            action_rate_hipz_hipx,
            body_ang_vel,
            lin_vel_z,
            dof_pos_limits,
            dof_vel,
            air_time,
            flight,
            single_support,
            foot_slip,
            foot_clearance,
            foot_orientation,
            feet_yaw_mean,
            feet_distance,
        }
    }

    /// Mirror error: pairs same-family left/right joints and accumulates the
    /// squared difference under the family's mirror sign.
    fn symmetry_error(&self, q: &[f32; NUM_JOINTS]) -> f32 {
        let mut err = 0.0;
        for i in 0..NUM_JOINTS {
            let name = self.robot.joints[i].name;
            let Some(stem) = name.strip_suffix("_left") else {
                continue;
            };
            // Find the right counterpart.
            let right = format!("{stem}_right");
            let Some(jr) = self.robot.joints.iter().position(|j| j.name == right) else {
                continue;
            };
            // Sagittal joints mirror equal; lateral joints mirror opposite.
            let sign = if stem == "hipy" || stem == "knee" || stem == "ankley" {
                1.0
            } else {
                -1.0
            };
            err += (q[i] - sign * q[jr]).powi(2);
        }
        err
    }

    /// Whether the episode should terminate from a fall: excessive tilt, sunk too
    /// low (the anti-reward-hack floor), or a non-finite base. The env adds the
    /// separate time-out termination at [`Self::max_steps`].
    pub fn fell_over(&self, base: &BaseState) -> bool {
        !base.height.is_finite()
            || base.height < self.min_base_height
            || self.upright_cos(base) < self.tilt_limit.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upright_state() -> RobotState {
        RobotState::default()
    }

    #[test]
    fn obs_dim_consistent() {
        assert_eq!(OBS_DIM, 43);
        assert_eq!(ACTION_DIM, 12);
        let task = VelocityFlatTask::new();
        let mut obs = vec![0.0; OBS_DIM];
        task.observe(&upright_state(), &VelocityCommand::default(), &mut obs);
        // Upright, neutral pose, zero command → projected gravity ≈ (0,0,-1)
        // (Z-up), everything else zero.
        assert!(obs.iter().take(OBS_DIM - 3).all(|&x| x == 0.0));
        assert!(
            (obs[OBS_DIM - 1] - (-1.0)).abs() < 1e-6,
            "up component of gravity"
        );
    }

    #[test]
    fn control_timing() {
        let task = VelocityFlatTask::new();
        assert!((task.control_dt() - 0.02).abs() < 1e-9);
        assert_eq!(task.max_steps(), 1000); // 20 s / 0.02 s
    }

    #[test]
    fn joint_targets_offset_by_scale() {
        let task = VelocityFlatTask::new();
        let mut a = [0.0; NUM_JOINTS];
        a[0] = 1.0; // anklex_left, scale 0.55
        let t = task.joint_targets(&a);
        assert!((t[0] - 0.55).abs() < 1e-6);
        assert_eq!(t[1], 0.0);
    }

    #[test]
    fn perfect_tracking_gives_full_reward() {
        let task = VelocityFlatTask::new();
        // Command zero velocity; robot at rest, upright, neutral pose. Standing is
        // commanded, so the (gated) pose term is active and tracking kernels are max.
        let r = task.reward(&upright_state(), &VelocityCommand::default());
        let dt = task.control_dt();
        let w = RewardWeights::default();
        assert!((r.track_lin_vel - w.track_lin_vel * dt).abs() < 1e-6);
        assert!((r.track_ang_vel - w.track_ang_vel * dt).abs() < 1e-6);
        assert!((r.upright - w.upright * dt).abs() < 1e-6);
        assert!((r.bilateral_symmetry).abs() < 1e-6); // symmetry term disabled
        // pose weight is 0 in the WBC port (they have no `pose` term).
        assert!((r.pose - w.pose * dt).abs() < 1e-6);
        // No motion/action → penalties zero.
        assert_eq!(r.action_rate, 0.0);
        assert_eq!(r.dof_vel, 0.0);
        assert!(r.total() > 0.0);
    }

    #[test]
    fn foot_rewards_behave() {
        let task = VelocityFlatTask::new();
        let cmd = VelocityCommand {
            vx: 0.5,
            vy: 0.0,
            yaw_rate: 0.0,
        };
        // The WBC port disables air_time (weight=0), so a touchdown produces zero
        // air-time reward regardless of swing duration.
        let mut s = RobotState::default();
        s.feet[0] = FootObs {
            contact: true,
            first_contact: true,
            air_time: 0.6,
            height: 0.0,
            planar_speed: 0.0,
            tilt: 0.0,
            yaw_rel_base: 0.0,
            pos_xy: [0.0, 0.0],
        };
        assert_eq!(task.reward(&s, &cmd).air_time, 0.0);
        // A foot sliding while in contact → negative slip penalty.
        let mut s2 = RobotState::default();
        s2.feet[0].planar_speed = 1.0;
        assert!(task.reward(&s2, &cmd).foot_slip < 0.0);
        // A foot tilted onto its edge while in contact → flat-foot penalty.
        let mut s3 = RobotState::default();
        s3.feet[0].tilt = 1.0; // ~57° off flat
        assert!(task.reward(&s3, &cmd).foot_orientation < 0.0);
        // The same tilt while airborne → no penalty (only stance feet count).
        let mut s4 = RobotState::default();
        s4.feet[0].contact = false;
        s4.feet[0].tilt = 1.0;
        s4.feet[1].contact = false;
        assert_eq!(task.reward(&s4, &cmd).foot_orientation, 0.0);
        // A foot yawed relative to the base → feet_yaw_mean penalty.
        let mut s5 = RobotState::default();
        s5.feet[0].yaw_rel_base = 0.5;
        assert!(task.reward(&s5, &cmd).feet_yaw_mean < 0.0);
    }

    #[test]
    fn velocity_error_reduces_tracking_reward() {
        let task = VelocityFlatTask::new();
        let mut s = upright_state();
        s.base.lin_vel_world = [0.5, 0.0, 0.0]; // moving forward
        // Command standing → big tracking error → reward below the max.
        let r = task.reward(&s, &VelocityCommand::default());
        let w = RewardWeights::default();
        assert!(r.track_lin_vel < 0.5 * w.track_lin_vel * task.control_dt());
    }

    #[test]
    fn fell_over_detects_tilt() {
        let task = VelocityFlatTask::new();
        let mut base = BaseState::default();
        assert!(!task.fell_over(&base));
        // Tip 80° about X (> 70° limit): (x,y,z,w)=(sin40,0,0,cos40).
        let a = 40.0_f32.to_radians();
        base.orientation = [a.sin(), 0.0, 0.0, a.cos()];
        assert!(task.fell_over(&base));
    }

    #[test]
    fn command_sampler_respects_ranges_and_standing() {
        let s = CommandSampler::default();
        let mut rng = Lcg::new(1);
        let mut stands = 0;
        for _ in 0..2000 {
            let c = s.sample(&mut rng);
            assert!(c.vx.abs() <= 0.8 + 1e-6);
            assert!(c.vy.abs() <= 0.4 + 1e-6);
            assert!(c.yaw_rate.abs() <= 0.2 + 1e-6);
            if c == VelocityCommand::default() {
                stands += 1;
            }
        }
        // ~10% standing; allow a wide band.
        assert!(
            (50..400).contains(&stands),
            "standing fraction off: {stands}/2000"
        );
    }

    #[test]
    fn symmetry_error_zero_for_mirrored_pose() {
        let task = VelocityFlatTask::new();
        // Neutral pose is trivially symmetric.
        assert_eq!(task.symmetry_error(&[0.0; NUM_JOINTS]), 0.0);
    }
}
