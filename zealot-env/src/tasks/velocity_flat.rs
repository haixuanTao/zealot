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
/// joint_pos_rel(12) + joint_vel(12) + projected_gravity(3) + gait_phase(2)`.
/// The trailing 2 are (sin 2πφ, cos 2πφ) of the gait clock so the policy can
/// time its steps to the periodic gait reward.
pub const OBS_DIM: usize = NUM_JOINTS + 4 + NUM_JOINTS + NUM_JOINTS + 3 + 2;
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
    /// World horizontal position of the base (torso), m. Used by the CoM-centering
    /// reward to keep the center of mass over the support foot — balancing on one
    /// foot with the CoM centered needs ~0 ankle torque, so this lets a fragile
    /// (15 N·m) ankle sustain single-support instead of saturating fighting an
    /// off-center CoM.
    pub pos_xy: [f32; 2],
}

impl Default for BaseState {
    fn default() -> Self {
        Self {
            orientation: [0.0, 0.0, 0.0, 1.0],
            lin_vel_world: [0.0; 3],
            ang_vel_world: [0.0; 3],
            height: 0.5,
            pos_xy: [0.0, 0.0],
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
    /// Touchdown this step that ALTERNATED feet — `first_contact` AND the *other*
    /// foot was the most recent to touch down (set by the env, which tracks the
    /// last-touchdown foot per env). The swing/air-time reward keys off this so a
    /// step only pays when feet alternate (L→R→L→R): a foot held permanently in
    /// the air never touches down (no reward), and double-tapping the same foot
    /// (hopping) earns nothing on the repeat. Forces a real alternating gait.
    pub alt_step: bool,
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
            alt_step: false,
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
    /// Gait-clock phase ∈ [0,1), advanced by the env each control step. Drives
    /// the periodic gait reward (foot 0 should swing near phase 0, foot 1 near
    /// phase 0.5) and is fed to the policy as (sin 2πφ, cos 2πφ) so it can lock
    /// its leg motion to the clock. The phase-clock reward provides a DENSE
    /// per-step gradient toward an alternating swing/stance pattern — which the
    /// sparse touchdown bonus (air_time) could not, since a step's payoff never
    /// beat the fall risk (the shuffle stayed a stable local optimum).
    pub phase: f32,
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
            phase: 0.0,
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
            // Fraction of command resamples that are a pure STAND (zero). Raising
            // it (BIPED_STAND_PROB) makes the robot stop more often → trains
            // explicit walk→stand→walk (go-stop-go) transitions and gives frequent
            // quasi-static "stabilize" checkpoints (helps the deliberate gait +
            // transfer). Resample interval (BIPED_RESAMPLE_S, "lo,hi" seconds);
            // shorter = more frequent transitions.
            standing_prob: std::env::var("BIPED_STAND_PROB")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.1),
            resample_s: std::env::var("BIPED_RESAMPLE_S")
                .ok()
                .and_then(|s| {
                    let p: Vec<f32> = s.split(',').filter_map(|x| x.parse().ok()).collect();
                    if p.len() == 2 {
                        Some((p[0], p[1]))
                    } else {
                        None
                    }
                })
                .unwrap_or((3.0, 8.0)),
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
    /// Linear forward-progress reward: `w · clamp(v·ĉmd, 0, |cmd|)`. A non-saturating
    /// gradient toward the commanded direction (breaks the march-in-place dead zone
    /// where the exp tracking kernel is flat). Folded into the `track_lin_vel` term.
    pub forward_progress: f32,
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
    /// Periodic gait-clock reward (dense). Each step, each foot earns up to this
    /// weight for matching its prescribed phase: airborne during its swing window,
    /// in contact during its stance window. Provides the dense gradient toward an
    /// alternating gait that the sparse touchdown bonus could not.
    pub gait_clock: f32,
    /// Fraction of each foot's gait cycle spent in swing (rest is stance). With
    /// the feet offset by half a cycle, `1 - 2·swing_ratio` of the cycle is
    /// double-support — the built-in "both feet down in the middle".
    pub gait_swing_ratio: f32,
    /// CoM-centering reward: keeps the base (CoM proxy) horizontally over the
    /// support point (centroid of contacting feet). With the CoM over the stance
    /// foot, the gravitational moment about the ankle ≈ 0, so single-support needs
    /// almost no ankle torque — letting the fragile 15 N·m ankle hold one-foot
    /// balance instead of saturating. This is what lets a real step survive.
    pub com_centering: f32,
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
            // TRACKING is now the DOMINANT objective (the task). Raised + the std
            // is tightened (below) so the reward is SHARP: large for tracking the
            // commanded velocity, ~0 for not tracking — i.e. not-following-the-
            // command is heavily penalized in effect (big opportunity cost). This
            // replaces the brittle "force stepping" gait machinery: walking emerges
            // because the robot MUST track velocity, and it may settle on two feet
            // between steps (double-support no longer penalized).
            track_lin_vel: 10.0,   // was 5.0 — make following velocity the point
            forward_progress: 8.0, // linear forward-velocity gradient (breaks march-in-place)
            track_ang_vel: 8.0,    // was 5.0
            // Stay-up lowered so it can't out-earn tracking (it used to: upright+
            // height ≈0.13/step > tracking ≈0.07 → the policy preferred to STAND).
            upright: 3.0,     // was 5.0
            base_height: 2.0, // restored to WBC value (was 1.0): at 1.0 the
            // pure-stand phase let the torso slowly crouch
            // 0.72→0.64 (lower CoM = more stable static stance).
            // 2.0 keeps it tall in BOTH stand and walk.
            base_height_target: 0.72, // WBC DEFAULT_TRUNK_HEIGHT (was 0.62 — crouch bug)
            pose: -8.0,               // hip yaw/roll deviation penalty (anti-limit-ride)
            bilateral_symmetry: 2.0,  // reward L/R-mirrored gait (natural, fixes lopsidedness)
            action_rate: -0.1,        // WBC -0.1 (was -0.25)
            action_rate_hipz_hipx: 0.0,
            body_ang_vel: -0.05,  // WBC ang_vel_xy -0.05 (was -0.25)
            lin_vel_z: -0.05,     // WBC -0.05 (was -0.25)
            dof_pos_limits: -0.5, // WBC -0.1; strengthened to discourage limit-bracing
            dof_vel: -2e-4,       // WBC -2e-4 (was -1e-4)
            termination: -2.0,    // WBC is_terminated -100 ×dt(0.02) (was -25 one-shot)
            // Gait shaping — turned ON to drive a clean, TRANSFERABLE alternating
            // stride. The un-shaped reward let the policy track velocity with a
            // nexus-specific foot-shuffle (ankle-slam propulsion) that didn't
            // survive MuJoCo (sim2sim ratio 0.19 walking vs 1.00 standing). These
            // push toward real stepping: swing duration (air_time), exactly one
            // foot planted (single_support), no double-flight (flight), foot lifted
            // to a target clearance while swinging (foot_clearance), and — the key
            // anti-shuffle / anti-exploit term — a strong penalty on a planted foot
            // sliding (foot_slip, 50× the old WBC value).
            // Forced-stepping terms OFF — they were band-aids for the old stand-bias
            // and made the gait gameable (slide/hop/march). With tracking now
            // dominant, stepping emerges from NEEDING to track velocity while
            // sliding is blocked (foot_slip) and settling on two feet is allowed
            // (double-support unpenalized). Keep only: no-hop (flight), no-slide
            // (foot_slip), lift the swing foot cleanly (foot_clearance).
            air_time: 1.0, // RE-ENABLED (was 0): pure emergence + a static
            // foot-lift reward got HACKED into a one-foot statue
            // (one foot held up 100% → farms clearance, never
            // steps; 0 transfer, MuJoCo fell in 0.66s). air_time
            // pays the completed-swing duration ONLY at touchdown,
            // so a permanently-raised foot earns nothing → forces
            // real alternating step cycles. Progress+command gated.
            flight: -1.0,        // keep: no hopping (both feet airborne)
            single_support: 0.5, // REPURPOSED → double-support SETTLE bonus (both feet
            // planted while moving). Modest, so it shapes a
            // "swing → settle → swing" cycle without farmable waddle.
            foot_slip: -1.0, // dialed back from -3.0: -3.0 suppressed motion
            // (slip penalty satisfied by NOT moving → backward
            // drift) rather than inducing lift. The positive
            // foot_clearance reward below now supplies the
            // "pick your feet up" incentive directly.
            foot_clearance: 0.0, // DROPPED. A static foot-height reward is farmable —
            // it got hacked into a one-foot statue (foot held up
            // 100% to farm clearance; 0 transfer, MuJoCo fell in
            // 0.66s). Step height now comes for free once steps are
            // real (alternation-gated air_time below).
            foot_clearance_target: 0.03, // (unused at weight 0; kept for the gated compute)
            foot_orientation: -0.01,     // WBC feet_roll_l2 -0.01 (was -0.5)
            feet_yaw_mean: -0.4,         // WBC feet_yaw_mean_vs_base -0.4 (was -2.0)
            feet_distance: -0.02,        // WBC feet_distance_from_ref -0.02 (was -0.1)
            feet_distance_ref: 0.2,
            gait_clock: 3.0, // dense periodic gait reward (the load-bearing
            // stepping signal). Symmetric ±: standing during a
            // swing window is penalized, so lifting on schedule
            // is clearly worth more than staying planted.
            gait_swing_ratio: 0.4, // 40% swing per foot → 20% double-support overlap
            com_centering: 2.0,    // keep CoM over the support foot → ~0 ankle torque
                                   // in single-support (fragile 15 N·m ankle can hold it)
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
            lin_vel: 0.15, // tightened (was 0.3): sharp tracking → standing-when-
            ang_vel: 0.1,  // commanded scores ~0, i.e. NOT tracking is penalized

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
    /// Periodic gait-clock contribution (dense reward for matching each foot's
    /// swing/stance to the gait phase).
    pub gait_clock: f32,
    /// CoM-centering contribution (CoM over the support point → low ankle torque).
    pub com_centering: f32,
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
            + self.gait_clock
            + self.com_centering
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
        // Gait clock as (sin, cos) so it's continuous across the 1→0 wrap.
        let ph = state.phase * std::f32::consts::TAU;
        put(obs, &mut o, ph.sin());
        put(obs, &mut o, ph.cos());
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

        // Tracking (exp kernels) + a LINEAR forward-progress term. The exp kernel
        // saturates to ~0 (and flat) when the robot is far below the commanded
        // speed, so a march-in-place policy sits in a dead zone with no gradient
        // pulling it forward (measured: track_lin_vel stuck ~0.014 for thousands
        // of iters while vx≈0.03). The linear term `w · clamp(v·ĉmd, 0, |cmd|)`
        // rewards forward velocity PROPORTIONALLY — a gradient at any speed, zero
        // for standing/backward — so any forward motion is rewarded and it can
        // climb out of the in-place optimum.
        let lin_err = (cmd.vx - v[FWD]).powi(2) + (cmd.vy - v[LAT]).powi(2);
        let cmd_speed = (cmd.vx * cmd.vx + cmd.vy * cmd.vy).sqrt();
        let v_along = if cmd_speed > 1e-6 {
            (v[FWD] * cmd.vx + v[LAT] * cmd.vy) / cmd_speed
        } else {
            0.0
        };
        let track_lin_vel =
            self.weights.track_lin_vel * (-lin_err / self.stds.lin_vel.powi(2)).exp() * dt
                + self.weights.forward_progress * v_along.clamp(0.0, cmd_speed) * dt;

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
        // Pay the completed swing duration ONLY at an ALTERNATING touchdown
        // (f.alt_step). This is the unfarmable core of the gait: a foot held up
        // forever never lands (no reward); hopping on one foot earns nothing on
        // the repeat; only L→R→L→R stepping pays. Capped at 0.4 s/step.
        let mut air = 0.0;
        for f in &state.feet {
            if f.alt_step {
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

        // Gait-phase shaping (while moving). A *bonus* for stepping (air_time) gives
        // no exploration pressure — you only collect it AFTER you already step — so
        // from a shuffle the policy never discovers the first step. And a settle
        // BONUS for double-support actively rewarded the shuffle (measured: v40 sat
        // in permanent double-support, air_time stuck at 0). The pressure must come
        // from making NOT-stepping costly. So this term PENALIZES both degenerate
        // modes and leaves only alternating stepping unpunished:
        //   contacts==2 (permanent double-support = slide-shuffle) → penalty,
        //   contacts==1 but the airborne foot is HELD (air_time > MAX_SWING, i.e. a
        //     one-foot statue — the other hack) → penalty,
        //   contacts==1 with an ACTIVE swing (air_time ≤ MAX_SWING) → 0 (allowed),
        //   contacts==0 (flight) → 0, handled by `flight`.
        // A brief double-support SETTLE between steps costs little (a few steps of
        // small penalty); only PERMANENT double- or single-support is expensive.
        // The reward for doing it right is the alternation-gated air_time above.
        let contacts = state.feet.iter().filter(|f| f.contact).count();
        let single_support = if moving {
            match contacts {
                2 => -self.weights.single_support * dt, // permanent double-support = shuffle
                1 => {
                    // One foot up: fine if it's an active swing, penalized if it's a
                    // held statue (foot airborne longer than a normal swing).
                    let held = state
                        .feet
                        .iter()
                        .any(|f| !f.contact && f.air_time > MAX_SWING_S);
                    if held {
                        -self.weights.single_support * dt
                    } else {
                        0.0
                    }
                }
                _ => 0.0, // flight: see `flight`
            }
        } else {
            // STANDING (command ~0): the inverse — reward BOTH feet planted,
            // penalize stepping. Without this the walking policy keeps lifting
            // feet / stamping in place at zero command (it drifts + fidgets,
            // since nothing rewarded standing still). Now "step when told to move,
            // plant when told to stand" is symmetric.
            match contacts {
                2 => self.weights.single_support * dt, // both feet planted = good
                1 => -self.weights.single_support * dt, // stepping while told to stand = bad
                _ => 0.0,
            }
        };

        // Slip: penalize horizontal foot speed while the foot is in contact.
        let mut slip = 0.0;
        for f in &state.feet {
            if f.contact {
                slip += f.planar_speed.powi(2);
            }
        }
        let foot_slip = self.weights.foot_slip * slip * dt;

        // Clearance: POSITIVE, capped reward for lifting an ACTIVE SWING foot above
        // its resting height, saturating at foot_clearance_target. Gated three ways
        // so it can't be farmed by holding one foot in the air (which is exactly how
        // the ungated version got hacked into a one-foot statue — 0 transfer):
        //   (1) f.contact == false   — only a lifted foot,
        //   (2) f.air_time < 0.45 s   — only an ACTIVE swing, not a held statue: a
        //       foot raised longer than a normal swing stops earning, so to keep
        //       collecting it must touch down (resetting air_time) and re-swing,
        //   (3) moving                — never at zero command (no stamping in place).
        const FOOT_REST_H: f32 = 0.035;
        const MAX_SWING_S: f32 = 0.45;
        let mut foot_h = 0.0;
        for f in &state.feet {
            if !f.contact && f.air_time < MAX_SWING_S {
                let lift = (f.height - FOOT_REST_H).max(0.0) / self.weights.foot_clearance_target;
                foot_h += lift.min(1.0);
            }
        }
        let foot_clearance = if moving {
            self.weights.foot_clearance * foot_h * dt
        } else {
            0.0
        };

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

        // Periodic gait clock (DENSE). Foot 0's cycle starts at `phase`, foot 1 is
        // offset half a cycle, so they alternate. Within a foot's cycle, the first
        // `gait_swing_ratio` is the SWING window (the foot should be airborne), the
        // rest is STANCE (it should be in contact). Each step, each foot earns the
        // weight if its actual contact matches its prescribed phase. This pays
        // every step (not just at touchdown), so the gradient pulls the foot up at
        // the right time even before a full step succeeds — the dense signal the
        // sparse air_time bonus lacked. Only while moving (no forced gait at stand).
        // Siekmann-style: each foot scores +1 when its contact MATCHES its phase
        // (airborne in swing / grounded in stance) and −1 when it MISMATCHES. The
        // −1 for contact-during-swing is the crucial part (my first version gave 0
        // there): it makes keeping a foot down during its swing window actively
        // COSTLY, so standing no longer farms the stance windows for free — the
        // only way to stop bleeding reward is to actually lift on schedule. The
        // −1 for airborne-during-stance also penalizes a held-up statue foot.
        let gait_clock = if moving {
            let sr = self.weights.gait_swing_ratio;
            let mut gc = 0.0;
            for (k, f) in state.feet.iter().enumerate() {
                let ph = (state.phase + 0.5 * k as f32).fract();
                let want_swing = ph < sr;
                let matched = if want_swing { !f.contact } else { f.contact };
                gc += if matched { 1.0 } else { -1.0 };
            }
            // PROGRESS-GATE the gait reward: on-schedule stepping only pays when the
            // steps actually carry the body toward the command. Without this the
            // policy farms gait_clock by marching IN PLACE (measured: v47 stepped
            // cleanly — 5 cm lifts, 8–9 touchdowns — but vx≈0.03 m/s). The gate
            // makes forward steps the only way to earn it (progress = (v·cmd)/|cmd|²).
            // The double-support penalty (single_support, ungated) still backstops
            // against simply standing, so it must step — now forward.
            self.weights.gait_clock * gc * dt * progress
        } else {
            0.0
        };

        // CoM centering: keep the base (CoM proxy) over the support point — the
        // centroid of whatever feet are in contact. Offset → 0 means the CoM is
        // over the base of support, so the ankle needs ~no torque to hold balance
        // (crucial in single-support, where an off-center CoM saturated the 15 N·m
        // ankle). exp kernel, active whenever at least one foot is down.
        const COM_STD: f32 = 0.12;
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut nc = 0u32;
        for f in &state.feet {
            if f.contact {
                sx += f.pos_xy[0];
                sy += f.pos_xy[1];
                nc += 1;
            }
        }
        let com_centering = if nc > 0 {
            let dx = state.base.pos_xy[0] - sx / nc as f32;
            let dy = state.base.pos_xy[1] - sy / nc as f32;
            let d2 = dx * dx + dy * dy;
            self.weights.com_centering * (-d2 / (COM_STD * COM_STD)).exp() * dt
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
            gait_clock,
            com_centering,
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
        assert_eq!(OBS_DIM, 45);
        assert_eq!(ACTION_DIM, 12);
        let task = VelocityFlatTask::new();
        let mut obs = vec![0.0; OBS_DIM];
        task.observe(&upright_state(), &VelocityCommand::default(), &mut obs);
        // Layout: last_action[0..12], command[12..16], joint_pos_rel[16..28],
        // joint_vel[28..40], projected_gravity[40..43], gait_phase(sin,cos)[43..45].
        // Upright, neutral pose, zero command, phase 0 → everything zero except
        // gravity up = -1 and cos(0) = 1.
        assert!(obs.iter().take(40).all(|&x| x == 0.0));
        assert!((obs[42] - (-1.0)).abs() < 1e-6, "up component of gravity");
        assert!(obs[43].abs() < 1e-6, "sin(phase 0) = 0");
        assert!((obs[44] - 1.0).abs() < 1e-6, "cos(phase 0) = 1");
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
        a[0] = 1.0; // anklex_left, scale 0.55, pos_limit ±0.175
        let t = task.joint_targets(&a);
        // 0 + 0.55·1 = 0.55, CLAMPED to the joint limit 0.175 (joint_targets caps
        // PD targets at pos_limit to stop limit-riding — see joint_targets()).
        assert!((t[0] - 0.175).abs() < 1e-6);
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
        // Neutral pose is L/R-mirrored, so sym_err=0 → full symmetry reward.
        assert!((r.bilateral_symmetry - w.bilateral_symmetry * dt).abs() < 1e-6);
        // `pose` is now the hip yaw/roll DEVIATION penalty: 0 at the neutral pose
        // (hipx/hipz = default), regardless of its (negative) weight.
        assert!(r.pose.abs() < 1e-6);
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
        // air_time pays ONLY on an alternating touchdown (alt_step). A plain
        // first_contact that did NOT alternate feet earns zero — this is what
        // blocks the one-foot-statue / same-foot-hop hacks.
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
            alt_step: false,
        };
        assert_eq!(task.reward(&s, &cmd).air_time, 0.0);
        // Same swing, but now it ALTERNATED while the base actually tracks the
        // forward command (progress > 0) → positive air-time reward.
        let mut s_alt = s;
        s_alt.feet[0].alt_step = true;
        s_alt.base.lin_vel_world = [0.5, 0.0, 0.0];
        assert!(task.reward(&s_alt, &cmd).air_time > 0.0);
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
