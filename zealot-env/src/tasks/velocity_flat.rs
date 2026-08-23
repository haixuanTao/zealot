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
use crate::robots::{RobotSpec, NUM_JOINTS};

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
// `+ 3` at the end is the base ANGULAR VELOCITY (gyro). Added after v21: the
// actor had no rotation sense at all — angular velocity was critic-only
// privileged information — so yaw control was open-loop and the policy could
// not cancel even its own heading drift (measured: yaw output scattered
// positive regardless of commanded sign). Every deployed locomotion stack
// feeds the gyro to the policy; it is free on hardware (IMU).
// `+ 5` at the very end is the STEP CUE (slots 48-52): distance to the edge,
// its signed height, the edge ORIENTATION as (sin, cos) in the body frame, and
// a validity flag. Supplied on hardware by a head RealSense: a plane fit plus
// edge extraction yields all three quantities directly, and unlike a foot
// probe it gives the edge DIRECTION -- which is what lets the robot meet a
// step at an angle rather than only head-on. Trained against a terrain oracle
// with sensor-shaped noise (see `StepCue`).
// `+ 2*NUM_HELD_OBS` at the very end (slots 53..79): the UPPER-BODY block —
// the 13 held-joint PD TARGETS and their finite-diff velocities. Until now the
// policy was blind to the arms: the AMASS playback moved 13 joints of mass and
// the legs only found out via the gyro, one control step after the torso
// already started tipping. Targets, not measured angles, on purpose: the
// controller COMMANDS the arms, so targets are exactly known on hardware too,
// and they LEAD the physical motion — feedforward, not just faster feedback.
// Zero-padded (targets = home, vel = 0) when the robot has fewer held joints
// or playback is off.
pub const NUM_HELD_OBS: usize = 13;
pub const OBS_DIM: usize =
    NUM_JOINTS + 4 + NUM_JOINTS + NUM_JOINTS + 3 + 2 + 3 + 5 + 2 * NUM_HELD_OBS;
/// Action vector length: one position target per leg DOF.
pub const ACTION_DIM: usize = NUM_JOINTS;
/// Privileged (critic) observation length: policy obs plus base linear & angular
/// velocity in the body frame. Foot/contact privileged terms are deferred.
// `+ 5` is the CLEAN step cue: the critic always sees the true edge state,
// un-noised and never dropped, even while the ACTOR's copy (slots 48..53 of
// the policy obs) is heavily dropout-masked early in training. Asymmetric
// actor-critic: the value function credits crossing correctly on steps the
// actor could not see, so the gradient learns "crossing pays" from blind
// practice -- the fix for the observed cue->caution trap, where a policy
// that has not yet learned to climb treats the cue purely as a fall
// predictor and stops.
pub const CRITIC_OBS_DIM: usize = OBS_DIM + 3 + 3 + 5;

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
    /// Foot vertical velocity, m/s (world +Z; negative = descending). Env
    /// finite-diff, 0 on the first step after reset. Drives the
    /// `touchdown_vz` slam penalty — the KINEMATIC impact term (a slam is a
    /// foot still descending fast near the ground; kinematics transfer
    /// across engines far better than contact-sensor forces do).
    pub vz: f32,
    /// One-step change in this foot's sensed normal contact force, in BODY
    /// WEIGHTS per control step (|F_t − F_{t−1}| / (m·g), ≥ 0). Filled by the
    /// env from the contact-force sensor (BIPED_CONTACT_SENSE); 0 when force
    /// sensing is off or on the first step after a reset. Drives the
    /// `force_rate` ground-reaction-smoothness penalty: a hard touchdown is a
    /// 1–2 BW jump in one step, and the standing tremor is a sustained
    /// left/right load dither — both are pure force-rate, while calm standing
    /// and normal gait weight transfer (~0.06–0.12 BW/step) are ~free.
    pub force_rate: f32,
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
            vz: 0.0,
            force_rate: 0.0,
        }
    }
}

/// What the head RealSense reports about the step ahead.
///
/// Extraction is a SCRIPTED perception step, not something the policy learns:
/// a plane fit plus edge detection on the depth image yields distance, height
/// and edge orientation directly. The policy's job is only to EXECUTE the step
/// given those numbers -- which is why this is 5 floats into the existing MLP
/// rather than a CNN on raw depth.
///
/// `valid = 0` must mean "no step information" and the policy must walk
/// normally on it -- otherwise a dropped detection on hardware becomes a fall
/// rather than a refusal.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepCue {
    /// Horizontal distance from the stance to the edge, m. Meaningless when
    /// `valid == 0`.
    pub distance: f32,
    /// Signed height of the surface beyond the edge, m. Positive = step UP,
    /// negative = step DOWN.
    pub height: f32,
    /// Edge normal direction in the BODY frame, as (sin, cos) of the angle
    /// between the robot's heading and the edge normal. (0, 1) means the edge
    /// is square-on. Split into sin/cos rather than a raw angle so it is
    /// continuous across the +/-pi wrap, and so the mirror transform is a
    /// clean sign flip on sin alone.
    pub edge_sin: f32,
    pub edge_cos: f32,
    /// 1.0 when a detection succeeded, 0.0 otherwise (no step, or no reading).
    pub valid: f32,
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
    /// Action before THAT — third point for the true second difference
    /// (`action_rate_rate` = Σ(a − 2a' + a″)²), the action-side tremor
    /// penalty: dither is action oscillation, so it's priced at the source
    /// with no contact sensor in the loop (the sensor-side force_rate term
    /// bred a straight-leg, transfer-breaking gait).
    pub prev2_action: [f32; NUM_JOINTS],
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
    /// Latest step-cue report AS THE ACTOR SEES IT (noised, dropout-masked).
    pub step_cue: StepCue,
    /// The clean oracle cue, critic-only (asymmetric AC). Zero when unknown.
    pub step_cue_clean: StepCue,
    /// Held-joint (upper body) PD targets, staging order. Home pose when
    /// playback is off; zero-padded past the robot's held-joint count.
    pub held_pos: [f32; NUM_HELD_OBS],
    /// Finite-diff velocity of `held_pos` (per control step / dt).
    pub held_vel: [f32; NUM_HELD_OBS],
}

impl Default for RobotState {
    fn default() -> Self {
        Self {
            base: BaseState::default(),
            joint_pos: [0.0; NUM_JOINTS],
            joint_vel: [0.0; NUM_JOINTS],
            last_action: [0.0; NUM_JOINTS],
            prev_action: [0.0; NUM_JOINTS],
            prev2_action: [0.0; NUM_JOINTS],
            feet: [FootObs::default(); NUM_FEET],
            phase: 0.0,
            step_cue: StepCue::default(),
            step_cue_clean: StepCue::default(),
            held_pos: [0.0; NUM_HELD_OBS],
            held_vel: [0.0; NUM_HELD_OBS],
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

/// Parse an env var holding a `"lo,hi"` range, e.g. `BIPED_YAW="-0.5,0.5"`.
fn range_env(var: &str, default: (f32, f32)) -> (f32, f32) {
    std::env::var(var)
        .ok()
        .and_then(|s| {
            let p: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (p.len() == 2).then(|| (p[0], p[1]))
        })
        .unwrap_or(default)
}

impl Default for CommandSampler {
    fn default() -> Self {
        // From the deployed Mjlab-Velocity-Flat-LeRobot config.
        Self {
            // Reachable at our budget: 0.5 m/s is achievable, 0.8 is not. With an
            // unreachable max command the curriculum forces the policy into a
            // regime where tracking reward is uniformly tiny → it gives up.
            // Override per-axis with BIPED_VX / BIPED_VY / BIPED_YAW ("lo,hi").
            lin_vel_x: range_env("BIPED_VX", (-0.8, 0.8)),
            lin_vel_y: range_env("BIPED_VY", (-0.5, 0.5)),
            // Yaw was ±0.2 through v21 — 5× narrower than WBC-AGILE T1's ±1.0,
            // and too narrow to learn from: at |yaw| ≤ 0.2 the exp tracking
            // kernel is nearly satisfied by NOT turning, so `track_ang_vel`
            // earned only 0.006/step (13% of linear tracking) and v21 turned
            // the WRONG WAY on a commanded ±0.2 (measured +0.03 / +0.02 rad/s
            // for ±0.5, −0.05 for +0.2). Raising the weight 5→8 had already
            // failed to fix it — the problem is the command range, not the
            // price. ±0.6 gives the kernel something to discriminate while
            // staying inside what the 12-DOF platform can turn.
            // NOTE evaluations: ±0.5 was OUTSIDE the old training range, so
            // every pre-v22 yaw probe at 0.5 was measuring extrapolation.
            ang_vel_z: range_env("BIPED_YAW", (-1.0, 1.0)),
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
                .unwrap_or((8.0, 12.0)),
        }
    }
}

impl CommandSampler {
    /// Draw a fresh command.
    pub fn sample(&self, rng: &mut Lcg) -> VelocityCommand {
        if rng.chance(self.standing_prob) {
            return VelocityCommand::default();
        }
        let mut cmd = VelocityCommand {
            vx: rng.range(self.lin_vel_x.0, self.lin_vel_x.1),
            vy: rng.range(self.lin_vel_y.0, self.lin_vel_y.1),
            yaw_rate: rng.range(self.ang_vel_z.0, self.ang_vel_z.1),
        };
        // Low-speed mass (BIPED_SLOW_PROB, default 0.25): uniform range
        // sampling puts almost no probability on SLOW commands relative to
        // the reward's sharp kernel, and the terrain curriculum pays for
        // travel — the 30k-iter policy learned exactly two speeds (0 and
        // ~±1 m/s), saturating every magnitude in between. Slow draws are
        // rescaled onto a speed in [0.12, 0.3]: the quasi-static regime where
        // single-support balance is hardest (and sim2sim falls live), kept
        // strictly above the 0.1 standing threshold so the gait clock runs.
        let slow_prob: f32 = std::env::var("BIPED_SLOW_PROB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        // Arc bias (BIPED_ARC_PROB, default 0.25): with the yaw range widened
        // to +/-0.6, independent uniform draws still put almost no mass on
        // CURVED walking -- measured only ~5% of moving commands were strong
        // arcs (|yaw|>=0.15 with |vx|>=0.3). Explicitly pair a real forward
        // speed with a real yaw rate so turning-while-walking is in the data
        // at all, not just turning-in-place.
        let arc_prob: f32 = std::env::var("BIPED_ARC_PROB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.25);
        if rng.chance(arc_prob) {
            let sx = if rng.chance(0.5) { 1.0 } else { -1.0 };
            let sy = if rng.chance(0.5) { 1.0 } else { -1.0 };
            cmd.vx = sx * rng.range(0.25, self.lin_vel_x.1.abs().max(0.25));
            cmd.yaw_rate = sy * rng.range(0.2, self.ang_vel_z.1.abs().max(0.2));
            cmd.vy = 0.0;
            return cmd;
        }
        if rng.chance(slow_prob) {
            let target = rng.range(0.12, 0.3);
            let speed = cmd.speed();
            if speed > 1e-6 {
                let k = target / speed;
                cmd.vx *= k;
                cmd.vy *= k;
                cmd.yaw_rate *= k;
            }
        }
        // A draw below the 0.1 standing threshold is contradictory: the gait
        // clock freezes and stand_planted engages while tracking still asks
        // for motion. Snap it to a true stand.
        if cmd.speed() < 0.1 {
            return VelocityCommand::default();
        }
        cmd
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
    /// Target base (torso) height, m (for `base_height`) while MOVING.
    pub base_height_target: f32,
    /// Target base height, m, while STANDING (`cmd.speed() < 0.1`, the same gate
    /// the gait clock freezes on). Defaults to `base_height_target`, so the
    /// split is inert until `BIPED_BASE_HEIGHT_STAND` is set.
    ///
    /// Standing tall is nearly free in height but expensive in knee angle:
    /// measured on this robot, height = 0.580*cos(knee/2) + 0.259, so the leg
    /// ceiling is 0.839 m at knee 0 and the curve goes FLAT as it straightens
    /// (15 deg -> 0 deg of knee buys only 5 mm). Two consequences:
    ///   * a target ABOVE ~0.839 is unreachable without hyperextending into the
    ///     -0.087 rad knee stop. That is what v22 did at 0.841 (locked at -4 deg,
    ///     60% of frames on the stop, CoT 1.00) -- do not set one.
    ///   * above ~0.837 the gradient falls under 0.5 mm/deg, so the only way to
    ///     chase the last millimetres is to ride the stop.
    /// 0.835 -> ~13.6 deg knee with 18.6 deg of stop margin, 4 mm under the
    /// ceiling: reachable at a legal knee angle, unlike v22's target.
    pub base_height_target_stand: f32,
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
    /// Penalty (per airborne foot, per step) while the command is STANDING.
    /// Makes the policy absorb small pushes with ankle/hip torque (feet
    /// planted, CoM shifted) instead of dance-stepping
    /// in place. MODERATE by design: for a big shove a protective step (brief
    /// penalty) must stay cheaper than falling (termination), mirroring the
    /// human ankle→hip→step strategy ladder. 0 = off.
    pub stand_planted: f32,
    /// Foot-slip penalty (horizontal foot speed while in contact).
    pub foot_slip: f32,
    /// Action second-difference penalty (NEGATIVE): Σ(a − 2a′ + a″)². The
    /// ACTION-side tremor term — high-frequency dither is literally an
    /// oscillating action, so this prices it at the policy's output with no
    /// engine-specific signal in the loop. Complements (does not stack with)
    /// `action_rate`, which only prices speed of change, not oscillation:
    /// a smooth fast ramp scores high action_rate but ~0 here. 0 = off.
    pub action_rate_rate: f32,
    /// Touchdown-impact penalty (NEGATIVE): per airborne foot below
    /// `touchdown_vz_h` (m), squared excess descent speed beyond
    /// `touchdown_vz_ok` (m/s). Kinematic slam term: a soft landing decel-
    /// erates BEFORE the ground, a slam is still fast at 5 cm. 0 = off.
    pub touchdown_vz: f32,
    /// Allowed descent speed at the gate height (m/s, default 0.3).
    pub touchdown_vz_ok: f32,
    /// Gate height above local ground (m, default 0.10). NOTE this compares
    /// against `FootObs.height`, which is the foot link ORIGIN — ~0.035 m
    /// above the sole even when planted (FOOT_REST_H). The first default
    /// (0.06) left a ~2 cm trigger window that contact sensing pre-empted:
    /// the term logged exactly 0 for 100 iters of newborn flailing. 0.10 =
    /// sole ~6.5 cm up, a real approach window.
    pub touchdown_vz_h: f32,
    /// Ground-reaction smoothness penalty (NEGATIVE): per foot, the one-step
    /// sensed-force change above `force_rate_deadband` (both in body weights
    /// per control step), squared. Prices the two behaviours v26 shipped with:
    /// foot-slam touchdowns (measured 0.9 BW mean / 2.1 BW peak first-contact
    /// jumps in MuJoCo) and the standing tremor (left/right load share
    /// dithering 0→1 at ~18 Hz with both feet planted — invisible in base
    /// motion, but exactly a sustained |ΔF|). Deadband keeps normal gait
    /// weight transfer (~0.06–0.12 BW/step) free, so walking itself is not
    /// taxed. 0 = off. Needs BIPED_CONTACT_SENSE (no sensor → term is 0).
    pub force_rate: f32,
    /// Deadband for `force_rate`, body weights per control step (default 0.15).
    pub force_rate_deadband: f32,
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
    /// WBC's `feet_yaw_diff_l2`: penalty on the squared wrapped yaw difference
    /// between the two feet (splayed/pigeon-toed stance). AGILE G1 weight −0.1,
    /// lerobot −0.02.
    pub feet_yaw_diff: f32,
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
            base_height_target_stand: 0.72, // = target: split is opt-in
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
            flight: -1.0,       // keep: no hopping (both feet airborne)
            stand_planted: 0.0, // OFF by default (A/B via BIPED_STAND_PLANTED_W):
            // per-airborne-foot penalty while the command is
            // standing → balance with ankles/hips, not dance-steps.
            single_support: 0.5, // REPURPOSED → double-support SETTLE bonus (both feet
            // planted while moving). Modest, so it shapes a
            // "swing → settle → swing" cycle without farmable waddle.
            action_rate_rate: 0.0, // off — enable with BIPED_W_ACTION_RATE_RATE
            touchdown_vz: 0.0,     // off — enable with BIPED_W_TOUCHDOWN_VZ
            touchdown_vz_ok: 0.3,
            touchdown_vz_h: 0.10,
            force_rate: 0.0, // off by default — enable with BIPED_W_FORCE_RATE
            force_rate_deadband: 0.15,
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
            feet_yaw_diff: 0.0, // OFF: keeps the tuned baseline reward unchanged
            // (runs are compared against a pinned iter-0 reference). WBC has it
            // in every config (G1 -0.1, lerobot -0.02); enable via agile() or a
            // struct update when the next baseline is re-pinned.
            feet_distance: -0.02,        // WBC feet_distance_from_ref -0.02 (was -0.1)
            feet_distance_ref: 0.2,
            gait_clock: 3.0, // dense periodic gait reward (the load-bearing
            // stepping signal). Symmetric ±: standing during a
            // swing window is penalized, so lifting on schedule
            // is clearly worth more than staying planted.
            gait_swing_ratio: 0.4, // 40% swing per foot → 20% double-support overlap
        }
    }
}

impl RewardWeights {
    /// WBC-AGILE's G1 `velocity_history` reward set, term for term
    /// (`BIPED_AGILE_REWARDS=1`). AGILE has NO stepping rewards — locomotion
    /// emerges from sharp tracking + terrain — so every zealot-only shaping
    /// term (gait_clock, air_time, single_support, bilateral_symmetry,
    /// pose, forward_progress) is zeroed here. Terms AGILE has
    /// that we can't express (action_rate_rate, root_acc, dof_vel_limits,
    /// torque family — those live in the env's BIPED_TORQUE_W/POWER_W hooks)
    /// are noted at the call site. `flight` carries AGILE's `jumping` weight.
    pub fn agile() -> Self {
        Self {
            track_lin_vel: 5.0,
            forward_progress: 0.0,
            track_ang_vel: 5.0,
            upright: 5.0,
            base_height: 2.5,
            base_height_target: 0.72, // AGILE DEFAULT_PELVIS_HEIGHT (bent-knee stance)
            base_height_target_stand: 0.72, // = target: split is opt-in
            pose: 0.0,
            bilateral_symmetry: 0.0,
            action_rate: -0.25,
            action_rate_hipz_hipx: 0.0,
            body_ang_vel: -0.25,
            lin_vel_z: -0.25,
            dof_pos_limits: -0.5,
            dof_vel: -1e-4,
            termination: -2.0, // AGILE is_terminated -100 × dt
            air_time: 0.0,
            flight: -20.0, // AGILE `jumping`
            single_support: 0.0,
            stand_planted: 0.0,
            action_rate_rate: 0.0, // WBC has action_rate_rate; weight unknown, keep off
            touchdown_vz: 0.0,
            touchdown_vz_ok: 0.3,
            touchdown_vz_h: 0.10,
            force_rate: 0.0, // no WBC equivalent (their contact_forces cap is absolute, not rate)
            force_rate_deadband: 0.15,
            foot_slip: -0.05, // AGILE feet_slip
            foot_clearance: 0.0,
            foot_clearance_target: 0.03,
            foot_orientation: -0.05, // AGILE feet_roll_l2
            feet_yaw_mean: -2.0,
            feet_yaw_diff: -0.1, // AGILE feet_yaw_diff_l2 (no turn reduction in G1 cfg)
            feet_distance: -0.1,
            feet_distance_ref: 0.2,
            gait_clock: 0.0,
            gait_swing_ratio: 0.4,
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

impl RewardStds {
    /// WBC-AGILE's exact kernel widths (`BIPED_AGILE_REWARDS=1`).
    pub fn agile() -> Self {
        Self {
            lin_vel: 0.2,
            ang_vel: 0.2,
            upright: 10_f32.to_radians(),
            base_height: 0.1,
            pose: 1.0,
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
    /// Feet-planted-while-standing penalty contribution.
    pub stand_planted: f32,
    /// Foot-slip penalty contribution.
    pub foot_slip: f32,
    /// Ground-reaction-smoothness (force-rate) penalty contribution.
    pub force_rate: f32,
    /// Action second-difference (tremor-at-source) penalty contribution.
    pub action_rate_rate: f32,
    /// Kinematic touchdown-impact (descent-speed) penalty contribution.
    pub touchdown_vz: f32,
    /// Foot-clearance penalty contribution.
    pub foot_clearance: f32,
    /// Flat-foot (sole-tilt) penalty contribution.
    pub foot_orientation: f32,
    /// Foot-yaw-vs-base penalty contribution.
    pub feet_yaw_mean: f32,
    /// Left/right foot-yaw-difference penalty contribution.
    pub feet_yaw_diff: f32,
    /// Lateral foot-distance penalty contribution.
    pub feet_distance: f32,
    /// Periodic gait-clock contribution (dense reward for matching each foot's
    /// swing/stance to the gait phase).
    pub gait_clock: f32,
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
            + self.stand_planted
            + self.foot_slip
            + self.force_rate
            + self.action_rate_rate
            + self.touchdown_vz
            + self.foot_clearance
            + self.foot_orientation
            + self.feet_yaw_mean
            + self.feet_yaw_diff
            + self.feet_distance
            + self.gait_clock
    }
}

/// The flat velocity-tracking task.
#[derive(Clone, Debug)]
pub struct VelocityFlatTask {
    /// The robot spec (gains, default pose, limits, joint order).
    pub robot: RobotSpec,
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
    /// Cue distance under which the step-manoeuvre relaxation applies (m).
    pub step_relax_dist: f32,
    /// Widened base-height kernel while stepping (m).
    pub step_std_base_h: f32,
    /// Widened upright kernel while stepping (rad).
    pub step_std_upright: f32,
    /// `|yaw_rate|` at which the LATERAL half of `bilateral_symmetry` is fully
    /// released (`BIPED_SYM_YAW_GATE`; 0 = off, historical behaviour). See
    /// `symmetry_error` for why turning is otherwise taxed by that term.
    pub sym_yaw_gate: f32,
}

impl Default for VelocityFlatTask {
    fn default() -> Self {
        Self::new()
    }
}

impl VelocityFlatTask {
    /// Hip yaw/roll joint indices — the subset `action_rate_hipz_hipx` sums
    /// over. Exposed so the GPU reward kernel can be handed the same mask.
    pub fn hip_yawroll_idx(&self) -> [usize; 4] {
        self.hip_yawroll_idx
    }

    /// Build the task with the deployed policy's settings, for the robot
    /// selected by `BIPED_ROBOT` (see [`RobotSpec::from_env`]).
    pub fn new() -> Self {
        Self::for_robot(RobotSpec::from_env())
    }

    /// Build the task for an explicit robot spec.
    pub fn for_robot(robot: RobotSpec) -> Self {
        // The hip yaw/roll DOFs for the targeted action-rate penalty (these
        // lateral-hip joints are the jittery ones) come from the spec.
        let hip_yawroll_idx = robot.hip_yawroll;
        // Reward-weight overrides for fast retuning without a rebuild. The
        // stand-still local optimum (policy collects upright + base_height +
        // free track_ang at zero command, ignores the velocity command) is the
        // walking blocker, so the key dials are the velocity-tracking weight vs
        // the standing magnets (upright / base_height). Set e.g.
        // `BIPED_W_TRACK_LIN=10 BIPED_W_UPRIGHT=3 BIPED_W_BASE_H=1.5` at launch.
        // BIPED_AGILE_REWARDS=1: WBC-AGILE's exact G1 term set — no stepping
        // rewards, no extra stand income, AGILE kernel widths, AGILE's 0.72 m
        // bent-knee height target (NOT the per-robot straight-leg default).
        // Pair with BIPED_POWER_W=0 (AGILE has no power term).
        let agile_rewards = std::env::var("BIPED_AGILE_REWARDS").is_ok_and(|v| v == "1");
        if agile_rewards {
            println!(
                "AGILE reward parity ENABLED: WBC-AGILE G1 term set (no stepping rewards), \
                 stds lin/ang 0.2, height target 0.72"
            );
        }
        let mut weights = if agile_rewards {
            RewardWeights::agile()
        } else {
            RewardWeights::default()
        };
        // The base-height target is per-robot (lerobot trunk 0.72 m, G1 pelvis
        // 0.78 m, H2 Plus pelvis 1.03 m); the RewardWeights default keeps the
        // lerobot value for struct-literal users.
        if !agile_rewards {
            weights.base_height_target = robot.base_height;
        }
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
        // Target height for the base_height kernel. AGILE's 0.72 is a
        // bent-knee stance; measured v21 sits 4-9 cm BELOW even that
        // (0.63-0.68 m vs 0.79 natural) because crouching buys balance
        // stability and the wide sigma=0.1 kernel makes the undershoot
        // cheap. 0.75 is the recommended first step up; 0.78+ risks the
        // straight-knee singularity where the leg loses vertical control
        // authority.
        // NOTE: BIPED_BASE_HEIGHT sets BOTH targets, so the historical
        // single-knob behaviour is preserved exactly. BIPED_BASE_HEIGHT_STAND
        // is applied AFTER it and overrides the standing one only -- order
        // matters, do not reorder these two blocks.
        if let Some(v) = env_f32("BIPED_BASE_HEIGHT") {
            weights.base_height_target = v;
            weights.base_height_target_stand = v;
        }
        // Command-conditioned standing height. The policy already prefers a
        // taller stance than its gait (v24 measured 0.816 standing vs 0.807
        // walking), so this codifies an existing preference rather than
        // fighting one. See the field docs for the knee-angle geometry and why
        // anything above ~0.837 rides the knee stop.
        weights.base_height_target_stand =
            env_f32("BIPED_BASE_HEIGHT_STAND").unwrap_or(weights.base_height_target + 0.01);
        // AGILE-alignment override: WBC has NO air-time reward — its gait
        // economy comes from torque/energy regularizers. Paying completed
        // swing DURATION (capped 0.4s ≈ our natural swing) selects for
        // maximal, exaggerated swings once a gait exists. 0 = AGILE parity.
        if let Some(v) = env_f32("BIPED_W_AIR_TIME") {
            weights.air_time = v;
        }
        let mut stds = if agile_rewards {
            RewardStds::agile()
        } else {
            RewardStds::default()
        };
        if let Some(v) = env_f32("BIPED_STD_LIN") {
            stds.lin_vel = v;
        }
        // Width of the height kernel (BIPED_STD_BASE_H, metres). The default
        // 0.1 is wide enough that a 4 cm undershoot still collects 85% of the
        // reward, which is why v21 sat 4 cm below its own target and stayed
        // there. Tightening to ~0.05 makes a crouch cost the height reward
        // outright -- the direct way to price posture, instead of taxing knee
        // torque and hoping the posture follows.
        stds.base_height = env_f32("BIPED_STD_BASE_H").unwrap_or(0.05);
        // Width of the ANGULAR tracking kernel. The default 0.1 is far too
        // narrow for the +/-0.6 command range: a policy sitting at the measured
        // 0.06 rad/s scores exp(-(0.4-0.06)^2/0.1^2) = 1e-5 on a 0.4 command,
        // i.e. NO gradient to climb. Measured on v24 at iter 32-42k, which
        // never learned to turn despite the gyro, the widened yaw range and 25%
        // arc sampling -- the arcs sample yaw in [0.2, 0.6], of which only the
        // 0.2 edge pays anything at all (14%; 0.35 pays 0.02%).
        //
        // v22 DID turn (+0.38) because it happened to land inside the basin,
        // after which a 0.4 command scores 96% and self-reinforces. That is
        // luck, not a gradient. 0.3 makes a 0.4 command pay 28% from a standing
        // start, so there is a slope the whole way in.
        stds.ang_vel = env_f32("BIPED_STD_ANG").unwrap_or(0.3);
        // Action-rate penalty gain (NEGATIVE). Exposed because it is denominated
        // in ACTION units, not radians: it charges (delta action)^2, so halving
        // BIPED_ACTION_SCALE makes the same PHYSICAL motion cost 4x more here.
        // Anyone running action_scale 0.25 must scale this by ~1/4 or the
        // penalty dominates and the policy simply stops moving.
        if let Some(v) = env_f32("BIPED_W_ACTION_RATE") {
            weights.action_rate = v;
        }
        Self {
            robot,
            weights,
            stds,
            sim_dt: 1.0 / 200.0,
            decimation: 4,
            episode_s: 20.0,
            tilt_limit: 70.0_f32.to_radians(),
            min_base_height: robot.min_base_height,
            hip_yawroll_idx,
            sym_yaw_gate: env_f32("BIPED_SYM_YAW_GATE").unwrap_or(0.0),
            step_relax_dist: env_f32("BIPED_STEP_RELAX_DIST").unwrap_or(0.6),
            step_std_base_h: env_f32("BIPED_STEP_STD_BASE_H").unwrap_or(0.15),
            step_std_upright: env_f32("BIPED_STEP_STD_UPRIGHT")
                .unwrap_or(25.0_f32.to_radians()),
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
    /// projected_gravity(3), gait_clock(2), base_ang_vel(3)]`.
    /// `joint_pos_rel = q − default_pos`.
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
        // Base angular velocity (body frame). Kept LAST so the preceding 45
        // slots keep their v21 meaning — the mirror transform, the sim2sim
        // harnesses and the lerobot controller all index by position.
        for w in self.base_ang_vel_body(&state.base) {
            put(obs, &mut o, w);
        }
        // Step cue LAST, so slots 0-47 keep their v22-v27 meaning. Distance is
        // clipped: beyond ~1.5 m the exact value carries no useful information
        // and an unclipped range would waste normalizer resolution on it.
        // When `valid == 0` the other two are forced to zero, so "no step" is a
        // single unambiguous pattern rather than stale numbers the policy might
        // still act on.
        let cue = state.step_cue;
        let live = cue.valid > 0.5;
        put(obs, &mut o, if live { cue.distance.clamp(-0.5, 1.5) } else { 0.0 });
        put(obs, &mut o, if live { cue.height.clamp(-0.4, 0.4) } else { 0.0 });
        put(obs, &mut o, if live { cue.edge_sin } else { 0.0 });
        put(obs, &mut o, if live { cue.edge_cos } else { 0.0 });
        put(obs, &mut o, if live { 1.0 } else { 0.0 });
        // Upper-body block LAST (slots 53..79), so every earlier slot keeps its
        // v22-v28 meaning — mirror transform / sim2sim / controller all index
        // by position. No noise: targets are the controller's own commands.
        for i in 0..NUM_HELD_OBS {
            put(obs, &mut o, state.held_pos[i]);
        }
        for i in 0..NUM_HELD_OBS {
            put(obs, &mut o, state.held_vel[i]);
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
        obs[OBS_DIM + 3..OBS_DIM + 6].copy_from_slice(&w);
        // Clean cue (see CRITIC_OBS_DIM): from step_cue_clean, which the env
        // fills with the raw oracle BEFORE actor-side noise and dropout.
        let cc = state.step_cue_clean;
        let live = cc.valid > 0.5;
        obs[OBS_DIM + 6] = if live { cc.distance.clamp(-0.5, 1.5) } else { 0.0 };
        obs[OBS_DIM + 7] = if live { cc.height.clamp(-0.4, 0.4) } else { 0.0 };
        obs[OBS_DIM + 8] = if live { cc.edge_sin } else { 0.0 };
        obs[OBS_DIM + 9] = if live { cc.edge_cos } else { 0.0 };
        obs[OBS_DIM + 10] = if live { 1.0 } else { 0.0 };
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

        // STEP MANOEUVRE GATE. Climbing a step is flatly irrational under the
        // flat-ground reward: the base MUST rise relative to the surface it is
        // leaving (no choice of reference foot avoids that), and it MUST pitch
        // forward to get the swing leg up. Measured against a 0.20 m riser,
        // holding the normal kernels costs ~0.040/step of base_height plus
        // ~0.072/step of upright at a 15 deg lean -- together ~42% of all
        // positive reward, against a ~0.002 behaviour threshold. The policy
        // would correctly learn to refuse.
        //
        // So while a step is cued and close, widen both kernels rather than
        // moving their targets: the target is still where we want the robot to
        // END UP, we are only declining to charge it for the transient.
        //
        // PROGRESS-GATED, and this is load-bearing. The first version relaxed
        // on proximity alone, and the policy found the hack within 15k iters:
        // standing inside the relax zone earns near-max posture reward in ANY
        // pose, so it learned "cue valid -> stop". Measured on the step eval,
        // same checkpoint, same 10 cm step: cue ON froze at x=0.16 m while cue
        // OFF (blind) walked to 1.28 m -- the cue made behaviour WORSE, and
        // the refusal strengthened with training while turning and gait
        // atrophied alongside it (loitering generalises). Requiring real
        // motion toward the edge means a loiterer faces the normal kernels and
        // the zone pays nothing unless the manoeuvre is actually happening.
        // Threshold 0.1 m/s matches the standing predicate's scale.
        let toward_step =
            v[FWD] * state.step_cue.edge_cos + v[LAT] * state.step_cue.edge_sin;
        let stepping = state.step_cue.valid > 0.5
            && state.step_cue.distance.abs() < self.step_relax_dist
            && toward_step > 0.1;
        let (std_h, std_up) = if stepping {
            (self.step_std_base_h, self.step_std_upright)
        } else {
            (self.stds.base_height, self.stds.upright)
        };

        // Upright: horizontal components of projected gravity → 0 when flat.
        let tilt_err = grav[FWD].powi(2) + grav[LAT].powi(2);
        let upright = self.weights.upright * (-tilt_err / std_up.powi(2)).exp() * dt;

        // Base height: stand tall (exp kernel around the target) so the policy
        // can't trivially crouch to avoid falling. The target is
        // command-conditioned: standing may ask for a taller, straighter-legged
        // stance than the gait wants, which cuts the continuous knee holding
        // torque (measured 28 N-m at a 31 deg stand knee) without forcing the
        // gait itself higher. Same `cmd.speed() < 0.1` gate the gait clock,
        // stand_planted and single_support already switch on, so this adds no
        // new discontinuity -- it rides one that is already there.
        let h_target = if cmd.speed() < 0.1 {
            self.weights.base_height_target_stand
        } else {
            self.weights.base_height_target
        };
        let h_err = (state.base.height - h_target).powi(2);
        let base_height = self.weights.base_height * (-h_err / std_h.powi(2)).exp() * dt;

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
        // Release the LATERAL half of the symmetry term in proportion to the
        // commanded turn: 1.0 going straight, 0.0 at |yaw_rate| >= the gate.
        // Off (gate 0) = historical behaviour, lateral at full weight always.
        let lateral_scale = if self.sym_yaw_gate > 0.0 {
            (1.0 - cmd.yaw_rate.abs() / self.sym_yaw_gate).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let sym_err = self.symmetry_error(&state.joint_pos, lateral_scale);
        let bilateral_symmetry = self.weights.bilateral_symmetry * (-sym_err).exp() * dt;

        // Penalties (negative weights).
        let mut da2 = 0.0;
        for i in 0..NUM_JOINTS {
            da2 += (state.last_action[i] - state.prev_action[i]).powi(2);
        }
        let action_rate = self.weights.action_rate * da2 * dt;

        // Action second difference (tremor at the source): Σ(a − 2a′ + a″)².
        let mut dda2 = 0.0;
        for i in 0..NUM_JOINTS {
            let dd = state.last_action[i] - 2.0 * state.prev_action[i] + state.prev2_action[i];
            dda2 += dd * dd;
        }
        let action_rate_rate = self.weights.action_rate_rate * dda2 * dt;

        // Touchdown impact (kinematic slam): an airborne foot still
        // descending faster than `touchdown_vz_ok` below the gate height.
        let mut slam = 0.0;
        for f in &state.feet {
            if !f.contact && f.height < self.weights.touchdown_vz_h {
                let ex = (-f.vz - self.weights.touchdown_vz_ok).max(0.0);
                slam += ex * ex;
            }
        }
        let touchdown_vz = self.weights.touchdown_vz * slam * dt;

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

        // Extra "balance, don't step" pressure at stand (NEGATIVE weight, 0 = off;
        // BIPED_STAND_PLANTED_W): per airborne foot per step while the command is
        // standing. Stacks on the binary standing branch of `single_support`
        // above. Sized so quiet ankle/hip balancing beats fidget-stepping under
        // small pushes, while a genuine protective step (a few penalized steps)
        // stays far cheaper than falling (termination) under big ones.
        let stand_planted = if standing {
            let airborne = state.feet.iter().filter(|f| !f.contact).count() as f32;
            self.weights.stand_planted * airborne * dt
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

        // Ground-reaction smoothness: squared excess |ΔF| (body weights per
        // control step) above the deadband, per foot. Slam touchdowns and the
        // standing load-dither tremor both live here; gait weight transfer
        // stays inside the deadband. See the `force_rate` weight doc.
        let mut frate = 0.0;
        for f in &state.feet {
            let ex = (f.force_rate - self.weights.force_rate_deadband).max(0.0);
            frate += ex * ex;
        }
        let force_rate = self.weights.force_rate * frate * dt;

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
        // REWARDING THE STEP. While a step-up is cued and close, raise the
        // clearance target to the riser plus a margin, so "enough lift" means
        // enough to actually clear THIS step rather than a fixed 3 cm. The
        // reward saturates at the target (lift.min(1.0)), so raising it is
        // exactly "you must lift higher here" and nothing else changes.
        //
        // This reuses the existing term ON PURPOSE. foot_clearance was dropped
        // once because a static foot-height reward got hacked into a one-foot
        // statue (foot held up to farm it; zero transfer, MuJoCo fell in
        // 0.66 s). The guards added in response -- swing-only, air_time <
        // 0.45 s so a held foot stops earning, moving-gated, capped -- are
        // right here and apply unchanged. A fresh "reward crossing a step"
        // term would have none of them.
        const STEP_CLEAR_MARGIN: f32 = 0.05;
        let clear_target = if stepping && state.step_cue.height > 0.0 {
            (state.step_cue.height + STEP_CLEAR_MARGIN)
                .max(self.weights.foot_clearance_target)
        } else {
            self.weights.foot_clearance_target
        };
        let mut foot_h = 0.0;
        for f in &state.feet {
            if !f.contact && f.air_time < MAX_SWING_S {
                let lift = (f.height - FOOT_REST_H).max(0.0) / clear_target;
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
        // UNGATED (AGILE feet_roll_l2 parity): penalize sole tilt on ALL feet,
        // not just loaded ones. The contact-gated form left swing-foot posture
        // free, and the learned policy exploited it (hard dorsiflexion — toes
        // pointing up through swing; functional-ish for toe-clearance but an
        // engine-flattered habit under box friction, and ugly).
        let mut tilt_sq = 0.0;
        for f in &state.feet {
            tilt_sq += f.tilt.powi(2);
        }
        let foot_orientation = self.weights.foot_orientation * tilt_sq * dt;

        // WBC's `feet_yaw_mean_vs_base`: sum of squared yaw (in base frame) over
        // both feet — drives each foot to point in the base's forward direction.
        let mut yaw_sq = 0.0;
        for f in &state.feet {
            yaw_sq += f.yaw_rel_base.powi(2);
        }
        let feet_yaw_mean = self.weights.feet_yaw_mean * yaw_sq * dt;

        // WBC's `feet_yaw_diff_l2`: squared wrapped yaw difference between the
        // two feet (splay / pigeon-toe). Base-relative yaws work here — the base
        // yaw cancels in the difference.
        let feet_yaw_diff = if NUM_FEET == 2 {
            let d = state.feet[1].yaw_rel_base - state.feet[0].yaw_rel_base;
            let d = (d + core::f32::consts::PI).rem_euclid(2.0 * core::f32::consts::PI)
                - core::f32::consts::PI;
            self.weights.feet_yaw_diff * d.powi(2) * dt
        } else {
            0.0
        };

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
            stand_planted,
            foot_slip,
            force_rate,
            action_rate_rate,
            touchdown_vz,
            foot_clearance,
            foot_orientation,
            feet_yaw_mean,
            feet_yaw_diff,
            feet_distance,
            gait_clock,
        }
    }

    /// Mirror error: pairs left/right joints via the spec's mirror permutation
    /// and accumulates the squared difference under the family's mirror sign
    /// (sagittal joints mirror equal, lateral joints mirror opposite).
    /// `lateral_scale` attenuates ONLY the mirror-opposite joints (hip_roll,
    /// hip_yaw, ankle_roll — `mirror_sign < 0`). Those are "symmetric" when
    /// `q_L = -q_R`, i.e. mirror images. Turning needs the opposite: both
    /// hip_yaws rotate the SAME way, so the error is `(q + q)^2 = 4q^2` — the
    /// worst case for that term. Measured on v26's weights, 0.2 rad of
    /// coordinated hip yaw costs ~0.005/step against a ~0.002 behaviour
    /// threshold, so a commanded turn is actively taxed for the one thing it
    /// must do. The sagittal terms (hip_pitch, knee, ankle_pitch) are what keep
    /// the gait even and are never attenuated.
    fn symmetry_error(&self, q: &[f32; NUM_JOINTS], lateral_scale: f32) -> f32 {
        let mut err = 0.0;
        for i in 0..NUM_JOINTS {
            let jr = self.robot.mirror[i];
            if jr <= i {
                continue; // count each L/R pair once
            }
            let e = (q[i] - self.robot.mirror_sign[i] * q[jr]).powi(2);
            err += if self.robot.mirror_sign[i] < 0.0 {
                lateral_scale * e
            } else {
                e
            };
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
        // 45 through v21; + base_ang_vel(3) from v22; + step_cue(5) from v28;
        // + upper-body block (13 held targets + 13 target vels) from v29.
        // This is the OBSERVATION CONTRACT -- the three sim2sim harnesses and
        // the lerobot controller all rebuild this vector by offset, so widening
        // it means updating them too, not just this number.
        assert_eq!(OBS_DIM, 79);
        assert_eq!(ACTION_DIM, 12);
        let task = VelocityFlatTask::for_robot(crate::robots::lerobot_bipedal::lerobot());
        let mut obs = vec![0.0; OBS_DIM];
        task.observe(&upright_state(), &VelocityCommand::default(), &mut obs);
        // Layout: last_action[0..12], command[12..16], joint_pos_rel[16..28],
        // joint_vel[28..40], projected_gravity[40..43], gait_phase(sin,cos)[43..45],
        // base_ang_vel[45..48], step_cue(dist, h, edge_sin, edge_cos, valid)[48..53].
        // Upright, neutral pose, zero command, phase 0 → everything zero except
        // gravity up = -1 and cos(0) = 1.
        assert!(obs.iter().take(40).all(|&x| x == 0.0));
        assert!((obs[42] - (-1.0)).abs() < 1e-6, "up component of gravity");
        assert!(obs[43].abs() < 1e-6, "sin(phase 0) = 0");
        assert!((obs[44] - 1.0).abs() < 1e-6, "cos(phase 0) = 1");
    }

    #[test]
    fn control_timing() {
        let task = VelocityFlatTask::for_robot(crate::robots::lerobot_bipedal::lerobot());
        assert!((task.control_dt() - 0.02).abs() < 1e-9);
        assert_eq!(task.max_steps(), 1000); // 20 s / 0.02 s
    }

    #[test]
    fn joint_targets_offset_by_scale() {
        let task = VelocityFlatTask::for_robot(crate::robots::lerobot_bipedal::lerobot());
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
        let task = VelocityFlatTask::for_robot(crate::robots::lerobot_bipedal::lerobot());
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
        let task = VelocityFlatTask::for_robot(crate::robots::lerobot_bipedal::lerobot());
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
            vz: 0.0,
            force_rate: 0.0,
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
        // Force-rate: off at weight 0; with a weight, a slam-sized 1.5 BW/step
        // jump is penalized, while normal gait weight transfer (0.1 BW/step,
        // inside the deadband) stays free.
        let mut task_fr = task.clone();
        task_fr.weights.force_rate = -1.0;
        let mut s_slam = RobotState::default();
        s_slam.feet[0].force_rate = 1.5;
        assert_eq!(task.reward(&s_slam, &cmd).force_rate, 0.0);
        assert!(task_fr.reward(&s_slam, &cmd).force_rate < 0.0);
        let mut s_walk = RobotState::default();
        s_walk.feet[0].force_rate = 0.1;
        s_walk.feet[1].force_rate = 0.1;
        assert_eq!(task_fr.reward(&s_walk, &cmd).force_rate, 0.0);
        // Sustained tremor-sized dither (0.4 BW/step on both feet) is taxed.
        let mut s_trem = RobotState::default();
        s_trem.feet[0].force_rate = 0.4;
        s_trem.feet[1].force_rate = 0.4;
        assert!(task_fr.reward(&s_trem, &cmd).force_rate < 0.0);
        // action_rate_rate: an OSCILLATING action (a, -a, a) is taxed; a
        // smooth constant-rate ramp (a, a+d, a+2d) has zero second
        // difference and is free — the distinction from plain action_rate.
        let mut task_dd = task.clone();
        task_dd.weights.action_rate_rate = -1.0;
        let mut s_osc = RobotState::default();
        s_osc.last_action[0] = 0.5;
        s_osc.prev_action[0] = -0.5;
        s_osc.prev2_action[0] = 0.5;
        assert!(task_dd.reward(&s_osc, &cmd).action_rate_rate < 0.0);
        let mut s_ramp = RobotState::default();
        s_ramp.last_action[0] = 0.6;
        s_ramp.prev_action[0] = 0.4;
        s_ramp.prev2_action[0] = 0.2;
        assert!(task_dd.reward(&s_ramp, &cmd).action_rate_rate.abs() < 1e-9);
        assert_eq!(task.reward(&s_osc, &cmd).action_rate_rate, 0.0); // off at weight 0
        // touchdown_vz: an airborne foot still dropping 1.0 m/s at 4 cm is a
        // slam; the same descent higher up, a decelerated (0.2 m/s) landing,
        // or a planted foot are all free.
        let mut task_td = task.clone();
        task_td.weights.touchdown_vz = -1.0;
        let mut s_slam = RobotState::default();
        s_slam.feet[0].contact = false;
        s_slam.feet[0].height = 0.04;
        s_slam.feet[0].vz = -1.0;
        assert!(task_td.reward(&s_slam, &cmd).touchdown_vz < 0.0);
        let mut s_high = RobotState::default();
        s_high.feet[0].contact = false;
        s_high.feet[0].height = 0.12;
        s_high.feet[0].vz = -1.0;
        assert_eq!(task_td.reward(&s_high, &cmd).touchdown_vz, 0.0);
        let mut s_soft = RobotState::default();
        s_soft.feet[0].contact = false;
        s_soft.feet[0].height = 0.04;
        s_soft.feet[0].vz = -0.2;
        assert_eq!(task_td.reward(&s_soft, &cmd).touchdown_vz, 0.0);
        // A foot tilted onto its edge while in contact → flat-foot penalty.
        let mut s3 = RobotState::default();
        s3.feet[0].tilt = 1.0; // ~57° off flat
        assert!(task.reward(&s3, &cmd).foot_orientation < 0.0);
        // The penalty is UNGATED: a tilted foot costs the same airborne as in
        // stance. Gating it to stance let the policy fly a toes-up swing foot
        // for free, which is exactly the heel-strike posture this term exists
        // to remove (AGILE's roll-only feet_roll_l2 cannot see it either).
        let mut s4 = RobotState::default();
        s4.feet[0].contact = false;
        s4.feet[0].tilt = 1.0;
        s4.feet[1].contact = false;
        assert!(task.reward(&s4, &cmd).foot_orientation < 0.0);
        // A foot yawed relative to the base → feet_yaw_mean penalty.
        let mut s5 = RobotState::default();
        s5.feet[0].yaw_rel_base = 0.5;
        assert!(task.reward(&s5, &cmd).feet_yaw_mean < 0.0);
    }

    #[test]
    fn velocity_error_reduces_tracking_reward() {
        let task = VelocityFlatTask::for_robot(crate::robots::lerobot_bipedal::lerobot());
        let mut s = upright_state();
        s.base.lin_vel_world = [0.5, 0.0, 0.0]; // moving forward
        // Command standing → big tracking error → reward below the max.
        let r = task.reward(&s, &VelocityCommand::default());
        let w = RewardWeights::default();
        assert!(r.track_lin_vel < 0.5 * w.track_lin_vel * task.control_dt());
    }

    #[test]
    fn fell_over_detects_tilt() {
        let task = VelocityFlatTask::for_robot(crate::robots::lerobot_bipedal::lerobot());
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
            assert!(c.vy.abs() <= 0.5 + 1e-6);
            assert!(c.yaw_rate.abs() <= 1.0 + 1e-6);
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

    /// A cued step-up must RAISE the clearance bar, so the same foot lift that
    /// saturated the reward on flat ground no longer does. Without this the
    /// policy is paid the same for a 3 cm lift whether or not there is a 20 cm
    /// riser in front of it.
    #[test]
    fn cued_step_up_raises_the_clearance_bar() {
        let mut task = VelocityFlatTask::new();
        task.weights.foot_clearance = 1.0; // off by default; enable to measure
        task.weights.foot_clearance_target = 0.03;

        // One foot mid-swing at 8 cm -- saturating on flat ground.
        let mut st = RobotState::default();
        st.feet[0].contact = false;
        st.feet[0].air_time = 0.1;
        st.feet[0].height = 0.08;
        st.feet[1].contact = true;
        // The adaptive bar inherits the progress gate: it only raises while
        // actually moving toward the edge.
        st.base.lin_vel_world = [0.4, 0.0, 0.0];
        let cmd = VelocityCommand { vx: 0.4, vy: 0.0, yaw_rate: 0.0 };

        let flat = task.reward(&st, &cmd).foot_clearance;
        let mut stepping = st;
        stepping.step_cue = StepCue {
            distance: 0.3, height: 0.20, edge_sin: 0.0, edge_cos: 1.0, valid: 1.0,
        };
        let on_step = task.reward(&stepping, &cmd).foot_clearance;
        assert!(
            on_step < flat,
            "8 cm of lift should NOT still saturate in front of a 20 cm riser \
             (flat {flat}, stepping {on_step})"
        );

        // And a lift that clears the riser earns it back.
        let mut cleared = stepping;
        cleared.feet[0].height = 0.20 + 0.05 + 0.035;
        assert!(
            (task.reward(&cleared, &cmd).foot_clearance - flat).abs() < 1e-6,
            "clearing riser + margin should saturate again"
        );
    }

    /// The step relaxation must fire ONLY when a step is cued and close, and
    /// must not change flat-ground reward at all -- otherwise every generation
    /// trained with the cue enabled is quietly running looser posture kernels
    /// everywhere, which is not the intent.
    #[test]
    fn step_relaxation_is_gated_and_inert_when_no_step() {
        let mut task = VelocityFlatTask::new();
        task.weights.base_height = 2.0;
        task.weights.upright = 4.0;
        task.stds.base_height = 0.05;
        task.weights.base_height_target = 0.82;
        task.weights.base_height_target_stand = 0.82;
        let cmd = VelocityCommand { vx: 0.4, vy: 0.0, yaw_rate: 0.0 };

        // Off the target by 0.20 m -- what crossing a 0.20 m edge looks like.
        let mut st = RobotState::default();
        st.base.height = 0.62;

        // Moving toward the edge at 0.3 m/s (body +x, edge square-on).
        st.base.lin_vel_world = [0.3, 0.0, 0.0];
        let no_cue = task.reward(&st, &cmd);
        let mut cued = st;
        cued.step_cue = StepCue { distance: 0.3, height: 0.2, edge_sin: 0.0, edge_cos: 1.0, valid: 1.0 };
        let with_cue = task.reward(&cued, &cmd);
        assert!(
            with_cue.base_height > no_cue.base_height,
            "relaxation did not fire while approaching a cued step ({} vs {})",
            with_cue.base_height, no_cue.base_height
        );

        // LOITERING in the zone (cued, close, but stationary) must get the
        // NORMAL kernels -- this is the reward hack the first version allowed:
        // the policy learned "cue valid -> stop" because standing in the relax
        // zone out-paid walking (measured: cue ON froze at x=0.16 m where cue
        // OFF walked to 1.28 m on the same checkpoint).
        let mut loiter = cued;
        loiter.base.lin_vel_world = [0.0, 0.0, 0.0];
        let still = task.reward(&loiter, &cmd);
        let mut still_no_cue = st;
        still_no_cue.base.lin_vel_world = [0.0, 0.0, 0.0];
        assert_eq!(
            still.base_height,
            task.reward(&still_no_cue, &cmd).base_height,
            "relaxation fired for a stationary robot parked at the edge"
        );

        // Cue valid but FAR: no relaxation, so the robot is not given a licence
        // to slouch merely because a step exists somewhere ahead.
        let mut far = st;
        far.step_cue = StepCue { distance: 1.4, height: 0.2, edge_sin: 0.0, edge_cos: 1.0, valid: 1.0 };
        assert_eq!(
            task.reward(&far, &cmd).base_height, no_cue.base_height,
            "relaxation fired for a step 1.4 m away"
        );
    }

    /// `valid = 0` must be a single unambiguous "no information" pattern:
    /// distance and height forced to zero, not stale values. On hardware a
    /// failed probe hits this path, and the policy must walk normally rather
    /// than act on whatever the last successful probe said.
    #[test]
    fn invalid_step_cue_zeroes_the_whole_block() {
        let task = VelocityFlatTask::for_robot(crate::robots::lerobot_bipedal::lerobot());
        let mut st = RobotState::default();
        let cmd = VelocityCommand::default();
        let mut obs = vec![0.0; OBS_DIM];

        st.step_cue = StepCue { distance: 0.42, height: 0.18, edge_sin: 0.5, edge_cos: 0.87, valid: 0.0 };
        task.observe(&st, &cmd, &mut obs);
        assert_eq!(&obs[48..53], &[0.0; 5], "stale cue leaked through valid=0");

        st.step_cue = StepCue { distance: 0.42, height: 0.18, edge_sin: 0.5, edge_cos: 0.87, valid: 1.0 };
        task.observe(&st, &cmd, &mut obs);
        assert_eq!(&obs[48..53], &[0.42, 0.18, 0.5, 0.87, 1.0]);
    }

    /// The cue must not disturb the slots every other consumer indexes by
    /// position -- the sim2sim harnesses and the lerobot controller all read
    /// 0..48 by offset.
    #[test]
    fn step_cue_is_appended_not_inserted() {
        let task = VelocityFlatTask::for_robot(crate::robots::lerobot_bipedal::lerobot());
        let mut st = RobotState::default();
        st.joint_pos[3] = 0.3;
        let cmd = VelocityCommand { vx: 0.4, vy: 0.0, yaw_rate: 0.0 };
        let mut with = vec![0.0; OBS_DIM];
        task.observe(&st, &cmd, &mut with);
        let mut cued = vec![0.0; OBS_DIM];
        let mut st2 = st;
        st2.step_cue = StepCue { distance: 1.0, height: 0.2, edge_sin: 0.0, edge_cos: 1.0, valid: 1.0 };
        task.observe(&st2, &cmd, &mut cued);
        assert_eq!(with[..48], cued[..48],
                   "setting the step cue changed a pre-existing observation slot");
    }

    #[test]
    fn symmetry_error_zero_for_mirrored_pose() {
        let task = VelocityFlatTask::for_robot(crate::robots::lerobot_bipedal::lerobot());
        // Neutral pose is trivially symmetric.
        assert_eq!(task.symmetry_error(&[0.0; NUM_JOINTS], 1.0), 0.0);
    }

    /// The lateral gate must attenuate ONLY the mirror-opposite joints
    /// (hip_roll/hip_yaw/ankle_roll) and leave the sagittal ones alone --
    /// otherwise gating a turn also stops enforcing an even gait.
    #[test]
    fn lateral_gate_spares_sagittal_symmetry() {
        let task = VelocityFlatTask::for_robot(crate::robots::lerobot_bipedal::lerobot());
        // Derive the indices from the SPEC rather than hardcoding a layout --
        // VelocityFlatTask::new() honours BIPED_ROBOT, so the default here is
        // lerobot, not the G1, and the joint order differs.
        let lat = (0..NUM_JOINTS)
            .find(|&i| task.robot.mirror[i] > i && task.robot.mirror_sign[i] < 0.0)
            .expect("spec has a mirror-opposite (lateral) pair");
        let sag = (0..NUM_JOINTS)
            .find(|&i| task.robot.mirror[i] > i && task.robot.mirror_sign[i] > 0.0)
            .expect("spec has a mirror-equal (sagittal) pair");
        let mut q = [0.0; NUM_JOINTS];
        // Coordinated motion on a LATERAL pair: what turning needs, and the
        // worst case for a mirror-opposite term.
        q[lat] = 0.2;
        q[task.robot.mirror[lat]] = 0.2;
        let full = task.symmetry_error(&q, 1.0);
        let released = task.symmetry_error(&q, 0.0);
        assert!(full > 0.0, "coordinated hip yaw should violate lateral mirror");
        assert_eq!(released, 0.0, "gate should fully release the lateral term");

        // Sagittal asymmetry must be charged at FULL weight whatever the gate.
        let mut s = [0.0; NUM_JOINTS];
        s[sag] = 0.2;
        assert_eq!(
            task.symmetry_error(&s, 0.0),
            task.symmetry_error(&s, 1.0),
            "sagittal symmetry must not depend on the lateral gate"
        );
    }
}

#[cfg(test)]
mod moving_gate_tests {
    use super::*;

    /// The `moving` gate is what keeps every stepping incentive (gait_clock,
    /// air_time, single_support) OFF at a standing command. If it ever leaks,
    /// the policy gets paid to march in place. Pin it directly.
    /// The launchers set these; the struct defaults do not (stand_planted
    /// shipped at 0 for several generations before v10 turned it on).
    fn launcher_task() -> VelocityFlatTask {
        let mut task = VelocityFlatTask::new();
        task.weights.stand_planted = -1.0;
        task.weights.gait_clock = 3.0;
        task.weights.air_time = 1.0;
        task.weights.single_support = 1.0;
        task
    }

    #[test]
    fn standing_gate_zeroes_every_stepping_term() {
        let task = launcher_task();
        // A state that WOULD earn every stepping term if the gate were open:
        // one foot airborne mid-swing, the other planted, base moving forward.
        let mut st = RobotState::default();
        st.feet[0].contact = false;
        st.feet[0].air_time = 0.1;
        st.feet[0].height = 0.12;
        st.feet[1].contact = true;
        st.base.lin_vel_world = [0.5, 0.0, 0.0];
        st.phase = 0.25;

        let stand = task.reward(&st, &VelocityCommand::default());
        assert_eq!(stand.gait_clock, 0.0, "gait_clock leaked at zero command");
        assert_eq!(stand.air_time, 0.0, "air_time leaked at zero command");
        // single_support is not gated off at stand -- it INVERTS: planted feet
        // are rewarded, stepping is penalized. Pin the sign, not a zero.
        assert!(
            stand.single_support < 0.0,
            "single_support should penalize a lifted foot at zero command: {}",
            stand.single_support
        );
        let mut planted = st.clone();
        planted.feet[0].contact = true;
        planted.feet[0].air_time = 0.0;
        assert!(
            task.reward(&planted, &VelocityCommand::default()).single_support > 0.0,
            "single_support should REWARD both feet planted at zero command"
        );
        // ...and stand_planted must actively CHARGE for the airborne foot.
        assert!(stand.stand_planted < 0.0, "stand_planted did not charge: {}", stand.stand_planted);

        // Same state under a moving command: the stepping terms come alive, so
        // the assertions above are testing the GATE, not a dead code path.
        let moving = task.reward(
            &st,
            &VelocityCommand { vx: 0.4, vy: 0.0, yaw_rate: 0.0 },
        );
        assert!(moving.gait_clock != 0.0, "gait_clock never fires even when moving");
        assert_eq!(moving.stand_planted, 0.0, "stand_planted fired while moving");
    }

    /// The standing height target must apply ONLY at a standing command, and
    /// must default to the moving target so the split stays inert unless asked
    /// for. A silent leak here would quietly re-target the whole gait.
    #[test]
    fn stand_height_target_is_command_gated() {
        let mut task = launcher_task();
        task.weights.base_height = 2.0;
        task.stds.base_height = 0.05;
        task.weights.base_height_target = 0.82;
        task.weights.base_height_target_stand = 0.835;

        // A base sitting exactly at the STANDING target: it should score better
        // at a standing command than the same height does while moving.
        let mut st = RobotState::default();
        st.base.height = 0.835;
        let standing = task.reward(&st, &VelocityCommand::default()).base_height;
        let moving = task
            .reward(&st, &VelocityCommand { vx: 0.4, vy: 0.0, yaw_rate: 0.0 })
            .base_height;
        assert!(
            standing > moving,
            "0.835 m should score higher standing ({standing}) than moving ({moving})"
        );

        // Yaw-only is MOVING (same predicate as the gait clock), so a pure turn
        // must use the walking target, not the standing one.
        let turning = task
            .reward(&st, &VelocityCommand { vx: 0.0, vy: 0.0, yaw_rate: 0.4 })
            .base_height;
        assert_eq!(turning, moving, "a pure yaw command must use the MOVING target");

        // Default: the stand target sits 1 cm above the moving target (the
        // production preference — v24 measured 0.816 standing vs 0.807 walking).
        let plain = launcher_task();
        assert!(
            (plain.weights.base_height_target_stand
                - (plain.weights.base_height_target + 0.01))
                .abs()
                < 1e-6,
            "the stand target must default to the moving target + 0.01"
        );
    }

    /// Yaw counts toward the speed magnitude, so a pure turn-in-place command
    /// is MOVING (this was a real v11 bug: the clock froze on linear speed
    /// only while the task's predicate included yaw).
    #[test]
    fn yaw_only_command_counts_as_moving() {
        let task = launcher_task();
        let mut st = RobotState::default();
        st.feet[0].contact = false;
        st.feet[0].air_time = 0.1;
        st.feet[1].contact = true;
        let turn = task.reward(&st, &VelocityCommand { vx: 0.0, vy: 0.0, yaw_rate: 0.4 });
        assert_eq!(turn.stand_planted, 0.0, "a yaw command was treated as standing");
        // And a sub-threshold command IS standing.
        let tiny = task.reward(&st, &VelocityCommand { vx: 0.05, vy: 0.0, yaw_rate: 0.0 });
        assert!(tiny.stand_planted < 0.0, "a 0.05 command was treated as moving");
    }
}
