//! SONIC-style whole-body motion tracking for the 25 live joints in zealot's
//! solver-compatible Unitree G1 model.
//!
//! The raw BONES-SEED G1 files contain a root transform and 29 joint angles at
//! 120 Hz. Zealot's G1 model welds the four low-torque wrist pitch/yaw joints,
//! so this module maps the remaining 25 columns by name and resamples the clip
//! to the 50 Hz policy rate. It intentionally implements the core motion
//! reference contract, not SONIC's universal-token network.

use crate::math::quat_rotate_inv;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// Number of live, policy-controlled joints in `unitree_g1_29dof.xml`.
pub const G1_CONTROLLED_JOINTS: usize = 25;
/// Number of joints in a raw BONES-SEED G1 CSV.
pub const BONES_G1_JOINTS: usize = 29;
/// SONIC/zealot control frequency.
pub const CONTROL_HZ: f32 = 50.0;
/// Default proprioceptive history length.
pub const PROP_HISTORY: usize = 10;
/// Default number of future reference frames.
pub const FUTURE_FRAMES: usize = 10;
/// Spacing between future reference frames, seconds.
pub const FUTURE_DT: f32 = 0.1;
/// Per-proprioception-frame dimension:
/// gravity(3), base angular velocity(3), q(25), dq(25), previous action(25).
pub const PROP_DIM: usize = 6 + 3 * G1_CONTROLLED_JOINTS;
/// Per-reference-frame dimension: local root displacement(3), relative root
/// quaternion(4), and target joint position(25).
pub const REFERENCE_DIM: usize = 7 + G1_CONTROLLED_JOINTS;
/// Actor observation dimension.
pub const ACTOR_OBS_DIM: usize = PROP_HISTORY * PROP_DIM + FUTURE_FRAMES * REFERENCE_DIM;
/// Privileged critic tail: root linear velocity(3), root position error(3),
/// root orientation error(1), joint position errors(25), joint velocity errors(25).
pub const CRITIC_TAIL_DIM: usize = 7 + 2 * G1_CONTROLLED_JOINTS;
/// Critic observation dimension.
pub const CRITIC_OBS_DIM: usize = ACTOR_OBS_DIM + CRITIC_TAIL_DIM;

/// MuJoCo actuator order used by BONES-SEED.
pub const BONES_G1_JOINT_NAMES: [&str; BONES_G1_JOINTS] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
];

/// Live joint order in zealot's solver-compatible G1 MJCF.
pub const G1_CONTROLLED_JOINT_NAMES: [&str; G1_CONTROLLED_JOINTS] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
];

/// Hard position limits copied from `assets/robots/unitree_g1_29dof.xml`.
pub const G1_CONTROLLED_LIMITS: [(f32, f32); G1_CONTROLLED_JOINTS] = [
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    (-2.5307, 2.8798),
    (-2.9671, 0.5236),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    (-2.618, 2.618),
    (-0.52, 0.52),
    (-0.52, 0.52),
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
];

/// One 50 Hz reference frame.
#[derive(Clone, Debug, PartialEq)]
pub struct MotionFrame {
    pub root_pos: [f32; 3],
    /// Root orientation `(x, y, z, w)`.
    pub root_quat: [f32; 4],
    pub root_lin_vel: [f32; 3],
    pub root_ang_vel: [f32; 3],
    pub joint_pos: [f32; G1_CONTROLLED_JOINTS],
    pub joint_vel: [f32; G1_CONTROLLED_JOINTS],
}

/// A validated and resampled motion.
#[derive(Clone, Debug, PartialEq)]
pub struct MotionClip {
    pub name: String,
    pub source: PathBuf,
    pub fps: f32,
    pub frames: Vec<MotionFrame>,
}

impl MotionClip {
    pub fn duration(&self) -> f32 {
        self.frames.len().saturating_sub(1) as f32 / self.fps
    }

    /// Sample with linear position/joint interpolation and quaternion slerp.
    /// Times outside the clip clamp to the first/last frame.
    pub fn sample(&self, time_s: f32) -> MotionFrame {
        assert!(!self.frames.is_empty(), "validated clips are non-empty");
        let x = (time_s.max(0.0) * self.fps).min((self.frames.len() - 1) as f32);
        let i0 = x.floor() as usize;
        let i1 = (i0 + 1).min(self.frames.len() - 1);
        interpolate_frame(&self.frames[i0], &self.frames[i1], x - i0 as f32)
    }
}

/// In-memory motion library. Loading is deterministic: paths are sorted before
/// applying `max_motions`.
#[derive(Clone, Debug, Default)]
pub struct MotionLibrary {
    pub clips: Vec<MotionClip>,
}

impl MotionLibrary {
    /// Load a raw BONES-SEED CSV or recursively load a directory of them.
    pub fn load(
        path: impl AsRef<Path>,
        max_motions: Option<usize>,
    ) -> Result<Self, MotionLoadError> {
        let path = path.as_ref();
        let mut paths = Vec::new();
        collect_csvs(path, &mut paths)?;
        paths.sort();
        if let Some(max) = max_motions {
            if max > 0 && paths.len() > max {
                // Sample the sorted corpus uniformly instead of taking its
                // alphabetic prefix. BONES-SEED is grouped by capture date, so
                // prefix truncation badly under-represents actors and skills.
                let total = paths.len();
                paths = (0..max)
                    .map(|i| paths[((2 * i + 1) * total / (2 * max)).min(total - 1)].clone())
                    .collect();
            } else {
                paths.truncate(max);
            }
        }
        if paths.is_empty() {
            return Err(MotionLoadError::new(path, "no CSV motion files found"));
        }
        let mut clips = Vec::with_capacity(paths.len());
        for csv in paths {
            clips.push(load_bones_csv(&csv)?);
        }
        Ok(Self { clips })
    }

    pub fn total_frames(&self) -> usize {
        self.clips.iter().map(|c| c.frames.len()).sum()
    }
}

/// Loader error carrying the offending source path.
#[derive(Debug)]
pub struct MotionLoadError {
    pub path: PathBuf,
    pub message: String,
}

impl MotionLoadError {
    fn new(path: impl AsRef<Path>, message: impl Into<String>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            message: message.into(),
        }
    }
}

impl fmt::Display for MotionLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.message)
    }
}

impl std::error::Error for MotionLoadError {}

/// Current simulated state needed by the task.
#[derive(Clone, Debug)]
pub struct MotionState {
    pub root_pos: [f32; 3],
    pub root_quat: [f32; 4],
    pub root_lin_vel: [f32; 3],
    pub root_ang_vel: [f32; 3],
    pub projected_gravity: [f32; 3],
    pub joint_pos: [f32; G1_CONTROLLED_JOINTS],
    pub joint_vel: [f32; G1_CONTROLLED_JOINTS],
    pub last_action: [f32; G1_CONTROLLED_JOINTS],
}

impl Default for MotionState {
    fn default() -> Self {
        Self {
            root_pos: [0.0, 0.0, 0.79],
            root_quat: [0.0, 0.0, 0.0, 1.0],
            root_lin_vel: [0.0; 3],
            root_ang_vel: [0.0; 3],
            projected_gravity: [0.0, 0.0, -1.0],
            joint_pos: [0.0; G1_CONTROLLED_JOINTS],
            joint_vel: [0.0; G1_CONTROLLED_JOINTS],
            last_action: [0.0; G1_CONTROLLED_JOINTS],
        }
    }
}

/// Current and future reference view for one policy step.
#[derive(Clone, Debug)]
pub struct MotionReference {
    pub now: MotionFrame,
    pub future: [MotionFrame; FUTURE_FRAMES],
}

impl MotionReference {
    pub fn at(clip: &MotionClip, time_s: f32) -> Self {
        Self {
            now: clip.sample(time_s),
            future: std::array::from_fn(|i| clip.sample(time_s + (i + 1) as f32 * FUTURE_DT)),
        }
    }
}

/// Scalar diagnostics for the core tracking objective.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrackingReward {
    pub root_pos: f32,
    pub root_ori: f32,
    pub root_lin_vel: f32,
    pub root_ang_vel: f32,
    pub joint_pos: f32,
    pub joint_vel: f32,
    pub action_rate: f32,
    pub joint_limit: f32,
    pub total: f32,
}

/// Dependency-free task logic shared by CPU tests and the nexus example.
#[derive(Clone, Debug)]
pub struct MotionTrackingTask {
    history: Vec<[f32; PROP_DIM]>,
}

impl Default for MotionTrackingTask {
    fn default() -> Self {
        Self {
            history: vec![[0.0; PROP_DIM]; PROP_HISTORY],
        }
    }
}

impl MotionTrackingTask {
    pub fn reset_history(&mut self, state: &MotionState) {
        let frame = proprio_frame(state);
        self.history.fill(frame);
    }

    pub fn push_state(&mut self, state: &MotionState) {
        self.history.rotate_left(1);
        self.history[PROP_HISTORY - 1] = proprio_frame(state);
    }

    pub fn actor_obs(&self, state: &MotionState, reference: &MotionReference) -> Vec<f32> {
        let mut out = Vec::with_capacity(ACTOR_OBS_DIM);
        for frame in &self.history {
            out.extend_from_slice(frame);
        }
        for target in &reference.future {
            let d_world = sub3(target.root_pos, state.root_pos);
            out.extend_from_slice(&quat_rotate_inv(state.root_quat, d_world));
            out.extend_from_slice(&quat_mul(quat_conj(state.root_quat), target.root_quat));
            out.extend_from_slice(&target.joint_pos);
        }
        debug_assert_eq!(out.len(), ACTOR_OBS_DIM);
        out
    }

    pub fn critic_obs(&self, state: &MotionState, reference: &MotionReference) -> Vec<f32> {
        let mut out = self.actor_obs(state, reference);
        out.extend_from_slice(&state.root_lin_vel);
        out.extend_from_slice(&sub3(reference.now.root_pos, state.root_pos));
        out.push(quat_angle(state.root_quat, reference.now.root_quat));
        for i in 0..G1_CONTROLLED_JOINTS {
            out.push(reference.now.joint_pos[i] - state.joint_pos[i]);
        }
        for i in 0..G1_CONTROLLED_JOINTS {
            out.push(reference.now.joint_vel[i] - state.joint_vel[i]);
        }
        debug_assert_eq!(out.len(), CRITIC_OBS_DIM);
        out
    }

    /// Default-offset joint targets with hard-limit clamping.
    pub fn joint_targets(
        &self,
        action: &[f32; G1_CONTROLLED_JOINTS],
    ) -> [f32; G1_CONTROLLED_JOINTS] {
        std::array::from_fn(|i| {
            action[i].clamp(G1_CONTROLLED_LIMITS[i].0, G1_CONTROLLED_LIMITS[i].1)
        })
    }

    pub fn reward(
        &self,
        state: &MotionState,
        reference: &MotionReference,
        previous_action: &[f32; G1_CONTROLLED_JOINTS],
    ) -> TrackingReward {
        let sq3 = |v: [f32; 3]| v.iter().map(|x| x * x).sum::<f32>();
        let root_pos_err = sq3(sub3(state.root_pos, reference.now.root_pos));
        let root_ori_err = quat_angle(state.root_quat, reference.now.root_quat).powi(2);
        let root_lin_err = sq3(sub3(state.root_lin_vel, reference.now.root_lin_vel));
        let root_ang_err = sq3(sub3(state.root_ang_vel, reference.now.root_ang_vel));
        let mut q_err = 0.0;
        let mut dq_err = 0.0;
        let mut da = 0.0;
        let mut limit = 0.0;
        for i in 0..G1_CONTROLLED_JOINTS {
            q_err += (state.joint_pos[i] - reference.now.joint_pos[i]).powi(2);
            dq_err += (state.joint_vel[i] - reference.now.joint_vel[i]).powi(2);
            da += (state.last_action[i] - previous_action[i]).powi(2);
            let (lo, hi) = G1_CONTROLLED_LIMITS[i];
            let margin = 0.05 * (hi - lo);
            limit += (lo + margin - state.joint_pos[i]).max(0.0).powi(2)
                + (state.joint_pos[i] - (hi - margin)).max(0.0).powi(2);
        }
        let mut r = TrackingReward {
            root_pos: (-20.0 * root_pos_err).exp(),
            root_ori: (-10.0 * root_ori_err).exp(),
            root_lin_vel: (-2.0 * root_lin_err).exp(),
            root_ang_vel: (-0.5 * root_ang_err).exp(),
            joint_pos: (-2.0 * q_err / G1_CONTROLLED_JOINTS as f32).exp(),
            joint_vel: (-0.05 * dq_err / G1_CONTROLLED_JOINTS as f32).exp(),
            action_rate: -0.01 * da,
            joint_limit: -5.0 * limit,
            total: 0.0,
        };
        r.total = 2.0 * r.root_pos
            + r.root_ori
            + 0.5 * r.root_lin_vel
            + 0.25 * r.root_ang_vel
            + 2.0 * r.joint_pos
            + 0.25 * r.joint_vel
            + r.action_rate
            + r.joint_limit;
        r
    }

    pub fn terminated(&self, state: &MotionState, reference: &MotionReference) -> bool {
        state.root_pos[2] < 0.35
            || quat_angle(state.root_quat, reference.now.root_quat) > 1.2
            || norm3(sub3(state.root_pos, reference.now.root_pos)) > 1.0
            || !state.root_pos.iter().all(|x| x.is_finite())
            || !state.joint_pos.iter().all(|x| x.is_finite())
    }
}

fn proprio_frame(state: &MotionState) -> [f32; PROP_DIM] {
    let mut out = [0.0; PROP_DIM];
    out[0..3].copy_from_slice(&state.projected_gravity);
    out[3..6].copy_from_slice(&state.root_ang_vel);
    out[6..31].copy_from_slice(&state.joint_pos);
    out[31..56].copy_from_slice(&state.joint_vel);
    out[56..81].copy_from_slice(&state.last_action);
    out
}

fn collect_csvs(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), MotionLoadError> {
    if path.is_file() {
        if path
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| x.eq_ignore_ascii_case("csv"))
        {
            out.push(path.to_path_buf());
            return Ok(());
        }
        return Err(MotionLoadError::new(path, "expected a .csv file"));
    }
    if !path.is_dir() {
        return Err(MotionLoadError::new(path, "path does not exist"));
    }
    let entries = fs::read_dir(path).map_err(|e| MotionLoadError::new(path, e.to_string()))?;
    for entry in entries {
        let entry = entry.map_err(|e| MotionLoadError::new(path, e.to_string()))?;
        let child = entry.path();
        if child.is_dir() {
            collect_csvs(&child, out)?;
        } else if child
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| x.eq_ignore_ascii_case("csv"))
        {
            out.push(child);
        }
    }
    Ok(())
}

fn load_bones_csv(path: &Path) -> Result<MotionClip, MotionLoadError> {
    let text = fs::read_to_string(path).map_err(|e| MotionLoadError::new(path, e.to_string()))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| MotionLoadError::new(path, "empty CSV"))?;
    let columns: Vec<&str> = header.split(',').map(str::trim).collect();
    let required_root = [
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
    ];
    let root_idx: Vec<usize> = required_root
        .iter()
        .map(|name| {
            columns
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| MotionLoadError::new(path, format!("missing column {name}")))
        })
        .collect::<Result<_, _>>()?;
    let joint_idx: Vec<usize> = BONES_G1_JOINT_NAMES
        .iter()
        .map(|name| {
            let expected = format!("{name}_dof");
            columns
                .iter()
                .position(|c| *c == expected)
                .ok_or_else(|| MotionLoadError::new(path, format!("missing column {expected}")))
        })
        .collect::<Result<_, _>>()?;

    let mut raw = Vec::new();
    for (row_no, line) in lines.enumerate() {
        let values: Vec<f32> = line
            .split(',')
            .map(|v| {
                v.trim().parse::<f32>().map_err(|_| {
                    MotionLoadError::new(path, format!("invalid number on row {}", row_no + 2))
                })
            })
            .collect::<Result<_, _>>()?;
        if values.len() != columns.len() {
            return Err(MotionLoadError::new(
                path,
                format!(
                    "row {} has {} columns, expected {}",
                    row_no + 2,
                    values.len(),
                    columns.len()
                ),
            ));
        }
        let root_pos = [
            values[root_idx[0]] / 100.0,
            values[root_idx[1]] / 100.0,
            values[root_idx[2]] / 100.0,
        ];
        let root_quat = quat_from_euler_xyz_deg([
            values[root_idx[3]],
            values[root_idx[4]],
            values[root_idx[5]],
        ]);
        let raw_q: [f32; BONES_G1_JOINTS] =
            std::array::from_fn(|i| values[joint_idx[i]].to_radians());
        let q: [f32; G1_CONTROLLED_JOINTS] = std::array::from_fn(|i| {
            let raw_i = BONES_G1_JOINT_NAMES
                .iter()
                .position(|n| n == &G1_CONTROLLED_JOINT_NAMES[i])
                .unwrap();
            raw_q[raw_i]
        });
        raw.push((root_pos, root_quat, q));
    }
    if raw.len() < 2 {
        return Err(MotionLoadError::new(
            path,
            "motion needs at least two frames",
        ));
    }

    // Raw BONES-SEED is 120 Hz. Sample it exactly onto the 50 Hz controller grid.
    let source_hz = 120.0;
    let duration = (raw.len() - 1) as f32 / source_hz;
    let output_len = (duration * CONTROL_HZ).floor() as usize + 1;
    let mut frames = Vec::with_capacity(output_len);
    for n in 0..output_len {
        let src = (n as f32 / CONTROL_HZ * source_hz).min((raw.len() - 1) as f32);
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(raw.len() - 1);
        let t = src - i0 as f32;
        frames.push(MotionFrame {
            root_pos: lerp3(raw[i0].0, raw[i1].0, t),
            root_quat: quat_slerp(raw[i0].1, raw[i1].1, t),
            root_lin_vel: [0.0; 3],
            root_ang_vel: [0.0; 3],
            joint_pos: std::array::from_fn(|j| raw[i0].2[j] + t * (raw[i1].2[j] - raw[i0].2[j])),
            joint_vel: [0.0; G1_CONTROLLED_JOINTS],
        });
    }
    // BONES clips live in capture-stage world coordinates and commonly begin
    // with an arbitrary heading. Each simulated episode starts at world XY=0
    // facing +X, so express the trajectory in that same episode-local frame.
    // Preserve source height and initial roll/pitch; only translation XY and
    // global yaw are gauge freedoms.
    let origin = frames[0].root_pos;
    let q0 = frames[0].root_quat;
    let initial_yaw =
        (2.0 * (q0[3] * q0[2] + q0[0] * q0[1])).atan2(1.0 - 2.0 * (q0[1] * q0[1] + q0[2] * q0[2]));
    let (sy, cy) = (0.5 * initial_yaw).sin_cos();
    let yaw = [0.0, 0.0, sy, cy];
    for frame in &mut frames {
        let local = quat_rotate_inv(
            yaw,
            [
                frame.root_pos[0] - origin[0],
                frame.root_pos[1] - origin[1],
                frame.root_pos[2],
            ],
        );
        frame.root_pos = local;
        frame.root_quat = quat_mul(quat_conj(yaw), frame.root_quat);
    }
    populate_velocities(&mut frames, 1.0 / CONTROL_HZ);
    let name = path
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("motion")
        .to_string();
    Ok(MotionClip {
        name,
        source: path.to_path_buf(),
        fps: CONTROL_HZ,
        frames,
    })
}

fn populate_velocities(frames: &mut [MotionFrame], dt: f32) {
    for i in 0..frames.len() {
        let a = i.saturating_sub(1);
        let b = (i + 1).min(frames.len() - 1);
        let span = ((b - a) as f32 * dt).max(dt);
        frames[i].root_lin_vel =
            std::array::from_fn(|k| (frames[b].root_pos[k] - frames[a].root_pos[k]) / span);
        frames[i].root_ang_vel =
            quat_delta_angular_velocity(frames[a].root_quat, frames[b].root_quat, span);
        frames[i].joint_vel =
            std::array::from_fn(|j| (frames[b].joint_pos[j] - frames[a].joint_pos[j]) / span);
    }
}

fn interpolate_frame(a: &MotionFrame, b: &MotionFrame, t: f32) -> MotionFrame {
    MotionFrame {
        root_pos: lerp3(a.root_pos, b.root_pos, t),
        root_quat: quat_slerp(a.root_quat, b.root_quat, t),
        root_lin_vel: lerp3(a.root_lin_vel, b.root_lin_vel, t),
        root_ang_vel: lerp3(a.root_ang_vel, b.root_ang_vel, t),
        joint_pos: std::array::from_fn(|i| a.joint_pos[i] + t * (b.joint_pos[i] - a.joint_pos[i])),
        joint_vel: std::array::from_fn(|i| a.joint_vel[i] + t * (b.joint_vel[i] - a.joint_vel[i])),
    }
}

fn quat_from_euler_xyz_deg(euler: [f32; 3]) -> [f32; 4] {
    let (rx, ry, rz) = (
        0.5 * euler[0].to_radians(),
        0.5 * euler[1].to_radians(),
        0.5 * euler[2].to_radians(),
    );
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();
    quat_normalize([
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
        cx * cy * cz - sx * sy * sz,
    ])
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    quat_normalize([
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ])
}

fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

fn quat_normalize(mut q: [f32; 4]) -> [f32; 4] {
    let n = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut q {
        *x /= n;
    }
    q
}

fn quat_slerp(a: [f32; 4], mut b: [f32; 4], t: f32) -> [f32; 4] {
    let mut dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    if dot < 0.0 {
        b = b.map(|x| -x);
        dot = -dot;
    }
    if dot > 0.9995 {
        return quat_normalize(std::array::from_fn(|i| a[i] + t * (b[i] - a[i])));
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let s = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / s;
    let wb = (t * theta).sin() / s;
    quat_normalize(std::array::from_fn(|i| wa * a[i] + wb * b[i]))
}

fn quat_angle(a: [f32; 4], b: [f32; 4]) -> f32 {
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>().abs();
    2.0 * dot.clamp(-1.0, 1.0).acos()
}

fn quat_delta_angular_velocity(a: [f32; 4], b: [f32; 4], dt: f32) -> [f32; 3] {
    let mut d = quat_mul(b, quat_conj(a));
    if d[3] < 0.0 {
        d = d.map(|x| -x);
    }
    let half = d[3].clamp(-1.0, 1.0).acos();
    let s = half.sin();
    if s.abs() < 1e-6 {
        return [0.0; 3];
    }
    let scale = 2.0 * half / (s * dt);
    [d[0] * scale, d[1] * scale, d[2] * scale]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|i| a[i] + t * (b[i] - a[i]))
}

fn norm3(v: [f32; 3]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn frame(x: f32) -> MotionFrame {
        MotionFrame {
            root_pos: [x, 0.0, 0.8],
            root_quat: [0.0, 0.0, 0.0, 1.0],
            root_lin_vel: [1.0, 0.0, 0.0],
            root_ang_vel: [0.0; 3],
            joint_pos: [x; G1_CONTROLLED_JOINTS],
            joint_vel: [1.0; G1_CONTROLLED_JOINTS],
        }
    }

    #[test]
    fn dimensions_are_stable() {
        assert_eq!(PROP_DIM, 81);
        assert_eq!(REFERENCE_DIM, 32);
        assert_eq!(ACTOR_OBS_DIM, 1130);
        assert_eq!(CRITIC_OBS_DIM, 1187);
    }

    #[test]
    fn future_reference_clamps_at_end() {
        let clip = MotionClip {
            name: "x".into(),
            source: "x.csv".into(),
            fps: 50.0,
            frames: vec![frame(0.0), frame(1.0)],
        };
        let r = MotionReference::at(&clip, 99.0);
        assert_eq!(r.now.root_pos[0], 1.0);
        assert!(r.future.iter().all(|f| f.root_pos[0] == 1.0));
    }

    #[test]
    fn perfect_tracking_beats_perturbed() {
        let reference = MotionReference {
            now: frame(0.0),
            future: std::array::from_fn(|_| frame(0.0)),
        };
        let mut state = MotionState::default();
        state.root_pos = reference.now.root_pos;
        state.root_quat = reference.now.root_quat;
        state.root_lin_vel = reference.now.root_lin_vel;
        state.joint_pos = reference.now.joint_pos;
        state.joint_vel = reference.now.joint_vel;
        let task = MotionTrackingTask::default();
        let perfect = task.reward(&state, &reference, &[0.0; G1_CONTROLLED_JOINTS]);
        state.root_pos[0] += 0.5;
        state.joint_pos[0] += 0.5;
        let bad = task.reward(&state, &reference, &[0.0; G1_CONTROLLED_JOINTS]);
        assert!(perfect.total > bad.total);
    }

    #[test]
    fn raw_bones_csv_maps_and_resamples() {
        let dir = std::env::temp_dir().join(format!("zealot-sonic-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("clip.csv");
        let mut file = fs::File::create(&path).unwrap();
        let mut header = vec![
            "Frame".to_string(),
            "root_translateX".into(),
            "root_translateY".into(),
            "root_translateZ".into(),
            "root_rotateX".into(),
            "root_rotateY".into(),
            "root_rotateZ".into(),
        ];
        header.extend(BONES_G1_JOINT_NAMES.iter().map(|n| format!("{n}_dof")));
        writeln!(file, "{}", header.join(",")).unwrap();
        for row in 0..121 {
            let mut values = vec![
                row.to_string(),
                row.to_string(),
                "0".into(),
                "80".into(),
                "0".into(),
                "0".into(),
                "0".into(),
            ];
            values.extend((0..BONES_G1_JOINTS).map(|j| (j as f32 + row as f32).to_string()));
            writeln!(file, "{}", values.join(",")).unwrap();
        }
        let clip = load_bones_csv(&path).unwrap();
        assert_eq!(clip.frames.len(), 51);
        assert!((clip.duration() - 1.0).abs() < 1e-6);
        assert_eq!(clip.frames[0].root_pos[0], 0.0);
        assert_eq!(clip.frames[0].root_pos[1], 0.0);
        assert!((clip.frames[50].root_pos[0] - 1.2).abs() < 1e-6);
        assert!((clip.frames[0].root_pos[2] - 0.8).abs() < 1e-6);
        assert!((clip.frames[0].joint_pos[20] - 22.0f32.to_radians()).abs() < 1e-6);
        let _ = fs::remove_file(path);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn observation_lengths_match_contract() {
        let clip = MotionClip {
            name: "x".into(),
            source: "x.csv".into(),
            fps: 50.0,
            frames: vec![frame(0.0), frame(0.0)],
        };
        let state = MotionState::default();
        let mut task = MotionTrackingTask::default();
        task.reset_history(&state);
        let reference = MotionReference::at(&clip, 0.0);
        assert_eq!(task.actor_obs(&state, &reference).len(), ACTOR_OBS_DIM);
        assert_eq!(task.critic_obs(&state, &reference).len(), CRITIC_OBS_DIM);
    }
}
