//! Retargeted mocap clips (AMASS / SONIC → G1 joint space), used as an
//! upper-body DISTURBANCE source for the balance curriculum: while the
//! command is "stand", the held (non-action) joints replay a random window
//! of a random clip, so the legs learn to keep the pelvis level under
//! realistic arm + waist motion instead of a frozen home pose. The policy
//! never observes the upper body — the moving mass is an unmodeled
//! perturbation, exactly like a human carrying/gesturing.
//!
//! File format: the SONIC CSV export. Header row
//! `Frame,root_translateX..Z,root_rotateX..Z,<joint>_dof,...`; angles are
//! DEGREES (root translation is centimetres — both ignored here). Columns
//! are matched to the caller's joint names as `<name>_dof`, so the same
//! file drives any subset of the model's joints: legs are simply never
//! requested, and clip columns the model doesn't have (the 29dof-agile G1
//! drops wrist pitch/yaw) are ignored. A REQUESTED joint missing from the
//! file is a hard error — silent zero-filling would quietly turn a dance
//! clip into a statue.

/// One clip: `frames[t][j]` in radians, `j` indexed in the caller's
/// requested-joint order (NOT csv column order).
pub struct MotionClip {
    pub name: String,
    pub fps: f32,
    n_joints: usize,
    /// Flattened `[t * n_joints + j]`, radians.
    frames: Vec<f32>,
}

impl MotionClip {
    /// Parse one CSV given the requested joint names (e.g. the env's held
    /// joints, in staging order).
    pub fn parse_csv(name: &str, text: &str, joints: &[String], fps: f32) -> Result<Self, String> {
        let mut lines = text.lines();
        let header = lines.next().ok_or_else(|| format!("{name}: empty file"))?;
        let cols: Vec<&str> = header.split(',').map(str::trim).collect();
        // Column index for each requested joint (`<name>_dof`).
        let mut col_of = Vec::with_capacity(joints.len());
        for j in joints {
            let want = format!("{j}_dof");
            let c = cols
                .iter()
                .position(|c| *c == want)
                .ok_or_else(|| format!("{name}: requested joint column {want} not in header"))?;
            col_of.push(c);
        }
        let deg2rad = std::f32::consts::PI / 180.0;
        let mut frames = Vec::new();
        for (ln, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let vals: Vec<&str> = line.split(',').collect();
            for &c in &col_of {
                let v: f32 = vals
                    .get(c)
                    .ok_or_else(|| format!("{name}: row {} truncated", ln + 2))?
                    .trim()
                    .parse()
                    .map_err(|_| format!("{name}: row {} col {} unparsable", ln + 2, c))?;
                frames.push(v * deg2rad);
            }
        }
        if frames.is_empty() {
            return Err(format!("{name}: no data rows"));
        }
        Ok(Self {
            name: name.to_string(),
            fps,
            n_joints: joints.len(),
            frames,
        })
    }

    /// Load every `*.csv` in `dir`. Errors on an unreadable dir, an unparsable
    /// file, or zero clips — the feature was explicitly requested, so an empty
    /// dataset must fail loudly, not train silently without it.
    pub fn load_dir(
        dir: &std::path::Path,
        joints: &[String],
        fps: f32,
    ) -> Result<Vec<Self>, String> {
        let mut clips = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "csv"))
            .collect();
        entries.sort(); // deterministic clip indices across runs
        for p in entries {
            let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let name = p.file_stem().unwrap_or_default().to_string_lossy().into_owned();
            clips.push(Self::parse_csv(&name, &text, joints, fps)?);
        }
        if clips.is_empty() {
            return Err(format!("{}: no .csv clips found", dir.display()));
        }
        Ok(clips)
    }

    pub fn num_frames(&self) -> usize {
        self.frames.len() / self.n_joints
    }

    /// Clip length in seconds.
    pub fn duration(&self) -> f32 {
        (self.num_frames().saturating_sub(1)) as f32 / self.fps
    }

    /// Linearly interpolated pose at time `t` seconds, clamped to the clip
    /// (t < 0 → first frame, t > duration → last frame — a finished clip
    /// freezes rather than wrapping, since a wrap is a teleport-sized
    /// discontinuity in the middle of a stand). Writes `n_joints` radians
    /// into `out`.
    pub fn sample(&self, t: f32, out: &mut [f32]) {
        assert_eq!(out.len(), self.n_joints);
        let last = self.num_frames() - 1;
        let ft = (t * self.fps).clamp(0.0, last as f32);
        let f0 = ft as usize;
        let f1 = (f0 + 1).min(last);
        let a = ft - f0 as f32;
        for j in 0..self.n_joints {
            let v0 = self.frames[f0 * self.n_joints + j];
            let v1 = self.frames[f1 * self.n_joints + j];
            out[j] = v0 + a * (v1 - v0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CSV: &str = "\
Frame,root_translateX,root_translateY,root_translateZ,waist_yaw_joint_dof,left_elbow_joint_dof,left_wrist_pitch_joint_dof
0,0,0,73.0,0.0,90.0,5.0
1,0,0,73.0,45.0,45.0,5.0
2,0,0,73.0,90.0,0.0,5.0
";

    fn joints() -> Vec<String> {
        vec!["waist_yaw_joint".into(), "left_elbow_joint".into()]
    }

    #[test]
    fn parses_by_name_in_request_order_and_converts_to_radians() {
        // Request order is reversed vs csv column order on purpose: output
        // must follow the REQUEST, and unrequested columns (root, wrist
        // pitch) must not leak in.
        let j = vec!["left_elbow_joint".to_string(), "waist_yaw_joint".to_string()];
        let c = MotionClip::parse_csv("t", CSV, &j, 30.0).unwrap();
        let mut out = [0.0f32; 2];
        c.sample(0.0, &mut out);
        assert!((out[0] - 90f32.to_radians()).abs() < 1e-6); // elbow first
        assert!(out[1].abs() < 1e-6); // waist yaw
    }

    #[test]
    fn missing_requested_joint_is_a_hard_error() {
        let j = vec!["right_elbow_joint".to_string()];
        assert!(MotionClip::parse_csv("t", CSV, &j, 30.0).is_err());
    }

    #[test]
    fn sample_interpolates_and_clamps_at_both_ends() {
        let c = MotionClip::parse_csv("t", CSV, &joints(), 30.0).unwrap();
        assert_eq!(c.num_frames(), 3);
        assert!((c.duration() - 2.0 / 30.0).abs() < 1e-6);
        let mut out = [0.0f32; 2];
        // Halfway between frames 0 and 1.
        c.sample(0.5 / 30.0, &mut out);
        assert!((out[0] - 22.5f32.to_radians()).abs() < 1e-5);
        assert!((out[1] - 67.5f32.to_radians()).abs() < 1e-5);
        // Before the start → frame 0; far past the end → frozen last frame.
        c.sample(-1.0, &mut out);
        assert!(out[0].abs() < 1e-6);
        c.sample(999.0, &mut out);
        assert!((out[0] - 90f32.to_radians()).abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
    }
}
