//! Drive a trained policy with a CONSTANT velocity command and report where
//! the robot actually goes. Loads a checkpoint, builds ONE nexus GPU env with
//! DR off, pins the command, rolls out deterministically (mean action), and
//! prints the base trajectory: per-second position/heading/velocity, a final
//! commanded-vs-achieved summary, and a top-down ASCII map of the path. Also
//! writes a CSV (t,x,y,z,yaw) for plotting and a rollout JSON in the
//! `render_biped.py` format, so the run can be rendered to video:
//!   `python3 examples/biped/render_biped.py /tmp/biped_drive.json /tmp/biped_drive.mp4`
//!
//! Run:
//!   `cargo run --release --example biped_drive --features biped_gpu -- \
//!         [vx] [vy] [yaw_rate] [seconds] [policy.safetensors] [out.csv] [out.json]`
//! e.g. forward at 0.3 m/s for 10 s with the latest Mac policy:
//!   `cargo run --release --example biped_drive --features biped_gpu -- 0.3`
//! turn in place:
//!   `cargo run --release --example biped_drive --features biped_gpu -- 0 0 0.5`

#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;

use biped_env_nexus::{BipedNexusBatchEnv, default_mjcf_path};
use std::fmt::Write as _;
use zealot_env::robots::lerobot_bipedal::{JOINT_NAMES, NUM_JOINTS};
use zealot_rl::ActorCritic;

/// Control-step period — matches `VelocityFlatTask` (50 Hz).
const DT: f32 = 0.02;

fn arg_f32(i: usize, default: f32) -> f32 {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Heading (rad) about +Z from an xyzw quaternion.
fn quat_yaw(q: [f32; 4]) -> f32 {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z))
}

fn wrap_angle(a: f32) -> f32 {
    let mut a = a % std::f32::consts::TAU;
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    } else if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

/// Top-down ASCII map of the XY path: S = start, E = end, * = path.
/// X axis (initial forward) points right, Y points up.
fn ascii_map(xy: &[(f32, f32)]) -> String {
    const W: usize = 61;
    const H: usize = 21;
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for &(x, y) in xy {
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    // Pad so a near-degenerate path (standing still) still renders.
    let pad = 0.05f32.max((xmax - xmin).max(ymax - ymin) * 0.05);
    xmin -= pad;
    xmax += pad;
    ymin -= pad;
    ymax += pad;
    let mut grid = vec![vec![' '; W]; H];
    let to_cell = |x: f32, y: f32| {
        let c = ((x - xmin) / (xmax - xmin) * (W - 1) as f32).round() as usize;
        let r = ((ymax - y) / (ymax - ymin) * (H - 1) as f32).round() as usize;
        (r.min(H - 1), c.min(W - 1))
    };
    for &(x, y) in xy {
        let (r, c) = to_cell(x, y);
        grid[r][c] = '*';
    }
    let (r, c) = to_cell(xy[0].0, xy[0].1);
    grid[r][c] = 'S';
    let last = xy[xy.len() - 1];
    let (r, c) = to_cell(last.0, last.1);
    grid[r][c] = 'E';
    let mut s = String::new();
    let _ = writeln!(
        s,
        "top-down path (x → right {:.2}..{:.2} m, y ↑ {:.2}..{:.2} m):",
        xmin, xmax, ymin, ymax
    );
    for row in grid {
        let _ = writeln!(s, "|{}|", row.iter().collect::<String>());
    }
    s
}

fn main() {
    let vx = arg_f32(1, 0.3);
    let vy = arg_f32(2, 0.0);
    let yaw_rate = arg_f32(3, 0.0);
    let seconds = arg_f32(4, 10.0);
    let policy_path = std::env::args()
        .nth(5)
        .unwrap_or_else(|| "walking_policy_mac_v4.safetensors".to_string());
    let csv_path = std::env::args()
        .nth(6)
        .unwrap_or_else(|| "/tmp/biped_drive_traj.csv".to_string());
    let json_path = std::env::args()
        .nth(7)
        .unwrap_or_else(|| "/tmp/biped_drive.json".to_string());
    let steps = (seconds / DT).round() as usize;

    let ac = ActorCritic::load(&policy_path).expect("load policy checkpoint");
    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");

    println!(
        "policy: {policy_path}\ncommand: vx={vx} m/s  vy={vy} m/s  yaw_rate={yaw_rate} rad/s  for {seconds} s ({steps} steps)"
    );

    pollster::block_on(async {
        // One env, one template — template 0 is the DR-off default scene, and
        // `reset_env_to_default_template` resets into exactly that.
        // BIPED_DRIVE_SEED varies the env RNG (push timing/direction) so
        // robustness sweeps can average over several perturbation sequences.
        let seed = std::env::var("BIPED_DRIVE_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0xC0FFEE);
        let mut env = BipedNexusBatchEnv::new(&xml, 1, 1, seed).await;
        let (mut obs, _) = {
            let o = env.reset_env_to_default_template(0).await;
            (o.0, o.1)
        };
        env.pin_command_for(0, vx, vy, yaw_rate);

        // Trajectory: (t, x, y, z, yaw) each control step, before acting.
        let mut traj: Vec<(f32, f32, f32, f32, f32)> = Vec::with_capacity(steps + 1);
        let mut falls: Vec<f32> = Vec::new();
        // Per-step world positions of every body, for the video renderer.
        let mut frames: Vec<Vec<[f32; 3]>> = Vec::with_capacity(steps + 1);
        let mut resets: Vec<usize> = Vec::new();
        // Base pose (pos + quat xyzw) and joint angles per step — the MuJoCo
        // mesh renderer replays these as qpos.
        let mut bases: Vec<[f32; 7]> = Vec::with_capacity(steps + 1);
        let mut joint_rec: Vec<[f32; NUM_JOINTS]> = Vec::with_capacity(steps + 1);
        // Mean achieved body-frame velocity, averaged over steps that have a
        // valid previous pose (skips the step right after a reset).
        let (mut sum_vxb, mut sum_vyb, mut sum_wz, mut vel_samples) =
            (0.0f64, 0.0f64, 0.0f64, 0u32);
        let mut prev: Option<(f32, f32, f32)> = None; // (x, y, yaw)

        for step in 0..=steps {
            let t = step as f32 * DT;
            let poses = env.snapshot().await;
            let (p, q) = env.base_pose_for(0, &poses);
            let yaw = quat_yaw(q);
            traj.push((t, p[0], p[1], p[2], yaw));
            frames.push(env.body_positions_for(0, &poses));
            bases.push([p[0], p[1], p[2], q[0], q[1], q[2], q[3]]);
            joint_rec.push(env.joint_angles_for(0, &poses));

            if let Some((px, py, pyaw)) = prev {
                let (dx, dy) = (p[0] - px, p[1] - py);
                // World → body frame via the current heading.
                let (s, c) = yaw.sin_cos();
                sum_vxb += ((c * dx + s * dy) / DT) as f64;
                sum_vyb += ((-s * dx + c * dy) / DT) as f64;
                sum_wz += (wrap_angle(yaw - pyaw) / DT) as f64;
                vel_samples += 1;
            }
            prev = Some((p[0], p[1], yaw));

            if step % 50 == 0 {
                println!(
                    "t={t:>5.1}s  pos=({:>6.2}, {:>6.2})  z={:.3}  heading={:>6.1}°",
                    p[0],
                    p[1],
                    p[2],
                    yaw.to_degrees()
                );
            }
            if step == steps {
                break;
            }

            // Deterministic mean action for env 0 (single env → CPU forward).
            let mut a = [0.0f32; NUM_JOINTS];
            a.copy_from_slice(&ac.mean(&obs)[..NUM_JOINTS]);
            let outs = env.step(&[a]).await;
            if outs[0].done {
                if outs[0].fell {
                    println!("  FELL at t={:.2}s — resetting", t + DT);
                    falls.push(t + DT);
                }
                resets.push(step);
                obs = env.reset_env_to_default_template(0).await.0;
                env.pin_command_for(0, vx, vy, yaw_rate);
                prev = None; // don't difference across the teleport
            } else {
                obs.clone_from(&outs[0].obs);
            }
        }

        // Summary: commanded vs achieved.
        let n = vel_samples.max(1) as f64;
        let (avx, avy, awz) = (sum_vxb / n, sum_vyb / n, sum_wz / n);
        let first = traj[0];
        let last = traj[traj.len() - 1];
        let dist = ((last.1 - first.1).powi(2) + (last.2 - first.2).powi(2)).sqrt();
        println!("\n=== summary ===");
        println!(
            "displacement: ({:+.2}, {:+.2}) m  |net| = {:.2} m in {:.1} s",
            last.1 - first.1,
            last.2 - first.2,
            dist,
            last.0
        );
        println!(
            "achieved body-frame velocity (mean): vx={avx:+.3}  vy={avy:+.3} m/s  yaw_rate={awz:+.3} rad/s"
        );
        println!(
            "commanded:                           vx={vx:+.3}  vy={vy:+.3} m/s  yaw_rate={yaw_rate:+.3} rad/s"
        );
        println!(
            "heading drift: {:+.1}°   falls: {}{}",
            wrap_angle(last.4 - first.4).to_degrees(),
            falls.len(),
            if falls.is_empty() {
                String::new()
            } else {
                format!(" (at {:?} s)", falls)
            }
        );

        let xy: Vec<(f32, f32)> = traj.iter().map(|f| (f.1, f.2)).collect();
        println!("\n{}", ascii_map(&xy));

        let mut csv = String::from("t,x,y,z,yaw\n");
        for (t, x, y, z, yaw) in &traj {
            let _ = writeln!(csv, "{t:.3},{x:.5},{y:.5},{z:.5},{yaw:.5}");
        }
        std::fs::write(&csv_path, csv).expect("write csv");
        println!("trajectory csv → {csv_path}");

        // Rollout JSON in the `render_biped.py` format (same hand-rolled
        // writer as biped_render_nexus — no serde dep).
        let (names, edges, feet) = env.skeleton();
        let mut s = String::from("{\n");
        let _ = write!(s, "  \"dt\": {DT:.4},\n");
        let names_json: Vec<String> = names.iter().map(|n| format!("\"{n}\"")).collect();
        let _ = write!(s, "  \"names\": [{}],\n", names_json.join(", "));
        let edges_json: Vec<String> = edges.iter().map(|(a, b)| format!("[{a},{b}]")).collect();
        let _ = write!(s, "  \"edges\": [{}],\n", edges_json.join(", "));
        let feet_json: Vec<String> = feet.iter().map(|i| i.to_string()).collect();
        let _ = write!(s, "  \"feet\": [{}],\n", feet_json.join(", "));
        let resets_json: Vec<String> = resets.iter().map(|i| i.to_string()).collect();
        let _ = write!(s, "  \"resets\": [{}],\n", resets_json.join(", "));
        let jn: Vec<String> = JOINT_NAMES.iter().map(|n| format!("\"{n}\"")).collect();
        let _ = write!(s, "  \"joint_names\": [{}],\n", jn.join(", "));
        let base_json: Vec<String> = bases
            .iter()
            .map(|b| {
                format!(
                    "[{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}]",
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6]
                )
            })
            .collect();
        let _ = write!(s, "  \"base\": [{}],\n", base_json.join(","));
        let joints_json: Vec<String> = joint_rec
            .iter()
            .map(|j| {
                let v: Vec<String> = j.iter().map(|a| format!("{a:.5}")).collect();
                format!("[{}]", v.join(","))
            })
            .collect();
        let _ = write!(s, "  \"joints\": [{}],\n", joints_json.join(","));
        s.push_str("  \"frames\": [\n");
        for (fi, frame) in frames.iter().enumerate() {
            let pts: Vec<String> = frame
                .iter()
                .map(|p| format!("[{:.4},{:.4},{:.4}]", p[0], p[1], p[2]))
                .collect();
            let comma = if fi + 1 < frames.len() { "," } else { "" };
            let _ = writeln!(s, "    [{}]{}", pts.join(","), comma);
        }
        s.push_str("  ]\n}\n");
        std::fs::write(&json_path, s).expect("write rollout json");
        println!(
            "rollout json → {json_path}\nrender: python3 examples/biped/render_biped.py {json_path} /tmp/biped_drive.mp4"
        );
    });
}
