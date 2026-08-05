//! Shared implementation of the website G1 demos: the Unitree G1 driven by
//! the RELEASED walking policy (`g1_v7_iter15k` from the
//! `checkpoints-2026-07-27-fixed-physics` GitHub release) on the ZEALOT
//! training environment — the exact nexus GPU env the trainer uses
//! (`biped_env_nexus.rs`, `BIPED_ROBOT=g1_29dof_agile`: policy legs + AGILE
//! holding gains for waist/arms, real Unitree PD gains), rendered with
//! kiss3d. MJCF + meshes + checkpoint are embedded, so the same executable
//! runs natively (dev) and as the browser wasm module.
//!
//! Two thin examples pick a [`DemoCfg`]:
//! - `g1_web`         — a 10-robot fleet on flat ground;
//! - `g1_terrain_web` — a single robot on the rough-terrain strips
//!   (v7 trained on them; trimesh contacts are the expensive part, which is
//!   why the fleet demo stays flat).
//!
//! `configure_env` mirrors the v7 launch script's DEPLOYMENT-relevant knobs
//! (obs layout: history 5 + force-sensed contact; physics: decimation 4 +
//! substep refresh, 4 iters, contact 240 Hz/ζ=1, corrective-velocity clamp
//! 0.2). The UI pins velocity commands (stand / walk / turn); the policy runs
//! deterministically (`ac.mean`) at 50 Hz.
//!
//! NOTE (native macOS): the unified-branch kernels currently mis-simulate
//! multibody contacts under naga's MSL backend (the robot falls no matter
//! what) — judge physics in Chrome (Tint) or on CUDA, not on Metal.

#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;

/// Which flavour of the demo to run.
pub struct DemoCfg {
    /// Number of robots (independent envs in one batched sim).
    pub n_robots: usize,
    /// Rough-terrain strips (BIPED_TERRAIN=1) vs flat ground.
    pub terrain: bool,
    /// Terrain spawn difficulty, 0 (flat-ish) ..= 19 (hardest patch).
    pub terrain_level: u32,
    /// Terrain amplitude multiplier in percent (100 = training terrain).
    pub terrain_amp_pct: u32,
    /// Uphill slope along the strip in degrees (0 = training terrain).
    pub terrain_slope_deg: u32,
    /// Where to get the policy from, if not the embedded default. Accepts a
    /// full URL, a Hugging Face `owner/repo/file.safetensors` (the form the
    /// website's model picker passes), or `owner/repo` — see [`ckpt_url`].
    pub ckpt: Option<String>,
}

use biped_env_nexus::{BipedNexusBatchEnv, parse_mjcf};
use nexus3d::rbd::math::Pose as NexusPose;
use glamx::{Pose3, Rot3, Vec3};
use kiss3d::camera::OrbitCamera3d;
use kiss3d::color::Color;
use kiss3d::scene::SceneNode3d;
use kiss3d::event::{Action as KAction, MouseButton as KMouseButton, WindowEvent as KWindowEvent};
use kiss3d::window::Window;
use zealot_env::robots::NUM_JOINTS;
use zealot_env::terrain::{TerrainFamily, TerrainParams, TerrainStrip};
use zealot_rl::ActorCritic;
// Trait methods (`begin_encoding`/`submit`) for the resident GPU loop.
use khal::backend::Backend as _;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

/// The 29-DOF G1 model (in-repo, primitive colliders — fully self-contained).
const MJCF_XML: &str = include_str!("../../assets/robots/unitree_g1_29dof.xml");

/// Baked G1 visual meshes in the converted link frames (see
/// `tools/bake_g1_visuals.py`): per kept body, decimated menagerie meshes.
const VISUALS_BIN: &[u8] = include_bytes!("assets/g1_visuals_29dof.bin");

/// The v26 walking checkpoint (iter 42290): ActorCritic weights + Welford
/// normalizer stats. 48-dim frames (gyro era), trained with the widened yaw
/// command range and the yaw-inclusive gait clock — markedly stronger
/// turning than v24.
const POLICY_BIN: &[u8] = include_bytes!("assets/g1_walk_v26.safetensors");

/// What the HUD calls the embedded checkpoint (see `POLICY_BIN`).
const DEFAULT_CKPT: &str = "g1_walk_v26 (embedded)";

/// Surface a demo-level problem where it can actually be seen: the browser
/// console, or stderr natively.
fn log_warn(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::warn_1(&format!("zealot demo: {msg}").into());
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("zealot demo: {msg}");
}

/// Parse `g1_visuals.bin` → body name → mesh groups (rgba, verts, tris).
#[allow(clippy::type_complexity)]
fn load_visuals() -> std::collections::HashMap<String, Vec<([f32; 4], Vec<Vec3>, Vec<[u32; 3]>)>> {
    let mut map: std::collections::HashMap<String, Vec<_>> = std::collections::HashMap::new();
    let d = VISUALS_BIN;
    let mut o = 0usize;
    let rd_u32 = |o: &mut usize| {
        let v = u32::from_le_bytes(d[*o..*o + 4].try_into().unwrap());
        *o += 4;
        v
    };
    let rd_f32 = |o: &mut usize| {
        let v = f32::from_le_bytes(d[*o..*o + 4].try_into().unwrap());
        *o += 4;
        v
    };
    let n = rd_u32(&mut o);
    for _ in 0..n {
        let l = d[o] as usize;
        o += 1;
        let name = std::str::from_utf8(&d[o..o + l]).unwrap().to_string();
        o += l;
        let rgba = [rd_f32(&mut o), rd_f32(&mut o), rd_f32(&mut o), rd_f32(&mut o)];
        let nv = rd_u32(&mut o) as usize;
        let nf = rd_u32(&mut o) as usize;
        let mut verts = Vec::with_capacity(nv);
        for _ in 0..nv {
            verts.push(Vec3::new(rd_f32(&mut o), rd_f32(&mut o), rd_f32(&mut o)));
        }
        let mut tris = Vec::with_capacity(nf);
        for _ in 0..nf {
            tris.push([rd_u32(&mut o), rd_u32(&mut o), rd_u32(&mut o)]);
        }
        map.entry(name).or_default().push((rgba, verts, tris));
    }
    map
}

/// Control-step period — matches `VelocityFlatTask` (50 Hz).
const DT: f32 = 0.02;

/// Live driving: each arrow / WASD press BUMPS the policy's velocity command
/// by a fixed step and the command latches, so you steer with taps instead of
/// holding a key down. Space (or Stand) zeroes it. A gamepad stick overrides
/// while deflected. Browser-only — natively the demo keeps its presets.
#[cfg(target_arch = "wasm32")]
mod drive {
    use std::cell::RefCell;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    /// Per-press increment, and the range each axis is held within.
    const STEP: f32 = 0.2;
    const VX: (f32, f32) = (-0.6, 1.0);
    const VY: (f32, f32) = (-0.4, 0.4);
    const YAW: (f32, f32) = (-1.0, 1.0);

    /// The latched command. `None` until the user first drives, so the demo's
    /// presets/sliders stay in charge before that.
    thread_local! {
        static CMD: RefCell<Option<[f32; 3]>> = const { RefCell::new(None) };
    }

    /// Adopt a command set elsewhere (a preset button or slider), so the next
    /// key press bumps from what the user currently sees.
    pub fn sync(cmd: [f32; 3]) {
        CMD.with(|c| *c.borrow_mut() = Some(cmd));
    }

    fn bump(axis: usize, delta: f32) {
        CMD.with(|c| {
            let mut c = c.borrow_mut();
            let mut cmd = c.unwrap_or([0.0; 3]);
            cmd[axis] += delta;
            let (lo, hi) = [VX, VY, YAW][axis];
            cmd[axis] = cmd[axis].clamp(lo, hi);
            *c = Some(cmd);
        });
    }

    /// Attach the key listeners once, at startup.
    pub fn install() {
        let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
            return;
        };
        let cb = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(
            move |e: web_sys::KeyboardEvent| {
                // Auto-repeat would ramp the command while a key is merely
                // held; one press should be one step.
                if e.repeat() {
                    return;
                }
                match e.code().as_str() {
                    "ArrowUp" | "KeyW" => bump(0, STEP),
                    "ArrowDown" | "KeyS" => bump(0, -STEP),
                    "KeyA" => bump(1, STEP),
                    "KeyD" => bump(1, -STEP),
                    "ArrowLeft" | "KeyQ" => bump(2, STEP),
                    "ArrowRight" | "KeyE" => bump(2, -STEP),
                    "Space" => CMD.with(|c| *c.borrow_mut() = Some([0.0; 3])),
                    _ => return,
                }
                // Arrows/space would otherwise scroll the embedding page.
                e.prevent_default();
            },
        );
        let _ = doc.add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref());
        cb.forget(); // lives for the page's lifetime
    }

    /// The command to drive with, or `None` while the user has never touched
    /// the controls (demo presets stay in charge until then).
    pub fn command() -> Option<[f32; 3]> {
        // A deflected gamepad stick overrides the latched value; sticks rest
        // near but not exactly at zero, hence the deadzone.
        if let Some(pads) = web_sys::window().and_then(|w| w.navigator().get_gamepads().ok()) {
            for pad in pads.iter().filter_map(|p| p.dyn_into::<web_sys::Gamepad>().ok()) {
                let axes = pad.axes();
                let axis = |i: u32| axes.get(i).as_f64().unwrap_or(0.0) as f32;
                let dead = |v: f32| if v.abs() < 0.15 { 0.0 } else { v };
                let (lx, ly, rx) = (dead(axis(0)), dead(axis(1)), dead(axis(2)));
                if lx != 0.0 || ly != 0.0 || rx != 0.0 {
                    // Stick up reads negative on the standard mapping.
                    let vx = if -ly > 0.0 { -ly * VX.1 } else { -ly * -VX.0 };
                    return Some([
                        vx.clamp(VX.0, VX.1),
                        (-lx * VY.1).clamp(VY.0, VY.1),
                        (-rx * YAW.1).clamp(YAW.0, YAW.1),
                    ]);
                }
            }
        }
        CMD.with(|c| *c.borrow())
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod drive {
    pub fn install() {}
    pub fn sync(_cmd: [f32; 3]) {}
    pub fn command() -> Option<[f32; 3]> {
        None
    }
}

/// Env knobs the demo needs regardless of target. `std::env::set_var` PANICS
/// on wasm32-unknown-unknown, so these go through programmatic overrides. Must
/// run BEFORE the env is constructed. Decimation 8 → 2.5 ms sim dt — the
/// passive stand needs the finer timestep. (Spawn DR is a non-issue: the demo
/// resets into template 0, which is always the DR-off template.)
fn probe_u32(key: &str, default: u32) -> u32 {
    #[cfg(target_arch = "wasm32")]
    if let Some(search) = web_sys::window().and_then(|w| w.location().search().ok()) {
        for kv in search.trim_start_matches('?').split('&') {
            if let Some(v) = kv.strip_prefix(key) {
                if let Ok(n) = v.parse::<u32>() {
                    return n;
                }
            }
        }
    }
    let _ = key;
    default
}

fn configure_env(terrain: bool, terrain_level: u32, terrain_amp_pct: u32, terrain_slope_deg: u32) {
    zealot_env::robots::set_robot_override("g1_29dof_agile");
    // Deployment-relevant slice of the v7 launch config (launch_v7.sh from the
    // checkpoint release). Physics: dt = 5 ms (decimation 4) + per-substep
    // joint-constraint refresh, 4 solver iters, contact ERP 240 Hz/ζ=1, max
    // corrective velocity 0.2 — the exact repaired-solver config v7 trained
    // on. Obs layout: 5-frame history + force-sensed foot contact + the v7
    // gait-clock swing ratio. Training-only knobs (DR, pushes, mirror aug,
    // LR/grad, terrain curriculum) are intentionally NOT replicated.
    let _ = biped_env_nexus::DECIMATION_OVERRIDE.set(4);
    let _ = zealot_env::obs_history::OBS_HISTORY_OVERRIDE.set(5);
    let mut overrides: std::collections::HashMap<&'static str, &'static str> = [
        ("NEXUS_SUBSTEP_REFRESH", "1"),
        ("BIPED_SOLVER_ITERS", "4"),
        ("BIPED_CONTACT_NF", "240"),
        ("BIPED_CONTACT_DR", "1"),
        ("BIPED_MAX_CORR_VEL", "0.2"),
        ("BIPED_OBS_HISTORY", "5"),
        ("BIPED_CONTACT_SENSE", "1"),
        ("BIPED_GAIT_SWING_RATIO", "0.5"),
        ("BIPED_CONTACT_CAP", "128"),
        // Hold the arms at the natural stand pose (joint zero = Unitree's
        // elbows-bent CAD zero — the "zombie arms").
    ]
    .into_iter()
    .collect();
    if terrain {
        // Rough-terrain strips (v7 trained on them): the robot spawns on its
        // difficulty patch; per-triangle contacts merged like training.
        overrides.insert("BIPED_TERRAIN", "1");
        overrides.insert("BIPED_CONTACT_REDUCE", "1");
        // Start on visibly rough ground (levels 0-1 are near-flat); the
        // curriculum still promotes/demotes from here.
        overrides.insert(
            "BIPED_TERRAIN_INIT_LEVEL",
            &*Box::leak(terrain_level.min(19).to_string().into_boxed_str()),
        );
        // Optional terrain-shape overrides (demo sliders; defaults = training
        // terrain, in which case the knobs are left unset).
        if terrain_amp_pct != 100 {
            overrides.insert(
                "BIPED_TERRAIN_AMP",
                &*Box::leak(format!("{}", terrain_amp_pct as f32 / 100.0).into_boxed_str()),
            );
        }
        if terrain_slope_deg != 0 {
            overrides.insert(
                "BIPED_TERRAIN_SLOPE_DEG",
                &*Box::leak(terrain_slope_deg.to_string().into_boxed_str()),
            );
        }
    }
    let _ = biped_env_nexus::ENV_OVERRIDES.set(overrides);
}

/// A capsule node spanning `a` → `b` (link-local coordinates), parented under
/// `parent`. kiss3d capsules are Y-aligned and origin-centered.
fn add_segment(parent: &mut SceneNode3d, a: Vec3, b: Vec3, r: f32, color: Color) {
    let d = b - a;
    let len = d.length();
    let mut node = parent.add_capsule(r, len.max(1e-4));
    let rot = if len > 1e-6 {
        Rot3::from_rotation_arc(Vec3::Y, d / len)
    } else {
        Rot3::IDENTITY
    };
    node.set_pose(Pose3::from_parts(a + d * 0.5, rot));
    node.set_color(color);
}

/// Native-only parity probe for the GPU-resident obs path: after a few CPU
/// steps, compare (a) joint angles read from `dof_values` at the assembly-dof
/// ids vs the pose-derived CPU angles, and (b) the ws link-0 LTW quat vs the
/// CPU base quat.
#[cfg(not(target_arch = "wasm32"))]
async fn resident_probe() {
    configure_env(false, 4, 100, 0);
    let mut env = BipedNexusBatchEnv::new(MJCF_XML, 1, 1, 0xC0FFEE).await;
    let _ = env.reset_env(0).await;
    env.pin_command_for(0, 0.4, 0.0, 0.0);
    for _ in 0..25 {
        let _ = env.step(&[[0.0; NUM_JOINTS]]).await;
    }
    let poses = env.snapshot().await;
    let q_cpu = env.joint_angles_for(0, &poses);
    let (_, base_q_cpu) = env.base_pose_for(0, &poses);
    let dofs = env.actuated_assembly_dofs();
    let backend = env.gpu_backend().clone();
    let (dv, ws) = env.resident_buffers();
    let dv_host: Vec<f32> = backend.slow_read_vec(dv.buffer()).await.expect("dv");
    let ws_host: Vec<glamx::Vec4> = {
        let raw: Vec<nexus3d::rbd::glamx::Vec4> =
            backend.slow_read_vec(ws.buffer()).await.expect("ws");
        raw.iter().map(|v| Vec3::new(v.x, v.y, v.z).extend(v.w)).collect()
    };
    println!("{:>24} {:>10} {:>10}", "joint", "dof_values", "cpu(pose)");
    for j in 0..NUM_JOINTS {
        println!(
            "{:>24} {:>10.4} {:>10.4}",
            j,
            dv_host[dofs[j] as usize],
            q_cpu[j]
        );
    }
    let bq = ws_host[5]; // link0, quad WS_LTW=5 (n_envs=1)
    println!("ws LTW quat: [{:.4},{:.4},{:.4},{:.4}]", bq.x, bq.y, bq.z, bq.w);
    println!("dof_values[0..40]:");
    for (i, v) in dv_host.iter().take(40).enumerate() {
        if v.abs() > 1e-6 {
            println!("  [{i:>2}] {v:+.4}");
        }
    }
    println!("  (len {}, nonzero {})", dv_host.len(), dv_host.iter().filter(|v| v.abs() > 1e-6).count());
    // Candidate: ws COORDS quad (link·15 + 1) component x per actuated link.
    let links = env.actuated_link_ids();
    println!("{:>8} {:>12} {:>10}", "link", "ws_coords.x", "cpu(pose)");
    for (j, &l) in links.iter().enumerate() {
        let c = ws_host[(l as usize) * 15 + 1];
        let jr = ws_host[(l as usize) * 15]; // WS_JOINT_ROT quad, xyzw
        let ang = 2.0 * jr.z.atan2(jr.w);
        println!("{:>8} {:>12.4} {:>10.4} jr_angle={:+.4}", l, c.x, q_cpu[j], ang);
    }
    println!(
        "cpu base quat (xyzw): [{:.4},{:.4},{:.4},{:.4}]",
        base_q_cpu[0], base_q_cpu[1], base_q_cpu[2], base_q_cpu[3]
    );
}

/// Native-only sanity rollout: 6 s of the walking policy at cmd 0.4 m/s,
/// print torso pos + falls.
#[cfg(not(target_arch = "wasm32"))]
async fn headless_check(cfg: &DemoCfg) {
    configure_env(cfg.terrain, cfg.terrain_level, cfg.terrain_amp_pct, cfg.terrain_slope_deg);
    let ac = ActorCritic::load_from_bytes(POLICY_BIN).expect("policy checkpoint");
    let mut env = BipedNexusBatchEnv::new(MJCF_XML, 1, 1, 0xC0FFEE).await;
    let (mut obs, _) = env.reset_env(0).await;
    env.pin_command_for(0, 0.4, 0.0, 0.0);
    let steps = (6.0 / DT) as usize;
    let mut falls = 0u32;
    for step in 0..steps {
        let mut a = [0.0f32; NUM_JOINTS];
        a.copy_from_slice(&ac.mean(&obs)[..NUM_JOINTS]);
        let outs = env.step(&[a]).await;
        if outs[0].done {
            if outs[0].fell {
                falls += 1;
            }
            obs = env.reset_env(0).await.0;
            env.pin_command_for(0, 0.4, 0.0, 0.0);
        } else {
            obs.clone_from(&outs[0].obs);
        }
        if step % 50 == 0 {
            let poses = env.snapshot().await;
            let (p, _) = env.base_pose_for(0, &poses);
            println!(
                "t={:>4.1}s x={:+.2} torso_z={:.3} falls={falls}",
                step as f32 * DT,
                p[0],
                p[2]
            );
        }
    }
    println!("headless-check: falls = {falls} over 6 s (0 = walks)");
}

/// What `?ckpt=` pointed at: either a specific file, or a whole Hugging Face
/// repo whose newest checkpoint we should go and find.
enum CkptRef {
    /// Ready to download.
    File(String),
    /// A repo id like `owner/name` — ask the Hub what is in it.
    Repo(String),
}

/// Parse anything a person might paste. The point is that a handle copied off
/// a model page, the page URL itself, a link to one file in it, or a direct
/// URL somewhere else entirely all mean "run this policy".
///
/// - `owner/repo`                                 → newest checkpoint in it
/// - `owner/repo/sub/dir/ckpt.safetensors`        → that file, on `main`
/// - `https://huggingface.co/owner/repo`          → newest checkpoint in it
/// - `.../blob/main/x.safetensors` (the page URL)  → that file
/// - `.../resolve/main/x.safetensors` (raw)       → that file
/// - any other `http(s)` URL                       → fetched as-is
/// - a bare word                                  → a file beside the demo
fn parse_ckpt(spec: &str) -> CkptRef {
    let spec = percent_decode(spec);
    let spec = spec.trim().trim_end_matches('/');

    // Hub URLs: reduce to the handle forms below, keeping the revision.
    let hub = spec
        .strip_prefix("https://huggingface.co/")
        .or_else(|| spec.strip_prefix("http://huggingface.co/"))
        .or_else(|| spec.strip_prefix("https://hf.co/"))
        .or_else(|| spec.strip_prefix("hf.co/"))
        .or_else(|| spec.strip_prefix("hf:"));
    if let Some(rest) = hub {
        // Strip a query string (`?download=true` comes with the copy button).
        let rest = rest.split('?').next().unwrap_or(rest);
        let p: Vec<&str> = rest.split('/').collect();
        return match p.as_slice() {
            // owner/repo/{blob,resolve}/rev/path…
            [owner, repo, kind, rev, path @ ..]
                if (*kind == "blob" || *kind == "resolve") && !path.is_empty() =>
            {
                CkptRef::File(format!(
                    "https://huggingface.co/{owner}/{repo}/resolve/{rev}/{}",
                    path.join("/")
                ))
            }
            // owner/repo/tree/rev… — a folder view; still just the repo.
            [owner, repo, ..] => CkptRef::Repo(format!("{owner}/{repo}")),
            _ => CkptRef::File(spec.to_string()),
        };
    }
    if spec.starts_with("http://") || spec.starts_with("https://") {
        return CkptRef::File(spec.to_string());
    }
    let p: Vec<&str> = spec.splitn(3, '/').collect();
    match p.as_slice() {
        [owner, repo, path] => CkptRef::File(format!(
            "https://huggingface.co/{owner}/{repo}/resolve/main/{path}"
        )),
        [owner, repo] => CkptRef::Repo(format!("{owner}/{repo}")),
        // Bare name: a checkpoint sitting next to the demo, which is what the
        // bench pages' `?ckpt=` has always meant.
        _ => CkptRef::File(format!("{spec}.safetensors")),
    }
}

/// Pull `"rfilename": "…"` values out of a Hub model response. A dependency-
/// free scan rather than a JSON parser: this is the only field we want, and
/// the demo ships as a wasm module where every crate costs download size.
fn rfilenames(json: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = json;
    while let Some(i) = rest.find("\"rfilename\"") {
        rest = &rest[i + "\"rfilename\"".len()..];
        let Some(q1) = rest.find('"') else { break };
        let after = &rest[q1 + 1..];
        let Some(q2) = after.find('"') else { break };
        out.push(after[..q2].to_string());
        rest = &after[q2 + 1..];
    }
    out
}

/// Order checkpoints newest-first by the numbers in their names, so
/// `g1_v24_iter32780` beats `g1_v21_iter4560` beats `g1_v19_iter2740` without
/// anyone having to follow a naming convention exactly.
fn newest_first(files: &mut [String]) {
    fn nums(s: &str) -> Vec<u64> {
        let mut v = Vec::new();
        let mut cur = String::new();
        for c in s.chars() {
            if c.is_ascii_digit() {
                cur.push(c);
            } else if !cur.is_empty() {
                v.push(cur.parse().unwrap_or(0));
                cur.clear();
            }
        }
        if !cur.is_empty() {
            v.push(cur.parse().unwrap_or(0));
        }
        v
    }
    files.sort_by(|a, b| nums(b).cmp(&nums(a)).then_with(|| b.cmp(a)));
}

/// A human-readable label for the HUD: the file name without extension.
fn ckpt_label(spec: &str) -> String {
    let spec = percent_decode(spec);
    let spec = spec.trim_end_matches('/');
    spec.rsplit('/')
        .next()
        .unwrap_or(spec)
        .trim_end_matches(".safetensors")
        .to_string()
}

/// Minimal `%XX` decoder — the picker hands us an encoded URL, and pulling in
/// a crate to undo `encodeURIComponent` would be silly.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Turn a spec into a downloadable URL, asking the Hub for the repo contents
/// when only a handle was given.
async fn resolve_ckpt(spec: &str) -> Result<String, String> {
    match parse_ckpt(spec) {
        CkptRef::File(url) => Ok(url),
        CkptRef::Repo(repo) => {
            let api = format!("https://huggingface.co/api/models/{repo}");
            let body = fetch_text(&api)
                .await
                .map_err(|e| format!("{repo}: {e}"))?;
            let mut files: Vec<String> = rfilenames(&body)
                .into_iter()
                .filter(|f| f.ends_with(".safetensors"))
                .collect();
            if files.is_empty() {
                return Err(format!("{repo}: no .safetensors in the repo"));
            }
            newest_first(&mut files);
            Ok(format!(
                "https://huggingface.co/{repo}/resolve/main/{}",
                files[0]
            ))
        }
    }
}

/// Resolve → download → parse, with the failure reason kept intact so the HUD
/// can show it. The obs width comes from the file, so v21-era (45×5) and
/// v24-era (48×5) checkpoints both load.
async fn load_ckpt(spec: &str) -> Result<(ActorCritic, String), String> {
    let url = resolve_ckpt(spec).await?;
    let bytes = fetch_ckpt(&url).await?;
    let ac = ActorCritic::load_from_bytes(&bytes)
        .map_err(|e| format!("{}: not a zealot checkpoint ({e})", ckpt_label(&url)))?;
    Ok((ac, ckpt_label(&url)))
}

/// Download a checkpoint. In the browser this is a plain `fetch` (Hugging
/// Face serves the LFS files with permissive CORS); natively `?ckpt=` names a
/// file on disk.
#[cfg(target_arch = "wasm32")]
async fn fetch_ckpt(url: &str) -> Result<Vec<u8>, String> {
    use wasm_bindgen::JsCast;
    let win = web_sys::window().ok_or("no window")?;
    let resp: web_sys::Response = wasm_bindgen_futures::JsFuture::from(win.fetch_with_str(url))
        .await
        .map_err(|e| format!("fetch failed: {e:?}"))?
        .dyn_into()
        .map_err(|_| "not a Response".to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {} for {url}", resp.status()));
    }
    let buf = wasm_bindgen_futures::JsFuture::from(
        resp.array_buffer().map_err(|e| format!("{e:?}"))?,
    )
    .await
    .map_err(|e| format!("body read failed: {e:?}"))?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

#[cfg(not(target_arch = "wasm32"))]
async fn fetch_ckpt(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{path}: {e}"))
}

/// The Hub's model API, for turning a handle into a file list.
#[cfg(target_arch = "wasm32")]
async fn fetch_text(url: &str) -> Result<String, String> {
    let bytes = fetch_ckpt(url).await?;
    String::from_utf8(bytes).map_err(|e| format!("bad response: {e}"))
}

#[cfg(not(target_arch = "wasm32"))]
async fn fetch_text(_url: &str) -> Result<String, String> {
    Err("resolving a repo handle needs the browser build".to_string())
}

/// Render poses read back from the GPU **one frame late**, without ever
/// blocking on the GPU.
///
/// `env.snapshot()` fences: it submits a copy and then waits for the whole
/// queue — every control step this frame, each with its physics substeps — to
/// retire, and for the map callback to come back through the browser's task
/// queue. That wait was ~90 ms/frame against ~1.5 ms of actual encode work,
/// i.e. the demo spent its frame watching the GPU rather than feeding it.
///
/// Nothing on screen needs *this* frame's poses. So the copy is started at the
/// end of a frame and the result is taken at the start of the next one; if it
/// has not landed yet the previous poses are drawn again. The robot is one
/// 20 ms control step stale, which is invisible, and the CPU and GPU finally
/// run at the same time.
///
/// WebGPU-backed only (the browser demo, and native Metal/Vulkan through
/// wgpu). On any other backend `poll` falls back to the blocking snapshot.
struct PoseStream {
    /// Staging-buffer ring. One copy is kicked EVERY frame (into any free
    /// slot) and the newest landed one wins — with a single slot the next
    /// copy could only start after the previous landed, which measured as
    /// 84% of frames re-drawing a stale pose (the robot visibly animated at
    /// ~4 Hz while the counter said 25 fps). Three slots cover the ~2-3
    /// frames a map takes to come back through the browser.
    slots: Vec<PoseSlot>,
    /// Most recent poses; drawn again on frames where nothing landed.
    poses: Vec<NexusPose>,
    /// Frames the renderer reused stale poses (HUD diagnostic).
    stale_frames: u32,
    /// Whether the poses changed this frame (see `fresh` uses below).
    fresh: bool,
    /// Sim time the poses in hand were captured at. Motion must be
    /// differentiated in SIM time — wall time between reads is not the sim
    /// time between captures.
    captured_at: f32,
}

struct PoseSlot {
    staging: wgpu::Buffer,
    /// `Some` while a copy into this slot is in flight; carries the sim time
    /// its snapshot was taken at.
    rx: Option<(std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>, f32)>,
}

impl PoseStream {
    fn new(poses: Vec<NexusPose>) -> Self {
        Self { slots: Vec::new(), poses, stale_frames: 0, fresh: true, captured_at: 0.0 }
    }

    /// The wgpu handles, if this backend is WebGPU.
    fn wgpu_parts<'a>(
        backend: &'a khal::backend::GpuBackend,
        buf: &'a khal::backend::GpuBuffer<NexusPose>,
    ) -> Option<(&'a wgpu::Device, &'a wgpu::Queue, &'a wgpu::Buffer)> {
        match (backend, buf) {
            (khal::backend::GpuBackend::WebGpu(w), khal::backend::GpuBuffer::WebGpu(b)) => {
                Some((w.device(), w.queue(), b))
            }
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    /// Harvest landed copies (newest wins) and kick this frame's copy.
    /// Returns `false` if this backend has no pipelined path.
    fn pump(
        &mut self,
        backend: &khal::backend::GpuBackend,
        src: &khal::backend::GpuBuffer<NexusPose>,
        sim_time: f32,
    ) -> bool {
        let Some((device, queue, src)) = Self::wgpu_parts(backend, src) else {
            return false;
        };
        self.fresh = false;
        let bytes = src.size();
        while self.slots.len() < 3 {
            self.slots.push(PoseSlot {
                staging: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("zealot_pose_readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                rx: None,
            });
        }

        // Native backends only progress map callbacks when polled; in the
        // browser the runtime drives them and this is a no-op.
        khal::backend::Backend::poll(backend);

        // Take every landed copy; keep only the newest snapshot's data.
        let mut best: Option<usize> = None;
        for i in 0..self.slots.len() {
            let Some((rx, at)) = &self.slots[i].rx else { continue };
            let at = *at;
            match rx.try_recv() {
                Ok(Ok(())) => {
                    match best {
                        Some(b) if self.best_at(b) >= at => {
                            // Older than what we already have: recycle unread.
                            self.slots[i].staging.unmap();
                            self.slots[i].rx = None;
                        }
                        _ => {
                            if let Some(b) = best {
                                self.slots[b].staging.unmap();
                                self.slots[b].rx = None;
                            }
                            best = Some(i);
                        }
                    }
                }
                Ok(Err(_)) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.slots[i].staging.unmap();
                    self.slots[i].rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if let Some(b) = best {
            {
                let view = self.slots[b].staging.slice(..).get_mapped_range();
                let n = (view.len() / core::mem::size_of::<NexusPose>()).min(self.poses.len());
                // SAFETY: `Pose` is plain old data; the mapped range may be
                // under-aligned for it, so copy bytes rather than casting.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        view.as_ptr(),
                        self.poses.as_mut_ptr() as *mut u8,
                        n * core::mem::size_of::<NexusPose>(),
                    );
                }
            }
            self.captured_at = self.best_at(b);
            self.slots[b].staging.unmap();
            self.slots[b].rx = None;
            self.fresh = true;
            self.stale_frames = 0;
        } else {
            self.stale_frames += 1;
        }

        // Kick this frame's copy into any free slot. All slots busy means the
        // GPU is several frames behind; adding more copies would not help.
        if let Some(free) = (0..self.slots.len()).find(|&i| self.slots[i].rx.is_none()) {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zealot_pose_copy"),
            });
            enc.copy_buffer_to_buffer(src, 0, &self.slots[free].staging, 0, bytes);
            queue.submit([enc.finish()]);
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            self.slots[free].staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.try_send(r);
            });
            self.slots[free].rx = Some((rx, sim_time));
        }
        true
    }

    fn best_at(&self, i: usize) -> f32 {
        self.slots[i].rx.as_ref().map_or(f32::MIN, |(_, at)| *at)
    }
}

/// Demo entry point — the thin `g1_web`/`g1_terrain_web` examples call this
/// from their `#[kiss3d::main] main`.
pub async fn run(cfg: DemoCfg) {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    #[cfg(not(target_arch = "wasm32"))]
    if std::env::args().any(|a| a == "--headless-check") {
        headless_check(&cfg).await;
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::args().any(|a| a == "--resident-probe") {
        resident_probe().await;
        return;
    }
    configure_env(cfg.terrain, cfg.terrain_level, cfg.terrain_amp_pct, cfg.terrain_slope_deg);

    let mut window = Window::new_with_size("zealot G1 — nexus GPU physics", 1200, 900).await;
    window.set_background_color(Color::new(0.051, 0.169, 0.180, 1.0));

    // Z-up orbit camera (the MJCF scene is Z-up). Terrain: close-follow
    // ROBOT 0 on its own lane, starting framed on its spawn patch (whose
    // ground height depends on the slope knob). Framing the whole fleet
    // instead would sit 14 m out, and at that zoom a 0.4 m/s walk covers so
    // few pixels it reads as a crawl. Scroll to pull back.
    let mut camera = if cfg.terrain {
        let (spawn_x, _) = TerrainStrip::patch_center(cfg.terrain_level.min(19));
        let spawn_z = (cfg.terrain_slope_deg.min(45) as f32).to_radians().tan()
            * (spawn_x - zealot_env::terrain::STRIP_X0);
        // Robot 0's lane — same rule as `offset_of` below (env % 3 lanes,
        // 9 m apart; a lone robot renders at the origin).
        let lane_y = if cfg.n_robots == 1 { 0.0 } else { -9.0 };
        let (back, up) = (4.0, 2.2);
        OrbitCamera3d::new(
            Vec3::new(spawn_x - back * 0.4, lane_y - back, spawn_z + up),
            Vec3::new(spawn_x, lane_y, spawn_z + 0.6),
        )
    } else {
        OrbitCamera3d::new(Vec3::new(-4.5, -5.5, 3.2), Vec3::new(0.0, 0.0, 0.6))
    };
    camera.set_up_axis(Vec3::Z);
    // The demo opens already framed on one robot, and the page scrolls under
    // the canvas — so the wheel may only pull OUT from here. Zooming further
    // in just buries the camera in the robot and steals the page's scroll.
    camera.set_min_dist(camera.dist());

    let mut scene = SceneNode3d::empty();
    scene.add_directional_light(Vec3::new(1.0, -2.0, -3.0));

    // Ground slab with stripes for motion cues; physics floor top is z = 0.
    // Terrain runs +x for 160 m — extend the slab under it.
    let (slab_cx, slab_len, stripe_hi) = if cfg.terrain {
        (80.0, 400.0, 100)
    } else {
        (0.0, 60.0, 14)
    };
    // Terrain strips carry near-zero heights (quantized flat cells at
    // exactly z = 0) — drop the visual slab a hair so the strip tops don't
    // z-fight against it.
    let slab_drop = if cfg.terrain { 0.02 } else { 0.0 };
    let mut ground = scene.add_cube(slab_len, 60.0, 0.1);
    ground.set_position(Vec3::new(slab_cx, 0.0, -0.05 - slab_drop));
    ground.set_color(Color::new(0.13, 0.30, 0.31, 1.0));
    for i in -14..=stripe_hi {
        let mut stripe = scene.add_cube(0.02, 60.0, 0.002);
        stripe.set_position(Vec3::new(i as f32, 0.0, 0.001 - slab_drop));
        stripe.set_color(Color::new(0.21, 0.42, 0.44, 1.0));
        if i <= 14 {
            let mut stripe = scene.add_cube(slab_len, 0.02, 0.002);
            stripe.set_position(Vec3::new(slab_cx, i as f32, 0.001 - slab_drop));
            stripe.set_color(Color::new(0.21, 0.42, 0.44, 1.0));
        }
    }

    // Robots: N independent envs (separate physics batches — they don't
    // collide with each other), rendered side by side via per-env visual
    // offsets. Terrain robots sit on their family strip's lane (strips are
    // 8 m wide — tile with a 1 m gap so they don't overlap/z-fight; family =
    // env % 3, same rule as the physics); flat-fleet robots stand in a
    // 2-row grid. One group node per MJCF body per robot; bodies with baked
    // visual meshes render those, anything without falls back to the
    // joint-sphere/capsule stick figure.
    let n_robots = cfg.n_robots;
    let terrain = cfg.terrain;
    let offset_of = move |e: usize| {
        if n_robots == 1 {
            Vec3::ZERO
        } else if terrain {
            Vec3::new(0.0, ((e % 3) as f32 - 1.0) * 9.0, 0.0)
        } else {
            let col = (e % 5) as f32 - 2.0;
            let row = (e / 5) as f32;
            Vec3::new(row * -2.2, col * 1.8, 0.0)
        }
    };
    // Straight lanes at a shared speed: staggered speeds stretch the
    // formation past the camera framing within a minute, and yaw would curve
    // robots across their neighbours visually (the envs don't collide, but
    // overlapping renders look wrong). Robots separate via spawn jitter.
    let preset_cmd = |_e: usize| -> [f32; 3] { [0.4, 0.0, 0.0] };
    let mjcf = parse_mjcf(MJCF_XML);
    let visuals = load_visuals();
    let body_color = Color::new(0.75, 0.78, 0.80, 1.0);
    let joint_color = Color::new(0.208, 0.757, 0.804, 1.0);
    let foot_color = Color::new(0.925, 0.702, 0.208, 1.0);
    // Terrain: regenerate the exact strips the env builds (deterministic from
    // the same seed + family-per-env rule) and draw one strip per family in
    // use, at that family's lane. Strip runs x ∈ [8, 168], harder with
    // distance.
    const ENV_SEED: u64 = 0xC0FFEE;
    // Kept beyond rendering: per-family height samplers so the demo's own
    // fall detection and camera work relative to the LOCAL ground (slope can
    // put the ground tens of meters up).
    let mut strips: Vec<TerrainStrip> = Vec::new();
    if cfg.terrain {
        // Must match the physics strips: same seed AND same shape overrides.
        let render_params = TerrainParams {
            amp: cfg.terrain_amp_pct as f32 / 100.0,
            slope: (cfg.terrain_slope_deg.min(45) as f32).to_radians().tan(),
        };
        let fams = [TerrainFamily::Boxes, TerrainFamily::Rough, TerrainFamily::Wave];
        for (f, fam) in fams.into_iter().enumerate() {
            let strip = TerrainStrip::generate_with(fam, ENV_SEED, render_params);
            if f < n_robots {
                let (v, t) = strip.mesh();
                let sv: Vec<Vec3> =
                    v.into_iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();
                let mut node = scene.add_trimesh(sv, t, Vec3::ONE, true);
                node.set_position(offset_of(f));
                node.set_color(match f {
                    0 => Color::new(0.30, 0.42, 0.40, 1.0),
                    1 => Color::new(0.33, 0.40, 0.36, 1.0),
                    _ => Color::new(0.28, 0.38, 0.42, 1.0),
                });
            }
            strips.push(strip);
        }
    }
    let mut robots: Vec<Vec<SceneNode3d>> = Vec::with_capacity(n_robots);
    for _e in 0..n_robots {
        let mut body_nodes: Vec<SceneNode3d> = Vec::with_capacity(mjcf.len());
        for body in &mjcf {
            let mut group = scene.add_group();
            if let Some(groups) = visuals.get(&body.name) {
                let color = if body.name == "pelvis" {
                    Color::new(0.80, 0.82, 0.85, 1.0)
                } else if body.name.contains("ankle_roll") {
                    Color::new(0.28, 0.30, 0.33, 1.0)
                } else {
                    Color::new(0.60, 0.63, 0.66, 1.0)
                };
                for (_rgba, verts, tris) in groups {
                    let mut node =
                        group.add_trimesh(verts.clone(), tris.clone(), Vec3::ONE, false);
                    node.set_color(color);
                }
            } else {
                let mut joint = group.add_sphere(0.035);
                joint.set_color(joint_color);
                for (p1, p2, r) in &body.capsules {
                    let a = Vec3::new(p1.x, p1.y, p1.z);
                    let b = Vec3::new(p2.x, p2.y, p2.z);
                    add_segment(&mut group, a, b, (*r).max(0.015), foot_color);
                }
            }
            body_nodes.push(group);
        }
        for body in &mjcf {
            if let Some(p) = body.parent {
                if visuals.contains_key(&body.name) && visuals.contains_key(&mjcf[p].name) {
                    continue;
                }
                let child_origin =
                    Vec3::new(body.local_pos.x, body.local_pos.y, body.local_pos.z);
                let mut parent_node = body_nodes[p].clone();
                add_segment(&mut parent_node, Vec3::ZERO, child_origin, 0.028, body_color);
            }
        }
        robots.push(body_nodes);
    }

    // Env: N envs, template 0 = DR off. The policy drives the legs; waist/arms
    // are PD-held by the env at the AGILE holding gains.
    //
    // `?ckpt=` swaps in a published checkpoint (Hugging Face by default). Its
    // obs width comes from the checkpoint itself, so pre-gyro policies (v21
    // and earlier, 45×5) and gyro ones (v24 on, 48×5) both just run. A fetch
    // or parse failure falls back to the embedded default rather than showing
    // a dead canvas — the HUD says which one is actually driving.
    let (ac, ckpt_name) = match &cfg.ckpt {
        Some(spec) => match load_ckpt(spec).await {
            Ok((ac, label)) => (ac, label),
            Err(e) => {
                log_warn(&e);
                (
                    ActorCritic::load_from_bytes(POLICY_BIN).expect("policy checkpoint"),
                    format!("{DEFAULT_CKPT} (fallback: {e})"),
                )
            }
        },
        None => (
            ActorCritic::load_from_bytes(POLICY_BIN).expect("policy checkpoint"),
            DEFAULT_CKPT.to_string(),
        ),
    };
    let ckpt_name = format!("{ckpt_name} — {}-dim obs", ac.obs_norm.state().0.len());
    let mut env = BipedNexusBatchEnv::new(MJCF_XML, n_robots, 1, 0xC0FFEE).await;
    let mut cmds: Vec<[f32; 3]> = (0..n_robots).map(preset_cmd).collect();
    for e in 0..n_robots {
        let _ = env.reset_env(e).await;
        env.pin_command_for(e, cmds[e][0], cmds[e][1], cmds[e][2]);
    }

    // ---- GPU-resident control loop: obs assembly + policy GEMM + PD-target
    // scatter all run as GPU compute; the CPU never reads observations back.
    // (The former CPU path — `ac.mean` per env + `env.step`'s fenced obs
    // readback — cost ~19 ms of wall time per control step in the browser;
    // this path is encode-only.)
    let backend = env.gpu_backend().clone();
    let mut gpol =
        crate::gpu_policy::GpuPolicy::new(&backend, &ac, n_robots).expect("gpu policy");
    let spec = zealot_env::robots::RobotSpec::from_env();
    let (nmean, nm2, ncount) = ac.obs_norm.state();
    let obs_cfg = zealot_gpu_obs::GpuObsConfig {
        link_ids: {
            let l = env.actuated_link_ids();
            core::array::from_fn(|j| l[j])
        },
        default_pos: core::array::from_fn(|j| spec.joints[j].default_pos),
        target_lo: core::array::from_fn(|j| spec.joints[j].pos_limit.0),
        target_hi: core::array::from_fn(|j| spec.joints[j].pos_limit.1),
        norm_mean: nmean.to_vec(),
        norm_std: nm2.iter().map(|&v| (v / ncount).max(1e-8).sqrt()).collect(),
        dt: DT,
        action_scale: spec.joints[0].action_scale,
    };
    let mut gobs = zealot_gpu_obs::GpuObs::new(&backend, n_robots, &obs_cfg).expect("gpu obs");
    for e in 0..n_robots {
        gobs.set_cmd(&backend, e, cmds[e]).expect("cmd");
    }
    // CPU-side episode counters (timeout + reset bookkeeping only).
    let mut ep_steps: Vec<u32> = vec![0; n_robots];
    let mut fallen: Vec<usize> = Vec::new();
    // Frames to skip fall-detection after a reset: the render snapshot lags
    // the teleport by a frame or two, so a fallen pose would be re-counted.
    let mut fall_cooldown: Vec<u32> = vec![0; n_robots];
    const FALL_Z: f32 = 0.45;
    const TILT_COS: f32 = 0.342; // cos 70°

    let t0 = Instant::now();
    let mut sim_time = 0.0f32;
    let mut falls: u32 = 0;
    let mut pending_reset = false;
    // Command sliders (applied to ALL robots on change) + measured planar
    // speed of the formation centroid (EMA over render frames; teleport
    // frames are skipped).
    let mut cmd_ui: [f32; 3] = [0.4, 0.0, 0.0];
    let mut prev_mean_base: Option<Vec3> = None;
    let mut prev_poses_at = 0.0f32;
    let mut meas_speed = 0.0f32;
    // Perf HUD: rolling averages over ~1 s windows. A control step = 1 env
    // GPU sim step (decimation substeps + solve) + N policy forwards + obs.
    let mut perf_frames = 0u32;
    let mut perf_steps = 0u32;
    let mut perf_step_ms_acc = 0.0f32;
    let mut perf_pol_ms_acc = 0.0f32;
    let mut perf_snap_ms_acc = 0.0f32;
    let mut perf_t = t0;
    let mut hud_fps = 0.0f32;
    let mut hud_step_ms = 0.0f32;
    let mut hud_pol_ms = 0.0f32;
    let mut hud_snap_ms = 0.0f32;
    let mut hud_realtime = 1.0f32;
    // Last frame's physics-truth foot clearance, for the title diagnostics.
    let mut hud_sole = 0.0f32;
    // Policy input/output magnitudes for the cross-browser diagnostics.
    let (mut dbg_in, mut dbg_out, mut dbg_nan) = (0.0f32, 0.0f32, 0usize);

    // `?diag=1` turns on the per-second policy I/O readbacks (see below).
    let diag = probe_u32("diag=", 0) == 1;
    // Physics submit granularity (`?phys=`): 0 per-phase — the DEFAULT, it
    // measured best for sim throughput (n=3: 83% RT vs 66% merged; merging
    // frees the CPU for more frames but the extra frames' GPU work comes out
    // of the physics budget) — 1 per-substep, 2 all-in-one.
    let phys_mode = probe_u32("phys=", 0);
    // Pose readback mode. DEFAULT = per-frame BLOCKING snapshot: every
    // rendered frame carries a fresh pose at regular sim intervals. The
    // pipelined path (`?snap=0`) posts higher fps by never fencing — but the
    // extra frames mostly RE-SHOW old poses (measured 52-84% stale, sim gaps
    // of 240-450 ms between displayed poses vs ≤170 ms blocking, at equal
    // sim speed), so the robot looks slower and choppier while the counters
    // look better. The fence is also natural backpressure: the queue drains
    // every frame, so input latency stays ~one frame by construction.
    let snap_block = probe_u32("snap=", 1) == 1;
    let (mut reg_frames, mut reg_stale) = (0u32, 0u32);
    let mut reg_last_disp = -1.0f32;
    let mut reg_max_gap = 0.0f32;
    let (mut hud_stale_pct, mut hud_gap_ms) = (0.0f32, 0.0f32);
    // `?prof=1`: per-pass GPU timings. Once a second, ONE control step runs
    // with timestamp queries (dedicated passes per label — slightly slower,
    // so only that step), read back without blocking; the accumulated
    // per-label averages go to the console every 5 s.
    let prof = probe_u32("prof=", 0) == 1;
    let mut prof_ts = prof.then(|| khal::backend::GpuTimestamps::new(&backend, 512));
    let mut prof_acc: std::collections::HashMap<String, (f64, u32)> =
        std::collections::HashMap::new();
    let mut prof_steps = 0u32;
    let mut prof_last_dump = 0u32;
    let mut prof_pending = false;
    let mut prof_want = false;
    // `?fused=1`: force nexus's fused colored-sweep kernels (one dispatch per
    // sweep instead of one per colour) — A/B for dispatch-latency-bound GPUs.
    if probe_u32("fused=", 0) == 1 {
        nexus3d::rbd::pipeline::FORCE_FUSED_SWEEPS
            .store(true, core::sync::atomic::Ordering::Relaxed);
    }


    drive::install();
    // Start the latched command at what the demo is already running, so the
    // first tap bumps up from the visible value instead of from zero.
    drive::sync(cmd_ui);

    // ---- Tap-to-walk: click/tap the ground and robot 0 walks there. The
    // policy still only ever sees a velocity command — a tiny P-controller
    // steers heading + the lateral channel toward the target, which is
    // exactly what a higher-level navigator does on hardware.
    let mut nav_target: Option<Vec3> = None; // render-space ground point
    let mut nav_marker = scene.add_cylinder(0.22, 0.02);
    nav_marker.set_color(Color::new(0.21, 0.76, 0.80, 1.0));
    nav_marker.set_visible(false);
    let mut cursor_px = (0.0f64, 0.0f64);
    let mut press_px: Option<(f64, f64)> = None;
    let mut last_drive_cmd = drive::command();
    // Wander mode: the demo opens with a random target so the robot is
    // visibly going SOMEWHERE, and each arrival draws the next one — until
    // any user input (tap, key, gamepad, slider, preset) takes over.
    let mut wander = true;
    let mut wander_seeded = false;
    // Cheap LCG; statistical quality is irrelevant for picking stroll points.
    let mut wander_rng: u32 = 0x9E37_79B9;
    let mut wander_rand = move || {
        wander_rng = wander_rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (wander_rng >> 8) as f32 / (1u32 << 24) as f32
    };

    // One blocking snapshot to prime the pipeline; every frame after this one
    // reads back without waiting.
    let mut pose_stream = PoseStream::new(env.snapshot().await);

    while window.render_3d(&mut scene, &mut camera).await {
        // Tap detection: a left-button press that releases within a few
        // pixels is a tap; anything longer is the camera orbit. Events the
        // UI already consumed are skipped, as is the HUD panel's corner.
        for event in window.events().iter() {
            if event.inhibited {
                continue;
            }
            match event.value {
                KWindowEvent::CursorPos(x, y, _) => cursor_px = (x, y),
                KWindowEvent::MouseButton(KMouseButton::Button1, KAction::Press, _) => {
                    press_px = Some(cursor_px);
                }
                KWindowEvent::MouseButton(KMouseButton::Button1, KAction::Release, _) => {
                    if let Some((px, py)) = press_px.take() {
                        let moved = (cursor_px.0 - px).hypot(cursor_px.1 - py);
                        let size = window.size();
                        let in_panel = px < 0.30 * size.x as f64 && py < 0.60 * size.y as f64;
                        if moved < 8.0 && !in_panel {
                            // Ray through the tap; walk it onto the ground
                            // surface (terrain height varies, so fixed-point
                            // iterate z = h(x, y) — converges in a few steps
                            // on these gentle slopes).
                            let (o, d) = kiss3d::camera::Camera3d::unproject(
                                &camera,
                                glamx::Vec2::new(px as f32, py as f32),
                                glamx::Vec2::new(size.x as f32, size.y as f32),
                            );
                            if d.z < -1e-3 {
                                let off0 = offset_of(0);
                                let mut t = -o.z / d.z;
                                for _ in 0..6 {
                                    let hit = o + d * t;
                                    let h = if terrain {
                                        strips[0].height(hit.x - off0.x, hit.y - off0.y)
                                    } else {
                                        0.0
                                    };
                                    t = (h - o.z) / d.z;
                                }
                                let hit = o + d * t;
                                wander = false;
                                nav_target = Some(hit);
                                nav_marker.set_pose(Pose3::from_parts(
                                    hit + Vec3::new(0.0, 0.0, 0.03),
                                    Rot3::from_rotation_x(core::f32::consts::FRAC_PI_2),
                                ));
                                nav_marker.set_visible(true);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Manual driving cancels the autopilot.
        let drive_now = drive::command();
        if drive_now != last_drive_cmd {
            last_drive_cmd = drive_now;
            wander = false;
            nav_target = None;
            nav_marker.set_visible(false);
        }

        // Wander: pick a stroll point 2–3.5 m out at a random bearing once
        // the spawn has settled, and again at every arrival below.
        if wander && !wander_seeded && sim_time > 1.5 {
            wander_seeded = true;
            let off0 = offset_of(0);
            let (base, _) = env.base_pose_for(0, &pose_stream.poses);
            let ang = wander_rand() * core::f32::consts::TAU;
            let dist = 2.0 + 1.5 * wander_rand();
            let tx = base[0] + off0.x + dist * ang.cos();
            // Keep the stroll on the strip (it is ~PATCH wide; lane center
            // is the robot's spawn y).
            let ty = (base[1] + off0.y + dist * ang.sin()).clamp(off0.y - 3.0, off0.y + 3.0);
            let tz = if terrain {
                strips[0].height(tx - off0.x, ty - off0.y)
            } else {
                0.0
            };
            let hit = Vec3::new(tx, ty, tz);
            nav_target = Some(hit);
            nav_marker.set_pose(Pose3::from_parts(
                hit + Vec3::new(0.0, 0.0, 0.03),
                Rot3::from_rotation_x(core::f32::consts::FRAC_PI_2),
            ));
            nav_marker.set_visible(true);
        }

        // Autopilot: steer robot 0's velocity command at the target. Heading
        // via yaw, but the LATERAL channel carries the approach too — the
        // policy tracks ±0.4 m/s sideways well, which closes on the target
        // even mid-turn.
        if let Some(tgt) = nav_target {
            let off0 = offset_of(0);
            let (base, brot) = env.base_pose_for(0, &pose_stream.poses);
            let bx = base[0] + off0.x;
            let by = base[1] + off0.y;
            let (dx, dy) = (tgt.x - bx, tgt.y - by);
            let dist = dx.hypot(dy);
            let c: [f32; 3] = if dist < 0.3 {
                if wander {
                    // Arrived at a stroll point: draw the next one.
                    wander_seeded = false;
                } else {
                    nav_target = None;
                    nav_marker.set_visible(false);
                }
                [0.0, 0.0, 0.0]
            } else {
                let q = Rot3::from_xyzw(brot[0], brot[1], brot[2], brot[3]);
                let fwd = q * Vec3::X;
                let yaw = fwd.y.atan2(fwd.x);
                let mut err = dy.atan2(dx) - yaw;
                err = err.sin().atan2(err.cos());
                // Target direction in the body frame.
                let cb = (-yaw).cos();
                let sb = (-yaw).sin();
                let fx = cb * dx - sb * dy;
                let fy = sb * dx + cb * dy;
                let slow = dist.min(1.0);
                [
                    (0.8 * fx * slow).clamp(-0.3, 0.6),
                    (0.8 * fy * slow).clamp(-0.4, 0.4),
                    (1.5 * err).clamp(-1.0, 1.0),
                ]
            };
            cmd_ui = c;
            drive::sync(c);
            last_drive_cmd = drive::command();
            for e in 0..n_robots {
                if cmds[e] != c {
                    cmds[e] = c;
                    env.pin_command_for(e, c[0], c[1], c[2]);
                    gobs.set_cmd(&backend, e, c).expect("cmd");
                }
            }
        } else if let Some(c) = drive_now {
            // Driving: each key press bumped the latched command, so apply it
            // whenever it differs from what the robots are already running.
            if cmds.iter().any(|p| *p != c) {
                cmd_ui = c;
                for e in 0..n_robots {
                    cmds[e] = c;
                    env.pin_command_for(e, c[0], c[1], c[2]);
                    gobs.set_cmd(&backend, e, c).expect("cmd");
                }
            }
        }

        if pending_reset {
            pending_reset = false;
            for e in 0..n_robots {
                let _ = env.reset_env(e).await;
                env.pin_command_for(e, cmds[e][0], cmds[e][1], cmds[e][2]);
                gobs.reset_env(&backend, e).expect("gobs reset");
                ep_steps[e] = 0;
            }
        }

        // Advance physics to wall-clock time, at most 4 control steps a
        // frame — and never more than 3 steps ahead of what the GPU has
        // retired (paced via the pose stream): past that, more submits only add
        // latency the browser answers with tab-level throttling.
        let wall = t0.elapsed().as_secs_f32();
        let mut steps_this_frame = 0;
        // Pace on the pose stream: its map landing means the GPU has caught
        // up to the copy submitted at the end of some recent frame, so its
        // staleness counts how far the GPU is running behind the loop. While
        // the GPU is behind, submitting more steps does not make the sim any
        // faster — it only grows the queue, and with it the delay between a
        // key press and the robot responding, until the browser throttles the
        // whole tab. (That unbounded queue was the "super laggy" demo: the
        // old per-second diagnostic fence used to drain it by accident, and
        // gating that off exposed it.) The threshold trades sim speed against
        // input latency; 3 measured best — beyond it, real-time gains flatten
        // while the frame rate and the responsiveness fall away:
        //   stale<2  28 fps  62% RT      stale<5  17 fps  92% RT
        //   stale<3  24 fps  83% RT      stale<8  14 fps  94% RT
        while sim_time < wall && steps_this_frame < 4 && pose_stream.stale_frames < 3 {
            // GPU-resident control step: obs kernel → policy GEMMs → commit
            // (targets + action ring) → motor scatter → decimation substeps.
            // Encode + submit only; the per-frame render snapshot below is
            // the sole GPU→CPU fence.
            let step_t0 = Instant::now();
            // Single-submit control step: obs assembly, policy GEMMs, action
            // commit and motor scatter all share ONE command buffer (each
            // submit is a wasm→JS→browser crossing).
            let mut enc = backend.begin_encoding();
            {
                let (_dv, ws) = env.resident_buffers();
                gobs.encode_assemble(&mut enc, ws, gpol.actor_input_mut())
                    .expect("obs assemble");
            }
            let mut cur = crate::cutile_gemm::EncCursor::from_encoder(&backend, enc);
            gpol.encode_actor(&backend, &mut cur).expect("actor");
            let mut enc = cur.into_encoder().expect("encoder");
            gobs.encode_commit(&mut enc, gpol.actor_output()).expect("commit");
            env.encode_scatter_targets(&mut enc, &gobs.targets);
            backend.submit(enc).expect("submit ctrl step");
            // Physics submit granularity, `?phys=`: 0 per-phase (original),
            // 1 per-substep, 2 all-in-one — measurement knob.
            match &mut prof_ts {
                // Profiled step when one was requested and the reader is free.
                Some(ts) if prof_want && !prof_pending && ts.is_idle() => {
                    ts.reset();
                    env.step_physics_profiled(ts);
                    ts.request_read(&backend);
                    prof_pending = true;
                    prof_want = false;
                }
                _ => match phys_mode {
                    1 => env.step_physics_substep_submits(),
                    2 => {
                        let mut penc = backend.begin_encoding();
                        env.step_physics_encoded(&mut penc);
                        backend.submit(penc).expect("submit physics");
                    }
                    _ => env.step_physics_only(),
                },
            }
            perf_step_ms_acc += step_t0.elapsed().as_secs_f32() * 1e3;
            perf_steps += 1;
            for e in 0..n_robots {
                ep_steps[e] += 1;
            }
            sim_time += DT;
            steps_this_frame += 1;
        }
        if sim_time < wall - 0.25 {
            // Fell too far behind to catch up; drop the debt rather than
            // spiral. `hud_realtime` is measured over the 1 s window below,
            // which already reflects the lost steps.
            sim_time = wall;
        }

        if let Some(ts) = &mut prof_ts {
            if prof_pending {
                if let Some(rows) = ts.try_take(&backend) {
                    prof_pending = false;
                    prof_steps += 1;
                    for r in rows {
                        let e = prof_acc.entry(r.label).or_insert((0.0, 0));
                        e.0 += r.duration_ms;
                        e.1 += 1;
                    }
                    if prof_steps >= prof_last_dump + 5 {
                        prof_last_dump = prof_steps;
                        let mut rows: Vec<(String, f64, u32)> = prof_acc
                            .iter()
                            .map(|(k, (ms, n))| (k.clone(), ms / prof_steps as f64, *n))
                            .collect();
                        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
                        let total: f64 = rows.iter().map(|r| r.1).sum();
                        let mut out = format!(
                            "[prof] {prof_steps} steps, GPU {total:.2} ms/ctrl-step\n"
                        );
                        for (label, ms, n) in rows.iter().take(14) {
                            out.push_str(&format!(
                                "[prof]  {ms:7.3} ms  x{:<3} {label}\n",
                                n / prof_steps
                            ));
                        }
                        log_warn(&out);
                    }
                }
            }
        }

        // Perf HUD bookkeeping (~1 s windows).
        perf_frames += 1;
        let win = perf_t.elapsed().as_secs_f32();
        if win >= 1.0 {
            hud_fps = perf_frames as f32 / win;
            hud_step_ms = if perf_steps > 0 {
                perf_step_ms_acc / perf_steps as f32
            } else {
                0.0
            };
            hud_pol_ms = if perf_steps > 0 {
                perf_pol_ms_acc / perf_steps as f32
            } else {
                0.0
            };
            hud_snap_ms = perf_snap_ms_acc / perf_frames.max(1) as f32;
            // Steps actually run vs steps needed for real time (50 Hz).
            hud_realtime = (perf_steps as f32 / (win / DT)).min(1.0);
            // Diagnostic: mean |value| of the policy's INPUT (normalized obs)
            // and OUTPUT (actions). This is what separates "physics broken"
            // from "policy broken" across browsers — a zero output with a
            // healthy input means the GEMM path miscompiled, not the sim.
            //
            // OFF unless `?diag=1`: each of these is a blocking readback, i.e.
            // a full pipeline fence, and doing two of them every second put a
            // ~60 ms stall into one frame per second. Average frame rate hid
            // it; the stutter did not. Everything else in the title line is
            // free (counters the CPU already has), so the cross-browser
            // channel still works without this.
            #[cfg(target_arch = "wasm32")]
            if diag {
                let mabs = |v: &[f32]| {
                    if v.is_empty() { 0.0 } else {
                        v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32
                    }
                };
                let inp = backend
                    .slow_read_vec(gpol.actor_input_mut().buffer())
                    .await
                    .unwrap_or_default();
                let out = backend
                    .slow_read_vec(gpol.actor_output().buffer())
                    .await
                    .unwrap_or_default();
                dbg_in = mabs(&inp);
                dbg_out = mabs(&out);
                dbg_nan = inp.iter().chain(out.iter()).filter(|x| !x.is_finite()).count();
            }
            // Mirror the key numbers into the document title: the canvas HUD
            // is unreadable without a screenshot, and the title can be read
            // from ANY browser (AppleScript, WebDriver) — which is how the
            // Safari/Firefox WebGPU behaviour gets diagnosed at all.
            #[cfg(target_arch = "wasm32")]
            if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                // GPU-boundary rates over the window (submits/passes/copies/
                // uploads/maps per second) — khal's wasm perf counters.
                let (c_sub, c_pass, c_copy, c_wr, c_map) =
                    khal::backend::webgpu::perf_counters::take();
                let cps = |v: u32| (v as f32 / win).round();
                doc.set_title(&format!(
                    "z| falls={} sole={:+.3} spd={:.2} cmd={:.2} rt={:.0}% fps={:.0} stale={:.0}% gap={:.0}ms gpu[sub={} pass={} cp={} wr={} map={}] |in|={:.4} |act|={:.4} nan={}",
                    falls,
                    hud_sole,
                    meas_speed,
                    cmd_ui[0],
                    hud_realtime * 100.0,
                    hud_fps,
                    hud_stale_pct,
                    hud_gap_ms,
                    cps(c_sub),
                    cps(c_pass),
                    cps(c_copy),
                    cps(c_wr),
                    cps(c_map),
                    dbg_in,
                    dbg_out,
                    dbg_nan,
                ));
            }
            hud_stale_pct = 100.0 * reg_stale as f32 / reg_frames.max(1) as f32;
            hud_gap_ms = reg_max_gap * 1e3;
            reg_frames = 0;
            reg_stale = 0;
            reg_max_gap = 0.0;
            prof_want = true;
            perf_frames = 0;
            perf_steps = 0;
            perf_step_ms_acc = 0.0;
            perf_pol_ms_acc = 0.0;
            perf_snap_ms_acc = 0.0;
            perf_t = Instant::now();
        }

        // Sync render poses from the physics snapshot (batch-major layout:
        // env e's colliders start at e·colliders_per_batch; the robot's
        // bodies are the first `mjcf.len()` of them). Pipelined: this takes
        // the copy started last frame and starts the next one, never waiting
        // on the GPU — see `PoseStream`.
        let snap_t0 = Instant::now();
        if snap_block || !pose_stream.pump(&backend, env.body_poses_buffer(), sim_time) {
            pose_stream.poses = env.snapshot().await;
            pose_stream.fresh = true;
            pose_stream.captured_at = sim_time;
        }
        let poses_fresh = pose_stream.fresh;
        let poses_at = pose_stream.captured_at;
        // Motion regularity: how often the screen re-shows an old pose, and
        // the worst sim-time jump between consecutive displayed poses. fps/RT
        // averages hide judder; these two numbers are the judder.
        reg_frames += 1;
        if poses_fresh {
            if reg_last_disp >= 0.0 {
                reg_max_gap = reg_max_gap.max(poses_at - reg_last_disp);
            }
            reg_last_disp = poses_at;
        } else {
            reg_stale += 1;
        }
        let poses = &pose_stream.poses;
        perf_snap_ms_acc += snap_t0.elapsed().as_secs_f32() * 1e3;
        let cpb = poses.len() / n_robots;
        let mut sole_bottom = f32::INFINITY;
        let mut mean_base = Vec3::ZERO;
        // Camera focus: robot 0 specifically. Following the fleet centroid
        // means framing lanes 9 m apart, and that zoom level makes a 0.4 m/s
        // walk read as a crawl — the motion covers too few pixels.
        let mut focus = Vec3::ZERO;
        for (e, body_nodes) in robots.iter_mut().enumerate() {
            let off = offset_of(e);
            for (i, node) in body_nodes.iter_mut().enumerate() {
                let p = &poses[e * cpb + i];
                let pos = Vec3::new(p.translation.x, p.translation.y, p.translation.z) + off;
                let rot =
                    Rot3::from_xyzw(p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w);
                node.set_pose(Pose3::from_parts(pos, rot));
                // Physics-truth foot clearance across the fleet.
                for (p1, p2, r) in &mjcf[i].capsules {
                    for lp in [p1, p2] {
                        let w = pos + rot * Vec3::new(lp.x, lp.y, lp.z);
                        sole_bottom = sole_bottom.min(w.z - r);
                    }
                }
            }
            let (base, brot) = env.base_pose_for(e, &poses);
            // Base height RELATIVE to the local ground surface (terrain can
            // put the ground well above z = 0, especially with slope).
            let ground = if terrain {
                strips[e % 3].height(base[0], base[1])
            } else {
                0.0
            };
            mean_base += Vec3::new(base[0], base[1], base[2]) + off;
            if e == 0 {
                focus = Vec3::new(base[0], base[1], base[2]) + off;
            }
            // Fall / timeout detection from the (already read) render poses —
            // the resident loop has no per-step obs readback to piggyback on.
            let up_z = 1.0 - 2.0 * (brot[0] * brot[0] + brot[1] * brot[1]);
            if fall_cooldown[e] > 0 {
                fall_cooldown[e] -= 1;
            } else if base[2] - ground < FALL_Z || up_z < TILT_COS || ep_steps[e] >= 1000 {
                if base[2] - ground < FALL_Z || up_z < TILT_COS {
                    falls += 1;
                }
                fallen.push(e);
            }
        }
        hud_sole = sole_bottom;
        mean_base /= n_robots as f32;
        // Measured planar speed of the centroid, differentiated ONLY across
        // frames that carried new poses: the readback is pipelined, so a frame
        // may redraw the previous poses, and pairing a two-frame displacement
        // with a one-frame dt reads as double the real speed. Teleport/reset
        // frames are skipped too — a reset yanks the centroid by meters.
        if poses_fresh {
            let dt = poses_at - prev_poses_at;
            if let Some(prev) = prev_mean_base {
                if dt > 1e-3 && fallen.is_empty() {
                    let d = mean_base - prev;
                    let v = (d.x * d.x + d.y * d.y).sqrt() / dt;
                    if v < 3.0 {
                        meas_speed += (v - meas_speed) * 0.1;
                    }
                }
            }
            prev_mean_base = Some(mean_base);
            prev_poses_at = poses_at;
        }
        for &e in &fallen {
            let _ = env.reset_env(e).await;
            env.pin_command_for(e, cmds[e][0], cmds[e][1], cmds[e][2]);
            gobs.reset_env(&backend, e).expect("gobs reset");
            ep_steps[e] = 0;
            fall_cooldown[e] = 10;
        }
        fallen.clear();

        // Camera follows the formation centroid smoothly (z tracks the mean
        // pelvis height minus a hair, so sloped/raised terrain stays framed).
        let target = Vec3::new(focus.x, focus.y, focus.z - 0.2);
        let at = camera.at();
        camera.set_at(at + (target - at) * 0.08);

        #[derive(Clone, Copy)]
        enum CmdChoice {
            All([f32; 3]),
            Presets,
        }
        let mut reset_clicked = false;
        let mut choice: Option<CmdChoice> = None;
        {
            let reset_clicked = &mut reset_clicked;
            let choice = &mut choice;
            let cmd_ui = &mut cmd_ui;
            let ckpt_label: &str = &ckpt_name;
            window.draw_ui(move |ctx| {
                kiss3d::egui::Window::new("zealot G1")
                    .anchor(kiss3d::egui::Align2::LEFT_TOP, [12.0, 12.0])
                    .resizable(false)
                    .collapsible(false)
                    .show(ctx, |ui| {
                        ui.label(format!(
                            "{n_robots}× Unitree G1 walking on the zealot"
                        ));
                        ui.label("training env: nexus GPU physics, real Unitree");
                        ui.label("PD gains, 5 ms dt + substep refresh, 50 Hz.");
                        ui.label(format!("policy: {ckpt_label}"));
                        ui.separator();
                        ui.horizontal(|ui| {
                            let mut preset = |v: [f32; 3]| {
                                *cmd_ui = v;
                                *choice = Some(CmdChoice::All(v));
                            };
                            if ui.button("Stand").clicked() {
                                preset([0.0, 0.0, 0.0]);
                            }
                            if ui.button("Walk").clicked() {
                                preset([0.4, 0.0, 0.0]);
                            }
                            if ui.button("Turn ↺").clicked() {
                                preset([0.2, 0.0, 0.5]);
                            }
                            if ui.button("Turn ↻").clicked() {
                                preset([0.2, 0.0, -0.5]);
                            }
                            if ui.button("Fan out").clicked() {
                                *choice = Some(CmdChoice::Presets);
                            }
                        });
                        // Velocity-command sliders — the policy's actual input
                        // (pinned; training resampler is off in the demo).
                        // Training ranges: vx/vy sampled within ±0.5/±0.3;
                        // beyond that is out-of-distribution territory.
                        ui.separator();
                        ui.label("command (the policy's velocity input):");
                        let mut changed = false;
                        changed |= ui
                            .add(
                                kiss3d::egui::Slider::new(&mut cmd_ui[0], -0.6..=1.0)
                                    .text("forward m/s"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                kiss3d::egui::Slider::new(&mut cmd_ui[1], -0.4..=0.4)
                                    .text("lateral m/s"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                kiss3d::egui::Slider::new(&mut cmd_ui[2], -1.0..=1.0)
                                    .text("yaw rad/s"),
                            )
                            .changed();
                        if changed {
                            *choice = Some(CmdChoice::All(*cmd_ui));
                        }
                        ui.label(format!(
                            "measured speed: {meas_speed:.2} m/s (cmd {:.2})",
                            cmd_ui[0]
                        ));
                        ui.separator();
                        ui.label(format!("falls: {falls}"));
                        ui.label(format!("sole bottom z (physics): {sole_bottom:+.4}"));
                        ui.label(format!(
                            "{hud_fps:.0} sim frames/s — every frame waits for its physics"
                        ));
                        ui.label(format!(
                            "ctrl-step encode: {hud_step_ms:.1} ms   pose fence: {hud_snap_ms:.1} ms/frame"
                        ));
                        ui.label(format!(
                            "sim speed: {:.0}% of real time ({} robots)",
                            hud_realtime * 100.0,
                            n_robots
                        ));
                        ui.separator();
                        if ui.button("Reset").clicked() {
                            *reset_clicked = true;
                        }
                        ui.label("drag: orbit camera   scroll: zoom out");
                        ui.label("tap ground: walk there · keys ↑↓ ±0.2, ←→ turn, space stops");
                    });
            });
        }
        if let Some(c) = choice {
            // Buttons and sliders take over from the autopilot.
            wander = false;
            nav_target = None;
            nav_marker.set_visible(false);
            last_drive_cmd = drive::command();
            for e in 0..n_robots {
                cmds[e] = match c {
                    CmdChoice::All(v) => v,
                    CmdChoice::Presets => preset_cmd(e),
                };
                env.pin_command_for(e, cmds[e][0], cmds[e][1], cmds[e][2]);
                gobs.set_cmd(&backend, e, cmds[e]).expect("cmd");
            }
            // Keep the latched drive command in step with the buttons and
            // sliders, so the next key press bumps from what's on screen.
            drive::sync(cmd_ui);
        }
        if reset_clicked {
            pending_reset = true;
        }
    }
}

#[cfg(test)]
mod ckpt_tests {
    use super::*;

    fn file(spec: &str) -> String {
        match parse_ckpt(spec) {
            CkptRef::File(u) => u,
            CkptRef::Repo(r) => panic!("{spec} parsed as repo {r}, wanted a file"),
        }
    }
    fn repo(spec: &str) -> String {
        match parse_ckpt(spec) {
            CkptRef::Repo(r) => r,
            CkptRef::File(u) => panic!("{spec} parsed as file {u}, wanted a repo"),
        }
    }

    #[test]
    fn handles_and_urls_people_actually_paste() {
        let want = "https://huggingface.co/haixuantao/zealot-g1-locomotion/resolve/main/g1_v24.safetensors";
        // The handle off the model page, with the file named.
        assert_eq!(file("haixuantao/zealot-g1-locomotion/g1_v24.safetensors"), want);
        // The page URL for that file ("blob"), and the raw one ("resolve").
        assert_eq!(
            file("https://huggingface.co/haixuantao/zealot-g1-locomotion/blob/main/g1_v24.safetensors"),
            want
        );
        assert_eq!(file(&format!("{want}?download=true")), want);
        // Bare handle, page URL, folder view, hf.co short domain, trailing
        // slash: all just "this repo".
        for spec in [
            "haixuantao/zealot-g1-locomotion",
            "https://huggingface.co/haixuantao/zealot-g1-locomotion",
            "https://huggingface.co/haixuantao/zealot-g1-locomotion/tree/main",
            "https://hf.co/haixuantao/zealot-g1-locomotion/",
            "hf:haixuantao/zealot-g1-locomotion",
        ] {
            assert_eq!(repo(spec), "haixuantao/zealot-g1-locomotion", "{spec}");
        }
        // A non-Hub URL is taken at its word, and a bare word is a local file.
        assert_eq!(file("https://example.org/p.safetensors"), "https://example.org/p.safetensors");
        assert_eq!(file("g1_walk_v24"), "g1_walk_v24.safetensors");
        // Percent-encoded, the way the picker passes it through the query.
        assert_eq!(repo("haixuantao%2Fzealot-g1-locomotion"), "haixuantao/zealot-g1-locomotion");
    }

    #[test]
    fn revision_and_subfolder_survive() {
        assert_eq!(
            file("https://huggingface.co/o/r/blob/v2.0/ckpts/deep/p.safetensors"),
            "https://huggingface.co/o/r/resolve/v2.0/ckpts/deep/p.safetensors"
        );
        assert_eq!(
            file("o/r/ckpts/deep/p.safetensors"),
            "https://huggingface.co/o/r/resolve/main/ckpts/deep/p.safetensors"
        );
    }

    #[test]
    fn newest_checkpoint_wins() {
        let mut f = vec![
            "g1_v19_iter2740.safetensors".to_string(),
            "g1_v24_iter32780.safetensors".to_string(),
            "g1_v21_iter4560.safetensors".to_string(),
            "g1_v21_iter21k.safetensors".to_string(),
        ];
        newest_first(&mut f);
        assert_eq!(f[0], "g1_v24_iter32780.safetensors");
        // iter 4560 > iter 21 (the "21k" name only carries the digits 21).
        assert_eq!(f[1], "g1_v21_iter4560.safetensors");
        assert_eq!(f[3], "g1_v19_iter2740.safetensors");
    }

    #[test]
    fn file_list_comes_out_of_the_hub_response() {
        let body = r#"{"id":"o/r","siblings":[{"rfilename":"README.md"},
            {"rfilename":"g1_v24.safetensors"},{"rfilename":"g1_v24.onnx"}]}"#;
        assert_eq!(rfilenames(body), ["README.md", "g1_v24.safetensors", "g1_v24.onnx"]);
    }

    #[test]
    fn label_is_the_file_stem() {
        assert_eq!(ckpt_label("o/r/g1_v24_iter32780.safetensors"), "g1_v24_iter32780");
        assert_eq!(
            ckpt_label("https://huggingface.co/o/r/resolve/main/a/b.safetensors"),
            "b"
        );
    }
}
