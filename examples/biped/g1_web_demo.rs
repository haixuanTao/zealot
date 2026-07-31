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
}

use biped_env_nexus::{BipedNexusBatchEnv, parse_mjcf};
use glamx::{Pose3, Rot3, Vec3};
use kiss3d::camera::OrbitCamera3d;
use kiss3d::color::Color;
use kiss3d::scene::SceneNode3d;
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

/// The v24 walking checkpoint (2026-07-31, iter 34k): ActorCritic weights +
/// Welford normalizer stats. First policy with the GYRO in its observation —
/// 48-dim frames instead of 45 — so it needs the matching env/shader obs.
const POLICY_BIN: &[u8] = include_bytes!("assets/g1_walk_v24.safetensors");

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
    pub fn command() -> Option<[f32; 3]> {
        None
    }
}

/// Env knobs the demo needs regardless of target. `std::env::set_var` PANICS
/// on wasm32-unknown-unknown, so these go through programmatic overrides. Must
/// run BEFORE the env is constructed. Decimation 8 → 2.5 ms sim dt — the
/// passive stand needs the finer timestep. (Spawn DR is a non-issue: the demo
/// resets into template 0, which is always the DR-off template.)
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
        ("NEXUS_FIXED_GRID", "1"),
        // Hold the arms at the natural stand pose (joint zero = Unitree's
        // elbows-bent CAD zero — the "zombie arms").
        ("BIPED_HELD_POSE", "natural"),
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

    // Env: N envs, template 0 = DR off. The released v7 policy drives the
    // legs; waist/arms are PD-held by the env at the AGILE holding gains.
    let ac = ActorCritic::load_from_bytes(POLICY_BIN).expect("policy checkpoint");
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
        gait_period: 0.7,
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
    let mut prev_frame_t = Instant::now();
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

    // Safari and Firefox run the physics correctly but the GPU policy path
    // returns wrong actions there (measured: healthy obs magnitudes, actions
    // that never produce motion), so the robot just stands. Until that kernel
    // is fixed, those browsers fall back to stepping the env normally and
    // running the tiny MLP on the CPU — same policy, same physics, a few
    // hundred microseconds a step for a handful of robots. Force either way
    // with ?policy=cpu / ?policy=gpu.
    let cpu_policy = {
        #[cfg(target_arch = "wasm32")]
        {
            let q = web_sys::window()
                .and_then(|w| w.location().search().ok())
                .unwrap_or_default();
            if q.contains("policy=cpu") {
                true
            } else if q.contains("policy=gpu") {
                false
            } else {
                let ua = web_sys::window()
                    .map(|w| w.navigator().user_agent().unwrap_or_default())
                    .unwrap_or_default();
                let chromium = ua.contains("Chrome") || ua.contains("Chromium") || ua.contains("Edg");
                !chromium
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    };
    // Actions for the CPU path, carried across control steps.
    let mut cpu_actions: Vec<[f32; NUM_JOINTS]> = vec![[0.0; NUM_JOINTS]; n_robots];

    drive::install();
    // Start the latched command at what the demo is already running, so the
    // first tap bumps up from the visible value instead of from zero.
    drive::sync(cmd_ui);

    while window.render_3d(&mut scene, &mut camera).await {
        // Driving: each key press bumped the latched command, so apply it
        // whenever it differs from what the robots are already running.
        if let Some(c) = drive::command() {
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

        // Advance physics to wall-clock time, at most 4 control steps a frame.
        let wall = t0.elapsed().as_secs_f32();
        let mut steps_this_frame = 0;
        while sim_time < wall && steps_this_frame < 4 {
            // GPU-resident control step: obs kernel → policy GEMMs → commit
            // (targets + action ring) → motor scatter → decimation substeps.
            // Encode + submit only; the per-frame render snapshot below is
            // the sole GPU→CPU fence.
            let step_t0 = Instant::now();
            // Single-submit control step: obs assembly, policy GEMMs, action
            // commit and motor scatter all share ONE command buffer (each
            // submit is a wasm→JS→browser crossing).
            if cpu_policy {
                // Classic path: step the env (which returns the stacked obs)
                // and run the actor on the CPU for the next step's actions.
                let outs = env.step(&cpu_actions).await;
                for (e, out) in outs.iter().enumerate() {
                    let a = ac.mean(&out.obs);
                    for k in 0..NUM_JOINTS {
                        cpu_actions[e][k] = a[k];
                    }
                }
            } else {
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
                env.step_physics_only();
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
            sim_time = wall;
            hud_realtime = 0.0; // fell behind and clamped this frame
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
            // and OUTPUT (actions), read back once a second. This is what
            // separates "physics broken" from "policy broken" across
            // browsers — a zero output with a healthy input means the GEMM
            // path miscompiled, not the sim.
            #[cfg(target_arch = "wasm32")]
            {
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
                doc.set_title(&format!(
                    "z| falls={} sole={:+.3} spd={:.2} cmd={:.2} rt={:.0}% fps={:.0}                      |in|={:.4} |act|={:.4} nan={}",
                    falls,
                    hud_sole,
                    meas_speed,
                    cmd_ui[0],
                    hud_realtime * 100.0,
                    hud_fps,
                    dbg_in,
                    dbg_out,
                    dbg_nan,
                ));
            }
            perf_frames = 0;
            perf_steps = 0;
            perf_step_ms_acc = 0.0;
            perf_pol_ms_acc = 0.0;
            perf_snap_ms_acc = 0.0;
            perf_t = Instant::now();
        }

        // Sync render poses from the physics snapshot (batch-major layout:
        // env e's colliders start at e·colliders_per_batch; the robot's
        // bodies are the first `mjcf.len()` of them).
        let snap_t0 = Instant::now();
        let poses = env.snapshot().await;
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
        // Measured planar speed of the centroid (skip teleport/reset frames —
        // a reset yanks the centroid by meters in one frame).
        {
            let now = Instant::now();
            let dt = (now - prev_frame_t).as_secs_f32();
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
            prev_frame_t = now;
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
            window.draw_ui(move |ctx| {
                kiss3d::egui::Window::new("zealot G1")
                    .anchor(kiss3d::egui::Align2::LEFT_TOP, [12.0, 12.0])
                    .resizable(false)
                    .collapsible(false)
                    .show(ctx, |ui| {
                        ui.label(format!(
                            "{n_robots}× Unitree G1 walking with the v24"
                        ));
                        ui.label("policy (v24, gyro obs) on the zealot training");
                        ui.label("env: nexus GPU physics, real Unitree PD gains,");
                        ui.label("5 ms dt + substep refresh, 50 Hz control.");
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
                            "render: {hud_fps:.0} fps   ctrl-step encode (obs+policy+physics): {hud_step_ms:.1} ms"
                        ));
                        ui.label(format!("pose snapshot: {hud_snap_ms:.1} ms/frame"));
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
                        ui.label("drive: tap ↑↓ ±0.2, ←→ turn, space stops");
                    });
            });
        }
        if let Some(c) = choice {
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
