//! Interactive walking demo of the LeRobot bipedal on nexus GPU physics,
//! driven in realtime by a trained zealot PPO policy. The MJCF model and the
//! `.safetensors` checkpoint are embedded in the binary, so the executable is
//! self-contained (and wasm-buildable, though browsers still reject the
//! nexus-cuda kernels — see below). The website's browser demo is the Unitree
//! G1 built from `../nexus` instead (`website/scripts/build-demos.sh`).
//!
//! ## Browser status (2026-07)
//!
//! The demo compiles to wasm and initializes the WebGPU device, but pipeline
//! creation still fails in Chrome on two nexus-cuda kernel issues the native
//! backends don't have:
//! 1. `gpu_mb_gravity_and_lu` / `gpu_mb_compute_dynamics_pre` bind 11 storage
//!    buffers in one stage; browsers cap `maxStorageBuffersPerShaderStage` at
//!    10 (upstream nexus's versions fit — the fork added frictionloss /
//!    local-mprops / num-multibodies bindings).
//! 2. Several multibody kernels place `workgroupBarrier` under control flow
//!    Tint can't prove uniform (loop bounds read from storage buffers), which
//!    strict browser WGSL validation rejects.
//! Until those kernels are made browser-clean (as upstream nexus's were for
//! nexus.dimforge.com), the wasm build loads but shows an initialization
//! error in the console. Natively the demo works today.
//!
//! (The rapier CPU twin env was tried as an in-browser fallback but currently
//! can't even hold the standing pose — see git history — so the nexus env
//! stays the single source of truth.)
//!
//! Rendering is a kiss3d stick figure mirroring the retired stick-figure renderer: one scene
//! node per MJCF body posed from the GPU snapshot each frame, with link
//! capsules parented statically (parent→child offsets are rigid), spheres at
//! the joints and the foot collision capsules on the feet.
//!
//! Controls: arrows / WASD command forward + turn, Q/E strafe, Space zeroes
//! the command, R resets the episode.
//!
//! Native run:  `cargo run --release --example lerobot_biped_web --features lerobot_biped_web`
//! Smoke check: `cargo run --release --example lerobot_biped_web --features lerobot_biped_web -- --headless-check`

#[path = "../../src/biped/biped_env_nexus.rs"]
mod biped_env_nexus;

use biped_env_nexus::{BipedNexusBatchEnv, parse_mjcf};
use glamx::{Pose3, Rot3, Vec3};
use kiss3d::camera::OrbitCamera3d;
use kiss3d::color::Color;
use kiss3d::event::{Action, Key, WindowEvent};
use kiss3d::scene::SceneNode3d;
use kiss3d::window::Window;
use zealot_env::robots::lerobot_bipedal::NUM_JOINTS;
use zealot_rl::ActorCritic;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

/// The deployed MuJoCo model (copy of `to_real_robot/RL_policy/robot.xml`).
const MJCF_XML: &str = include_str!("assets/robot.xml");
/// Trained velocity-tracking checkpoint (`walking_policy_mac_v4`).
const POLICY_BYTES: &[u8] = include_bytes!("assets/walking_policy.safetensors");

/// Control-step period — matches `VelocityFlatTask` (50 Hz).
const DT: f32 = 0.02;
/// Command clamp ranges (roughly the training command distribution).
const VX_RANGE: (f32, f32) = (-0.3, 0.6);
const VY_RANGE: (f32, f32) = (-0.3, 0.3);
const YAW_RANGE: (f32, f32) = (-0.8, 0.8);

/// Velocity command being tracked, mutated by the keyboard handler.
struct Command {
    vx: f32,
    vy: f32,
    yaw: f32,
}

impl Command {
    fn clamp(&mut self) {
        self.vx = self.vx.clamp(VX_RANGE.0, VX_RANGE.1);
        self.vy = self.vy.clamp(VY_RANGE.0, VY_RANGE.1);
        self.yaw = self.yaw.clamp(YAW_RANGE.0, YAW_RANGE.1);
    }
}

/// Mean-action policy step; returns (done, fell).
async fn policy_step(
    env: &mut BipedNexusBatchEnv,
    ac: &ActorCritic,
    obs: &mut Vec<f32>,
) -> (bool, bool) {
    let mut a = [0.0f32; NUM_JOINTS];
    a.copy_from_slice(&ac.mean(obs)[..NUM_JOINTS]);
    let outs = env.step(&[a]).await;
    let flags = (outs[0].done, outs[0].fell);
    if !outs[0].done {
        obs.clone_from(&outs[0].obs);
    }
    flags
}

/// Native-only sanity rollout: 6 s at vx=0.3, print displacement + falls.
/// Confirms the embedded policy walks on this env without opening a window.
/// An optional path after `--headless-check` evaluates that checkpoint
/// instead of the embedded one (for sweeping candidates).
#[cfg(not(target_arch = "wasm32"))]
async fn headless_check(policy_path: Option<&str>) {
    let cmd = Command {
        vx: 0.3,
        vy: 0.0,
        yaw: 0.0,
    };
    let ac = match policy_path {
        Some(p) => ActorCritic::load(p).expect("load policy checkpoint"),
        None => ActorCritic::load_from_bytes(POLICY_BYTES).expect("load embedded policy"),
    };
    let mut env = BipedNexusBatchEnv::new(MJCF_XML, 1, 1, 0xC0FFEE).await;
    let (mut obs, _) = env.reset_env_to_default_template(0).await;
    env.pin_command_for(0, cmd.vx, cmd.vy, cmd.yaw);
    let steps = (6.0 / DT) as usize;
    let poses = env.snapshot().await;
    let (start, _) = env.base_pose_for(0, &poses);
    let mut falls = 0u32;
    for _ in 0..steps {
        let (done, fell) = policy_step(&mut env, &ac, &mut obs).await;
        if fell {
            falls += 1;
        }
        if done {
            obs = env.reset_env_to_default_template(0).await.0;
            env.pin_command_for(0, cmd.vx, cmd.vy, cmd.yaw);
        }
    }
    let poses = env.snapshot().await;
    let (end, _) = env.base_pose_for(0, &poses);
    let (dx, dy) = (end[0] - start[0], end[1] - start[1]);
    println!("headless-check: displacement = ({dx:+.2}, {dy:+.2}) m over 6 s, falls = {falls}");
}

/// A capsule node spanning `a` → `b` (link-local coordinates), parented under
/// `parent`. kiss3d capsules are Y-aligned and origin-centered, so pose =
/// midpoint + shortest-arc rotation from +Y to the segment direction.
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

#[kiss3d::main]
pub async fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    #[cfg(not(target_arch = "wasm32"))]
    {
        let args: Vec<String> = std::env::args().collect();
        if let Some(i) = args.iter().position(|a| a == "--headless-check") {
            headless_check(args.get(i + 1).map(String::as_str)).await;
            return;
        }
    }

    let mut window = Window::new_with_size("zealot biped — nexus GPU physics", 1200, 900).await;
    window.set_background_color(Color::new(0.051, 0.169, 0.180, 1.0));

    // Z-up orbit camera (the MJCF scene is Z-up), looking at the spawn point.
    let mut camera = OrbitCamera3d::new(Vec3::new(1.6, -1.6, 1.0), Vec3::new(0.0, 0.0, 0.5));
    camera.set_up_axis(Vec3::Z);

    let mut scene = SceneNode3d::empty();
    scene.add_directional_light(Vec3::new(1.0, -2.0, -3.0));

    // Ground slab, top face at z = 0, with a stripe pattern for motion cues.
    let mut ground = scene.add_cube(60.0, 60.0, 0.1);
    ground.set_position(Vec3::new(0.0, 0.0, -0.05));
    ground.set_color(Color::new(0.13, 0.30, 0.31, 1.0));
    for i in -14..=14 {
        let mut stripe = scene.add_cube(0.02, 60.0, 0.002);
        stripe.set_position(Vec3::new(i as f32, 0.0, 0.001));
        stripe.set_color(Color::new(0.21, 0.42, 0.44, 1.0));
        let mut stripe = scene.add_cube(60.0, 0.02, 0.002);
        stripe.set_position(Vec3::new(0.0, i as f32, 0.001));
        stripe.set_color(Color::new(0.21, 0.42, 0.44, 1.0));
    }

    // Robot: one group node per MJCF body, posed from the GPU snapshot each
    // frame. Static children: a sphere at the body origin, a capsule to each
    // child body origin (rigid offset), and the foot collision capsules.
    let mjcf = parse_mjcf(MJCF_XML);
    let body_color = Color::new(0.75, 0.78, 0.80, 1.0);
    let joint_color = Color::new(0.208, 0.757, 0.804, 1.0);
    let foot_color = Color::new(0.925, 0.702, 0.208, 1.0);
    let mut body_nodes: Vec<SceneNode3d> = Vec::with_capacity(mjcf.len());
    for body in &mjcf {
        let mut group = scene.add_group();
        let mut joint = group.add_sphere(0.028);
        joint.set_color(joint_color);
        for (p1, p2, r) in &body.capsules {
            let a = Vec3::new(p1.x, p1.y, p1.z);
            let b = Vec3::new(p2.x, p2.y, p2.z);
            add_segment(&mut group, a, b, *r, foot_color);
        }
        body_nodes.push(group);
    }
    for body in &mjcf {
        if let Some(p) = body.parent {
            // `local_pos` is the child-body origin in the parent frame (MJCF),
            // in the physics stack's own math types — convert by component.
            let child_origin = Vec3::new(body.local_pos.x, body.local_pos.y, body.local_pos.z);
            let mut parent_node = body_nodes[p].clone();
            add_segment(&mut parent_node, Vec3::ZERO, child_origin, 0.02, body_color);
        }
    }

    // Policy + env (1 env, template 0 = DR off).
    let ac = ActorCritic::load_from_bytes(POLICY_BYTES).expect("load embedded policy");
    let mut env = BipedNexusBatchEnv::new(MJCF_XML, 1, 1, 0xC0FFEE).await;
    let (mut obs, _) = env.reset_env_to_default_template(0).await;
    let mut cmd = Command {
        vx: 0.3,
        vy: 0.0,
        yaw: 0.0,
    };
    env.pin_command_for(0, cmd.vx, cmd.vy, cmd.yaw);

    // Fixed-dt control loop paced to the wall clock (render rate varies with
    // the display), capped so a slow frame can't spiral.
    let t0 = Instant::now();
    let mut sim_time = 0.0f32;
    let mut falls: u32 = 0;

    while window.render_3d(&mut scene, &mut camera).await {
        // Keyboard → command.
        let mut reset_requested = false;
        for event in window.events().iter() {
            if let WindowEvent::Key(key, Action::Press, _) = event.value {
                match key {
                    Key::Up | Key::W => cmd.vx += 0.1,
                    Key::Down | Key::S => cmd.vx -= 0.1,
                    Key::Left | Key::A => cmd.yaw += 0.2,
                    Key::Right | Key::D => cmd.yaw -= 0.2,
                    Key::Q => cmd.vy += 0.1,
                    Key::E => cmd.vy -= 0.1,
                    Key::Space => {
                        cmd.vx = 0.0;
                        cmd.vy = 0.0;
                        cmd.yaw = 0.0;
                    }
                    Key::R => reset_requested = true,
                    _ => {}
                }
                cmd.clamp();
                env.pin_command_for(0, cmd.vx, cmd.vy, cmd.yaw);
            }
        }

        if reset_requested {
            obs = env.reset_env_to_default_template(0).await.0;
            env.pin_command_for(0, cmd.vx, cmd.vy, cmd.yaw);
        }

        // Advance physics to wall-clock time, at most 4 control steps a frame.
        let wall = t0.elapsed().as_secs_f32();
        let mut steps_this_frame = 0;
        while sim_time < wall && steps_this_frame < 4 {
            let (done, fell) = policy_step(&mut env, &ac, &mut obs).await;
            if fell {
                falls += 1;
            }
            if done {
                obs = env.reset_env_to_default_template(0).await.0;
                env.pin_command_for(0, cmd.vx, cmd.vy, cmd.yaw);
            }
            sim_time += DT;
            steps_this_frame += 1;
        }
        if sim_time < wall - 0.25 {
            // Fell behind (tab hidden, slow frame): drop the debt instead of
            // fast-forwarding through it.
            sim_time = wall;
        }

        // Sync render poses from the physics snapshot.
        let poses = env.snapshot().await;
        for (i, node) in body_nodes.iter_mut().enumerate() {
            let p = &poses[i];
            node.set_pose(Pose3::from_parts(
                Vec3::new(p.translation.x, p.translation.y, p.translation.z),
                Rot3::from_xyzw(p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w),
            ));
        }

        // Camera follows the torso smoothly (keeps user orbit offset).
        let (base, _) = env.base_pose_for(0, &poses);
        let target = Vec3::new(base[0], base[1], 0.45);
        let at = camera.at();
        camera.set_at(at + (target - at) * 0.08);

        // HUD.
        let (vx, vy, yaw) = (cmd.vx, cmd.vy, cmd.yaw);
        window.draw_ui(move |ctx| {
            kiss3d::egui::Window::new("zealot biped")
                .anchor(kiss3d::egui::Align2::LEFT_TOP, [12.0, 12.0])
                .resizable(false)
                .collapsible(false)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "command  vx {vx:+.1} m/s  vy {vy:+.1} m/s  yaw {yaw:+.1} rad/s"
                    ));
                    ui.label(format!("falls: {falls}"));
                    ui.separator();
                    ui.label("arrows/WASD: walk + turn   Q/E: strafe");
                    ui.label("space: stop   R: reset   drag: orbit camera");
                });
        });
    }
}
