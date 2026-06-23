#!/usr/bin/env python3
"""Render the CLOSED-LOOP MuJoCo rollout of a zealot policy to an mp4.

Reuses the validated policy + MuJoCo model from `sim2sim_xval.py` (same net,
same obs construction, same lag-2 / warmup conventions), but instead of stopping
at the first fall it RE-INITIALISES to the start pose on each fall — so the clip
shows several independent attempts, like the nexus render. The robot.xml visual
STL meshes are missing on disk, so this renders the physics model's capsule
collision geoms (functionally identical motion, just blockier looking).

Headless GL via EGL (the box has a GPU); falls back to osmesa.

Usage:
  python3 examples/biped/render_sim2sim_mujoco.py [rollout.json] [policy.safetensors] [out.mp4] [steps]
defaults: /tmp/biped_xval.json  /tmp/biped_policy_v7.safetensors  /tmp/biped_sim2sim_mujoco.mp4  500
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU rendering

import json
import numpy as np
import mujoco
import imageio.v2 as imageio

# Pull the validated pieces from the cross-val harness.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim_xval as X  # noqa: E402

ROLLOUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/biped_xval.json"
POLICY = sys.argv[2] if len(sys.argv) > 2 else "/tmp/biped_policy_v7.safetensors"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/biped_sim2sim_mujoco.mp4"
N_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 500
W, H = 640, 480  # within MuJoCo's default offscreen framebuffer
FALL_Z = X.FALL_Z
# Velocity command [vx,vy,yaw] for the closed loop. Override with BIPED_CMD,
# e.g. BIPED_CMD="0,0,0" to test a stand-trained policy fairly.
_cmd = os.environ.get("BIPED_CMD", "0.4,0,0").split(",")
CMD = np.array([float(_cmd[0]), float(_cmd[1]), float(_cmd[2]), 0.0])


def main():
    gt = json.load(open(ROLLOUT))
    policy = X.Policy(POLICY)
    jnames = gt["joint_names"]
    base = np.array(gt["base"], dtype=np.float64)
    joints = np.array(gt["joints"], dtype=np.float64)

    model = X.build_mujoco_model(jnames)
    model.opt.timestep = X.PHYS_DT
    data = mujoco.MjData(model)

    free_q = model.joint(X.FREEJOINT).qposadr[0]
    hinge_q = {n: int(model.joint(n).qposadr[0]) for n in jnames}
    hinge_d = {n: int(model.joint(n).dofadr[0]) for n in jnames}
    act_id = {n: int(model.actuator(f"act_{n}").id) for n in jnames}
    scale = np.array([X.pick_by_prefix(X.ACTION_SCALE, n) for n in jnames])

    # Per-reset randomization so each attempt differs (the policy + physics are
    # both deterministic, so a fixed start pose would make every reset bit-
    # identical). Seeded for reproducibility; mirrors the nexus render's
    # randomized resets (spawn perturbation) in spirit.
    SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    rng = np.random.default_rng(SEED)

    def reset_random():
        data.qpos[free_q:free_q + 3] = base[0, :3]
        # small random base yaw + slight roll/pitch tilt around the start orient
        qx, qy, qz, qw = base[0, 3:7]
        dy, dr, dp = rng.uniform(-0.25, 0.25), rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)
        cy, sy = np.cos(dy / 2), np.sin(dy / 2)
        cr, sr = np.cos(dr / 2), np.sin(dr / 2)
        cp, sp = np.cos(dp / 2), np.sin(dp / 2)
        # compose start quat (xyzw) with a small random rotation (yaw*roll*pitch)
        dq = np.array([sr * cp * cy + cr * sp * sy,   # x
                       cr * sp * cy - sr * cp * sy,   # y
                       cr * cp * sy + sr * sp * cy,   # z
                       cr * cp * cy - sr * sp * sy])  # w
        sq = np.array([qx, qy, qz, qw])
        sx, sy_, sz, sw = sq
        dx, dyq, dz, dw = dq
        # quaternion product start * delta (xyzw)
        nq = np.array([
            sw * dx + sx * dw + sy_ * dz - sz * dyq,
            sw * dyq - sx * dz + sy_ * dw + sz * dx,
            sw * dz + sx * dyq - sy_ * dx + sz * dw,
            sw * dw - sx * dx - sy_ * dyq - sz * dz,
        ])
        data.qpos[free_q + 3:free_q + 7] = [nq[3], nq[0], nq[1], nq[2]]  # WXYZ
        for k, n in enumerate(jnames):
            data.qpos[hinge_q[n]] = joints[0, k] + rng.uniform(-0.08, 0.08)
        data.qvel[:] = 0.0
        # small random joint velocities to diversify the divergence
        for n in jnames:
            data.qvel[hinge_d[n]] = rng.normal(0.0, 0.3)
        mujoco.mj_forward(model, data)

    reset_random()
    act_hist = [np.zeros(12), np.zeros(12)]  # [t-2, t-1]
    since_reset = 0
    traj = []   # (base_xyzw7, joints12) per control step
    resets = []
    falls = 0

    prev_qj = None
    for t in range(N_STEPS):
        last_action = act_hist[0] if since_reset >= 2 else np.zeros(12)
        qj = np.array([data.qpos[hinge_q[n]] for n in jnames])
        if since_reset < 2:
            jvel = np.zeros(12)
        elif os.environ.get("BIPED_JVEL_FD"):
            # Finite-diff joint vel over the control step — MATCHES how the nexus
            # env builds the obs joint_vel (q_now-q_prev)/control_dt). Default path
            # uses instantaneous data.qvel, which the policy never trained on.
            jvel = (qj - prev_qj) / X.CONTROL_DT
        else:
            jvel = np.array([data.qvel[hinge_d[n]] for n in jnames])
        prev_qj = qj
        cmd = CMD  # command not zeroed mid-run; warmup only zeros action/vel
        qw_, qx_, qy_, qz_ = data.qpos[free_q + 3:free_q + 7]
        # Gait clock: held at 0 for the fresh step then advances (matches the env's
        # 1-step lag) — see sim2sim_xval phase3.
        phase = (max(0, since_reset - 1) * X.CONTROL_DT / X.GAIT_PERIOD) % 1.0
        obs = np.zeros(45)
        obs[0:12] = last_action
        obs[12:16] = cmd
        obs[16:28] = qj
        obs[28:40] = jvel
        obs[40:43] = X.projected_gravity((qx_, qy_, qz_, qw_))
        obs[43] = np.sin(2.0 * np.pi * phase)
        obs[44] = np.cos(2.0 * np.pi * phase)
        action = policy.act(obs)
        act_hist = [act_hist[1], action.copy()]
        target = scale * action  # default_pos = 0
        for k, n in enumerate(jnames):
            data.ctrl[act_id[n]] = target[k]
        for _ in range(X.DECIMATION):
            mujoco.mj_step(model, data)
        since_reset += 1

        p = data.qpos[free_q:free_q + 3].copy()
        wq = data.qpos[free_q + 3:free_q + 7].copy()  # WXYZ
        traj.append((np.array([p[0], p[1], p[2], wq[1], wq[2], wq[3], wq[0]]),
                     np.array([data.qpos[hinge_q[n]] for n in jnames])))

        if data.qpos[free_q + 2] < FALL_Z:
            resets.append(t)
            falls += 1
            reset_random()
            act_hist = [np.zeros(12), np.zeros(12)]
            since_reset = 0

    print(f"rollout: {len(traj)} control steps, {falls} falls "
          f"(reset every ~{len(traj)/max(falls,1):.1f} steps)")

    # --- render the recorded qpos trajectory through the SAME faithful model ---
    # (build_mujoco_model includes the visual meshes, so physics + render share it)
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
    model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    vdata = mujoco.MjData(model)
    vfree_q, vhinge_q = free_q, hinge_q

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance = 1.8
    cam.elevation = -8.0
    cam.azimuth = 100.0
    cam.lookat[:] = [0.0, 0.0, 0.40]
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)  # default groups: visual meshes shown, collisions hidden

    fps = int(round(1.0 / gt["dt"]))
    import tempfile
    import subprocess
    fdir = tempfile.mkdtemp()
    for i, (b7, jq) in enumerate(traj):
        vdata.qpos[vfree_q:vfree_q + 3] = b7[:3]
        vdata.qpos[vfree_q + 3:vfree_q + 7] = [b7[6], b7[3], b7[4], b7[5]]  # XYZW->WXYZ
        for k, n in enumerate(jnames):
            vdata.qpos[vhinge_q[n]] = jq[k]
        mujoco.mj_forward(model, vdata)
        renderer.update_scene(vdata, camera=cam, scene_option=opt)
        imageio.imwrite(os.path.join(fdir, f"f{i:05d}.png"), renderer.render())
        if i % 100 == 0:
            print(f"  rendered {i}/{len(traj)}")
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{fdir}/f%05d.png",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUT],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {OUT}  ({len(traj)/fps:.1f}s @ {fps}fps)")


if __name__ == "__main__":
    main()
