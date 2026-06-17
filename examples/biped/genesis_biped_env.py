#!/usr/bin/env python3
"""Genesis biped env mirroring the nexus/rapier LeRobot-bipedal sim, for the
Genesis-vs-nexus benchmark (throughput + fidelity).

Replicates the pinned sim config (see biped_genesis_common.py): MJCF model,
physics dt 0.005, decimation 4 (50 Hz control), per-joint PD (kp/kd) + effort
cap + armature, batched over N parallel envs. Obs is the same 43-dim layout as
velocity_flat.rs; reward is a structural stand-in (same terms/cost — exact
weights aren't needed for a throughput comparison; fidelity is gated separately
by cross_engine_eval_genesis.py).

gs.init() is process-global; construct exactly one env per process.
"""
import os
import re
import tempfile

import numpy as np
import torch
import genesis as gs

import biped_genesis_common as C


def _patched_mjcf():
    """Copy the robot MJCF with meshdir pointed at a temp dir of symlinked STLs;
    stub any referenced-but-missing mesh with a tiny box (mirrors
    cross_engine_eval.py / render_biped_mujoco.py). Returns the patched XML path."""
    import trimesh

    tmp = tempfile.mkdtemp(prefix="genesis_biped_")
    md = os.path.join(tmp, "meshes")
    os.makedirs(md)
    for fn in os.listdir(C.ASSETS):
        if fn.endswith(".stl"):
            os.symlink(os.path.join(C.ASSETS, fn), os.path.join(md, fn))
    xml = open(C.ROBOT_XML).read()
    for ref in set(re.findall(r'file="([^"]+\.stl)"', xml)):
        base = os.path.basename(ref)
        if not os.path.exists(os.path.join(md, base)):
            trimesh.creation.box((0.02, 0.02, 0.02)).export(os.path.join(md, base))
    xml = xml.replace('meshdir="assets"', f'meshdir="{md}"')
    out = os.path.join(tmp, "robot.xml")
    open(out, "w").write(xml)
    return out


_INITED = False


def _ensure_init(backend):
    global _INITED
    if not _INITED:
        gs.init(backend=backend)
        _INITED = True


class GenesisBipedEnv:
    def __init__(self, n_envs, on_gpu=True):
        self.n = n_envs
        self.device = "cuda" if on_gpu else "cpu"
        _ensure_init(gs.gpu if on_gpu else gs.cpu)

        rigid = gs.options.RigidOptions(dt=C.PHYS_DT, iterations=C.SOLVER_ITERS)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=C.PHYS_DT, gravity=C.GRAVITY),
            rigid_options=rigid,
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=_patched_mjcf()))
        self.scene.build(n_envs=n_envs)

        # Map canonical joint names -> local DOF indices.
        name_to_dof = {}
        for j in self.robot.joints:
            didx = getattr(j, "dofs_idx_local", None)
            if didx is not None and len(didx) == 1:
                name_to_dof[j.name] = int(didx[0])
        self.dof_idx = [name_to_dof[n] for n in C.JOINT_NAMES]

        dev = self.device
        self.kp = torch.tensor(C.KP, device=dev)
        self.kd = torch.tensor(C.KD, device=dev)
        self.effort = torch.tensor(C.EFFORT, device=dev)
        self.armature = torch.tensor(C.ARMATURE, device=dev)
        self.act_scale = torch.tensor(C.ACTION_SCALE, device=dev)
        self.default_pos = torch.tensor(C.DEFAULT_POS, device=dev)

        # PD + material params on the leg DOFs.
        self.robot.set_dofs_kp(self.kp, self.dof_idx)
        self.robot.set_dofs_kv(self.kd, self.dof_idx)
        self.robot.set_dofs_force_range(-self.effort, self.effort, self.dof_idx)
        self.robot.set_dofs_armature(self.armature, self.dof_idx)

        self.last_action = torch.zeros(n_envs, C.ACT_DIM, device=dev)
        self.command = torch.zeros(n_envs, 4, device=dev)
        self.reset()

    def reset(self):
        n, dev = self.n, self.device
        # Base at spawn height, identity orientation; legs at default pose.
        base_pos = torch.zeros(n, 3, device=dev)
        base_pos[:, 2] = C.SPAWN_Z
        self.robot.set_pos(base_pos)
        quat = torch.zeros(n, 4, device=dev)
        quat[:, 0] = 1.0  # wxyz identity
        self.robot.set_quat(quat)
        q = self.default_pos.unsqueeze(0).repeat(n, 1)
        self.robot.set_dofs_position(q, self.dof_idx)
        self.robot.zero_all_dofs_velocity()
        self.last_action.zero_()

    # --- helpers --------------------------------------------------------
    def _projected_gravity(self, quat_wxyz):
        # gravity dir [0,0,-1] rotated into base frame by quat^{-1}
        w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        # R^T @ [0,0,-1]  (inverse rotation of -Z)
        gx = -2.0 * (x * z - w * y)
        gy = -2.0 * (y * z + w * x)
        gz = -(1.0 - 2.0 * (x * x + y * y))
        return torch.stack([gx, gy, gz], dim=1)

    def get_obs(self):
        q = self.robot.get_dofs_position(self.dof_idx)
        qd = self.robot.get_dofs_velocity(self.dof_idx)
        quat = self.robot.get_quat()  # [N,4] wxyz
        pg = self._projected_gravity(quat)
        # [last_action(12), command(4), joint_pos_rel(12), joint_vel(12), proj_gravity(3)] = 43
        return torch.cat([self.last_action, self.command, q - self.default_pos, qd, pg], dim=1)

    def reward(self, obs):
        # Structural stand-in (matches per-step cost, not exact weights). Mirrors
        # the dominant velocity_flat.rs terms so the rollout's reward compute is
        # representative. Fidelity is verified separately.
        base_vel = self.robot.get_vel()  # [N,3] linear
        ang = self.robot.get_ang() if hasattr(self.robot, "get_ang") else torch.zeros_like(base_vel)
        track_lin = torch.exp(-((base_vel[:, :2] - self.command[:, :2]) ** 2).sum(1) / 0.25)
        track_ang = torch.exp(-((ang[:, 2] - self.command[:, 2]) ** 2) / 0.25)
        upright = obs[:, -1]  # proj_gravity_z ~ -1 upright
        return 5.0 * track_lin + 5.0 * track_ang + 0.5 * upright

    def step(self, action):
        # action in ~[-1,1]; q_target = default + scale*action
        targets = self.default_pos + self.act_scale * action
        self.robot.control_dofs_position(targets, self.dof_idx)
        for _ in range(C.DECIMATION):
            self.scene.step()
        self.last_action = action
        obs = self.get_obs()
        rew = self.reward(obs)
        torso_z = self.robot.get_pos()[:, 2]
        return obs, rew, torso_z


if __name__ == "__main__":
    # Self-test: small batch, zero action, confirm it loads + steps + stays up.
    import sys
    on_gpu = "--cpu" not in sys.argv
    n = 4
    env = GenesisBipedEnv(n, on_gpu=on_gpu)
    print(f"built {n} envs, dof_idx={env.dof_idx}")
    act = torch.zeros(n, C.ACT_DIM, device=env.device)
    for s in range(50):
        obs, rew, z = env.step(act)
        if s % 10 == 0:
            print(f"step {s:3d}  obs{tuple(obs.shape)}  torso_z mean={z.mean().item():.3f}  rew={rew.mean().item():.3f}")
    print("OK")
