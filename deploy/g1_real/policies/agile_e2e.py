"""WBC-AGILE end-to-end ONNX velocity policy (Unitree G1 29-DOF).

Wraps `unitree_g1_velocity_e2e.onnx` + its I/O-descriptor YAML (exported by
NVIDIA's WBC-AGILE, Apache-2.0). The graph is self-contained: observation
normalization, per-term 5-frame histories, default offsets and action scaling
are all baked in. The runner only feeds RAW state each 20 ms tick:

  root_link_quat_w   pelvis orientation (w, x, y, z)   <- IMU
  root_ang_vel_b     pelvis angular velocity (body)    <- gyro
  velocity_commands  [vx, vy, yaw_rate]                <- joystick
  joint_pos/joint_vel  29 joints in the DESCRIPTOR's order (Isaac BFS
                       interleaved order, from the yaml element_names)

plus the recurrent state tensors (`is_state: true`), which the model updates
itself — each `*_out` output is fed back to its matching input next tick
(`pipeline.feedback_connections`). States start at zeros on reset.

Outputs: absolute joint position targets AND per-joint kp/kd for the 14
controlled joints (12 legs + waist_roll + waist_pitch), mapped back to
hardware motor indices by name.
"""
import os

import numpy as np
import yaml

from common.g1_constants import G1_JOINT_LIMITS, MOTOR_INDEX
from . import BasePolicy, RobotObs


class AgileE2EPolicy(BasePolicy):
    def __init__(self, cfg: dict, base_dir: str):
        import onnxruntime as ort

        onnx_path = cfg["path"]
        if not os.path.isabs(onnx_path):
            onnx_path = os.path.join(base_dir, onnx_path)
        desc_path = cfg.get("descriptor", os.path.splitext(onnx_path)[0] + ".yaml")
        if not os.path.isabs(desc_path):
            desc_path = os.path.join(base_dir, desc_path)
        with open(desc_path) as f:
            desc = yaml.safe_load(f)

        (model_name, model), = desc["models"].items()
        self.inputs = {i["name"]: i for i in model["inputs"]}
        self.outputs = {o["name"]: o for o in model["outputs"]}

        # joint_pos/joint_vel element order -> hardware motor indices.
        joint_names = self.inputs["joint_pos"]["element_names"][0]
        self.in_motor_idx = np.array([MOTOR_INDEX[n] for n in joint_names])

        # 14 controlled joints, output order -> hardware motor indices + limits.
        out_names = self.outputs["action_joint_pos"]["element_names"][0]
        self.out_motor_idx = np.array([MOTOR_INDEX[n] for n in out_names])
        limits = np.array([G1_JOINT_LIMITS[n] for n in out_names])
        self.lo, self.hi = limits[:, 0], limits[:, 1]

        # Recurrent state plumbing: output name -> input name it feeds.
        self.feedback = {}
        for out_name, in_names in desc["pipeline"]["feedback_connections"].items():
            self.feedback[out_name.split("/", 1)[1]] = [
                n.split("/", 1)[1] for n in in_names]

        self.state_inputs = [n for n, i in self.inputs.items() if i.get("is_state")]
        self.control_dt = float(desc["semantic"]["scene"]["dt"])
        self.kp_scale = float(cfg.get("kp_scale", 1.0))
        self.kd_scale = float(cfg.get("kd_scale", 1.0))

        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])
        onnx_inputs = {i.name for i in self.session.get_inputs()}
        missing = set(self.inputs) - onnx_inputs
        if missing:
            raise ValueError(f"descriptor inputs missing from ONNX graph: {missing}")

        print(f"[agile_e2e] {model_name}: {len(joint_names)} joints in, "
              f"{len(out_names)} controlled, dt {self.control_dt}s")
        self.reset()

    def reset(self):
        self.state = {
            name: np.zeros(self.inputs[name]["shape"], dtype=np.float32)
            for name in self.state_inputs
        }

    def step(self, obs: RobotObs, cmd: np.ndarray) -> dict:
        feed = {
            "root_link_quat_w": obs.quat.reshape(1, 4).astype(np.float32),
            "root_ang_vel_b": obs.gyro.reshape(1, 3).astype(np.float32),
            "velocity_commands": np.asarray(cmd[:3]).reshape(1, 3).astype(np.float32),
            "joint_pos": obs.q[self.in_motor_idx].reshape(1, -1).astype(np.float32),
            "joint_vel": obs.dq[self.in_motor_idx].reshape(1, -1).astype(np.float32),
        }
        feed.update(self.state)

        names = list(self.outputs)
        values = dict(zip(names, self.session.run(names, feed)))

        for out_name, in_names in self.feedback.items():
            for in_name in in_names:
                self.state[in_name] = values[out_name]

        target = np.clip(values["action_joint_pos"][0], self.lo, self.hi)
        kp = values["action_joint_pos_kp_gains"][0] * self.kp_scale
        kd = values["action_joint_pos_kd_gains"][0] * self.kd_scale

        return {int(m): (float(target[i]), float(kp[i]), float(kd[i]))
                for i, m in enumerate(self.out_motor_idx)}
