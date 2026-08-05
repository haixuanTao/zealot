# G1 real-hardware deployment

Runs a trained locomotion policy on a physical Unitree G1 (29-DOF EDU) over
the low-level DDS interface (`rt/lowcmd` / `rt/lowstate`, hg message set,
PR ankle mode), at 50 Hz PD position control.

Two policies are wired up:

| config | policy | source |
|---|---|---|
| `configs/g1_agile_e2e.yaml` | **WBC-AGILE pretrained velocity policy** — `assets/unitree_g1_velocity_e2e.onnx` (+ I/O descriptor yaml). End-to-end graph: normalization, 5-frame histories, action scaling baked in; outputs 14 joint targets **and per-joint kp/kd** (12 legs + waist roll/pitch). NVIDIA-validated sim2real. | WBC-AGILE `agile/data/policy/velocity_g1` (Apache-2.0) |
| `configs/g1_zealot.yaml` | **zealot-trained policy** — `checkpoints/g1_zealot_v17.safetensors` (nexus GPU training, obs45×5 → 12 leg actions). Obs conventions ported 1:1 from `examples/biped/sim2sim_g1_mujoco.py` (LAG-2 last action, finite-diff joint velocity, 0.7 s gait clock, Welford normalizer). | this repo (`v17_eval` from champagne; `g1_zealot_v14.safetensors` also included) |

**Start with the AGILE policy.** It is the proven-on-hardware one; use it to
validate the whole comms/PD chain before trying the zealot checkpoint, which
so far has only been validated in sim (nexus) and sim2sim (MuJoCo).

## Install (robot-side PC, Linux recommended)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# unitree_sdk2py from source if the wheel is unavailable:
#   git clone https://github.com/unitreerobotics/unitree_sdk2_python
#   pip install -e unitree_sdk2_python
```

## Network

Connect the PC to the robot's ethernet. Give the interface a static IP on the
robot LAN, e.g. `192.168.123.222/24`, and note the interface name
(`ip addr` / `ifconfig`) — it is the first argument to `deploy_real.py`.

## Dry run (no robot needed, works on the Mac)

```bash
python3 deploy_real.py --mock --config configs/g1_agile_e2e.yaml
python3 deploy_real.py --mock --config configs/g1_zealot.yaml
```

The mock simulates motors + IMU and scripts the remote (start → ramp → A →
8 s of policy with a forward command → select). Use it to verify the policy
loads, the loop holds 50 Hz, and targets stay within limits.

## Real robot — procedure

**Safety first: for the first trials hang the robot from a hoist with feet
just touching the ground, keep the area clear, keep a hand on the remote.**

1. Power on the G1 hanging; wait for it to finish booting (default damping).
2. On the Unitree remote press **L2+R2** to enter debug mode — this releases
   the built-in motion service so low-level commands take effect. Joints go
   limp (damping); the robot must be supported.
3. Start the runner:
   ```bash
   python3 deploy_real.py eth0 --config configs/g1_agile_e2e.yaml
   ```
4. `START` → the robot ramps to the default (slightly crouched) pose over 3 s.
5. `A` → the policy takes over. Sticks: left = vx/vy, right x = yaw rate.
6. `SELECT` or `B` → immediate damping stop. Also auto-damps on pelvis tilt
   > 55°, stale LowState (> 200 ms), or Ctrl-C.

Lower the hoist gradually once the robot balances in place. Only then try
small stick commands (`max_cmd` in the config caps them — the zealot config
ships conservative caps of 0.4/0.2/0.4).

## Safety machinery in the loop

- Per-tick target clamp: no motor is commanded further than
  `safety.max_target_step` (0.6 rad) from its **measured** position.
- Joint targets clamped to MJCF joint limits.
- Watchdog: stale LowState → damping. Tilt monitor → damping.
- Waist + arms always position-held (`startup.hold_*`); the AGILE policy
  additionally drives waist roll/pitch itself, incl. its own gains.

## Layout

```
deploy_real.py       state machine: zero-torque → ramp → hold → policy → damping
real_robot.py        unitree_sdk2py DDS I/O (hg LowCmd/LowState, CRC, PR mode)
mock_robot.py        simulated I/O for --mock
policies/zealot_policy.py   safetensors MLP + obs45×hist builder (training conventions)
policies/agile_e2e.py       AGILE e2e ONNX + descriptor parsing + state feedback
common/              G1 motor map & joint limits, remote parsing, quaternion helpers
configs/             per-policy deployment configs (gains, holds, cmd caps, safety)
assets/              AGILE ONNX + descriptor (vendored)
checkpoints/         zealot G1 checkpoints (v14, v17 eval snapshots)
```

## Notes / caveats

- The zealot leg gains (hip 100/2.5, knee 200/5, ankle 20/0.2 & 20/0.1) are
  the WBC-AGILE actuator parametrization the checkpoint was trained against —
  don't "fix" them to the stiffer unitree_rl_gym values without retraining.
- `joint_vel_source: fd` reproduces training (finite-diff of 50 Hz joint
  positions). Switch to `measured` only as an experiment.
- The zealot checkpoints are eval snapshots of an in-progress training run on
  champagne (`/tmp/g1_v14.safetensors`, `/tmp/v16_eval`, `/tmp/v17_eval`);
  fetch newer ones from there as the run progresses.
- Requires the 29-DOF G1 (EDU) with low-level DDS access. The 23-DOF variant
  cannot run the AGILE policy (it drives waist roll/pitch).
