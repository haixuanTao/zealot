# VR teleop — PICO whole-body → G1 in MuJoCo, live in the browser

Advanced demo: drive a **zealot-trained G1 velocity policy** with a PICO VR
headset. The v28 policy balances and walks (the same 53-dim sim2sim contract as
`examples/biped/sim2sim_g1_mujoco.py`); your tracked upper body is retargeted
onto the PD-held arms in real time; the whole thing renders in a browser tab —
no native viewer, works on a headless host.

```
PICO headset ──XRoboToolkit──▶ G1 Orin (orin/pose_pub.py, ZMQ :5556)
                                   │ smpl joints + wrist quats + sticks + triggers
                                   ▼
                Mac/laptop: g1_vr_stand.py
                • v28 policy legs @50 Hz (torque PD @200 Hz, MuJoCo)
                • position-IK arm retarget (fractional-reach depth mapping)
                • wrist twist from hand orientation
                • thumbstick → velocity command (walk/steer)
                • trigger → magnet-grab welds (pick-and-place scene)
                                   │ link poses (ws :8766)
                                   ▼
                robot.html (three.js) @ http://localhost:8001
```

## Setup

```bash
./setup.sh          # fetches playground G1 scene, menagerie meshes, v28 checkpoint
pip install mujoco zmq websockets safetensors   # python 3.10+
```

Robot side (one-time; survives reboots once in `~unitree`):
```bash
scp orin/pose_pub.py unitree@<robot-ip>:~/
# XRoboToolkit PC service must be installed (GR00T-WholeBodyControl install_pico.sh)
```

## Run

```bash
# 1. Orin
ssh unitree@<robot-ip>
source ~/GR00T-WholeBodyControl/.venv_teleop/bin/activate
nohup python -u ~/pose_pub.py > /tmp/pose_pub.log 2>&1 &

# 2. Headset: XRoboToolkit app → PC Service = <robot-ip> → Status WORKING
#    (toggles: Data/Control "Send", Motion Tracker "Full body", Head+Controller;
#     ankle trackers strapped + calibrated)

# 3. This machine
python3 g1_vr_stand.py --host <robot-ip>
open http://localhost:8001
```

**Controls** — arms mirror your body continuously. Left stick: walk forward/
back, steer. Trigger: magnet-grab the ball (within 0.55 m), release to drop.
Page buttons / arrow keys: shove the robot (0.5 m/s; 💥 = 2.5 m/s, enough to
floor it). `r` / ↺: reset. Ball auto-respawns onto the table after 2 s on the
floor.

Extras in this directory:
- `g1_web.py` — kinematic-only mirror (no policy), same viewer
- `bridge.py` + `index.html` — raw SMPL skeleton debug viewer (:8000)
- `fake_publisher.py` — synthetic pose source; test everything with zero hardware
- `g1_mirror.py --test` — offline retarget sanity renders (needs the GR00T
  `g1_gear_wbc` model in `g1_model/`, scp'able from the robot)
- `orin/start_pico.sh` — swaps the publisher for GR00T's real `--manager`
  streamer (button-driven SONIC pipeline); both bind :5556, one at a time

## Notes & failure modes

- Measured with this stack: v28 standing survives velocity kicks ≤2.0 m/s (any
  direction), falls at 2.5 m/s backward. Trained push was 0.5 m/s.
- Headset **suspends when lifted off the eyes** (proximity sensor) — the stream
  freezes. `pose_pub.py` content-gates freshness and logs `BODY FROZEN` (ankle
  trackers asleep) vs device drops.
- Repeated drops leave zombie sessions in the PC service: symptoms = connected
  but no data; fix = `sudo systemctl restart roboticsservice` on the Orin,
  relaunch pose_pub, force-quit + reopen the headset app, Reconnect.
- Wrist-twist axis conventions are set by `TWIST_AXIS_COL` / `TWIST_SIGN` in
  `g1_mirror.py` — flip there if your pronation mirrors.
