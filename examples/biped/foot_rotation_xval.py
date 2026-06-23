#!/usr/bin/env python3
"""MuJoCo foot-rotation analog of the nexus `[stance] rot` metric.

Runs the SAME policy closed-loop in MuJoCo and measures, per stance phase, how
much the foot LINK rotates in the WORLD (touchdown→liftoff) — the kinematic
"does the planted foot stay flat" outcome. Compare to nexus's ~8 deg/stance to
decide whether nexus's foot rotation is excess (contact-flattening failure) or
just legitimate ankle articulation (which MuJoCo would show too).

Usage: python3 foot_rotation_xval.py [rollout.json] [policy.safetensors] [steps]
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim_xval as X

ROLLOUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v56_final_rollout.json"
POLICY  = sys.argv[2] if len(sys.argv) > 2 else "/tmp/v56_net.safetensors"
N_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
CMD = np.array([0.4, 0.0, 0.0, 0.0])

gt = json.load(open(ROLLOUT))
jnames = gt["joint_names"]
base = np.array(gt["base"], dtype=np.float64)
joints = np.array(gt["joints"], dtype=np.float64)
policy = X.Policy(POLICY)
model = X.build_mujoco_model(jnames); model.opt.timestep = X.PHYS_DT
data = mujoco.MjData(model)
free_q = model.joint(X.FREEJOINT).qposadr[0]
hinge_q = {n: int(model.joint(n).qposadr[0]) for n in jnames}
hinge_d = {n: int(model.joint(n).dofadr[0]) for n in jnames}
act_id = {n: int(model.actuator(f"act_{n}").id) for n in jnames}
scale = np.array([X.pick_by_prefix(X.ACTION_SCALE, n) for n in jnames])
foot_bids = [int(model.body("foot_subassembly").id), int(model.body("foot_subassembly_2").id)]

# init to nexus step-0 pose
data.qpos[free_q:free_q+3] = base[0, :3]
data.qpos[free_q+3:free_q+7] = [base[0,6], base[0,3], base[0,4], base[0,5]]
for k,n in enumerate(jnames): data.qpos[hinge_q[n]] = joints[0,k]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

def foot_contact_force(bid):
    """Total normal force on the foot body's geoms this step."""
    f = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        g1b = model.geom_bodyid[c.geom1]; g2b = model.geom_bodyid[c.geom2]
        if g1b == bid or g2b == bid:
            ft = np.zeros(6); mujoco.mj_contactForce(model, data, i, ft)
            f += abs(ft[0])  # normal component
    return f

def ang_between(qa, qb):  # wxyz
    d = abs(float(np.dot(qa, qb))); d = min(1.0, d)
    return np.degrees(2.0*np.arccos(d))

act_hist=[np.zeros(12),np.zeros(12)]
stance={b:dict(loaded=False, q0=None, steps=0) for b in foot_bids}
rots=[]
for t in range(N_STEPS):
    last_action = act_hist[0] if t>=2 else np.zeros(12)
    jvel = np.array([data.qvel[hinge_d[n]] for n in jnames]) if t>=2 else np.zeros(12)
    qj = np.array([data.qpos[hinge_q[n]] for n in jnames])
    qw_,qx_,qy_,qz_ = data.qpos[free_q+3:free_q+7]
    phase = (max(0,t-1)*X.CONTROL_DT/X.GAIT_PERIOD)%1.0
    obs=np.zeros(45); obs[0:12]=last_action; obs[12:16]=CMD; obs[16:28]=qj; obs[28:40]=jvel
    obs[40:43]=X.projected_gravity((qx_,qy_,qz_,qw_)); obs[43]=np.sin(2*np.pi*phase); obs[44]=np.cos(2*np.pi*phase)
    action=policy.act(obs); act_hist=[act_hist[1],action.copy()]
    tgt=scale*action
    for k,n in enumerate(jnames): data.ctrl[act_id[n]]=tgt[k]
    for _ in range(X.DECIMATION): mujoco.mj_step(model, data)
    if data.qpos[free_q+2] < X.FALL_Z: break
    for b in foot_bids:
        q = data.xquat[b].copy()  # wxyz
        loaded = foot_contact_force(b) > 5.0  # N (well-loaded, ~½ body weight)
        s = stance[b]
        if loaded and not s["loaded"]:
            s.update(loaded=True, q0=q, steps=1)
        elif loaded and s["loaded"]:
            s["steps"] += 1
        elif (not loaded) and s["loaded"]:
            if s["steps"] >= 2:
                rots.append(ang_between(s["q0"], q))
            s["loaded"]=False

if rots:
    print(f"MuJoCo foot rotation per stance: mean {np.mean(rots):.1f} deg, max {np.max(rots):.0f}, n={len(rots)} stances")
else:
    print("no completed stance phases (policy fell early?)")
print(f"(nexus comparison: ~8 deg/stance at iters=2)")
