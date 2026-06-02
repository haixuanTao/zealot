#!/usr/bin/env python3
"""Same knife-edge / one-tip foot-stability scenarios as `foot_tip_stability.rs`,
but in Newton (NVIDIA's Warp-based physics framework) using `SolverMuJoCo`.

Outputs a JSON pose dump in the same schema as the Rust example so the existing
`render_foot_tip.py` can replay it side-by-side with the rapier result.

Run:
    python3 examples/biped/foot_tip_stability_newton.py
    python3 examples/biped/render_foot_tip.py /tmp/foot_tip_poses_newton.json /tmp/foot_tip_newton.mp4

Newton's MuJoCo solver uses MuJoCo's soft-contact formulation, so the prediction
is: the knife_edge_94mm artefact rapier shows will tip flat correctly here.
"""
import json
import math

import numpy as np
import warp as wp

import newton

# Foot half-extents (m). Same as the Rust repro.
HE = (0.047, 0.0675, 0.015)
DT = 1.0 / 200.0
TOTAL_STEPS = 1000  # 5 s


def quat_axis_angle(axis, angle):
    """Axis-angle to (qx, qy, qz, qw)."""
    s = math.sin(angle / 2)
    c = math.cos(angle / 2)
    n = np.asarray(axis, dtype=float)
    n /= np.linalg.norm(n)
    return (float(n[0] * s), float(n[1] * s), float(n[2] * s), float(c))


def quat_mul(q1, q2):
    """(x,y,z,w) ⊗ (x,y,z,w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_rotate(q, v):
    """Rotate world vector v=(x,y,z) by quaternion q=(x,y,z,w)."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    # v' = v + 2*qxyz × (qxyz × v + qw*v)
    cx1 = qy * vz - qz * vy
    cy1 = qz * vx - qx * vz
    cz1 = qx * vy - qy * vx
    tx = cx1 + qw * vx
    ty = cy1 + qw * vy
    tz = cz1 + qw * vz
    cx2 = qy * tz - qz * ty
    cy2 = qz * tx - qx * tz
    cz2 = qx * ty - qy * tx
    return (vx + 2 * cx2, vy + 2 * cy2, vz + 2 * cz2)


def lowest_corner_z(quat, he):
    """World-z of the lowest box corner after applying `quat` to the box."""
    a, b, c = he
    lo = float("inf")
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                _, _, rz = quat_rotate(quat, (sx * a, sy * b, sz * c))
                if rz < lo:
                    lo = rz
    return lo


def sole_tilt_deg(quat):
    """Same convention as the rapier example: angle between R*Z and world +Z."""
    _, _, nz = quat_rotate(quat, (0.0, 0.0, 1.0))
    return math.degrees(math.acos(min(1.0, abs(nz))))


def run_scenario(name, init_quat):
    """Build a fresh Newton model with a single box and ground, spawn it at the
    knife-edge configuration (lowest corner ~ on z=0), simulate 5 s, return the
    per-step (px, py, pz, qx, qy, qz, qw) trajectory."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane(height=0.0)

    # Drop test: spawn 5 cm above the just-touching height so impact adds a real
    # perturbation, instead of starting at a strict unstable equilibrium.
    spawn_z = -lowest_corner_z(init_quat, HE) + 0.05
    spawn_xform = wp.transform(
        wp.vec3(0.0, 0.0, spawn_z),
        wp.quat(init_quat[0], init_quat[1], init_quat[2], init_quat[3]),
    )
    bid = builder.add_body(xform=spawn_xform, mass=0.3)

    # Box shape with friction 1.0 and density 1000 (so the body mass ~= 0.3 kg
    # via the density; explicit `mass` above is redundant but harmless).
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=1.0)
    builder.add_shape_box(bid, hx=HE[0], hy=HE[1], hz=HE[2], cfg=cfg)

    # builder.gravity is a magnitude scalar (default -9.81); up axis Z (matches rapier).
    model = builder.finalize(device="cpu")

    solver = newton.solvers.SolverMuJoCo(model, iterations=8, use_mujoco_cpu=True)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    # MuJoCo manages contacts internally — passing None is the supported pattern,
    # see newton/examples/sensors/example_sensor_contact.py.

    poses = []
    # Initial pose (sim time 0)
    q0 = state_0.body_q.numpy()[0]
    poses.append([float(q0[0]), float(q0[1]), float(q0[2]),
                  float(q0[3]), float(q0[4]), float(q0[5]), float(q0[6])])

    tipped_at = None
    print(f"\n[{name}]  initial tilt = {sole_tilt_deg(init_quat):.2f}°  spawn z = {spawn_z:.4f} m (5 cm drop)")
    print("   t (s)   tilt (deg)   z (m)     |omega| (rad/s)")
    for s in range(TOTAL_STEPS):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, DT)
        # swap buffers
        state_0, state_1 = state_1, state_0
        q = state_0.body_q.numpy()[0]
        poses.append([float(q[0]), float(q[1]), float(q[2]),
                      float(q[3]), float(q[4]), float(q[5]), float(q[6])])
        t = (s + 1) * DT
        tilt_deg = sole_tilt_deg((q[3], q[4], q[5], q[6]))
        if tipped_at is None and tilt_deg < 10.0:
            tipped_at = t
        if (s + 1) % 100 == 0:
            qd = state_0.body_qd.numpy()[0]
            omega = math.sqrt(qd[0] ** 2 + qd[1] ** 2 + qd[2] ** 2)
            print(f"  {t:5.2f}    {tilt_deg:6.2f}      {q[2]:+.4f}    {omega:.4f}")

    final_tilt = sole_tilt_deg((q[3], q[4], q[5], q[6]))
    if tipped_at is not None:
        print(f"VERDICT: settled flat (tilt < 10°) at t = {tipped_at:.2f} s — final tilt {final_tilt:.2f}°")
    else:
        print(f"VERDICT: came to rest at {final_tilt:.2f}° after 5 s.")
    return poses


def main():
    # Knife-edge configurations — same as the rapier repro:
    a, b, c = HE
    theta_94 = math.atan(b / c)  # X-rot, 9.4 cm contact edge, ≈ 77.47°
    phi_135 = math.atan(a / c)  # Y-rot, 13.5 cm contact edge, ≈ 72.34°

    # One-tip: rotation arc from body-diagonal (+a, -b, -c) to world -Z.
    # We compute it as quat-from-two-vectors using the cross/half-angle formula.
    d = math.sqrt(a * a + b * b + c * c)
    n = np.array([a / d, -b / d, -c / d])
    target = np.array([0.0, 0.0, -1.0])
    axis = np.cross(n, target)
    s = np.linalg.norm(axis)
    cdot = float(np.dot(n, target))
    angle = math.atan2(s, cdot)
    if s < 1e-9:
        # already aligned (or anti-aligned) — no rotation needed (or 180°)
        one_tip = (0.0, 0.0, 0.0, 1.0)
    else:
        axis /= s
        one_tip = quat_axis_angle(axis.tolist(), angle)

    scenarios = {
        "flat_baseline": run_scenario("flat_baseline", (0.0, 0.0, 0.0, 1.0)),
        "knife_edge_94mm": run_scenario(
            "knife_edge_94mm", quat_axis_angle((1.0, 0.0, 0.0), theta_94)
        ),
        "knife_edge_135mm": run_scenario(
            "knife_edge_135mm", quat_axis_angle((0.0, 1.0, 0.0), phi_135)
        ),
        "one_tip_corner": run_scenario("one_tip_corner", one_tip),
    }

    out = {
        "dt": DT,
        "half_extents": list(HE),
        "scenarios": scenarios,
    }
    path = "/tmp/foot_tip_poses_newton.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"\nwrote pose trajectories → {path}")
    print("render with: python3 examples/biped/render_foot_tip.py "
          f"{path} /tmp/foot_tip_newton.mp4")


if __name__ == "__main__":
    main()
