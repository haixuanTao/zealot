#!/usr/bin/env python3
"""Bake the self-contained, mesh-free G1 MJCF for the browser MuJoCo demo.

Source of truth: mujoco_playground's `scene_mjx_feetonly_flat_terrain.xml`
(the exact model `scripts/sim2sim_g1_mujoco.py` validates against).
The browser page renders with zealot's own baked visual meshes
(`g1_visuals_29dof.bin`), so every `<mesh>` asset and every mesh geom is
stripped — what remains is the physics: primitive feet-only collision,
joints/inertials/actuators, and the `home` keyframe. Includes are inlined so
the result is ONE file loadable via `MjModel.from_xml_string` with no
filesystem. The `sensor.xml` include (contact sensors) is dropped — the demo
doesn't read sensors.

Fetches the XMLs from the mujoco_playground GitHub (pinned ref below), so the
tool works without the (jax-heavy) pip package installed. Verifies the result
loads in the local `mujoco` and contains the 12 policy joints, then writes
website/static/bench/g1_mjcf_web.xml.
"""
import io
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REF = "main"
BASE = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_playground/"
    f"{REF}/mujoco_playground/_src/locomotion/g1/xmls/"
)
OUT = Path(__file__).resolve().parent.parent / "website/static/bench/g1_mjcf_web.xml"

POLICY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def fetch(name: str) -> ET.Element:
    with urllib.request.urlopen(BASE + name) as r:
        return ET.parse(io.BytesIO(r.read())).getroot()


def main() -> None:
    scene = fetch("scene_mjx_feetonly_flat_terrain.xml")

    # Inline includes (drop sensor.xml entirely).
    for inc in list(scene.iter("include")):
        pass  # ET.iter can't give parents; do it manually below.
    for parent in [scene]:
        for inc in list(parent):
            if inc.tag != "include":
                continue
            fname = inc.attrib["file"]
            parent.remove(inc)
            if fname == "sensor.xml":
                continue
            sub = fetch(fname)
            # Insert the included root's children at the top so <default>/
            # <asset> blocks precede the scene's worldbody additions.
            for i, child in enumerate(sub):
                parent.insert(i, child)

    # Strip everything mesh/visual-file related.
    for asset in scene.findall("asset"):
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)
    n_geoms = 0
    for body in scene.iter():
        for geom in list(body.findall("geom")):
            if "mesh" in geom.attrib or geom.attrib.get("type") == "mesh":
                body.remove(geom)
                n_geoms += 1
    # The visual-geom default class may reference nothing now; harmless.

    xml = ET.tostring(scene, encoding="unicode")

    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
    ]
    missing = [n for n in POLICY_JOINTS if n not in names]
    assert not missing, f"missing policy joints: {missing}"
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key >= 0, "home keyframe missing"
    assert model.nmesh == 0, model.nmesh
    print(
        f"ok: njnt={model.njnt} nq={model.nq} ngeom={model.ngeom} "
        f"(stripped {n_geoms} mesh geoms), home key id {key}"
    )

    OUT.write_text(xml)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.exit(main())
