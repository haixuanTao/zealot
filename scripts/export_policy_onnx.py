#!/usr/bin/env python3
"""Export a zealot locomotion checkpoint to ONNX as a *reference* artifact.

The graph is the complete observation->action function the deployed controller
runs: Welford normalize, clip to +/-5, then the ELU MLP, deterministic mean
action. Normalization is BAKED IN on purpose -- an ONNX file holding only the
MLP would be a trap, since whoever loads it would have to rediscover the
normalizer and the clip from the safetensors anyway.

What this artifact is FOR:
  * a golden numeric reference for reimplementations (onboard C++/Rust, TensorRT)
  * a portable definition that does not depend on our safetensors reader

What it is NOT, and cannot be:
  * the deployment contract. The graph starts at a fully-assembled 240- or
    288-float observation vector. Everything that has actually bitten us lives
    OUTSIDE it: the lag-2 action slot, joint velocity from FINITE DIFFERENCES
    (not encoder dq), the command-derived gait clock, the body-frame gyro, the
    PD gains, and action_scale (0.5 for v19/v21, 0.25 for v24). An ONNX file
    gives no hint that any of those exist. Read the model card.

Usage:
    python3 scripts/export_policy_onnx.py <ckpt.safetensors> [out.onnx]

Exits non-zero if the exported graph disagrees with the numpy reference, so a
bad export cannot be published silently.
"""
import json
import os
import struct
import sys

import numpy as np


def load_safetensors(path):
    dtypes = {"F32": np.float32, "F64": np.float64, "I64": np.int64,
              "I32": np.int32, "U32": np.uint32, "F16": np.float16}
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        blob = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        out[name] = np.frombuffer(blob[s:e], dtype=dtypes[meta["dtype"]]).reshape(meta["shape"])
    return out


def numpy_reference(sd, obs):
    """Bit-for-bit the controller's `_Policy.act` (float64, as it runs today)."""
    mean = sd["obs_norm.mean"].astype(np.float64)
    m2 = sd["obs_norm.m2"].astype(np.float64)
    count = float(sd["obs_norm.count"].reshape(-1)[0])
    var = np.maximum(m2 / count, 1e-8)
    a = np.clip((obs.astype(np.float64) - mean) / np.sqrt(var), -5.0, 5.0)
    layer = 0
    while f"actor.w_{layer}" in sd:
        w = sd[f"actor.w_{layer}"].astype(np.float64)
        b = sd[f"actor.b_{layer}"].astype(np.float64)
        z = w @ a + b
        a = z if f"actor.w_{layer + 1}" not in sd else np.where(z > 0, z, np.expm1(z))
        layer += 1
    return a


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    ckpt = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(ckpt)[0] + ".onnx"
    sd = load_safetensors(ckpt)

    import torch
    import torch.nn as nn

    n_layers = 0
    while f"actor.w_{n_layers}" in sd:
        n_layers += 1
    if n_layers == 0:
        raise SystemExit(f"no actor.w_* tensors in {ckpt}")
    obs_dim = sd["actor.w_0"].shape[1]
    act_dim = sd[f"actor.w_{n_layers - 1}"].shape[0]

    mean = torch.tensor(sd["obs_norm.mean"].astype(np.float32))
    count = float(sd["obs_norm.count"].reshape(-1)[0])
    std = torch.tensor(np.sqrt(np.maximum(sd["obs_norm.m2"].astype(np.float64) / count,
                                          1e-8)).astype(np.float32))

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
            self.fcs = nn.ModuleList()
            for i in range(n_layers):
                w = sd[f"actor.w_{i}"]
                lin = nn.Linear(w.shape[1], w.shape[0])
                lin.weight.data = torch.tensor(w.astype(np.float32))
                lin.bias.data = torch.tensor(sd[f"actor.b_{i}"].astype(np.float32))
                self.fcs.append(lin)

        def forward(self, obs):
            a = torch.clamp((obs - self.mean) / self.std, -5.0, 5.0)
            for i, fc in enumerate(self.fcs):
                a = fc(a)
                if i != len(self.fcs) - 1:
                    a = torch.nn.functional.elu(a)
            return a

    model = Policy().eval()
    dummy = torch.zeros(1, obs_dim)
    torch.onnx.export(
        model, (dummy,), out,
        input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )

    # torch >=2.13 writes tensors to a SIDECAR <name>.onnx.data by default, so
    # the .onnx alone is a ~3 kB shell that silently resolves against whatever
    # sidecar happens to sit next to it. That validates fine locally and then
    # ships broken. Force everything back inside one self-contained file.
    import onnx
    m = onnx.load(out)                      # resolves the sidecar while it exists
    onnx.save_model(m, out, save_as_external_data=False)
    sidecar = out + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
    reloaded = onnx.load(out, load_external_data=False)
    if any(i.data_location != 0 for i in reloaded.graph.initializer):
        raise SystemExit("FAILED: initializers still reference external data")

    # --- equivalence gate: the export is worthless if it does not match ---
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(256):
        # Exercise the clip on both sides, not just the linear region.
        obs = rng.normal(0.0, 3.0, size=obs_dim).astype(np.float32)
        got = sess.run(None, {"obs": obs[None, :]})[0][0]
        want = numpy_reference(sd, obs)
        worst = max(worst, float(np.abs(got - want).max()))
    print(f"wrote {out}  ({os.path.getsize(out) / 1024:.0f} kB, self-contained)")
    print(f"  obs {obs_dim} -> action {act_dim}, {n_layers} layers, opset 17")
    print(f"  normalizer + clip(+/-5) baked in; float32 graph vs float64 numpy")
    print(f"  worst |onnx - numpy| over 256 random observations: {worst:.3e}")
    # float32 graph against a float64 reference: 1e-4 is generous headroom for
    # accumulation order, and still ~1000x tighter than any action difference
    # that could move the robot (action_scale 0.25 rad on a 0.05 rad/step gait).
    if not np.isfinite(worst) or worst > 1e-4:
        raise SystemExit(f"FAILED: ONNX disagrees with the numpy reference ({worst:.3e})")
    print("  OK: within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
