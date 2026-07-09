#!/usr/bin/env python3
"""Export a zealot biped actor checkpoint (safetensors) to ONNX for deployment.

The graph bakes in the full deploy-time forward: obs normalizer
(clip((obs - mean) / sqrt(max(m2/count, 1e-8)), ±5)) followed by the ELU MLP,
linear output = deterministic mean action. Input "obs" [N, obs_dim] float32,
output "action" [N, act_dim] float32 — 12 joint-position offsets; the caller
applies q_target = default_pos + action_scale * action (see sim2sim_xval.py
ACTION_SCALE / GAINS for the deployed values).

Verifies the export two ways before writing:
  * against the float64 numpy reference from sim2sim_xval.py (the same code
    validated to ~1e-5 vs the Rust rollout dump in xval phase 1),
  * on a rollout obs dump when one is given (the exact obs the policy saw).

Usage:
  python3 examples/biped/export_onnx.py policy.safetensors out.onnx [rollout.json]
"""
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2sim_xval import Policy  # noqa: E402  (validated numpy reference)

CLIP = 5.0


def build_model(pol):
    obs_dim, act_dim = pol.obs_dim, pol.act_dim
    var = np.maximum(pol.m2 / pol.count, 1e-8)
    std = np.sqrt(var).astype(np.float32)
    mean = pol.mean.astype(np.float32)

    inits = [
        helper.make_tensor("norm_mean", TensorProto.FLOAT, [obs_dim], mean),
        helper.make_tensor("norm_std", TensorProto.FLOAT, [obs_dim], std),
        helper.make_tensor("clip_min", TensorProto.FLOAT, [], [-CLIP]),
        helper.make_tensor("clip_max", TensorProto.FLOAT, [], [CLIP]),
    ]
    nodes = [
        helper.make_node("Sub", ["obs", "norm_mean"], ["centered"]),
        helper.make_node("Div", ["centered", "norm_std"], ["scaled"]),
        helper.make_node("Clip", ["scaled", "clip_min", "clip_max"], ["h0"]),
    ]
    h = "h0"
    for i, (w, b) in enumerate(zip(pol.W, pol.b)):
        inits.append(helper.make_tensor(
            f"w{i}", TensorProto.FLOAT, list(w.shape), w.astype(np.float32).ravel()))
        inits.append(helper.make_tensor(
            f"b{i}", TensorProto.FLOAT, [len(b)], b.astype(np.float32)))
        z = f"z{i}"
        # Gemm with transB: [N, in] @ w[out, in]^T + b
        nodes.append(helper.make_node("Gemm", [h, f"w{i}", f"b{i}"], [z], transB=1))
        if i < pol.n_layers - 1:
            h = f"a{i}"
            nodes.append(helper.make_node("Elu", [z], [h], alpha=1.0))
        else:
            nodes.append(helper.make_node("Identity", [z], ["action"]))

    graph = helper.make_graph(
        nodes, "zealot_biped_actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["N", obs_dim])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["N", act_dim])],
        inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)],
        producer_name="zealot export_onnx",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main():
    ckpt = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(ckpt)[0] + ".onnx"
    rollout = sys.argv[3] if len(sys.argv) > 3 else None

    pol = Policy(ckpt)
    print(f"{ckpt}: {pol.n_layers} layers, obs {pol.obs_dim} -> act {pol.act_dim}")
    model = build_model(pol)

    import onnxruntime as ort
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    # Parity vs the validated numpy reference on random obs + (optionally) the
    # exact rollout obs the Rust policy saw.
    rng = np.random.default_rng(0)
    batches = [rng.normal(0.0, 1.0, size=(256, pol.obs_dim)).astype(np.float32)]
    if rollout:
        gt = json.load(open(rollout))
        batches.append(np.array(gt["obs"], dtype=np.float32))
    worst = 0.0
    for xb in batches:
        got = sess.run(["action"], {"obs": xb})[0]
        want = np.stack([pol.act(x.astype(np.float64)) for x in xb])
        worst = max(worst, float(np.abs(got - want.astype(np.float32)).max()))
    print(f"parity vs numpy reference: max abs err = {worst:.3e} "
          f"({'PASS' if worst < 1e-4 else 'FAIL'} @ 1e-4)")
    if worst >= 1e-4:
        sys.exit(1)

    onnx.save(model, out)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
