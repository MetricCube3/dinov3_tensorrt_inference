#!/usr/bin/env python3
"""
Insert IdentityBarrier plugin nodes around the attention matmuls.

A plugin node is OPAQUE to TensorRT's Myelin fusion compiler, so Myelin cannot
fuse across it. By wrapping the inputs/outputs of the two attention matmuls
(QK^T and attn@V) with no-op IdentityBarrier plugins, we force QK^T, softmax and
attn@V to be built as PLAIN TensorRT layers -- which were proven correct -- instead
of Myelin's buggy fused-attention region.

Per block we barrier 5 tensors:
    QK^T (/attn/MatMul):    input0 (Q_rope), input1 (K_rope^T), output (scores)
    attn@V (/attn/MatMul_1): input0 (probs),  input1 (V)
This isolates both matmuls completely and leaves the softmax sitting between two
barriers (scores -> ... -> probs), so it too becomes a plain region.

Usage:
    python3 insert_plugin_barrier.py \
        --in  dinov3_vits16_features_512.onnx \
        --out dinov3_vits16_features_512_plugin.onnx
"""
import argparse

import onnx
import onnx_graphsurgeon as gs

PLUGIN_OP = "IdentityBarrier"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = ""


def barrier_tensor(graph: gs.Graph, t: gs.Variable, idx: int) -> None:
    """Route every current consumer of tensor `t` through an IdentityBarrier."""
    consumers = list(t.outputs)  # capture BEFORE we wire in the barrier node
    if not consumers:
        return  # graph output / unused: nothing to re-route

    safe = t.name.replace("/", "_").replace(".", "_")
    bar_out = gs.Variable(name=f"{safe}_pb{idx}", dtype=t.dtype, shape=t.shape)
    node = gs.Node(
        op=PLUGIN_OP,
        name=f"IdentityBarrier_{idx}_{safe}",
        inputs=[t],
        outputs=[bar_out],
        attrs={
            "plugin_version": PLUGIN_VERSION,
            "plugin_namespace": PLUGIN_NAMESPACE,
        },
    )
    graph.nodes.append(node)

    for c in consumers:
        c.inputs = [bar_out if x is t else x for x in c.inputs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="dinov3_vits16_features_512.onnx")
    ap.add_argument("--out", dest="out", default="dinov3_vits16_features_512_plugin.onnx")
    args = ap.parse_args()

    graph = gs.import_onnx(onnx.load(args.inp))

    qk_nodes = [n for n in graph.nodes if n.name.endswith("/attn/MatMul")]
    av_nodes = [n for n in graph.nodes if n.name.endswith("/attn/MatMul_1")]
    print(f"Found {len(qk_nodes)} QK^T and {len(av_nodes)} attn@V matmul nodes.")
    if not qk_nodes or not av_nodes:
        raise RuntimeError("Could not find attention matmul nodes; check naming.")

    # Collect tensors to barrier (dedup by identity, preserve order).
    seen = set()
    tensors = []

    def add(t: gs.Variable) -> None:
        if t is not None and id(t) not in seen:
            seen.add(id(t))
            tensors.append(t)

    for n in qk_nodes:
        add(n.inputs[0])   # Q (rope-applied)
        add(n.inputs[1])   # K^T (rope-applied, transposed)
        add(n.outputs[0])  # raw scores
    for n in av_nodes:
        add(n.inputs[0])   # softmax probs
        add(n.inputs[1])   # V

    for i, t in enumerate(tensors):
        barrier_tensor(graph, t, i)

    print(f"Inserted {len(tensors)} IdentityBarrier plugin nodes.")

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), args.out)
    print(f"Saved patched model -> {args.out}")


if __name__ == "__main__":
    main()
