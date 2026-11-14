# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Graph checkpoints must preserve executable interstate arithmetic."""
import json

import numpy as np
import pytest

import dace


@pytest.mark.parametrize("expression", [
    "(x + float32(1.0)) == float32(1.1)",
    "(x + float32(1e20)) - float32(1e20)",
    "not (x != x)",
])
def test_assignment_source_roundtrip(expression):
    edge = dace.InterstateEdge(assignments={"result": expression})
    restored = dace.InterstateEdge.from_json(json.loads(json.dumps(edge.to_json())))
    assert restored.assignments["result"] == expression


def test_checkpoint_retains_single_precision_comparison(tmp_path):
    sdfg = dace.SDFG("checkpoint_float_comparison")
    sdfg.add_symbol("x", dace.float32)
    sdfg.add_symbol("result", dace.int32)
    sdfg.add_array("out", [1], dace.int32)
    start = sdfg.add_state(is_start_block=True)
    finish = sdfg.add_state()
    sdfg.add_edge(start, finish, dace.InterstateEdge(assignments={
        "result": "(x + float32(1.0)) == float32(1.1)"
    }))
    task = finish.add_tasklet("publish", {}, {"value"}, "value = result")
    finish.add_edge(task, "value", finish.add_write("out"), None, dace.Memlet("out[0]"))
    checkpoint = tmp_path / "graph.sdfgz"
    sdfg.save(str(checkpoint))
    restored = dace.SDFG.from_file(str(checkpoint))
    restored.build_folder = str(tmp_path / "build")
    result = np.zeros(1, dtype=np.int32)
    restored.compile()(out=result, x=np.float32(0.1))
    assert result[0] == 1
