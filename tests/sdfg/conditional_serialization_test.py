"""Nested conditional serialization visits each branch exactly once."""

import json
from collections import Counter

import dace
import numpy as np
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState


def graph(depth=8):
    sdfg = dace.SDFG("nested_conditional_serialization")
    sdfg.add_array("out", [1], dace.int32)
    sdfg.add_symbol("selector", dace.int32)
    region = sdfg
    for level in range(depth):
        cond = ConditionalBlock(f"if_{level}", sdfg=sdfg)
        region.add_node(cond, is_start_block=True)
        yes = ControlFlowRegion(f"yes_{level}", sdfg=sdfg)
        no = ControlFlowRegion(f"no_{level}", sdfg=sdfg)
        state = no.add_state(f"write_{level}", is_start_block=True)
        t = state.add_tasklet("write", {}, {"value"}, f"value = {level}")
        state.add_edge(t, "value", state.add_write("out"), None, dace.Memlet("out[0]"))
        cond.add_branch(f"selector > {level}", yes)
        cond.add_branch(None, no)
        region = yes
    state = region.add_state("deepest", is_start_block=True)
    t = state.add_tasklet("write", {}, {"value"}, f"value = {depth}")
    state.add_edge(t, "value", state.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


def test_nested_branches_serialize_once_and_round_trip(monkeypatch, tmp_path):
    sdfg = graph()
    visits = Counter()
    original = SDFGState.to_json

    def counted(state, *args, **kwargs):
        visits[id(state)] += 1
        return original(state, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(SDFGState, "to_json", counted)
        serialized = sdfg.to_json()
    assert len(visits) == 9
    assert set(visits.values()) == {1}
    loaded = dace.SDFG.from_json(json.loads(json.dumps(serialized)))
    loaded.validate()
    loaded.build_folder = str(tmp_path / "build")
    out = np.zeros(1, dtype=np.int32)
    for selector in [-1, 0, 3, 9]:
        loaded(out=out, selector=selector)
        assert out[0] == min(max(selector, 0), 8)


def test_serialized_format_matches_legacy_traversal(monkeypatch):
    sdfg = graph(depth=3)
    expected = sdfg.to_json()

    def legacy(block, parent=None):
        result = super(ConditionalBlock, block).to_json(parent)
        del result["nodes"]
        del result["edges"]
        result["branches"] = [(cond.to_json() if cond is not None else None, body.to_json())
                              for cond, body in block.branches]
        return result

    with monkeypatch.context() as patch:
        patch.setattr(ConditionalBlock, "to_json", legacy)
        legacy_result = sdfg.to_json()
    assert expected == legacy_result
