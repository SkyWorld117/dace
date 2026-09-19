# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Detecting the Map that could not collapse, because its body holds SIBLING Maps.

The cost of missing this is 372x on a real kernel and nothing fails -- the arithmetic is right, the
kernel is simply uncoalesced.  A transformation that silently declines is what made that expensive to
find, so the condition is exposed as a question a caller can ask.
"""
import dace
from dace.transformation.dataflow.map_collapse import uncoalesced_device_maps


def _two_sibling_maps(nested=True):
    """A GPU Map whose body holds either two sibling maps or a single one."""
    sdfg = dace.SDFG("m")
    state = sdfg.add_state("s")
    sdfg.add_array("A", (8, 8), dace.float64)
    sdfg.add_array("B", (8, 8), dace.float64)
    outer, outer_exit = state.add_map("outer", {"i": "0:8"}, dace.dtypes.ScheduleType.GPU_Device)
    if nested:
        for tag in ("l", "r"):
            m, mx = state.add_map(tag, {"j": "0:8"}, dace.dtypes.ScheduleType.Default)
            # NEST the inner map inside `outer`: without these scope edges the two maps are
            # siblings at the state level, which is a different (and correctly-flagged-nothing) shape.
            state.add_edge(outer, None, m, None, dace.Memlet())
            state.add_edge(mx, None, outer_exit, None, dace.Memlet())
            t = state.add_tasklet(tag, {"a"}, {"b"}, "b = a")
            an = state.add_access("A")
            bn = state.add_access("B")
            state.add_edge(m, None, an, None, dace.Memlet("A[j]"))
            state.add_edge(an, None, t, "a", dace.Memlet("A[j]"))
            state.add_edge(t, "b", bn, None, dace.Memlet("B[j]"))
            state.add_edge(bn, None, mx, None, dace.Memlet("B[j]"))
    else:
        m, mx = state.add_map("only", {"j": "0:8"}, dace.dtypes.ScheduleType.Default)
        state.add_edge(outer, None, m, None, dace.Memlet())
        state.add_edge(mx, None, outer_exit, None, dace.Memlet())
        t = state.add_tasklet("only_t", {"a"}, {"b"}, "b = a")
        an = state.add_access("A")
        bn = state.add_access("B")
        state.add_edge(m, None, an, None, dace.Memlet("A[j]"))
        state.add_edge(an, None, t, "a", dace.Memlet("A[j]"))
        state.add_edge(t, "b", bn, None, dace.Memlet("B[j]"))
        state.add_edge(bn, None, mx, None, dace.Memlet("B[j]"))
    return sdfg


def test_two_sibling_maps_inside_a_device_map_are_reported():
    """Neither chain can collapse -- collapsing one would have to duplicate the enclosing map -- so
    the enclosing map keeps its own dimensionality and the inner loops run serially per thread."""
    hits = uncoalesced_device_maps(_two_sibling_maps(nested=True))
    assert len(hits) == 1
    outer, siblings = hits[0]
    assert len(siblings) == 2
    assert len(outer.map.params) == 1          # the signature: the map is ONLY its own dims


def test_a_collapsed_map_is_not_reported():
    """The guard has to be able to say "fine", or it is noise rather than information."""
    assert uncoalesced_device_maps(_two_sibling_maps(nested=False)) == []


def test_a_cpu_map_is_not_reported():
    """On a CPU a serial inner loop is not a defect -- the check is about coalescing."""
    sdfg = dace.SDFG("cpu")
    state = sdfg.add_state("s")
    sdfg.add_array("A", (8, 8), dace.float64)
    state.add_map("outer", {"i": "0:8"}, dace.dtypes.ScheduleType.CPU_Multicore)
    for tag in ("l", "r"):
        m, _ = state.add_map(tag, {"j": "0:8"}, dace.dtypes.ScheduleType.Default)
        t = state.add_tasklet(tag, {"a"}, {"b"}, "b = a")
        an = state.add_access("A")
        state.add_edge(m, None, an, None, dace.Memlet("A[j]"))
        state.add_edge(an, None, t, "a", dace.Memlet("A[j]"))
    assert uncoalesced_device_maps(sdfg) == []
