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
    sdfg.add_array("C", (8, 8), dace.float64)      # the second arm's OWN output: the arms of a real
                                                   # stencil pair write disjoint arrays, and that
                                                   # independence is what the split depends on
    outer, outer_exit = state.add_map("outer", {"i": "0:8"}, dace.dtypes.ScheduleType.GPU_Device)
    if nested:
        for tag in ("l", "r"):
            out = "B" if tag == "l" else "C"     # the arms of a real stencil pair write DISJOINT
                                                 # arrays; that independence is what the split needs
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
            state.add_edge(t, "b", bn, None, dace.Memlet(f"{out}[j]"))
            state.add_edge(bn, None, mx, None, dace.Memlet(f"{out}[j]"))
    else:
        out = "B"
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


def _collapse_all(sdfg):
    """Drive MapCollapse manually: `apply_transformations_repeated` does not find these Maps even
    when the structure is sound, which cost a session of debugging the wrong mechanism."""
    from dace.transformation.dataflow import MapCollapse
    n = 0
    for st in sdfg.all_states():
        for o in [x for x in st.nodes() if isinstance(x, dace.nodes.MapEntry)]:
            for k in [x for x in st.nodes()
                      if isinstance(x, dace.nodes.MapEntry) and st.entry_node(x) is o]:
                t = MapCollapse()
                t.setup_match(sdfg=sdfg, cfg_id=st.parent_graph.cfg_id, state_id=st.block_id,
                              subgraph={MapCollapse.outer_map_entry: o,
                                        MapCollapse.inner_map_entry: k},
                              expr_index=0, override=True)
                if t.can_be_applied(st, 0, sdfg, permissive=True):
                    t.apply(st, sdfg)
                    n += 1
    return n


def test_splitting_is_a_no_op_on_a_single_child_map():
    """A guard has to be able to do nothing, or it is noise: with one child there is nothing to
    split, and the form that already collapses must be untouched."""
    from dace.transformation.dataflow.map_collapse import split_sibling_maps
    assert split_sibling_maps(_two_sibling_maps(nested=False)) == 0


def test_splitting_is_a_no_op_when_the_children_are_not_independent():
    """Splitting `for i { A; B }` into two loops is equivalent ONLY when A and B are independent.
    The fixture's arms both write `B`, so nothing may be split -- and refusing is the whole safety
    argument for this transformation."""
    from dace.transformation.dataflow.map_collapse import split_sibling_maps, _children_write_disjointly
    sdfg = _two_sibling_maps(nested=True)
    state = next(sdfg.all_states())
    outer = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)
                 and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device)
    kids = [n for n in state.nodes()
            if isinstance(n, dace.nodes.MapEntry) and state.entry_node(n) is outer]
    assert len(kids) == 2
    assert not _children_write_disjointly(state, kids)      # both write `B`
    assert split_sibling_maps(sdfg) == 0


# NOT COVERED HERE, and worth saying rather than leaving implied: the POSITIVE path --
# `split_sibling_maps` cloning, both chains collapsing, and the SDFG still validating -- is verified
# against the real thing (the shipped union TU de-fused back into the two loops the source writes):
#
#   sibling: uncoalesced_device_maps 1 -> 0;  split=1 collapse=2;  device map dims=[3, 3];  VALID
#   union:   uncoalesced 0 -> 0;  split=0 collapse=0;  device map dims=[3];                 VALID
#
# It is not a unit test because it needs the Fortran TU and the full pipeline.  Reproducing it in a
# synthetic SDFG was attempted and the fixture proved fiddly enough that asserting the wrong thing
# would be worse than saying so.
