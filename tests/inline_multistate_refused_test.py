# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""`InlineSDFG.apply()` must refuse a nested SDFG it cannot correctly inline.

`can_be_applied` is not re-checked by the transformation machinery, so a caller that applies this
transformation directly -- a pass that forces past the check, or any hand-written fixup -- arrives
having verified nothing.  `apply()` then takes `nodes()[0]`, which for a MULTI-state nested SDFG
silently produces a wrong SDFG instead of raising.

That is not hypothetical: forcing past this transformation's structural guard is how applying it
library-wide turned a working program into one that produced NaN at step 1.  The failure is silent in
the SDFG and appears only as wrong numbers.
"""
import dace
import pytest
from dace.transformation.interstate import InlineSDFG


def _outer(nested: dace.SDFG) -> dace.SDFG:
    """An SDFG holding `nested` as a NestedSDFG node."""
    sdfg = dace.SDFG("outer")
    state = sdfg.add_state("s0")
    node = state.add_nested_sdfg(nested, set(), set())
    node.label = "inner"
    return sdfg


def _apply(node, sdfg, state):
    t = InlineSDFG()
    t.setup_match(sdfg=sdfg,
                  cfg_id=state.parent_graph.cfg_id,
                  state_id=state.block_id,
                  subgraph={InlineSDFG.nested_sdfg: node},
                  expr_index=0,
                  override=True)
    t.apply(state, sdfg)


def test_a_multistate_nested_sdfg_is_refused():
    inner = dace.SDFG("inner")
    a = inner.add_state("a")
    b = inner.add_state("b")
    inner.add_edge(a, b, dace.InterstateEdge())      # a second, reachable state
    outer = _outer(inner)
    state = next(outer.all_states())
    node = next(n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    with pytest.raises(ValueError, match="single-state inliner"):
        _apply(node, outer, state)


def test_a_no_inline_nested_sdfg_is_refused():
    """`no_inline` is a promise made by whoever built the nested SDFG.  Honouring it here as well
    means an explicit request not to inline cannot be overridden by accident."""
    inner = dace.SDFG("inner")
    inner.add_state("a")
    outer = _outer(inner)
    state = next(outer.all_states())
    node = next(n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    node.no_inline = True
    with pytest.raises(ValueError, match="no_inline"):
        _apply(node, outer, state)


def test_a_single_state_nested_sdfg_still_inlines():
    """The guard must not be so strict that it breaks the case the transformation is FOR."""
    inner = dace.SDFG("inner")
    inner.add_state("a")
    outer = _outer(inner)
    state = next(outer.all_states())
    node = next(n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    _apply(node, outer, state)
    assert not any(isinstance(n, dace.nodes.NestedSDFG) for n in state.nodes())
