# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest
import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.simplification.prune_empty_conditional_branches import PruneEmptyConditionalBranches


def test_prune_empty_else():
    N = dace.symbol('N')

    @dace.program
    def prune_empty_else(A: dace.int32[N]):
        A[:] = 0
        if N == 32:
            for i in range(N):
                A[i] = 1
        else:
            A[:] = 0

    sdfg = prune_empty_else.to_sdfg(simplify=False)

    conditional: ConditionalBlock = None
    for n in sdfg.nodes():
        if isinstance(n, ConditionalBlock):
            conditional = n
            break

    assert len(conditional.branches) == 2

    conditional._branches[-1][0] = None
    else_branch = conditional._branches[-1][1]
    else_branch.remove_nodes_from(else_branch.nodes())
    else_branch.add_state('empty')

    res = PruneEmptyConditionalBranches().apply_pass(sdfg, {})

    assert res[conditional.cfg_id] == 1
    assert len(conditional.branches) == 1

    N1 = 32
    N2 = 31
    A1 = np.zeros((N1, ), dtype=np.int32)
    A2 = np.zeros((N2, ), dtype=np.int32)
    verif1 = np.full((N1, ), 1, dtype=np.int32)
    verif2 = np.zeros((N2, ), dtype=np.int32)

    sdfg(A1, N=N1)
    sdfg(A2, N=N2)

    assert np.allclose(A1, verif1)
    assert np.allclose(A2, verif2)


def test_prune_empty_if_with_else():
    N = dace.symbol('N')

    @dace.program
    def prune_empty_if_with_else(A: dace.int32[N]):
        A[:] = 0
        if N == 32:
            for i in range(N):
                A[i] = 2
        else:
            A[:] = 1

    sdfg = prune_empty_if_with_else.to_sdfg(simplify=False)

    conditional: ConditionalBlock = None
    for n in sdfg.nodes():
        if isinstance(n, ConditionalBlock):
            conditional = n
            break

    assert len(conditional.branches) == 2

    conditional._branches[-1][0] = None
    if_branch = conditional._branches[0][1]
    if_branch.remove_nodes_from(if_branch.nodes())
    if_branch.add_state('empty')

    res = PruneEmptyConditionalBranches().apply_pass(sdfg, {})

    assert res[conditional.cfg_id] == 1
    assert len(conditional.branches) == 1
    assert conditional.branches[0][0] is not None

    N1 = 32
    N2 = 31
    A1 = np.zeros((N1, ), dtype=np.int32)
    A2 = np.zeros((N2, ), dtype=np.int32)
    verif1 = np.zeros((N1, ), dtype=np.int32)
    verif2 = np.full((N2, ), 1, dtype=np.int32)

    sdfg(A1, N=N1)
    sdfg(A2, N=N2)

    assert np.allclose(A1, verif1)
    assert np.allclose(A2, verif2)


@pytest.mark.parametrize(
    "conditions, empty, taken",
    [
        # An empty first branch: the second one is still not taken when the first condition holds.
        (["a > 0", "a > -5"], [0], {1: 0, -1: 2, -9: 0}),
        (["a > 0", "a > -5", None], [0], {1: 0, -1: 2, -9: 3}),
        # Two empty branches before a conditional and an else branch.
        (["a > 5", "a == 3", "a > 0", None], [0, 1], {9: 0, 3: 0, 1: 3, -1: 4}),
        # An empty branch between two others: only the branches after it test its negation.
        (["a == 1", "a < 3", "a < 5", None], [1], {1: 1, 2: 0, 4: 3, 7: 4}),
    ],
)
def test_prune_empty_branch_before_others(conditions, empty, taken):
    """Every branch after an eliminated one keeps not being taken when the eliminated condition holds. ``empty``:
    the branches without work; ``taken``: per value of ``a``, the branch (1-based, 0 for none) whose work runs."""
    sdfg = dace.SDFG("prune_empty_branch_before_others")
    sdfg.add_symbol("a", dace.int32)
    sdfg.add_array("out", [1], dace.int32)
    start = sdfg.add_state("start", is_start_block=True)
    conditional = ConditionalBlock("cond")
    sdfg.add_node(conditional)
    sdfg.add_edge(start, conditional, dace.InterstateEdge())
    for i, condition in enumerate(conditions):
        branch = ControlFlowRegion(f"branch_{i}", sdfg)
        state = branch.add_state(f"write_{i}", is_start_block=True)
        if i not in empty:
            tasklet = state.add_tasklet(f"set_{i}", {}, {"o"}, f"o = {i + 1}")
            state.add_edge(tasklet, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
        conditional.add_branch(None if condition is None else CodeBlock(condition), branch)
    sdfg.add_edge(conditional, sdfg.add_state("end"), dace.InterstateEdge())

    res = PruneEmptyConditionalBranches().apply_pass(sdfg, {})
    assert res[conditional.cfg_id] == len(empty)
    assert len(conditional.branches) == len(conditions) - len(empty)
    sdfg.validate()

    compiled = sdfg.compile()
    for a, branch in taken.items():
        out = np.zeros([1], dtype=np.int32)
        compiled(out=out, a=a)
        assert out[0] == branch, (a, out[0], branch)


if __name__ == "__main__":
    test_prune_empty_else()
    test_prune_empty_if_with_else()
    test_prune_empty_branch_before_others(["a > 0", "a > -5", None], [0], {1: 0, -1: 2, -9: 3})
