# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import copy
from typing import Optional
from dace import properties
from dace.frontend.python import astutils
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class PruneEmptyConditionalBranches(ppl.ControlFlowRegionPass):
    """
    Prunes empty (or no-op) conditional branches from conditional blocks.
    """

    CATEGORY: str = 'Simplification'

    def __init__(self):
        super().__init__()
        self.apply_to_conditionals = True

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def apply(self, region: ControlFlowRegion, _) -> Optional[int]:
        if not isinstance(region, ConditionalBlock):
            return None
        removed_branches = 0
        # A conditional block takes the first branch whose condition holds, so every branch after an eliminated one
        # was only taken when the eliminated condition did not hold. ``guard`` is the conjunction of the negated
        # conditions of the branches eliminated so far, which every remaining branch after them must test: an else
        # branch becomes a conditional branch on it, any other branch tests it before its own condition.
        guard = None
        kept = []
        for cond, branch in region.branches:
            branch_nodes = branch.nodes()
            if (len(branch_nodes) == 0 or (len(branch_nodes) == 1 and isinstance(branch_nodes[0], SDFGState)
                                           and len(branch_nodes[0].nodes()) == 0)):
                # Found a branch we can eliminate.
                if cond is not None:
                    negated_condition = astutils.negate_expr(cond.code[0])
                    guard = negated_condition if guard is None else astutils.and_expr(guard, negated_condition)
                removed_branches += 1
                continue
            if guard is not None:
                guarded = copy.deepcopy(guard) if cond is None else astutils.and_expr(guard, cond.code[0])
                cond = properties.CodeBlock([guarded])
            kept.append([cond, branch])
        if removed_branches > 0:
            region._branches = kept

        if len(region.branches) == 0:
            # The conditional has become entirely empty, remove it.
            replacement_node_before = region.parent_graph.add_state_before(region)
            replacement_node_after = region.parent_graph.add_state_after(region)
            region.parent_graph.add_edge(replacement_node_before, replacement_node_after, InterstateEdge())
            region.parent_graph.remove_node(region)

        if removed_branches > 0:
            region.reset_cfg_list()
            return removed_branches
        return None
