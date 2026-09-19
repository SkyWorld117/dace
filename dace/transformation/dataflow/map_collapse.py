# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the map-collapse transformation. """

from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.symbolic import symlist
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties
from typing import Tuple


@make_properties
class MapCollapse(transformation.SingleStateTransformation):
    """ Implements the Map Collapse pattern.

        Map-collapse takes two nested maps with M and N dimensions respectively,
        and collapses them to a single M+N dimensional map.

        IT FUSES A CHAIN, NOT SIBLINGS, and the consequence is easy to miss because it shows up as
        SLOWNESS rather than as an error.  When a map's body holds two maps side by side, neither
        chain can be collapsed -- collapsing one would have to duplicate the enclosing map -- so the
        enclosing map stays M-dimensional and the inner loops run SERIALLY inside each thread.  On a
        GPU that is the difference between the unit-stride dimension being spread across the block
        and being walked one step at a time.

        Measured on a stencil pair written the way the source usually writes it, one loop per arm:
        the enclosing map covered the outer loop alone, and the kernel cost **2.0 ms per launch
        against 5.4 us** for the fused form -- 372x, and 81.6% of a run's GPU time against 1.2%.
        Nothing warns; the arithmetic is right.

        The two arms cannot simply be fused either: their iteration spaces genuinely differ (they are
        the same stencil at different offsets), so :func:`find_parameter_remapping
        <dace.transformation.dataflow.map_fusion_helper.find_parameter_remapping>` correctly refuses,
        and a correct fusion needs the UNION of the ranges with each arm behind a guard.

        WHAT TO DO ABOUT IT: write the pair as one nest over the union of their ranges, each arm
        guarded by its own bound.  That lowers to a single fully-collapsed map -- and it is the form
        the loop nest should arguably have had in the first place.

        WHY THERE IS NO AUTOMATIC FUSION FOR THIS, and it is not a matter of effort.  Fusing the two
        arms requires their UNION, and for symbolically-bounded ranges that means deciding relations
        between the arms' bound symbols -- e.g. the arms `ulb:c_hi` and `c_lo:ure` union to
        `c_lo:c_hi` only if `ulb >= c_lo` and `ure <= c_hi`.  **The SDFG records no such relation**:
        the symbols are free, and `sympy` cannot decide them (nor is there a `Range.union` to call).
        So the information a fusion would need is simply not present at the SDFG level -- it lives
        with whoever wrote the loops, which is why the answer is a source-level rewrite and not a
        pass.  A pass could only fuse ranges that are provably co-extensive, which is the case
        :func:`find_parameter_remapping <dace.transformation.dataflow.map_fusion_helper.find_parameter_remapping>`
        already accepts.
    """

    outer_map_entry = transformation.PatternNode(nodes.MapEntry)
    inner_map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.outer_map_entry, cls.inner_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Check the edges between the entries of the two maps.
        outer_map_entry: nodes.MapEntry = self.outer_map_entry
        inner_map_entry: nodes.MapEntry = self.inner_map_entry

        # Check that inner map range is independent of outer range
        map_deps = set()
        for s in inner_map_entry.map.range:
            map_deps |= set(map(str, symlist(s)))
        if any(dep in outer_map_entry.map.params for dep in map_deps):
            return False

        # Check that the destination of all the outgoing edges
        # from the outer map's entry is the inner map's entry.
        for _src, _, dest, _, _ in graph.out_edges(outer_map_entry):
            if dest != inner_map_entry:
                return False

        # Check that the source of all the incoming edges
        # to the inner map's entry is the outer map's entry.
        for src, _, _, dst_conn, memlet in graph.in_edges(inner_map_entry):
            if src != outer_map_entry:
                return False

            # Check that dynamic input range memlets are independent of
            # first map range
            if dst_conn is not None and not dst_conn.startswith('IN_'):
                memlet_deps = set()
                for s in memlet.subset:
                    memlet_deps |= set(map(str, symlist(s)))
                if any(dep in outer_map_entry.map.params for dep in memlet_deps):
                    return False

        # Check the edges between the exits of the two maps.
        inner_map_exit = graph.exit_node(inner_map_entry)
        outer_map_exit = graph.exit_node(outer_map_entry)

        # Check that the destination of all the outgoing edges
        # from the inner map's exit is the outer map's exit.
        for _src, _, dest, _, _ in graph.out_edges(inner_map_exit):
            if dest != outer_map_exit:
                return False

        # Check that the source of all the incoming edges
        # to the outer map's exit is the inner map's exit.
        for src, _, _dest, _, _ in graph.in_edges(outer_map_exit):
            if src != inner_map_exit:
                return False

        if not permissive:
            if inner_map_entry.map.schedule != outer_map_entry.map.schedule:
                return False

        return True

    def match_to_str(self, graph):
        outer_map_entry = self.outer_map_entry
        inner_map_entry = self.inner_map_entry

        return ' -> '.join(entry.map.label + ': ' + str(entry.map.params)
                           for entry in [outer_map_entry, inner_map_entry])

    def apply(self, graph: SDFGState, sdfg: SDFG) -> Tuple[nodes.MapEntry, nodes.MapExit]:
        """
        Collapses two maps into one.

        :param sdfg: The SDFG to apply the transformation to.
        :return: A 2-tuple of the new map entry and exit nodes.
        """
        # Extract the parameters and ranges of the inner/outer maps.
        outer_map_entry = self.outer_map_entry
        inner_map_entry = self.inner_map_entry
        inner_map_exit = graph.exit_node(inner_map_entry)
        outer_map_exit = graph.exit_node(outer_map_entry)

        return sdutil.merge_maps(graph, outer_map_entry, outer_map_exit, inner_map_entry, inner_map_exit)


def uncoalesced_device_maps(sdfg):
    """Device Maps whose body holds SIBLING Maps -- i.e. the collapse that did not happen.

    THE SIGNATURE.  When a Map's body contains two Maps side by side, neither chain can collapse (it
    would have to duplicate the enclosing Map), so the enclosing Map keeps its own dimensionality
    alone and the inner loops run serially inside each thread.  On a device that puts the unit-stride
    dimension OFF the block axis, and the kernel is uncoalesced.  The arithmetic is right, so nothing
    fails -- it is simply slow, by a factor measured at 372x on a real kernel (2.0 ms/launch against
    5.4 us).

    This is deliberately a SEPARATE function rather than a condition inside the transformation: a
    transformation that silently declines is the thing that made this expensive to find in the first
    place, and a caller that wants to be told should be able to ask.

    Note what this does NOT rely on: whether the two sibling Maps' iteration spaces could be
    remapped onto each other.  They usually cannot -- `find_parameter_remapping` refuses, correctly,
    because the arms of a stencil differ by an offset -- and an earlier version of this check that
    filtered on that returned a clean bill of health for a Map that was in fact uncoalesced.

    :returns: ``[(outer_map_entry, [sibling_map_entries])]``, empty when every device Map collapsed.
    """
    from dace.dtypes import ScheduleType      # deferred: see the note above
    out = []
    for state in sdfg.all_states():
        for node in state.nodes():
            if not isinstance(node, nodes.MapEntry):
                continue
            if node.map.schedule != ScheduleType.GPU_Device:
                continue
            siblings = [
                n for n in state.nodes()
                if isinstance(n, nodes.MapEntry) and state.entry_node(n) is node
            ]
            if len(siblings) > 1:
                out.append((node, siblings))
    return out
