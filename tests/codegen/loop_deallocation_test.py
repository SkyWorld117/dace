"""A body-local last use cannot free scratch allocated before a repeated loop."""
import numpy as np
import dace
from dace.codegen.targets.framecode import (
    _get_dominator_and_postdominator, dominator_trees, pad_control_flow_region_boundaries,
)
from dace.sdfg.state import LoopRegion


def test_nested_loop_scratch_lifetime(tmp_path):
    graph = dace.SDFG('nested_loop_scratch_lifetime')
    graph.add_symbol('n', dace.int64)
    graph.add_transient('scratch', ['n'], dace.float64)
    graph.add_array('out', [4, 3], dace.float64)
    pre = graph.add_state('pre', is_start_block=True)
    before = graph.add_state('before')
    graph.add_edge(pre, before, dace.InterstateEdge(assignments={'n': '3'}))
    outer = LoopRegion('outer', loop_var='i', initialize_expr='i = 0',
                       condition_expr='i < 4', update_expr='i = i + 1')
    graph.add_node(outer)
    graph.add_edge(before, outer, dace.InterstateEdge())
    for name in ['initialize', 'consume']:
        loop = LoopRegion(name, loop_var='j', initialize_expr='j = 0',
                          condition_expr='j < n', update_expr='j = j + 1')
        outer.add_node(loop, is_start_block=name == 'initialize')
        body = loop.add_state(name + '_body', is_start_block=True)
        if name == 'initialize':
            task = body.add_tasklet('write', {}, {'value'}, 'value = i * 10 + j')
            body.add_edge(task, 'value', body.add_write('scratch'), None, dace.Memlet('scratch[j]'))
            first = body
            previous = loop
        else:
            task = body.add_tasklet('read', {'value'}, {'result'}, 'result = value + 100')
            body.add_edge(body.add_read('scratch'), None, task, 'value', dace.Memlet('scratch[j]'))
            body.add_edge(task, 'result', body.add_write('out'), None, dace.Memlet('out[i,j]'))
            outer.add_edge(previous, loop, dace.InterstateEdge())
            last = body
    after = graph.add_state('after')
    graph.add_edge(outer, after, dace.InterstateEdge())
    pad_control_flow_region_boundaries(graph)
    allocation, release = _get_dominator_and_postdominator(
        [(first, dace.sdfg.nodes.AccessNode('scratch')), (last, dace.sdfg.nodes.AccessNode('scratch'))],
        *dominator_trees(graph))
    assert allocation.parent_graph is graph
    assert release.parent_graph is graph
    graph.build_folder = str(tmp_path / 'build')
    compiled = graph.compile()
    for _ in range(2):
        out = np.zeros((4, 3))
        compiled(out=out)
        np.testing.assert_array_equal(out, np.arange(4)[:, None] * 10 + np.arange(3)[None, :] + 100)
