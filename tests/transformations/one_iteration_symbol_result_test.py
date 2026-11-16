"""A one-iteration loop's scalar result must survive LoopToMap analysis."""
import numpy as np
import pytest
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap


@pytest.mark.parametrize('start', [2, 3])
def test_factorial_symbol_survives_trivial_loop(tmp_path, start):
    graph = dace.SDFG('one_iteration_factorial_' + str(start))
    graph.add_symbol('factorial', dace.int32)
    graph.add_symbol('j', dace.int32)
    graph.add_array('out', [1], dace.int32)
    pre = graph.add_state('pre', is_start_block=True)
    loop = LoopRegion('factorial_loop', loop_var='j',
                      initialize_expr='j = '+str(start), condition_expr='j < 3',
                      update_expr='j = j + 1')
    graph.add_node(loop)
    graph.add_edge(pre, loop, dace.InterstateEdge(assignments={'factorial': '1'}))
    first = loop.add_state('first', is_start_block=True)
    last = loop.add_state('last')
    loop.add_edge(first, last, dace.InterstateEdge(assignments={'factorial': 'factorial * j'}))
    after = graph.add_state('after')
    graph.add_edge(loop, after, dace.InterstateEdge())
    task = after.add_tasklet('result', {}, {'value'}, 'value = factorial')
    after.add_edge(task, 'value', after.add_write('out'), None, dace.Memlet('out[0]'))
    assert graph.apply_transformations_repeated(LoopToMap) == 0
    graph.build_folder = str(tmp_path/'build')
    compiled = graph.compile()
    for _ in range(2):
        out = np.zeros(1, dtype=np.int32)
        compiled(out=out)
        assert out[0] == (2 if start == 2 else 1)
