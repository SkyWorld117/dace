# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest


def test_interstate_multidim_access():
    sdfg = dace.SDFG('ima')
    sdfg.add_array('A', [2, 2], dace.float32)
    s = sdfg.add_state()
    s1 = sdfg.add_state()
    s2 = sdfg.add_state()
    sdfg.add_edge(s, s1, dace.InterstateEdge('(1.0 - A[1, 0]) > 0'))
    sdfg.add_edge(s, s2, dace.InterstateEdge('(1.0 - A[1, 0]) <= 0'))

    sdfg.compile()


if __name__ == '__main__':
    test_interstate_multidim_access()


def _nested_multidim_interstate_access(order, gpu=False):
    import numpy as np

    shapes = {'A': (2, 5, 3), 'index': (2, 3), 'row': (2, 3), 'output': (2, 3)}
    arrays = {name: np.zeros(shape, dtype=np.int32, order=order) for name, shape in shapes.items()}
    arrays['A'][:] = np.arange(30).reshape(2, 5, 3) - 14
    arrays['index'][:] = [[0, 4, 2], [3, 1, 4]]
    arrays['row'][:] = [[1, 0, 1], [0, 1, 0]]
    body = dace.SDFG('indirect_control_body')
    root = dace.SDFG('indirect_control_' + order + ('_gpu' if gpu else '_cpu'))
    for graph in (body, root):
        for name, array in arrays.items():
            graph.add_array(name, array.shape, dace.int32,
                            strides=[stride // array.itemsize for stride in array.strides])
    for name in ('i', 'j', 'selected'):
        body.add_symbol(name, dace.int32)
    start = body.add_state('start', is_start_block=True)
    positive = body.add_state('positive')
    negative = body.add_state('negative')
    done = body.add_state('done')
    read = 'A[i, index[row[i,j],j], j]'
    body.add_edge(start, positive, dace.InterstateEdge(read + ' > 0', assignments={'selected': read}))
    body.add_edge(start, negative, dace.InterstateEdge(read + ' <= 0', assignments={'selected': '-(' + read + ')'}))
    body.add_edge(positive, done, dace.InterstateEdge())
    body.add_edge(negative, done, dace.InterstateEdge())
    task = done.add_tasklet('store', {}, {'out'}, 'out = selected')
    done.add_edge(task, 'out', done.add_write('output'), None, dace.Memlet('output[i,j]'))
    state = root.add_state('columns', is_start_block=True)
    entry, exit = state.add_map('columns', {'i': '0:2', 'j': '0:3'})
    nested = state.add_nested_sdfg(body, {'A', 'index', 'row'}, {'output'},
                                   symbol_mapping={'i': 'i', 'j': 'j'})
    for name in ('A', 'index', 'row'):
        state.add_memlet_path(state.add_read(name), entry, nested, dst_conn=name,
                              memlet=dace.Memlet.from_array(name, root.arrays[name]))
    state.add_memlet_path(nested, exit, state.add_write('output'), src_conn='output',
                          memlet=dace.Memlet.from_array('output', root.arrays['output']))
    root.validate()
    if gpu:
        root.apply_gpu_transformations()
    expected = np.array([[abs(arrays['A'][i, arrays['index'][arrays['row'][i, j], j], j])
                          for j in range(3)] for i in range(2)])
    root(**arrays)
    np.testing.assert_array_equal(arrays['output'], expected)


def test_nested_multidim_interstate_access():
    for order in ('C', 'F'):
        _nested_multidim_interstate_access(order)


@pytest.mark.gpu
def test_nested_multidim_interstate_access_gpu():
    for order in ('C', 'F'):
        _nested_multidim_interstate_access(order, gpu=True)
