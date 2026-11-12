"""Runtime-sized scratch remains private to each CUDA thread, including nested code."""
import os

import dace
import numpy as np
import pytest


def private_scratch_graph(nested):
    n = dace.symbol('N', dtype=dace.int32)
    m = dace.symbol('M', dtype=dace.int32)
    graph = dace.SDFG('gpu_private_dynamic_' + str(int(nested)))
    graph.add_array('B', [m], dace.int32, storage=dace.StorageType.GPU_Global)
    graph.add_symbol('N', dace.int32)
    state = graph.add_state()
    entry, exit = state.add_map('threads', {'i': '0:M'}, schedule=dace.ScheduleType.GPU_Device)
    if nested:
        child = dace.SDFG('private_column')
        child.add_symbol('N', dace.int32)
        child.add_symbol('i', dace.int32)
        child.add_array('out', [1], dace.int32)
        child.add_transient('scratch', [n], dace.int32, storage=dace.StorageType.Register)
        fill = child.add_state()
        fill.add_mapped_tasklet('fill', {'k': '0:N'}, {}, 's = i * 1000 + k',
                                {'s': dace.Memlet('scratch[k]')}, external_edges=True,
                                schedule=dace.ScheduleType.Sequential)
        use = child.add_state_after(fill)
        tasklet = use.add_tasklet('use', {'first', 'last'}, {'value'}, 'value = first + last')
        use.add_edge(use.add_read('scratch'), None, tasklet, 'first', dace.Memlet('scratch[0]'))
        use.add_edge(use.add_read('scratch'), None, tasklet, 'last', dace.Memlet('scratch[N-1]'))
        use.add_edge(tasklet, 'value', use.add_write('out'), None, dace.Memlet('out[0]'))
        tasklet = state.add_nested_sdfg(child, set(), {'out'},
                                      symbol_mapping={'N': n, 'i': dace.symbol('i')})
        state.add_edge(entry, None, tasklet, None, dace.Memlet())
        state.add_memlet_path(tasklet, exit, state.add_write('B'),
                              src_conn='out', memlet=dace.Memlet('B[i]'))
    else:
        graph.add_transient('scratch', [n], dace.int32, storage=dace.StorageType.Register)
        tasklet = state.add_tasklet('private_fill', {}, {'s'},
                                    'for (int k = 0; k < N; ++k) s[k] = i * 1000 + k;',
                                    language=dace.Language.CPP)
        tasklet.out_connectors['s'] = dace.pointer(dace.int32)
        scratch = state.add_access('scratch')
        state.add_edge(entry, None, tasklet, None, dace.Memlet())
        state.add_edge(tasklet, 's', scratch, None, dace.Memlet('scratch[0:N]'))
        use = state.add_tasklet('use', {'first', 'last'}, {'value'}, 'value = first + last')
        state.add_edge(scratch, None, use, 'first', dace.Memlet('scratch[0]'))
        state.add_edge(scratch, None, use, 'last', dace.Memlet('scratch[N-1]'))
        state.add_memlet_path(use, exit, state.add_write('B'),
                              src_conn='value', memlet=dace.Memlet('B[i]'))
    graph.validate()
    return graph


@pytest.mark.parametrize('nested', [False, True])
def test_dynamic_private_cuda_codegen(nested):
    code = '\n'.join(obj.code for obj in private_scratch_graph(nested).generate_code())
    assert 'alloca(' in code
    assert 'delete[] scratch' not in code


@pytest.mark.skipif(os.environ.get('DACE_TEST_GPU') != '1', reason='requires CUDA GPU')
@pytest.mark.parametrize('nested', [False, True])
def test_dynamic_private_cuda_runtime(nested):
    import ctypes as ct
    runtime = ct.CDLL('libcudart.so')
    runtime.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
    runtime.cudaFree.argtypes = [ct.c_void_p]
    runtime.cudaMemcpy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]
    host = np.full(131, -1, dtype=np.int32)
    pointer = ct.c_void_p()
    assert runtime.cudaMalloc(ct.byref(pointer), host.nbytes) == 0
    class DeviceArray:
        __cuda_array_interface__ = dict(shape=host.shape, strides=None, typestr=host.dtype.str,
                                       data=(pointer.value, False), version=3)
    try:
        compiled = private_scratch_graph(nested).compile()
        for n in (3, 29, 7, 31, 3):
            host[:] = -1
            assert runtime.cudaMemcpy(pointer, host.ctypes.data, host.nbytes, 1) == 0
            compiled(B=DeviceArray(), N=np.int32(n), M=np.int32(len(host)))
            assert runtime.cudaMemcpy(host.ctypes.data, pointer, host.nbytes, 2) == 0
            np.testing.assert_array_equal(host, np.arange(131) * 2000 + n - 1)
    finally:
        assert runtime.cudaFree(pointer) == 0
