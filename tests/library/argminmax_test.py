# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness tests for :class:`ArgMin` / :class:`ArgMax` library nodes.

Covers:

* whole-array reduction (Fortran ``MINLOC``/``MAXLOC`` scalar form)
* per-axis reduction via the ``dim`` property
* tie-break direction toggle via ``back``
* 0-based vs 1-based output via ``one_based``
"""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import ArgMin, ArgMax


def _build_whole(node_cls, in_shape, dtype, *, one_based=True, back=False):
    """Build the whole-array (``dim=None``) form.

    Fortran ``MINLOC`` / ``MAXLOC`` without ``DIM`` return a rank-1
    integer array of length ``rank(input)`` with the multi-dim subscript
    of the located element, NOT a flat scalar -- so the output buffer
    here has shape ``[rank]``.
    """
    rank = len(in_shape)
    sdfg = dace.SDFG(f"{node_cls.__name__}_whole_{'_'.join(map(str, in_shape))}_{int(back)}_{int(one_based)}")
    sdfg.add_array("v", list(in_shape), dtype)
    sdfg.add_array("idx", [rank], dace.int32)
    state = sdfg.add_state()
    node = node_cls(node_cls.__name__.lower(), one_based=one_based, back=back, dim=None)
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, '_x', dace.Memlet.from_array("v", sdfg.arrays["v"]))
    state.add_edge(node, '_idx', state.add_write("idx"), None, dace.Memlet.from_array("idx", sdfg.arrays["idx"]))
    sdfg.expand_library_nodes()
    return sdfg


def _build_dim(node_cls, in_shape, dim, dtype, *, one_based=True, back=False):
    out_shape = [s for d, s in enumerate(in_shape) if d != (dim - 1)] or [1]
    sdfg = dace.SDFG(f"{node_cls.__name__}_dim{dim}")
    sdfg.add_array("v", list(in_shape), dtype)
    sdfg.add_array("idx", out_shape, dace.int32)
    state = sdfg.add_state()
    node = node_cls(node_cls.__name__.lower(), one_based=one_based, back=back, dim=dim)
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, '_x', dace.Memlet.from_array("v", sdfg.arrays["v"]))
    state.add_edge(node, '_idx', state.add_write("idx"), None, dace.Memlet.from_array("idx", sdfg.arrays["idx"]))
    sdfg.expand_library_nodes()
    return sdfg


def test_argmin_whole_array_one_based():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16)
    sdfg = _build_whole(ArgMin, x.shape, dace.float64)
    idx = np.zeros(1, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert idx[0] == int(np.argmin(x)) + 1


def test_argmax_whole_array_one_based():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(32)
    sdfg = _build_whole(ArgMax, x.shape, dace.float64)
    idx = np.zeros(1, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert idx[0] == int(np.argmax(x)) + 1


def test_argmin_whole_array_zero_based():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(16)
    sdfg = _build_whole(ArgMin, x.shape, dace.float64, one_based=False)
    idx = np.zeros(1, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert idx[0] == int(np.argmin(x))


def test_argmin_first_occurrence_default():
    """Default (``back=False``) selects the first minimum position."""
    x = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
    sdfg = _build_whole(ArgMin, x.shape, dace.float64, one_based=False)
    idx = np.zeros(1, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert idx[0] == 1  # the first 0.0


def test_argmin_last_occurrence_back():
    """``back=True`` selects the last minimum position."""
    x = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
    sdfg = _build_whole(ArgMin, x.shape, dace.float64, one_based=False, back=True)
    idx = np.zeros(1, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert idx[0] == 3  # the last 0.0


def test_argmin_2d_whole_array_returns_subscript():
    """``MINLOC`` without ``DIM`` returns the multi-dim subscript as rank-1."""
    x = np.array([[1.0, 2.0, 3.0], [0.5, 4.0, 5.0]])
    sdfg = _build_whole(ArgMin, x.shape, dace.float64, one_based=True)
    idx = np.zeros(2, dtype=np.int32)
    sdfg(v=x, idx=idx)
    # 0.5 lives at (1, 0) -- Fortran 1-based -> [2, 1].
    assert tuple(idx) == (2, 1), idx


def test_argmax_3d_whole_array_returns_subscript():
    x = np.zeros((3, 4, 2))
    x[2, 1, 0] = 99.0  # unique max
    sdfg = _build_whole(ArgMax, x.shape, dace.float64, one_based=True)
    idx = np.zeros(3, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert tuple(idx) == (3, 2, 1), idx


def test_argmax_2d_dim1():
    """Reduce along Fortran dim=1 (rows): result has shape (#cols,)."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 6))
    sdfg = _build_dim(ArgMax, x.shape, 1, dace.float64)
    idx = np.zeros((6, ), dtype=np.int32)
    sdfg(v=x, idx=idx)
    np.testing.assert_array_equal(idx, np.argmax(x, axis=0) + 1)


def test_argmin_2d_dim2():
    """Reduce along Fortran dim=2 (cols): result has shape (#rows,)."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((5, 7))
    sdfg = _build_dim(ArgMin, x.shape, 2, dace.float64)
    idx = np.zeros((5, ), dtype=np.int32)
    sdfg(v=x, idx=idx)
    np.testing.assert_array_equal(idx, np.argmin(x, axis=1) + 1)


if __name__ == '__main__':
    test_argmin_whole_array_one_based()
    test_argmax_whole_array_one_based()
    test_argmin_whole_array_zero_based()
    test_argmin_first_occurrence_default()
    test_argmin_last_occurrence_back()
    test_argmax_2d_dim1()
    test_argmin_2d_dim2()
    print('ArgMin / ArgMax library node tests PASS')


@pytest.mark.parametrize("node_cls", [ArgMin, ArgMax])
@pytest.mark.parametrize("back", [False, True])
@pytest.mark.parametrize("rank", [1, 2])
def test_dynamic_reduction_axis(tmp_path, node_cls, back, rank):
    """Runtime section lengths retain first/last ties and empty-section results."""
    length = dace.symbolic.pystr_to_symbolic("Max(0, N + 1)")
    graph = _build_dim(node_cls, [length] if rank == 1 else [length, 3], 1, dace.float64, back=back)
    graph.name = f"{node_cls.__name__}_dynamic_{int(back)}_rank{rank}"
    graph.build_folder = str(tmp_path / "build")
    compiled = graph.compile()
    for size in (2, 7, 0, 4):
        values = np.zeros((size,) if rank == 1 else (size, 3), dtype=np.float64)
        output = np.full(1 if rank == 1 else 3, -99, dtype=np.int32)
        if size:
            extreme = -7.0 if node_cls is ArgMin else 7.0
            values[0] = extreme
            values[-1] = extreme
        compiled(v=values, idx=output, N=size - 1)
        expected = (size if back else 1) if size else 0
        np.testing.assert_array_equal(output, np.full(output.shape, expected, dtype=np.int32))


def nested_device_argmin():
    root = dace.SDFG('device_argmin')
    child = dace.SDFG('argmin_leaf')
    for graph in (root,child):
        graph.add_array('v',['N'],dace.float32,storage=dace.StorageType.GPU_Global)
        graph.add_array('idx',[1],dace.int32,storage=dace.StorageType.GPU_Global)
    state = child.add_state()
    node = ArgMin('minimum',dim=1,one_based=True,schedule=dace.ScheduleType.Sequential)
    state.add_node(node)
    state.add_edge(state.add_read('v'),None,node,'_x',dace.Memlet('v[0:N]'))
    state.add_edge(node,'_idx',state.add_write('idx'),None,dace.Memlet('idx[0]'))
    state = root.add_state()
    entry,exit = state.add_map('whole_leaf',{'lane':'0:1'},schedule=dace.ScheduleType.GPU_Device)
    call = state.add_nested_sdfg(child,['v','idx'],['idx'],symbol_mapping={'N':'N'})
    for name in ['v','idx']:
        state.add_memlet_path(state.add_read(name),entry,call,dst_conn=name,
                              memlet=dace.Memlet.from_array(name,root.arrays[name]))
    state.add_memlet_path(call,exit,state.add_write('idx'),src_conn='idx',memlet=dace.Memlet('idx[0]'))
    return root


def test_argmin_in_device_thread_keeps_private_storage():
    graph = nested_device_argmin()
    graph.expand_library_nodes()
    scratch = [desc for child in graph.all_sdfgs_recursive() for name,desc in child.arrays.items()
               if name in ['__best_val','__best_idx']]
    assert len(scratch) == 2
    assert all(desc.storage == dace.StorageType.Register for desc in scratch)
    graph.generate_code()


@pytest.mark.gpu
@pytest.mark.skipif(__import__('os').environ.get('DACE_TEST_GPU') != '1',reason='requires GPU')
def test_argmin_in_device_thread_repeats_on_gpu(tmp_path):
    import ctypes as ct
    runtime = ct.CDLL('libcudart.so')
    runtime.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p),ct.c_size_t]
    runtime.cudaMemcpy.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int]
    runtime.cudaFree.argtypes = [ct.c_void_p]
    graph = nested_device_argmin()
    graph.build_folder = str(tmp_path/'device_argmin')
    compiled = graph.compile()
    class DeviceArray:
        def __init__(self,array):
            self.pointer = ct.c_void_p()
            assert runtime.cudaMalloc(ct.byref(self.pointer),array.nbytes) == 0
            self.__cuda_array_interface__ = dict(shape=array.shape,strides=array.strides,
                typestr=array.dtype.str,data=(self.pointer.value,False),version=3)
    for length,position in [(7,2),(11,9),(7,5)]:
        values = np.ones(length,dtype=np.float32)
        values[position-1:position+1] = -3.
        result = np.full(1,-19,dtype=np.int32)
        buffers = [DeviceArray(array) for array in [values,result]]
        try:
            for host,gpu in zip([values,result],buffers):
                assert runtime.cudaMemcpy(gpu.pointer,host.ctypes.data,host.nbytes,1) == 0
            compiled(v=buffers[0],idx=buffers[1],N=np.int64(length))
            assert runtime.cudaMemcpy(result.ctypes.data,buffers[1].pointer,result.nbytes,2) == 0
            np.testing.assert_array_equal(result,[position])
        finally:
            for gpu in buffers:
                assert runtime.cudaFree(gpu.pointer) == 0
