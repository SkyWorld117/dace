# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Unit tests for the GPU to-device transformation. """

import dace
import numpy as np
import pytest
from dace.transformation.interstate import GPUTransformSDFG


def test_toplevel_transient_lifetime():
    N = dace.symbol('N')

    @dace.program
    def program(A: dace.float64[20, 20]):
        for i in range(20):
            tmp = A[:i, :i]
            tmp2 = A[:5, :N]
            tmp *= 5
            tmp2 *= 10

    sdfg = program.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, options=dict(toplevel_trans=True))

    for name, desc in sdfg.arrays.items():
        if name == 'tmp2' and type(desc) is dace.data.Array:
            assert desc.lifetime is dace.AllocationLifetime.SDFG
        else:
            assert desc.lifetime is not dace.AllocationLifetime.SDFG


@pytest.mark.gpu
def test_scalar_to_symbol_in_nested_sdfg():
    """
    GPUTransformSDFG will automatically create copy-out states for GPU scalars that are used in host-side interstate
    edges. However, this process may only be applied in top-level SDFGs and not in NestedSDFGs that have GPU-device
    schedule but are not part of a single GPU kernel, leading to illegal memory accesses.
    """

    @dace.program
    def nested_program(a: dace.int32, out: dace.int32[10]):
        for i in range(10):
            if a < 5:
                out[i] = 0
                a *= 2
            else:
                out[i] = 10
                a /= 2

    @dace.program
    def main_program(a: dace.int32):
        out = np.ndarray((10, ), dtype=np.int32)
        nested_program(a, out)
        return out

    sdfg = main_program.to_sdfg(simplify=False)
    sdfg.apply_transformations(GPUTransformSDFG)
    out = sdfg(a=4)
    assert np.array_equal(out, np.array([0, 10] * 5, dtype=np.int32))


@pytest.mark.gpu
def test_write_subset():

    @dace.program
    def write_subset(A: dace.int32[20, 20]):
        for i, j in dace.map[2:18, 2:18]:
            A[i, j] = i + j

    sdfg = write_subset.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)

    ref = np.ones((20, 20), dtype=np.int32)
    val = np.copy(ref)

    write_subset.f(ref)
    sdfg(A=val)

    assert np.array_equal(ref, val)


def test_write_full():

    M, N = dace.symbol('M'), dace.symbol('N')

    @dace.program
    def write_full(A: dace.int32[M, N]):
        for i, j in dace.map[0:M, 0:N]:
            A[i, j] = i + j

    sdfg = write_full.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode) and node.data == 'A':
                assert state.out_degree(node) == 0


@pytest.mark.gpu
def test_write_subset_dynamic():

    @dace.program
    def write_subset_dynamic(A: dace.int32[20, 20], x: dace.int32[20], y: dace.int32[20]):
        for i, j in dace.map[2:18, 2:18]:
            A[x[i], y[j]] = i + j

    sdfg = write_subset_dynamic.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)

    ref = np.ones((20, 20), dtype=np.int32)
    val = np.copy(ref)

    x = np.random.permutation(20).astype(np.int32)
    y = np.random.permutation(20).astype(np.int32)

    write_subset_dynamic.f(ref, x, y)
    sdfg(A=val, x=x, y=y)

    assert np.array_equal(ref, val)


@pytest.mark.parametrize(["transient", "scalar"], [[False, False], [False, True], [True, False], [True, True]])
def test_free_tasklet(transient, scalar):
    sdfg = dace.SDFG("assign")

    state = sdfg.add_state("main")
    if scalar:
        arr_name, arr = sdfg.add_scalar("A", dace.float32, transient=transient)
    else:
        arr_name, arr = sdfg.add_array("A", (4, ), dace.float32, transient=transient)

    an = state.add_access(arr_name)

    t = state.add_tasklet("assign", {}, {"_out"}, "_out = 2.0")
    state.add_edge(t, "_out", an, None, dace.memlet.Memlet("A" if scalar else "A[0]"))

    sdfg.validate()

    sdfg.apply_gpu_transformations(validate=True,
                                   validate_all=True,
                                   permissive=True,
                                   sequential_innermaps=True,
                                   register_transients=False,
                                   simplify=False)

    sdfg.validate()


def _row_doubling_body(shape):
    """``b[0, j] = 2 * a[0, j]``, with both connectors describing the whole container."""
    sdfg = dace.SDFG("body")
    sdfg.add_array("a", shape, dace.float64)
    sdfg.add_array("b", shape, dace.float64)
    sdfg.add_symbol("j", dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("t", {"x"}, {"y"}, "y = x * 2")
    state.add_edge(state.add_read("a"), None, tasklet, "x", dace.Memlet("a[0, j]"))
    state.add_edge(tasklet, "y", state.add_write("b"), None, dace.Memlet("b[0, j]"))
    return sdfg


def test_gpu_local_storage_of_a_nested_sdfg_row():
    """The copy on the device holds one row, so the connectors describe a row rather than a matrix.

    ``GPUTransformLocalStorage`` copies only the part of each array the map reads, dropping the
    dimensions the copy is a single index of. Under the nested SDFG contract (see
    ``dace.sdfg.dealias.integrate_nested_sdfg``) a connector is the container it is connected to,
    so the connectors below have to be restated the same way.
    """
    shape = (4, 5)
    sdfg = dace.SDFG("gpu_local_storage_nested")
    sdfg.add_array("A", shape, dace.float64)
    sdfg.add_array("B", shape, dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", dict(j="0:%d" % shape[1]))
    node = state.add_nested_sdfg(_row_doubling_body(shape), {"a"}, {"b"}, {"j": "j"})
    state.add_memlet_path(state.add_read("A"), entry, node, dst_conn="a", memlet=dace.Memlet("A[0, j]"))
    state.add_memlet_path(node, exit_, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[0, j]"))
    sdfg.validate()

    assert sdfg.apply_transformations(GPUTransformLocalStorage) == 1

    for edge in state.all_edges(node):
        connector = edge.dst_conn if edge.dst is node else edge.src_conn
        assert len(node.sdfg.arrays[connector].shape) == 1
        assert node.sdfg.arrays[connector].is_equivalent(sdfg.arrays[edge.data.data])
    sdfg.validate()


@pytest.mark.gpu
def test_an_interstate_condition_reading_a_gpu_map_output_reads_a_host_copy():
    """A GPU map writes ``flag`` and the next interstate condition reads ``flag[0]`` on the host. The
    transformation must register the moved array so a host copy-out precedes the reading edge;
    without it the condition names device memory and validation rejects the SDFG."""
    sdfg = dace.SDFG("gpu_output_gate")
    sdfg.add_array("flag", [4], dace.int32, transient=True)
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("out", [8], dace.float64)
    set_flag = sdfg.add_state("set_flag", is_start_block=True)
    set_flag.add_mapped_tasklet(
        "write_flag", {"i": "0:4"}, {}, "o = 1", {"o": dace.Memlet("flag[i]")}, external_edges=True
    )
    compute = sdfg.add_state("compute")
    compute.add_mapped_tasklet(
        "double",
        {"j": "0:8"},
        {"a": dace.Memlet("A[j]")},
        "o = 2 * a",
        {"o": dace.Memlet("out[j]")},
        external_edges=True,
    )
    end = sdfg.add_state("end")
    sdfg.add_edge(set_flag, compute, dace.InterstateEdge("flag[0] > 0"))
    sdfg.add_edge(set_flag, end, dace.InterstateEdge("not (flag[0] > 0)"))
    sdfg.add_edge(compute, end, dace.InterstateEdge())

    sdfg.apply_transformations(GPUTransformSDFG, options=dict(simplify=False))

    assert sdfg.arrays["flag"].storage == dace.StorageType.GPU_Global
    host_copies = [
        name
        for name, desc in sdfg.arrays.items()
        if name.startswith("host_flag") and desc.storage == dace.StorageType.CPU_Heap
    ]
    assert len(host_copies) == 1, sorted(sdfg.arrays)
    conditions = [e.data.condition.as_string for e in sdfg.all_interstate_edges() if not e.data.is_unconditional()]
    assert len(conditions) == 2, conditions
    assert all(host_copies[0] in c and not re.search(r"\bflag\[", c) for c in conditions), conditions

    A = np.random.rand(8)
    out = np.zeros(8)
    sdfg(A=A, out=out)
    assert np.allclose(out, 2 * A)


def test_free_tasklet_with_an_ordering_edge():
    """A free tasklet the GPU transformation wraps in a one-thread map keeps an edge that only orders it (an empty
    memlet, no connector): the edge passes through the map instead of the map getting a connector for it."""
    sdfg = dace.SDFG("free_tasklet_ordering_edge")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_array("B", [4], dace.float64)
    state = sdfg.add_state()
    a, b = state.add_read("A"), state.add_write("B")
    tasklet = state.add_tasklet("copy", {"inp"}, {"out"}, "out = inp")
    state.add_edge(a, None, tasklet, "inp", dace.Memlet("A[0]"))
    state.add_edge(tasklet, "out", b, None, dace.Memlet("B[0]"))
    state.add_nedge(tasklet, state.add_write("A"), dace.Memlet())  # orders a later use of A after the tasklet

    sdfg.apply_transformations(GPUTransformSDFG)
    sdfg.validate()
    wrappers = [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]
    assert len(wrappers) == 1
    exit_node = state.exit_node(wrappers[0])
    assert all(c is not None for c in exit_node.in_connectors)
    assert any(e.data.is_empty() for e in state.out_edges(exit_node))


def _program_with_two_ordered_free_tasklets() -> dace.SDFG:
    """Two free tasklets at the top level joined by an ordering edge (an empty memlet), and a map whose body is a
    nested SDFG."""
    sdfg = dace.SDFG("ordered_free_tasklets")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_array("B", [4], dace.float64)
    sdfg.add_scalar("s", dace.float64, transient=True)
    state = sdfg.add_state()
    first = state.add_tasklet("first", {"inp"}, {"out"}, "out = inp + 1")
    second = state.add_tasklet("second", {"inp"}, {"out"}, "out = inp * 2")
    state.add_edge(state.add_read("A"), None, first, "inp", dace.Memlet("A[0]"))
    state.add_edge(first, "out", state.add_write("s"), None, dace.Memlet("s"))
    state.add_edge(state.add_read("A"), None, second, "inp", dace.Memlet("A[1]"))
    state.add_edge(second, "out", state.add_write("B"), None, dace.Memlet("B[1]"))
    state.add_nedge(first, second, dace.Memlet())  # the source ordered the two statements
    body = dace.SDFG("body")
    body.add_scalar("a", dace.float64)
    body.add_scalar("b", dace.float64)
    bstate = body.add_state()
    bt = bstate.add_tasklet("t", {"x"}, {"y"}, "y = x + 2")
    bstate.add_edge(bstate.add_read("a"), None, bt, "x", dace.Memlet("a"))
    bstate.add_edge(bt, "y", bstate.add_write("b"), None, dace.Memlet("b"))
    mstate = sdfg.add_state_after(state)
    nested = mstate.add_nested_sdfg(body, {"a"}, {"b"})
    me, mx = mstate.add_map("m", {"i": "0:4"})
    mstate.add_memlet_path(mstate.add_read("A"), me, nested, dst_conn="a", memlet=dace.Memlet("A[i]"))
    mstate.add_memlet_path(nested, mx, mstate.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    return sdfg


def test_ordering_edge_between_free_tasklets_does_not_block_the_transformation():
    """Two top-level tasklets joined by an empty memlet only order each other: the transformation applies (each
    becomes a one-thread kernel, the edge passes between them) instead of refusing the SDFG as it does for a
    memlet carrying data between two tasklets."""
    sdfg = _program_with_two_ordered_free_tasklets()
    sdfg.apply_gpu_transformations()
    sdfg.validate()
    state = sdfg.start_state
    kernels = [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]
    assert len(kernels) == 2 and all(k.map.schedule == dace.ScheduleType.GPU_Device for k in kernels)


def test_the_transformation_applies_to_the_given_sdfg_not_to_a_nested_one():
    """When the SDFG itself cannot take the transformation (a memlet carrying data between two top-level tasklets),
    apply_gpu_transformations says so instead of transforming the body of one of its maps as if it were a program
    of its own."""
    sdfg = _program_with_two_ordered_free_tasklets()
    state = sdfg.start_state
    first, second = (n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet))
    for e in list(state.edges_between(first, second)):
        state.remove_edge(e)
    first.add_out_connector("link")
    second.add_in_connector("link")
    state.add_edge(first, "link", second, "link", dace.Memlet("s"))
    nested = next(n for n in sdfg.all_nodes_recursive() if isinstance(n[0], dace.nodes.NestedSDFG))[0]
    with pytest.raises(ValueError):
        sdfg.apply_gpu_transformations()
    assert not any(name.startswith("gpu_") for name in nested.sdfg.arrays)


if __name__ == "__main__":
    test_toplevel_transient_lifetime()
    test_scalar_to_symbol_in_nested_sdfg()
    test_write_subset()
    test_write_full()
    test_write_subset_dynamic()
    for scalar in [False, True]:
        for transient in [False, True]:
            test_free_tasklet(transient, scalar)
    test_gpu_local_storage_of_a_nested_sdfg_row()
    test_an_interstate_condition_reading_a_gpu_map_output_reads_a_host_copy()
    test_free_tasklet_with_an_ordering_edge()
    test_ordering_edge_between_free_tasklets_does_not_block_the_transformation()
    test_the_transformation_applies_to_the_given_sdfg_not_to_a_nested_one()
