# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import re

import dace
import numpy as np
from dace import dtypes


def test_nsdfg_input():
    """ Tests constexpr array passed as input argument to a NestedSDFG. """

    @dace.program
    def constexpr_nsdfg():
        a = np.array([1., 2., 3.])
        b = np.max(a)

    with dace.config.set_temporary('compiler', 'inline_sdfgs', value=False):
        constexpr_nsdfg()


def test_tasklet_input_cpu():
    """ Tests constexpr array passed as input argument to a Tasklet (CPU)."""

    @dace.program
    def constexpr_tasklet_cpu():
        a = np.array([1., 2., 3.])
        b = np.max(a)

    with dace.config.set_temporary('optimizer', 'autooptimize', value=True):
        constexpr_tasklet_cpu()


def _map_reading_constant(through_nested: bool, gpu: bool) -> dace.SDFG:
    """``out[i] = C[i % 3]`` in a map (a GPU kernel or a CPU loop), ``C`` a constant array -- a transient with a
    compile-time value -- read from outside the map: by the map's tasklet, or by a nested SDFG in the map that holds
    the same constant under the connector's name."""
    value = np.array([1.0, 2.0, 3.0])
    storage = dtypes.StorageType.GPU_Global if gpu else dtypes.StorageType.Default
    sdfg = dace.SDFG("constant_in_map_" + ("nested" if through_nested else "tasklet") + ("_gpu" if gpu else "_cpu"))
    sdfg.add_array("C", [3], dace.float64, transient=True)
    sdfg.add_constant("C", value, sdfg.arrays["C"])
    sdfg.add_array("out", [12], dace.float64, storage=storage)
    state = sdfg.add_state()
    schedule = dtypes.ScheduleType.GPU_Device if gpu else dtypes.ScheduleType.Default
    entry, exit_ = state.add_map("fill", dict(i="0:12"), schedule=schedule)
    if through_nested:
        inner = dace.SDFG("body")
        inner.add_array("C", [3], dace.float64)
        inner.add_constant("C", value, inner.arrays["C"])
        inner.add_array("out", [12], dace.float64, storage=storage)
        inner.add_symbol("k", dace.int64)
        istate = inner.add_state()
        tasklet = istate.add_tasklet("pick", {"_c"}, {"_o"}, "_o = _c")
        istate.add_edge(istate.add_read("C"), None, tasklet, "_c", dace.Memlet("C[k % 3]"))
        istate.add_edge(tasklet, "_o", istate.add_write("out"), None, dace.Memlet("out[k]"))
        node = state.add_nested_sdfg(inner, {"C"}, {"out"}, symbol_mapping={"k": "i"})
        state.add_memlet_path(state.add_read("C"), entry, node, dst_conn="C", memlet=dace.Memlet("C[0:3]"))
        state.add_memlet_path(node, exit_, state.add_write("out"), src_conn="out", memlet=dace.Memlet("out[i]"))
    else:
        tasklet = state.add_tasklet("pick", {"_c"}, {"_o"}, "_o = _c")
        state.add_memlet_path(state.add_read("C"), entry, tasklet, dst_conn="_c", memlet=dace.Memlet("C[i % 3]"))
        state.add_memlet_path(tasklet, exit_, state.add_write("out"), src_conn="_o", memlet=dace.Memlet("out[i]"))
    return sdfg


def test_nsdfg_holding_its_input_as_constant():
    """A nested SDFG that holds an input as a constant of its own declares it in its function: the input is not also
    a parameter of that function (a redeclaration)."""
    sdfg = _map_reading_constant(through_nested=True, gpu=False)
    with dace.config.set_temporary("compiler", "inline_sdfgs", value=False):
        out = np.zeros(12)
        sdfg(out=out)
    np.testing.assert_array_equal(out, np.tile([1.0, 2.0, 3.0], 4))


def _kernel_code(sdfg: dace.SDFG) -> str:
    """The generated CUDA code of ``sdfg`` (code generation only: no GPU or compiler is needed)."""
    return "\n".join(code.clean_code for code in sdfg.generate_code() if code.language == "cu")


def test_gpu_kernel_reading_constant_in_tasklet():
    """A kernel declares a constant it reads itself: no illegal host-to-device copy, and no kernel parameter of the
    same name."""
    code = _kernel_code(_map_reading_constant(through_nested=False, gpu=True))
    kernel = re.search(r"__global__ void[^{]*?\b\w+\(([^()]*)\)\s*\{\s*constexpr double C\[3\]", code)
    assert kernel, "the kernel does not declare the constant first"
    assert not re.search(r"\bC\b", kernel.group(1)), kernel.group(1)


def test_gpu_kernel_reading_constant_in_nested_sdfg():
    """The same with the constant read by a nested SDFG in the kernel: code generation used to stop at its check that
    the targets it predicted (an illegal copy for the constant) are the ones it used."""
    code = _kernel_code(_map_reading_constant(through_nested=True, gpu=True))
    headers = re.findall(r"(?:__global__|DACE_DFI) void[^{]*?\b\w+\(([^()]*)\)\s*\{", code)
    assert len(headers) >= 2, "no kernel or nested function found"
    for header in headers:
        assert not re.search(r"\bC\b", header), header
    assert len(re.findall(r"constexpr double C\[3\]", code)) >= 2  # the kernel's and the nested function's



def test_constant_of_a_nested_sdfg_in_a_gpu_kernel_generates():
    """A constant array transient in a nested SDFG inside a kernel is a constexpr there: collecting the kernel's
    arguments looked up where it is allocated (it is not) and raised a KeyError (MFC's chemistry, the molecular
    weights in its reaction rates)."""
    inner = dace.SDFG("weights_body")
    inner.add_array("x", [4], dace.float64)
    inner.add_array("y", [4], dace.float64)
    inner.add_symbol("i", dace.int64)
    inner.add_transient("W", [4], dace.float64)
    inner.add_constant("W", np.array([1.0, 2.0, 3.0, 4.0]), inner.arrays["W"])
    st = inner.add_state("body")
    t = st.add_tasklet("scale", {"a", "w"}, {"b"}, "b = a * w")
    st.add_edge(st.add_read("x"), None, t, "a", dace.Memlet("x[i]"))
    st.add_edge(st.add_read("W"), None, t, "w", dace.Memlet("W[i]"))
    st.add_edge(t, "b", st.add_write("y"), None, dace.Memlet("y[i]"))

    sdfg = dace.SDFG("nested_constant_in_kernel")
    sdfg.add_array("x", [4], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("y", [4], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("k", {"i": "0:4"}, schedule=dace.ScheduleType.GPU_Device)
    node = state.add_nested_sdfg(inner, {"x"}, {"y"}, {"i": "i"})
    state.add_memlet_path(state.add_read("x"), entry, node, dst_conn="x", memlet=dace.Memlet("x[0:4]"))
    state.add_memlet_path(node, exit_, state.add_write("y"), src_conn="y", memlet=dace.Memlet("y[0:4]"))
    sdfg.validate()
    code = sdfg.generate_code()
    assert any("W" in c.clean_code for c in code)

if __name__ == "__main__":
    test_nsdfg_input()
    test_tasklet_input_cpu()
    test_nsdfg_holding_its_input_as_constant()
    test_gpu_kernel_reading_constant_in_tasklet()
    test_gpu_kernel_reading_constant_in_nested_sdfg()
