# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
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


if __name__ == "__main__":
    test_nsdfg_input()
    test_tasklet_input_cpu()
    test_nsdfg_holding_its_input_as_constant()
