# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A kernel parameter written inside the kernel is not declared const, even when the last write has no path to the
map exit (an in-place update of a global array whose later reads stay in the scope)."""
import re

import dace
from dace import dtypes


def test_in_scope_write_is_not_const():
    n = dace.symbol("n")
    sdfg = dace.SDFG("gpu_in_scope_write")
    sdfg.add_array("a", [n], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("out", [n], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("main")
    me, mx = state.add_map("k", {"i": "0:n"}, schedule=dtypes.ScheduleType.GPU_Device)
    a_in = state.add_read("a")
    a_mid = state.add_access("a")  # written inside the scope, then read: no path to the exit
    out = state.add_write("out")
    double = state.add_tasklet("double", {"x"}, {"y"}, "y = 2 * x")
    use = state.add_tasklet("use", {"x"}, {"y"}, "y = x + 1")
    state.add_memlet_path(a_in, me, double, dst_conn="x", memlet=dace.Memlet("a[i]"))
    state.add_edge(double, "y", a_mid, None, dace.Memlet("a[i]"))
    state.add_edge(a_mid, None, use, "x", dace.Memlet("a[i]"))
    state.add_memlet_path(use, mx, out, src_conn="y", memlet=dace.Memlet("out[i]"))
    sdfg.validate()
    cu = next(c for c in sdfg.generate_code() if c.language == "cu").clean_code
    head = re.search(r"__global__ void\s+(?:__launch_bounds__\([^)]*\)\s+)?k_\w*\(([^)]*)\)", cu)
    assert head is not None
    params = [p.strip() for p in head.group(1).split(",")]
    assert not any(p.startswith("const") and p.endswith(" a") for p in params), params
