# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The synchronisation before a host node that reads what a device copy wrote, in a state inside a control flow
region: the line annotation names the state within its region, not within the SDFG (where the same index may be the
region itself). Asserts on emitted code, so it needs no GPU."""

import dace
from dace import dtypes
from dace.sdfg.state import LoopRegion


def kernel_copy_then_host_read() -> dace.SDFG:
    """``for t in 0:2: G[i] = 2 * i (kernel); H = G (to the host); out[t] = H[0] + 1 (host)``."""
    sdfg = dace.SDFG("presync_in_loop")
    sdfg.add_array("G", [8], dace.float64, storage=dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_array("H", [8], dace.float64, storage=dtypes.StorageType.CPU_Heap, transient=True)
    sdfg.add_array("out", [2], dace.float64)
    loop = LoopRegion("loop", "t < 2", "t", "t = 0", "t = t + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    g = body.add_access("G")
    body.add_mapped_tasklet("fill", {"i": "0:8"}, {}, "o = 2.0 * i", {"o": dace.Memlet("G[i]")},
                            schedule=dtypes.ScheduleType.GPU_Device, output_nodes={g}, external_edges=True)
    h = body.add_access("H")
    body.add_nedge(g, h, dace.Memlet("G[0:8]"))
    t = body.add_tasklet("use", {"x"}, {"y"}, "y = x + 1.0")
    body.add_edge(h, None, t, "x", dace.Memlet("H[0]"))
    body.add_edge(t, "y", body.add_write("out"), None, dace.Memlet("out[t]"))
    sdfg.validate()
    return sdfg


def test_presynchronisation_inside_a_loop_region_generates():
    code = "\n".join(c.clean_code for c in kernel_copy_then_host_read().generate_code())
    assert "StreamSynchronize(" in code
