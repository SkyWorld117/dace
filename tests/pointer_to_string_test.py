# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace


def test_pointer_to_string_evaluates_back_to_the_pointer():
    """``to_string`` of a pointer is usable where a type hint ``dace.<to_string()>`` is built (dead-dataflow
    elimination does so for a removed tasklet connector)."""
    for t in (dace.pointer(dace.float64), dace.pointer(dace.pointer(dace.int32))):
        assert eval("dace." + t.to_string(), {"dace": dace}) == t
