# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Integer powers must preserve the rounding of repeated squaring."""
import numpy as np
import pytest

import dace


def _power_graph(name, dtype, count, exponents):
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [count], dtype)
    sdfg.add_array('E', [exponents], dace.uint32)
    sdfg.add_array('B', [exponents, count], dtype)
    sdfg.add_state().add_mapped_tasklet(
        'power', dict(e=f'0:{exponents}', i=f'0:{count}'),
        {'x': dace.Memlet('A[i]'), 'exponent': dace.Memlet('E[e]')},
        'out = dace::math::ipow(x, exponent);',
        {'out': dace.Memlet('B[e, i]')}, external_edges=True,
        language=dace.Language.CPP)
    return sdfg


@pytest.mark.parametrize('dtype', [dace.float32, dace.float64])
def test_ipow_fourth_power_rounding(dtype):
    # These values expose the extra rounding in ((x*x)*x)*x in RRTMG.
    values = np.array([-0.5891060829162598, -0.1320328712463379,
                       -0.7392387390136719, -0.7820024490356445,
                       -0.0, 0.0, 0.25, 1.0, -2.0], dtype=dtype.as_numpy_dtype())
    powers = np.array([0, 1, 2, 4], dtype=np.uint32)
    output = np.empty((len(powers), len(values)), dtype=values.dtype)
    graph = _power_graph('ipow_rounding_' + str(dtype), dtype, len(values), len(powers))
    graph(A=values, E=powers, B=output)
    square = values * values
    expected = np.array([np.ones_like(values), values, square, square * square])
    assert output.tobytes() == expected.tobytes()


def test_ipow_zero_and_large_integral_exponent():
    values = np.array([-1, 0, 1], dtype=np.int64)
    powers = np.array([0, 1, 65536, 65537], dtype=np.uint32)
    output = np.empty((len(powers), len(values)), dtype=values.dtype)
    graph = _power_graph('ipow_integral', dace.int64, len(values), len(powers))
    graph(A=values, E=powers, B=output)
    expected = np.power(values[None, :], powers[:, None].astype(np.int64))
    assert np.array_equal(output, expected)
