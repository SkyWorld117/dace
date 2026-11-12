# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from sympy import Min, Max

from dace.symbolic import simplify_ext, symbol


def test_simplify_ext_min() -> None:
    N = symbol("N")

    assert simplify_ext(Min(N, 4) + 1) == Min(N + 1, 5)
    assert simplify_ext(Max(N, 4) + 1) == Max(N + 1, 5)

    untouched = Min(N, 4)
    assert simplify_ext(untouched) == untouched


if __name__ == "__main__":
    test_simplify_ext_min()


def test_numeric_evaluation_does_not_inspect_unrelated_constants():
    import numpy as np
    import sympy as sp
    from dace.symbolic import evaluate

    class UnrelatedConstants(dict):
        def keys(self):
            raise AssertionError('numeric expression inspected unrelated names')

        def items(self):
            raise AssertionError('numeric expression substituted unrelated values')

    values = [2, 0.25, np.float32(0.25), sp.Integer(2), sp.Float(0.25)]
    for value in values:
        result = evaluate(value, UnrelatedConstants(unused=9))
        assert result == value
        assert type(result) is type(value)
