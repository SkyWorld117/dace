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

    values = [2, 0.25, np.float32(0.25), sp.Integer(2), sp.Float(0.25),
              sp.pi, sp.Function('float32')(sp.Rational(1, 3))]
    for value in values:
        result = evaluate(value, UnrelatedConstants(unused=9))
        assert result == value
        assert type(result) is type(value)


def test_boolean_expression_numeric_truth_comparisons_preserve_guards():
    import sympy as sp
    from dace.symbolic import pystr_to_symbolic

    expressions = ("x < 3", "not (x < 3)", "x < 3 and x > 0", "not (x < 3 or x > 7)")
    for expression in expressions:
        for literal in (0, 1, 0.0, 1.0):
            for op in ("==", "!="):
                for text in (f"({expression}) {op} {literal}", f"{literal} {op} ({expression})"):
                    actual = pystr_to_symbolic(text)
                    assert actual.free_symbols, text
                    for x in (-1, 1, 3, 8):
                        expected = eval(text, {}, {"x": x})
                        value = actual.subs({symbol: x for symbol in actual.free_symbols})
                        assert bool(sp.simplify(value)) == expected, text
