# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""First-grade numeric typecast functions (``int32`` / ``int64`` /
``float32`` / ``float64``) in symbolic expressions.

A Fortran kind coercion lands as ``int32(x)`` / ``float64(x)`` in BOTH
tasklet bodies and symbolic expressions (interstate edges, memlet
subsets, conditions).  These are registered ``sympy.Function``s so they
parse cleanly, carry integer/real-ness for downstream symbolic
reasoning, and print to the matching ``dace::<type>(x)`` C++ cast
(truncating for int) -- so the SAME bare spelling round-trips through the
sympy printer and cppunparse to identical code.
"""
import numpy as np
import pytest
import sympy

import dace
from dace.symbolic import free_symbols_and_functions, pystr_to_symbolic
from dace.codegen.targets.cpp import sym2cpp


@pytest.mark.parametrize("expr,cpp", [
    ("int32(qm) + 1", "(dace::int32(qm) + 1)"),
    ("int64(x)", "(dace::int64(x))"),
    ("float32(i) * 2", "(2*dace::float32(i))"),
    ("float64(i) - r", "(-r + dace::float64(i))"),
])
def test_typecast_prints_to_dace_cast(expr, cpp):
    assert sym2cpp(pystr_to_symbolic(expr)) == cpp


def test_typecast_roundtrips():
    """The bare spelling survives parse -> str (no ``dace.`` prefix, not
    folded to an attribute)."""
    for expr in ("int32(qm)", "float64(i)", "int64(a + b)"):
        s = pystr_to_symbolic(expr)
        assert str(s).replace(" ", "") == expr.replace(" ", "")


def test_int_typecast_is_integer():
    """``int32``/``int64`` report as integers so they can index a memlet
    subset / drive a loop bound."""
    qm = dace.symbol("qm")
    assert pystr_to_symbolic("int32(qm)").is_integer is True
    assert pystr_to_symbolic("int64(qm)").is_integer is True


def test_dace_prefixed_cast_is_accepted():
    """Safety net: a stray ``dace.int32(x)`` (the attribute-call spelling)
    still parses to the bare typecast rather than raising
    ``'Attr' object is not callable``."""
    assert sym2cpp(pystr_to_symbolic("dace.int32(qm) + 1")) == "(dace::int32(qm) + 1)"


@pytest.mark.parametrize("expr,expected", [
    ("int32(2.9)", sympy.Integer(2)),
    ("int32(-2.9)", sympy.Integer(-2)),
    ("int64(7)", sympy.Integer(7)),
    ("float64(0.5)", sympy.Float(0.5)),
    # REAL(4) narrowing: the folded value is the float32 value widened back.
    ("float32(0.1)", sympy.Float(float(np.float32(0.1)))),
])
def test_typecast_of_literal_folds(expr, expected):
    """A cast of a concrete literal is a number, not an unevaluated function."""
    folded = pystr_to_symbolic(expr)
    assert folded == expected
    assert folded.is_number and folded.is_comparable


@pytest.mark.parametrize("expr", [
    "max(tke_1d[get_pblh_k-1], float32(0.0))",
    "min(x, float64(1.0))",
    "max(i, int32(0))",
    "min(i, int64(7))",
])
def test_typed_constant_bound_is_parseable(expr):
    """Regression (W16-YSU): a Max/Min bound against a *typed* literal must parse.

    Before the literal-folding fix, ``max(tke_1d[get_pblh_k-1], float32(0.0))``
    raised ``ValueError: The argument 'float32(0.0)' is not comparable`` from
    ``sympy.Max``.  On the WRF YSU slice that ValueError propagated through
    ``InterstateEdge.new_symbols`` -> ``ScalarToSymbolPass.remove_symbol_indirection``
    and aborted the whole pipeline.
    """
    parsed = pystr_to_symbolic(expr)
    assert parsed.func in (sympy.Max, sympy.Min)
    # No unevaluated cast of a literal may survive inside a bound.
    assert not parsed.has(dace.symbolic.int32, dace.symbolic.int64, dace.symbolic.float32, dace.symbolic.float64)


def test_free_symbols_of_typed_constant_bound():
    """The exact crash entry: ``free_symbols_and_functions`` on the YSU string."""
    syms = free_symbols_and_functions("max(tke_1d[get_pblh_k-1], float32(0.0))")
    assert "get_pblh_k" in syms
