"""``dace.codegen.manifest.describe`` -- the ABI as data.

The SYMBOL case is the one worth pinning, and it is the one that was wrong.  ``arglist()`` hands
back ``dt.Scalar(sdfg.symbols[name])`` for a free symbol -- so its dtype is available -- and
``describe`` took only ``str(a)`` from each entry, then looked the name up in ``sdfg.arrays`` (where
a symbol is not), and recorded ``"unknown"`` for every ``kind="symbol"`` argument.

A language binding cannot write a declaration from ``"unknown"``.  MEASURED on the Fortran port
that consumes this manifest: its generated C had ``int jb, int je, int kb, int ke, int lb, int le``,
and the shim had to carry a hardcoded list of exactly those six names to declare them, because the
manifest said nothing.  A seventh symbol, or a rename, would have fallen through to a silent
default.  These tests fail if the dtype is discarded again.
"""
import dace
from dace.codegen import manifest


def _sdfg_with_symbols():
    """A mapped tasklet over `A`, so `n` and `m` are USED and therefore free.

    A symbol that nothing in the body uses is not in the arglist at all -- MEASURED: an SDFG whose
    only reference to `n` is an array's shape describes as `['A']` alone.
    """
    sdfg = dace.SDFG("Sym")
    state = sdfg.add_state(is_start_block=True)
    n = dace.symbol(sdfg.add_symbol("n", dace.int32))
    m = dace.symbol(sdfg.add_symbol("m", dace.int64))
    sdfg.add_array("A", dtype=dace.float64, shape=(n, ), transient=False)
    sdfg.add_array("B", dtype=dace.float64, shape=(n, ), transient=False)
    sdfg.add_array("C", dtype=dace.float64, shape=(m, ), transient=False)
    state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i0": "0:n"},
        inputs={"__in0": dace.Memlet("A[__i0]")},
        code="__out = __in0",
        outputs={"__out": dace.Memlet("B[__i0]")},
        external_edges=True,
    )
    # A SECOND nest over `m`, for the same reason: a symbol only reaches the arglist if the body
    # uses it.
    sdfg.add_array("D", dtype=dace.float64, shape=(m, ), transient=False)
    state.add_mapped_tasklet(
        "width",
        map_ranges={"__j0": "0:m"},
        inputs={"__in0": dace.Memlet("C[__j0]")},
        code="__out = __in0",
        outputs={"__out": dace.Memlet("D[__j0]")},
        external_edges=True,
    )
    return sdfg


def _by_name(sdfg):
    return {a["name"]: a for a in manifest.describe(sdfg)["args"]}


def test_symbol_records_its_dtype_and_not_unknown():
    by_name = _by_name(_sdfg_with_symbols())
    assert by_name["n"]["kind"] == "symbol"
    # `str(dace.int32)` renders "int", NOT the member name "int32" -- the manifest carries DaCe's
    # own rendering, and a consumer keying on the dtype has to know which vocabulary it is reading.
    assert by_name["n"]["dtype"] == "int", "a symbol's dtype must survive, not be recorded unknown"


def test_symbol_dtype_keeps_its_width():
    """`int` and `int64_t` are different C parameters, so the width must not be flattened."""
    by_name = _by_name(_sdfg_with_symbols())
    assert by_name["m"]["kind"] == "symbol"
    assert by_name["m"]["dtype"] == "int64_t"


def test_the_symbol_fix_leaves_the_other_kinds_alone():
    described = manifest.describe(_sdfg_with_symbols())
    by_name = {a["name"]: a for a in described["args"]}
    assert by_name["A"]["kind"] == "array"
    assert by_name["A"]["dtype"] == "double"
    # every described argument appears in the flat arglist, and vice versa
    assert set(described["order"]) == set(by_name)
    assert [a["name"] for a in described["args"]] == described["order"]
