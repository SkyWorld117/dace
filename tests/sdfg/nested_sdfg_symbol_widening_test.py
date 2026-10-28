import copy
import dace


def test_a_nested_sdfg_declares_a_parent_symbol_with_the_parent_type(tmp_path):
    """The nested SDFG took the type of the symbol object in the parent's descriptor (``int``) over the parent's
    declaration (``int64``). Both levels agree in memory, where the objects are shared, but a saved SDFG re-types each
    descriptor from its own level's declaration, and the reloaded connector no longer matched (ICON velocity
    tendencies after LoopToMap)."""
    sdfg = dace.SDFG("parent_declares_n")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("A", ["n"], dace.float64)  # the shape holds an untyped symbol object
    inner = dace.SDFG("body")
    inner.arrays["A"] = copy.deepcopy(sdfg.arrays["A"])  # as LoopToMap nests a loop body
    body = inner.add_state()
    one = body.add_tasklet("one", {}, {"o"}, "o = 1")
    body.add_edge(one, "o", body.add_write("A"), None, dace.Memlet("A[0]"))
    state = sdfg.add_state()
    nsdfg = state.add_nested_sdfg(inner, {}, {"A"})
    state.add_edge(nsdfg, "A", state.add_write("A"), None, dace.Memlet("A[0:n]"))

    assert inner.symbols["n"] == dace.int64, inner.symbols["n"]
    sdfg.save(str(tmp_path / "parent_declares_n.sdfg"))
    dace.SDFG.from_file(str(tmp_path / "parent_declares_n.sdfg")).validate()
