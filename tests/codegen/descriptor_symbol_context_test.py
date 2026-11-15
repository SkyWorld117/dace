"""Descriptor type snapshots stay local to each code generation."""
from types import SimpleNamespace
import dace
from dace.codegen.targets.cpp import _descriptor_symbol_types


def test_snapshot_matches_uncached_scope_and_refreshes_between_codegen():
    graph=dace.SDFG('descriptor_symbols')
    n=dace.symbol('n',dtype=dace.int64)
    m=dace.symbol('m',dtype=dace.int32)
    graph.add_array('A',[n],dace.float32)
    graph.add_array('B',[n],dace.float32)
    state=graph.add_state()
    tasklet=state.add_mapped_tasklet('copy',{'i':'0:n'},{'a':dace.Memlet('A[i]')},
        'b=a+i',{'b':dace.Memlet('B[i]')},external_edges=True)[0]
    codegen=SimpleNamespace()
    types=_descriptor_symbol_types(graph,codegen)
    assert types=={'n':dace.int64}
    assert _descriptor_symbol_types(graph,codegen) is types
    expected=state.symbols_defined_at(tasklet)
    actual=state.symbols_defined_at(tasklet,types)
    assert actual==expected and 'i' in actual
    actual['private']=dace.int32
    assert 'private' not in types
    graph.add_array('new',[m],dace.int32)
    assert _descriptor_symbol_types(graph,codegen)=={'n':dace.int64,'m':dace.int32}
    graph.arrays['new'].shape=(n,)
    graph.arrays['new'].total_size=n
    assert _descriptor_symbol_types(graph,SimpleNamespace())=={'n':dace.int64}


def test_repeated_tasklet_codegen_retains_compiled_values(tmp_path):
    import numpy as np
    graph=dace.SDFG('typed_tasklets')
    n=dace.symbol('n',dtype=dace.int64)
    graph.add_array('A',[n],dace.float32)
    graph.add_array('B',[n],dace.float32)
    graph.add_array('tmp',[n],dace.float32,transient=True)
    first=graph.add_state()
    first.add_mapped_tasklet('one',{'i':'0:n'},{'a':dace.Memlet('A[i]')},
        'b=a+i',{'b':dace.Memlet('tmp[i]')},external_edges=True)
    last=graph.add_state_after(first)
    last.add_mapped_tasklet('two',{'i':'0:n'},{'a':dace.Memlet('tmp[i]')},
        'b=2*a',{'b':dace.Memlet('B[i]')},external_edges=True)
    graph.build_folder=str(tmp_path/'build')
    compiled=graph.compile()
    for length in (3,11):
        a=np.arange(length,dtype=np.float32)+1
        b=np.zeros_like(a)
        compiled(A=a,B=b,n=np.int64(length))
        np.testing.assert_array_equal(b,2*(a+np.arange(length)))
