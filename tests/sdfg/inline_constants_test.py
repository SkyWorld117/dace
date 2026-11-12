"""Inlining must preserve constant collisions without repeatedly casting every value."""
import dace
import numpy as np

from dace.transformation.interstate import InlineSDFG


def test_inline_constant_names_avoid_repeated_value_materialization(monkeypatch):
    parent = dace.SDFG('parent_constants')
    parent.add_array('out', [1], dace.int32)
    parent.add_constant('value', np.int32(99))
    state = parent.add_state()
    child = dace.SDFG('child_constants')
    child.add_array('result', [1], dace.int32)
    child.add_constant('value', np.int32(7))
    for i in range(128):
        child.add_constant(f'unused_{i}', np.int32(i))
    inner = child.add_state()
    tasklet = inner.add_tasklet('constant', {}, {'y'}, 'y = value')
    inner.add_edge(tasklet, 'y', inner.add_write('result'), None, dace.Memlet('result[0]'))
    nested = state.add_nested_sdfg(child, set(), {'result'})
    state.add_edge(nested, 'result', state.add_write('out'), None, dace.Memlet('out[0]'))
    calls = 0
    getter = dace.SDFG.constants.fget

    def counted(graph):
        nonlocal calls
        calls += 1
        return getter(graph)

    with monkeypatch.context() as patch:
        patch.setattr(dace.SDFG, 'constants', property(counted))
        InlineSDFG.apply_to(parent, nested_sdfg=nested)
    assert calls < 16
    parent.validate()
    assert parent.constants['value'] == 99
    output = np.zeros(1, dtype=np.int32)
    parent(out=output)
    np.testing.assert_array_equal(output, [7])


def test_inline_array_constant_keeps_the_child_value():
    parent = dace.SDFG('parent_array_constant')
    parent.add_array('out', [1], dace.int32)
    parent.add_constant('table', np.array([11, 13], dtype=np.int32))
    state = parent.add_state()
    child = dace.SDFG('child_array_constant')
    child.add_array('result', [1], dace.int32)
    child.add_constant('table', np.array([2, 3], dtype=np.int32))
    inner = child.add_state()
    tasklet = inner.add_tasklet('constant', {}, {'y'}, 'y = table[1]')
    inner.add_edge(tasklet, 'y', inner.add_write('result'), None, dace.Memlet('result[0]'))
    nested = state.add_nested_sdfg(child, set(), {'result'})
    state.add_edge(nested, 'result', state.add_write('out'), None, dace.Memlet('out[0]'))
    InlineSDFG.apply_to(parent, nested_sdfg=nested)
    parent.validate()
    output = np.zeros(1, dtype=np.int32)
    parent(out=output)
    np.testing.assert_array_equal(output, [3])


def test_constant_name_analysis_includes_live_parent_edits_without_casting(monkeypatch):
    parent = dace.SDFG('constant_names_parent')
    parent.add_constant('parent_value', np.int32(99))
    child = dace.SDFG('constant_names_child')
    child.add_constant('child_value', np.int32(7))
    inner = child.add_state()
    child.add_array('out', [1], dace.int32)
    tasklet = inner.add_tasklet('use_constants', {}, {'y'}, 'y = parent_value + child_value')
    inner.add_edge(tasklet, 'y', inner.add_write('out'), None, dace.Memlet('out[0]'))
    parent.add_array('result', [1], dace.int32)
    state = parent.add_state()
    nested = state.add_nested_sdfg(child, set(), {'out'})
    state.add_edge(nested, 'out', state.add_write('result'), None, dace.Memlet('result[0]'))

    def must_not_cast(graph):
        raise AssertionError('name analysis materialized typed constant values')

    with monkeypatch.context() as patch:
        patch.setattr(dace.SDFG, 'constants', property(must_not_cast))
        assert child.constant_names == {'parent_value', 'child_value'}
        parent.add_constant('later', np.int32(3))
        assert child.constant_names == {'parent_value', 'child_value', 'later'}
        del parent.constants_prop['later']
        assert child.constant_names == {'parent_value', 'child_value'}
        assert not inner.used_symbols(all_symbols=False)
        assert not inner.arglist()['out'].transient
