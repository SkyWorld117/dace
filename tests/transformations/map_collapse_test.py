# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg.utils import consolidate_edges
from dace.transformation.dataflow import MapCollapse


@dace.program
def tocollapse(A: dace.float64[3, 3], B: dace.float64[2, 2]):
    for i in dace.map[0:2]:
        for j in dace.map[0:2]:
            with dace.tasklet:
                a1 << A[i, j]
                a2 << A[i + 1, j]
                a3 << A[i, j + 1]
                a4 << A[i + 1, j + 1]
                b >> B[i, j]
                b = a1 + a2 + a3 + a4


@dace.program
def memset_map(A: dace.float64[5, 5, 5]):
    for i in dace.map[0:5]:
        for j in dace.map[0:5]:
            for k in dace.map[0:5]:
                A[i, j, k] = 0.0


def test_mapcollapse_tree():
    sdfg: dace.SDFG = tocollapse.to_sdfg()
    sdfg.simplify()
    sdfg.validate()
    assert sdfg.apply_transformations(MapCollapse) == 1
    sdfg.validate()


def test_mapcollapse_consolidated():
    sdfg: dace.SDFG = tocollapse.to_sdfg()
    sdfg.simplify()
    consolidate_edges(sdfg)
    sdfg.validate()
    assert sdfg.apply_transformations(MapCollapse) == 1
    sdfg.validate()


def test_mapcollapse_dep_edges():
    sdfg: dace.SDFG = memset_map.to_sdfg()
    sdfg.simplify()
    sdfg.validate()
    sdfg.apply_transformations_repeated(MapCollapse)

    maps = {n for n, g in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
    assert len(maps) == 1
    map_entry = maps.pop()
    assert map_entry.map.range == dace.subsets.Range([
        (0, 4, 1), (0, 4, 1), (0, 4, 1)
    ]), f"Excepted 3 ranges all with range 0:5:1, got {map_entry.map.range}"
    assert map_entry.map.params == ["i", "j", "k"]
    sdfg.validate()


def test_mapcollapse_shared_dynamic_input():
    """A map bound also read by its body needs distinct memlets after collapse."""
    sdfg = dace.SDFG('shared_dynamic_input')
    sdfg.add_array('bound', [1], dace.int32)
    sdfg.add_array('A', [2, 5], dace.int32)
    state = sdfg.add_state()
    outer, outer_exit = state.add_map('outer', {'i': '0:2'})
    inner, inner_exit = state.add_map('inner', {'j': '0:limit'})
    read = state.add_read('bound')
    write = state.add_write('A')
    tasklet = state.add_tasklet('copy_bound', {'b'}, {'out'}, 'out = b')
    state.add_memlet_path(read, outer, inner, tasklet,
                          dst_conn='b', memlet=dace.Memlet('bound[0]'))
    # The same outer connector feeds both the dynamic range and the tasklet.
    source = state.in_edges(inner)[0]
    inner.add_in_connector('limit', dace.int32)
    state.add_edge(outer, source.src_conn, inner, 'limit', dace.Memlet('bound[0]'))
    state.add_memlet_path(tasklet, inner_exit, outer_exit, write,
                          src_conn='out', memlet=dace.Memlet('A[i, j]'))
    sdfg.validate()
    assert sdfg.apply_transformations(MapCollapse) == 1
    sdfg.validate()
    memlets = [edge.data for edge in state.edges()]
    assert len({id(memlet) for memlet in memlets}) == len(memlets)


if __name__ == '__main__':
    test_mapcollapse_tree()
    test_mapcollapse_consolidated()
    test_mapcollapse_dep_edges()
