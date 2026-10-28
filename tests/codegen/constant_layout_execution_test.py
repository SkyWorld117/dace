"""CPU and CUDA constant arrays follow their descriptor layout."""
import os
import dace
import numpy as np
import pytest

@pytest.mark.parametrize('gpu',[False,True])
@pytest.mark.parametrize('strides',[(1,2),(1,4),(3,1)])
def test_descriptor_layout(tmp_path,gpu,strides):
    if gpu and os.environ.get('DACE_TEST_GPU')!='1':
        pytest.skip('set DACE_TEST_GPU=1 on a CUDA node')
    table=np.arange(6,dtype=np.float64).reshape(2,3)
    sdfg=dace.SDFG('audited_constant_layout')
    sdfg.add_array('T',[2,3],dace.float64,strides=strides,transient=True)
    sdfg.add_constant('T',table,sdfg.arrays['T'])
    sdfg.add_array('out',[2,3],dace.float64)
    sdfg.add_state().add_mapped_tasklet('read',{'i':'0:2','j':'0:3'},
        {'t':dace.Memlet('T[i,j]')},'o=t',{'o':dace.Memlet('out[i,j]')},external_edges=True)
    if gpu:sdfg.apply_gpu_transformations()
    sdfg.build_folder=str(tmp_path/'build')
    out=np.zeros((2,3));sdfg(out=out)
    np.testing.assert_array_equal(out,table)


def test_multiple_kernels_do_not_duplicate_scalar_tables():
    sdfg=dace.SDFG('constant_scope_budget')
    sdfg.add_array('T',[3],dace.float64,transient=True)
    sdfg.add_constant('T',np.array([2.,3.,5.]),sdfg.arrays['T'])
    for index in range(256):sdfg.add_constant('coefficient_'+str(index),index,dace.data.Scalar(dace.int32))
    sdfg.add_array('out',[2,3],dace.float64)
    for row in range(2):
        state=sdfg.add_state(is_start_block=row==0)
        if row:sdfg.add_edge(previous,state,dace.InterstateEdge())
        state.add_mapped_tasklet('read_'+str(row),{'i':'0:3'},
            {'t':dace.Memlet('T[i]')},'o=t',{'o':dace.Memlet('out['+str(row)+',i]')},external_edges=True)
        previous=state
    sdfg.apply_gpu_transformations()
    code=next(c.clean_code for c in sdfg.generate_code() if c.language=='cu')
    assert code.count('constexpr int coefficient_0 =')==1
    # Two device-local arrays plus the host declaration used by host code.
    assert code.count('constexpr double T[3]')==3
