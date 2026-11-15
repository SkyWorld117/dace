"""Scalar storage and interstate assignments share one declaration."""
import dace
import numpy as np
import pytest
from dace.sdfg.state import ConditionalBlock,ControlFlowRegion
from dace.properties import CodeBlock


@pytest.mark.parametrize('lifetime',[dace.AllocationLifetime.Scope,dace.AllocationLifetime.SDFG])
def test_interstate_assignment_to_transient_scalar(tmp_path,lifetime):
    graph=dace.SDFG('interstate_scalar_storage')
    graph.add_symbol('x',dace.int32)
    graph.add_scalar('local_flag',dace.int32,transient=True,lifetime=lifetime)
    graph.add_array('out',[1],dace.int32)
    first=graph.add_state('first')
    definition=ConditionalBlock('definition',sdfg=graph)
    graph.add_node(definition)
    graph.add_edge(first,definition,dace.InterstateEdge())
    branch=ControlFlowRegion('set_scalar',sdfg=graph)
    before=branch.add_state('before')
    after=branch.add_state('after')
    branch.add_edge(before,after,dace.InterstateEdge(assignments={'local_flag':'x+7'}))
    definition.add_branch(CodeBlock('x > 0'),branch)
    block=ConditionalBlock('use_scalar',sdfg=graph)
    graph.add_node(block)
    graph.add_edge(definition,block,dace.InterstateEdge())
    for label,condition,expression in [('positive',CodeBlock('local_flag > 7 if x > 0 else False'),'x+1'),('other',None,'-3')]:
        branch=ControlFlowRegion(label,sdfg=graph)
        state=branch.add_state()
        tasklet=state.add_tasklet('write',{}, {'value'},'value = '+expression)
        state.add_edge(tasklet,'value',state.add_write('out'),None,dace.Memlet('out[0]'))
        block.add_branch(condition,branch)
    graph.build_folder=str(tmp_path/'build')
    compiled=graph.compile()
    for value in (-2,0,3,9,-1,1):
        result=np.zeros(1,dtype=np.int32)
        compiled(out=result,x=np.int32(value))
        np.testing.assert_array_equal(result,[value+1 if value>0 else -3])
