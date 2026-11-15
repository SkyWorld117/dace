"""Stable duplicate removal preserves the first objects and sequence type."""
import pytest
from dace.dtypes import deduplicate


@pytest.mark.parametrize('container',[list,tuple])
def test_order_and_first_representative(container):
    value=container([3,True,3,1,False,0,2,True])
    result=deduplicate(value)
    assert type(result) is container
    assert list(result)==[3,True,False,2]
    assert result[1] is True and result[2] is False
    assert deduplicate(container())==container()


def test_unique_items_do_not_require_quadratic_equality_searches():
    class Item:
        comparisons=0
        def __init__(self,key):self.key=key
        def __hash__(self):return self.key
        def __eq__(self,other):
            Item.comparisons+=1
            return isinstance(other,Item) and self.key==other.key
    values=[Item(i) for i in range(1000)]
    result=deduplicate(values+[values[0],values[-1]])
    assert len(result)==len(values)
    assert all(a is b for a,b in zip(values,result))
    assert Item.comparisons<=len(values)
