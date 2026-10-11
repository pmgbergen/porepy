import pytest

import porepy
from porepy.numerics.ad import Operator, OperatorSpace

_operations = porepy.numerics.ad._operations.Operations

operators = [
    ("+", _operations.add),
    ("-", _operations.sub),
    ("*", _operations.mul),
    ("/", _operations.div),
    ("**", _operations.pow),
]


@pytest.mark.parametrize("operator", operators)
def test_elementary_operations(operator):
    """Test that performing elementary arithmetic operations on operators return
    operator trees with the expected structure.

    The test does not consider evaluation of the numerical values of the operators.
    """
    # Generate two generic operators
    space = OperatorSpace.scalar()
    a = Operator(source=space, target=space)
    b = Operator(source=space, target=space)

    # Combine the operators with the provided operation.
    c = eval(f"a {operator[0]} b")

    # Check that the combined operator has the expected structure.
    assert c.operation == operator[1]

    # Need to check the id of the objects since the equality of pp.ad.Operator (or
    # rather the _key method which is called by eq) does not allow for generic void
    # operators like a and b.
    assert id(c.children[0]) == id(a)
    assert id(c.children[1]) == id(b)
