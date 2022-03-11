# TODO DOC
import porepy as pp
import numpy as np

def is_Edge(e) -> bool:
    """Implementation if isinstance(g, Tuple[pp.Grid, pp.Grid])."""

    return (
        isinstance(e, tuple) and
        len(e) == 2 and 
        isinstance(e[0], pp.Grid) and
        isinstance(e[1], pp.Grid)
    )

def isTupleOf_entity_str(pt, entity) -> bool:
    """Implementation of isinstance(pt, Tuple[Union[pp.Grid, List[pp.Grid]], str]), if entity = "node", and
    isinstance(pt, Tuple[Union[Tuple[pp.Grid, pp.Grid], List[Tuple[pp.Grid, pp.Grid]]], str]) if entity = "edge".
    """

    if entity not in ["node", "edge"]:
        raise ValueError(f"Entity type {entity} not supported.")

    return (
        # Check whether the input is a tuple with two entries
        isinstance(pt, tuple) and
        len(pt) == 2 and
        (
            # Check whether the first component is a grid or a list of grids
            entity == "node" and
            (
                isinstance(pt[0], pp.Grid) or
                isinstance(pt[0], list) and all([isinstance(g, pp.Grid) for g in pt[0]])
            ) or

            # Check whether the first component is an edge or a list of edges
            entity == "edge" and
            (
                is_Edge(pt[0]) or
                isinstance(pt[0], list) and all([is_Edge(g) for g in pt[0]])
            )
        ) and
        # Check whether the second component is a string
        isinstance(pt[1], str)
    )

def isTupleOf_entity_str_array(pt, entity) -> bool:
    """Implementation of isinstance(pt, Tuple[pp.Grid, str, np.ndarray] if entity="node",
    and isinstance(pt, Tuple[Tuple[pp.Grid, pp.Grid], str, np.ndarray]) if entity="edge"."""

    if entity not in ["node", "edge"]:
        raise ValueError(f"Entity type {entity} not supported.")

    return (
        isinstance(pt, tuple) and
        len(pt) == 3 and 
        (
            entity == "node" and isinstance(pt[0], pp.Grid) or
            entity == "edge" and is_Edge(pt[0])
        ) and
        isinstance(pt[1], str) and
        isinstance(pt[2], np.ndarray)
    )

def isTupleOf_str_array(pt) -> bool:
    """Implementation if isinstance(pt, Tuple[str, np.ndarray]."""

    return (
        isinstance(pt, tuple) and
        len(pt) == 2 and
        isinstance(pt[0], str) and
        isinstance(pt[1], np.ndarray)
    )
