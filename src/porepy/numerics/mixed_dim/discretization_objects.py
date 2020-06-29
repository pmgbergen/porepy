from typing import Tuple, Any, List

from porepy.numerics.discretization import Discretization
from pydantic import BaseModel, validator, root_validator
import porepy as pp


class NodeDiscretization(BaseModel):
    """ Node discretization object"""
    var: str
    term_key: str
    term: Discretization


class CouplingDiscretization(BaseModel):
    """ Coupling discretization object

    An edge, edge variable and edge term is required.
    At least one of
        (g_h, master_var, master_term_key), or
        (g_l, slave_var, slave_term_key)
    is required.
    """
    coupling_key: str

    g_h: pp.Grid = None
    master_var: str = None
    master_term_key: str = None

    g_l: pp.Grid = None
    slave_var: str = None
    slave_term_key: str = None

    edge: Tuple[pp.Grid, pp.Grid]
    edge_var: str
    edge_term: Any

    class Config:
        # Needed to allow pp.Grid objects
        arbitrary_types_allowed = True

    # Validate the input parameters
    @root_validator()
    def check_at_least_gh_or_gl_is_set(self, values):
        g_h, master_var, master_term_key = values["g_h"], values["master_var"], values["master_term_key"]
        g_l, slave_var, slave_term_key = values["g_l"], values["slave_var"], values["slave_term_key"]
        if not g_h:
            # If a master grid is not provided, a slave grid has to be provided
            assert g_l and slave_var and slave_term_key
        elif not g_l:
            # Vice, versa: If a slave grid is not provided, a master grid has to be provided
            assert g_h and master_var and master_term_key
        else:
            # If both are provided, check that all variables and terms are also provided
            assert g_h and master_var and master_term_key and g_l and slave_var and slave_term_key
        return values

    @validator("edge")
    def check_gl_gh_in_edge(self, v, values):
        g_l, g_h = values["g_l"], values["g_h"]
        if g_l:
            assert g_l in v
        if g_h:
            assert g_h in v


def coupling_discr_dict_to_model(discr: dict) -> List[CouplingDiscretization]:
    """ Convert a coupling discretization dictionary to a CouplingDiscretization Model

    Parameters
    ----------
    discr : dict
        The contents of d[pp.COUPLING_DISCRETIZATION]

    Examples
    --------
    This method converts a dictionary of the form:
        d[pp.COUPLING_DISCRETIZATION] = {
                    "scalar_coupling_term": {                           <-- coupling_key
                        g_h: ("pressure", "diffusion"),                 <-- (master_var, master_term_key)
                        g_l: ("pressure", "diffusion"),                 <-- (slave_var, slave_term_key)
                        e: (
                            "mortar_pressure",                          <-- edge_var
                            pp.RobinCoupling("flow", pp.Mpfa("flow"),   <-- edge_term
                        ),
                    },
                }

    to a list of CouplingDiscretization objects.

    """
    result = []
    for coupling_term, coupling_data in discr.items():
        # coupling data is a dictionary where g_h, g_l and e are keys,
        # and their respective values are tuples of (variable, term or term_key)
        grids: List[pp.Grid] = [g for g in coupling_data if isinstance(g, pp.Grid)]
        edges: List[Tuple[pp.Grid, pp.Grid]] = [e for e in coupling_data if isinstance(e, Tuple)]
        assert len(edges) == 1, "You can only have one edge"
        edge: Tuple[pp.Grid, pp.Grid] = edges[0]

        if len(grids) == 1:
            # Either a master grid or slave grid is defined, not both
            # Find out whether the grid corresponds to a master or slave grid
            g = grids[0]
            edge_dims: List[int] = [g.dim for g in edge]
            if g.dim == max(edge_dims):
                g_h = g
                master_coupling_data = coupling_data[g]
                g_l = slave_coupling_data = None
            else:
                g_h = master_coupling_data = None
                g_l = g
                slave_coupling_data = coupling_data[g]
        elif len(grids) == 2:
            # You have a master and a slave grid
            sorted(grids, key=lambda g: g.dim)  # Sort grids: Lowest dim first
            g_h = grids[0]
            master_coupling_data = coupling_data[g_h]
            g_l = grids[1]
            slave_coupling_data = coupling_data[g_l]
            assert g_l in edge and g_h in edge, "The slave and master grids should correspond to the edge grids"
            assert g_l.dim < g_h.dim, "The slave and master grids should be of different dimension"
        else:
            raise ValueError("You can at most couple to two grids")

        cd = CouplingDiscretization(
            coupling_key=coupling_term,
            g_h=g_h,
            master_var=master_coupling_data[0],
            master_term_key=master_coupling_data[1],
            g_l=g_l,
            slave_var=slave_coupling_data[0],
            slave_term_key=slave_coupling_data[1],
            edge=edge,
            edge_var=coupling_data[edge][0],
            edge_term=coupling_data[edge][1],
        )
        result.append(cd)

    return result


def node_discr_dict_to_model(discr: dict) -> List[NodeDiscretization]:
    """ Convert the dictionary d[pp.DISCRETIZATION] to a list of NodeDiscretization"""
    result = []
    for variable, terms in discr.items():
        for term_key, term in terms.items():
            nd = NodeDiscretization(var=variable, term_key=term_key, term=term)
            result.append(nd)
    return result
