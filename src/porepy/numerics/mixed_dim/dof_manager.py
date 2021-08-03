""" Implementation of a degree of freedom manager.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp


class DofManager:
    """
    block_dof: Is a dictionary with keys that are either
        Tuple[pp.Grid, variable_name: str] for nodes in the GridBucket, or
        Tuple[Tuple[pp.Grid, pp.Grid], str] for edges in the GridBucket.

        The values in block_dof are integers 0, 1, ..., that identify the block
        index of this specific grid (or edge) - variable combination.

    full_dof: Is a np.ndarray of int that store the number of degrees of
        freedom per key-item pair in block_dof. Thus
          len(full_dof) == len(block_dof).
        The total size of the global system is full_dof.sum()

    """

    def __init__(self, gb: pp.GridBucket) -> None:

        self.gb = gb

        # Counter for block index
        block_dof_counter = 0

        # Dictionary that maps node/edge + variable combination to an index.
        block_dof: Dict[Tuple[Union[pp.Grid, Tuple[pp.Grid, pp.Grid]], str], int] = {}

        # Storage for number of dofs per variable per node/edge, with respect
        # to the ordering specified in block_dof
        full_dof: List[int] = []

        for g, d in gb:
            if pp.PRIMARY_VARIABLES not in d:
                continue

            for local_var, local_dofs in d[pp.PRIMARY_VARIABLES].items():
                # First assign a block index.
                # Note that the keys in the dictionary is a tuple, with a grid
                # and a variable name (str)
                block_dof[(g, local_var)] = block_dof_counter
                block_dof_counter += 1

                # Count number of dofs for this variable on this grid and store it.
                # The number of dofs for each grid entitiy type defaults to zero.
                total_local_dofs = (
                    g.num_cells * local_dofs.get("cells", 0)
                    + g.num_faces * local_dofs.get("faces", 0)
                    + g.num_nodes * local_dofs.get("nodes", 0)
                )
                full_dof.append(total_local_dofs)

        for e, d in gb.edges():
            if pp.PRIMARY_VARIABLES not in d:
                continue

            mg: pp.MortarGrid = d["mortar_grid"]

            for local_var, local_dofs in d[pp.PRIMARY_VARIABLES].items():

                # First count the number of dofs per variable. Note that the
                # identifier here is a tuple of the edge and a variable str.
                block_dof[(e, local_var)] = block_dof_counter
                block_dof_counter += 1

                # We only allow for cell variables on the mortar grid.
                # This will not change in the foreseeable future
                total_local_dofs = mg.num_cells * local_dofs.get("cells", 0)
                full_dof.append(total_local_dofs)

        # Array version of the number of dofs per node/edge and variable
        self.full_dof: np.ndarray = np.array(full_dof)
        self.block_dof: Dict[
            Tuple[Union[pp.Grid, Tuple[pp.Grid, pp.Grid]], str], int
        ] = block_dof

    def dof_ind(
        self, g: Union[pp.Grid, Tuple[pp.Grid, pp.Grid]], name: str
    ) -> np.ndarray:
        """Get the indices in the global system of variables associated with a
        given node / edge (in the GridBucket sense) and a given variable.

        Parameters:
            g (pp.Grid or pp.GridBucket edge): Either a grid, or an edge in the
                GridBucket.
            name (str): Name of a variable. Should be an active variable.

        Returns:
            np.array (int): Index of degrees of freedom for this variable.

        """
        block_ind = self.block_dof[(g, name)]
        dof_start = np.hstack((0, np.cumsum(self.full_dof)))
        return np.arange(dof_start[block_ind], dof_start[block_ind + 1])

    def num_dofs(
        self,
        g: Optional[Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]] = None,
        var: Optional[Union[List[str], str]] = None,
    ) -> int:
        """Get the number of degrees of freedom for a specific grid and/or variable.

        Four scenarios are possible:
          * If a grid (or interface) is specified, the total size for all variables of
            this grid is specified.
          * If a variable is specified, the total size for instances of this variable
            over all grids is returned.
          * If both grid and variable are specified, the variable size for the unique
            combination is returned.
          * If neither variable nor variable are specified, the total system size is
            returned.

        Parameters:
            g (pp.Grid or tuple of grids): Grid used in inquiery.
            var (str): Variable name.

        Returns:
            int: Size of subsystem.

        """
        if g is None and var is None:
            return np.sum(self.full_dof)  # type: ignore
        elif var is None:
            num = 0
            for grid, variable in self.block_dof:
                if grid == g:
                    bi = self.block_dof[(grid, variable)]
                    num += self.full_dof[bi]
            return num
        elif g is None:
            num = 0
            if not isinstance(var, list):  # type: ignore
                var = [var]  # type: ignore
            for grid, variable in self.block_dof:
                if variable in var:
                    bi = self.block_dof[(grid, variable)]
                    num += self.full_dof[bi]
            return num
        else:
            if isinstance(var, list):
                # TODO Include something here!
                raise RuntimeError("Not implemented!")
            return self.full_dof[(g, var)]

    def distribute_variable(
        self,
        values: np.ndarray,
        variable_names: Optional[List[str]] = None,
        additive: bool = False,
        to_iterate: bool = False,
    ) -> None:
        """Distribute a vector to the nodes and edges in the GridBucket.

        The intended use is to split a multi-physics solution vector into its
        component parts.

        Parameters:
            values (np.array): Vector to be split. It is assumed that it corresponds
                to the ordering implied in block_dof and full_dof, e.g. that it is
                the solution of a linear system assembled with the assembler.
            variable_names (list of str, optional): Names of the variable to be
                distributed. If not provided, all variables found in block_dof
                will be distributed
            additive (bool, optional): If True, the variables are added to the current
                state or iterate, instead of overwrite the existing value.
            to_iterate (bool, optional): If True, distribute to iterates, and not the
                state itself. Set to True inside a non-linear scheme (Newton), False
                at the end of a time step.

        """
        if variable_names is None:
            variable_names = []
            for pair in self.block_dof.keys():
                variable_names.append(pair[1])

        dof = np.cumsum(np.append(0, np.asarray(self.full_dof)))

        for var_name in set(variable_names):
            for pair, bi in self.block_dof.items():
                g = pair[0]
                name = pair[1]
                if name != var_name:
                    continue
                else:
                    if isinstance(g, tuple):
                        # This is really an edge
                        data = self.gb.edge_props(g)
                    else:
                        data = self.gb.node_props(g)

                    if pp.STATE in data.keys():
                        vals = values[dof[bi] : dof[bi + 1]]
                        if additive:
                            if to_iterate:
                                vals += data[pp.STATE][pp.ITERATE][var_name]
                            else:
                                vals += data[pp.STATE][var_name]

                        if to_iterate:
                            data[pp.STATE][pp.ITERATE][var_name] = vals
                        else:
                            data[pp.STATE][var_name] = vals
                    else:
                        # If no values exist, there is othing to add to
                        if to_iterate:
                            data[pp.STATE] = {
                                pp.ITERATE: {var_name: values[dof[bi] : dof[bi + 1]]}
                            }
                        else:
                            data[pp.STATE] = {var_name: values[dof[bi] : dof[bi + 1]]}

    def assemble_variable(
        self, variable_names: Optional[List[str]] = None, from_iterate: bool = False
    ) -> np.ndarray:
        if variable_names is None:
            variable_names = []
            for pair in self.block_dof.keys():
                variable_names.append(pair[1])

        dof = np.cumsum(np.append(0, np.asarray(self.full_dof)))
        vals = np.zeros(dof[-1])

        for var_name in set(variable_names):
            for pair, bi in self.block_dof.items():
                g = pair[0]
                name = pair[1]
                if name != var_name:
                    continue
                else:
                    if isinstance(g, tuple):
                        # This is really an edge
                        data = self.gb.edge_props(g)
                    else:
                        data = self.gb.node_props(g)

                    if from_iterate:
                        vals[dof[bi] : dof[bi + 1]] = data[pp.STATE][pp.ITERATE][
                            var_name
                        ]
                    else:
                        vals[dof[bi] : dof[bi + 1]] = data[pp.STATE][var_name]
        return vals

    def transform_dofs(
        self, dofs: np.ndarray, var: Optional[list] = None
    ) -> np.ndarray:
        """Transforms dofs associated to full list of dofs to a restricted list of dofs."""

        # TODO this procedure should only be performed once! One could consider storing
        # the projection matrix.
        # Double-check whether dofs are actually represented by var.
        total_dofs: int = np.sum(self.full_dof)  # type: ignore
        if var is not None:
            is_var_dofs = np.zeros(total_dofs, dtype=bool)
            if not isinstance(var, list):
                var = [var]
            for grid, variable in self.block_dof:
                if variable in var:
                    var_dof_ind = self.dof_ind(grid, variable)
                    is_var_dofs[var_dof_ind] = True
            assert all(is_var_dofs[dofs])

        # Input dofs in the context of all dofs managed by DofManager
        total_dofs: int = np.sum(self.full_dof)  # type: ignore
        global_dofs = np.zeros(total_dofs)
        global_dofs[dofs] = 1

        # Projection matrix from space of all dofs to restricted list of var
        num_dofs = self.num_dofs(var=var)  # type: ignore
        projection = sps.coo_matrix(
            (np.arange(num_dofs), (np.arange(num_dofs), dofs)),
            shape=(num_dofs, total_dofs),
        )

        # Return the restricted dofs
        return projection * global_dofs

    def __str__(self) -> str:
        grid_likes = [key[0] for key in self.block_dof]
        unique_grids = list(set(grid_likes))

        num_grids = 0
        num_interfaces = 0
        for g in unique_grids:
            if isinstance(g, pp.Grid):
                num_grids += 1
            else:
                num_interfaces += 1

        names = [key[1] for key in self.block_dof]
        unique_vars = list(set(names))
        s = (
            f"Degree of freedom manager for {num_grids} "
            f"subdomains and {num_interfaces} interfaces.\n"
            f"Total number of degrees of freedom: {self.num_dofs()}\n"
            "Total number of subdomain and interface variables:"
            f"{len(self.block_dof)}\n"
            f"Variable names: {unique_vars}"
        )

        return s

    def __repr__(self) -> str:

        grid_likes = [key[0] for key in self.block_dof]
        unique_grids = list(set(grid_likes))

        num_grids = 0
        num_interfaces = 0

        dim_max = -1
        dim_min = 4

        for g in unique_grids:
            if isinstance(g, pp.Grid):
                num_grids += 1
                dim_max = max(dim_max, g.dim)
                dim_min = min(dim_min, g.dim)
            else:
                num_interfaces += 1

        s = (
            f"Degree of freedom manager with in total {self.full_dof.sum()} dofs"
            f" on {num_grids} subdomains and {num_interfaces} interface variables.\n"
            f"Maximum grid dimension: {dim_max}\n"
            f"Minimum grid dimension: {dim_min}\n"
        )

        return s
