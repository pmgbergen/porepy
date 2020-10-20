"""
Module indended for combining fracture propagation with complex multiphysics,
as represented in the model classes.

This module contains a partial implementation of propagation, confer other modules
for full implementation of the propagation.

WARNING: This should be considered experimental code and should be used with
    extreme caution. In particular, the code likely contains bugs, possibly of a
    severe character. Moreover, simulation of fracture propagation may cause
    numerical stability issues that it will likely take case-specific adaptations
    to resolve.



"""
import abc
import numpy as np
import scipy.sparse as sps
import porepy as pp
from typing import Dict


class FracturePropagation(abc.ABC):
    """ Abstract base class for fracture propagation methods.

    The class is indended used together with a subclass of AbstractModel,
    using dual inheritance.

    Known subclasses are:
        ConformingFracturePropagation

    """

    @abc.abstractmethod
    def evaluate_propagation(self) -> None:
        """ Evaluate propagation of fractures based on the current solution.

        Impementation of the method will differ between propagation criretia,
        whether the adaptive meshing is applied etc.
        """

    @abc.abstractmethod
    def has_propagated(self) -> bool:
        """ Should return True if fractures were propagated in the previous step.
        """

    @abc.abstractmethod
    def keep_propagating(self) -> bool:
        """ Whether or not another propagation step should be performed. 
        Typically depends on has_propagated and/or propagation_index.
        """

    def _initialize_new_variable_values(
        self, g: pp.Grid, d: Dict, var: str, dofs: int
    ) -> np.ndarray:
        """
        

        Parameters
        ----------
        g : pp.Grid
            Grid.
        d : Dict
            Data dictionary.
        var : str
            Name of variable.
        dofs : int
            Number of DOFs per cell (or face/node).

        Returns
        -------
        vals : np.ndarray
            Values for the new DOFs.

        """
        cell_dof = dofs.get("cells")
        n_new = d["cell_index_map"].shape[0] - d["cell_index_map"].shape[1]
        vals = np.zeros(n_new * cell_dof)
        return vals

    def _map_variables(self, x) -> None:
        """
        Map variables from old to new grids in d[pp.STATE] and d[pp.STATE][pp.ITERATE].
        Also call update of self.assembler.update_dof_count and update the current
        solution vector accordingly.
        
        Newly created DOFs are assigned values by _initialize_new_variable_values,
        which for now returns zeros,, but can be tailored for specific variables
        etc.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Obtain old solution vector. The values are extracted in the first two loops
        # and mapped and updated in the last two, after update_dof_count has been called.
        for g, d in self.gb:
            if not ("cell_index_map" in d and "face_index_map" in d):
                continue

            cell_map = d["cell_index_map"]
            face_map = d["face_index_map"]
            d[pp.STATE]["old_solution"] = {}
            for var, dofs in d[pp.PRIMARY_VARIABLES].items():
                # Copy old solution vector values
                d[pp.STATE]["old_solution"][var] = x[self.assembler.dof_ind(g, var)]

                # Only cell-based dofs have been considered so far.
                # It should not be difficult to handle other types of variables,
                # but the need has not been there.
                face_dof: int = dofs.get("faces", 0)
                node_dof: int = dofs.get("nodes", 0)
                if face_dof != 0 or node_dof != 0:
                    raise NotImplementedError(
                        "Have only implemented variable mapping for face dofs"
                    )

                cell_dof: int = dofs.get("cells")

                # Map old solution
                mapping = sps.kron(cell_map, sps.eye(cell_dof))
                d[pp.STATE][var] = mapping * d[pp.STATE][var]
                # Initialize new values
                new_ind = self._new_dof_inds(mapping)
                new_vals = self._initialize_new_variable_values(g, d, var, dofs)
                d[pp.STATE][var][new_ind] = new_vals

                # Repeat for iterate:
                if var in d[pp.STATE][pp.ITERATE].keys():
                    d[pp.STATE][pp.ITERATE][var] = (
                        mapping * d[pp.STATE][pp.ITERATE][var]
                    )
                    d[pp.STATE][pp.ITERATE][var][new_ind] = new_vals

        for e, d in self.gb.edges():
            if not "cell_index_map" in d:
                continue
            d[pp.STATE]["old_solution"] = {}
            cell_map = d["cell_index_map"]

            for var, dofs in d[pp.PRIMARY_VARIABLES].items():
                # Copy old solution vector values
                d[pp.STATE]["old_solution"][var] = x[self.assembler.dof_ind(e, var)]

                # Only cell-based dofs have been considered so far.
                cell_dof: int = dofs.get("cells")

                # Map old solution
                mapping = sps.kron(cell_map, sps.eye(cell_dof))
                d[pp.STATE][var] = mapping * d[pp.STATE][var]

                # Initialize new values
                new_ind = self._new_dof_inds(mapping)
                new_vals = self._initialize_new_variable_values(e, d, var, dofs)
                d[pp.STATE][var][new_ind] = new_vals

                # Repeat for iterate
                if var in d[pp.STATE][pp.ITERATE].keys():
                    d[pp.STATE][pp.ITERATE][var] = (
                        mapping * d[pp.STATE][pp.ITERATE][var]
                    )
                    d[pp.STATE][pp.ITERATE][var][new_ind] = new_vals

        # Update the assembler's counting of dofs
        self.assembler.update_dof_count()
        x_new = np.zeros(self.assembler.num_dof())
        # For each grid-variable pair, map old solution and initialize for new
        # DOFs.
        for g, d in self.gb:
            if not ("cell_index_map" in d and "face_index_map" in d):
                continue

            cell_map = d["cell_index_map"]

            for var, dofs in d[pp.PRIMARY_VARIABLES].items():
                cell_dof: int = dofs.get("cells")
                mapping = sps.kron(cell_map, sps.eye(cell_dof))
                x_new[self.assembler.dof_ind(g, var)] = (
                    mapping * d[pp.STATE]["old_solution"][var]
                )
                new_ind = self._new_dof_inds(mapping)
                new_vals = self._initialize_new_variable_values(g, d, var, dofs)
                x_new[new_ind] = new_vals

        for e, d in self.gb.edges():
            if not "cell_index_map" in d:
                continue

            cell_map = d["cell_index_map"]

            for var, dofs in d[pp.PRIMARY_VARIABLES].items():
                cell_dof: int = dofs.get("cells")
                mapping = sps.kron(cell_map, sps.eye(cell_dof))
                x_new[self.assembler.dof_ind(e, var)] = (
                    mapping * d[pp.STATE]["old_solution"][var]
                )
                new_ind = self._new_dof_inds(mapping)
                new_vals = self._initialize_new_variable_values(e, d, var, dofs)
                x_new[new_ind] = new_vals

        # Store the mapped solution vector
        return x_new

    def _new_dof_inds(self, mapping: sps.spmatrix) -> np.ndarray:
        """
        The new DOFs/geometric entities are those which do not correspond to an
        old entity, i.e. their row is empty in the mapping matrix.

        Parameters
        ----------
        mapping : sps.spmatrix
            Mapping between old and new geometric entities.

        Returns
        -------
        np.ndarray
            Boolean vector of length n_new_entities, true for newly created entities.

        """
        # Find rows with only zeros. Can also get this information from the matrix
        # sparsity information, but this approach should be sufficient.
        return (np.sum(mapping, axis=1) == 0).nonzero()[0]
