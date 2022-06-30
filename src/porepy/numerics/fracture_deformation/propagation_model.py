"""
Module indented for combining fracture propagation with complex multi-physics,
as represented in the model classes.

This module contains a partial implementation of propagation, confer other modules
for full implementation of the propagation.

WARNING: This should be considered experimental code and should be used with
    extreme caution. In particular, the code likely contains bugs, possibly of a
    severe character. Moreover, simulation of fracture propagation may cause
    numerical stability issues that it will likely take case-specific adaptations
    to resolve.

    The code structure for fracture propagation cannot be considered fixed, and it
    may be fundamentally restructured at unknown points in the future. If you use
    this functionality, please notify the maintainers (preferably by an email to
    Eirik.Keilegavlen@uib.no), so that we may keep your use-cases in mind if a major
    overhaul of the code is undertaken.

"""
from __future__ import annotations

import abc

import numpy as np
import scipy.sparse as sps

import porepy as pp


class FracturePropagation(abc.ABC):
    """Abstract base class for fracture propagation methods.

    The class is indented used together with a subclass of AbstractModel,
    using dual inheritance.

    WARNING: This should be considered experimental code and should be used with
        extreme caution. In particular, the code likely contains bugs, possibly of a
        severe character. Moreover, simulation of fracture propagation may cause
        numerical stability issues that it will likely take case-specific adaptations
        to resolve.

        The code structure for fracture propagation cannot be considered fixed, and it
        may be fundamentally restructured at unknown points in the future. If you use
        this functionality, please notify the maintainers (Eirik.Keilegavlen@uib.no),
        so that we may keep your use-cases in mind if a major overhaul of the code is
        undertaken.

    Known subclasses are:
        ConformingFracturePropagation

    """

    @abc.abstractmethod
    def __init__(self, assembler):
        # Abstract init, aimed at appeasing mypy. In practice, these attributes should
        # come from combining this class with a mechanical model.
        self.assembler = assembler
        self.mdg = self.assembler.mdg
        self.nd = self.mdg.dim_max()

    @abc.abstractmethod
    def evaluate_propagation(self) -> None:
        """Evaluate propagation of fractures based on the current solution.

        Implementation of the method will differ between propagation criteria,
        whether the adaptive meshing is applied etc.
        """

    @abc.abstractmethod
    def has_propagated(self) -> bool:
        """Should return True if fractures were propagated in the previous step."""

    def _initialize_new_variable_values(
        self, sd: pp.Grid, d: dict, var: str, dofs: dict[str, int]
    ) -> np.ndarray:
        """
        Initialize a new variable field with the right size for a new variable.

        Only cell variables are covered, extension to face and nodes should be
        straightforward.

        Parameters
        ----------
        sd : pp.Grid
            Subdomain grid.
        d : Dict
            Data dictionary. Should contain a field cell_index_map (an sps.spmatrix)
            which maps from old to new cell indices.
        var : str
            Name of variable.
        dofs : Dict[str, int]
            Dictionary with number of DOFs per cell (or face/node). In practice,
            use the standard way of defining variables.

        Returns
        -------
        vals : np.ndarray
            Values for the new DOFs.

        """
        # Number of cell dofs for this variable
        cell_dof = dofs.get("cells")

        # Number of new variables is given by the size of the cell map.
        cell_map: sps.spmatrix = d["cell_index_map"]
        n_new = cell_map.shape[0] - cell_map.shape[1]

        vals = np.zeros(n_new * cell_dof)
        return vals

    def _map_variables(self, x: np.ndarray) -> np.ndarray:
        """
        Map variables from old to new grids in d[pp.STATE] and d[pp.STATE][pp.ITERATE].
        Also call update of self.assembler.update_dof_count and update the current
        solution vector accordingly.

        Newly created DOFs are assigned values by _initialize_new_variable_values,
        which for now returns zeros, but can be tailored for specific variables
        etc.

        Parameters
        ------
        x: np.ndarray
            Solution vector, or other vector to be mapped.

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
        for sd, data in self.mdg.subdomains(return_data=True):

            # First check if cells and faces have been updated, by checking if index maps are
            # available. If this is not the case, there is no need to map variables.
            if not ("cell_index_map" in data and "face_index_map" in data):
                continue

            cell_map: sps.spmatrix = data["cell_index_map"]

            data[pp.STATE]["old_solution"] = {}
            for var, dofs in data[pp.PRIMARY_VARIABLES].items():
                # Copy old solution vector values
                data[pp.STATE]["old_solution"][var] = x[
                    self.assembler._dof_manager.grid_and_variable_to_dofs(sd, var)
                ]

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
                data[pp.STATE][var] = mapping * data[pp.STATE][var]

                # Initialize new values
                new_ind = self._new_dof_inds(mapping)
                new_vals = self._initialize_new_variable_values(sd, data, var, dofs)
                data[pp.STATE][var][new_ind] = new_vals

                # Repeat for iterate:
                if var in data[pp.STATE][pp.ITERATE].keys():
                    data[pp.STATE][pp.ITERATE][var] = (
                        mapping * data[pp.STATE][pp.ITERATE][var]
                    )
                    data[pp.STATE][pp.ITERATE][var][new_ind] = new_vals

        for intf, data in self.mdg.interfaces(return_data=True):

            # Check if the mortar grid geometry has been updated.
            if "cell_index_map" not in data:
                # No need to do anything
                continue

            data[pp.STATE]["old_solution"] = {}
            cell_map = data["cell_index_map"]

            for var, dofs in data[pp.PRIMARY_VARIABLES].items():
                # Copy old solution vector values
                data[pp.STATE]["old_solution"][var] = x[
                    self.assembler._dof_manager.grid_and_variable_to_dofs(intf, var)
                ]

                # Only cell-based dofs have been considered so far.
                cell_dof = dofs.get("cells")

                # Map old solution
                mapping = sps.kron(cell_map, sps.eye(cell_dof))
                data[pp.STATE][var] = mapping * data[pp.STATE][var]

                # Initialize new values
                new_ind = self._new_dof_inds(mapping)
                new_vals = self._initialize_new_variable_values(intf, data, var, dofs)
                data[pp.STATE][var][new_ind] = new_vals

                # Repeat for iterate
                if var in data[pp.STATE][pp.ITERATE].keys():
                    data[pp.STATE][pp.ITERATE][var] = (
                        mapping * data[pp.STATE][pp.ITERATE][var]
                    )
                    data[pp.STATE][pp.ITERATE][var][new_ind] = new_vals

        # Update the assembler's counting of dofs
        self.assembler.update_dof_count()

        x_new = np.zeros(self.assembler.num_dof())
        # For each grid-variable pair, map old solution and initialize for new
        # DOFs.
        for sd, data in self.mdg.subdomains(return_data=True):
            # Check if there has been updates to this grid.
            if not ("cell_index_map" in data and "face_index_map" in data):
                continue

            cell_map = data["cell_index_map"]

            for var, dofs in data[pp.PRIMARY_VARIABLES].items():
                # Update consist of two parts: First map the old solution to the new
                # grid, second populate newly formed cells.

                # Mapping of old variables
                cell_dof = dofs.get("cells")
                mapping = sps.kron(cell_map, sps.eye(cell_dof))
                x_new[
                    self.assembler._dof_manager.grid_and_variable_to_dofs(sd, var)
                ] = (mapping * data[pp.STATE]["old_solution"][var])

                # Index of newly formed variables
                new_ind = self._new_dof_inds(mapping)
                # Values of newly formed variables
                new_vals = self._initialize_new_variable_values(sd, data, var, dofs)
                # Update newly formed variables
                x_new[
                    self.assembler._dof_manager.grid_and_variable_to_dofs(sd, var)[
                        new_ind
                    ]
                ] = new_vals

        for intf, data in self.mdg.interfaces(return_data=True):
            # Same procedure as for subdomains, see above for comments
            if "cell_index_map" not in data:
                continue

            cell_map = data["cell_index_map"]

            for var, dofs in data[pp.PRIMARY_VARIABLES].items():
                cell_dof = dofs.get("cells")
                mapping = sps.kron(cell_map, sps.eye(cell_dof))
                x_new[
                    self.assembler._dof_manager.grid_and_variable_to_dofs(intf, var)
                ] = (mapping * data[pp.STATE]["old_solution"][var])
                new_ind = self._new_dof_inds(mapping)
                new_vals = self._initialize_new_variable_values(intf, data, var, dofs)
                x_new[
                    self.assembler._dof_manager.grid_and_variable_to_dofs(intf, var)[
                        new_ind
                    ]
                ] = new_vals

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
