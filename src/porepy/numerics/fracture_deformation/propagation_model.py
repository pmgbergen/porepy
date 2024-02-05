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
from typing import Literal

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

    equation_system: pp.EquationSystem

    @abc.abstractmethod
    def __init__(self, mdg):
        # Abstract init, aimed at appeasing mypy. In practice, these attributes should
        # come from combining this class with a mechanical model.
        self.mdg = mdg
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
        self,
        domain: pp.Grid | pp.MortarGrid,
        d: dict,
        var: str,
        dofs: dict[Literal["cells", "faces", "nodes"], int],
    ) -> np.ndarray:
        """Initialize a new variable field with the right size for a new variable.

        Only cell variables are covered, extension to face and nodes should be
        straightforward.

        Parameters:
            domain: Subdomain or interface grid.
            d: Data dictionary. Should contain a field cell_index_map (an sps.spmatrix)
                which maps from old to new cell indices.
            var: Name of variable.
            dofs: Dictionary with number of DOFs per cell (or face/node). In practice,
                use the standard way of defining variables.

        Returns:
            vals: np.ndarray
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
        Map variables from old to new grids in d[pp.TIME_STEP_SOLUTIONS] and
        d[pp.ITERATE_SOLUTIONS].
        Also call update of self.assembler.update_dof_count and update the current
        solution vector accordingly.

        Newly created DOFs are assigned values by _initialize_new_variable_values, which
        for now returns zeros, but can be tailored for specific variables etc.

        Assumes only one stored time step vector, i.e.:
        time step indexes = iterate indexes = [0].

        Parameters:
            x: Solution vector, or other vector to be mapped.

        Raises:
            NotImplementedError

        Returns:
            Mapped solution vector.

        """
        # Obtain old solution vector. The values are extracted in the first two loops
        # and mapped and updated in the last two, after update_dof_count has been called.
        # for sd, data in self.mdg.subdomains(return_data=True):
        # First check if cells and faces have been updated, by checking if index
        # maps are available. If this is not the case, there is no need to map
        # variables.

        # Make temporary storage for old solution.

        for var in self.equation_system.get_variables():
            domain = var.domain
            if isinstance(domain, pp.Grid):
                # Subdomain
                data = self.mdg.subdomain_data(domain)
                is_subdomain = True
            elif isinstance(domain, pp.MortarGrid):
                # Interface
                data = self.mdg.interface_data(domain)
                is_subdomain = False
            else:
                raise ValueError("Unknown domain type for variable {}".format(var.name))

            if "cell_index_map" not in data:
                # No need to do anything
                continue
            if is_subdomain and "face_index_map" not in data:
                # No need to do anything. Both maps should be present for subdomains.
                continue

            cell_map: sps.spmatrix = data["cell_index_map"]
            data["old_solution"] = {}
            # Copy old solution vector values and store for next outer loop.

            data["old_solution"][var] = x[self.equation_system.dofs_of([var])]

            # Only cell-based dofs have been considered so far.
            # It should not be difficult to handle other types of variables,
            # but the need has not been there.
            dofs = self.equation_system._variable_dof_type[var]
            face_dof: int = dofs.get("faces", 0)
            node_dof: int = dofs.get("nodes", 0)
            if face_dof != 0 or node_dof != 0:
                raise NotImplementedError(
                    "Have only implemented variable mapping for face dofs"
                )

            cell_dof: int = dofs["cells"]

            # Map old solution
            mapping = sps.kron(cell_map, sps.eye(cell_dof))
            # New values for the new dofs.
            new_ind = self._new_dof_inds(mapping)
            new_vals = self._initialize_new_variable_values(
                domain, data, var.name, dofs
            )
            # Loop over stored time steps. Loop on keys to avoid bad practice of
            # changing looped quantity using items method.
            for ind in data[pp.TIME_STEP_SOLUTIONS][var.name].keys():
                values = pp.get_solution_values(
                    name=var.name, data=data, time_step_index=ind
                )
                values = mapping * values
                values[new_ind] = new_vals
                pp.set_solution_values(var.name, values, data, time_step_index=ind)

            # Repeat for iterate:
            for ind in data[pp.ITERATE_SOLUTIONS][var.name].keys():
                values = pp.get_solution_values(
                    name=var.name, data=data, iterate_index=ind
                )
                values = mapping * values
                values[new_ind] = new_vals
                pp.set_solution_values(var.name, values, data, iterate_index=ind)

        # Update the equation system's counting of dofs. Note that this update has no
        # proper testing and may not cover all necessary updates of dof management. In
        # particular, the image spaces of the equations are left untouched here.
        self.equation_system.update_variable_num_dofs()

        x_new = np.zeros(self.equation_system.num_dofs())
        # For each variable, map old solution and initialize for new DOFs.
        for var in self.equation_system.get_variables():
            domain = var.domain
            if isinstance(domain, pp.Grid):
                # Subdomain
                data = self.mdg.subdomain_data(domain)
                is_subdomain = True
            elif isinstance(domain, pp.MortarGrid):
                # Interface
                data = self.mdg.interface_data(domain)
                is_subdomain = False
            else:
                raise ValueError("Unknown domain type for variable {}".format(var.name))

            if "cell_index_map" not in data:
                # No need to do anything
                continue
            if is_subdomain and "face_index_map" not in data:
                # No need to do anything. Both maps should be present for subdomains.
                continue

            cell_map = data["cell_index_map"]

            # Update consist of two parts: First map the old solution to the new
            # grid, second populate newly formed cells.

            # Mapping of old variables.
            dofs = self.equation_system._variable_dof_type[var]
            cell_dof = dofs["cells"]
            mapping = sps.kron(cell_map, sps.eye(cell_dof))
            x_new[self.equation_system.dofs_of([var])] = (
                mapping * data["old_solution"][var]
            )

            # Index of newly formed variables.
            new_ind = self._new_dof_inds(mapping)
            # Values of newly formed variables.
            new_vals = self._initialize_new_variable_values(
                domain, data, var.name, dofs
            )
            # Update newly formed variables.
            x_new[self.equation_system.dofs_of([var])[new_ind]] = new_vals
            # Purge temporary storage of old solution.
            del data["old_solution"]

        # Store the mapped solution vector
        return x_new

    def _new_dof_inds(self, mapping: sps.spmatrix) -> np.ndarray:
        """
        The new DOFs/geometric entities are those which do not correspond to an
        old entity, i.e. their row is empty in the mapping matrix.

        Parameters:
            mapping: Mapping between old and new geometric entities.

        Returns:
            Boolean array of length n_new_entities, true for newly created entities.

        """
        # Find rows with only zeros. Can also get this information from the matrix
        # sparsity information, but this approach should be sufficient.
        return (np.sum(mapping, axis=1) == 0).nonzero()[0]
