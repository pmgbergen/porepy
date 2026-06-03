"""This module defines the class DofManager - a layer of translation between a PorePy
model and the equation-variable groups defined in `equation_variable_groups.py`.
Given the `MassBalancePressureGroup()` as example, the DofManager can tell us:
- Is this group present in the problem?
- If yes, what PorePy DoFs correspond to this equation?

This is done by:
```
dof_manager = DofManager(...)
mass_balance_group = dof_manager.indices_of_groups([MassBalancePressureGroup()])[0]

dofs_mass_balance_eq = dof_manager.eq_dofs()[mass_balance_group]
dofs_pressure_var = dof_manager.var_dofs()[mass_balance_group]

# These dofs now can be used to slice the matrix, produced by the PorePy model:
mat, rhs = model.linear_system

submatrix_mass_balance_pressure = mat[dofs_mass_balance_eq, dofs_pressure_var]
rhs_mass_balance = rhs[dofs_mass_balance_eq]
```

"""

from __future__ import annotations

from collections import defaultdict
from itertools import count
from typing import Optional
from weakref import ReferenceType, ref

import numpy as np

import porepy as pp
from porepy.numerics.linear_solver.block_linear_system import (
    concatenate_dof_indices,
)
from porepy.numerics.linear_solver.equation_variable_groups import (
    ContactMechanicsGroup,
    EquationNames,
    EquationOnDomains,
    EquationVariableGroup,
)

__all__ = ["DofManager"]


class DofManager:
    """Takes care of translation of PorePy equations and variables (from EquationSystem
    format) to group indices, suited to construct a `BlockLinearSystem` to be solved
    with an iterative solver.

    One particular "exception" or "edge case" is the contact mechanics equation, which
    requires extra care, since PorePy treats tangential and normal equations as
    different entities, and it requires reordering before solving with an iterative
    solver. This reordering is done in this class.

    A general problem would outsource the contact reordering to a subclass, but right
    now we have no reason to do so, since it's the only known exception.

    """

    def __init__(self, model: pp.PorePyModel, groups: list[EquationVariableGroup]):
        """Constructs the DoFs mapping for the passed groups of equations and
        variables.

        Raises:
            ValueError: If a group defines a variable or an equation on the same domain
            more than once.

        """
        # We need a weak reference here to avoid a reference cycle, which can lead to a
        # memory leak. The weak reference is alive until the PorePy model is alive.
        # DofManager is used inside a PorePy model, so a DofManager without an active
        # PorePy model would not make sense anyway.
        self._model: ReferenceType[pp.PorePyModel] = ref(model)
        """The PorePy model of the given problem."""

        self._groups: list[EquationVariableGroup] = groups
        """Groups that define the DofManager."""

        # Extracting equation and variable names, these are used for debugging purposes.
        self._equation_names: list[str] = [g.equation_name(model) for g in groups]
        self._variable_names: list[str] = [g.variable_name(model) for g in groups]

        # Collecting and validation equation and variable groups. This ensures no
        # duplicates. More validation regarding meaningful dofs is made in
        # BlockLinearSystem constructor.

        equation_groups = [g.equation_group(model) for g in groups]
        self._equation_groups: list[EquationOnDomains] = equation_groups
        _validate_equation_groups(equation_groups=equation_groups)

        variable_groups = [g.variable_group(model) for g in groups]
        self._variable_groups: list[pp.ad.MixedDimensionalVariable] = variable_groups
        _validate_variable_groups(variable_groups=variable_groups)

        # Assembling DoFs that correspond to each group:
        # 1. PorePy provides us with a list of arrays, each array corresponds to the
        #   DoFs of a single equation/variable on a single (not mixed-dimensional)
        #   grid.
        # 2. We construct a mapping from what PorePy provided to the groups, which
        #   equations/variables on which collection grids we will treat monolithically
        #   in this DofManager.
        # 3. We concatenate arrays of DoFs, so that now a single array correspond to a
        #   single group.

        # First, we treat equations:
        eq_dofs_porepy_order = self._eq_dofs_porepy_order()
        mapping_equation_groups = self._equation_block_indices()
        self._eq_dofs: list[np.ndarray] = [
            concatenate_dof_indices([eq_dofs_porepy_order[i] for i in dofs_in_group])
            for dofs_in_group in mapping_equation_groups
        ]
        """List of arrays, i-th array contains the DoFs of the i-th equation group."""

        # Second, we treat variables:
        var_dofs_porepy_order = self._var_dofs_porepy_order()
        mapping_variable_groups = self._variable_block_indices()
        self._var_dofs: list[np.ndarray] = [
            concatenate_dof_indices([var_dofs_porepy_order[i] for i in dofs_in_group])
            for dofs_in_group in mapping_variable_groups
        ]
        """List of arrays, i-th array contains the DoFs of the i-th variable group."""

        # Contact mechanics permutation.
        try:
            contact_group = self.indices_of_groups([ContactMechanicsGroup()])[0]
        except ValueError:
            pass  # Do nothing if no contact groups is present.
        else:
            self._eq_dofs[contact_group] = self._permute_contact_dofs(contact_group)

    @property
    def model(self) -> pp_solvers.IterativeSolverMixin:
        """The PorePy model of the given problem."""
        model = self._model()
        if model is None:
            # This should never happen, as the DofManager is meant to be used together
            # with a PorePy model.
            raise ValueError("The underlying PorePy model is destroyed.")
        return model

    def groups(self) -> list[EquationVariableGroup]:
        """Groups of equations and variables that define the DofManager."""
        return self._groups

    def indices_of_groups(self, groups: list[EquationVariableGroup]):
        """Return unique numerical identifiers of the passed groups.

        Raises:
            ValueError: If any of the groups is not found in this DofManager, or if
                repeating groups are requestsd.

        """
        indices = [self._groups.index(x) for x in groups]
        if len(indices) != len(set(indices)):
            # YZ does not see a situation when this can be desired behavior, but
            # clearly sees how it can lead to bugs later on. This can be caused by
            # comparison of custom EquationVariableGroups, when they are not equal but
            # treated as so.
            raise ValueError(f"Repeating group indices are produced by {self.groups}.")
        return indices

    def equation_names(self) -> list[str]:
        """Get the names of equations in the DofManager. These names are not generally
        equal to the PorePy model equation names, and are intended for debugging.

        Returns:
            A list of strings containing the names of equations in the DofManager.

        """
        return self._equation_names

    def variable_names(self) -> list[str]:
        """Get the names of variables in the DofManager. These names are not generally
        equal to the PorePy model variable names, and are intended for debugging.

        Returns:
            A list of strings containing the names of equations in the DofManager.

        """
        return self._variable_names

    def eq_dofs(self) -> list[np.ndarray]:
        """List of arrays, i-th array contains the DoFs of the i-th equation group."""
        return self._eq_dofs

    def var_dofs(self) -> list[np.ndarray]:
        """List of arrays, i-th array contains the DoFs of the i-th variable group."""
        return self._var_dofs

    def build_projection(self, groups: Optional[list[EquationVariableGroup]] = None):
        if groups is None:
            groups = self.groups()
        group_indices = self.indices_of_groups(groups)
        eq_dofs = self.eq_dofs()
        var_dofs = self.var_dofs()

        projection_row = []
        projection_col = []
        for group_idx in group_indices:
            projection_row.append(eq_dofs[group_idx])
            projection_col.append(var_dofs[group_idx])
        return projection_row, projection_col

    def _eq_dofs_porepy_order(self) -> list[np.ndarray]:
        """Equation degrees of freedom (rows of the Jacobian) in the PorePy order (how
        they are arranged in the PorePy model).

        Returns:
            List of numpy arrays. Each array contains the degrees of freedom for a
                single equation on a single (not mixed-dimensional) grid.

        """
        eq_dofs: list[np.ndarray] = []
        model = self.model
        offset = 0
        for data in model.equation_system._equation_image_space_composition.values():
            local_offset = 0
            for dofs in data.values():
                eq_dofs.append(dofs + offset)
                local_offset += len(dofs)
            offset += local_offset
        return eq_dofs

    def _var_dofs_porepy_order(self) -> list[np.ndarray]:
        """Variable degrees of freedom (columns of the Jacobian) in the PorePy order
        (how they are arranged in the PorePy model).

        Returns:
            List of numpy arrays. Each array contains the degrees of freedom for a
                single variable on a single (not mixed-dimensional) grid.

        """
        model = self.model

        var_dofs: list[np.ndarray] = []
        for var in model.equation_system.variables:
            var_dofs.append(model.equation_system.dofs_of([var]))
        return var_dofs

    def _permute_contact_dofs(self, contact_group: int) -> np.ndarray:
        """Get a permuted array of the DoFs in the contact group.

        This is used to reorder the equations so that the contact equations for single
        fracture cells form a diagonal block.

        The PorePy arrangement in 3D is:

            [C_n^0, C_n^1, ..., C_n^K, C_y^0, C_z^0, C_y^1, C_z^1, ..., C_z^K, C_z^k],

        where `C_n` is a normal component, `C_y` and `C_z` are two tangential
        components. The superscript corresponds to cell index. We permute it to

            `[C_n^0, C_y^0, C_z^0, ..., C_n^K, C_y^K, C_z^K]`.

        Parameters:
            contact_group: The group index of the contact mechanics equations.

        Raises:
            ValueError: If the model dimension is not 2 or 3.

        Returns:
            A numpy array with the permuted DoFs for the contact group.

        """
        # Get the dofs of the contact mechanics equations.
        dofs_contact = self._eq_dofs[contact_group]

        if len(dofs_contact) == 0:
            # If contact is formally present, but no equations are defined for it,
            # no permutation is needed.
            return dofs_contact

        nd = self.model.nd
        num_contact_cells = dofs_contact.size // nd

        # Extracting normal equation DoFs: [C_n^0, C_n^1, ...]. They go first, as
        # defined in the DofManager._equation_block_indices method.
        dofs_normal = dofs_contact[:num_contact_cells]

        # Extracting tangential equation DoFs.
        if nd == 2:
            # For 2D, it is a single DoF per cell: [C_y^0, C_y^1, ...].
            dofs_tangential = [dofs_contact[num_contact_cells:]]
        elif nd == 3:
            # For 3D, it is two DoFs per equation, already interleaved:
            # [C_y^0, C_z^0, C_y^1, C_z^1, ...]. We extract two arrays: for C_y and C_z.
            dofs_tangential = [
                dofs_contact[num_contact_cells::2],
                dofs_contact[num_contact_cells + 1 :: 2],
            ]
        return np.vstack([dofs_normal] + dofs_tangential).ravel("F")

    def _variable_block_indices(self) -> list[list[int]]:
        """Used to assemble the index that will later help accessing the submatrix
        corresponding to a group of variables, which may include one or more variable.

        Example: Group 0 corresponds to the pressure on all the subdomains. It will
        contain indices [0, 1, 2] which point to the pressure variable dofs on sd1, sd2
        and sd3, respectively. Combination of different variables in one group is also
        possible.

        Returns:
            List of lists of integers. i-th inner list contains the indices of the
                variables defined in the i-th group of `self._variable_groups`.

        """
        # Create a 0-based index for each variable.
        counter = count(0)
        variable_to_idx = {
            var: next(counter) for var in self.model.equation_system.variables
        }
        indices = []
        for md_var in self._variable_groups:
            # If we ever get a variable in here, we need to handle it directly, and
            # not call sub_vars.
            assert isinstance(md_var, pp.ad.MixedDimensionalVariable)
            indices.append([variable_to_idx.pop(var) for var in md_var.sub_vars])
        assert len(variable_to_idx) == 0, "Some variables are not used."
        return indices

    def _equation_block_indices(self) -> list[list[int]]:
        """Assembles the index that will later help accessing the submatrix
        corresponding to a group of equation, which may include one or more equation.

        The contact mechanics equation is defined in PorePy as two equations: normal and
        tangential. Here, we compose them into a single equation group.

        Returns:
            List of lists of integers. i-th inner list contains the indices of the
                equations in defined in the i-th item in `self._equation_groups`.
                The indices refer to the block indices defined in
                model.equation_system._equation_image_space_composition.

        """
        # Assign a unique index to each equation-domain pair.
        equation_to_idx: dict[tuple[str, pp.GridLike], int] = {}
        idx: int = 0
        for (
            eq_name,
            domains,
        ) in self.model.equation_system._equation_image_space_composition.items():
            for domain in domains:
                equation_to_idx[(eq_name, domain)] = idx
                idx += 1

        indices: list[list[int]] = []
        # The outer loop define different groups of equations (to become blocks in the
        # block matrix).
        for equation_on_domains in self._equation_groups:
            eq_name = equation_on_domains.name
            domains = equation_on_domains.domains
            # Items in the group will contain a single equation defined on one or more
            # domains (subdomains or interfaces). Loop over equations an over all their
            # domains to add the indices to the group.
            indices_group: list[int] = []
            for domain in domains:
                if (eq_name, domain) in equation_to_idx:
                    indices_group.append(equation_to_idx.pop((eq_name, domain)))

            # Exception: Special treatment for contact.
            if eq_name == EquationNames.CONTACT.value:
                # PorePy model contains 2 equations: one for normal and one for
                # tangential contact mechanics. If the "CONTACT" group is passed, we
                # treat them as a single group.
                for eq_name in [
                    EquationNames.CONTACT_NORMAL.value,
                    EquationNames.CONTACT_TANGENTIAL.value,
                ]:
                    # First, we append the normal equation in all domains of definition.
                    # Second - the tangential equation. The method
                    # DofManager._permute_contact_dofs relies on this order.
                    for domain in domains:
                        if (eq_name, domain) in equation_to_idx:
                            indices_group.append(equation_to_idx.pop((eq_name, domain)))

            indices.append(indices_group)

        # TODO EK: Added this assert just to verify that my understanding of the
        # function is correct. Delete it later.
        assert len(indices) == len(self._equation_groups)
        if len(equation_to_idx) != 0:
            raise ValueError(
                "Some equations are not used on some subdomains: "
                f"{set([k[0] for k in equation_to_idx.keys()])}"
            )

        return indices


def _validate_equation_groups(equation_groups: list[EquationOnDomains]):
    """Ensures no duplicates in equation_groups.

    Raises:
        ValueError: If a a pair of equation_names and subdomains is encountered more
            than once.

    """
    # The key is a tuple (equation_name: str, domain: pp.GridLike). The value is how
    # many times we encountered this key. We make sure we encounter each combination
    # only once.
    equation_domain_counter = defaultdict(lambda: 0)
    for group in equation_groups:
        for domain in group.domains:
            equation_domain_counter[(group.name, domain)] += 1

    for (eq_name, domain), count in equation_domain_counter.items():
        if count > 1:
            raise ValueError(
                f"{eq_name}, {domain} encountered more than once. Check the"
                " equation groups."
            )


def _validate_variable_groups(variable_groups: list[pp.ad.MixedDimensionalVariable]):
    """Ensures no duplicates in variable_groups.

    Raises:
        ValueError: If a variable defined on a single domain is encountered more than
        once.

    """
    # The key is a tuple (variable_name: str, domain: pp.GridLike). The value is how
    # many times we encountered this key. We make sure we encounter each combination
    # only once.
    variable_domain_counter = defaultdict(lambda: 0)
    for md_var in variable_groups:
        for domain in md_var.domains:
            variable_domain_counter[(md_var.name, domain)] += 1

    for (var_name, domain), count in variable_domain_counter.items():
        if count > 1:
            raise ValueError(
                f"Variable group encountered more than once: {var_name} on {domain}"
            )
