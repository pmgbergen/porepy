"""This module defines `EquationTag` and `VariableTag`, which identify equations
and variables in a model, optionally restricted to specific grids.

These tags are intended for defining nonlinear and linear solution strategies. Unlike
`pp.ad.Variable`, `pp.ad.MixedDimensionalVariable`, and `pp.ad.Operator`, they can be
created without initializing a PorePy model, which requires constructing and meshing its
grids. This makes them less expensive to use and easier to incorporate into modular
code, particularly in tests.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import porepy as pp

__all__ = [
    "EquationTag",
    "VariableTag",
    "DefaultEquationTags",
    "DefaultVariableTags",
    "DomainFilter",
    "Anywhere",
]


@dataclass(frozen=True)
class DomainFilter(ABC):
    """A filter to restrict domains of a variable or an equation.

    Used in :class:`EquationTag` and :class:`VariableTag`."""

    @abstractmethod
    def filter(self, domain: pp.GridLike) -> bool:
        """Whether this `domain` is included in the domains where the equation /
        variable tag operates.

        """


@dataclass(frozen=True)
class Anywhere(DomainFilter):
    """A default filter that includes all domains."""

    def filter(self, domain: pp.GridLike) -> bool:
        return True


@dataclass(frozen=True)
class EquationTag:
    """An identifier of a single equation defined on multiple domains. Used to define
    nonlinear and linear solvers outside PorePy models, where identification by
    `pp.ad.Operator` or a list of `pp.GridLike` is unavailable.

    """

    name: str
    """Equation name."""
    defined_on: DomainFilter = Anywhere()
    """Operational domains of the equation identified by this tag. Possibly a subset of
    all domains where this equation is defined by a PorePy model.

    """


@dataclass(frozen=True)
class VariableTag:
    """An identifier of a single variable defined on multiple domains. Used to define
    nonlinear and linear solvers outside PorePy models, where identification by
    `pp.ad.Variable`, `pp.ad.MixedDimensionalVariable` or a list of `pp.GridLike` is
    unavailable.

    """

    name: str
    """Variable name."""
    defined_on: DomainFilter = Anywhere()
    """Operational domains of the variable identified by this tag. Possibly a subset of
    all domains where this variable is defined by a PorePy model.

    """


# Below are namespaces of default tags for equations and variables, default to PorePy.
# Compositional model equations and variables are omitted, since their names depend on
# the composition. See how to construct custom tags for them in
# schur_complement_reduction_linear_tracer_3p.py example.


class DefaultEquationTags:
    """A namespace for all known equations defined by standard PorePy models."""

    # Mass balance
    mass_balance = EquationTag(name="mass_balance_equation")
    interface_darcy_flux = EquationTag(name="interface_darcy_flux_equation")
    well_flux = EquationTag(name="well_flux_equation")
    # Energy balance
    energy_balance = EquationTag(name="energy_balance_equation")
    interface_fourier_flux = EquationTag(name="interface_fourier_flux_equation")
    interface_enthalpy_flux = EquationTag(name="interface_enthalpy_flux_equation")
    well_enthalpy_flux = EquationTag(name="well_enthalpy_flux_equation")
    # Momentum balance MPSA
    momentum_balance = EquationTag(name="momentum_balance_equation")
    # Momentum balance TPSA
    angular_momentum_balance = EquationTag(name="angular_momentum_balance_equation")
    solid_mass = EquationTag(name="solid_mass_equation")
    poromechanics_solid_mass = EquationTag(name="Solid_mass_equation_poromechanics")
    # Contact mechanics
    normal_fracture_deformation = EquationTag(
        name="normal_fracture_deformation_equation"
    )
    tangential_fracture_deformation = EquationTag(
        name="tangential_fracture_deformation_equation"
    )
    interface_force_balance = EquationTag(name="interface_force_balance_equation")
    # Fracture damage (?)
    dilation_damage = EquationTag(name="dilation_damage_equation")
    friction_damage = EquationTag(name="friction_damage_equation")


class DefaultVariableTags:
    """A namespace for all known variables defined by standard PorePy models."""

    # Mass balance
    pressure = VariableTag(name="pressure")
    interface_darcy_flux = VariableTag(name="interface_darcy_flux")
    well_flux = VariableTag(name="well_flux")
    # Energy balance
    temperature = VariableTag(name="temperature")
    enthalpy = VariableTag(name="enthalpy")
    interface_fourier_flux = VariableTag(name="interface_fourier_flux")
    interface_enthalpy_flux = VariableTag(name="interface_enthalpy_flux")
    well_enthalpy_flux = VariableTag(name="well_enthalpy_flux")
    # Momentum balance MPSA and TPSA
    displacement = VariableTag(name="u")
    # Momentum balance TPSA
    rotation_stress = VariableTag(name="rotation_stress")
    total_pressure = VariableTag(name="total_pressure")
    # Contact mechanics
    interface_displacement = VariableTag(name="u_interface")
    contact_traction = VariableTag(name="contact_traction")
    # Fracture damage (?)
    dilation_damage_history = VariableTag(name="dilation_damage_history")
    friction_damage_history = VariableTag(name="friction_damage_history")
