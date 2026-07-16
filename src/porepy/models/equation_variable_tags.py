from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Final, Optional, TypeVar

import numpy as np
from scipy.sparse import csr_matrix

import porepy as pp


@dataclass(frozen=True)
class DomainFilter:
    @abstractmethod
    def filter(self, domain: pp.GridLike) -> bool:
        pass


@dataclass(frozen=True)
class Anywhere(DomainFilter):
    def filter(self, domain: pp.GridLike) -> bool:
        return True


@dataclass(frozen=True)
class EquationTag:
    name: str
    defined_on: DomainFilter = Anywhere()


@dataclass(frozen=True)
class VariableTag:
    name: str
    defined_on: DomainFilter = Anywhere()


class DefaultEquationTags:
    # Mass balance
    mass_balance = EquationTag(name=pp.SinglePhaseFlow.primary_equation_name())
    interface_darcy_flux = EquationTag(name="interface_darcy_flux_equation")
    well_flux = EquationTag(name="well_flux_equation")
    # Energy balance
    energy_balance = EquationTag(
        name=pp.energy_balance.TotalEnergyBalanceEquations.primary_equation_name()
    )
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


T = TypeVar("T")


def apply_cumulative_dof_offsets(
    dict_of_dofs: dict[T, np.ndarray],
) -> dict[T, np.ndarray]:
    result: dict[T, np.ndarray] = {}
    offset = 0
    for key, dofs in dict_of_dofs.items():
        result[key] = dofs + offset
        offset += len(dofs)
    return result



"""
Why do we need an equation identifier? Can't we just take the equation from the equation
system?

- What if outside model?
- ordering of dofs?
- compare them, find intersection, etc.
- give me indices of this equation (equation system does this, but is it convenient?)
- give me indices of these equations if we only use 

Why do we need a variable identifier? Is MdVariable bad?
- outside model
- compare them

"""

"""
Operations we want:
- give me residual for these tags
- give me jacobian for these tags
- give me solution for these tags
- give me indexer for these tags

- switch physics based on something
- set new t and dt ...

"""