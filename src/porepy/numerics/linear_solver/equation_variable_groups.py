from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import porepy as pp
from porepy.models.energy_balance import TotalEnergyBalanceEquations
from porepy.models.fluid_mass_balance import FluidMassBalanceEquations
from porepy.numerics.ad.operators import MixedDimensionalVariable


class EquationNames(Enum):
    """Enum for the names of the equations in the model."""

    MASS_BALANCE = FluidMassBalanceEquations.primary_equation_name()
    ENERGY_BALANCE = TotalEnergyBalanceEquations.primary_equation_name()
    INTERFACE_DARCY_FLUX = "interface_darcy_flux_equation"

    INTERFACE_ENTHALPY_FLUX = "interface_enthalpy_flux_equation"
    INTERFACE_FOURIER_FLUX = "interface_fourier_flux_equation"

    MECHANICS = "momentum_balance_equation"
    INTERFACE_FORCE_BALANCE = "interface_force_balance_equation"
    CONTACT = "contact_mechanics_equation"
    CONTACT_NORMAL = "normal_fracture_deformation_equation"
    CONTACT_TANGENTIAL = "tangential_fracture_deformation_equation"

    WELL_FLUX = "well_flux_equation"
    WELL_ENTHALPY_FLUX = "well_enthalpy_flux_equation"


@dataclass
class EquationOnDomains:
    """A PorePy equation defined on a collection of domains."""

    name: str
    """Should be identical to the PorePy equation name."""
    domains: list[pp.GridLike]
    """A list of domains of definition for the given equation."""


@dataclass(frozen=True)
class EquationVariableGroup(ABC):
    """A base class for all groups. This represents a diagonal submatrix in a block
    matrix, thus the number of DoFs for the equation and the variable should match.

    This is a dataclass, because it is: (i) comparable, (ii) hashable and (iii)
    immutable. The subclasses may contain a state, e.g., `ComponentGroup("CO2")`. The
    state must fully define what data generates by PorePy in `equation_group` and
    `variable_group` methods.

    """

    @abstractmethod
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        pass

    @abstractmethod
    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        pass

    def equation_name(self, model: pp.PorePyModel) -> str:
        """Human-readible equation name for debugging and diagnostics. Should not
        necessarily match the PorePy name. Feel free to override.

        """
        return self.equation_group(model).name

    def variable_name(self, model: pp.PorePyModel) -> str:
        """Human-readible variable name for debugging and diagnostics. Should not
        necessarily match the PorePy name. Feel free to override.

        """
        return self.variable_group(model).name


@dataclass(frozen=True)
class InterfaceDarcyFluxGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.INTERFACE_DARCY_FLUX.value
        return EquationOnDomains(name=name, domains=model.mdg.interfaces())

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.interface_darcy_flux(model.mdg.interfaces())


@dataclass(frozen=True)
class InterfaceEnthalpyFluxGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.INTERFACE_ENTHALPY_FLUX.value
        return EquationOnDomains(name=name, domains=model.mdg.interfaces())

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.interface_enthalpy_flux(model.mdg.interfaces())


@dataclass(frozen=True)
class InterfaceFourierFluxGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.INTERFACE_FOURIER_FLUX.value
        return EquationOnDomains(name=name, domains=model.mdg.interfaces())

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.interface_fourier_flux(model.mdg.interfaces())


@dataclass(frozen=True)
class WellFluxGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.WELL_FLUX.value
        return EquationOnDomains(name=name, domains=model.mdg.interfaces())

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.well_flux(model.mdg.interfaces())


@dataclass(frozen=True)
class WellEnthalpyFluxGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.WELL_ENTHALPY_FLUX.value
        return EquationOnDomains(name=name, domains=model.mdg.interfaces())

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.well_enthalpy_flux(model.mdg.interfaces())


@dataclass(frozen=True)
class InterfaceForceBalanceGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        name = EquationNames.INTERFACE_FORCE_BALANCE.value
        return EquationOnDomains(
            name=name, domains=model.mdg.interfaces(dim=model.nd - 1)
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        interfaces = model.mdg.interfaces(dim=model.nd - 1)
        return model.interface_displacement(interfaces)


@dataclass(frozen=True)
class MechanicsGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        return EquationOnDomains(
            name=EquationNames.MECHANICS.value,
            domains=model.mdg.subdomains(dim=model.nd),
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        subdomains = model.mdg.subdomains(dim=model.nd)
        return model.displacement(subdomains)


@dataclass(frozen=True)
class ContactMechanicsGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        # PorePy has no single contact mechanics equation: it has two: normal and
        # tangential. However, we treat them as one in the linear solver context. This
        # is an exception, which is manually treated in the DofManager.
        return EquationOnDomains(
            name=EquationNames.CONTACT.value,
            domains=model.mdg.subdomains(dim=model.nd - 1),
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        subdomains = model.mdg.subdomains(dim=model.nd - 1)
        return model.contact_traction(subdomains)


@dataclass(frozen=True)
class MassBalancePressureMatrixGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        return EquationOnDomains(
            name=EquationNames.MASS_BALANCE.value,
            domains=model.mdg.subdomains(dim=model.nd),
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.pressure(model.mdg.subdomains(dim=model.nd))

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.MASS_BALANCE.value + " (matrix)"

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.pressure_variable + " (matrix)"


@dataclass(frozen=True)
class MassBalancePressureFracturesGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        return EquationOnDomains(
            name=EquationNames.MASS_BALANCE.value,
            domains=model.mdg.subdomains(dim=model.nd - 1),
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.pressure(model.mdg.subdomains(dim=model.nd - 1))

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.MASS_BALANCE.value + " (fractures)"

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.pressure_variable + " (fractures)"


@dataclass(frozen=True)
class MassBalancePressureIntersectionsGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        intersections = [
            sd
            for dim in reversed(range(0, model.nd - 1))
            for sd in model.mdg.subdomains(dim=dim)
        ]
        return EquationOnDomains(
            name=EquationNames.MASS_BALANCE.value,
            domains=intersections,
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        intersections = [
            sd
            for dim in reversed(range(0, model.nd - 1))
            for sd in model.mdg.subdomains(dim=dim)
        ]
        return model.pressure(intersections)

    def equation_name(self, model: pp.PorePyModel) -> str:
        return EquationNames.MASS_BALANCE.value + " (intersections)"

    def variable_name(self, model: pp.PorePyModel) -> str:
        return model.pressure_variable + " (intersections)"


@dataclass(frozen=True)
class MassBalancePressureGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        return EquationOnDomains(
            name=EquationNames.MASS_BALANCE.value, domains=model.mdg.subdomains()
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.pressure(model.mdg.subdomains())


@dataclass(frozen=True)
class EnergyBalanceTemperatureGroup(EquationVariableGroup):
    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        return EquationOnDomains(
            name=EquationNames.ENERGY_BALANCE.value, domains=model.mdg.subdomains()
        )

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        return model.temperature(model.mdg.subdomains())


@dataclass(frozen=True)
class DefinitionDomain(ABC):
    @abstractmethod
    def domains(self, model: pp.PorePyModel) -> pp.GridLikeSequence:
        pass


@dataclass(frozen=True)
class OnWell(DefinitionDomain):
    well_type: Literal["production", "injection"]

    def domains(self, model: pp.PorePyModel) -> pp.GridLikeSequence:
        well, not_well = model._filter_wells(model.mdg.subdomains(), self.well_type)
        return well


@dataclass(frozen=True)
class NotOnWell(DefinitionDomain):
    well_type: Literal["production", "injection"]

    def domains(self, model: pp.PorePyModel) -> pp.GridLikeSequence:
        wells, not_well = model._filter_wells(model.mdg.subdomains(), self.well_type)
        return not_well


@dataclass(frozen=True)
class CustomEquationVariableGroup(EquationVariableGroup):
    eq_name: str
    var_name: str
    defined_on: Optional[DefinitionDomain] = None

    def equation_group(self, model: pp.PorePyModel) -> EquationOnDomains:
        if self.defined_on is not None:
            domains = self.defined_on.domains(model)
        else:
            domains = model.mdg.subdomains()
        return EquationOnDomains(name=self.eq_name, domains=domains)

    def variable_group(self, model: pp.PorePyModel) -> MixedDimensionalVariable:
        if self.defined_on is not None:
            domains = self.defined_on.domains(model)
        else:
            domains = model.mdg.subdomains()
        return model.equation_system.md_variable(name=self.var_name, domains=domains)
