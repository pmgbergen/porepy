""" Contains the private base class for all phases.

This module is not imported by default into the composite subpackage,
since the user is not supposed to be able to create phase classes, only the composition class.

"""
from __future__ import annotations

import abc
from typing import Dict, Generator, List

import numpy as np
import porepy as pp

from .component import Component, VarLike
from ._composite_utils import R_IDEAL, T_REF, P_REF, CP_REF, V_REF, H_REF

__all__ = ["Phase", "IncompressibleFluid", "IdealGas"]


class Phase(abc.ABC):
    """Private base class for phases in a multiphase multicomponent mixture.

    The term 'phase' includes both, states of matter and general fields.
    A phase is identified by the (time-dependent) region/volume it occupies and a
    respective velocity field (or flux) in that region.

    This class is only meant to be instantiated by a Composition, since the number of phases
    is an unknown in the thermodynamic equilibrium problem.

    Phases have abstract physical properties, dependent on the thermodynamic state and the
    composition. The composition variables (molar fractions of present components)
    can be accessed by internal reference.

    Parameters:
        name: given name for this phase. Used as an unique identifier for singletons.
        ad_system (optional): If given, this class will use the AD framework and the respective
            mixed-dimensional domain to represent fractions cell-wise in each subdomain.

    """

    # contains per mdg the singleton, using the given name as a unique identifier
    __ad_singletons: Dict[pp.MixedDimensionalGrid, Dict[str, Phase]] = dict()
    # flag if a singleton has recently been accessed, to skip re-instantiation
    __singleton_accessed: bool = False

    def __new__(cls, name: str, ad_system: pp.ad.ADSystem) -> Phase:
        # check for AD singletons per grid
        mdg = ad_system.dof_manager.mdg
        if mdg in Phase.__ad_singletons:
            if name in Phase.__ad_singletons[mdg]:
                # flag that the singleton has been accessed and return it.
                Phase.__singleton_accessed = True
                return Phase.__ad_singletons[mdg][name]
        else:
            Phase.__ad_singletons.update({mdg: dict()})

        # create a new instance and store it
        new_instance = super().__new__(cls)
        Phase.__ad_singletons[mdg].update({name: new_instance})

        return new_instance

    def __init__(self, name: str, ad_system: pp.ad.ADSystem) -> None:
        # skipping re-instantiation if class if __new__ returned the previous reference
        if Phase.__singleton_accessed:
            Phase.__singleton_accessed = False
            return

        super().__init__()

        ### PUBLIC

        self.ad_system: pp.ad.ADSystem = ad_system
        """The AD system optionally passed at instantiation."""

        #### PRIVATE

        self._name = name
        self._present_components: List[pp.composite.Component] = list()

        # Instantiate saturation and molar phase fraction (secondary variables)
        self._s: pp.ad.MergedVariable = ad_system.create_variable(self.saturation_name)
        self._fraction: pp.ad.MergedVariable = ad_system.create_variable(self.fraction_name)
        nc = ad_system.dof_manager.mdg.num_subdomain_cells()
        ad_system.set_var_values(self.saturation_name, np.zeros(nc), True)
        ad_system.set_var_values(self.fraction_name, np.zeros(nc), True)
        # contains extended fractional values per present component name (key)
        self._ext_composition: Dict[str, pp.ad.MergedVariable] = dict()
        # contains regular fractional values per present component name (key)
        self._composition: Dict[str, pp.ad.MergedVariable] = dict()

    def __iter__(self) -> Generator[pp.composite.Component, None, None]:
        """Generator over components present in this phase.

        Notes:
            The order from this iterator is used for choosing e.g. the values in a
            list of 'numpy.array' when setting initial values.
            Use the order returned here every time you deal with component-related values
            for components in this phase.

        Yields:
            components present in this phase.
        """
        for substance in self._present_components:
            yield substance

    @property
    def name(self) -> str:
        """Name of this phase given at instantiation."""
        return self._name

    @property
    def num_components(self) -> int:
        """Number of present (added) components."""
        return len(self._present_components)

    @property
    def saturation_name(self) -> str:
        """Name for the saturation variable, given by the general symbol and :meth:`name`."""
        return f"s_{self.name}"

    @property
    def fraction_name(self) -> str:
        """Name for the molar fraction variable, given by the general symbol and :meth:`name`."""
        return f"y_{self.name}"

    @property
    def saturation(self) -> pp.ad.MergedVariable:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        Returns:
            saturation (volumetric fraction), a secondary variable on the whole domain.
            Indicates how much of the (local) volume is occupied by this phase (per cell).
            It is supposed to represent the value at thermodynamic equilibrium.

        """
        return self._s

    @property
    def fraction(self) -> pp.ad.MergedVariable:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        Returns:
            molar phase fraction, a secondary variable on the whole domain.
            Indicates how many of the total moles belong to this phase (per cell).
            It is supposed to represent the value at thermodynamic equilibrium.

        """
        return self._fraction

    def ext_fraction_of_component(
        self, component: pp.composite.Component
    ) -> pp.ad.MergedVariable:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        If a phase is present (phase fraction is strictly positive), the extended component
        fraction (this one) coincides with the regular component fraction.
        If a phase vanishes (phase fraction is zero), the extended fractions represent
        non-physical values at equilibrium.
        The extended phase composition does not fulfill unity.
        In the case of a vanished phase, the regular phase composition is obtained by
        re-normalizing the extended phase composition, such that unity is fulfilled.

        Notes:
            Currently there is no checking if the component was added to the phase.
            If it was not added, zero is simply returned.

        Parameters:
            component: a component present in this phase

        Returns:
            extended molar fraction of a component in this phase,
            a secondary variable on the whole domain (cell-wise).
            Indicates how many of the moles in this phase belong to the component.
            It is supposed to represent the value at thermodynamic equilibrium.
            Returns always zero if a component is not modelled (added) to this phase.

        """
        return self._ext_composition.get(self.ext_component_fraction_name(component), 0.0)

    def ext_component_fraction_name(self, component: pp.composite.Component) -> str:
        """
        Parameters:
            component: component for which the respective name is requested.

        Returns:
            name of the respective variable, given by the general symbol, the
            component name and the phase name.

        """
        return f"xi_{component.name}_{self.name}"

    def fraction_of_component(self, component: pp.composite.Component) -> pp.ad.MergedVariable:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        If a phase is present (phase fraction is strictly positive), the regular component
        fraction coincides with the extended component fraction.
        If a phase vanishes (phase fraction is zero), the regular component fraction (this one)
        is obtained by re-normalizing the extended component fraction.

        Notes:
            Currently there is no checking if the component was added to the phase.
            If it was not added, zero is simply returned.

        Parameters:
            component: a component present in this phase

        Returns:
            extended molar fraction of a component in this phase,
            a secondary variable on the whole domain (cell-wise).
            Indicates how many of the moles in this phase belong to the component.
            It is supposed to represent the value at thermodynamic equilibrium.
            Returns always zero if a component is not modelled (added) to this phase.

        """
        return self._composition.get(self.ext_component_fraction_name(component), 0.0)

    def component_fraction_name(self, component: pp.composite.Component) -> str:
        """
        Parameters:
            component: component for which the respective name is requested.

        Returns:
            name of the respective variable, given by the general symbol, the
            component name and the phase name.

        """
        return f"chi_{component.name}_{self.name}"

    def add_component(
        self,
        component: Component | list[Component],
    ) -> None:
        """Adds components which are expected by the modeler in this phase.

        If a component was already added, nothing happens. Components appear uniquely in a
        phase.

        Parameters:
            a component, or list of components, which are expected in this phase.

        Raises:
            RuntimeError: if the component was instantiated using a different AD system than
            the one used for the phase.

        """
        if isinstance(component, pp.composite.Component):
            component = [component]  # type: ignore
        present_components = [ps.name for ps in self._present_components]

        for comp in component:
            # sanity check when using the AD framework
            if self.ad_system != comp.ad_system:
                raise RuntimeError(
                    f"Component '{comp.name}' instantiated with a different AD system."
                )
            # skip already present components:
            if comp.name in present_components:
                continue

            # create compositional variables for the component in this phase
            ext_fraction_name = self.ext_component_fraction_name(comp)
            fraction_name = self.component_fraction_name(comp)
            ext_comp_fraction = self.ad_system.create_variable(ext_fraction_name)
            comp_fraction = self.ad_system.create_variable(fraction_name)

            # set fractional values to zero
            nc = self.ad_system.dof_manager.mdg.num_subdomain_cells()
            self.ad_system.set_var_values(ext_fraction_name, np.zeros(nc), True)
            self.ad_system.set_var_values(fraction_name, np.zeros(nc), True)

            # store reference to present substance
            self._present_components.append(comp)
            # store the compositional variable
            self._ext_composition.update({ext_fraction_name: ext_comp_fraction})
            self._composition.update({fraction_name: comp_fraction})

    ### Physical properties -------------------------------------------------------------------

    def mass_density(self, p: VarLike, T: VarLike) -> pp.ad.MergedVariable:
        """Uses the  molar mass in combination with the molar masses and fractions
        of components in this phase, to compute the mass density of the phase.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kg / REV]

        Parameters:
            p: pressure
            T: temperature

        Returns: mass density of this phase.

        """
        weight = 0.0
        # add the mass-weighted fraction for each present substance.
        # if no components are present, the weight is zero!
        for component in self._present_components:
            weight += component.molar_mass() * self._composition[component.name]

        # Multiply the mass weight with the molar density and return the operator
        return weight * self.density(p, T)

    @abc.abstractmethod
    def density(self, p: VarLike, T: VarLike) -> VarLike:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [mol / REV]

        Parameters:
            p: pressure
            T: temperature

        Returns: mass density of this phase.

        """
        pass

    @abc.abstractmethod
    def specific_enthalpy(self, p: VarLike, T: VarLike) -> VarLike:
        """
        | Math. Dimension:        scalar
        | Phys.Dimension:         [kJ / mol / K]

        Parameters:
            p: pressure
            T: temperature

        Returns: specific molar enthalpy of this phase.

        """
        pass

    @abc.abstractmethod
    def dynamic_viscosity(self, p: VarLike, T: VarLike) -> VarLike:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [mol / m / s]

        Parameters:
            p: pressure
            T: temperature

        Returns: dynamic viscosity of this phase.

        """
        pass

    @abc.abstractmethod
    def thermal_conductivity(self, p: VarLike, T: VarLike) -> VarLike:
        """
        | Math. Dimension:    2nd-order tensor
        | Phys. Dimension:    [W / m / K]

        Parameters:
            p: pressure
            T: temperature

        Returns: thermal conductivity of this phase.

        """
        pass


# TODO ADify properly
class IncompressibleFluid(Phase):
    """Ideal, Incompressible fluid with constant density of 1 mole per V_REF.
    
    The EOS is reduced to
    
    const rho = 1 / V_REF ( = 1 / V )
    V = V_REF
    
    """

    def density(self, p, T):
          # TODO p / p is a hack to create cell-wise dofs
        return pp.ad.Scalar(1000000. / V_REF) * p / p

    def specific_enthalpy(self, p, T):
        return H_REF + CP_REF * (T - T_REF) + V_REF * (p - P_REF)

    def dynamic_viscosity(self, p, T):
        return pp.ad.Scalar(1.) 

    def thermal_conductivity(self, p, T):
        return pp.ad.Scalar(1.)


class IdealGas(Phase):
    """Ideal water vapor phase with EoS:
    
     rho = n / V  = p / (R * T)

    """

    def density(self, p, T):
        return p / (T * R_IDEAL)

    def specific_enthalpy(self, p, T):
        # enthalpy at reference state is
        # h = u + p / rho(p,T)
        # which due to the ideal gas law simplifies to
        # h = u + R * T
        return H_REF + CP_REF * (T - T_REF)

    def dynamic_viscosity(self, p, T):
        return pp.ad.Scalar(1.)

    def thermal_conductivity(self, p, T):
        return pp.ad.Scalar(1.)
