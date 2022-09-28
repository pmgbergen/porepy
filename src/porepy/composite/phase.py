""" Contains the abstract base class for all phases."""
from __future__ import annotations

import abc
from typing import Dict, Generator, List, Optional, Union

import porepy as pp

from .component import VarLike

__all__ = ["Phase"]


class Phase(abc.ABC):
    """Abstract base class for phases in a multiphase multicomponent mixture.

    The term 'phase' includes both, states of matter and general fields.
    A phase is identified by the (time-dependent) region/volume it occupies and a
    respective velocity field (or flux) in that region.

    Similar to :class:`~porepy.composite.Component`, if a :class:`~porepy.ad.ADSystemManager`
    is passed at instantiation, the AD framework is is used to represent the fractional
    variables and the phase class becomes a singleton with respect to the mixed-dimensional
    domain contained in the AD system.
    Unique instantiation over a given domain is assured by using the given as an unique
    identifier.
    Ambiguities and uniqueness must be assured due to central storage of the fractional values
    in the grid data dictionaries.

    If the AD system is not passed, this class can be used in standalone mode.
    Respective fractional variables are not existent in this case and the
    :class:`~porepy.composite.Composition` takes over the computation and storage of values.

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

    def __new__(
        cls, name: str, ad_system: Optional[pp.ad.ADSystemManager] = None
    ) -> Phase:
        # check for AD singletons per grid
        if ad_system:
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
        if ad_system:
            Phase.__ad_singletons[mdg].update({name: new_instance})

        return new_instance

    def __init__(
        self, name: str, ad_system: Optional[pp.ad.ADSystemManager] = None
    ) -> None:
        # skipping re-instantiation if class if __new__ returned the previous reference
        if Phase.__singleton_accessed:
            Phase.__singleton_accessed = False
            return

        super().__init__()

        ### PUBLIC

        self.ad_system: Optional[pp.ad.ADSystemManager] = ad_system
        """The AD system optionally passed at instantiation."""

        #### PRIVATE

        self._name = name
        self._present_components: List[pp.composite.Component] = list()

        # Instantiate saturation and molar phase fraction (secondary variables)
        self._s: Optional[pp.ad.MergedVariable] = None
        self._fraction: Optional[pp.ad.MergedVariable] = None
        if ad_system:
            self._s = ad_system.create_variable(self.saturation_var_name, False)
            self._fraction = ad_system.create_variable(
                self.fraction_var_name, False
            )
        # contains fractional values per present component name (key)
        self._composition: Dict[str, VarLike] = dict()

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
    def saturation_var_name(self) -> str:
        """Name for the saturation variable, given by the general symbol and :meth:`name`."""
        return f"s_{self.name}"

    @property
    def fraction_var_name(self) -> str:
        """Name for the molar fraction variable, given by the general symbol and :meth:`name`."""
        return f"y_{self.name}"

    @property
    def saturation(self) -> Optional[pp.ad.MergedVariable]:
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
    def fraction(self) -> Optional[pp.ad.MergedVariable]:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        Returns:
            molar phase fraction, a secondary variable on the whole domain.
            Indicates how many of the total moles belong to this phase (per cell).
            It is supposed to represent the value at thermodynamic equilibrium.

        """
        return self._fraction

    def fraction_of_component(self, component: pp.composite.Component) -> VarLike:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        Notes:
            Currently there is no checking if the component was added to the phase.
            If it was not added, zero is simply returned.

        Parameters:
            component: a component present in this phase

        Returns:
            molar fraction of a component in this phase,
            a secondary variable on the whole domain (cell-wise).
            Indicates how many of the moles in this phase belong to the component.
            It is supposed to represent the value at thermodynamic equilibrium.
            Returns always zero if a component is not modelled (added) to this phase.

        Raises:
            RuntimeError: if the AD framework is used and the component was instantiated using
                a different AD system than the one used for this composition.

        """
        if component.ad_system != self.ad_system:
            raise RuntimeError(
                f"Component '{component.name}' instantiated with a different AD system."
            )
        return self._composition.get(self.fraction_of_component_var_name(component), 0.0)
    
    def fraction_of_component_var_name(self, component: pp.composite.Component) -> str:
        """
        Parameters:
            component: component for which the respective name is requested.

        Returns:
            name of the respective variable, given by the general symbol, the
            component name and the phase name.

        """
        return f"x_{self.name}_{component.name}"

    def add_component(
        self,
        component: Union[List[pp.composite.Component], pp.composite.Component],
    ) -> None:
        """Adds components which are expected by the modeler in this phase.

        If a component was already added, nothing happens.

        If the AD framework is used (AD System is not None), creates the fractional variable
        for the component in this phase.
        If not, instantiates the respective fractional as zero (float).

        Parameters: a component, or list of components, which are expected in this phase.

        Raises:
            RuntimeError: if the AD framework is used and the component was instantiated using
                a different AD system than the one used for the phase.

        """

        if isinstance(component, pp.composite.Component):
            component = [component]

        present_components = [ps.name for ps in self._present_components]

        for comp in component:
            # sanity check when using the AD framework
            if self.ad_system:
                if self.ad_system != comp.ad_system:
                    raise RuntimeError(
                        f"Component '{comp.name}' instantiated with a different AD system."
                    )
            # skip already present components:
            if comp.name in present_components:
                continue
            # create the name for the variable 'component fraction in this phase'
            fraction_name = self.fraction_of_component_var_name(comp)
            # create the fraction of the component in this phase
            comp_fraction: VarLike
            if self.ad_system:
                comp_fraction = self.ad_system.create_variable(fraction_name, False)
            else:
                comp_fraction = 0.0
            # store reference to present substance
            self._present_components.append(comp)
            # store the compositional variable
            self._composition.update({fraction_name: comp_fraction})

    def mass_density(self, p: VarLike, T: VarLike) -> VarLike:
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

    # ------------------------------------------------------------------------------
    ### Abstract, phase-related physical properties
    # ------------------------------------------------------------------------------

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
