"""This module contains the (private) base class for all phases.

The abstract phase class contains relevant fractional variables, as well as abstract
methods for thermodynamic properties used in the unified formulation.

"""
from __future__ import annotations

import abc
from typing import Generator

import numpy as np

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from ._core import COMPOSITIONAL_VARIABLE_SYMBOLS
from .component import Component
from .composite_utils import CompositionalSingleton, safe_sum


class Phase(abc.ABC, metaclass=CompositionalSingleton):
    """Abstract base class for phases in a multiphase multicomponent mixture.

    The term 'phase' includes both, states of matter and general fields.
    A phase is identified by the (time-dependent) region/volume it occupies (saturation)
    and the (molar) fraction of mass belonging to this phase.

    Phases have physical properties,
    dependent on the thermodynamic state and the composition.
    The composition variables (molar fractions of present components)
    can be accessed by internal reference (see overload of ``__iter__``).

    Important:
        Due to the user being able to access component fractions in this phase by
        reference, the signature of all thermodynamic properties contains **optional**
        arguments ``*X`` representing the phase composition.

        I.e., when implementing custom Phases using a specific EoS, **always** include
        computations for the case when specific fractions are passed as ``*X``.

    The Phase is a Singleton per AD system,
    using the **given** name as a unique identifier.
    A Phase class with name ``X`` can only be present once in a system.
    Ambiguities must be avoided due to central storage of the fractional values in the
    grid data dictionaries.

    Note:
        The variables representing saturation and molar fraction of a phase are created
        and their value is set to zero.

    Parameters:
        ad_system: AD system in which this phase is present cell-wise in each subdomain.
        name: ``default=''``

            Given name for this phase. Used as an unique identifier for singletons.

    """

    def __init__(self, ad_system: pp.ad.EquationSystem, name: str = "") -> None:
        super().__init__()

        ### PUBLIC

        self.ad_system: pp.ad.EquationSystem = ad_system
        """The AD system passed at instantiation."""

        #### PRIVATE
        self._name = name
        """Name given to the phase at instantiation."""

        self._components: list[Component] = []

        self._composition: dict[Component, pp.ad.MixedDimensionalVariable] = dict()
        """A dictionary containing the composition variable for each
        added/modelled component in this phase."""

        self._s: pp.ad.MixedDimensionalVariable = ad_system.create_variables(
            f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_saturation']}_{self.name}",
            subdomains=ad_system.mdg.subdomains(),
        )
        """Saturation Variable in AD form."""

        self._fraction: pp.ad.MixedDimensionalVariable = ad_system.create_variables(
            f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_fraction']}_{self.name}",
            subdomains=ad_system.mdg.subdomains(),
        )
        """Molar fraction variable in AD form."""

        # Set values for fractions of phase
        nc = ad_system.mdg.num_subdomain_cells()
        ad_system.set_variable_values(
            np.zeros(nc),
            variables=[self.saturation.name],
            to_state=True,
            to_iterate=True,
        )
        ad_system.set_variable_values(
            np.zeros(nc),
            variables=[self.fraction.name],
            to_state=True,
            to_iterate=True,
        )

    def __iter__(self) -> Generator[Component, None, None]:
        """Generator over components present in this phase.

        Notes:
            The order from this iterator is used for choosing e.g. the values in a
            list of 'numpy.array' when setting initial values.
            Use the order returned here every time you deal with
            component-related values for components in this phase.

        Yields:
            Components present in this phase.

        """
        for component in self._components:
            yield component

    @property
    def name(self) -> str:
        """Name of this phase given at instantiation."""
        return self._name

    @property
    def num_components(self) -> int:
        """Number of added components."""
        return len(self._composition)

    @property
    def saturation(self) -> pp.ad.MixedDimensionalVariable:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        The name of this variable is composed of the general symbol and the name
        assigned to this phase at instantiation
        (see :data:`~porepy.composite._composite_utils.VARIABLE_SYMBOLS`).

        Returns:
            Saturation (volumetric fraction), a secondary variable on the whole domain.
            Indicates how much of the (local) volume is occupied by this phase per cell.

        """
        return self._s

    @property
    def fraction(self) -> pp.ad.MixedDimensionalVariable:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        The name of this variable is composed of the general symbol and the name
        assigned to this phase at instantiation
        (see :data:`~porepy.composite._composite_utils.VARIABLE_SYMBOLS`).

        Returns:
            Molar phase fraction, a primary variable on the whole domain.
            Indicates how many of the total moles belong to this phase per cell.

        """
        return self._fraction

    def fraction_of_component(
        self, component: Component
    ) -> pp.ad.MixedDimensionalVariable | pp.ad.Scalar:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        If a phase is present (phase fraction is strictly positive),
        the component fraction (this one) has physical meaning.

        If a phase vanishes (phase fraction is zero),
        the extended fractions represent non-physical values at equilibrium.
        The extended phase composition does not fulfill unity if a phase is not present.

        In the case of a vanished phase, the regular phase composition is obtained by
        re-normalizing the extended phase composition,
        see :meth:`normalized_fraction_of_component`.

        Parameters:
            component: A component present in this phase.

        Returns:
            Extended molar fraction of a component in this phase,
            a primary variable on the whole domain (cell-wise).
            Indicates how many of the moles in this phase belong to the component.

            Returns always zero if a component is not modelled (added) to this phase.

        """
        if component in self._composition:
            return self._composition[component]
        else:
            return pp.ad.Scalar(0.0)

    def normalized_fraction_of_component(
        self, component: Component
    ) -> pp.ad.Operator | pp.ad.Scalar:
        """Performs a normalization of the component fraction by dividing it through
        the sum of the phase composition.

        If a phase is present (phase fraction is strictly positive),
        the normalized component fraction coincides with the component fraction due to
        the sum of the phase composition fulfilling unity.

        If a phase vanishes (phase fraction is zero), the normalized fraction has
        no physical meaning but fulfils unity, contrary to the (extended) fraction.

        Parameters:
            component: A component present in this phase.

        Returns:
            Normalized molar fraction of a component in this phase in AD operator form.

            Returns always zero (wrapped in AD) if a component is not modelled
            (added) to this phase.

        """
        if component in self._composition:
            # normalization by division through fraction sum
            norm_frac = self.fraction_of_component(component) / safe_sum(
                [self.fraction_of_component(comp) for comp in self]
            )
            norm_frac.set_name(
                f"{self.fraction_of_component(component).name}_normalized"
            )
            return norm_frac
        else:
            return pp.ad.Scalar(0.0)

    @property
    def components(self) -> list[Component]:
        """A container for all components modelled in this phase.

        The user does not need to set the components directly in a phase.
        This is done during
        :meth:`~porepy.composite.composition.Mixture.initialize`.

        By setting components, variables representing the phase composition
        are created or overwritten with zero.

        Warning:
            There is a lose end in the framework, variables once created can not be
            deleted and are for always in the global DOF order.

            Abstain from overwriting the components in a phase once set.

        Parameters:
            components: List of components to be modelled in this phase.

        """
        # deep copy list
        return [c for c in self]

    @components.setter
    def components(self, components: list[Component]) -> None:
        self._composition = dict()
        self._components = list()
        # to avoid double setting
        added_components = list()
        for comp in components:
            # sanity check when using the AD framework
            if self.ad_system != comp.ad_system:
                raise ValueError(
                    f"Component '{comp.name}' instantiated with a different AD system."
                )

            # check if already added
            if comp.name in added_components:
                continue
            else:
                added_components.append(comp.name)

            # create compositional variables for the component in this phase
            fname = f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_composition']}"
            fname += f"_{comp.name}_{self.name}"
            comp_fraction = self.ad_system.create_variables(
                fname,
                subdomains=self.ad_system.mdg.subdomains(),
            )

            # set fractional values to zero
            nc = self.ad_system.mdg.num_subdomain_cells()
            self.ad_system.set_variable_values(
                np.zeros(nc),
                variables=[comp_fraction.name],
                to_state=True,
                to_iterate=True,
            )

            # store the compositional variable and component
            self._composition.update({comp: comp_fraction})
            self._components.append(comp)

    ### Physical properties ------------------------------------------------------------

    def mass_density(
        self, p: NumericType, T: NumericType, *X: tuple[NumericType]
    ) -> NumericType | pp.ad.Operator:
        """Uses the mass of this phase in combination with the molar masses and
        fractions of present components, to compute the mass density of the phase.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kg / REV]

        Note:
            A call to :meth:`density` is performed using the input.
            If ``*X`` is not provided, the normalized phase compositions are evaluated
            and passed as *X to :meth:`density`.

        Parameters:
            p: Pressure.
            T: Temperature.
            *X: (Normalized) Component fractions in this phase.

        Returns:
            The mass density of this phase in AD compatible form.

        """
        # add the mass-weighted fraction for each present substance.
        # if no components are present, the weight is zero!
        weight = 0.0
        if X:
            assert len(X) == len(
                self._components
            ), f"Mismatch fractions: Need {len(self._components)} got {len(X)}"
            for i, component in enumerate(self._components):
                weight += component.molar_mass() * X[i]
            return weight * self.density(p, T, *X)
        else:
            X_ = [
                self.fraction_of_component(comp).evaluate(self.ad_system)
                for comp in self
            ]
            normalization = safe_sum(X_)
            X_ = tuple([x / normalization for x in X_])
            for component in self._composition:
                weight += component.molar_mass() * X_[component]
            return weight * self.density(p, T, *X_)

    @abc.abstractmethod
    def density(
        self, p: NumericType, T: NumericType, *X: tuple[NumericType]
    ) -> NumericType:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [mol / REV]

        Parameters:
            p: Pressure.
            T: Temperature.
            *X: (Normalized) Component fractions in this phase.

        Returns:
            The molar density of this phase in AD compatible form.

        """
        pass

    @abc.abstractmethod
    def specific_enthalpy(
        self, p: NumericType, T: NumericType, *X: tuple[NumericType]
    ) -> NumericType:
        """
        | Math. Dimension:        scalar
        | Phys.Dimension:         [kJ / mol / K]

        Parameters:
            p: Pressure.
            T: Temperature.
            *X: (Normalized) Component fractions in this phase.

        Returns:
            The specific molar enthalpy of this phase in AD compatible form.

        """
        pass

    @abc.abstractmethod
    def dynamic_viscosity(
        self, p: NumericType, T: NumericType, *X: tuple[NumericType]
    ) -> NumericType:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [mol / m / s]

        Parameters:
            p: Pressure.
            T: Temperature.
            *X: (Normalized) Component fractions in this phase.

        Returns:
            The dynamic viscosity of this phase in AD compatible form.

        """
        pass

    @abc.abstractmethod
    def thermal_conductivity(
        self, p: NumericType, T: NumericType, *X: tuple[NumericType]
    ) -> NumericType:
        """
        | Math. Dimension:    2nd-order tensor
        | Phys. Dimension:    [W / m / K]

        Parameters:
            p: Pressure.
            T: Temperature.
            *X: (Normalized) Component fractions in this phase.

        Returns:
            The thermal conductivity of this phase in AD compatible form.

        """
        pass

    @abc.abstractmethod
    def fugacity_of(
        self,
        component: Component,
        p: NumericType,
        T: NumericType,
        *X: tuple[NumericType],
    ) -> NumericType:
        """
        | Math. Dimension:    scalar
        | Phys. Dimension:    [Pa]

        Parameters:
            component: A component present in this mixture.
            p: Pressure.
            T: Temperature.
            *X: (Normalized) Component fractions in this phase.

        Returns:
            The fugacity of ``component`` in this phase in AD compatible form.

        """
        pass
