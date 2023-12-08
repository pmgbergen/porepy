"""This module contains the (private) base class for all phases.

The abstract phase class contains relevant fractional variables, as well as abstract
methods for thermodynamic properties used in the unified formulation.

"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Generator, Literal, overload

import numpy as np

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from .component import Component
from .composite_utils import AdProperty, safe_sum

__all__ = ["PhaseProperties", "AbstractEoS", "Phase"]


@dataclass(frozen=True)
class PhaseProperties:
    """Basic data class for general phase properties, relevant for this framework.

    Use this dataclass to extend the list of relevant phase properties for a specific
    equation of state.

    """

    rho: NumericType
    """Molar density ``[mol / m^3]``."""

    rho_mass: NumericType
    """Mass density ``[kg / m^3]``."""

    v: NumericType
    """Molar volume ``[m^3 / mol]``."""

    h_ideal: NumericType
    """Specific ideal enthalpy ``[J / mol / K]``, which is a sum of ideal enthalpies of
    components weighed with their fraction. """

    h_dep: NumericType
    """Specific departure enthalpy ``[J / mol / K]``."""

    h: NumericType
    """Specific enthalpy ``[J / mol / K]``, a sum of :attr:`h_ideal` and :attr:`h_dep`.
    """

    phis: list[NumericType]
    """Fugacity coefficients per component, ordered as compositional fractions."""

    kappa: NumericType
    """Thermal conductivity ``[W / m / K]``."""

    mu: NumericType
    """Dynamic molar viscosity ``[mol / m / s]``."""


class AbstractEoS(abc.ABC):
    """Abstract class representing an equation of state.

    Child classes have to implement the method :meth:`compute` which must return
    relevant :class:`PhaseProperties` using a specific EoS.

    The purpose of this class is the abstraction of the property computations, as well
    as providing the necessary information about the supercritical state and the
    extended state in the unified setting.

    Parameters:
        gaslike: A bool indicating if the EoS is represents a gas-like state.

            Since in general there can only be one gas-phase, this flag must be
            provided.
        *args: In case of inheritance.
        **kwargs: In case of inheritance.

    """

    def __init__(self, gaslike: bool, *args, **kwargs) -> None:
        super().__init__()

        self._components: list[Component] = list()
        """Private container for components with species data. See :meth:`components`.
        """

        self.is_supercritical: np.ndarray = np.array([], dtype=bool)
        """A boolean array flagging if the mixture became super-critical.

        In vectorized computations, the results are stored component-wise.

        Important:
            It is unclear, what the meaning of super-critical phases is using this EoS
            and values in this phase region should be used with suspicion.

        """

        self.is_extended: np.ndarray = np.array([], dtype=bool)
        """A boolean array flagging if an extended state is computed in the unified
        setting.

        In vectorized computations, the results are stored component-wise.

        Important:
            Extended states are un-physical in general.

        """

        self.gaslike: bool = bool(gaslike)
        """Flag passed at instantiation indicating if gas state or not."""

    @property
    def components(self) -> list[Component]:
        """A list of (compatible) components, which hold relevant chemical data.

        A setter is provided, which concrete EoS can overwrite to perform one-time
        computations, if any.

        Important:
            This attribute is set by a the setter of :meth:`Phase.components`,
            and should not be meddled with.

            The order in this list is crucial for computations involving fractions.

            Order of passed fractions must coincide with the order of components
            passed here.

            **This is a design choice**.
            Alternatively, component-related parameters and
            functions can be passed during instantiation, which would render the
            signature of the constructor quite hideous.

        Parameters:
            components: A list of component for the EoS containing species data.

        """
        return self._components

    @components.setter
    def components(self, components: list[Component]) -> None:
        # deep copy
        self._components = [c for c in components]

    @abc.abstractmethod
    def compute(
        self,
        p: NumericType,
        T: NumericType,
        X: list[NumericType],
        **kwargs,
    ) -> PhaseProperties:
        """Abstract method for computing all thermodynamic properties based on the
        passed, thermodynamic state.

        Warning:
            ``p``, ``T``, ``*X`` have a union type, meaning the results will be of
            the same. When mixing numpy arrays, porepy's Ad arrays and numbers,
            the user must make sure there will be no compatibility issues.

            This method is not supposed to be used with AD Operator instances.

        Important:
            This method must update :attr:`is_supercritical` and :attr:`is_extended`.

        Parameters:
            p: Pressure
            T: Temperature
            X: ``len=num_components``

                (Normalized) Fractions to be used in the computation.
            **kwargs: Any options necessary for specific computations can be passed as
                keyword arguments.

        Returns:
            A dataclass containing the basic phase properties. The basic properties
            include those, which are required for the reminder of the framework to
            function as intended.

        """
        raise NotImplementedError("Call to abstract method.")

    def get_h_ideal(
        self, p: NumericType, T: NumericType, X: list[NumericType]
    ) -> NumericType:
        """
        Parameters:
            p: Pressure.
            T: Temperature.
            X: ``len=num_components``

                (Normalized) Fraction per component to be used in the computation,
                ordered as in :attr:`components`.

        Returns:
            The ideal part of the enthalpy, which is a sum of ideal component enthalpies
            weighed with their fraction.

        """
        return safe_sum([x * comp.h_ideal(p, T) for x, comp in zip(X, self.components)])

    def get_rho_mass(self, rho_mol: NumericType, X: list[NumericType]) -> NumericType:
        """
        Parameters:
            rho_mol: Molar density resulting from :meth:`compute`.
            X: ``len=num_components``

                (Normalized) Fraction per component to be used in the computation,
                ordered as in :attr:`components`.

        Returns:
            The mass density, which is the molar density multiplied with the sum of
            fractions weighed with component molar masses.

        """
        return rho_mol * safe_sum(
            [x * comp.molar_mass for x, comp in zip(X, self.components)]
        )


class Phase:
    """Base class for phases in a multiphase multicomponent mixture.

    The term 'phase' includes both, states of matter and general fields.
    A phase is identified by the (time-dependent) region/volume it occupies (saturation)
    and the (molar) fraction of mass belonging to this phase.

    Phases have physical properties, dependent on pressure, temperature and component
    fractions.
    Modelled components can be accessed by iterating over a phase.

    Thermodynamic phase properties relevant for flow and flash problems are stored as
    AD-compatible objects, whose values can be written and accessed directly.
    To compute the values based on the passed equation of state, use
    :meth:`compute_properties`.
    These properties are:

    - :attr:`density`
    - :attr:`enthalpy`
    - :attr:`viscosity`
    - :attr:`conductivity`
    - :attr:`fugacities_of`

    The variables representing the molar fraction and saturation are assigned by
    a mixture instance, and are only available when the phase was put into context
    through this way.

    Note:
        Dependent on whether this phase is assigned as the reference phase or not,
        the operator representing the fraction or saturation might either be a genuine
        variable (:class:`~porepy.numerics.ad.operators.MixedDimensionalVariable`)
        or a dependent :class:`~porepy.numerics.ad.operators.Operator`,
        where the fraction and saturation were eliminated by unity respectively.

    Also, variables and operators representing phase compositions (per component) and
    normalized fractions respectively, are also assigned by a mixture.

    Note:
        All compositional fractions in :attr:`fraction_of` are genuine variables in the
        flash (:class:`~porepy.numerics.ad.operators.MixedDimensionalVariable`).

        All normalized fractions in :attr:`normalized_fractions_of` are dependent
        :class:`~porepy.numerics.ad.operators.Operator` -instances,
        created by normalization of fractions in :attr:`fraction_of`.

    Important:
        The mixture class is the only class supposed to use the setter of
        :meth:`components`.

    Parameters:
        eos: An equation of state providing computations for properties.
        name: Given name for this phase. Used as an unique identifier and for naming
            various variables.

    """

    def __init__(
        self,
        eos: AbstractEoS,
        name: str,
    ) -> None:
        self._name: str = name
        """Name given to the phase at instantiation."""

        self._components: list[Component] = []
        """Private list of present components (see :meth:`components`)."""

        ### PUBLIC

        self.eos: AbstractEoS = eos
        """The equation of state passed at instantiation."""

        self.density: AdProperty = AdProperty(f"phase-density-{self.name}")
        """Molar density of this phase.

        - Math. Dimension:        scalar
        - Phys. Dimension:        [mol / m^3]


        This is an AD-compatible representation, whose value is computed using
        :meth:`compute_properties`.

        """

        self.volume: AdProperty = AdProperty(f"phase-volume-{self.name}")
        """Molar volume of this phase.

        - Math. Dimension:        scalar
        - Phys. Dimension:        [m^3 / mol]


        This is an AD-compatible representation, whose value is computed using
        :meth:`compute_properties`.

        """

        self.enthalpy: AdProperty = AdProperty(f"phase-enthalpy-{self.name}")
        """Specific molar enthalpy of this phase.

        - Math. Dimension:        scalar
        - Phys.Dimension:         [J / mol / K]

        This is an AD-compatible representation, whose value is computed using
        :meth:`compute_properties`.

        """

        self.viscosity: AdProperty = AdProperty(f"phase-viscosity-{self.name}")
        """Dynamic molar viscosity of this phase.

        - Math. Dimension:        scalar
        - Phys. Dimension:        [mol / m / s]

        This is an AD-compatible representation, whose value is computed using
        :meth:`compute_properties`.

        """

        self.conductivity: AdProperty = AdProperty(f"phase-conductivity-{self.name}")
        """Thermal conductivity of this phase.

        - Math. Dimension:    2nd-order tensor
        - Phys. Dimension:    [W / m / K]

        This is an AD-compatible representation, whose value is computed using
        :meth:`compute_properties`.

        """

        self.fugacity_of: dict[Component, AdProperty] = dict()
        """A map of fugacity coefficients for each present component (key).

        - Math. Dimension:    scalar
        - Phys. Dimension:    [Pa]

        The properties are set, once the components are set for this phase using
        :meth:`components`.

        This is an AD-compatible representation, whose values are computed using
        :meth:`compute_properties`.

        """

        self.fraction: pp.ad.Operator
        """Molar phase fraction, a primary variable on the whole domain.

        Indicates how many of the total moles belong to this phase per cell.

        - Math. Dimension:        scalar
        - Phys. Dimension:        [-] fractional

        This attribute is assigned by a mixture instance, when this phase is added.

        If the phase is assigned as the reference phase, this is a dependent operator.
        Otherwise it is a variable.

        """

        self.saturation: pp.ad.Operator
        """Saturation (volumetric fraction), a secondary variable on the whole domain.

        Indicates how much of the (local) volume is occupied by this phase per cell.

        - Math. Dimension:        scalar
        - Phys. Dimension:        [-] fractional

        This attribute is assigned by a mixture instance, when this phase is added.

        If the phase is assigned as the reference phase, this is a dependent operator.
        Otherwise it is a variable.

        """

        self.fraction_of: dict[Component, pp.ad.MixedDimensionalVariable] = dict()
        """A dictionary containing the composition variable for each component (key).

        The fractions are created by a mixture instance, when this phase is added.

        """

        self.normalized_fraction_of: dict[Component, pp.ad.Operator] = dict()
        """A dictionary containing operators representing normalized fractions per
        component (key) based on variables in :attr:`fraction_of`.

        The operators are created, once the components are set for this phase using
        :meth:`components`.

        """

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
    def type(self) -> int:
        """Integer indicating the physical type of the phase.

        - ``0``: liquid-like
        - ``1``: gas-like
        - ``>1``: open for further development.

        """
        return int(self.eos.gaslike)

    @property
    def num_components(self) -> int:
        """Number of added components."""
        return len(self._components)

    @property
    def components(self) -> list[Component]:
        """A container for all components modelled in this phase.

        The user does not need to set the components directly in a phase.
        This is done during
        :meth:`~porepy.composite.composition.Mixture.initialize`.

        By setting components, variables representing the phase composition
        are created or overwritten with zero.

        Warning:
            Components in a phase should only be set once!

            There is a lose end in the framework, variables once created can not be
            deleted and are for always in the global DOF order.

        Parameters:
            components: List of components to be modelled in this phase.

        """
        # deep copy list
        return [c for c in self._components]

    @components.setter
    def components(self, components: list[Component]) -> None:
        self._components = list()
        # to avoid double setting
        added_components = list()
        for comp in components:
            # check if already added
            if comp.name in added_components:
                continue

            added_components.append(comp.name)
            self._components.append(comp)

            # create fugacity operatpr
            fug_i = AdProperty(f"fugacity-of-{comp.name}-in-{self.name}")
            self.fugacity_of.update({comp: fug_i})

        self.eos.components = self.components

    @overload
    def compute_properties(
        self,
        p: NumericType,
        T: NumericType,
        X: list[NumericType],
        store: Literal[True] = True,
        **kwargs,
    ) -> None:
        # Typing overload for default return value: None, properties are stored
        ...

    @overload
    def compute_properties(
        self,
        p: NumericType,
        T: NumericType,
        X: list[NumericType],
        store: Literal[False] = False,
        **kwargs,
    ) -> PhaseProperties:
        # Typing overload for default return value: Properties are returned
        ...

    def compute_properties(
        self,
        p: NumericType,
        T: NumericType,
        X: list[NumericType],
        store: bool = True,
        **kwargs,
    ) -> None | PhaseProperties:
        """Abstract method for computing thermodynamic properties of this phase.

        This is a wrapper for :meth:`AbstractEoS.compute`, where the results are stored
        in this class' thermodynamic properties.

        Parameters:
            p: Pressure.
            T: Temperature.
            X: ``len=num_components``

                (Normalized) Component fractions in this phase.
            store: ``default=True``

                A flag to store the computed values in the respective attributes
                representing thermodynamic properties.
            **kwargs: Keyword arguments for :meth:`AbstractEoS.compute`.

        Returns:
            If ``store=False``, returns the computed properties.

        """
        props = self.eos.compute(p, T, X, **kwargs)

        if store:
            self.density.value = props.rho
            self.volume.value = props.v
            self.viscosity.value = props.mu
            self.conductivity.value = props.kappa
            self.enthalpy.value = props.h

            for i, comp in enumerate(self):
                self.fugacity_of[comp].value = props.phis[i]
        else:
            return props
