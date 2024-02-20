"""This module contains the (abstract) base class for elements of a mixture:

- Components
- Compounds (Pure component altered by presence of solutes)
- Phase (Physical phase like liquid or gas)
- Mixture

Components are models for phase-changing chemical species inside a mixture.
They are either pure components (with relevant fractional variables) or
compounds, where other, present species act as parameters.

The hierarchy is as follows:

1. :class:`Component`:
   A phase-changing representation of a species involving some physical constants
   Additionally, this class represents a variable quantity in the equilibrium
   problem. It can appear in multiple phases and has a fraction of the overall mass
   associated with it.
2. :class:`Compound`:
   Additionally to being a variable quantity, this class has other species
   with related solute fractions functioning as parameters for thermodynamic
   properties. The solute fractions are **not** variables of the equilibrium problem.
   They might nevertheless be transportable by f.e. a coupled flow problem.
3. :class:`Phase`:
   An object representing a physical phase like gas-phase or a liquid-phase.
   A phase can contain multiple phase-changing components and the modeller must set
   those explicitly (see :attr:`Phase.components`).
   Components in a phase are characterized by their fraction of mass
   (:attr:`Phase.fraction_of`), relative to the
   fraction of mass in a phase (:attr:`Phase.fraction`).
   Components have additionally an fugacity coefficient in a phase
   (:attr:`Phase.fugacity_of`)
   In the unified setting, all phases contain every modelled component.

   The phase has also physical properties (like density and enthalpy) which come into
   play when formulating more complex equilibrium models coupled with flow & transport.
4. :class:`Mixture`:
    A basic representation of a mixture which is a collection of anticipated phases and
    present components.

    It puts phases and components into context and assignes fractional unknowns.
    The mixture expects the overall fraction of a component (:attr:`Component.fraction`)
    to never be zero. Otherwise a smaller mixture can be modelled.

    Serves as a managing instance and provides functionalities to formulate the flash
    equations using PorePy's AD framework.

Note:
    Phases are meant to be based on an Equation of State.
    A basic interface for such an equation of state is defined by :class:`AbstractEoS`.

"""
from __future__ import annotations

import abc
from dataclasses import asdict
from typing import Any, Callable, Generator, Literal, Sequence, overload

import numpy as np

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from ._core import R_IDEAL
from .chem_species import ChemicalSpecies, FluidSpecies
from .composite_utils import DomainProperty, safe_sum
from .states import PhaseState

__all__ = [
    "Component",
    "Compound",
    "AbstractEoS",
    "Phase",
    "Mixture",
]


class Component(abc.ABC, FluidSpecies):
    """Abstract base class for components modelled inside a mixture.

    Components are chemical species inside a mixture, which possibly go through phase
    transitions and appear in multiple :class:`Phase`.
    A component is identified by the (time-dependent) fraction of total mass belonging
    to the component. The fraction is assigned by an mixture instance, once the it has
    been added to a mixture.

    Note:
        Rather than instantiating a component directly, it is easier to use the
        class factory based on loaded species data (see :meth:`from_species`).

    Parameters:
        **kwargs: See parent (data-) class and its attributes.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            **kwargs
            # **{
            # k: v for k, v in kwargs.items()
            # if k in FluidSpeciesData.__match_args__
            # }  # Python 3.10
        )

        # creating the overall molar fraction variable
        self.fraction: Callable[[Sequence[pp.Grid]], pp.ad.Operator]
        """Overall fraction, or feed fraction, for this component.

        It indicates how many of the total moles belong to this component (cell-wise).

        - Math. Dimension:        scalar
        - Phys. Dimension:        [%] fractional

        Note:
            Feed fractions are constant in non-reactive mixtures, since there the number
            of moles of a species is assumed to be constant.

            Nevertheless, they are instantiated as variables for coupling with flow
            and transport, and for future development.

        This callable is assigned by a mixture mixin.

        If the component is assigned as the reference component and the reference
        component is eliminated, this is a dependent operator.
        Otherwise it is a variable.

        If there is only one component in the mixture, this is a scalar 1.

        """

    @classmethod
    def from_species(cls, species: FluidSpecies) -> Component:
        """An instance factory creating an instance of this class based on a load
        fluid species represented by respective data class.

        Parameters:
            species: Chemical species with loaded data.

        Returns:
            A genuine mixture component.

        """
        return cls(**asdict(species))

    @abc.abstractmethod
    def h_ideal(self, p: Any, T: Any) -> Any:
        """Abstract method for implementing the component-specific ideal part of the
        specific molar enthalpy.

        This function depends on experimental data and heuristic laws.

        - Math. Dimension:        scalar
        - Phys. Dimension:        [J / mol]

        Parameters:
            p: The pressure of the mixture.
            T: The temperature of the mixture.

        Returns:
            Ideal specific enthalpy for given pressure and temperature.

        """
        pass

    def u_ideal(self, p: Any, T: Any) -> Any:
        """
        - Math. Dimension:        scalar
        - Phys. Dimension:        [J / mol]

        Parameters:
            p: The pressure of the mixture.
            T: The temperature of the mixture.

        Returns:
            Ideal specific internal energy based on the relation

            :math:`u_{id}(T) = h_{id}(T) - TR`.

        """
        return self.h_ideal(p, T) - T * R_IDEAL


class Compound(Component):
    """Abstract base class for compounds in a mixture.

    A compound is a simplified, but meaningfully generalized set of components inside a
    mixture, for which it makes sense to treat it as a single component.

    It is represents one species, the solvent, and contains arbitrary many solutes.

    A compound can appear in multiple phases and its thermodynamic properties are
    influenced by the presence of solutes.

    Solutes are transportable and are represented by a molar fraction relative to
    the :attr:`~Component.fraction` of the compound, i.e. the moles of a solute are
    given by a product of mixture density, compound fraction and solute fraction.

    The solvent fraction is eliminated by unity.

    Note:
        Due to the generalization, the solvent and solutes alone are not considered as
        genuine components which can transition into various phases,
        but rather as parameters in the flash problem.
        Only the compound as a whole splits into various phases. Fractions in phases
        are associated with the compound.
        Solvent and solute fractions are not variables in the flash problem.

    Example:
        Brines with species salt and water as solute and solvent, where it is
        sufficient to calculate how much brine is in vapor or liquid form,
        and the information about how the salt distributes across phases is irrelevant.
        The salt in this case is a **transportable** quantity,
        whose concentration acts as a parameter in the flash.

        Another example would be the black-oil model, where black-oil is treated as a
        compound with various hydrocarbons as pseudo-components.

    """

    def __init__(self, **kwargs) -> None:
        Warning("Not up-to-date with remaining framework.")
        super().__init__(**kwargs)

        self._solutes: list[ChemicalSpecies] = list()
        """A list containing present solutes."""

        self.molalities: list[NumericType] = list()
        """A list containing the molality for the solvent, followed by molalities for
        present solutes.

        Important:
            Molalities must be computed and stored using (relative) fractions per
            solute (see :meth:`compute_molalities`).

        """

        self.solute_fraction_of: dict[
            ChemicalSpecies, Callable[[Sequence[pp.Grid]], pp.ad.Operator]
        ] = dict()
        """A dictionary containing per solute (key) the relative fraction of it
        in this compound.

        Solute fractions indicate how many of the moles in the compound belong to the
        solute.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        Important:
            1. Solute fractions are transportable quantities!
            2. The Solvent fraction is not included. It can be obtained by unity of
               fractions.

        This callable is assigned by a mixture mixin.

        """

    def __iter__(self) -> Generator[ChemicalSpecies, None, None]:
        """Iterator overload to iterate over present solutes"""
        for solute in self._solutes:
            yield solute

    @property
    def solutes(self) -> list[ChemicalSpecies]:
        """Solutes present in this compound.

        Important:
            Solutes must be set before the compound is added to a mixture.

        Parameters:
            solutes: A list of solutes to be added to the compound.

        """
        return [s for s in self._solutes]

    @solutes.setter
    def solutes(self, solutes: list[ChemicalSpecies]) -> None:
        # avoid double species
        double = []
        self._solutes = []
        for s in solutes:
            if s.CASr_number not in double:
                self._solutes.append(s)
                double.append(s.CASr_number)

    @pp.ad.admethod
    def compound_molar_mass(self, *X: tuple[NumericType]) -> NumericType:
        """The molar mass of a compound depends on how much of the solutes is available.

        It is a sum of the molar masses of present species, weighed with their
        respective fraction, including the solvent.

        Parameters:
            *X: (Relative) solute fractions.

        Raises:
            AssertionError: If the number of provided values does not match the number
                of present solutes.

        Returns:
            The molar mass of the compound.

        """
        assert len(X) == len(self._solutes), f"Need {len(self._solutes)} values."
        M = self.molar_mass * (1 - safe_sum(X))

        for solute, x in zip(self._solutes, X):
            M += solute.molar_mass * x

        return M

    @overload
    def compute_molalities(
        self, *X: tuple[NumericType], store: Literal[True] = True
    ) -> None:
        # signature overload for default args
        ...

    @overload
    def compute_molalities(
        self, *X: tuple[NumericType], store: Literal[False] = False
    ) -> list[NumericType]:
        # signature overload for ``store==False``
        ...

    def compute_molalities(
        self, *X: tuple[NumericType], store: bool = True
    ) -> list[NumericType] | None:
        """Computes the molalities of present species, including the solvent.

        The first molality value belongs to the solvent, the remainder are ordered
        as given by :meth:`solutes`.

        Note:
            The solvent molality is always the reciprocal of the solvent molar mass.
            Hence, it is always a scalar.

        Parameters:
            *X: Relative solute fractions.
            store: ``default=True``

                If True, the molalities are stored.

        Raises:
            AssertionError: If the number of provided values does not match the number
                of present solutes.

        Returns:
            A list of molality values if ``store=False``.

        """
        assert len(X) == len(self._solutes), f"Need {len(self._solutes)} values."

        molalities = []

        # molality of solvent
        molalities.append(1 / self.molar_mass)

        # solvent fraction
        x_s = 1 - safe_sum(X)
        for x in X:
            m_i = x / (x_s * self.molar_mass)
            molalities.append(m_i)

        if store:
            # for storage, derivatives are removed
            molalities = [
                m.val if isinstance(m, pp.ad.AdArray) else m for m in molalities
            ]
            self.molalities = molalities
        else:
            return molalities

    def fractions_from_molalities(
        self, molalities: list[NumericType]
    ) -> list[np.ndarray]:
        """
        Note:
            Molalities must only be given for solutes, not the solvent.

        Parameters:
            molalities: A list of molalities per present solute.

        Raises:
            AssertionError: If the number of provided values does not match the number
                of present solutes.

        Returns:
            A list of relative solute fractions calculated from molalities,
            where the first fraction always corresponds to the solute fraction.

        """
        # strip off derivatives if Ad
        molalities = [m.val if isinstance(m, pp.ad.AdArray) else m for m in molalities]

        m_sum = safe_sum(molalities)

        X: list[np.ndarray] = []

        for m in molalities:
            # See https://en.wikipedia.org/wiki/Molality
            x_i = self.molar_mass * m / (1 + self.molar_mass * m_sum)
            X.append(x_i)

        # Return including solvent fraction
        return [1 - safe_sum(X)] + X


class AbstractEoS(abc.ABC):
    """Abstract EoS class defining the interface between thermodynamic input
    and resulting structure containing thermodynamic properties of a phase.

    Component properties required for computations can be extracted in the constructor.

    Parameters:
        components: A sequence of components for which the EoS is instantiated.

    """

    def __init__(self, components: Sequence[Component]) -> None:
        self._nc: int = len(components)
        """Number of components passed at instantiation."""

    @abc.abstractmethod
    def compute_phase_state(
        self, phase_type: int, p: np.ndarray, T: np.ndarray, xn: Sequence[np.ndarray]
    ) -> PhaseState:
        """ "Abstract method to compute the properties of a phase based on pressure,
        temperature and a sequence of (normalized) fractions for each component passed
        at instantiation.

        Parameters:
            phase_type: See :attr:`Phase.type`
            p: ``shape=(N,)``

                Pressure values.
            T:``shape=(N,)``

                Temperature values.
            xn: ``shape=(num_comp, N)``

                Normalized fractions per component.

        Returns:
            A datastructure containing all relevant phase properties and their
            derivatives.

        """
        ...


class Phase:
    """Base class for phases in a multiphase multicomponent mixture.

    The term 'phase' as used here refers to physical states of matter.
    A phase is identified by the (time-dependent) region/volume it occupies (saturation)
    and the (molar) fraction of mass belonging to this phase.

    Phases have physical properties, dependent on pressure, temperature and composition.
    Modelled components can be set and accessed using :attr:`components`.

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

    Parameters:
        phase_type: Integer indicating the physical type of the phase.

            - ``0``: liquid-like
            - ``1``: gas-like
            - ``>1``: open for further development.
        eos: An EoS which provides means to compute physical properties of the phase.

            Expected to be already compiled.
        name: Given name for this phase. Used as an unique identifier and for naming
            various variables.

    """

    def __init__(
        self,
        eos: AbstractEoS,
        type: int,
        name: str,
    ) -> None:
        ### PUBLIC

        self.components: Sequence[Component]
        """A sequence of all components modelled in this phase.

        Important:
            In principle, the user has to manually set which components are in which
            phase.

            This influences the operators created during :meth:`Mixture.set_up_ad`.

            Models in the unified setting
            (see :class:`~porepy.composite.equilibrium_models.MixtureUNR`) set all
            components to be present in every phase by default.

        Once set, it should not be modified. Avoid multiple occurences of components.

        """

        self.eos: AbstractEoS = eos
        """The EoS passed at instantiation."""

        self.type: int = int(type)
        """Physical type of this phase. Passed at instantiation."""

        self.name: str = str(name)
        """Name given to the phase at instantiation."""

        self.density: DomainProperty
        """Molar density of this phase.

        - Math. Dimension:        scalar
        - Phys. Dimension:        [mol / m^3]

        This is an AD-compatible representation, whose value is computed using
        :meth:`compute_properties`.

        It is available once :meth:`Mixture.set_up_ad` is performed.

        """

        self.volume: DomainProperty
        """Molar volume of this phase.

        - Math. Dimension:        scalar
        - Phys. Dimension:        [m^3 / mol]

        This is an AD-compatible representation, whose value is computed using
        :meth:`compute_properties`.

        It is available once :meth:`Mixture.set_up_ad` is performed.

        """

        self.enthalpy: DomainProperty
        """Specific molar enthalpy of this phase.

        - Math. Dimension:        scalar
        - Phys.Dimension:         [J / mol / K]

        This is an AD-compatible representation, whose value is computed using
        :meth:`compute_properties`.

        It is available once :meth:`Mixture.set_up_ad` is performed.

        """

        self.viscosity: DomainProperty
        """Dynamic molar viscosity of this phase.

        - Math. Dimension:        scalar
        - Phys. Dimension:        [mol / m / s]

        This is an AD-compatible representation, whose value is computed using
        :meth:`compute_properties`.

        It is available once :meth:`Mixture.set_up_ad` is performed.

        """

        self.conductivity: DomainProperty
        """Thermal conductivity of this phase.

        - Math. Dimension:    2nd-order tensor
        - Phys. Dimension:    [W / m / K]

        This is an AD-compatible representation, whose value is computed using
        :meth:`compute_properties`.

        It is available once :meth:`Mixture.set_up_ad` is performed.

        """

        self.fugacity_of: dict[Component, DomainProperty]
        """Fugacitiy coefficients per component in this phase.

        - Math. Dimension:    scalar
        - Phys. Dimension:    [-]

        The callables are assigned by a mixture mixin.

        Fugacity coefficients depend in general on pressure, temperature and every
        fraction of components in a phase.

        """

        self.fraction: Callable[[Sequence[pp.Grid]], pp.ad.Operator]
        """Molar phase fraction indicates how many of the total moles belong to this
        phase.

        - Math. Dimension:        scalar
        - Phys. Dimension:        [-] fractional

        This callable is assigned by a mixture mixin.

        If the phase is assigned as the reference phase and the reference
        phase is eliminated, this is a dependent operator.
        Otherwise it is a variable.

        If there is only one phase in the mixture, this is a scalar 1.

        """

        self.saturation: Callable[[Sequence[pp.Grid]], pp.ad.Operator]
        """Phase saturation indicates how much of the (local) volume is occupied by this
        phase.

        - Math. Dimension:        scalar
        - Phys. Dimension:        [-] fractional

        This callable is assigned by a mixture mixin.

        If the phase is assigned as the reference phase and the reference
        phase is eliminated, this is a dependent operator.
        Otherwise it is a variable.

        If there is only one phase in the mixture, this is a scalar 1.

        """

        self.fraction_of: dict[Component, Callable[[Sequence[pp.Grid]], pp.ad.Operator]]
        """Molar fractions per component in a phase.

        Indicates how many of the moles in a phase belong to a component
        (relative fraction).

        - Math. Dimension:        scalar
        - Phys. Dimension:        [-] fractional

        The callables are assigned by a mixture mixin.

        As of now, fractions of components in a phase are always introduced as
        independent variables.

        """

        self.normalized_fraction_of: dict[
            Component, Callable[[Sequence[pp.Grid]], pp.ad.Operator]
        ]
        """Normalized versions of :attr:`fraction_of`.

        For a component i it holds

        .. math::

            x_{n, ij} = \\dfrac{x_{ij}}{\\sum_k x_{kj}}

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
    def num_components(self) -> int:
        """Number of set components."""
        return len(self.components)

    def compute_properties(
        self,
        p: np.ndarray,
        T: np.ndarray,
        xn: Sequence[np.ndarray],
        store: bool = True,
    ) -> PhaseState:
        """Convenience method to compute the phase properties using the underlying EoS
        and store the values in the respective AD operator.

        Parameters:
            p: ``shape=(N,)``

                Pressure.
            T: ``shape=(N,)``

                Temperature.
            xn: ``shape=(num_components, N)``

                (Normalized) Component fractions in this phase.
                The number of values must correspond to the number expected by the
                underlying EoS.
            store: ``default=True``

                A flag to store the computed values in the respective operators
                representing thermodynamic properties.

        Returns:
            A data structure containing the phase properties in numerical format.

        """
        state = self.eos.compute_phase_state(self.type, p, T, xn)

        if store:
            self.density.value = state.rho
            self.density.derivatives = state.drho
            self.volume.value = state.v
            self.volume.derivatives = state.dv
            self.enthalpy.value = state.h
            self.enthalpy.derivatives = state.dh

            self.viscosity.value = state.mu
            self.viscosity.derivatives = state.dmu
            self.conductivity.value = state.kappa
            self.conductivity.derivatives = state.dkappa

            for i, comp in enumerate(self):
                self.fugacity_of[comp].value = state.phis[i]
                self.fugacity_of[comp].derivatives = state.dphis[i]

        return state


class Mixture:
    """Basic mixture class managing modelled components and phases.

    The mixture class serves as a container for components and phases and to determine
    the reference component and phase.

    If also allocates attributes for some thermodynamic properites of a mixture, which
    are required by the remaining framework.

    The Mixture allows only one gas-like phase, and it must be modelled with at least
    1 component and 1 phase.

    Flash algorithms are built around the mixture management utilities of this class.

    Important:
        - The first, non-gas-like phase is treated as the reference phase.
        - The first component is set as reference component.
        - Choice of reference phase and component influence the choice of equations and
          variables, keep that in mind. It might have numeric implications.

    Notes:
        Be careful when modelling mixtures with 1 component. This singular case is not
        fully supported by the framework.

    Parameters:
        components: A list of components to be added to the mixture.
            This are the chemical species which can appear in multiple phases.
        phases: A list of phases to be modelled.

    Raises:
        AssertionError: If the model assumptions are violated.

            - 1 gas phase must be modelled.
            - At least 1 component must be present.
            - At least 1 phase must be modelled.

    """

    def __init__(
        self,
        components: list[Component],
        phases: list[Phase],
    ) -> None:
        # modelled phases and components
        self._components: list[Component] = []
        """A list of components passed at instantiation."""
        self._phases: list[Phase] = []
        """A list of phases passed at instantiation."""

        # a container holding objects already added, to avoid double adding
        doubles = []
        # Lists of gas-like and liquid-like phases
        gaslike_phases: list[Phase] = list()
        other_phases: list[Phase] = list()

        for comp in components:
            # add, Avoid double components
            if comp.name not in doubles:
                doubles.append(comp.name)
                self._components.append(comp)

        for phase in phases:
            # add, avoid double phases
            if phase.name not in doubles:
                doubles.append(phase.name)
                # add phase
                if phase.type == 1:
                    gaslike_phases.append(phase)
                else:
                    other_phases.append(phase)

        self._phases = other_phases + gaslike_phases

        # checking model assumptions
        assert len(self._components) > 0, "At least 1 component required."
        assert len(self._phases) > 0, "At least 1 phase required."
        assert len(gaslike_phases) == 1, "Only 1 gas-like phase is permitted."

        ### PUBLIC

        self.enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        """A representation of the mixture enthalpy as a callable on some domains.

        This callable is assigned by a mixture mixin.

        """

        self.density: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        """A representation of the mixture density as a callable on some domains.

        This callable is assigned by a mixture mixin.

        """

        self.volume: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        """A representation of the mixture volume as a callable on some domains.

        This callable is assigned by a mixture mixin.

        """

    def __str__(self) -> str:
        """
        Returns:
            A string representation of the composition, with information about present
            components and phases.

        """
        out = f"Composition with {self.num_components} components:"
        for component in self.components:
            out += f"\n\t{component.name}"
        out += f"\nand {self.num_phases} phases:"
        for phase in self.phases:
            out += f"\n\t{phase.name}"
        return out

    @property
    def num_components(self) -> int:
        """Number of components in this mixture."""
        return len(self._components)

    @property
    def num_phases(self) -> int:
        """Number of phases phases in this mixture."""
        return len(self._phases)

    @property
    def num_equilibrium_equations(self) -> int:
        """Number of necessary equilibrium equations for this composition, based on the
        number of added components and phases ``num_components * (num_phases - 1)``.

        Equilibrium equation (isofugacity constraints) are always formulated w.r.t. the
        reference phase, hence ``num_phases - 1``.

        """
        return self.num_components * (self.num_phases - 1)

    @property
    def components(self) -> Generator[Component, None, None]:
        """
        Note:
            The first component is always the reference component.

        Yields:
            Components added to the composition.

        """
        for C in self._components:
            yield C

    @property
    def phases(self) -> Generator[Phase, None, None]:
        """
        Note:
            The first phase is always the reference phase.

        Yields:
            Phases modelled by the composition class.

        """
        for P in self._phases:
            yield P

    @property
    def reference_phase(self) -> Phase:
        """Returns the reference phase.

        As of now, the first **non-gas-like** phase is declared as reference
        phase (based on :attr:`~porepy.composite.phase.Phase.gaslike`).

        The implications of a phase being the reference phase include:

        - Phase fraction and saturation can be dependent expression, not variables
          (elimination by unity).

        """
        return self._phases[0]

    @property
    def reference_component(self) -> Component:
        """Returns the reference component.

        As of now, the first component is declared as reference component.

        The implications of a component being the reference component include:

        - The mass constraint can be eliminated, since it is linear dependent on the
          other mass constraints due to various unity constraints.

        """
        return self._components[0]

    def compute_properties(
        self,
        p: np.ndarray,
        T: np.ndarray,
        xn: Sequence[Sequence[np.ndarray]],
        store: bool = True,
    ) -> list[PhaseState]:
        """Convenience function to evaluate and optionally store the physical
        properties of all phases in the mixture.

        See :meth:`Phase.compute_properties` for more.

        Parameters:
            p: ``shape=(N,)``

                Pressure.
            T: ``shape=(N,)``

                Temperature.
            X: ``shape=(num_phases, num_components, N)``

                A nested sequence containing for each phase a sub-sequence of
                (normalized) component fractions in that phase.
            store: ``default=True``

                Flag to store or return the results

        """
        results: list[PhaseState] = list()
        for j, phase in enumerate(self.phases):
            x_j = xn[j]
            props = phase.compute_properties(p, T, x_j, store=store)
            results.append(props)
        return results
