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
   problem. It can appear in multiple phases.
   It also has abstract thermodynamic properties, which need to be
   implemented for each component based on some experimental data.
2. :class:`Compound`:
   Additionally to being a variable quantity, this class has other species
   with related solute fractions functioning as parameters for thermodynamic
   properties. The solute fractions are **not** variables of the equilibrium problem.
   They might nevertheless be transportable by f.e. a coupled flow problem.
3. :class:`Phase`:
   An object representing a physical phase like gas-phase or a liquid-phase.
   In the unified setting, all phases contain every modelled component.
   They must ergo provide representations of that component's fraction in a phase and
   it's fugacity.
   The phase has also physical properties (like density) which come into play when
   formulating the phase.
4. :class:`Mixture`:
    A basic representation of a mixture which is a collection of anticipated phases and
    present components. The overall fraction of a component must never be zero.
    Serves as a managing instance and provides functionalities to formulate the flash
    equations using PorePy's AD framework.

"""
from __future__ import annotations

import abc
from dataclasses import asdict
from typing import Any, Generator, Literal, Optional, Sequence, overload

import numpy as np

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from ._core import COMPOSITIONAL_VARIABLE_SYMBOLS, R_IDEAL
from .chem_species import ChemicalSpecies, FluidSpecies
from .composite_utils import AdProperty, safe_sum
from .states import FluidState, PhaseState

__all__ = [
    "Component",
    "Compound",
    "Phase",
    "Mixture",
]


class Component(abc.ABC, FluidSpecies):
    """Abstract base class for components modelled inside a mixture.

    Components are chemical species which possibly go through phase transitions and
    appear in multiple phases.
    They represent a genuine component in the flash problem.

    Provides a variable representing the molar fraction of this component
    (feed fraction), cell-wise in a computational domain.

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
        self.fraction: pp.ad.Operator
        """Overall fraction, or feed fraction, for this component.

        It indicates how many of the total moles belong to this component (cell-wise).

        - Math. Dimension:        scalar
        - Phys. Dimension:        [%] fractional

        The overall fraction is always considered constant in the flash problem,
        but possible a primary variable in other physics.

        This attribute is assigned by a mixture instance, when this component is added.

        If the component is assigned as the reference component, this is a dependent
        operator. Otherwise it is a variable.

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
            ChemicalSpecies, pp.ad.MixedDimensionalVariable
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


class Phase:
    """Base class for phases in a multiphase multicomponent mixture.

    The term 'phase' as used here refers to physical states of matter.
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
        eos: pp.composite.EoSCompiler,
        type: int,
        name: str,
    ) -> None:
        self._components: list[Component] = []
        """Private list of present components (see :meth:`components`)."""

        ### PUBLIC

        self.eos: pp.composite.EoSCompiler = eos
        """The EoS passed at instantiation."""

        self.type: int = int(type)
        """Physical type of this phase. Passed at instantiation."""

        self.name: str = str(name)
        """Name given to the phase at instantiation."""

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
    def num_components(self) -> int:
        """Number of added components."""
        return len(self._components)

    @property
    def components(self) -> list[Component]:
        """A container for all components modelled in this phase.

        Important:
            The user does not need to set the components directly in a phase.
            The mixture class sets the components of every phase in the unified setting.
            Manipulate with care once the mixture is set up.

            By setting components, previous references to fugiacities and fractions of
            components are dereferenced.

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
        self.fugacity_of = dict()
        self.fraction_of = dict()
        self.normalized_fraction_of = dict()
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

    @overload
    def compute_properties(
        self,
        p: np.ndarray,
        T: np.ndarray,
        xn: Sequence[np.ndarray],
        store: Literal[True] = True,
    ) -> None:
        # Typing overload for default return value: None, properties are stored
        ...

    @overload
    def compute_properties(
        self,
        p: np.ndarray,
        T: np.ndarray,
        xn: Sequence[np.ndarray],
        store: Literal[False] = False,
    ) -> PhaseState:
        # Typing overload for default return value: Properties are returned
        ...

    def compute_properties(
        self,
        p: np.ndarray,
        T: np.ndarray,
        xn: Sequence[np.ndarray],
        store: bool = True,
    ) -> PhaseState | None:
        """Convenience method to compute the phase properties using the underlying EoS
        and store the values in the respective AD operator.

        Parameters:
            p: Pressure.
            T: Temperature.
            xn: ``len=num_components``

                (Normalized) Component fractions in this phase.
                The number of values must correspond to the number expected by the
                underlying EoS.
            store: ``default=True``

                A flag to store the computed values in the respective attributes
                representing thermodynamic properties.

        Returns:
            If ``store=False``, returns the computed properties.

        """
        state = self.eos.compute_phase_state(self.type, p, T, xn)

        if store:
            self.density.value = state.rho
            self.volume.value = state.v
            self.viscosity.value = state.mu
            self.conductivity.value = state.kappa
            self.enthalpy.value = state.h

            for i, comp in enumerate(self):
                self.fugacity_of[comp].value = state.phis[i]
        else:
            return state


class Mixture:
    """Basic mixture class managing modelled components and phases.

    This is a layer-1 implementation of a mixture, containing the core functionality
    and objects representing the variables and basic thermodynamic properties.

    The equilibrium problem is set in the unified formulation (unified flash).
    It allows one gas-like phase and an arbitrary number of liquid-like phases.

    Flash algorithms are built around the mixture management utilities of this class.

    This class implements also basic intersection points with PorePy's AD framework and
    hence the remaining modelling concepts in this package.

    Important:
        - The first, non-gas-like phase is treated as the reference phase.
          Its molar fraction and saturation will not be part of the primary variables.
        - The first component is set as reference component.
          Its mass conservation will not be part of the equilibrium equations.
        - Choice of reference phase and component influence the choice of equations and
          variables, keep that in mind. It might have numeric implications.

    Notes:
        If the user wants to model a single-component mixture, a dummy component must be
        added as the first component (reference component for elimination),
        with a feed fraction small enough s.t. its effects on the
        thermodynamic properties are negligible.

        This approximates a single-component mixture, due to the flash system being
        inherently singular in this case. Numerical issues can appear if done so!

    Parameters:
        components: A list of components to be added to the mixture.
            This are the chemical species which can appear in multiple phases.
        phases: A list of phases to be modelled.

    Raises:
        AssertionError: If the model assumptions are violated.

            - 1 gas phase must be modelled.
            - At least 2 components must be present.
            - At least 2 phases must be modelled.

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
        assert len(gaslike_phases) == 1, "Only 1 gas-like phase is permitted."
        assert len(self._components) > 1, "At least 2 components required."
        assert len(self._phases) > 1, "At least 2 phases required."

        self._num_cells: int
        """The fluid is present in each cell and each state function has one DOF per
        cell. This is calculated during :meth:`set_up_ad`."""

        ### PUBLIC

        self.system: pp.ad.EquationSystem
        """The AD-system set during :meth:`set_up_ad`.

        This attribute is not available prior to that.

        """

        self.molar_fraction_variables: list[str]
        """A list of names of molar fractional variables, which are unknowns in the
        equilibrium problem.

        These include

        - phase fractions
        - phase compositions

        This list is created in :meth:`set_up_ad`.

        """

        self.saturation_variables: list[str]
        """A list of names of saturation variables, which are unknowns in the
        equilibrium problem.

        Note:
            Saturations are only relevant in equilibrium problems involving the volume
            of the mixtures.
            Otherwise they can be calculated a posterior.

        This list is created in :meth:`set_up_ad`.

        """

        self.feed_fraction_variables: list[str]
        """A list of names of feed fraction variables per present components.

        Note:
            Feed fractions are constant in non-reactive mixtures, since there the number
            of moles of a species is assumed to be constant.

        This list is created in :meth:`set_up_ad`.

        """

        self.solute_fraction_variables: dict[Compound, list[str]] = dict()
        """A dictionary containing per compound (key) names of solute fractions
        for each solute in that compound.

        Note:
            Solute fractions are assumed constant in non-reactive mixtures.
            They are not used anywhere in the flash by default.

        This map is created in :meth:`set_up_ad`.

        """

        self.enthalpy: pp.ad.Operator
        """An operator representing the mixture enthalpy as a sum of
        :attr:`~porepy.composite.phase.Phase.enthalpy` weighed with
        :attr:`~porepy.composite.phase.Phase.fraction` for each phase.

        This operator is created in :meth:`set_up_ad`.

        """

        self.density: pp.ad.Operator
        """An operator representing the mixture density as a sum of
        :attr:`~porepy.composite.phase.Phase.density` weighed with
        :attr:`~porepy.composite.phase.Phase.saturation`.

        This operator is created in :meth:`set_up_ad`.

        """

        self.volume: pp.ad.Operator
        """An operator representing the mixture volume as a reciprocal of
        :attr:`density`.

        This operator is created in :meth:`set_up_ad`.

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

    def _instantiate_frac_var(
        self, ad_system: pp.ad.EquationSystem, name: str, subdomains: list[pp.Grid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Auxiliary function to instantiate variables with 1 degree per cell.

        STATE and ITERATE are set to zero.

        """
        var = ad_system.create_variables(name=name, subdomains=subdomains)
        ad_system.set_variable_values(
            np.zeros(self._num_cells),
            variables=[name],
            iterate_index=0,
            time_step_index=0,
        )
        return var

    def set_up_ad(
        self,
        ad_system: Optional[pp.ad.EquationSystem] = None,
        subdomains: list[pp.Grid] = None,
        eliminate_ref_phase: bool = True,
        eliminate_ref_feed_fraction: bool = True,
    ) -> list[pp.Grid]:
        """Basic set-up of mixture.

        This creates the fractional variables in the phase-equilibrium problem.

        The following AD operators are created (with 0-values, if variable):

        - :attr:`Component.fraction`
        - attr:`Phase.fractions` (variables, except for ref. phase if eliminated).
        - :attr:`Phase.saturation` (variables, except for ref. phase if eliminated).
        - :attr:`Phase.fraction_of` (variables for each components).
        - :attr:`Phase.normalized_fraction_of`
          (expressions dependent on :attr:`Phase.fraction_of` for each phase).

        Parameters:
            ad_system: ``default=None``

                If given, this class will use the AD system and the respective
                mixed-dimensional domain to represent all involved variables cell-wise
                in each subdomain.

                If not given (None), a single-cell domain and respective AD system are
                created.
            subdomains: ``default=None``

                If ``ad_system`` is not None, restrict this mixture to a given set of
                grids by defining this keyword argument.

                Otherwise, every subdomain found in the grid stored in ``ad_system``
                will be used as domain for this mixture.

                Important:
                    All components and phases are present in each subdomain-cell.
                    Their fractions are introduced as cell-wise, scalar unknowns.

            eliminate_reference_phase: ``default=True``

                An optional flag to eliminate reference phase variables from the
                system, and hence reduce the system.

                The saturation and fraction can be eliminated by unity using other
                saturations and fractions.

                If True, the attributes :attr:`Phase.fraction` and
                :attr:`Phase.saturation` will **not** be variables, but expressions.
            eliminate_ref_feed_Fraction: ``default=True``

                An optional flag to eliminate the feed fraction of the reference
                component from the system as a variable, hence to reduce the system.

                It can be eliminated by unity using the other feed fractions.

                If True, the attribute :attr:`~Component.fraction` will **not** be a
                variable, but an expression.

        Returns:
            A list of subdomains on which the mixture unknowns are defined.

            If ``subdomains`` and ``ad_system`` are given, this returns the intersection
            of ``subdomains`` and the subdomains found in the mixed-dimensional grid of
            ``ad_system``.

        """
        domains: list[pp.Grid]
        if ad_system is None:
            sg = pp.CartGrid([1, 1], [1, 1])
            mdg = pp.MixedDimensionalGrid()
            mdg.add_subdomains(sg)
            mdg.compute_geometry()

            ad_system = pp.ad.EquationSystem(mdg)
            domains = mdg.subdomains()
        else:
            if subdomains is None:
                domains = ad_system.mdg.subdomains()
            else:
                domains = [sd for sd in subdomains if sd in ad_system.mdg.subdomains()]

        self.system = ad_system
        self._num_cells = int(sum([sd.num_cells for sd in domains]))
        self.reference_phase_eliminated = bool(eliminate_ref_phase)

        ### Creating fractional variables.
        ## First, create all component fractions and solute fraction for compounds
        self.feed_fraction_variables = list()
        self.solute_fraction_variables = dict()

        if eliminate_ref_feed_fraction:
            z_R: pp.ad.Operator = 1.0 - safe_sum(
                [c.fraction for c in self.components if c != self.reference_component]
            )  # type: ignore
            z_R.set_name("ref-component-fraction-by-unity")
            self.reference_component.fraction = z_R
        else:
            name = (
                f"{COMPOSITIONAL_VARIABLE_SYMBOLS['component_fraction']}"
                + f"_{self.reference_component.name}"
            )
            self.reference_component.fraction = self._instantiate_frac_var(
                ad_system, name, domains
            )
            self.feed_fraction_variables.append(name)

        for comp in self.components:
            if comp != self.reference_component:
                name = (
                    f"{COMPOSITIONAL_VARIABLE_SYMBOLS['component_fraction']}"
                    + f"_{comp.name}"
                )
                comp.fraction = self._instantiate_frac_var(ad_system, name, domains)
                self.feed_fraction_variables.append(name)

            if isinstance(comp, Compound):
                self.solute_fraction_variables[comp] = list()
                for solute in comp.solutes:
                    name = (
                        f"{COMPOSITIONAL_VARIABLE_SYMBOLS['solute_fraction']}"
                        + f"_{solute.name}_{comp.name}"
                    )
                    comp.solute_fraction_of[solute] = self._instantiate_frac_var(
                        ad_system, name, domains
                    )
                    self.solute_fraction_variables[comp].append(name)

        ## Second, create all saturations and molar phase fractions.
        # adding all components to every phase, according to unified procedure
        # this dereferences nay previously set lists
        for phase in self.phases:
            phase.components = list(self.components)
        self.saturation_variables = list()
        self.molar_fraction_variables = list()

        if self.reference_phase_eliminated:
            s_R: pp.ad.Operator = 1.0 - safe_sum(
                [p.saturation for p in self.phases if p != self.reference_phase]
            )  # type: ignore
            s_R.set_name("ref-phase-saturation-by-unity")
            self.reference_phase.saturation = s_R

            y_R: pp.ad.Operator = 1.0 - safe_sum(
                [p.fraction for p in self.phases if p != self.reference_phase]
            )  # type: ignore
            y_R.set_name("ref-phase-fraction-by-unity")
            self.reference_phase.fraction = y_R
        else:
            name = (
                f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_saturation']}"
                + f"_{self.reference_phase.name}"
            )
            self.reference_phase.saturation = self._instantiate_frac_var(
                ad_system, name, domains
            )
            self.saturation_variables.append(name)

            name = (
                f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_fraction']}"
                + f"_{self.reference_phase.name}"
            )
            self.reference_phase.fraction = self._instantiate_frac_var(
                ad_system, name, domains
            )
            self.molar_fraction_variables.append(name)

        for phase in self.phases:
            if phase != self.reference_phase:
                name = (
                    f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_saturation']}_{phase.name}"
                )
                phase.saturation = self._instantiate_frac_var(ad_system, name, domains)
                self.saturation_variables.append(name)

                name = (
                    f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_fraction']}_{phase.name}"
                )
                phase.fraction = self._instantiate_frac_var(ad_system, name, domains)
                self.molar_fraction_variables.append(name)

        ## Third, create all phase composition varaiables per phase per component
        for phase in self.phases:
            for comp in self.components:
                name = (
                    f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_composition']}"
                    + f"_{comp.name}_{phase.name}"
                )
                phase.fraction_of.update(
                    {comp: self._instantiate_frac_var(ad_system, name, domains)}
                )
                self.molar_fraction_variables.append(name)

        ## Fifth, create operators representing normalized fractions
        for phase in self.phases:
            sum_j: pp.ad.Operator = safe_sum(
                list(phase.fraction_of.values())
            )  # type: ignore
            for comp in self.components:
                name = f"{phase.fraction_of[comp].name}_normalized"
                x_ij_n = phase.fraction_of[comp] / sum_j
                x_ij_n.set_name(name)
                phase.normalized_fraction_of.update({comp: x_ij_n})

        ### Creating mixture properties
        ## First, mixture density
        self.density: pp.ad.Operator = safe_sum(
            [phase.saturation * phase.density for phase in self.phases]
        )  # type: ignore
        self.density.set_name("mixture-density")

        ## Second, mixture volume as the reciprocal of density
        self.volume: pp.ad.Operator = self.density ** (-1)
        self.volume.set_name("mixture-volume")

        ## Third, mixture enthalpy
        self.enthalpy: pp.ad.Operator = safe_sum(
            [phase.fraction * phase.enthalpy for phase in self.phases]
        )  # type: ignore
        self.enthalpy.set_name("mixture-enthalpy")

        return domains

    @overload
    def compute_properties(
        self,
        p: NumericType,
        T: NumericType,
        xn: Sequence[Sequence[NumericType]],
        store: Literal[True] = True,
    ) -> None:
        # Typing overload for default return value: None, properties are stored
        ...

    @overload
    def compute_properties(
        self,
        p: np.ndarray,
        T: np.ndarray,
        xn: Sequence[Sequence[np.ndarray]],
        store: Literal[False] = False,
    ) -> list[PhaseState]:
        # Typing overload for default return value: Properties are returned
        ...

    def compute_properties(
        self,
        p: np.ndarray,
        T: np.ndarray,
        xn: Sequence[Sequence[np.ndarray]],
        store: bool = True,
    ) -> list[PhaseState] | None:
        """Convenience function to evaluate and optionally store the physical
        properties of all phases in the mixture.

        See :meth:`Phase.compute_properties` for more.

        Parameters:
            p: Pressure.
            T: Temperature.
            X: ``shape=(num_phases, num_components)``

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
        if not store:
            return results

    def fractional_state_from_vector(
        self,
        state: Optional[np.ndarray] = None,
    ) -> FluidState:
        """Uses the AD framework the currently stored values of fractions.

        Convenience function to get the values for fractions in iterative procedures.

        Evaluates:

        1. Overall fractions per component
        2. Molar fractions per phase
        3. Volumetric fractions per phase (saturations)
        4. Extended fractions per phase per component

        Parameters:
            state: ``default=None``

                Argument for the evaluation methods of the AD framework.
                Can be used to assemble a fluid state from an alternative global vector
                of unknowns.

        Returns:
            A partially filled fluid state data structure containing the above
            fractional values.

        """

        z = np.array(
            [c.fraction.evaluate(self.system, state).val for c in self.components]
        )

        y = np.array([p.fraction.evaluate(self.system, state).val for p in self.phases])

        sat = np.array(
            [p.saturation.evaluate(self.system, state).val for p in self.phases]
        )

        x = [
            np.array(
                [
                    p.fraction_of[c].evaluate(self.system, state).val
                    for c in self.components
                ]
            )
            for p in self.phases
        ]

        return FluidState(z=z, y=y, sat=sat, phases=[PhaseState(x=x_) for x_ in x])
