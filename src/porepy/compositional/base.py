"""This module contains the base classes for elements of a mixture:

1. :class:`Component`:
   A phase-changing representation of a species involving some physical constants.
   Additionally, this class represents a variable quantity in the equilibrium
   problem. It can appear in multiple phases and has a fraction of the overall mass
   associated with it.

2. :class:`Compound`:
   A compound represents a combination of chemical species, which can be bundled into
   1 component-like instance. While one chemical species serves as the solvent,
   an arbitrary number of other species can be set as active tracers.
   Pseudo-components are not considered individually in the equilibrium problem, but the
   compound as a whole.
   But they are transportable quantities, with a fractional value relative to the
   overall fraction of the compound.

3. :class:`Phase`:
   An object representing a physical phase like gas-phase or a liquid-phase.
   A phase can contain multiple phase-changing components and the modeller must set
   those explicitly (see :attr:`Phase.components`).
   Components in a phase are characterized by their fraction of mass
   (:attr:`Phase.partial_fraction_of`), relative to the
   fraction of mass in a phase (:attr:`Phase.fraction`).

   The phase has also physical properties (like density and enthalpy) which come into
   play when formulating more complex equilibrium models coupled with flow & transport.

4. :class:`FluidMixture`:
    A basic representation of a mixture which is a collection of anticipated phases and
    present components, putting them into their contexts.

    Serves as a managing instance and provides functionalities to formulate flow &
    transport & flash equations using PorePy's AD framework.

Note:
    Phases are meant to be based on an Equation of State.
    A basic interface for such an equation of state is defined by :class:`AbstractEoS`.

Important:
    The physical units used here are in general SI units.
    For specific quantities, the framework can be used for both, molar and massic
    settings. Once chosen, it must be consistent throughout the set-up.

    Fractions are respectively molar or massic as well, though they are always
    dimensionless.

"""

from __future__ import annotations

import abc
from dataclasses import asdict
from typing import Callable, Generator, Sequence, Type, TypeVar

import numpy as np

import porepy as pp
from porepy.numerics.ad.functions import FloatType

from ._core import PhysicalState
from .chem_species import ChemicalSpecies
from .states import PhaseProperties
from .utils import CompositionalModellingError, safe_sum

__all__ = [
    "Component",
    "Compound",
    "AbstractEoS",
    "Phase",
    "FluidMixture",
]


class Component(ChemicalSpecies):
    """Base class for components modelled inside a mixture.

    Components are chemical species inside a mixture, which can go through phase
    transitions and appear in multiple :class:`Phase`.
    A component is identified by the (time-dependent) :meth:`fraction` of total mass
    belonging to the component.

    The fractions are assigned by the AD interface
    :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`, once the
    component is added to a mixture context.

    Note:
        Rather than instantiating a component directly, it is easier to use the
        class factory based on loaded species data (see :meth:`from_species`).

    Parameters:
        **kwargs: See parent (data-) class and its attributes.

    """

    def __init__(self, **kwargs) -> None:
        # NOTE Only for Python >= 3.10
        # Filter away kwargs that will not be recognized by ChemicalSpecies
        chem_species_kwargs = {
            k: v for k, v in kwargs.items() if k in ChemicalSpecies.__match_args__
        }
        super().__init__(**chem_species_kwargs)

        # creating the overall molar fraction variable
        self.fraction: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        """Overall fraction, or feed fraction, for this component, indicating how much
        of the total mass or moles belong to this component.

        Dimensionless, scalar field bound to the interval ``[0, 1]``.
        The sum of overall fractions must always equal 1.

        Note:
            This is a variable in flow and transport.
            The feed fraction of one arbitrarily chosen component can be eliminated
            by unity.

            If there is only 1 component, this should be a wrapped scalar with value 1.

        """

    @classmethod
    def from_species(
        cls: Type[_ComponentLike], species: ChemicalSpecies
    ) -> _ComponentLike:
        """Factory method for creating an instance of this class based on some chemical
        data.

        Parameters:
            species: Chemical species with constant parameters characterizing the
                component.

        Returns:
            A component instance to be used in PorePy.

        """
        return cls(**asdict(species))


_ComponentLike = TypeVar("_ComponentLike", bound=Component)


class Compound(Component):
    """A compound is a simplified, but meaningfully generalized set of chemical species
    inside a mixture, for which it makes sense to treat it as a single component.

    It is represents one species, the solvent, and contains arbitrary many
    active tracers (pseudo-components).

    A compound can appear in multiple phases and its thermodynamic properties are
    determined by the tracers present.

    Tracers are transportable and are represented by a fraction relative to
    the :attr:`~Component.fraction` of the compound, i.e. the moles/mass of them are
    given by a product of mixture density, compound fraction and tracer fraction.
    :attr:`tracer_fraction_of` are assigned by the AD interface
    :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`.

    Note:
        Due to the generalization, the solvent and individual tracers are not considered as
        genuine components which can transition into various phases,
        but rather as parameters in the equilibrium problem problem.
        Only the compound as a whole splits into various phases. Fractions in phases
        are associated with the compound.
        Solvent and tracer fractions are not variables in the flash problem.

    Example:
        1. Brines with species salt and water as tracer and solvent, where it is
           sufficient to calculate how much brine is in vapor or liquid form,
           and the information about how the salt distributes across phases is
           irrelevant. The salt in this case is a **transportable** quantity,
           whose concentration acts as a parameter in the flash.

        2. The black-oil model, where black-oil is treated as a compound with various
           hydrocarbons as active tracers.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._active_tracers: list[ChemicalSpecies] = []
        """A list containing present tracers as species."""

        self.tracer_fraction_of: dict[
            ChemicalSpecies, Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        ] = {}
        """A dictionary containing per present tracer (key) the tracer
        fraction of it with respect to the compound's overall fraction.

        Dimensionless, scalar field bound to the interval ``[0, 1]``.
        The sum of relative fractions must be bound by 1.

        Note:
            Pseudo-components are transportable quantities!
            This is hence a variable in flow and transport.
            The fraction of the solvent is eliminated by unity.

        """

    def __iter__(self) -> Generator[ChemicalSpecies, None, None]:
        """Iterator overload to iterate over present tracers."""
        for tracer in self._active_tracers:
            yield tracer

    @property
    def active_tracers(self) -> list[ChemicalSpecies]:
        """
        Important:
            Pseudo-components must be set before the compound is added to a mixture.

        Parameters:
            tracers: A list of chemical species to be added to the compound as active
                tracers. Uniqueness of the species is enforced in the setter.

        Raises:
            ValueError: If names or CASr numbers are not unique per tracer.

        Returns:
            Active tracers present in this compound.

        """
        return [s for s in self._active_tracers]

    @active_tracers.setter
    def active_tracers(self, tracers: list[ChemicalSpecies]) -> None:
        # avoid double species
        double_names = []
        double_casr = []
        self._active_tracers = []
        for s in tracers:
            double_names.append(s.name)
            double_casr.append(s.CASr_number)
            self._active_tracers.append(s)
        if len(set(double_casr)) < len(tracers):
            raise ValueError("CASr numbers must be unique per species.")
        if len(set(double_names)) < len(tracers):
            raise ValueError("Names must be unique per species.")

    def compound_molar_mass(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """The molar mass of a compound depends on how much of the tracers are present.

        It is a sum of the molar masses of present species, weighed with their
        respective fraction, including the solvent.

        Important:
            This is the *average* molar mass (https://en.wikipedia.org/wiki/Molar_mass),
            assuming :attr:`tracer_fraction_of` contains *molar* fractions.

        Parameters:
            domains: A sequence of subdomains or boundaries on which the operator
                should be assembled. Used to call :attr:`tracer_fraction_of`.

        Returns:
            The molar mass of the compound.

        """
        X = [self.tracer_fraction_of[pc](domains) for pc in self.active_tracers]
        M = pp.ad.Scalar(self.molar_mass) * (pp.ad.Scalar(1.0) - safe_sum(X))
        # But the molar mass [kg / mol] can be computed using molar masses of the
        # solvent and tracers, weighing them with respective fractions and summing them.
        # NOTE molar masses are required for the gravitational flux, in a setting with
        # molar units.

        for pc, x in zip(self._active_tracers, X):
            M += pp.ad.Scalar(pc.molar_mass) * x
        M.set_name(f"compound_molar_mass_{self.name}")
        return M

    def molalities_from_fractions(self, *fractions: FloatType) -> list[FloatType]:
        """Computes the molalities of present active tracers, based on given
        ``fractions``.

        Notes:
            1. The order of ``fractions`` must match the order in
               :meth:`active_tracers`
            2. The solvent molality is always the reciprocal of the solvent molar mass.
               Hence, it is always a scalar, and not computed here.

        Parameters:
            *fractions: Tracer fractions of present tracers in numerical format.

        Raises:
            ValueError: If the number of provided values does not match the number
                of present tracers.

        Returns:
            A list of molality values per tracer in :attr:`active_tracers`.

        """
        if len(fractions) != len(self._active_tracers):
            raise ValueError(
                f"Need {len(self._active_tracers)} values, {len(fractions)} given."
            )

        molalities = []

        # NOTE for more information on the formula applied here see
        # https://en.wikipedia.org/wiki/Molality

        # solvent fraction
        x_s = 1 - safe_sum(fractions)
        for fraction in fractions:
            m_i = fraction / (x_s * self.molar_mass)
            molalities.append(m_i)

        return molalities

    def fractions_from_molalities(self, *molalities: FloatType) -> list[FloatType]:
        """Reverse operation for :meth:`molalities_from_fractions`.

        Parameters:
            *molalities: Molalities per present tracer.

        Raises:
            ValueError: If the number of provided values does not match the number
                of tracers in :attr:`active_tracers`.

        Returns:
            A list of tracer fractions calculated from molalities.

        """
        if len(molalities) != len(self._active_tracers):
            raise ValueError(
                f"Need {len(self._active_tracers)} values, {len(molalities)} given."
            )

        m_sum = safe_sum(molalities)
        X = []

        # NOTE for more information on the formula applied here see
        # https://en.wikipedia.org/wiki/Molality

        for m in molalities:
            x_i = self.molar_mass * m / (1 + self.molar_mass * m_sum)
            X.append(x_i)

        return X


class AbstractEoS(abc.ABC):
    """Abstract EoS class defining the interface between thermodynamic input
    and resulting structure containing thermodynamic properties of a phase.

    Component properties required for computations can be extracted in the constructor.

    Note:
        This class is called 'abstract EoS'. Users can implement any correlations but
        are encouraged to focus on thermodynamic consistency.

        Phase properties are defined as secondary expressions, and the framework is
        able to pick up the dependencies and call :meth:`compute_phase_properties` with
        the right values.

        For more information on this, see
        :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin` and
        :meth:`~porepy.compositional.compositional_mixins.FluidMixtureMixin.
        dependencies_of_phase_properties`.

    Parameters:
        components: A sequence of components for which the EoS is instantiated.

    Raises:
        CompositionalModellingError: If no components passed.

    """

    def __init__(self, components: Sequence[Component]) -> None:
        self._nc: int = len(components)
        """Number of components passed at instantiation."""

        if self._nc == 0:
            raise CompositionalModellingError("Cannot create an EoS with no components")

    @abc.abstractmethod
    def compute_phase_properties(
        self, phase_state: PhysicalState, *thermodynamic_input: np.ndarray
    ) -> PhaseProperties:
        """ "Abstract method to compute the properties of a phase based any
        thermodynamic input.

        Examples:
            1. For a single component mixture, the thermodynamic input may consist of
               just pressure and temperature.
               For isothermal models, it may be just pressure.
            2. For general multiphase-multicomponent mixtures, the thermodynamic input
               may consist of pressure, temperature and partial fractions of components
               in a phase.
            3. For correlations which indirectly represent the solution of the
               fluid phase equilibrium problem, the signature might as well be
               pressure, temperature and independent overall fractions.
            4. For complex models, temperature can be replaced by enthalpy, for example.

        Parameters:
            phase_state: The physical phase state for which to compute values.
            *thermodynamic_input: Vectors with consistent shape ``(N,)`` representing
                any combination of thermodynamic input variables.

        Returns:
            A datastructure containing all relevant phase properties and their
            derivatives w.r.t. the dependencies (``thermodynamic_input``).

        """
        ...


class Phase:
    """Base class for phases in a fluid mixture.

    The term 'phase' as used here refers to physical states of matter.
    A phase is identified by the (time-dependent) region/volume it occupies (saturation)
    and the fraction of moles/mass belonging to this phase.

    Phases have physical properties, dependent on some thermodynamic input.
    They are usually assigned by an instance of
    :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`, and include
    **only** properties relevant for flow & transport problems:

    - :attr:`density`
    - :attr:`specific_volume`
    - :attr:`specific_enthalpy`
    - :attr:`viscosity`
    - :attr:`thermal_conductivity`
    - :attr:`fugacities_of`

    Components must be modelled explicitly in a phase, by setting :attr:`components`.
    (see also :class:`~porepy.compositional.compositional_mixins.
    FluidMixtureMixin.set_components_in_phases`).

    Important:
        The components must be set in a phase, before adding the two contexts into
        a mixture.

    The mixin creates fractional unknowns as well, including

    - :attr:`fraction`
    - :attr:`saturation`
    - :attr:`extended_fraction_of`
    - :attr:`partial_fraction_of`

    Both, properties and fractional unknowns, are only available once put into a
    context by creating a :class:`Mixture`.

    Note:
        Dependent on whether this phase is assigned as the reference phase or not,
        the operator representing the fraction or saturation might either be a genuine
        variable (:class:`~porepy.numerics.ad.operators.MixedDimensionalVariable`)
        or a dependent :class:`~porepy.numerics.ad.operators.Operator`,
        where the fraction and saturation were eliminated by unity respectively.

    Note:
        All extended fractions :attr:`extended_fraction_of` are genuine variables in the
        unified flash (:class:`~porepy.numerics.ad.operators.MixedDimensionalVariable`).

        All partial fractions in :attr:`partial_fraction_of` are dependent
        :class:`~porepy.numerics.ad.operators.Operator` -instances,
        created by normalization of fractions in :attr:`extended_fraction_of`.

        If the flow & transport model does not include an equilibrium formulation,
        the extended fractions are meaningless and the partialf ractions are independent
        variables instead.

    Parameters:
        eos: An EoS which provides means to compute physical properties of the phase.
            Can be different for different phases.
        state: The physical state this phase represents.
        name: Given name for this phase. Used as an unique identifier and for naming
            various variables and properties.

    """

    def __init__(
        self,
        eos: AbstractEoS,
        state: PhysicalState,
        name: str,
    ) -> None:

        self._ref_component_index: int = 0
        """See :meth:`reference_component_index`."""

        ### PUBLIC

        self.components: Sequence[Component]
        """A sequence of all components modelled in this phase.

        To be set by the user, or by some instance of
        :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`

        Once set, it should not be modified. Avoid multiple occurences of components.

        """

        self.eos: AbstractEoS = eos
        """The EoS passed at instantiation."""

        self.state: PhysicalState = state
        """Physical state declared at instantiation (see :attr:`PhysicalStates`)."""

        self.name: str = str(name)
        """Name given to the phase at instantiation."""

        self.density: pp.ad.SurrogateFactory
        """Density of this phase.

        Scalar field with physical dimensions ``[mol / m^3]`` or ``[kg / m^3]``.

        """

        self.specific_volume: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        """Specific volume of this phase.

        Scalar field with physical dimensions ``[m^3 / mol]`` or ``[m^3 / mol]``.

        Note:
            The specific volume is not a surrogate factory, like the other properties.
            It is always calculated as the reciprocal of density.

        """

        self.specific_enthalpy: pp.ad.SurrogateFactory
        """Specific enthalpy of this phase.

        Scalar field with physical dimensions ``[J / mol K]`` or ``[J / kg K]``.

        """

        self.viscosity: pp.ad.SurrogateFactory
        """Dynamic viscosity of this phase.

        Scalar field with physical dimensions``[kg / m / s]``.

        """

        self.conductivity: pp.ad.SurrogateFactory
        """Thermal conductivity of this phase.

        Scalar field with physical dimensions``[W / m / K]``.

        """

        self.fugacity_coefficient_of: dict[Component, pp.ad.SurrogateFactory]
        """Fugacitiy coefficients per component in this phase.

        Dimensionless, scalar field.

        """

        self.fraction: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        """Fraction of mass or moles in this phase, relative to the total amount.

        Dimensionless, scalar field bound to the interval ``[0, 1]``.
        The sum of phase fractions must always equal 1.

        Note:
            The reference phase fraction is usually eliminated by unity, hence not an
            independent variable.

            If only 1 phase is modelled in the mixture, it is a constant scalar 1.

        Fractions do not usually appear in the flow and transport problem, since they
        can be expressed through densities and saturations.
        In the standard set-up, they are purely local variables.

        """

        self.saturation: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        """Fraction of (pore) volume occupied by this phase.

        Dimensionless, scalar field bound to the interval ``[0, 1]``.
        The sum of saturations must always equal 1.

        Note:
            Same as for :meth:`fraction`, the reference phase saturation is usually
            eliminated by unity, hence not an independent variable.

            If only 1 phase is modelled in the mixture, it is a constant scalar 1.

        """

        self.extended_fraction_of: dict[
            Component, Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        ]
        """Extended molar fractions per component in a phase, used in the unified
        phase equilibrium formulation (see :attr:`partial_fraction_of`).

        Extended fractions are unique in the unified formulation and their sum does
        not necessarily equal 1, if a phase is absent (phase fraction is zero.)

        Extended fractions are independent variables in the CFLE setting, for all
        phases and all components in them.

        """

        self.partial_fraction_of: dict[
            Component, Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        ]
        """Partial (physical) fraction of a component, relative to the phase fraction.

        Dimensionless, scalar field bound to the interval ``[0, 1]``.
        The sum of partial fractions per phase must always equal 1.

        In the unified flash with extended fractions, this must be the normalized
        version of :attr:`fraction_of`:

        .. math::

            x_{ij} = \\dfrac{\\chi_{ij}}{\\sum_k \\chi_{kj}}

        Partial fractions are used in flow and transport to define mobilities.

        """

    def __iter__(self) -> Generator[Component, None, None]:
        """Iterator over components present in this phase.

        Notes:
            The order from this iterator is used for choosing e.g. the values in a
            list of 'numpy.array' when setting initial values.
            Use the order returned here every time you deal with
            component-related values for components in this phase.

        Yields:
            Components present in this phase.

        """
        for component in self.components:
            yield component

    @property
    def num_components(self) -> int:
        """Number of set components."""
        return len(self.components) if hasattr(self, "components") else 0

    @property
    def reference_component_index(self) -> int:
        """Returns the index of the component in :meth:`components`, which is designated
        as the reference component *in this phase*.

        Not to be confused with :meth:`FluidMixture.reference_component_index`.

        By default, the first component (0) is designated as the reference component.

        Important:
            Changing the index of the reference component changes which partial fraction
            is eliminated by unity of fractions.
            Its representation will be by unity, and more importantly, it will reduce
            the size of the system by 1.

        Parameters:
            index: A new index to be assigned.

        Raises:
            IndexError: If index is out of range of :meth:`components`.

        Returns:
            The index of the current component designated as the reference component in
            this phase.

        """
        return self._ref_component_index

    @reference_component_index.setter
    def reference_component_index(self, index: int) -> None:
        max_index = len(self.components) - 1
        if index < 0 or index > max_index:
            raise IndexError(f"Component index {index} out of range [0, {max_index}].")
        self._ref_component_index = int(index)

    @property
    def reference_component(self) -> Component:
        """The component in :attr:`components` corresponding to the
        :meth:`reference_component_index`."""
        return self.components[self.reference_component_index]

    def compute_properties(self, *thermodynamic_input: np.ndarray) -> PhaseProperties:
        """Shortcut to compute the properties calling
        :meth:`AbstractEoS.compute_phase_properties` of :attr:`eos` with :attr:`type` as
        argument."""
        return self.eos.compute_phase_properties(self.state, *thermodynamic_input)


class FluidMixture:
    """Basic fluid mixture class managing modelled components and phases.

    The mixture class serves as a container for components and phases and contains the
    specification of the reference component and phase.

    It also allocates attributes for some thermodynamic properites of a mixture, which
    are required by the remaining framework, which are assigned by an instance of
    :class:`~porepy.compositional.compositional_mixins.FluidMixtureMixin`.

    - :attr:`density`
    - :attr:`specific_enthalpy`
    - :attr:`specific_volume` as the reciprocal of :attr:`density`

    The mixture allows only one gas-like phase, and it must be modelled with at least
    1 component and 1 phase.

    Flash algorithms are built around the mixture management utilities of this class.

    Important:
        Phases are re-ordered once passed as arguments according to the following rules:

        - If more than 1 phase, the first, non-gas-like phase is treated as the
          reference phase per default.
        - The single gas-like phase is always the last one.
        - The first component is set as reference component per default.

    Parameters:
        components: A list of components to be added to the mixture.
            These are the chemical species which can appear in multiple phases.
        phases: A list of phases to be modelled.

    Raises:
        CompositionalModellingError: If the model assumptions are violated.

            - at most 1 gas phase must be modelled.
            - At least 1 component must be present.
            - At least 1 phase must be modelled.
            - Any phase has no components in it.
        CompositionalModellingError: If there is 1 component, which is not in any phase.
        ValueError: If any two components or phases have the same name
            (storage conflicts), or any two components have the same CASr number.

    """

    def __init__(
        self,
        components: list[Component],
        phases: list[Phase],
    ) -> None:

        self._ref_phase_index: int = 0
        """See :meth:`reference_phase_index`."""
        self._ref_component_index: int = 0
        """See :meth:`reference_component_index`."""
        self._components: list[Component] = []
        """A list of components passed at instantiation."""
        self._phases: list[Phase] = []
        """A list of phases passed at instantiation."""

        # a container holding names already added, to avoid storage conflicts
        double_names: list[str] = []
        double_casr: list[str] = []
        # Lists of gas-like and other phases
        gaslike_phases: list[Phase] = []
        other_phases: list[Phase] = []

        for comp in components:
            double_names.append(comp.name)
            double_casr.append(comp.CASr_number)
            self._components.append(comp)

        if len(set(double_casr)) < len(self._components):
            raise ValueError("CASr numbers must be unique per component.")

        for phase in phases:
            double_names.append(phase.name)
            if phase.state == PhysicalState.gas:
                gaslike_phases.append(phase)
            else:
                other_phases.append(phase)

            if phase.num_components == 0:
                raise CompositionalModellingError(
                    f"Phase {phase.name} has no components assigned."
                )

        self._phases = other_phases + gaslike_phases

        if len(set(double_names)) < len(self._phases) + len(self._components):
            raise ValueError("Phases and components must have unique names each.")

        # checking model assumptions
        if len(self._components) == 0:
            raise CompositionalModellingError("At least 1 component required")
        if len(self._phases) == 0:
            raise CompositionalModellingError("At least 1 phase required.")
        if len(gaslike_phases) > 1:
            raise CompositionalModellingError("At most 1 gas-like phase is permitted.")

        # Checking no dangling components
        for comp in self._components:
            its_phases = []
            for phase in self._phases:
                if comp in phase.components:
                    its_phases.append(phase)
            if len(its_phases) == 0:
                raise CompositionalModellingError(
                    f"Component {comp.name} not in any phase."
                )

        # NOTE by logic, length of gas-like phases can only be 1 at this point
        self._has_gas: bool = True if len(gaslike_phases) == 1 else False
        """Flag indicating if a gas-like phase is present."""
        ### PUBLIC

        self.density: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        """Density of the fluid mixture.

        Its thermodynamically consistent representation is
        :math:`\\sum_j s_j \\rho_j`, with :math:`s_j` being :attr:`Phase.saturation`
        and :math:`\\rho_j` being :attr:`Phase.density`

        """

        self.specific_volume: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        """The specific volume of the fluid mixture as the reciprocal of
        :attr:`density`."""

        self.specific_enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
        """Specific enthalpy of the fluid mixture.

        Its thermodynamically consistent representation is
        :math:`\\sum_j y_j h_j`, with :math:`y_j` being :attr:`Phase.fraction` and
        :math:`h_j` being :attr:`Phase.specific_enthalpy`

        """

    def __str__(self) -> str:
        """
        Returns:
            A string representation of the composition, with information about present
            components and phases.

        """
        out = f"Fluid mixture with {self.num_components} components:"
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
    def components(self) -> Generator[Component, None, None]:
        """
        Yields:
            Components in this fluid mixture.
        """
        for C in self._components:
            yield C

    @property
    def phases(self) -> Generator[Phase, None, None]:
        """
        Yields:
            Phases modelled in this fluid mixture.
        """
        for P in self._phases:
            yield P

    @property
    def gas_phase_index(self) -> int | None:
        """Returns the index of the gas-like phase in :meth:`phases`.

        Only 1 gas-like phase is supported and as of now it is always the last one
        in :meth:`phases`, if present.

        Note:
            The return value can be used as a boolean check whether gas is modelled or
            not.

        Returns:
            The index of the gas-like phase (:meth:`num_phases` - 1), if gas is
            modelled. Returns None otherwise.

        """
        if self._has_gas:
            # NOTE for safety reasons, we avoid using -1 for the last index to be
            # compatible with non-pythonic indexation (numba)
            return self.num_phases - 1
        else:
            return None

    @property
    def reference_phase_index(self) -> int:
        """Returns the index of the phase in :meth:`phases`, which is designated
        as the reference phase.

        By default, the first phase (0) is designated as the reference phase, which
        coincides with the first liquid-like phase (if any).

        Important:
            Changing the index of the reference phase changes also which phase
            :attr:`~Phase.fraction` and :attr:`~Phase.saturation` are eliminated by
            unity.

            Depending on the simulation set-up, this has numerical implications.

        Parameters:
            index: A new index to be assigned.

        Raises:
            IndexError: If index is out of range of :meth:`phases`.

        Returns:
            The index of the current phase designated as the reference phase.

        """
        return self._ref_phase_index

    @reference_phase_index.setter
    def reference_phase_index(self, index: int) -> None:
        max_index = len(self._phases) - 1
        if index < 0 or index > max_index:
            raise IndexError(f"Phase index {index} out of range [0, {max_index}].")
        self._ref_phase_index = int(index)

    @property
    def reference_component_index(self) -> int:
        """Returns the index of the component in :meth:`components`, which is designated
        as the reference component.

        By default, the first component (0) is designated as the reference component.

        Important:
            Changing the index of the reference component changes which mass
            conservation equations are eliminated by unity.

        Parameters:
            index: A new index to be assigned.

        Raises:
            IndexError: If index is out of range of :meth:`components`.

        Returns:
            The index of the current component designated as the reference component.

        """
        return self._ref_component_index

    @reference_component_index.setter
    def reference_component_index(self, index: int) -> None:
        max_index = len(self._components) - 1
        if index < 0 or index > max_index:
            raise IndexError(f"Component index {index} out of range [0, {max_index}].")
        self._ref_component_index = int(index)

    @property
    def reference_phase(self) -> Phase:
        """Returns the reference phase as designated by :meth:`reference_phase_index`."""
        return self._phases[self.reference_phase_index]

    @property
    def reference_component(self) -> Component:
        """Returns the reference component as designated by
        :meth:`reference_component_index`."""
        return self._components[self.reference_component_index]
