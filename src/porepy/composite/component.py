"""Contains the abstract base class for all components (species/ pure substances)
used in this framework.
"""

from __future__ import annotations

import abc
from typing import Generator

import porepy as pp

from ._composite_utils import VARIABLE_SYMBOLS, CompositionalSingleton


class PseudoComponent(abc.ABC, metaclass=CompositionalSingleton):
    """Abstract base class for all component-like instances inside a mixture.

    Pseudo-components are defined by a name and a molar mass.

    They are used as a starting point for all genuine components (species inside a mixture
    which change phases), but also as simplified surrogate models (parts of a compound),
    which sufficiently enough approximate reality.

    Example:
        A surrogate model would be salt, which influences the phase
        behavior of a compound 'brine' with its concentration, but for which it is **not**
        meaningful to model salt alone as a component in vapor and liquid phase.

        The alternative to the surrogate model is the rigorous model, where salt is split into
        the genuine components Sodium and Chlorine,
        which switch phases (dissolution, precipitation and vaporization)

    Note:
        When pairing the equilibrium problem with a flow problem, the pseudo-component might
        need nevertheless a transport equation! Take above example. If the amount of salt
        entering the system is unequal to the amount leaving it, one needs to formulate the
        respective transport. If that is not the case, i.e. the amount of salt is constant
        at all times, this is not necessary.

        Keep this in mind when formulating a model, its equations and variables.
        See :class:`Compound` for more information.

    Parameters:
        ad_system: AD system in which this pseud-component is present.

    """

    def __init__(self, ad_system: pp.ad.ADSystem) -> None:

        super().__init__()

        self.ad_system: pp.ad.ADSystem = ad_system
        """The AD system passed at instantiation."""

    @property
    def name(self) -> str:
        """
        Returns:
            name of the class, used as a unique identifier.

        """
        return str(self.__class__.__name__)

    @staticmethod
    @abc.abstractmethod
    def molar_mass() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kg / mol]

        Returns:
            molar mass.

        """
        pass


class Component(PseudoComponent):
    """Abstract base class for components modelled inside a mixture.

    Provides a variable representing the molar fraction of this component,
    cell-wise in a computational domain.

    Components are chemical species which possibly go through phase transitions and appear in
    multiple phases. They represent a genuine component in the flash problem.

    Note:
        The component is a Singleton per AD system,
        using the class name as a unique identifier.
        A component class with name ``X`` can only be present once in a system.
        Ambiguities must be avoided due to central storage of the fractional values in the
        grid data dictionaries.

    Parameters:
        ad_system: AD system in which this component is present cell-wise in each subdomain.

    """

    def __init__(self, ad_system: pp.ad.ADSystem) -> None:

        super().__init__(ad_system=ad_system)

        # creating the overall molar fraction variable
        self._fraction: pp.ad.MergedVariable = ad_system.create_variable(
            self.fraction_name
        )

    @property
    def fraction_name(self) -> str:
        """Name of the feed fraction variable, given by the general symbol and :meth:`name`."""
        return f"{VARIABLE_SYMBOLS['component_fraction']}_{self.name}"

    @property
    def fraction(self) -> pp.ad.MergedVariable:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        Returns:
            feed fraction, a primary variable on the whole domain (cell-wise).
            Indicates how many of the total moles belong to this component.

        """
        return self._fraction


class Compound(Component):
    """Abstract base class for all compounds in a mixture.

    A compound is a simplified, but meaningfully generalized component inside a mixture, for
    which it makes sense to treat it as a single component.

    It has one pseudo-component declared as the solvent, and arbitrary many other
    pseudo-components declared as solutes.

    A compound can appear in multiple phases and its thermodynamic properties are influenced
    by the presence of solutes.

    Note:
        Due to the generalization, the solvent and solutes are not considered as species which
        can transition into various phases, but rather as parameters in the flash problem.
        Only the compound as a whole splits into various phases and phase compositions are
        associated with the compound.

    This class provides variables representing fractions of solutes. The solute fractions are
    formulated with respect to the component fraction.
    Solute fractions are secondary variables in the flash problem, but primary (transportable)
    in the flow problem.

    Note:
        There is no variable representing the fraction of the solvent.
        The solvent fraction is always expressed by unity through the solute fractions.

    Example:
        See :class:`PseudoComponent`:

        Brine with a pseudo-components salt and water as solute and solvent, where it is
        sufficient to calculate how much brine is in vapor or liquid form,
        and the information about how the salt distributes across phases is irrelevant.
        The salt in this case is a **transportable** quantity,
        whose concentration acts as a parameter in the flash.

        Another example would be the black-oil model, where black-oil is treated as a compound
        with various hydrocarbons as pseudo-components.

    Parameters:
        ad_system: AD system in which this component is present cell-wise in each subdomain.
        solvent: A pseudo-component representing the solvent

    """

    def __init__(self, ad_system: pp.ad.ADSystem, solvent: PseudoComponent) -> None:

        super().__init__(ad_system=ad_system)

        self._solvent: PseudoComponent = solvent
        """The solvent passed at instantiation."""

        self._solutes: list[PseudoComponent] = list()
        """A list containing present solutes."""

        self._solute_fractions: dict[PseudoComponent, pp.ad.MergedVariable] = dict()
        """A dictionary containing the variables representing solute fractions for a given
        pseudo-component (key)."""

    @property
    def solvent(self) -> PseudoComponent:
        """The pseudo-component representing the solvent, passed at instantiation"""

    @property
    def solutes(self) -> Generator[PseudoComponent, None, None]:
        """
        Yields:
            Pseudo-components modelled as solutes in this compound.

        """
        for solute in self._solutes:
            yield solute

    def solute_fraction_name(self, solute: PseudoComponent) -> str:
        """
        Returns:
            the name of the solute fraction, composed of the general symbol and the solute
            name.

        """
        return f"{VARIABLE_SYMBOLS['solute_fraction']}_{solute.name}"

    def solution_fraction_of(self, pseudo_component: PseudoComponent) -> pp.ad.Operator:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        Parameters:
            pseudo_component: a pseudo-component present in this compound.

        Returns:
            A representation of the molar fraction of a present pseudo_component.

            Fractions of solutes are represented by their respective variable.
            The fraction of the solvent is represented by unity where

                ``fraction_solute = 1 - sum_solutes solute_fraction``

            Returns zero for any unknown pseudo-component.

        """

        if pseudo_component == self.solvent:
            # represent solvent fraction by unity
            fraction = pp.ad.Scalar(1.0)
            for solute in self.solutes:
                fraction -= self._solute_fractions[solute]

            return fraction

        elif pseudo_component in self.solutes:
            return self._solute_fractions[pseudo_component]
        else:
            return pp.ad.Scalar(0.0)
        
    def molality_of(self, solute: PseudoComponent) -> pp.ad.Operator | pp.ad.Scalar:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [mol / kg]

        The molality ``m`` of a pseudo-component is calculated with respect to the total
        number of moles in the mixture ``n``:

            ``n_compound = n * fraction_compound``
            ``m_solute = (n * fraction_compound * solute_fraction) ``
            ``/ (n * fraction_compound * solvent_fraction * molar_mass*solvent)``
            
            ``m_solute = solute_fraction / (solvent_fraction * molar_mass*solvent)``

        Note:
            The molality of the solvent is its molar mass.

            The molality of a solute not present in a compound is zero.

        Returns:
            An operator representing the molality of a solute, i.e. number of moles of solute
            per ``kg`` of solvent.

        """
        if solute == self.solvent:
            return pp.ad.Scalar(1. / self.solvent.molar_mass())
        elif solute in self.solutes:
            # get fractions of solute and solvent
            fraction_solute = self.solution_fraction_of(solute)
            fraction_solvent = self.solution_fraction_of(self.solvent)
            # apply above formula
            return fraction_solute / (fraction_solvent * self.solvent.molar_mass())
        else:
            return pp.ad.Scalar(0.)

    def add_solute(self, solutes: PseudoComponent | list[PseudoComponent]) -> None:
        """Adds a solute to the compound.

        This method introduces new variables into the system, the solute fraction.

        If a solute was already added, it is ignored.

        Parameters:
            solutes: one or multiple pseudo-components to be added to this compound.

        Raises:
            TypeError: if a user attempts to add a genuine component to a compound.

        """
        if isinstance(solutes, PseudoComponent):
            solutes = [solutes]  # type: ignore

        for solute in solutes:
            if solutes not in self.solutes:
                # sanity check if attempt to add a genuine components.
                if isinstance(solute, Component):
                    raise TypeError(
                        f"Cannot add genuine component '{solute.name}' to a compound."
                    )
                # create name of solute fraction and respective variable
                fraction_name = self.solute_fraction_name(solute)
                solute_fraction = self.ad_system.create_variable(fraction_name)

                # store fraction and solute
                self._solutes.append(solute)
                self._solute_fractions[solute] = solute_fraction
