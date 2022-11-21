"""Contains the abstract base class for all components (species/ pure substances)
used in this framework.
"""

from __future__ import annotations

import abc

import porepy as pp

from ._composite_utils import CompositionalSingleton


class Component(abc.ABC, metaclass=CompositionalSingleton):
    """Abstract base class for chemical components modelled inside a mixture.

    Provides and manages component-related physical quantities and properties.

    Components are chemical species which possibly go through phase transitions and appear in
    multiple phases.

    The component is a Singleton per AD system, using the class name as a unique identifier.
    A component class with name ``X`` can only be present once in a system.
    Ambiguities must be avoided due to central storage of the fractional values in the
    grid data dictionaries.

    Parameters:
        ad_system: AD system in which this component is present cell-wise in each subdomain.

    """

    def __init__(self, ad_system: pp.ad.ADSystem) -> None:

        super().__init__()

        ### PUBLIC

        self.ad_system: pp.ad.ADSystem = ad_system
        """The AD system passed at instantiation."""

        #### PRIVATE

        # creating the overall molar fraction variable
        self._fraction: pp.ad.MergedVariable = ad_system.create_variable(
            self.fraction_name
        )

    @property
    def name(self) -> str:
        """
        Returns: name of the class, used as a unique identifier.

        """
        return str(self.__class__.__name__)

    @property
    def fraction_name(self) -> str:
        """Name of the feed fraction variable, given by the general symbol and :meth:`name`."""
        return "z" + "_" + self.name

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

    @staticmethod
    @abc.abstractmethod
    def molar_mass() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kg / mol]

        Returns: molar mass.

        """
        pass


class Compound(Component):
    """Abstract base class for all compounds in a mixture.

    A compound is a simplified, but meaningfully generalized component inside a mixture, for
    which it makes sense to treat it as a single component.

    A compound can appear in multiple phases and it can have multiple pseudo-components, whose
    presence influence the thermodynamic behavior.

    An example would be brine with a pseudo-component representing salt, where it is
    sufficient to calculate how much brine is in vapor or liquid form,
    and the information about how the salt distributes across phases is irrelevant.
    The salt in this case is a **transportable** quantity,
    whose concentration acts as a parameter in the flash.

    Another example would be the black-oil model, where black-oil is treated as a component
    with various hydrocarbons as pseudo-components.

    Parameters:
        ad_system: AD system in which this component is present cell-wise in each subdomain.

    """
    pass


class PseudoComponent(abc.ABC, metaclass=CompositionalSingleton):
    """Abstract base class for components inside a mixture, which influence the properties of
    the mixture, but are more similar to parameters than to actual unknowns for the flash.

    Pseudo-components can be used to provide simplified surrogate models, which sufficiently
    enough approximate reality.

    An example would be salt, which influences the phase
    behavior of a compound 'brine' with its concentration, but for which it is not meaningful
    to model salt alone as a component in vapor and liquid phase.

    Note:
        When pairing the equilibrium problem with a flow problem, the pseudo-component might
        need nevertheless a transport equation! Take above example. If the amount of salt
        entering the system is unequal to the amount leaving it, one needs to formulate the
        respective transport. If that is not the case, i.e. the amount of salt is constant
        at all times, this is not necessary.

        Keep this in mind when formulating a model, its equations and variables.
    
    Note:
        For above reasons, the pseudo-component does not inherit from component.
        Its mathematical implications are fundamentally different.

    Parameters:
        ad_system: AD system in which this pseud-component is present,
            cell-wise in each subdomain.

    """
    pass
