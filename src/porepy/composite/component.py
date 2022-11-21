"""Contains the abstract base class for all components (species/ pure substances)
used in this framework.
"""

from __future__ import annotations

import abc

import porepy as pp


class Component(abc.ABC):
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

    # contains per mdg the singleton, using the class name as a unique identifier
    __ad_singletons: dict[pp.MixedDimensionalGrid, dict[str, Component]] = dict()
    # flag if a singleton has recently been accessed, to skip re-instantiation
    __singleton_accessed: bool = False

    def __new__(cls, ad_system: pp.ad.ADSystem) -> Component:
        # class name is used as unique identifier
        name = str(cls.__name__)
        # check for AD singletons per grid
        mdg = ad_system.dof_manager.mdg
        if mdg in Component.__ad_singletons:
            if name in Component.__ad_singletons[mdg]:
                # flag that the singleton has been accessed and return it.
                Component.__singleton_accessed = True
                return Component.__ad_singletons[mdg][name]
        else:
            Component.__ad_singletons.update({mdg: dict()})

        # create a new instance and store it, if no previous instantiations were found
        new_instance = super().__new__(cls)
        Component.__ad_singletons[mdg].update({name: new_instance})

        return new_instance

    def __init__(self, ad_system: pp.ad.ADSystem) -> None:

        # skipping re-instantiation if class if __new__ returned the previous reference
        if Component.__singleton_accessed:
            Component.__singleton_accessed = False
            return

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


class PseudoComponent(abc.ABC):
    """Abstract base class for components inside a mixture, which influence the properties of
    the mixture, but more more similar to parameters, then to actual unknowns.
    
    Pseudo-components can be used to provide simplified surrogate models, which sufficiently
    enough approximate the reality. An example would be salt, which influences the phase
    behavior of a compound 'brine' with its concentration, but for which it is not meaningful
    to model salt alone as a component in vapor and liquid phase.
    
    Parameters:
        ad_system: AD system in which this pseud-component is present,
            cell-wise in each subdomain.

    """
    pass


class Compound(abc.ABC):
    """Abstract base class for all compounds in a mixture.
    
    A compound is a simplified, but meaningful generalized component inside a mixture, for
    which it is meaningful to treat it as a component.
    
    A compound can appear in multiple phases and it can have multiple pseudo-components, whose
    presence influence the thermodynamic behavior.
    
    An example would be brine with a pseudo-component representing a salt, where it is
    sufficient to calculate how much brine is vapor or liquid form, but the information about
    how the salt distributes across phases is irrelevant for a model. The salt in this case is
    a transportable quantity, whose concentration acts as a parameter in the flash.

    Parameters:
        ad_system: AD system in which this component is present cell-wise in each subdomain.

    """