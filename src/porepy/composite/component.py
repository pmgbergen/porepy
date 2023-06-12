"""This module contains the abstract base class for all components
(species/ pure substances) used in this framework.

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

"""

from __future__ import annotations

import abc
from dataclasses import asdict
from typing import Literal, overload

import numpy as np
import porepy as pp

from porepy.numerics.ad.operator_functions import NumericType

from ._core import R_IDEAL
from .chem_species import ChemicalSpecies, FluidSpecies
from .composite_utils import safe_sum

__all__ = [
    "Component",
    "Compound",
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
    def h_ideal(self, p: NumericType, T: NumericType) -> NumericType:
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

    def u_ideal(self, p: NumericType, T: NumericType) -> NumericType:
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
