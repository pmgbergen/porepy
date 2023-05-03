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
from typing import Generator

import numpy as np
from scipy import sparse as sps

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from ._core import COMPOSITIONAL_VARIABLE_SYMBOLS
from .chem_species import ChemicalSpeciesData, FluidSpeciesData

__all__ = [
    "Component",
    "Compound",
]


class Component(abc.ABC, FluidSpeciesData):
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
    def from_species(cls, species: FluidSpeciesData) -> Component:
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
        - Phys. Dimension:        [-]

        Parameters:
            p: The pressure of the mixture.
            T: The temperature of the mixture.

        Returns:
            Ideal specific enthalpy for given pressure and temperature.

        """
        pass


class Compound(Component):  # TODO fix molality to make it an ad.Function call
    """Abstract base class for compounds in a mixture.

    A compound is a simplified, but meaningfully generalized set of components inside a
    mixture, for which it makes sense to treat it as a single component.

    It is represented by one component declared as the solvent, and arbitrary many other
    species declared as solutes.

    A compound can appear in multiple phases and its thermodynamic properties are
    influenced by the presence of solutes.

    Note:
        Due to the generalization, the solvent and solutes alone are not considered as
        genuine components which can transition into various phases,
        but rather as parameters in the flash problem.
        Only the compound as a whole splits into various phases. Fractions in phases
        are associated with the compound.

    This class provides variables representing fractions of solutes.
    The solute fractions are formulated with respect to the component
    :attr:`~Component.fraction`.
    Solute fractions are secondary variables in the flash problem,
    but primary (transportable) in the flow problem.

    Note:
        There is no variable representing the fraction of the solvent.
        The solvent fraction is always expressed by unity through the solute fractions.

    Example:
        Brines with species salt and water as solute and solvent, where it is
        sufficient to calculate how much brine is in vapor or liquid form,
        and the information about how the salt distributes across phases is irrelevant.
        The salt in this case is a **transportable** quantity,
        whose concentration acts as a parameter in the flash.

        Another example would be the black-oil model, where black-oil is treated as a
        compound with various hydrocarbons as pseudo-components.

    Parameters:
        ad_system: AD system in which this component is present cell-wise in each
            subdomain.
        solvent: A pseudo-component representing the solvent.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._solutes: list[ChemicalSpeciesData] = list()
        """A list containing present solutes."""

        self._solute_fractions: dict[
            PseudoComponent, pp.ad.MixedDimensionalVariable
        ] = dict()
        """A dictionary containing the variables representing solute fractions for a
        given pseudo-component (key)."""

    @property
    def solvent(self) -> PseudoComponent:
        """The pseudo-component representing the solvent, passed at instantiation"""
        return self._solvent

    @property
    def solutes(self) -> Generator[PseudoComponent, None, None]:
        """
        Yields:
            Pseudo-components modelled as solutes in this compound.

        """
        for solute in self._solutes:
            yield solute

    @property
    def molar_mass(self) -> pp.ad.Operator:
        """The molar mass of a compound depends on how much of the solutes is available.

        It is a sum of the molar masses of present pseudo-components, weighed with their
        respective fraction.

        Returns:
            An operator representing the molar mass of the compound,
            depending on solute fractions.

        """
        M = self.solvent.molar_mass() * self.solution_fraction_of(self.solvent)

        for solute in self.solutes:
            M += solute.molar_mass() * self.solution_fraction_of(solute)

        return M

    def solute_fraction_name(self, solute: PseudoComponent) -> str:
        """
        Returns:
            The name of the solute fraction,
            composed of the general symbol and the solute name.

        """
        return f"{COMPOSITIONAL_VARIABLE_SYMBOLS['solute_fraction']}_{solute.name}"

    def solution_fraction_of(self, pseudo_component: PseudoComponent) -> pp.ad.Operator:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        Parameters:
            pseudo_component: A pseudo-component present in this compound.

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
        number of moles in the mixture ``n``

            ``n_compound = n * fraction_compound``,
            ``m_solute = (n * fraction_compound * solute_fraction) ``,
            ``/ (n * fraction_compound * solvent_fraction * molar_mass*solvent)``,
            ``m_solute = solute_fraction / (solvent_fraction * molar_mass*solvent)``.

        Note:
            The molality of the solvent is its molar mass.

            The molality of a solute not present in a compound is zero.

        Returns:
            An operator representing the molality of a solute,
            i.e. number of moles of solute per ``kg`` of solvent.

        """
        if solute == self.solvent:
            return pp.ad.Scalar(1.0 / self.solvent.molar_mass())
        elif solute in self.solutes:
            # get fractions of solute and solvent
            fraction_solute = self.solution_fraction_of(solute)
            fraction_solvent = self.solution_fraction_of(self.solvent)
            # apply above formula
            return fraction_solute / (fraction_solvent * self.solvent.molar_mass())
        else:
            return pp.ad.Scalar(0.0)

    def add_solute(self, solutes: PseudoComponent | list[PseudoComponent]) -> None:
        """Adds a solute to the compound.

        This method introduces new variables into the system, the solute fraction.

        If a solute was already added, it is ignored.

        Parameters:
            solutes: One or multiple pseudo-components to be added to this compound.

        Raises:
            TypeError: If a user attempts to add a genuine component to a compound.

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
                solute_fraction = self.ad_system.create_variables(
                    fraction_name, subdomains=self.ad_system.mdg.subdomains()
                )

                # store fraction and solute
                self._solutes.append(solute)
                self._solute_fractions[solute] = solute_fraction

    def set_solute_fractions(
        self,
        fractions: dict[PseudoComponent, float | np.ndarray],
        copy_to_state: bool = True,
    ) -> None:
        """Set the solute fractions per solute in this component.

        The fraction can either be given as an array with entries per cell,
        or as float for a homogenous distribution.

        Only fractions for solutes are to be passed,
        since the solvent fraction is represented by unity.

        Note:
            Per cell, the sum of fractions have to be in ``[0,1)``,
            where the right interval end is open on purpose.
            This means the solvent **has** to be always present.
            As of now, the chemistry when the solvent disappears is not considered.

        Parameters:
            fractions: A dictionary containing per solute (key) the respective fraction.
            copy_to_state: ``default=True``

                Copies the values to the STATE of the AD variable,
                additionally to ITERATE. Defaults to True.

        Raises:
            ValueError: If

                - any value breaches above restriction,
                - the cell-wise sum is greater or equal to 1,
                - values are missing for a present solute.

            AssertionError: If array-like fractions don't have ``num_cells`` values.

        """

        nc = self.ad_system.mdg.num_subdomain_cells()
        # sum of fractions for validity check
        fraction_sum = np.zeros(nc)

        # loop over present solutes to ensure we have fractions for every solute.
        for solute in self.solutes:

            # check if fractions are passed for present solute
            if solute not in fractions:
                raise ValueError(f"Missing fraction for solute {solute.name}.")

            frac_val = fractions[solute]
            # convert to array
            if not isinstance(frac_val, np.ndarray):
                frac_val = np.ones(nc) * frac_val

            # assert enough values are present
            assert (
                len(frac_val) == nc
            ), f"Array values for solute {solute.name} do not cover the whole domain."
            # check validity of fraction values
            if np.any(frac_val < 0.0) or np.any(frac_val >= 1.0):
                raise ValueError(f"Values for solute {solute.name} not in [0,1).")

            # sum values
            fraction_sum += frac_val

            # set values for fraction
            frac_name = self.solute_fraction_name(solute)
            self.ad_system.set_variable_values(
                frac_val,
                variables=[frac_name],
                to_iterate=True,
                to_state=copy_to_state,
            )

        # last validity check to ensure the solvent is present everywhere
        if np.any(fraction_sum >= 1.0):
            raise ValueError("Sum of solute fractions is >= 1.")

    def set_solute_fractions_with_molality(
        self, molalities: dict[PseudoComponent, np.ndarray | float]
    ) -> None:
        """Uses the formula given in :meth:`molality_of` to set solute fractions using
        values for molality per solute.

        After computing fractions based on given molalities,
        :meth:`set_solute_fractions` is called to set the fraction,
        i.e. the the same restrictions apply and errors will be raised if molality
        values violate the restrictions on respective fractions.

        Analogously to fractions, molal values can be set cell-wise in an array or
        homogeneously using numbers.

        It holds for every solute ``c``:

            ``molality_c * (1 - sum_i fraction_i) * molar_M_solvent = fraction_c``,
            ``molality_c * molar_M_solvent =``
            ``(1 + molality_c * molar_M_solvent)  * fraction_c``
            ``+ molality_c * molar_M_solvent * (sum_{i != c} fraction_i``.

        Parameters:
            molalities: A dictionary containing per solute (key) the respective molality
                values.

        Raises:
            ValueError: If the molality for a present solute is missing.

        """

        nc = self.ad_system.mdg.num_subdomain_cells()

        # data structure to set resulting fractions
        fractions: dict[PseudoComponent, np.ndarray] = dict()
        # rhs and matrix for linear system to convert to fractions
        rhs_: list[np.ndarray] = list()
        A_: list[sps.spmatrix] = list()
        # column slicing to solute fractions
        fraction_names = [self.solute_fraction_name(solute) for solute in self.solutes]
        projection = self.ad_system.projection_to(fraction_names)

        # loop over present solutes and construct row-blocks of linear system for
        # conversion
        for solute in self.solutes:
            if solute not in molalities:
                raise ValueError(f"Missing molality for solute {solute.name}.")

            molality = molalities[solute]
            # convert to array
            if not isinstance(molality, np.ndarray):
                molality = np.ones(nc) * molality

            rhs_.append(self.solvent.molar_mass() * molality)

            A_block = (1 + rhs_[-1]) * self.solution_fraction_of(solute)
            for other_solute in self.solutes:
                if other_solute != solute:
                    A_block += rhs_[-1] * self.solution_fraction_of(other_solute)

            A_block = A_block.evaluate(self.ad_system).jac * projection.transpose()
            A_.append(A_block)

        # compute fractions by solving the linear system
        A = sps.vstack(A_, format="csr")
        rhs = np.concatenate(rhs_)
        fractions = sps.linalg.spsolve(A, rhs)

        # extract computed fractions and assemble dictionary for setting fractions
        for i, solute in enumerate(self.solutes):
            idx_start = i * nc
            idx_end = (i + 1) * nc
            solute_fraction = fractions[idx_start:idx_end]
            fractions[solute] = solute_fraction

        # set computed fractions
        self.set_solute_fractions(fractions)
