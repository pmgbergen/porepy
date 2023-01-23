"""This module contains the abstract base class for all components
(species/ pure substances) used in this framework.

Components are models for chemical species inside a mixture.
They are either genuine components (with relevant fractional variables) or
pseudo-components, which act as parameters in compounds.

"""

from __future__ import annotations

import abc
from typing import Generator

import numpy as np
from scipy import sparse as sps

import porepy as pp

from ._composite_utils import VARIABLE_SYMBOLS, CompositionalSingleton


class PseudoComponent(abc.ABC, metaclass=CompositionalSingleton):
    """Abstract base class for instances inside a mixture, which represent a chemical
    species.

    Pseudo-components are identified by their name and have a molar mass,
    critical pressure and critical temperature.

    They are used as a starting point for all genuine components
    (species inside a mixture which change phases),
    but also as simplified surrogate models (parts of a compound),
    which sufficiently enough approximate reality.

    Example:
        A surrogate model would be salt, which influences the phase
        behavior of a compound 'brine' with its concentration,
        but for which it is **not** meaningful to model salt alone as a component
        in vapor and liquid phase.

        The alternative to the surrogate model is the rigorous model,
        where salt is split into the genuine components Sodium and Chlorine,
        which switch phases (dissolution, precipitation and vaporization)

    Note:
        When pairing the equilibrium problem with a flow problem,
        the pseudo-component might need nevertheless a transport equation!
        Take above example.
        If the amount of salt entering the system is unequal to the amount leaving it,
        one needs to formulate the respective transport.
        If that is not the case, i.e. the amount of salt is constant at all times,
        this is not necessary.

        Keep this in mind when formulating a model, its equations and variables.
        See :class:`Compound` for more information.

    Parameters:
        ad_system: AD system in which this pseudo-component is present.

    """

    def __init__(self, ad_system: pp.ad.EquationSystem) -> None:

        super().__init__()

        self.ad_system: pp.ad.EquationSystem = ad_system
        """The AD system passed at instantiation."""

    @property
    def name(self) -> str:
        """
        Returns:
            Name of the class, used as a unique identifier in the composite framework.

        """
        return str(self.__class__.__name__)

    @staticmethod
    @abc.abstractmethod
    def molar_mass() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kg / mol]

        Returns:
            Molar mass.

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def critical_pressure() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [MPa]

        Returns:
            Critical pressure of this pseudo-component.

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def critical_temperature() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [K]

        Returns:
            Critical temperature of this pseudo-component.

        """
        pass


class Component(PseudoComponent):
    """Abstract base class for components modelled inside a mixture.

    Provides a variable representing the molar fraction of this component
    (feed fraction), cell-wise in a computational domain.

    Components are chemical species which possibly go through phase transitions and
    appear in multiple phases.
    They represent a genuine component in the flash problem.

    Note:
        The component is a Singleton per AD system,
        using the class name as a unique identifier.
        A component class with name ``X`` can only be present once in a system.
        Ambiguities must be avoided due to central storage of the fractional values
        in the grid data dictionaries.

    Parameters:
        ad_system: AD system in which this component is present cell-wise in each
            subdomain.

    """

    def __init__(self, ad_system: pp.ad.EquationSystem) -> None:

        super().__init__(ad_system=ad_system)

        # creating the overall molar fraction variable
        self._fraction: pp.ad.MixedDimensionalVariable = ad_system.create_variables(
            self.fraction_name, subdomains=ad_system.mdg.subdomains()
        )

    @property
    def fraction_name(self) -> str:
        """Name of the feed fraction variable,
        given by the general symbol and :meth:`name`."""
        return f"{VARIABLE_SYMBOLS['component_fraction']}_{self.name}"

    @property
    def fraction(self) -> pp.ad.MixedDimensionalVariable:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [%] fractional

        Returns:
            Feed fraction of this component,
            a primary variable on the whole domain (cell-wise).
            Indicates how many of the total moles belong to this component.

        """
        return self._fraction


class Compound(Component):
    """Abstract base class for all compounds in a mixture.

    A compound is a simplified, but meaningfully generalized component inside a mixture,
    for which it makes sense to treat it as a single component.

    It has one pseudo-component declared as the solvent, and arbitrary many other
    pseudo-components declared as solutes.

    A compound can appear in multiple phases and its thermodynamic properties are
    influenced by the presence of solutes.

    Note:
        Due to the generalization,
        the solvent and solutes are not considered as genuine components which
        can transition into various phases,
        but rather as parameters in the flash problem.
        Only the compound as a whole splits into various phases and fractions in phases
        are associated with the compound.

    This class provides variables representing fractions of solutes.
    The solute fractions are formulated with respect to the component fraction.
    Solute fractions are secondary variables in the flash problem,
    but primary (transportable) in the flow problem.

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

        Another example would be the black-oil model, where black-oil is treated as a
        compound with various hydrocarbons as pseudo-components.

    Parameters:
        ad_system: AD system in which this component is present cell-wise in each
            subdomain.
        solvent: A pseudo-component representing the solvent.

    """

    def __init__(
        self, ad_system: pp.ad.EquationSystem, solvent: PseudoComponent
    ) -> None:

        super().__init__(ad_system=ad_system)

        self._solvent: PseudoComponent = solvent
        """The solvent passed at instantiation."""

        self._solutes: list[PseudoComponent] = list()
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
        return f"{VARIABLE_SYMBOLS['solute_fraction']}_{solute.name}"

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
