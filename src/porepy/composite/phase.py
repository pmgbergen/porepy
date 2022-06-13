""" Contains the abstract base class for all phases."""
from __future__ import annotations

import abc
import warnings
from typing import Dict, Generator, List, Union

import numpy as np

import porepy as pp

from ._composite_utils import COMPUTATIONAL_VARIABLES, create_merged_variable

# from weakref import WeakValueDictionary, WeakKeyDictionary

__all__ = ["PhaseField"]


class PhaseField(abc.ABC):
    """
    Base class for all phases.

    The name PhaseFIELD was chosen on purpose to avoid ambiguities when talking about general
    phases or states of matter.
    Both of the latter ones are mathematically identified by the (time-dependent) region they
    occupy and a respective general velocity field (or flux) in that region.
    Those are expressed by the 'PhaseField', the basis for both interpretations
    of the term phase.

    The phase related to a state of matter is characterized by a dominating substance and
    can be treated less abstract in a child class.

    In order to create a general phase, like e.g. a Black-Oil-Model,
    use this abstract base class.

    Currently, the modeler has to ANTICIPATE the appearance of a substance in a phase or
    state of matter.
    There is no dynamic creation of phases/states or a dynamic appearance of
    a substance in a phase

    NOTE proposition:
    Turns every child in a conditional singleton class: It can only be instantiated once per
    computational domain.
    Any further instantiation will return a reference to the same, first class.
    Assures uniqueness and integrity of variable values (like concentrations) in
    grid-specific dictionaries.
    Currently this is only indirectly given due to how the access to the variables is
    implemented (indirectly using CD)

    Functionality:
        - handles anticipated substances
        - instantiates and provides direct access to phase-related AD-variables.
        - serves as a "container" object for anticipated substances.

    PhaseField-related variables:
        - saturation
        - phase molar fraction
        - molar fraction in phase (per present component)
    """

    """ For a grid bucket (keys), contains a list of present phases (values). """
    __phase_instances: Dict[pp.GridBucket, list] = dict()

    def __new__(cls, name: str, gb: pp.GridBucket) -> PhaseField:
        """
        Declarator assures the phase name is unique for a given computational domain.
        Ambiguities must be avoided due to the central storage of the AD variables and usage
        of the name as a key.
        """
        name = str(name)
        if gb in PhaseField.__phase_instances.keys():
            if name in PhaseField.__phase_instances[gb]:
                raise RuntimeError(
                    "Phase with name '" + name + "' already present in\n" + str(gb)
                )
        else:
            PhaseField.__phase_instances.update({gb: list()})

        PhaseField.__phase_instances[gb].append(name)
        return super().__new__(cls)

    def __init__(self, name: str, gb: pp.GridBucket) -> None:
        """Base class constructor. Initiates phase-related AD-variables.

        :param name: name of the phase
        :type name: str
        :param gb: geometry in which the phase field is modelled
        :type gb: :class:`~porepy.GridBucket`
        """
        super().__init__()

        ## PUBLIC
        self.gb: pp.GridBucket = gb

        ## PRIVATE
        self._name = str(name)
        self._present_substances: List[pp.composite.Substance] = list()

        # Instantiate saturation variable
        self._saturation = create_merged_variable(gb, {"cells": 1}, self.saturation_var)
        # Instantiate phase molar fraction variable
        self._molar_fraction = create_merged_variable(
            self.gb, {"cells": 1}, self.molar_fraction_var
        )

    def __iter__(self) -> Generator[pp.composite.Substance, None, None]:
        """
        Iterator over substances present in this phase field instance.

        IMPORTANT:
        The order from this iterator is used for choosing e.g. the values in a
        list of 'numpy.array' when setting initial values.
        Use the order returned here every time you deal with substance-related values
        for substances in this phase.

        :return: yields present substance
        :rtype: :class:`~porepy.composite.substance.Substance`
        """
        for substance in self._present_substances:
            yield substance

    @property
    def name(self) -> str:
        """
        The name is used to construct names for AD variables and keys to store them.

        :return: name of the phase field.
        :rtype: str
        """
        return self._name

    @property
    def saturation(self) -> pp.ad.MergedVariable:
        """
        As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`~porepy.ad.MergedVariable` representing the
            saturation of this phase field.
        :rtype: :class:`~porepy.ad.MergedVariable`
        """
        return self._saturation

    @property
    def saturation_var(self) -> str:
        """
        The name is used as name for the MergedVariable anf key for data in grid dictionaries.

        :return: name of the saturation variable
        :rtype: str
        """
        return COMPUTATIONAL_VARIABLES["saturation"] + "_" + self.name

    @property
    def molar_fraction(self) -> pp.ad.MergedVariable:
        """
        As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the
            molar fraction of this phase
        :rtype: :class:`~porepy.ad.MergedVariable`
        """
        return self._molar_fraction

    @property
    def molar_fraction_var(self) -> str:
        """
        The name is used as name for the MergedVariable anf key for data in grid dictionaries.

        :return: name of the phase molar fraction variable
        :rtype: str
        """
        return COMPUTATIONAL_VARIABLES["phase_molar_fraction"] + "_" + self.name

    def add_substance(
        self,
        substances: Union[List[pp.composite.Substance], pp.composite.Substance],
    ) -> None:
        """Adds substances which are expected by the model in this phase.

        The Substances must be instantiated on the same :class:`~porepy.GridBucket`
        used for the phasefield.

        :param substances: substance or list of substances to be added to this phase
        :type substances: :class:`~porepy.composite.substance.Substance`
        """

        if isinstance(substances, pp.composite.Substance):
            substances = [substances]

        for subst in substances:
            # check if objects are meant for same domain
            if subst.gb != self.gb:
                raise ValueError(
                    "Substance '%s' instantiated using unknown grid bucket."
                    % (subst.name)
                )

            # skip already present substances:
            if subst.name in [ps.name for ps in self._present_substances]:
                warnings.warn(
                    "Substance '%s' has already been added to phase '%s'. Skipping..."
                    % (subst.name, self.name)
                )
                continue

            # store reference to present substance
            self._present_substances.append(subst)

            # instantiate molar fraction in phase variable
            subst.fraction_in_phase(self.name)

    def set_initial_fractions(
        self, fractions: Union[List[List[np.ndarray]], List[List[float]]]
    ) -> None:
        """
        Sets the (initial) molar fractions for present substances.

        This methods assumes the order of the gridbuckets iterator to be used for the argument
        'fractions', containing the initial values per subdomain per substance.
        The initial values themselves have to be given as numpy.array or floats
        per subdomain per substance.

        in detail, the input must have the following, nested structure:
            - the top list ('fractions') contains lists per grid in grid bucket
            - the lists per grid contain arrays per substance in this phase

        For homogenous fraction values per grid per substance, pass a float.
        For heterogeneous fraction values per grid per substance, use numpy.array with length
        equal to number of cells, per grid per substance

        Finally, this methods asserts the initial unitarity of the values per cell.

        :param fractions: list of lists of arrays or floats,
            containing the fractions per component per grid,
            in the order of the gridbuckets iterator and this class' iterator.
        :param fractions: list
        """
        # loop over top list: fractions per grid in gridbucket
        for grid_data, vals_per_grid in zip(self.gb, fractions):

            grid = grid_data[0]
            data = grid_data[1]

            sum_per_grid = np.zeros(grid.num_cells)

            # loop over next level list: fractions per substance (per grid)
            for subst, vals in zip(self, vals_per_grid):

                # convert homogenous fractions to values per cell
                if isinstance(vals, float):
                    vals = np.ones(grid.num_cells) * vals

                # this throws an error if the dimensions should mismatch when giving
                # fractions for a grid in array form
                sum_per_grid += vals

                # check if data dictionary has necessary keys, if not, create them.
                # (normally they should NOT be missing)
                # TODO check if we should also prepare the 'previous_timestep' values here
                if pp.STATE not in data:
                    data[pp.STATE] = {}
                if pp.ITERATE not in data[pp.STATE]:
                    data[pp.STATE][pp.ITERATE] = {}

                fraction_name = subst.fraction_in_phase_var(self.name)
                data[pp.STATE][fraction_name] = np.copy(vals)
                data[pp.STATE][pp.ITERATE][fraction_name] = np.copy(vals)

            # assert the fractional character (sum equals 1) in each cell
            # if not np.allclose(sum_per_grid, 1.):
            if np.any(sum_per_grid != 1.0):  # TODO check sensitivity
                raise ValueError(
                    "Initial fractions do not sum up to 1.0 on each cell on grid:\n"
                    + str(grid)
                )

    def mass_density(
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        """
        Uses the  molar mass values in combination with the molar fractions in this phase
        to compute the mass density of the phase.

        rho_e_mass = sum_s density_e_molar() * chi_s_e * M_s

        This holds since
            density_e_molar() * chi_s_e
        give the number of poles of the substance present in this phase and
        the molar mass M_s converts the physical dimension.

        Math. Dimension:        scalar
        Phys. Dimension:        [kg / m^3]

        :return: mass density of the phase
        (dependent on thermodynamic state and the composition)
        :rtype: :class:`~porepy.ad.operators.Operator`
        """

        weight = pp.ad.Scalar(0.0, "mass-density-%s" % (self.name))
        # if there are no substances in this phase, return a zero
        if not self._present_substances:
            return weight

        # add the mass-weighted fraction for each present substance.
        for substance in self._present_substances:
            weight += substance.molar_mass() * substance.fraction_in_phase(self.name)

        # Multiply the mass weight with the molar density and return the operator
        return weight * self.molar_density(pressure, temperature)

    # ------------------------------------------------------------------------------
    ### Abstract, phase-related physical properties
    # ------------------------------------------------------------------------------

    @abc.abstractmethod
    def molar_density(
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        """
        Abstract physical property, dependent on thermodynamic state and the composition.
        The composition variables (molar fractions of present substances) can be accessed
        by reference.

        Math. Dimension:        scalar
        Phys. Dimension:        [mol / m^3]

        :param pressure: global pressure variable
        :type pressure: :class:`~porepy.ad.MergedVariable`
        :param temperature: global temperature variable
        :type temperature: :class:`~porepy.ad.MergedVariable`

        :return: molar density of the phase
        :rtype: :class:`~porepy.ad.Operator`
        """
        pass

    @abc.abstractmethod
    def specific_molar_enthalpy(
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        """
        Abstract physical quantity, dependent on thermodynamic state and the composition.
        The composition variables (molar fractions of present substances) can be accessed
        by reference.

        Math. Dimension:        scalar
        Phys.Dimension:         [kJ / mol / K]

        :param pressure: global pressure variable
        :type pressure: :class:`~porepy.ad.MergedVariable`
        :param temperature: global temperature variable
        :type temperature: :class:`~porepy.ad.MergedVariable`

        :return: specific molar enthalpy of the phase
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        pass

    @abc.abstractmethod
    def dynamic_viscosity(
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        """
        Abstract physical property, dependent on thermodynamic state and the composition.
        The composition variables (molar fractions of present substances) can be accessed
        by reference.

        Math. Dimension:        scalar
        Phys. Dimension:        [mol / m / s]

        :param pressure: global pressure variable
        :type pressure: :class:`~porepy.ad.MergedVariable`
        :param temperature: global temperature variable
        :type temperature: :class:`~porepy.ad.MergedVariable`

        :return: dynamic viscosity of the phase
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        pass

    @abc.abstractmethod
    def thermal_conductivity(
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        """
        Abstract physical property, dependent on thermodynamic state and composition.
        The composition variables (molar fractions of present substances) can be accessed
        by reference.

        Math. Dimension:    2nd-order tensor
        Phys. Dimension:    [W / m / K]

        :param pressure: global pressure variable
        :type pressure: :class:`~porepy.ad.MergedVariable`
        :param temperature: global temperature variable
        :type temperature: :class:`~porepy.ad.MergedVariable`

        :return: thermal conductivity of phase for Fourier's Law
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        pass
