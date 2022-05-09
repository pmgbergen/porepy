""" Contains the abstract base class for all phases."""
from __future__ import annotations

import abc
import warnings
from typing import Dict, Iterator, List, Union

import numpy as np

import porepy as pp

from ._composite_utils import COMPUTATIONAL_VARIABLES, STATES_OF_MATTER

# from weakref import WeakValueDictionary, WeakKeyDictionary

__all__ = [
    "PhaseField",
    "PhysicalState"
]


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

    """ For a computational domain (keys), contains a list of present phases (values). """
    __phase_instances: Dict["pp.composite.CompositionalDomain", list] = dict()

    def __new__(
        cls, name: str, computational_domain: "pp.composite.CompositionalDomain"
    ) -> PhaseField:
        """
        Declarator.
        Assures the phase name is unique for a given computational domain.
        Ambiguities must be avoided due to the central storage of the AD variables and usage
        of the name as a key.
        """
        name = str(name)
        if computational_domain in PhaseField.__phase_instances.keys():
            if name in PhaseField.__phase_instances[computational_domain]:
                raise RuntimeError(
                    "Phase with name '"
                    + name
                    + "' already present in\n"
                    + str(computational_domain)
                )
        else:
            PhaseField.__phase_instances.update({computational_domain: list()})

        PhaseField.__phase_instances[computational_domain].append(name)
        return super().__new__(cls)

    def __init__(self, name: str,
        computational_domain: "pp.composite.CompositionalDomain") -> None:
        """
        Base class constructor. Initiates phase-related AD-variables.

        :param name: name of the phase
        :type name: str
        :param solvent: solvent substance of this phase. Influences the phase bahaviour.
        :type solvent: :class:`~porepy.composite.substance.FluidSubstance`

        :param computational_domain: domain of computations containing the phase
        :type computational_domain:
            :class:`~porepy.composite.computational_domain.ComputationalDomain`
        """
        super().__init__()

        ## PUBLIC
        self.cd = computational_domain

        ## PRIVATE
        self._name = str(name)
        self._anticipated_substances: List["pp.composite.Substance"] = list()

        # Instantiate saturation variable
        self.cd(self.saturation_var, {"cells": 1})
        # Instantiate phase molar fraction variable
        self.cd(self.molar_fraction_var, {"cells": 1})

    def __iter__(self) -> Iterator["pp.composite.Substance"]:
        """
        Iterator over substances present in phase.
        The first substance will always be the solvent passed at instantiation.

        IMPORTANT: The order in this iterator (tuple) is used for choosing e.g. the values in a
        list of 'numpy.array' when setting initial values.
        Use the order returns here everytime you deal with substance-related values or
        other for substances in this phase.

        :return: yields present substance
        :rtype: :class:`~porepy.composite.substance.Substance`
        """
        for substance in self._anticipated_substances:
            yield substance

    @property
    def name(self):
        """
        :return: name of the class. The name is used to construct names for AD variables and
            keys to store them.
        :rtype: str
        """
        return self._name

    @property
    def saturation(self) -> pp.ad.MergedVariable:
        """
        As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the
            saturation of this phase
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self.cd(self.saturation_var)

    @property
    def saturation_var(self) -> str:
        """
        :return: name of the saturation variable under which it is stored in the
            grid data dictionaries
        :rtype: str
        """
        return COMPUTATIONAL_VARIABLES["saturation"] + "_" + self.name

    @property
    def molar_fraction(self) -> pp.ad.MergedVariable:
        """
        As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing
        the molar fraction of this phase
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self.cd(self.molar_fraction_var)

    @property
    def molar_fraction_var(self) -> str:
        """
        :return: name of the phase molar fraction variable under which it is stored in the
        grid data dictionaries
        :rtype: str
        """
        return COMPUTATIONAL_VARIABLES["phase_molar_fraction"] + "_" + self.name

    def add_substance(self,
        substances: Union[List["pp.composite.Substance"], "pp.composite.Substance"]) -> None:
        """
        Adds substances which are anticipated in this phase.

        The Substances must be instantiated on the same
        :class:`~porepy.composite.computational_domain.ComputationalDomain` instance used
        for this object.

        :param substances: substance or list of substances to be added to this phase
        :type substances: :class:`~porepy.composite.substance.Substance`
        """

        if isinstance(substances, pp.composite.Substance):
            substances = [substances]

        for subst in substances:
            # check if objects are meant for same domain
            if subst.cd != self.cd:
                raise ValueError(
                    "Substance '%s' instantiated on unknown ComputationalDomain."
                    % (subst.name)
                )

            # skip already present substances:
            if subst.name in [ps.name for ps in self._anticipated_substances]:
                warnings.warn(
                    "Substance '%s' has already been added to phase '%s'. Skipping..."
                    % (subst.name, self.name)
                )
                continue

            # store reference to present substance
            self._anticipated_substances.append(subst)
            # instantiate molar fraction in phase variable
            self.cd(subst.mfip_var(self.name), {"cells": 1})
            # update the data structures of the computational domain
            # NOTE this is not so nice, there may be a nicer solution in the future
            self.cd.resolve_composition()

    def set_molar_fractions_in_phase(
        self, fractions: Union[List[List["np.array"]], List[List[float]]]
    ) -> None:
        """
        Sets the (initial) molar fractions for present components.

        This methods assumes the order of the gridbuckets iterator to be used for the argument
        'fractions', containing the initial values per grid.
        The initial values themselves have to be given as numpy.array per grid per substance.

        in detail, the input must have the following, nested structure:
            - the top list ('fractions') contains lists per grid in gridbucket
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
        for grid_data, vals_per_grid in zip(self.cd.gb, fractions):

            grid = grid_data[0]
            data = grid_data[1]

            sum_per_grid = np.zeros(grid.num_cells)

            # loop over next level: fractions per substance (per grid)
            for subst, vals in zip(self, vals_per_grid):

                # convert homogenous fractions to values per cell
                if isinstance(vals, float):
                    vals = np.ones(grid.num_cells) * vals

                # this throws an error if the dimensions should mismatch when giving
                # fractions for a grid in array form
                sum_per_grid += vals

                # check if data dictionary has necessary keys
                # If not, create them.
                # TODO check if we should also prepare the 'previous_timestep' values here
                if pp.STATE not in data:
                    data[pp.STATE] = {}
                if pp.ITERATE not in data[pp.STATE]:
                    data[pp.STATE][pp.ITERATE] = {}

                data[pp.STATE][subst.mfip_var(self.name)] = np.copy(vals)
                data[pp.STATE][pp.ITERATE][subst.mfip_var(self.name)] = np.copy(vals)

            # assert the fractional character (sum equals 1) in each cell
            if np.any(sum_per_grid != 1.0):  # TODO check sensitivity
                raise ValueError(
                    "Initial fractions do not sum up to 1. on each cell on grid:\n"
                    + str(grid)
                )

    def mass_phase_density(self, *args, **kwargs) -> pp.ad.Operator:
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
        if not self._anticipated_substances:
            return weight

        # add the mass-weighted fraction for each present substance.
        for substance in self._anticipated_substances:
            weight += substance.molar_mass() * substance.molar_fraction_in_phase(
                self.name
            )

        # Multiply the mass weight with the molar density and return the operator
        return weight * self.molar_phase_density(*args, **kwargs)

    # ------------------------------------------------------------------------------
    ### Abstract, phase-related physical properties
    # ------------------------------------------------------------------------------

    @abc.abstractmethod
    def molar_phase_density(self, *args, **kwargs) -> pp.ad.Operator:
        """
        Abstract physical property.

        Math. Dimension:        scalar
        Phys. Dimension:        [mol / m^3]

        :return: molar density of the phase
            (dependent on thermodynamic state and the composition)
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        pass

    @abc.abstractmethod
    def phase_viscosity(self, *args, **kwargs) -> pp.ad.Operator:
        """
        Abstract physical property.

        Math. Dimension:        scalar
        Phys. Dimension:        [kg / m / s]

        :return: dynamic viscosity of the phase
            (dependent on thermodynamic state and the composition)
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        pass
        # raise NotImplementedError("Viscosity for phase '%s' not implemented."%(self.name))

    @abc.abstractmethod
    def relative_permeability(self, law: str, *args, **kwargs) -> pp.ad.Operator:
        """
        Currently supported heuristic laws (values for 'law'):
            - 'brooks_corey':   Brook-Corey model TODO finish
            - 'quadratic':      quadratic power law for saturation

        Inherit this class and overwrite this method if you want to implement special models
        for the relative permeability.
        Use positional arguments 'args' and keyword arguments 'kwargs' to provide arguments
        for the heuristic law.

        Math. Dimension:        scalar
        Phys. Dimension:        [-] (fractional)

        :param law: name of the law to be applied (see valid values above)
        :type law: str

        :return: relative permeability using the respectie law
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        law = str(law)
        if law == "quadratic":
            return pp.ad.Function(
                lambda S: S**2, "rel-perm-%s-%s" % (law, self.name)
            )(self.saturation)
        else:
            raise NotImplementedError(
                "Unknown 'law' keyword for rel.Perm.: %s \n" % (law)
                + "Available: 'quadratic,'"
            )


class PhysicalState(PhaseField):
    """
    This class represent the 'state of matter'-interpretation of of the term phase.
    The phase(-field) is associated with a specific, dominating substance in a specific
    state of matter, which serves as a solvent for in the region occupied by this phase.

    Physical properties like density and viscosity of the phase are dominated by the respective
    physical property of the solvent.

    Functionalities:
        - provides classifications of states of matter (currently: solid, liquid, gas)
        - provides simple physical properties of the phase associated
          with the underlying substance
        - based on the last point, serves as a concrete implementation of a
          phase-field and is not abstract

    Physical properties are mainly heuristic laws. In order to provide more models/laws
    create a child class and override the respective method.
    Follow the if-else construction given in the methods here and return a call to
    '__super__().<physical property>' in the 'else' branch.
    """

    def __init__(
        self,
        name: str,
        computational_domain: "pp.composite.CompositionalDomain",
        solvent: "pp.composite.FluidSubstance",
        state_of_matter: str,
    ) -> None:
        """
        Constructor with additional arguments.

        :param solvent: substance filling this phase field and for which the
            state of matter is assumed.
        :type solvent: :class:`~porepy.composite.substance.FluidSubstance`

        :param state_of_matter: currently supported states of matter are
            'solid', 'liquid', 'gas'
        :type state_of_matter: str
        """
        # stringify to make sure string operations work as intended
        state_of_matter = str(state_of_matter)

        if state_of_matter not in STATES_OF_MATTER:
            raise ValueError("Unknown state of matter '%s'." % (state_of_matter))

        super().__init__(name, computational_domain)

        # Add the solvent to the present substances.
        # It will always be the first substance in the iterator this way.
        self.add_substance(solvent)

        ## PRIVATE
        self._solvent: "pp.composite.FluidSubstance" = solvent
        self._state = state_of_matter

    # ------------------------------------------------------------------------------
    ### HEURISTIC LAWS NOTE all heuristic laws can be modularized somewhere and referenced here
    # ------------------------------------------------------------------------------

    @abc.abstractmethod
    def molar_phase_density(self, law: str, *args, **kwargs) -> pp.ad.Operator:
        """
        Currently supported heuristic laws (values for 'law'):
            - 'solvent':       uses the unmodified solvent density

        Inherit this class and overwrite this method if you want to implement special models
        for the phase density.
        Use positional arguments 'args' and keyword arguments 'kwargs' to provide arguments
        for the heuristic law.

        Math. Dimension:        scalar
        Phys. Dimension:        [mol / m^3]

        :param law: name of the law to be applied (see valid values above)
        :type law: str

        :return: molar density of the phase
            (dependent on thermodynamic state and the composition)
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        # stringify to ensure the string operations below work
        law = str(law)
        # assumed thermodynamic state variables of the current implementation
        pressure = self.cd(COMPUTATIONAL_VARIABLES["pressure"])
        enthalpy = self.cd(COMPUTATIONAL_VARIABLES["enthalpy"])

        if law == "solvent":

            return pp.ad.Function(
                lambda p, h: self._solvent.molar_density(p, h),
                "molar-density-%s-%s" % (self.name, law),
            )(pressure, enthalpy)

        else:
            raise NotImplementedError(
                "Unknown 'law' keyword for phase viscosity.: %s \n" % (law)
                + "Available: 'solvent,'"
            )

    @abc.abstractmethod
    def phase_viscosity(self, law: str, *args, **kwargs) -> pp.ad.Operator:
        """
        Currently supported heuristic laws (values for 'law'):
            - 'solvent':        uses the unmodified solvent viscosity

        Inherit this class and overwrite this method if you want to implement special models
        for the phase viscosity.
        Use positional arguments 'args' and keyword arguments 'kwargs' to provide arguments
        for the heuristic law.

        Math. Dimension:        scalar
        Phys. Dimension:        [kg / m / s]

        :param law: name of the law to be applied (see valid values above)
        :type law: str

        :return: dynamic viscosity of the phase
            (dependent on thermodynamic state and the composition)
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        # stringify to ensure the string operations below work
        law = str(law)
        # assumed thermodynamic state variables of the current implementation
        pressure = self.cd(COMPUTATIONAL_VARIABLES["pressure"])
        enthalpy = self.cd(COMPUTATIONAL_VARIABLES["enthalpy"])

        if law == "solvent":
            return pp.ad.Function(
                lambda p, h: self._solvent.dynamic_viscosity(p, h),
                "viscosity-%s-%s" % (self.name, law),
            )(pressure, enthalpy)
        else:
            raise NotImplementedError(
                "Unknown 'law' keyword for phase viscosity.: %s \n" % (law)
                + "Available: 'solvent,'"
            )

    @abc.abstractmethod
    def relative_permeability(self, law: str, *args, **kwargs) -> pp.ad.Operator:
        """
        Currently supported heuristic laws (values for 'law'):
            - 'brooks_corey':   Brook-Corey model TODO finish
            - 'quadratic':      quadratic power law for saturation

        Inherit this class and overwrite this method if you want to implement special models
        for the relative permeability.
        Use positional arguments 'args' and keyword arguments 'kwargs' to provide arguments
        for the heuristic law.

        Math. Dimension:        scalar
        Phys. Dimension:        [-] (fractional)

        :param law: name of the law to be applied (see valid values above)
        :type law: str

        :return: relative permeability using the respectie law
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        law = str(law)
        if law == "quadratic":
            return pp.ad.Function(
                lambda S: S**2, "rel-perm-%s-%s" % (law, self.name)
            )(self.saturation)
        else:
            raise NotImplementedError(
                "Unknown 'law' keyword for rel.Perm.: %s \n" % (law)
                + "Available: 'quadratic,'"
            )
