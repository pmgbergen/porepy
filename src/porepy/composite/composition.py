""" Contains the physical extension for :class:`~porepy.grids.grid_bucket.GridBucket`."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from ._composite_utils import COMPUTATIONAL_VARIABLES, create_merged_variable
from .phase import PhaseField

__all__ = ["Composition"]


class Composition:
    """Representation of a composition in a flow problem.

    Combines functionalities of :class:`~porepy.composite.phase.PhaseField` and
    :class:`~porepy.composite.substance.Substance` to obtain a compositional flow model in
    molar formulation

    The physical properties of the domain are assembled using a material representations of
    each grid in the gridbucket,
    namely :class:`~porepy.composite.material_subdomain.MaterialSubDomain`.
    Currently they are accessed grid-wise in the model-classes.

    Public attributes:
        - 'gb': GridBucket for which the CompositionalDomain was initiated
        - 'INITIAL_EQUILIBRIUM': bool. Flagged false if composition changes.
                                 Only flagged True after initial equilibrium is computed,
                                 i.e. after initial variables were set.
                                 DO NOT flag this bool yourself as you please.
        - 'pressure': MergedVariable representing the global pressure
        - 'enthalpy': MergedVariable representing the global enthalpy

    """

    def __init__(self, gb: "pp.GridBucket") -> None:
        """Constructor.

        :param gridbucket: geometrical representation of domain
        :type gridbucket: :class:`~porepy.grids.grid_bucket.GridBucket`

        Instantiate default material subdomains using the unit solid class.
        NOTE this approach should be discussed. One could also instantiate None and demand
        certain steps by the modeler.
        The current solution keeps the model 'runable' without the modeler explicitely setting
        material properties for the grids.
        """

        ## PUBLIC
        self.gb: "pp.GridBucket" = gb
        self.INITIAL_EQUILIBRIUM: bool = False
        self.dof_manager: "pp.DofManager" = pp.DofManager(gb)
        self.eq_manager: "pp.ad.EquationManager" = pp.ad.EquationManager(
            gb, self.dof_manager
        )
        # store phase equilibria equations for each substance name (key)
        # equations are stored in dicts per substance (key: equ name, value: operator)
        self.phase_equilibrium_equations: Dict[
            str, Dict[str, "pp.ad.Operator"]
        ] = dict()

        ## PRIVATE
        # set containing all present substances
        self._present_substances: Set["pp.composite.Substance"] = set()
        # key: phase name, value: tuple of present substance names
        self._phases_per_substance: Dict[
            "pp.composite.Substance", Set[PhaseField]
        ] = dict()
        # instances of added phases
        self._present_phases: List[PhaseField] = list()

        # initiate system-wide primary variables
        self._pressure_var: str = COMPUTATIONAL_VARIABLES["pressure"]
        self._enthalpy_var: str = COMPUTATIONAL_VARIABLES["enthalpy"]
        self._temperature_var: str = COMPUTATIONAL_VARIABLES["temperature"]
        self._pressure: "pp.ad.MergedVariable" = create_merged_variable(
            gb, {"cells": 1}, self._pressure_var
        )
        self._enthalpy: "pp.ad.MergedVariable" = create_merged_variable(
            gb, {"cells": 1}, self._enthalpy_var
        )
        self._temperature: "pp.ad.MergedVariable" = create_merged_variable(
            gb, {"cells": 1}, self._temperature_var
        )
        # store subsystem components (references) for faster assembly
        self._phase_equilibrium_subsystem: dict = dict()
        self._saturation_flash_subsystem: dict = dict()
        self._isenthalpic_flash_subsystem: dict = dict()

        self.dof_manager.update_dofs()

    def __str__(self) -> str:
        """Returns string representation of instance,
        with information about invoked variables and phases.
        Concatenates the string representation of the underlying gridbucket.
        """

        out = "Composition with "

        out += "\n%s phases:\n" % (str(self.num_phases))

        for phase_name in [phase.name for phase in self._present_phases]:
            out += phase_name + "\n"

        out += "\n%s substances:\n" % (str(self.num_substances))

        for substance_name in [
            substance.name for substance in self._present_substances
        ]:
            out += substance_name + "\n"

        out += "\non gridbucket\n"

        return out + str(self.gb)

    def __iter__(self) -> Generator[PhaseField, None, None]:
        """Returns an iterator over all anticipated phases.

        IMPORTANT: The order in this iterator (tuple) is used for choosing e.g.,
        the values in a list of 'numpy.array' when setting initial values.
        Use the order returns here everytime you deal with phase-related values or other.
        """
        for phase in self._present_phases:
            yield phase

    @property
    def num_phases(self) -> int:
        """
        :return: number of added phases
        :rtype: int
        """
        return len(self._present_phases)

    @property
    def num_substances(self) -> int:
        """
        :return: total number of distinct substances in all phases
        :rtype: int
        """
        return len(self._present_substances)

    @property
    def num_phase_equilibrium_equations(self) -> Dict[str, int]:
        """Returns a dict containing the number of necessary phase equilibrium equations
        per substance.
        The number is based on the number of phases in which a substance is present i.e.,
        if substance c is present in n_p(c) phases, we need n_p(c) - 1 equilibrium equations.

        :return: dictionary with substance names as keys and equation numbers as values
        :rtype: dict
        """
        equ_nums = dict()

        for substance in self._present_substances:
            equ_nums.update(
                {substance.name: len(self._phases_per_substance[substance]) - 1}
            )

        return equ_nums

    @property
    def substances(self) -> Tuple["pp.composite.Substance"]:
        """
        :return: substances present in phases
        :rtype: tuple
        """
        return tuple(self._present_substances)

    @property
    def subsystem(self) -> Dict[str, List]:
        """Returns the subsystem representing the phase equilibrium calculations.
        - 'equations' : names of respective equations in the equation manager
        - 'vars' : list of MergedVariables associated with the subsystem
        - 'var_names' : names of above MergedVariable instances
        """
        return self._phase_equilibrium_subsystem

    @property
    def pressure(self) -> "pp.ad.MergedVariable":
        """(Global) pressure. Primary variable in the compositional flow.
        Given per cell

        Math. Dimension:        scalar
        Phys. Dimension:        [Pa] = [N / m^2]

        :return: primary variable pressure
        :rtype: :class:`~porepy.numerics.ad.operators.MergedVariable`
        """
        return self._pressure

    @property
    def enthalpy(self) -> "pp.ad.MergedVariable":
        """Specific molar enthalpy of the composition.
        Given per cell.

        Math. Dimension:        scalar
        Phys. Dimension:        [J / mol / K]

        :return: primary variable (specific molar) enthalpy
        :rtype: :class:`~porepy.numerics.ad.operators.MergedVariable`
        """
        return self._enthalpy

    @property
    def temperature(self) -> "pp.ad.MergedVariable":
        """Temperature of the composition. Given per cell.

        Math. Dimension:        scalar
        Phys. Dimension:        [K]

        :return: secondary variable temperature
        :rtype: :class:`~porepy.numerics.ad.operators.MergedVariable`
        """
        return self._temperature

    def composit_density(
        self, prev_time: Optional[bool] = False
    ) -> Union["pp.ad.Operator", int, float]:
        """
        :param prev_time: indicator to use values at previous time step
        :type prev_time: bool

        :return: overall molar density of the composition using the caloric relation.
        :rtype: :class:`~porepy.numerics.ad.operators.MergedVariable`
        """
        if prev_time:
            rho = [
                phase.saturation.previous_timestep()
                * phase.molar_density(
                    self.pressure.previous_timestep(), self.enthalpy.previous_timestep()
                )
                for phase in self
            ]
        else:
            rho = [
                phase.saturation * phase.molar_density(self.pressure, self.enthalpy)
                for phase in self
            ]
        # density = 0.0
        # for phase in self:
        #     density += phase.saturation * phase.molar_density(
        #         self.pressure, self.enthalpy
        #     )
        return sum(rho)

    def add_phase(self, phases: Union[List[PhaseField], PhaseField]) -> None:
        """
        Adds the phases to the compositional flow.
        Resolves the composition of the flow (which substance appears in which phase).
        Skips phases which were already added.

        The phases must be instantiated using the same
        :class:`~porepy.GridBucket` instance.

        :param phases: a p instance to be added or multiple phase instances in a list.
        :type phases: :class:`~porepy.composite.phase.PhaseField`
        """

        if isinstance(phases, PhaseField):
            phases = [phases]

        old_names = {phase.name for phase in self._present_phases}
        # check if phase is instantiated on same domain or
        # if its name is already among the present phases
        for phase in phases:
            if phase.gb != self.gb:
                raise ValueError(
                    "Phase '%s' instantiated on unknown grid bucket." % (phase.name)
                )

            if phase.name in old_names:
                warnings.warn(
                    "Phase '%s' has already been added. Skipping..." % (phase.name)
                )
                continue
            else:
                self._present_phases.append(phase)

        self._resolve_composition()

    def add_phase_equilibrium_equation(
        self,
        substance: "pp.composite.Substance",
        equation: "pp.ad.Operator",
        equ_name: str,
    ) -> None:
        """Adds a phase equilibrium equation for closing the system.

        This is modularized and for now it is completely up to the modeler to assure
        a non-singular system of equations.

        NOTE: Make sure the equation is such, that the right-hand-side is zero and the passed
        operator represent the left-hand-side
        NOTE: Make sure the name is unique. Unknown behavior else.

        For the number of necessary equations see
        :method:`~porepy.composite.composition.Composition.num_phase_equilibrium_equations`.

        :param substance: Substance instance present in the composition
        :type substance: :class:`~porepy.composite.substance.Substance`
        :param equation: AD Operator representing the equation s.t. the right-hand-side is 0
        :type equation: :class:`~porepy.ad.Operator`
        :param equ_name: name of the given equation
        :type equ_name: str
        """

        if substance in self._present_substances:
            raise_error = False
            eq_num_max = self.num_phase_equilibrium_equations[substance.name]
            eq_num_is = len(self.phase_equilibrium_equations[substance.name])
            # if not enough equations available, add it
            if eq_num_is < eq_num_max:
                self.phase_equilibrium_equations[substance.name].update(
                    {equ_name: equation}
                )
            # if enough are available, check if one is replacable (same name)
            elif eq_num_is == eq_num_max:
                if equ_name in self.phase_equilibrium_equations[substance.name].keys():
                    self.phase_equilibrium_equations[substance.name].update(
                        {equ_name: equation}
                    )
                else:
                    raise_error = True
            else:
                raise_error = True
            # if the supposed number of equation would be violated, raise an error
            if raise_error:
                raise RuntimeError(
                    "Maximal number of phase equilibria equations "
                    + "for substance %s (%i) exceeded." % (substance.name, eq_num_max)
                )
        else:
            raise ValueError(
                "Substance '%s' not present in composition." % (substance.name)
            )

    def remove_phase_equilibrium_equation(
        self, equ_name: str
    ) -> Union[None, "pp.ad.Operator"]:
        """Removes the equation with name 'equ_name'.

        :param equ_name: name of the equation passed to
            :method:`~porepy.composit.Composition.add_phase_equilibrium_equation`
        :type equ_name: str

        :return: If the equation name is valid, returns the las operator found under this name.
            Returns None otherwise
        """
        operator = None
        for substance in self._present_substances:
            operator = self.phase_equilibrium_equations[substance.name].pop(
                equ_name, None
            )

        return operator

    def set_initial_state(
        self,
        pressure: Union[List[float], List["np.ndarray"]],
        temperature: Union[List[float], List["np.ndarray"]],
        saturations: List[Union[List[float], List["np.ndarray"]]],
    ) -> None:
        """Sets the initial compositional and thermodynamic state of the system.
        Natural variables are used as arguments, since they are more feasible.

        Enthalpy is computed using an isenthalpic flash.

        THE FOLLOWING IS ASSUMED FOR THE ARGUMENTS:
            - the top list contains lists/arrays/floats per grid in gridbucket
            - for pressure and enthalpy, the list contains arrays/floats per subdomain
            - for saturations, the list contains lists (per subdomain)
              with saturation arrays/floats per phase
            - the order of the gridbuckets iterator is used for the list-entries per grid
            - the order of this instances iterator is assumed for the saturation values
              in the nested lists
            - each variable is either given homogenously (float per variable) or
            heterogeneously (float per cell, i.e. array per grid)

        Finally, this methods asserts the initial unitarity of the saturation values per cell.

        :param pressure: initial pressure values per grid
        :type pressure: list
        :param temperature: initial temperature per grid
        :type temperature: list
        :param saturations: saturation values per grid per anticipated phase
        :type saturations: list
        """

        for idx, grid_data in enumerate(self.gb):

            grid, data = grid_data

            # check if data dictionary has necessary keys, if not, create them.
            # TODO check if we should also prepare the 'previous_timestep' values here
            if pp.STATE not in data:
                data[pp.STATE] = {}
            if pp.ITERATE not in data[pp.STATE]:
                data[pp.STATE][pp.ITERATE] = {}

            ## setting initial pressure values for this grid
            vals = pressure[idx]
            # convert homogenous fractions to values per cell
            if isinstance(vals, float):
                vals = np.ones(grid.num_cells) * vals
            data[pp.STATE][self._pressure_var] = np.copy(vals)
            data[pp.STATE][pp.ITERATE][self._pressure_var] = np.copy(vals)

            ## setting initial temperature values for this grid
            vals_t = temperature[idx]
            # convert homogenous fractions to values per cell
            if isinstance(vals, float):
                vals_t = np.ones(grid.num_cells) * vals_t
            data[pp.STATE][self._temperature_var] = np.copy(vals_t)
            data[pp.STATE][pp.ITERATE][self._temperature_var] = np.copy(vals_t)

            ## setting initial saturation values for this grid
            # assertions of unitarity of saturation per grid (per cell actually)
            sum_saturation_per_grid = np.zeros(grid.num_cells)

            # loop over next level: fractions per phase (per grid)
            # this throws an error if there are values missing for a phase (or too many given)
            phases = [phase for phase in self]
            for phase, values in zip(phases, saturations[idx]):

                # convert homogenous fractions to values per cell
                if isinstance(values, float):
                    values = np.ones(grid.num_cells) * values

                # this throws an error if the dimensions should mismatch when giving fractions
                # for a grid in array form
                sum_saturation_per_grid += values

                data[pp.STATE][phase.saturation_var] = np.copy(values)
                data[pp.STATE][pp.ITERATE][phase.saturation_var] = np.copy(values)

            # assert the fractional character (sum equals 1) in each cell
            # if not np.allclose(sum_saturation_per_grid, 1.):
            if np.any(sum_saturation_per_grid != 1.0):  # TODO check sensitivity
                raise ValueError(
                    "Initial saturations do not sum up to 1. on each cell on grid:\n"
                    + str(grid)
                )

    def phases_of_substance(
        self, substance: "pp.composite.Substance"
    ) -> Tuple[PhaseField, ...]:
        """
        :return: for given substance, tuple of phases is returned which contain this substance.
        :rtype: tuple
        """
        return tuple(self._phases_per_substance[substance])

    def initialize_composition(self, initial_calc: Optional[bool] = False) -> bool:
        """
        Sets the equations for this model and executes optionally the initial calculations.
        (see :method:`~porepy.composite.CompositionalFlow.do_initial_calculations`)

        Throws an error if not all initial values have been set.

        Initial values of molar variables are computed using the natural variables.

        Use :method:`~porepy.compostie.phase.PhaseField.set_initial_fractions`
        to set initial molar fractions per phase
        Use
        :method:`~porepy.compostie.compositional_domain.CompositionalDomain.set_initial_state`
        to set the rest.

        Flag `initial_calc` as True to compute the initial calculations
        If successful, flags this instance by setting
        :data:`~~porepy.compostie.compositional_domain.CompositionalDomain.INITIAL_EQUILIBRIUM`
        true.

        :param initial_calc: flag for computing initial calculations
        :type initial_calc: bool

        :return: Always True, if no initial calculations are done. Otherwise result of calc.
        :rtype: bool
        """

        self._calculate_initial_molar_phase_fractions()
        self._calculate_initial_overall_substance_fractions()
        self._check_num_phase_equilibrium_equations()

        # setting of equations and subsystems
        equations = dict()
        subset: Any
        subsystem_dict: Dict[str, list] = {
            "equations": list(),
            "vars": list(),
            "var_names": list(),
        }
        # deep copies
        phase_equilibrium_subsystem = dict(subsystem_dict)
        saturation_flash_subsystem = dict(subsystem_dict)
        isenthalpic_flash_subsystem = dict(subsystem_dict)

        ### EQUILIBRIUM CALCULATIONS
        # num_substances overall fraction equations
        subset = self.overall_substance_fraction_equations()
        for c, substance in enumerate(self._present_substances):
            name = "overall_substance_fraction_%s" % (substance.name)
            equations.update({name: subset[c]})
            phase_equilibrium_subsystem["equations"].append(name)
            phase_equilibrium_subsystem["vars"].append(substance.overall_fraction)
            phase_equilibrium_subsystem["var_names"].append(
                substance.overall_fraction_var
            )

        # num_phases - 1 substance in phase sum equations
        subset = self.substance_in_phase_sum_equations()
        phases = [phase for phase in self]
        for i in range(self.num_phases - 1):
            name = "substance_in_phase_sum_%s_%s" % (phases[i].name, phases[i + 1].name)
            equations.update({name: subset[i]})
            phase_equilibrium_subsystem["equations"].append(name)

        # num_substances * (num_phases(of substance) - 1) phase equilibrium equations
        for substance in self._present_substances:
            for equ_name in self.phase_equilibrium_equations[substance.name]:
                equation = self.phase_equilibrium_equations[substance.name][equ_name]
                equations.update({equ_name: equation})
                phase_equilibrium_subsystem["equations"].append(equ_name)

        # adding substance fractions in phases to subsystem
        for substance in self._present_substances:
            for phase in self._phases_per_substance[substance]:
                phase_equilibrium_subsystem["vars"].append(
                    substance.fraction_in_phase(phase.name)
                )
                phase_equilibrium_subsystem["var_names"].append(
                    substance.fraction_in_phase_var(phase.name)
                )

        # 1 phase fraction unity equation
        equ = self.molar_phase_fraction_sum()
        name = "molar_phase_fraction_sum"
        equations.update({name: equ})
        phase_equilibrium_subsystem["equations"].append(name)
        for phase in self:
            phase_equilibrium_subsystem["vars"].append(phase.molar_fraction)
            phase_equilibrium_subsystem["var_names"].append(phase.molar_fraction_var)

        # store respective subsystem
        self._phase_equilibrium_subsystem = phase_equilibrium_subsystem

        ### SATURATION FLASH
        # num_phases saturation fraction equations for saturation flash calculations
        subset = self.saturation_flash_equations()
        for e, phase in enumerate(self._present_phases):
            name = "saturation_flash_%s" % (phase.name)
            equations.update({name: subset[e]})
            saturation_flash_subsystem["equations"].append(name)
            saturation_flash_subsystem["vars"].append(phase.saturation)
            saturation_flash_subsystem["var_names"].append(phase.saturation_var)
        # store respective subsystem
        self._saturation_flash_subsystem = saturation_flash_subsystem

        ### ISENTHALPIC FLASH
        # 1 isenthalpic equation for isenthalpic flash
        equ = self.isenthalpic_flash_equation()
        name = "isenthalpic_flash"
        equations.update({name: equ})
        # store respective subsystem
        isenthalpic_flash_subsystem["equations"].append(name)
        isenthalpic_flash_subsystem["vars"].append(self._temperature)
        isenthalpic_flash_subsystem["var_names"].append(self._temperature_var)
        self._isenthalpic_flash_subsystem = isenthalpic_flash_subsystem

        self.eq_manager.equations = equations

        success = True
        if initial_calc:
            success = self.do_calculations()

        return success

    def do_calculations(self) -> bool:
        """Performs the following calculations:
            - the initial equilibrium
            - initial saturations flash
            - initial isenthalpic flash.

        This step is put in a separate method in case this class gets inherited.
        NOTE VL: All computations here can be parallelized since they are local (per cell).

        :return: True if all calculations are successful, False otherwise
        :rtype: bool
        """
        equilibrium = self.compute_phase_equilibrium()
        saturations = self.saturation_flash()
        isenthalpic = self.isenthalpic_flash()

        if equilibrium and saturations and isenthalpic:
            return True
        else:
            return False

    # -----------------------------------------------------------------------------------------
    ### Equilibrium and flash calculations
    # -----------------------------------------------------------------------------------------

    def compute_phase_equilibrium(
        self, max_iterations: int = 100, eps: float = 1.0e-10
    ) -> bool:
        """Computes the equilibrium using the following equations:
            - overall substance fraction equations (num_substances)
            - fugacity equations (num_substances * (num_phases -1))
            - substance fraction in phase unity equations (num_phases - 1)
            - molar phase fraction unity equation (1)

        Equilibrium for fixed pressure and enthalpy is assumed.

        :param max_iterations: set maximal number for Newton iterations
        :type max_iterations: int
        :param eps: tolerance for Newton residual
        :type eps: float

        :return: True if successful, False otherwise
        :rtype: bool
        """
        # get objects defining the subsystem
        subsystem = self._phase_equilibrium_subsystem

        return self._subsystem_newton(
            subsystem["equations"],
            subsystem["vars"],
            subsystem["var_names"],
            max_iterations,
            eps,
        )

    def saturation_flash(self, max_iterations: int = 100, eps: float = 1.0e-10) -> bool:
        """Performs a flash calculation using
        :method:`~porepy.composite.CompositionaFlow.saturation_fraction_equations` in order to
        obtain saturation values.

        :param max_iterations: set maximal number for Newton iterations
        :type max_iterations: int
        :param eps: tolerance for Newton residual
        :type eps: float

        :return: True if successful, False otherwise
        :rtype: bool
        """
        # get objects defining the subsystem
        subsystem = self._saturation_flash_subsystem

        return self._subsystem_newton(
            subsystem["equations"],
            subsystem["vars"],
            subsystem["var_names"],
            max_iterations,
            eps,
        )

    def isenthalpic_flash(
        self, max_iterations: int = 100, eps: float = 1.0e-10
    ) -> bool:
        """Performs an isenthalpic flash to obtain temperature values.

        :param max_iterations: set maximal number for Newton iterations
        :type max_iterations: int
        :param eps: tolerance for Newton residual
        :type eps: float

        :return: True if successful, False otherwise
        :rtype: bool
        """
        # get objects defining the subsystem
        subsystem = self._isenthalpic_flash_subsystem

        return self._subsystem_newton(
            subsystem["equations"],
            subsystem["vars"],
            subsystem["var_names"],
            max_iterations,
            eps,
        )

    # -----------------------------------------------------------------------------------------
    ### Model equations
    # -----------------------------------------------------------------------------------------

    def overall_component_fractions_sum(self) -> "pp.ad.Operator":
        """Returns 1 equation representing the unity of the overall component fractions.

        sum_c zeta_c - 1 = 0
        """
        # -1
        equation = -1.0 * np.ones(self.gb.num_cells())

        for substance in self._present_substances:
            # + zeta_c
            equation += substance.overall_fraction

        return equation

    def overall_substance_fraction_equations(self) -> List["pp.ad.MergedVariable"]:
        """Returns num_substances equations representing the definition of the
        overall component fraction.
        The order of equations per substance equals the order of substances as they appear in
        phases in this composition (see iterator).

        zeta_c - \sum_e chi_ce * xi_e = 0
        """
        equations = list()

        for substance in self._present_substances:
            # zeta_c
            equation = substance.overall_fraction

            for phase in self._phases_per_substance[substance]:
                # - xi_e * chi_ce
                equation -= phase.molar_fraction * substance.fraction_in_phase(
                    phase.name
                )
            # equations per substance
            equations.append(equation)

        return equations

    def molar_phase_fraction_sum(self) -> "pp.ad.Operator":
        """Returns 1 equation representing the unity of molar phase fractions.

        sum_e xi_e -1 = 0
        """
        # -1
        equation = -1.0 * np.ones(self.gb.num_cells())

        for phase in self:
            # + xi_e
            equation += phase.molar_fraction

        return equation

    def substance_in_phase_sum_equations(self) -> List["pp.ad.Operator"]:
        """Returns num_phases -1 equations representing the unity of the
        molar component fractions per phase.
        For phases in composition (`__iter__`) returns an equation for two neighboring phases.

        sum_c chi_ci - sum_c chi_cj = 0 , i != j phases
        """
        phases = [phase for phase in self]
        equations = list()

        for i in range(self.num_phases - 1):
            # two distinct phases
            phase_i = phases[i]
            phase_j = phases[i + 1]

            # sum_c chi_ci
            sum_i = sum(
                [substance.fraction_in_phase(phase_i.name) for substance in phase_i]
            )
            # sum_c chi_cj
            sum_j = sum(
                [substance.fraction_in_phase(phase_j.name) for substance in phase_j]
            )
            # sum_c chi_ci - sum_c chi_cj
            equations.append(sum_i - sum_j)

        return equations

    def saturation_flash_equations(self) -> List["pp.ad.Operator"]:
        """Returns num_phases equations representing the relation between molar phase fractions
        and saturation (volumetric phase fractions).

        xi_e * (sum_j s_j rho_j) - s_e * rho_e = 0
        """
        equations = list()

        for phase in self:
            # xi_e * (sum_j s_j rho_j)
            equation = phase.molar_fraction * self.composit_density
            # - s_e * rho_e
            equation -= phase.saturation * phase.molar_density(
                self.pressure, self.enthalpy
            )

            equations.append(equation)

        return equations

    def isenthalpic_flash_equation(self) -> "pp.ad.Operator":
        """Returns an operator representing the isenthalpic flash equation.

        rho(p,h) * h = sum_e s_e * rho_e(p,h) * h_e(p, h, T)

        We still express densities in terms of p, h since density is not supposed to change
        during the flash calculation. This leads to solely the phase enthalpies being dependent
        on temperature.
        """
        # rho(p, h, s_e) * h
        equ = self.composit_density * self._enthalpy

        for phase in self._present_phases:
            # - s_e * rho_e(p,h) * h_e(p,h,T)
            equ -= (
                phase.saturation
                * phase.molar_density(self._pressure, self._enthalpy)
                * phase.enthalpy(
                    self._pressure, self._enthalpy, temperature=self._temperature
                )
            )

        return equ

    # -----------------------------------------------------------------------------------------
    ### private methods
    # -----------------------------------------------------------------------------------------

    def _calculate_initial_molar_phase_fractions(self) -> None:
        """
        Name is self-explanatory.

        These calculations have to be done everytime everytime new initial values are set.
        """
        molar_fraction_sum = 0.0

        for phase in self:
            # definition of molar fraction of phase
            molar_fraction = (
                phase.saturation
                * phase.molar_density(self.pressure, self.enthalpy)
                / self.composit_density
            )
            # evaluate the AD expression and get the values
            # this is a global DOF vector and has therefor many zeros
            molar_fraction = molar_fraction.evaluate(self.dof_manager).val
            molar_fraction = molar_fraction[molar_fraction != 0.0]

            molar_fraction_sum += molar_fraction

            # distribute values to respective variable
            self.dof_manager.distribute_variable(
                molar_fraction, variables=[phase.saturation_var]
            )
            self.dof_manager.distribute_variable(
                molar_fraction, variables=[phase.saturation_var], to_iterate=True
            )

        # assert the fractional character (sum equals 1) in each cell
        # if not np.allclose(sum_per_grid, 1.):
        # TODO check if this is really necessary here (equilibrium is computed afterwards)
        if np.any(molar_fraction_sum != 1.0):  # TODO check sensitivity
            raise ValueError(
                "Initial phase molar fractions do not sum up " + "to 1.0 on each cell."
            )

    def _calculate_initial_overall_substance_fractions(self) -> None:
        """Name is self-explanatory.

        These calculations have to be done everytime everytime new initial values are set.
        """
        for grid, data in self.gb:

            sum_per_grid = np.zeros(grid.num_cells)

            for substance in self._present_substances:

                overall_fraction = np.zeros(grid.num_cells)

                for phase in self._phases_per_substance[substance]:

                    substance_fraction = data[pp.STATE][
                        substance.fraction_in_phase_var(phase.name)
                    ]
                    phase_fraction = data[pp.STATE][phase.molar_fraction_var]
                    # calculate the overall fraction as is defined
                    overall_fraction += phase_fraction * substance_fraction

                # sum the overall fractions. should be close to one per cell
                sum_per_grid += overall_fraction

                data[pp.STATE][substance.overall_fraction_var] = np.copy(
                    overall_fraction
                )
                data[pp.STATE][pp.ITERATE][substance.overall_fraction_var] = np.copy(
                    overall_fraction
                )

            # assert the fractional character (sum equals 1) in each cell
            # if not np.allclose(sum_per_grid, 1.):
            # TODO check if this is really necessary here (equilibrium is computed afterwards)
            if np.any(sum_per_grid != 1.0):  # TODO check sensitivity
                raise ValueError(
                    "Initial overall substance fractions do not sum up "
                    + "to 1.0 on each cell on grid:\n"
                    + str(grid)
                )

    def _check_num_phase_equilibrium_equations(self) -> None:
        """Checks whether enough phase equilibria equations were passed.
        Raises en error if not.
        """

        missing_num = 0

        for substance in self._present_substances:
            # should-be-number
            equ_num = self.num_phase_equilibrium_equations[substance.name]
            # summing discrepancy
            missing_num += equ_num - len(
                self.phase_equilibrium_equations[substance.name].keys()
            )

        if missing_num > 0:
            raise RuntimeError(
                "Missing %i phase equilibria equations to initialize the composition."
                % (missing_num)
            )

    def _resolve_composition(self) -> None:
        """Analyzes the composition, i.e. presence of substances in phases.
        Information about substances which are anticipated in multiple phases is stored.

        This method is called internally everytime any new component is added.
        """
        # for given substance (keys), save set of phases containing the substance (values)
        phases_per_substance: Dict["pp.composite.Substance", Set[PhaseField]] = dict()
        # unique substance names, over whole composition
        unique_substances: Set["pp.composite.Substance"] = set()
        # loop through composition, safe references and appearance of substances in phases
        for phase in self:
            for substance in phase:
                unique_substances.add(substance)

                if substance in phases_per_substance.keys():
                    phases_per_substance[substance].add(phase)
                else:
                    phases_per_substance.update({substance: set()})
                    phases_per_substance[substance].add(phase)

        self._present_substances = unique_substances
        self._phases_per_substance = phases_per_substance

        for substance in unique_substances:
            if substance.name not in self.phase_equilibrium_equations.keys():
                self.phase_equilibrium_equations.update({substance.name: dict()})

        # update DOFs since new unknowns were introduced
        self.dof_manager.update_dofs()
        self.INITIAL_EQUILIBRIUM = False

    def _subsystem_newton(
        self,
        equations: List[str],
        variables: List["pp.ad.MergedVariable"],
        var_names: List[str],
        max_iterations: int,
        eps: float,
    ) -> bool:
        """Performs Newton iterations on a specified subset of equations and variables.

        :param equations: names of equations in equation manager
        :type equations: List[str]
        :param variables: subsystem variables
        :type variables: List[MergedVariable]
        :param var_names: names of subsystem variables
        :type var_names: List[str]
        :param max_iterations: set maximal number for Newton iterations
        :type max_iterations: int
        :param eps: tolerance for Newton residual
        :type eps: float

        :return: True if successful, False otherwise
        :rtype: bool
        """
        success = False

        A, b = self.eq_manager.assemble_subsystem(equations, variables)

        if np.linalg.norm(b) <= eps:
            success = True
        else:
            for i in range(max_iterations):

                dx = sps.linalg.spsolve(A, b)

                self.dof_manager.distribute_variable(
                    dx, variables=var_names, additive=True, to_iterate=True
                )

                A, b = self.eq_manager.assemble_subsystem(equations, variables)

                if np.linalg.norm(b) <= eps:
                    # setting state to newly found solution
                    X = self.dof_manager.assemble_variable(
                        variables=var_names, from_iterate=True
                    )
                    self.dof_manager.distribute_variable(X[X != 0], variables=var_names)
                    success = True
                    break

        # if not successful, replace iterate values with initial state values
        if not success:
            X = self.dof_manager.assemble_variable(variables=var_names)
            self.dof_manager.distribute_variable(
                X[X != 0], variables=var_names, to_iterate=True
            )

        return success
