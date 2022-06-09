""" Contains the physical extension for :class:`~porepy.grids.grid_bucket.GridBucket`."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Generator, List, Literal, Optional, Set, Tuple, Union

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
        - 'pressure': MergedVariable representing the global pressure
        - 'enthalpy': MergedVariable representing the global enthalpy

    TODO: pressure, enthalpy, overall molar substance fractions,...
        all these operators need only be evaluated ones.
        Phase densities might change due to changing substance fractions in phases...
    TODO: All computations here can be parallelized since they are local...
    """

    def __init__(self, gb: pp.GridBucket) -> None:
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
        self.gb: pp.GridBucket = gb
        self.dof_manager: pp.DofManager = pp.DofManager(gb)
        self.eq_manager: pp.ad.EquationManager = pp.ad.EquationManager(
            gb, self.dof_manager
        )
        # store phase equilibria equations for each substance name (key)
        # equations are stored in dicts per substance (key: equ name, value: operator)
        self.phase_equilibrium_equations: Dict[str, Dict[str, pp.ad.Operator]] = dict()
        # contains chronologically information about past applications of the Newton algorithm
        self.newton_history: List[Dict[str, Any]] = list()

        ## PRIVATE
        # set containing all present substances
        self._present_substances: Set[pp.composite.Substance] = set()
        # key: phase name, value: tuple of present substance names
        self._phases_per_substance: Dict[
            pp.composite.Substance, Set[PhaseField]
        ] = dict()
        # instances of added phases
        self._present_phases: List[PhaseField] = list()

        # initiate system-wide primary variables
        self._pressure_var: str = COMPUTATIONAL_VARIABLES["pressure"]
        self._enthalpy_var: str = COMPUTATIONAL_VARIABLES["enthalpy"]
        self._temperature_var: str = COMPUTATIONAL_VARIABLES["temperature"]
        self._pressure: pp.ad.MergedVariable = create_merged_variable(
            gb, {"cells": 1}, self._pressure_var
        )
        self._enthalpy: pp.ad.MergedVariable = create_merged_variable(
            gb, {"cells": 1}, self._enthalpy_var
        )
        self._temperature: pp.ad.MergedVariable = create_merged_variable(
            gb, {"cells": 1}, self._temperature_var
        )
        # store subsystem components (references) for faster assembly
        self._phase_equilibrium_subsystem: dict = dict()
        self._saturation_flash_subsystem: dict = dict()
        self._isenthalpic_flash_subsystem: dict = dict()
        self._isothermal_flash_subsystem: dict = dict()
        # defines the maximal number of Newton history entries
        self._max_history = 200

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
    def substances(self) -> Tuple[pp.composite.Substance, ...]:
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
    def pressure(self) -> pp.ad.MergedVariable:
        """(Global) pressure. Primary variable in the compositional flow.
        Given per cell

        Math. Dimension:        scalar
        Phys. Dimension:        [Pa] = [N / m^2]

        :return: primary variable pressure
        :rtype: :class:`~porepy.numerics.ad.operators.MergedVariable`
        """
        return self._pressure

    @property
    def enthalpy(self) -> pp.ad.MergedVariable:
        """Specific molar enthalpy of the composition.
        Given per cell.

        Math. Dimension:        scalar
        Phys. Dimension:        [J / mol / K]

        :return: primary variable (specific molar) enthalpy
        :rtype: :class:`~porepy.numerics.ad.operators.MergedVariable`
        """
        return self._enthalpy

    @property
    def temperature(self) -> pp.ad.MergedVariable:
        """Temperature of the composition. Given per cell.

        Math. Dimension:        scalar
        Phys. Dimension:        [K]

        :return: secondary variable temperature
        :rtype: :class:`~porepy.numerics.ad.operators.MergedVariable`
        """
        return self._temperature

    def composit_density(
        self,
        prev_time: Optional[bool] = False,
        temperature: Union["pp.ad.MergedVariable", None] = None,
    ) -> Union[pp.ad.Operator, Literal[0]]:
        """
        :param prev_time: (optional) indicator to use values at previous time step
        :type prev_time: bool
        :param temperature: if passed, calculates temperature based phase enthalpies
        :type temperature: :class:`~porepy.ad.MergedVariable`

        :return: overall molar density of the composition using the caloric relation.
        :rtype: :class:`~porepy.numerics.ad.operators.MergedVariable`
        """
        if prev_time:
            if temperature:
                temperature=temperature.previous_timestep()

            rho = [
                phase.saturation.previous_timestep()
                * phase.molar_density(
                    self.pressure.previous_timestep(),
                    self.enthalpy.previous_timestep(),
                    temperature=temperature,
                )
                for phase in self
            ]
        else:
            rho = [
                phase.saturation
                * phase.molar_density(
                    self.pressure, self.enthalpy, temperature=temperature
                )
                for phase in self
            ]

        return sum(rho)

    def add_phases(self, phases: Union[List[PhaseField], PhaseField]) -> None:
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
            for phase, values in zip(self, saturations[idx]):

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

    def initialize_composition(self) -> None:
        """
        Sets the equations for this model.
        Throws an error if not all initial values have been set.
        Initial values of molar variables are computed using the natural variables.

        Use :method:`~porepy.compostie.phase.PhaseField.set_initial_fractions`
        to set initial molar fractions per phase
        Use
        :method:`~porepy.compostie.compositional_domain.CompositionalDomain.set_initial_state`
        to set the rest.
        """
        # at this point we assume all DOFs are defined and we reset the following
        # to get correct DOF mappins
        self.dof_manager = pp.DofManager(self.gb)
        self.eq_manager = pp.ad.EquationManager(self.gb, self.dof_manager)

        # setting of equations and subsystems
        equations = dict()
        subset: Any
        isothermal_flash_subsystem: Dict[str, list] = self._get_subsystem_dict()
        phase_equilibrium_subsystem: Dict[str, list] = self._get_subsystem_dict()
        saturation_flash_subsystem: Dict[str, list] = self._get_subsystem_dict()
        isenthalpic_flash_subsystem: Dict[str, list] = self._get_subsystem_dict()

        self._calculate_initial_molar_phase_fractions()
        self._calculate_initial_overall_substance_fractions()
        self._check_num_phase_equilibrium_equations()

        ### ISOTHERMAL FLASH
        # compute the initial enthalpy using an isothermal flash
        name = "isothermal_flash"
        eq = self.isothermal_flash_equation()

        equations.update({name: eq})

        isothermal_flash_subsystem["equations"].append("isothermal_flash")
        isothermal_flash_subsystem["vars"].append(self._enthalpy)
        isothermal_flash_subsystem["var_names"].append(self._enthalpy_var)

        self._isothermal_flash_subsystem = isothermal_flash_subsystem
        self.isothermal_flash()

        ### EQUILIBRIUM CALCULATIONS
        # num_substances overall fraction equations
        subset = self.overall_substance_fraction_equations()
        for c, substance in enumerate(self._present_substances):
            name = "overall_substance_fraction_%s" % (substance.name)
            equations.update({name: subset[c]})
            phase_equilibrium_subsystem["equations"].append(name)

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

        # Adding all dynamically created MergedVariables to the equation manager
        # necessary to register the variables per grid
        self.eq_manager.update_variables_from_merged(self.pressure)
        self.eq_manager.update_variables_from_merged(self.enthalpy)
        self.eq_manager.update_variables_from_merged(self.temperature)
        for phase in self:
            self.eq_manager.update_variables_from_merged(phase.molar_fraction)
            self.eq_manager.update_variables_from_merged(phase.saturation)
            for substance in phase:
                self.eq_manager.update_variables_from_merged(
                    substance.fraction_in_phase(phase.name)
                )
        for substance in self._present_substances:
            self.eq_manager.update_variables_from_merged(substance.overall_fraction)

    # -----------------------------------------------------------------------------------------
    ### Equilibrium and flash calculations
    # -----------------------------------------------------------------------------------------

    def compute_phase_equilibrium(
        self,
        max_iterations: int = 100,
        tol: float = 1.0e-10,
        copy_to_state: bool = False,
        trust_region: bool = False,
        eliminate_unitarity: Optional[Tuple[str, str, str]] = None,
    ) -> bool:
        """Computes the equilibrium using the following equations:
            - overall substance fraction equations (num_substances)
            - fugacity equations (num_substances * (num_phases -1))
            - substance fraction in phase unity equations (num_phases - 1)
            - molar phase fraction unity equation (1)

        Equilibrium for fixed pressure and enthalpy is assumed.

        :param max_iterations: set maximal number for Newton iterations
        :type max_iterations: int
        :param tol: tolerance for Newton residual
        :type tol: float

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
            tol,
            copy_to_state=copy_to_state,
            trust_region=trust_region,
            eliminate_unitarity=eliminate_unitarity,
        )

    def saturation_flash(self, copy_to_state: bool = False) -> None:
        """Performs a saturation flash calculation using phase molar fractions.
        Two different procedures are applied, dependent on the number of present phases.

        For a 2-phase-system, a forward evaluation is done.
        For more than 2 phases, a linear system has to be solved.
        """
        # depending on number of phases, two different procedures
        if self.num_phases == 2:
            self._2phase_saturation_flash(copy_to_state=copy_to_state)
        elif self.num_phases > 2:
            self._multi_phase_saturation_flash(copy_to_state=copy_to_state)
        else:
            raise NotImplementedError("Single-phase saturation flash not implemented.")

    def isenthalpic_flash(
        self,
        max_iterations: int = 100,
        tol: float = 1.0e-10,
        copy_to_state: bool = False,
    ) -> bool:
        """Performs an isenthalpic flash to obtain temperature values.

        :param max_iterations: set maximal number for Newton iterations
        :type max_iterations: int
        :param tol: tolerance for Newton residual
        :type tol: float
        :param copy_to_state: indicate if results should be copied to state
            (by default to iterate)
        :type copy_to_state: bool

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
            tol,
            copy_to_state=copy_to_state,
        )

    def isothermal_flash(self) -> None:
        """Performs an isothermal flash to obtain the molar enthalpy of the composition.

        The "flash" here is a simple forward evaluation of the caloric relation for enthalpy.

        :param max_iterations: set maximal number for Newton iterations
        :type max_iterations: int
        :param tol: tolerance for Newton residual
        :type tol: float
        """

        equ = list()
        # caloric relation for enthalpy
        for phase in self._present_phases:
            # s_e * rho_e(p,h) * h_e(p,h,T)
            equ.append(
                phase.saturation
                * phase.molar_density(
                    self._pressure, self._enthalpy, temperature=self._temperature
                )
                * phase.enthalpy(
                    self._pressure, self._enthalpy, temperature=self._temperature
                )
            )
        # h = (sum_e s_e * rho_e(p,h) * h_e(p,h,T)) / (sum_e sum_e s_e * rho_e(p,h))
        equ = sum(equ) / self.composit_density(temperature=self._temperature)
        h = equ.evaluate(self.dof_manager).val

        dof = self.dof_manager.dof_var([self._enthalpy_var])
        X = np.zeros(self.dof_manager.num_dofs())
        X[dof] = h

        self.dof_manager.distribute_variable(X, variables=[self._enthalpy_var])
        self.dof_manager.distribute_variable(
            X, variables=[self._enthalpy_var], to_iterate=True
        )

    # -----------------------------------------------------------------------------------------
    ### Model equations
    # -----------------------------------------------------------------------------------------

    def overall_substance_fractions_sum(self) -> pp.ad.Operator:
        """Returns 1 equation representing the unity of the overall component fractions.

        sum_c zeta_c - 1 = 0
        """
        # -1
        equation = pp.ad.Array(-1.0 * np.ones(self.gb.num_cells()))

        for substance in self._present_substances:
            # + zeta_c
            equation += substance.overall_fraction

        return equation

    def overall_substance_fraction_equations(self) -> List[pp.ad.MergedVariable]:
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

    def molar_phase_fraction_sum(self) -> pp.ad.Operator:
        """Returns 1 equation representing the unity of molar phase fractions.

        sum_e xi_e -1 = 0
        """
        # -1
        equation = pp.ad.Array(-1.0 * np.ones(self.gb.num_cells()))

        for phase in self:
            # + xi_e
            equation += phase.molar_fraction

        return equation

    def substance_in_phase_sum_equations(self) -> List[pp.ad.Operator]:
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

    def saturation_flash_equations(self) -> List[pp.ad.Operator]:
        """Returns num_phases equations representing the relation between molar phase fractions
        and saturation (volumetric phase fractions).

        xi_e * (sum_j s_j rho_j) - s_e * rho_e = 0
        """
        equations = list()

        for phase in self:
            # xi_e * (sum_j s_j rho_j)
            equation = phase.molar_fraction * self.composit_density()
            # - s_e * rho_e
            equation -= phase.saturation * phase.molar_density(
                self.pressure, self.enthalpy
            )

            equations.append(equation)

        return equations

    def isenthalpic_flash_equation(self) -> pp.ad.Operator:
        """Returns an operator representing the isenthalpic flash equation.

        rho(p,h) * h = sum_e s_e * rho_e(p,h) * h_e(p, T)

        We still express densities in terms of p, h since density is not supposed to change
        during the flash calculation. This leads to solely the phase enthalpies being dependent
        on temperature.
        """
        # rho(p, h, s_e) * h
        equ = self.composit_density() * self._enthalpy

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

    def isothermal_flash_equation(self) -> pp.ad.Operator:
        """Returns an operator representing the isothermal flash equation for calculating the
        global specific molar entropy.

        rho(p,T) * h = sum_e s_e * rho_e(p,T) * h_e(p,T)

        We express phase enthalpies in terms of p, T since they are not supposed to change
        during the flash calculation. This leads to a simpler form of the equation.
        """
        # rho(p, h, s_e) * h
        equ = self.composit_density() * self._enthalpy

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

    def get_permutation_to_local(
        self, variables: Optional[List[str]] = None
    ) -> sps.spmatrix:
        """Returns a permutation matrix grouping together
            1. cell-wise values
            2. face-wise values
            3. node-wise values
        per node, then per edge in grid bucket.

        :param variables: Returns the permutation matrix for a subsystem spanned by
            given variable names
        :type variables: List[str]

        :return: permutation matrix
        :rtype: scipy.sparse.spmatrix
        """
        pass

    # -----------------------------------------------------------------------------------------
    ### private methods
    # -----------------------------------------------------------------------------------------

    def _calculate_initial_molar_phase_fractions(self) -> None:
        """
        Name is self-explanatory.

        These calculations have to be done every time new initial values are set.
        """
        molar_fraction_sum = 0.0

        for phase in self:
            # definition of molar fraction of phase
            molar_fraction = (
                phase.saturation
                * phase.molar_density(self.pressure, self.enthalpy)
                / self.composit_density()
            )
            # evaluate the AD expression and get the values
            # this is a global DOF vector and has therefor many zeros
            molar_fraction = molar_fraction.evaluate(self.dof_manager).val

            molar_fraction_sum += molar_fraction

            dof = self.dof_manager.dof_var([phase.molar_fraction_var])
            X = np.zeros(self.dof_manager.num_dofs())
            X[dof] = molar_fraction

            # distribute values to respective variable
            self.dof_manager.distribute_variable(
                X, variables=[phase.molar_fraction_var]
            )
            self.dof_manager.distribute_variable(
                X, variables=[phase.molar_fraction_var], to_iterate=True
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

        These calculations have to be done every time new initial values are set.
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
                + "\nNeed:\n%s" % (str(self.num_phase_equilibrium_equations))
            )

    def _resolve_composition(self) -> None:
        """Analyzes the composition, i.e. presence of substances in phases.
        Information about substances which are anticipated in multiple phases is stored.

        This method is called internally everytime any new component is added.
        """
        # for given substance (keys), save set of phases containing the substance (values)
        phases_per_substance = dict()
        # unique substance names, over whole composition
        unique_substances = set()
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

    def _subsystem_newton(
        self,
        equations: List[str],
        variables: List[pp.ad.MergedVariable],
        var_names: List[str],
        max_iterations: int,
        eps: float,
        copy_to_state: bool = False,
        trust_region: Optional[bool] = False,
        eliminate_unitarity: Optional[Tuple[str, str, str]] = None,
    ) -> bool:
        """Performs Newton iterations on a specified subset of equations and variables.

        Trust region update:
            If the update would leave the trusted region of 0 and 1 (for fractional variables),
            scales the update such that the largest offset-value gets scaled down to 0 or 1.
            Scales the other uniformly.

        Unity elimination:
            For a given variable symbol x
            (see :data:`~porepy.composite._composite_utils.COMPUTATIONAL_VARIABLES`)
            assumes the constr of type
                sum_i x_i = 1
            for all occurrences of x in `var_names`.
            Constructs respective affine-linear projections and eliminates respective
            columns and rows in the linear system of equations.
            Eliminates the `x_i` fulfilling the given criterion `elimination_criterion`.

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
        :param eps: if true, enforces a trust region of [0,1] for the variables
        :type trust_region: bool
        :param eliminate_unitarity: list of strings containing
            - symbol of eliminated variable
            - name of eliminated unitary equation
            - elimination criterion
        :type eliminate_unitarity: List[str]

        :return: True if successful, False otherwise
        :rtype: bool
        """
        success = False
        trust_region_updates = 0

        A, b = self.eq_manager.assemble_subsystem(equations, variables)
        # print(A.todense())
        # print(np.linalg.cond(A.todense()))
        # print(self.dof_manager.assemble_variable())

        if np.linalg.norm(b) <= eps:
            success = True
            iter_final = 0
        else:
            # get prolongation to global vector
            prolongation = self._prolongation_matrix(variables)

            if eliminate_unitarity:
                # get information about the group of variables for the elimination procedure
                eliminated_var = eliminate_unitarity[0]
                eliminated_eq = eliminate_unitarity[1]
                elimination_criterion = eliminate_unitarity[2]

                # list of vars belonging to the unitarity group
                unitary_vars = [var for var in variables if eliminated_var in var._name]
                # remove the redundant equation
                equations.remove(eliminated_eq)

                # choose which var out of the unitary group to eliminate
                unitary_var_vals = [
                    var.evaluate(self.dof_manager).val for var in unitary_vars
                ]
                avg_val = [np.sum(vals) / len(vals) for vals in unitary_var_vals]

                if elimination_criterion == "min":
                    to_eliminate = avg_val.index(min(avg_val))
                elif elimination_criterion == "max":
                    to_eliminate = avg_val.index(max(avg_val))
                else:
                    raise ValueError(
                        "Unknown criterion '%s' for unitarity elimination."
                        % (str(elimination_criterion))
                    )

                # get variables for which an identity block is to be computed
                other_vars = list(set(variables).difference(set(unitary_vars)))
                # get eliminated var from the group of unitary vars
                eliminated_var = unitary_vars.pop(to_eliminate)

                # construct affine-linear transformation for eliminated unitary variable
                expansion, affine = self._unitary_expansion(
                    unitary_vars, eliminated_var, other_vars
                )

                # get the eliminated equations (unitary constraint): start with identity
                elimination = sps.diags(np.ones(A.shape[0])).tocsr()
                # get indices of not eliminated equations
                not_eliminated = np.ones(b.size, dtype=bool)
                eliminated_idx = self.eq_manager.dofs_per_equation_last_assembled[
                    eliminated_eq
                ]
                not_eliminated[eliminated_idx] = False
                # get projection onto not eliminated equations
                elimination = elimination[not_eliminated]
                # remove the eliminated equation
                A = elimination * A * expansion
                b = elimination * b
                # print(A.todense())

                # the alternative is to re-assemble the rectangular system without the equation
                # Upper way should be faster, though not critical..
                # equations.remove(eliminated_eq)
                # A, b = self.eq_manager.assemble_subsystem(equations, variables)

            for i in range(max_iterations):

                dx = sps.linalg.spsolve(A, b)

                if eliminate_unitarity:
                    # if the unitarity has still to hold, than the update for
                    # the eliminated var has to be the negative of the sum of the other updates
                    dx = expansion * dx

                dX = prolongation * dx
                # Trust Region Update scaling, defaults to 1
                TRU_scaling = 1.0

                # trust region update for values between 0 and 1
                # find values exceeding 1 and values below 0
                # find scaling coefficients so that WHOLE update stays within 0 and 1
                # choose minimal scaling coefficient and scale down update uniformly
                if trust_region:
                    X = self.dof_manager.assemble_variable(
                        variables=var_names, from_iterate=True
                    )

                    X_preliminary = X + dX
                    too_large = X_preliminary > 1.0
                    too_small = X_preliminary < 0.0
                    scale_too_large = 1.0
                    scale_too_small = 1.0

                    if np.any(too_large):
                        max_idx = X_preliminary.argmax()
                        # x + alpha dx = 1 <-> alpha = (1-x)/dx
                        scale_too_large = (1 - X[max_idx]) / dX[max_idx]
                    if np.any(too_small):
                        min_idx = X_preliminary.argmin()
                        # x + beta dx = 0 <-> beta = - x / dx
                        scale_too_small = -X[min_idx] / dX[min_idx]

                    TRU_scaling = min([scale_too_large, scale_too_small])

                    if TRU_scaling < 1.0:
                        trust_region_updates += 1

                self.dof_manager.distribute_variable(
                    TRU_scaling * dX,
                    variables=var_names,
                    additive=True,
                    to_iterate=True,
                )

                A, b = self.eq_manager.assemble_subsystem(equations, variables)
                if eliminate_unitarity:
                    A = A * expansion                    

                if np.linalg.norm(b) <= eps:
                    # setting state to newly found solution
                    X = self.dof_manager.assemble_variable(
                        variables=var_names, from_iterate=True
                    )
                    if copy_to_state:
                        self.dof_manager.distribute_variable(X, variables=var_names)
                    success = True
                    iter_final = i
                    break

        # if not successful, replace iterate values with initial state values
        if not success:
            print("\nComposition failed to solve:\n%s\nwith iterate state\n%s\n"
            % (str(equations), str(self.dof_manager.assemble_variable(from_iterate=True))))
            X = self.dof_manager.assemble_variable()
            self.dof_manager.distribute_variable(X, to_iterate=True)
            iter_final = max_iterations
        # append history entry and delete old ones if necessary
        self.newton_history.append(
            {
                "variables": var_names,
                "iterations": iter_final,
                "trust": trust_region_updates,
                "success": success,
            }
        )
        if len(self.newton_history) > self._max_history:
            self.newton_history.pop(0)

        return success

    def _get_subsystem_dict(self) -> Dict[str, list]:
        """Returns a template for subsystem dictionaries."""
        return {
            "equations": list(),
            "vars": list(),
            "var_names": list(),
        }

    def _prolongation_matrix(
        self, variables: List[pp.ad.MergedVariable]
    ) -> sps.spmatrix:
        """Constructs a prolongation mapping for a subspace of given variables to the
        global vector.
        Credits to EK.

        :param variables: variables spanning the subspace
        :type: :class:`~porepy.ad.MergedVariable`

        :return: prolongation matrix
        :rtype: scipy.sparse.spmatrix
        """
        nrows = self.dof_manager.num_dofs()
        rows = np.unique(
            np.hstack(
                # The use of private variables here indicates that something is wrong
                # with the data structures. TODO..
                [
                    self.dof_manager.grid_and_variable_to_dofs(s._g, s._name)
                    for var in variables
                    for s in var.sub_vars
                ]
            )
        )
        ncols = rows.size
        cols = np.arange(ncols)
        data = np.ones(ncols)

        return sps.coo_matrix((data, (rows, cols)), shape=(nrows, ncols)).tocsr()

    def _unitary_expansion(
        self,
        vars: List[pp.ad.MergedVariable],
        to_eliminate: pp.ad.MergedVariable,
        other_vars: Optional[List[pp.ad.MergedVariable]] = [],
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        """Returns the unitary expansion mapping for variables fulfilling the
        unitary constraint.
        The unitary expansion is an affine-linear mapping from n-1 to n variables which
        fulfill
            sum_i x_i = 1
        The linear part contains the linear combination for the eliminated variable.
        The affine part contains the 1.

        Other variables can be included in the mapping. The map will contain an identity block
        for them.

        NOTE: assumes cell-wise values for x_i

        :param vars: names of variables fulfilling the unitary constraint.
            DOES NOT contain the eliminated one
        :type vars: List[:class:`~porepy.ad.MergedVariable`]
        :param to_eliminate: variable to be eliminated
        :type to_eliminate: :class:`~porepy.ad.MergedVariable`
        :param other_vars: variables for which an identity block is to be included
        :type other_vars: List[:class:`~porepy.ad.MergedVariable`]

        :return: sparse linear matrix containing the linear combination and a vector for
            the affine part of the map
        :rtype: Tuple[scipy.sparse.spmatrix, numpy.ndarray]
        """
        linear_vals = list()
        linear_rows = list()
        linear_cols = list()
        eliminated_dofs = dict()

        affine = np.zeros(self.dof_manager.num_dofs())
        subvars = self.eq_manager._variables_as_list(vars)
        elim_subvars = self.eq_manager._variables_as_list([to_eliminate])
        other_subvars = self.eq_manager._variables_as_list(other_vars)

        # affine part for eliminated variable
        for var in elim_subvars:
            local_dofs = self.dof_manager.grid_and_variable_to_dofs(var._g, var._name)
            affine[local_dofs] = 1.0
            eliminated_dofs[var._g] = local_dofs

        # identity block plus unitary block for unitary variables
        for var in subvars:
            local_dofs = self.dof_manager.grid_and_variable_to_dofs(var._g, var._name)
            num_local_dofs = local_dofs.size

            linear_vals.append(np.ones(num_local_dofs))
            linear_rows.append(local_dofs)
            linear_cols.append(local_dofs)

            linear_vals.append(-np.ones(num_local_dofs))
            linear_rows.append(eliminated_dofs[var._g])
            linear_cols.append(local_dofs)

        # identity block for other variables
        for var in other_subvars:

            local_dofs = self.dof_manager.grid_and_variable_to_dofs(var._g, var._name)
            num_local_dofs = local_dofs.size

            linear_vals.append(np.ones(num_local_dofs))
            linear_rows.append(local_dofs)
            linear_cols.append(local_dofs)

        linear_vals = np.hstack(linear_vals)
        linear_cols = np.hstack(linear_cols)
        linear_rows = np.hstack(linear_rows)
        # slice through and remove rows with only zeros
        linear = sps.coo_matrix((linear_vals, (linear_rows, linear_cols))).tocsr()
        num_non_zeros = np.diff(linear.indptr)
        linear = linear[num_non_zeros != 0]
        # slice through and remove columns with only zeros
        linear = linear.tocsc()
        num_non_zeros = np.diff(linear.indptr)
        linear = linear[:, num_non_zeros != 0]

        # use a global projection to get the properly sliced affine part
        projection = self.eq_manager._column_projection(
            subvars + elim_subvars + other_subvars
        )
        affine = (affine.T * projection).T

        return (linear, affine)

    def _2phase_saturation_flash(self, copy_to_state: bool = True) -> None:
        """Calculates the saturation value assuming phase molar fractions are given.
        Valid for two-phase flow.

        It holds:
            s_i = 1 / (1 + xi_j / (1 - x_j) * rho_i / rho_j) , i != j
        """
        # get reference to phases
        phase1 = self._present_phases[0]
        phase2 = self._present_phases[1]

        # get phase molar fraction values
        xi1 = phase1.molar_fraction.evaluate(self.dof_manager).val
        xi2 = phase2.molar_fraction.evaluate(self.dof_manager).val

        # get density values for given pressure and enthalpy
        rho1 = phase1.molar_density(self.pressure, self.enthalpy).evaluate(
            self.dof_manager
        )
        if isinstance(rho1, pp.ad.Ad_array):
            rho1 = rho1.val
        rho2 = phase2.molar_density(self.pressure, self.enthalpy).evaluate(
            self.dof_manager
        )
        if isinstance(rho2, pp.ad.Ad_array):
            rho2 = rho2.val

        # allocate saturations, size must be the same
        s1 = np.zeros(xi1.size)
        s2 = np.zeros(xi1.size)

        # TODO test sensitivity of this
        phase1_saturated = xi1 == 1.0  # equal to phase2_vanished
        phase2_saturated = xi2 == 1.0  # equal to phase1_vanished

        # calculate only non-saturated cells to avoid division by zero
        # set saturated or "vanishing" cells explicitly to 1., or 0. respectively
        idx = np.logical_not(phase2_saturated)
        xi2_idx = xi2[idx]
        rho1_idx = rho1[idx]
        rho2_idx = rho2[idx]
        s1[idx] = 1.0 / (1.0 + xi2_idx / (1.0 - xi2_idx) * rho1_idx / rho2_idx)
        s1[phase1_saturated] = 1.0
        s1[
            phase2_saturated
        ] = 0.0  # even if initiated as zero array. remove numerical artifacts

        idx = np.logical_not(phase1_saturated)
        xi1_idx = xi1[idx]
        rho1_idx = rho1[idx]
        rho2_idx = rho2[idx]
        s2[idx] = 1.0 / (1.0 + xi1_idx / (1.0 - xi1_idx) * rho2_idx / rho1_idx)
        s2[
            phase1_saturated
        ] = 0.0  # even if initiated as zero array. remove numerical artifacts
        s2[phase2_saturated] = 1.0

        # distribute saturation values to global DOF
        X = np.zeros(self.dof_manager.num_dofs())
        # saturation of phase 1
        dof = self.dof_manager.dof_var([phase1.saturation_var])
        X[dof] = s1
        # saturation of phase 2
        dof = self.dof_manager.dof_var([phase2.saturation_var])
        X[dof] = s2
        self.dof_manager.distribute_variable(
            X, variables=[phase1.saturation_var, phase2.saturation_var], to_iterate=True
        )
        if copy_to_state:
            self.dof_manager.distribute_variable(
                X, variables=[phase1.saturation_var, phase2.saturation_var]
            )

    def _multi_phase_saturation_flash(self, copy_to_state: bool = True) -> None:
        """Calculates the saturation value assuming phase molar fractions are given.
        Valid for flow with at least 3 phases.

        It holds for all i = 1... m, where m is the number of phases:
            1 = sum_{j != i} (1 + rho_j / rho_i * xi_i / (1 - xi_i)) s_j
        """
        # get phases, phase molar fractions (xi) and densities (rho)
        phases = [phase for phase in self._present_phases]
        xi = [phase.molar_fraction.evaluate(self.dof_manager).val for phase in phases]
        rho = list()
        for phase in phases:
            rho_p = phase.molar_density(self.pressure, self.enthalpy).evaluate(
                self.dof_manager
            )
            if isinstance(rho_p, pp.ad.Ad_array):
                rho_p = rho_p.val
            rho.append(rho_p)

        mat_per_eq = list()
        num_cells = self.gb.num_cells()

        # list of indicators per phase, where the phase is fully saturated
        saturated = list()
        # where one phase is saturated, the other vanish
        vanished = [np.zeros(num_cells, dtype=bool) for _ in phases]

        for i in range(self.num_phases):
            # get the DOFS where one phase is fully saturated
            # TODO check sensitivity of this
            saturated_i = xi[i] == 1.0
            saturated.append(saturated_i)

            # store information that other phases vanish at these DOFs
            for j in range(self.num_phases):
                if j == i:
                    # a phase can not vanish and be saturated at the same time
                    continue
                else:
                    # where phase i is saturated, phase j vanishes
                    # Use OR in order to accumulate the bools per i-loop without overwriting
                    vanished[j] = np.logical_or(vanished[j], saturated_i)

        # indicator which DOFs are saturated for the vector of stacked, discrete saturations
        saturated = np.hstack(saturated)
        # indicator which DOFs vanish
        vanished = np.hstack(vanished)
        # all other DOFs are in multiphase regions
        multiphase = np.logical_not(np.logical_or(saturated,vanished))

        # construct the matrix for saturation flash
        # first loop, per block row (equation per phase)
        for i in range(self.num_phases):
            mats = list()
            # second loop, per block column (block per phase per equation)
            for j in range(self.num_phases):
                # diagonal values are zero
                # This matrix is just a placeholder
                if i == j:
                    mats.append(sps.diags([np.zeros(num_cells)]))
                # diagonals of blocks which are not on the main diagonal, are non-zero
                else:
                    denominator = 1 - xi[i]
                    # to avoid a division by zero error, we set it to one
                    # this is arbitrary, but respective matrix entries will be sliced out later
                    # since they correspond to cells where one phase is saturated,
                    # i.e. the respective saturation is 1., the other 0.
                    denominator[denominator == 0.0] = 1.0
                    d = 1.0 + rho[j] / rho[i] * xi[i] / denominator

                    mats.append(sps.diags([d]))

            # rectangular matrix per equation
            mat_per_eq.append(np.hstack(mats))

        # Stack matrices per equation on each other
        # This matrix corresponds to the vector of stacked, discretized saturations per phase
        mat = np.vstack(mat_per_eq)
        # TODO permute DOFS to get a block diagonal matrix. This one has a large band width
        mat = sps.csr_matrix(mat)

        # projection matrix to DOFs in multiphase region
        # start with identity in CSR format
        projection = sps.diags([np.ones(len(multiphase))]).tocsr()
        # slice image of canonical projection out of identity
        projection = projection[multiphase]

        # get sliced system
        rhs = projection * np.ones(num_cells * self.num_phases)
        mat = projection * mat * projection.T

        s = sps.linalg.spsolve(mat.tocsr(), rhs)

        # prolongate the values from the multiphase region to global DOFs
        saturations = projection.T * s
        # set values where phases are saturated or have vanished
        saturations[saturated] = 1.0
        saturations[vanished] = 0.0

        # distribute results to the saturation variables
        X = np.zeros(self.dof_manager.num_dofs())
        var_names = list()
        for i, phase in enumerate(phases):
            dof = self.dof_manager.dof_var([phase.saturation_var])
            X[dof] = saturations[i * num_cells : (i + 1) * num_cells]
            var_names.append(phase.saturation_var)

        self.dof_manager.distribute_variable(X, variables=var_names, to_iterate=True)
        if copy_to_state:
            self.dof_manager.distribute_variable(X, variables=var_names)
