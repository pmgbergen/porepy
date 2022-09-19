"""Contains a class representing a multiphase multicomponent mixture (composition)"""

from __future__ import annotations

import numbers
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from ._composite_utils import COMPUTATIONAL_VARIABLES, create_merged_subdomain_variable

__all__ = ["Composition"]


class Composition:
    """Representation of a composition of multiple components (chemical substances).
    Performs thermodynamically consistent phase stability and equilibrium calculations.

    Whether specific enthalpy or temperature is temporarily a primary, depends on the chosen
    flash procedure.

    The primary variables are
        - pressure,
        - specific enthalpy of the mixture (depending on the flash procedure),
        - temperature of the mixture (depending on the flash procedure),
        - feed fractions per component.
    Primary Variables are assumed to be given. The equilibrium is performed for fixed p-T or
    fixed p-h, and additionally a fixed feed composition.

    Secondary variables are fractions, i.e.
        - molar phase fractions
        - molar component fractions in a phase
        - volumetric phase fractions (saturations)

    While the molar fractions are the actual unknowns in the flash procedure, the saturation
    values are computed once the equilibrium converges using a relation between molar and
    volumetric fractions for phases.

    References to the secondary variables are stored in respective classes representing
    components and phases.

    All variables are stored as :class:`~porepy.ad.MergedVariable` and the whole concept is
    based on the AD framework provided by PorePy.

    The isenthalpic flash and isothermal flash procedure are implemented.
    The persistent variable approach is utilized based on the work of [1,2], and the references
    therein.

    Attributes:
        gb (:class:`~porepy.GridBucket`): domain of computation.
            A composition is defined locally in each cell.
        dof_manager (:class:`porepy.DofManager`): Degree of Freedom manager for the composition
            Use this instance when imbedding the equilibrium calculations in another model.
        eq_manager (:class:`porepy.ad.EquationManager`): Contains the flash equations in form
            of AD operators. Use this instance when imbedding the equilibrium calculations
            in another model.
        pressure (:class:`~porepy.ad.MergedVariable`): the mixture pressure at equilibrium
        specific_enthalpy (:class:`~porepy.ad.MergedVariable`): the specific molar enthalpy
            of the mixture
        temperature (:class:`~porepy.ad.MergedVariable`): the mixture temperature
            at equilibrium
        num_phases (int): number of present phases, resulting from the last flash.
            Before any flash procedure is performed, this number is zero
        num_components (int): number of components. This is a static variable, depending on
            how many components were added to the composition prior to initialization.
        phases (Generator[:class:`~porepy.composite.Phase`]): Can be used to iterate over
            present phases. This order is also used internally for e.g., global DOFs.
            *Use this generator to iterate over phases outside of this class*.
            *Use this generator to for ordering passed initial values*.
        components (Generator[:class:`~porepy.composite.Component`]): Analogous to the
            generator for phases.
            Only the components are static and the order corresponds to the order
            in which the components where added prior to initialization.
        flash_tolerance (float): convergence criterion for the flash algorithm
        max_iter_flash (int): maximal number of iterations tolerated for the flash algorithm
        pT_subsystem (dict): a specially structured dictionary containing information about
            variables and equations necessary for the isothermal flash. Created after
            :meth:`initialize`
        pT_subsystem (dict): a specially structured dictionary containing information about
            variables and equations necessary for the isenthalpic flash. Created after
            :meth:`initialize`

    References:
        [1] Lauser, A. et. al.:
            A new approach for phase transitions in miscible multi-phase flow in porous media
            DOI: 10.1016/j.advwatres.2011.04.021
        [2] Ben Gharbia, I. et. al.:
            An analysis of the unified formulation for the equilibrium problem of
            compositional multiphase mixture
            DOI: 10.1051/m2an/2021075

    Notes:
        :caption: Implementation
        - Equilibrium calculations can be done cell-wise.
          I.e. the computations can be done smarter or parallelized. This is not yet exploited.
        - number of cells is assumed to be fixed and computed only once at instantiation.
        - Currently the first phase added will be used as the reference phase, keep this in
          mind when assembling the composition. It might have numeric implications.

    """

    def __init__(self, mdg: pp.MixedDimensionalGrid) -> None:
        """
        Args:
            mdg: A grid bucket representing the geometric domain.
                Equilibrium calculations are performed locally, i.e. per cell.

        """
        # public attributes
        self.mdg: pp.MixedDimensionalGrid = mdg
        self.dof_manager: pp.DofManager = pp.DofManager(mdg)
        self.eq_manager: pp.ad.EquationManager = pp.ad.EquationManager(
            mdg, self.dof_manager
        )
        # contains chronologically information about past flash procedures
        self.flash_history: List[Dict[str, Any]] = list()
        # convergence criterion for the flash algorithm
        self.flash_tolerance: float = 1e-8
        # maximal number of iterations for the flash algorithm
        self.max_iter_flash: int = 1000
        # phase equilibrium equations per component
        self.equilibrium_equations: Dict[str, Dict[str, pp.ad.Operator]] = dict()
        # dictionary representing the subsystem for the p-h flash
        self.ph_subsystem: dict = dict()
        # dictionary representing the subsystem for the p-T flash
        self.pT_subsystem: dict = dict()

        # private attributes
        # primary variables
        self._p_var: str = COMPUTATIONAL_VARIABLES["pressure"]
        self._h_var: str = COMPUTATIONAL_VARIABLES["enthalpy"]
        self._T_var: str = COMPUTATIONAL_VARIABLES["temperature"]
        self._p: pp.ad.MergedVariable = create_merged_subdomain_variable(
            mdg, {"cells": 1}, self._p_var
        )
        self._h: pp.ad.MergedVariable = create_merged_subdomain_variable(
            mdg, {"cells": 1}, self._h_var
        )
        self._T: pp.ad.MergedVariable = create_merged_subdomain_variable(
            mdg, {"cells": 1}, self._T_var
        )
        

        # composition
        self._components: List[pp.composite.Component] = list()
        self._phases: List[pp.composite.Phase] = list()
        self._phases_of_component: Dict[str, List[pp.composite.Phase]] = dict()

        # other
        # maximal number of flash history entries (FiFo)
        self._max_history: int = 100
        # this is set true once a proper feed composition was set
        self._feed_composition_set: bool = False
        # Number of cells. The equilibrium is computed cell-wise.
        self._nc = self.mdg.num_subdomain_cells()

        # names of equations
        self._mass_conservation: str = "flash_mass"
        self._phase_fraction_unity: str = "flash_phase_unity"
        self._complementary: str = "flash_complementary"  # complementary conditions
        self._enthalpy_constraint: str = "flash_h_constraint"  # for p-h flash
        # complementary condition tuple TODO better solution...
        self._complementary_eq: dict = dict()

    def __str__(self) -> str:
        """Returns string representation of the composition,
        with information about present components.

        """
        out = "Composition with %s components:" % (str(self.num_components))
        for name in [component.name for component in self.components]:
            out += "\n" + name
        return out

    @property
    def num_components(self) -> int:
        """
        Returns:
            int: number of components in the composition

        """
        return len(self._components)

    @property
    def num_phases(self) -> int:
        """
        Returns:
            int: number of currently present phases

        """
        return len(self._phases)

    @property
    def num_equilibrium_equations(self) -> Dict[str, int]:
        """Returns a dictionary containing the number of necessary phase equilibrium equations
        per component.

        The equation has to be formulated with respect to the reference phase.

        The number is based on the number of phases in which a substance is anticipated i.e.,
        if substance c is present in n_p(c) phases, we need n_p(c) - 1 equilibrium equations,
        since one phase is chosen as a reference phase.

        Returns: dictionary with component names as keys and number of necessary equations as
        values.
        """
        equ_nums = dict()

        for component in self._components:
            equ_nums.update(
                {component.name: len(self._phases_of_component[component.name]) - 1}
            )

        return equ_nums

    @property
    def components(self) -> Generator[pp.composite.Component, None, None]:
        """
        Returns:
            Generator: Returns an iterable object over all components in the composition.

        """
        for C in self._components:
            yield C

    @property
    def phases(self) -> Generator[pp.composite.Phase, None, None]:
        """
        Returns:
            Generator: Returns an iterable object over all anticipated phases.

        """
        for P in self._phases:
            yield P

    @property
    def pressure(self) -> pp.ad.MergedVariable:
        """Initialized with 1 atm (101.325 kPa).

        Math. Dimension:        scalar
        Phys. Dimension:        [kPa] = [kN / m^2]

        Returns:
            :class:`~porepy.ad.MergedVariable`:
                the primary variable pressure on the whole domain,
                assumed to represent values at equilibrium.

        """
        return self._p

    @property
    def specific_enthalpy(self) -> pp.ad.MergedVariable:
        """Initialized with zero globally.

        Math. Dimension:        scalar
        Phys. Dimension:        [kJ / mol / K]

        Returns:
            :class:`~porepy.ad.MergedVariable`:
                the primary variable specific molar enthalpy on the whole domain,
                assumed to represent values at equilibrium.

        """
        return self._h

    @property
    def temperature(self) -> pp.ad.MergedVariable:
        """Temperature of the composition. Given per cell.

        Math. Dimension:        scalar
        Phys. Dimension:        [K]

        Returns:
            :class:`~porepy.ad.MergedVariable`:
                the primary variable temperature on the whole domain,
                assumed to represent values at equilibrium.

        """
        return self._T

    def density(
        self, prev_time: Optional[bool] = False
    ) -> Union[pp.ad.Operator, Literal[0]]:
        """
        Args:
            prev_time (bool): indicator to use values from the previous time step,
                as provided by the AD framework

        Returns:
            :class:`porepy.ad.Operator`:
                Returns the overall molar density of the composition
                given by the saturation-weighted sum of all phase densities.
                The phase densities are computed using the current temperature and pressure
                values.

        """
        # creating a list of saturation-weighted phase densities
        # If the value from the previous time step (STATE) is requested,
        # we do so using the functionality of the AD framework
        if prev_time:
            rho = [
                phase.saturation.previous_timestep()
                * phase.density(
                    self.pressure.previous_timestep(),
                    self.temperature.previous_timestep(),
                )
                for phase in self._phases
            ]
        else:
            rho = [
                phase.saturation * phase.density(self.pressure, self.temperature)
                for phase in self._phases
            ]
        # summing the elements of the list results in the mixture density
        return sum(rho)

    def add_component(
        self, component: Union[List[pp.composite.Component], pp.composite.Component]
    ) -> None:
        """Adds components to the composition. Adding or removing components invalidates the
        last, computed equilibrium.

        Args:
            component (:class:`porepy.composite.Component`): a component,
                or list of components, which are modelled in this mixture.

        """
        if isinstance(component, pp.composite.Component):
            component = [component]

        added_components = [comp.name for comp in self._components]

        for comp in component:

            if comp.name in added_components:
                # already added components are skipped
                continue
            else:
                self._components.append(comp)
                self.equilibrium_equations.update({comp.name: dict()})

        self._feed_composition_set = False

    def add_phase(
        self, phases: Union[List[pp.composite.Phase], pp.composite.Phase]
    ) -> None:
        """Adds one or multiple phases which are anticipated in the current model.

        As of now, the modeler has to anticipate every phase which might appear or disappear.
        A phase is only allowed to contain components, which are known to the composition.

        Note:
            The first phase added will be used as the reference phase (unitarity).

        Args:
            phases (list): a phase or a list of phases anticipated in this composition.
        """

        if isinstance(phases, pp.composite.Phase):
            phases = [phases]

        added_phases = {phase.name for phase in self._phases}
        # check if phase is instantiated on same domain or
        # if its name is already among the present phases
        for phase in phases:

            if phase.name in added_phases:
                # Already added phases are simply skipped
                continue
            else:
                # if 'empty' phases are passed, raise an error
                if phase.num_components == 0:
                    raise RuntimeError("Phases without components are nonphysical.")

                # check if the components are known
                for component in phase:
                    if component not in self._components:
                        raise ValueError(
                            f"Unknown component {component.name} in phase {phase.name}"
                        )
                # add to list of anticipated phases
                self._phases.append(phase)

        self._resolve_composition()

    def add_equilibrium_equation(
        self,
        component: pp.composite.Component,
        equation: pp.ad.Operator,
        equ_name: str,
    ) -> None:
        """Adds a phase equilibrium equation to the flash.

        This is modularized and for now it is completely up to the modeler to assure
        a non-singular system of equations.

        Notes:
            - Make sure the equation is such, that the right-hand-side is zero and the passed
            operator represents the left-hand-side
            - Make sure the name is unique. Unknown behavior else.
            - Make sure the equilibrium is formulated with respect to the reference phase

        For the number of necessary equations see :method:`num_equilibrium_equations`.

        Args:
            component: component for which the equilibrium equation is intended
            equation: A general AD operator representing the left-hand side of the equation
            equ_name: A name for the equation, used as an identifier

        """

        if component in self._components:
            raise_error = False
            eq_num_max = self.num_equilibrium_equations[component.name]
            eq_num_is = len(self.equilibrium_equations[component.name])
            # if not enough equations available, add it
            if eq_num_is < eq_num_max:
                self.equilibrium_equations[component.name].update({equ_name: equation})
            # if enough are available, check if one is replaceable (same name)
            elif eq_num_is == eq_num_max:
                if equ_name in self.equilibrium_equations[component.name].keys():
                    self.equilibrium_equations[component.name].update(
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
                    + "for component %s (%i) exceeded." % (component.name, eq_num_max)
                )
        else:
            raise ValueError(f"Component {component.name} not present in composition.")

    def set_feed_composition(self, feed: List[Union[float, np.ndarray]]) -> None:
        """Set the feed fraction per component.
        Fractions can be passed homogeneously (float) or heterogeneously
        (array, float per cell) for each present component.

        Args:
            feed (ArrayLike): A list of floats or numpy arrays per component with fractions.
                If a float is passed for a components, a homogeneous distribution is assumed.
                Use :meth:`~Composition.components` for the order of fractions in ``feed``.

        Raises:
            ValueError:
                If the length of argument ``feed`` does not match the number of components.
            ValueError:
                If the feed fractions do not sum up to 1 on each cell.
            ValueError:
                If a feed in form of an array has not enough values (number of cells)

        """

        if len(feed) != self.num_components:
            raise ValueError(
                f"{len(feed)} fraction given, but {self.num_components} components present."
            )

        fraction_sum = np.zeros(self._nc)
        X = np.zeros(self.dof_manager.num_dofs())
        var_names = list()

        for fraction, component in zip(feed, self.components):
            if isinstance(fraction, numbers.Real):
                fraction = fraction * np.ones(self._nc)
            else:
                if len(fraction) != self._nc:
                    raise ValueError(
                        f"Array-like feed has {len(fraction)} entries, require {self._nc}."
                    )
            fraction_sum += fraction

            dof = self.dof_manager.dof_var([component.fraction_var])
            X[dof] = fraction
            var_names.append(component.fraction_var)

        if not np.allclose(fraction_sum, 1.0):
            raise ValueError("Sum of feed fraction does not fulfill unity.")

        self.dof_manager.distribute_variable(X, variables=var_names, to_iterate=True)
        self.dof_manager.distribute_variable(X, variables=var_names)
        self._feed_composition_set = True

    def set_state(self, p: Union[float, np.ndarray], T: Union[float, np.ndarray]) -> None:
        """Sets the thermodynamic state of the composition in terms of pressure and temperature
        at equilibrium.

        Args:
            p (ArrayLike, number): Pressure
            T (ArrayLike, number): Temperature

        """

        var_names = [self._p_var, self._T_var]
        X = np.zeros(self.dof_manager.num_dofs())

        if isinstance(p, numbers.Real):
            p = p * np.ones(self._nc)
        if isinstance(T, numbers.Real):
            T = T * np.ones(self._nc)

        if len(p) != self._nc:
            raise ValueError(
                f"Array-like 'p' has {len(p)} entries, require {self._nc}."
            )
        if len(T) != self._nc:
            raise ValueError(
                f"Array-like 'T' has {len(T)} entries, require {self._nc}."
            )

        dof = self.dof_manager.dof_var([self._p_var])
        X[dof] = p
        dof = self.dof_manager.dof_var([self._T_var])
        X[dof] = T
        self.dof_manager.distribute_variable(X, variables=var_names, to_iterate=True)
        self.dof_manager.distribute_variable(X, variables=var_names)

    def print_last_flash(self) -> None:
        """Prints the result of the last flash calculation."""
        entry = self.flash_history[-1]
        msg = "\nProcedure: %s\n" % (str(entry["flash"]))
        msg += "SUCCESS: %s\n" % (str(entry["success"]))
        msg += "Method: %s\n" % (str(entry["method"]))
        msg += "Remarks: %s" % (str(entry["other"]))

    def initialize(self) -> None:
        """Initializes the flash equations for this system, based on the added components,
        phases and equilibrium equations.

        Raises:
            RuntimeError: If the system is not closed with a sufficient number of
                equilibrium equations
        """
        # check the system closure, i.e. if enough equilibrium equations are provided
        missing_num = 0
        for component in self._components:
            # should-be-number
            equ_num = self.num_equilibrium_equations[component.name]
            # summing discrepancy
            missing_num += equ_num - len(self.equilibrium_equations[component.name])

        if missing_num > 0:
            raise RuntimeError(
                "Missing %i phase equilibria equations to initialize the composition."
                % (missing_num)
                + "\nNeed: \n%s" % (str(self.num_equilibrium_equations))
            )

        # at this point we assume all DOFs are defined and we reset the following
        # to get a complete DOF mapping including all variables
        self.dof_manager = pp.DofManager(self.mdg)
        self.eq_manager = pp.ad.EquationManager(self.mdg, self.dof_manager)
        # allocating place for the subsystem
        equations = dict()
        pT_subsystem: Dict[str, list] = self._get_subsystem_dict()
        ph_subsystem: Dict[str, list] = self._get_subsystem_dict()
        self._set_subsystem_vars(ph_subsystem, pT_subsystem)

        ### Mass conservation equations
        eqs = self.mass_conservation_equations()
        for c, component in enumerate(self._components):

            name = "%s_%s" % (self._mass_conservation, component.name)

            equations.update({name: eqs[c]})
            pT_subsystem["equations"].append(name)
            ph_subsystem["equations"].append(name)

        ### equilibrium equations
        for component in self.components:
            for equ_name in self.equilibrium_equations[component.name]:
                equ = self.equilibrium_equations[component.name][equ_name]
                equations.update({equ_name: equ})
                pT_subsystem["equations"].append(equ_name)
                ph_subsystem["equations"].append(equ_name)

        ### phase fraction unity (fraction of reference equation)
        # TODO this can be done in post-processing, add reference phase fraction to sec vars
        equ = self.phase_fraction_equation()
        equations.update({self._phase_fraction_unity: equ})
        pT_subsystem["equations"].append(self._phase_fraction_unity)
        ph_subsystem["equations"].append(self._phase_fraction_unity)

        ### enthalpy constraint for p-H flash
        equations.update({self._enthalpy_constraint: self.specific_enthalpy_equation()})
        ph_subsystem["equations"].append(self._enthalpy_constraint)

        ### Semi-smooth complementary conditions per phase
        eqs = self.phase_composition_equations()
        self._complementary_eq = dict()
        # the first complementary condition will be eliminated by the phase fraction unity
        for phase, equation in zip(self._phases[1:], eqs[1:]):
            equ_name = "%s_%s" % (self._complementary, phase.name)

            self._complementary_eq.update({equ_name: (phase.fraction, equation)})
            pT_subsystem["equations"].append(equ_name)
            ph_subsystem["equations"].append(equ_name)

        self.eq_manager.equations = equations # TODO conflict between MergedVar and vars stored in eq manager
        self.pT_subsystem = pT_subsystem
        self.ph_subsystem = ph_subsystem

    # -----------------------------------------------------------------------------------------
    ### Flash methods
    # -----------------------------------------------------------------------------------------

    def isothermal_flash(
        self, copy_to_state: bool = True, initial_guess: str = "iterate"
    ) -> bool:
        """Isothermal flash procedure to determine the composition based on given
        temperature of the mixture, pressure and feed fraction per component.

        Args:
            copy_to_state (bool): Copies the values to the STATE of the AD variables,
                additionally to ITERATE.
            initial_guess ({'iterate', 'feed', 'uniform'}): strategy for choosing the initial
                guess:
                    - ``iterate``: values from ITERATE or STATE, if ITERATE not existent,
                    - ``feed``: feed composition values are used as initial guesses
                    - ``uniform``: uniform fractions adding up to 1 are used as initial guesses

        Returns:
            indicator if flash was successful or not. If not successful, the ITERATE will 
            **not** be copied to the STATE, even if flagged ``True`` by ``copy_to_state``.
        """
        success = self._Newton_min(self.pT_subsystem, copy_to_state, initial_guess)
        if success:
            self._renormalize_absent_phase_composition(copy_to_state)
        else: # if not successful, we re-normalize only the iterate # if not successful, we re-normalize only the iterate
            self._renormalize_absent_phase_composition(False)
        return success

    def isenthalpic_flash(
        self, copy_to_state: bool = True, initial_guess: str = "iterate"
    ) -> bool:
        """Isenthalpic flash procedure to determine the composition based on given
        specific enthalpy of the mixture, pressure and feed fractions per component.

        Args:
            copy_to_state (bool): Copies the values to the STATE of the AD variable,
                additionally to ITERATE.
            initial_guess ({'iterate', 'feed', 'uniform'}): strategy for choosing the initial
                guess:
                    - ``iterate``: values from ITERATE or STATE, if ITERATE not existent,
                    - ``feed``: feed composition values are used as initial guesses
                    - ``uniform``: uniform fractions adding up to 1 are used as initial guesses

        Returns:
            indicator if flash was successful or not. If not successful, the ITERATE will 
            **not** be copied to the STATE, even if flagged ``True`` by ``copy_to_state``.
        """
        success = self._Newton_min(self.ph_subsystem, copy_to_state, initial_guess)
        if success:
            self._renormalize_absent_phase_composition(copy_to_state)
        else: # if not successful, we re-normalize only the iterate
            self._renormalize_absent_phase_composition(False)
        return success

    def evaluate_saturations(self, copy_to_state: bool = True) -> None:
        """Assuming molar phase fractions, pressure and temperature are given (and correct),
        evaluates the volumetric phase fractions (saturations) based on the number of present
        phases.
        If no phases are present (e.g. before any flash procedure), this method does nothing.

        Notes:
            It is enough to call this method once after the (any) flash procedure converged.

        Args:
            copy_to_state (bool): Copies the values to the STATE of the AD variable,
                additionally to ITERATE.

        """
        if len(self._phases) == 1:
            self._single_phase_saturation_evaluation(copy_to_state)
        if len(self._phases) == 2:
            self._2phase_saturation_evaluation(copy_to_state)
        elif len(self._phases) >= 3:
            self._multi_phase_saturation_evaluation(copy_to_state)

    def evaluate_specific_enthalpy(self, copy_to_state: bool = True) -> None:
        """Based on current pressure, temperature and phase fractions, evaluates the
        specific molar enthalpy. Use with care, if the equilibrium problem is coupled with
        e.g., the flow.

        Args:
            copy_to_state (bool): Copies the values to the STATE of the AD variable,
                additionally to ITERATE.

        """

        # obtain values by forward evaluation
        equ_ = list()
        for phase in self.phases:
            equ_.append(phase.fraction * phase.specific_enthalpy(self._p, self._T))
        equ = sum(equ_)

        # if no phase present (list empty) zero is returned and enthalpy is zero
        if equ == 0:
            h = np.zeros(self._nc)
        # else evaluate this operator
        elif isinstance(equ, pp.ad.Operator):
            h = equ.evaluate(self.dof_manager).val
        else:
            raise RuntimeError("Something went terribly wrong.")

        # insert values in global dof vector
        X = np.zeros(self.dof_manager.num_dofs())
        dof = self.dof_manager.dof_var([self._h_var])
        X[dof] = h

        self.dof_manager.distribute_variable(
            X, variables=[self._h_var], to_iterate=True
        )
        if copy_to_state:
            self.dof_manager.distribute_variable(X, variables=[self._h_var])

    # -----------------------------------------------------------------------------------------
    ### Model equations
    # -----------------------------------------------------------------------------------------

    def mass_conservation_equations(self) -> List[pp.ad.Operator]:
        """Returns ``num_components`` equations representing the definition of the
        overall component fraction (mass conservation) per component.

        zeta_c - \sum_e chi_ce * y_e = 0

        The order of equations is given by the iterator :meth:`components`.

        Returns: list of AD operators representing the left-hand side of the equations (rhs=0).
        """
        equations = list()

        for component in self.components:
            # zeta_c
            equation = component.fraction

            for phase in self._phases_of_component[component.name]:
                # - xi_e * chi_ce
                equation -= phase.fraction * component.fraction_in_phase(phase.name)
            # equations per substance
            equations.append(equation)

        return equations

    def phase_fraction_equation(self) -> pp.ad.Operator:
        """Returns the equation representing the phase fraction unity

        1 - sum_phases y_e = 0

        Returns: AD operator representing the left-hand side of the equation (rhs=0).
        """
        equ = pp.ad.Scalar(1.0)

        for phase in self.phases:
            equ -= phase.fraction

        return equ

    def specific_enthalpy_equation(self) -> pp.ad.Operator:
        """Returns an operator representing the specific molar enthalpy of the composition,
        based on it's definition:

        h - sum_phases phase.fraction * phase.specific_enthalpy(p,T) = 0

        Can be used to for the p-h flash as enthalpy constraint (T is an additional variable).
        This is for a simple, p-T-based evaluation. Can be used for an initial guess or the
        final computation after the p-T-flash.

        """

        equ = list()
        for phase in self.phases:
            equ.append(phase.fraction * phase.specific_enthalpy(self._p, self._T))

        return self._h - sum(equ)

    def phase_composition_equations(self) -> List[pp.ad.Operator]:
        """For all phases it holds

        1 - sum chi_ce = 0 , where the sum goes over components c present in phase e

        Returns: list of AD operators representing the phase composition unity per phase. The
        order matches the iterator :meth:`phases`.
        """
        equations = list()

        for phase in self.phases:

            eq = pp.ad.Scalar(1.0)

            for component in phase:
                eq -= component.fraction_in_phase(phase.name)

            equations.append(eq)

        return equations

    # -----------------------------------------------------------------------------------------
    ### Flash methods
    # -----------------------------------------------------------------------------------------

    def _Newton_min(self, subsystem: dict, copy_to_state: bool, initial_guess: str) -> bool:
        """Performs a semi-smooth newton (Newton-min), where the complementary conditions are
        the semi-smooth part.

        Args:
            subsystem: specially structured dict containing information about vars and equs.
            copy_to_state: flag to save the result as STATE, additionally to ITERATE.

        Returns: a bool indicating the success of the method.

        """
        success = False
        vars = subsystem["primary_vars"]
        var_names = subsystem["primary_var_names"]
        if self._T_var in var_names:
            flash_type = "isenthalpic"
        else:
            flash_type = "isothermal"
        # separating smooth and non-smooth parts
        equations = set(subsystem["equations"])
        complementary_cond = set(self._complementary_eq.keys())
        equations = equations.difference(complementary_cond)

        if flash_type == "isenthalpic":
            self._set_initial_guess(initial_guess, True)
        else:
            self._set_initial_guess(initial_guess)

        # assemble linear system of eq for semi-smooth subsystem
        A, b = self._assemble_semi_smooth_system(equations, complementary_cond, vars)

        # if residual is already small enough
        if np.linalg.norm(b) <= self.flash_tolerance:
            success = True
            iter_final = 0
        else:
            # this changes dependent on flash type but also if other models accessed it
            prolongation = self._prolongation_matrix(vars)

            for i in range(self.max_iter_flash):

                # solve iteration and add to ITERATE state additively
                dx = sps.linalg.spsolve(A, b)
                DX = prolongation * dx
                self.dof_manager.distribute_variable(
                    DX,
                    variables=var_names,
                    additive=True,
                    to_iterate=True,
                )
                # counting necessary number of iterations
                iter_final = i + 1  # shift since range() starts with zero

                # assemble new matrix and residual
                A, b = self._assemble_semi_smooth_system(equations, complementary_cond, vars)

                # in case of convergence
                if np.linalg.norm(b) <= self.flash_tolerance:

                    # setting STATE to newly found solution
                    if copy_to_state:
                        X = self.dof_manager.assemble_variable(
                            variables=var_names, from_iterate=True
                        )
                        self.dof_manager.distribute_variable(X, variables=var_names)

                    success = True

                    # compute the inverse Jacobian, for Schur complement with flow
                    # TODO make inverter more efficient (block inverter?)
                    self._last_inverted = np.linalg.inv(A.A)

                    break

        # append history entry
        self._history_entry(
            flash=flash_type,
            method="Newton-min",
                iterations=iter_final,
            success=success,
            variables=[v._name for v in vars],
            equations=equations,
            complementary=complementary_cond,
        )

        return success

    def _assemble_semi_smooth_system(
        self,
        equations: List[pp.ad.Operator],
        complementary_cond: Tuple[pp.ad.Operator],
        vars: List[pp.ad.MergedVariable],
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        """Returns an element of the subgradient of the ``min(-,-)`` function and the
        respective right-hand side of the Newton-min linearized system.

        References:
            - Pang, J.S.: Newton's Method for B-Differentiable Equations
              https://www.jstor.org/stable/3689785

        Args:
            equations: list of operators representing the smooth part of the system
            complementary_cond: a 2-tuple of AD operators representing the arguments for min
            vars: list of variables w.r.t. which the lin. equations should be assembled

        Returns:
            spmatrix: subgradient element w.r.t. to the given variables
            ndarray: residual vector for the current iterate state.
        """
        # assemble smooth subsystem
        A_s, b_s = self.eq_manager.assemble_subsystem(equations, vars)

        # assemble non-smooth subsystem
        all_b_ns = list()
        all_A_ns = list()
        # projection to primary variables
        # TODO this is dirty, since access to private methods
        variable_list = self.eq_manager._variables_as_list(vars)
        projection = self.eq_manager._column_projection(variable_list)

        for comp_cond in complementary_cond:
            op1, op2 = self._complementary_eq[comp_cond]

            b1 = op1.evaluate(self.dof_manager)
            b2 = op2.evaluate(self.dof_manager)
            # see reference for this active-set-strategy
            active_set = (b1.val - b2.val) > 0

            b_ns = -b1.val.copy()
            b_ns[active_set] = -b2.val[active_set]

            A_ns = b1.jac.tolil()
            # TODO scipy.sparse gives an efficiency warning here, told me to change to
            # lil. Is this really the way to go?
            A_ns[active_set] = b2.jac.tolil()[active_set]

            all_b_ns.append(b_ns)
            all_A_ns.append([A_ns.tocsr()])
        
        A_ns = sps.bmat(all_A_ns, format="csr") * projection
        
        # stack smooth and non-smooth part to global semi-smooth system
        A = sps.bmat([[A_s], [A_ns]], format="csr")
        b = np.hstack([b_s] + all_b_ns)

        return A, b

    def _set_initial_guess(self, initial_guess: str, guess_temperature: bool = False) -> None:
        """Auxillary function to set the initial values for phase fractions, phase compositions
        and temperature, based on the chosen strategy.
        """

        if initial_guess != "iterate":
            values = np.zeros(self.dof_manager.num_dofs())
            var_names = list()

            if initial_guess == "feed":
                # 'phase feed' is a temporary value, sum of feeds of present components
                # y_e = sum z_c if c in e
                all_phase_feeds = list()
                # setting the values for phase compositions per phase
                for phase in self.phases:
                    # feed fractions from components present in this phase
                    feed = [
                        comp.fraction.evaluate(self.dof_manager).val for comp in phase
                    ]
                    # this is not necessarily one if not all components are present in phase,
                    # but never zero because empty phases are not accepted
                    phase_feed = sum(feed)
                    all_phase_feeds.append(phase_feed)

                    for component in phase:
                        fraction_in_phase = component.fraction.evaluate(self.dof_manager).val
                        # re-normalize the fraction,
                        # dependent on number of components in this phase.
                        # component fractions have to sum up to 1.
                        fraction_in_phase = fraction_in_phase / phase_feed

                        var_names.append(component.fraction_in_phase_var(phase.name))
                        idx = self.dof_manager.dof_var(var_names[-1])
                        values[idx] = fraction_in_phase

                # by re-normalizing the phase feeds,
                # we obtain an initial guess for the phase fraction.
                # phase fractions have to sum up to 1.
                phase_feed_sum = sum(all_phase_feeds)
                for e, phase in enumerate(self._phases):
                    phase_feed = all_phase_feeds[e]
                    phase_fraction = phase_feed / phase_feed_sum

                    var_names.append(phase.fraction_var)
                    idx = self.dof_manager.dof_var(var_names[-1])
                    values[idx] = phase_fraction

            elif initial_guess == "uniform":
                # uniform values for phase fraction
                val_phases = 1. / self.num_phases
                
                for phase in self.phases:
                    var_names.append(phase.fraction_var)
                    idx = self.dof_manager.dof_var(var_names[-1])
                    values[idx] = np.ones(self._nc) * val_phases

                    # uniform values for composition of this phase
                    val_phase_comp = 1. / phase.num_components
                    for component in phase:
                        var_names.append(component.fraction_in_phase_var(phase.name))
                        idx = self.dof_manager.dof_var(var_names[-1])
                        values[idx] = np.ones(self._nc) * val_phase_comp
            else:
                raise ValueError(f"Unknown initial guess strategy '{initial_guess}'")

            # make variable names unique, just in case
            var_names = list(set(var_names))
            self.dof_manager.distribute_variable(values, variables=var_names, to_iterate=True)

            if guess_temperature:
                # TODO implement
                pass

    def _prolongation_matrix(
        self, variables: List[pp.ad.MergedVariable]
    ) -> sps.spmatrix:
        """Constructs a prolongation mapping for a subspace of given variables to the
        global vector.
        Credits to EK.

        Args:
            variables: variables spanning the subspace

        Returns: sparse prolongation matrix to global DOFs
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

    def _renormalize_absent_phase_composition(self, copy_to_state: bool) -> None:
        """Phase compositions (fractions of components in that phase) are nonphysical if a 
        phase is not present. The unified flash procedure yields nevertheless values, possibly
        violating the unity constraint. Respective fractions have to be re-normalized in a 
        post-processing step.

        Args:
            copy_to_state (bool): Copies the values to the STATE of the AD variable,
                additionally to ITERATE.

        """
        X = np.zeros(self.dof_manager.num_dofs())
        var_names = list()

        for phase in self.phases:
            phase_composition = list()
            phase_frac = phase.fraction.evaluate(self.dof_manager).val
            
            # remove numerical artifacts
            phase_frac[phase_frac < 0.] = 0.
            phase_frac[phase_frac > 1.] = 1.
            var_names.append(phase.fraction_var)
            idx = self.dof_manager.dof_var(var_names[-1])
            X[idx] = phase_frac

            for comp in phase:
                comp_frac = comp.fraction_in_phase(phase.name).evaluate(self.dof_manager).val
                phase_composition.append(comp_frac)
            
            comp_sum = sum(phase_composition)

            for c, comp in enumerate(phase):
                # re-normalize everything.
                # DOFS where unity is already fulfilled remain unchanged.
                comp_frac = phase_composition[c]
                comp_frac = comp_frac / comp_sum
                # remove numerical artifacts
                comp_frac[comp_frac < 0.] = 0.
                comp_frac[comp_frac > 1.] = 1.

                var_names.append(comp.fraction_in_phase_var(phase.name))
                idx = self.dof_manager.dof_var(var_names[-1])
                X[idx] = comp_frac

        # make variable names unique, just in case
        var_names = list(set(var_names))
        self.dof_manager.distribute_variable(X, variables=var_names, to_iterate=True)
        if copy_to_state:
            self.dof_manager.distribute_variable(X, variables=var_names)

    # -----------------------------------------------------------------------------------------
    ### other private methods
    # -----------------------------------------------------------------------------------------

    def _history_entry(
        self,
        flash: str = "isenthalpic",
        method: str = "standard",
        iterations: int = 0,
        success: bool = False,
        variables: List[str] = list(),
        equations: List[str] = list(),
        **kwargs,
    ) -> None:
        """Makes an entry in the flash history"""

        self.flash_history.append(
            {
                "flash": flash,
                "method": method,
                "iterations": iterations,
                "success": success,
                "variables": str(variables),
                "equations": str(equations),
                "other": str(kwargs),
            }
        )
        if len(self.flash_history) > self._max_history:
            self.flash_history.pop(0)

    def _get_subsystem_dict(self) -> Dict[str, list]:
        """Returns a template for subsystem dictionaries."""
        return {
            "equations": list(),
            "primary_vars": list(),
            "primary_var_names": list(),
            "secondary_vars": list(),
            "secondary_var_names": list(),
        }

    def _resolve_composition(self) -> None:
        """Resolves the composition by storing which component is anticipated in which phase.
        """
        phases_of_component: Dict[str, list] = dict()

        for component in self._components:
            # allocate the list of phases for this component
            if component.name not in phases_of_component.keys():
                phases_of_component.update({component.name: list()})

            # check in each phase if this component is anticipated
            for phase in self._phases:
                if component in phase:
                    phases_of_component[component.name].append(phase)

        self._phases_of_component = phases_of_component

    def _set_subsystem_vars(
        self, ph_subsystem: Dict[str, list], pT_subsystem: Dict[str, list]
    ) -> None:
        """Auxiliary function to set the variables in respective subsystems."""
        
        # pressure is always a secondary var in the flash
        pT_subsystem["secondary_vars"].append(self._p)
        pT_subsystem["secondary_var_names"].append(self._p_var)
        ph_subsystem["secondary_vars"].append(self._p)
        ph_subsystem["secondary_var_names"].append(self._p_var)
        # for the p-H flash, enthalpy is a secondary var
        ph_subsystem["secondary_vars"].append(self._h)
        ph_subsystem["secondary_var_names"].append(self._h_var)
        # for the p-T flash, temperature AND enthalpy are secondary vars,
        # because h can be evaluated for given T and fractions
        pT_subsystem["secondary_vars"].append(self._h)
        pT_subsystem["secondary_var_names"].append(self._h_var)
        pT_subsystem["secondary_vars"].append(self._T)
        pT_subsystem["secondary_var_names"].append(self._T_var)
        # feed fractions are always secondary vars
        for component in self.components:
            pT_subsystem["secondary_vars"].append(component.fraction)
            pT_subsystem["secondary_var_names"].append(component.fraction_var)
            ph_subsystem["secondary_vars"].append(component.fraction)
            ph_subsystem["secondary_var_names"].append(component.fraction_var)
        # saturations are always secondary vars
        for phase in self.phases:
            pT_subsystem["secondary_vars"].append(phase.saturation)
            pT_subsystem["secondary_var_names"].append(phase.saturation_var)
            ph_subsystem["secondary_vars"].append(phase.saturation)
            ph_subsystem["secondary_var_names"].append(phase.saturation_var)
    
        # primary vars which are same for both subsystems
        # phase fractions
        for phase in self.phases:
            pT_subsystem["primary_vars"].append(phase.fraction)
            pT_subsystem["primary_var_names"].append(phase.fraction_var)
            ph_subsystem["primary_vars"].append(phase.fraction)
            ph_subsystem["primary_var_names"].append(phase.fraction_var)
        # phase composition
        for component in self.components:
            for phase in self._phases_of_component[component.name]:
                var = component.fraction_in_phase(phase.name)
                var_name = component.fraction_in_phase_var(phase.name)
                pT_subsystem["primary_vars"].append(var)
                pT_subsystem["primary_var_names"].append(var_name)
                ph_subsystem["primary_vars"].append(var)
                ph_subsystem["primary_var_names"].append(var_name)
        # for the p-h flash, T is an additional var
        ph_subsystem["primary_vars"].append(self._T)
        ph_subsystem["primary_var_names"].append(self._T_var)

    def _single_phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """If only one phase is present, we assume it occupies the whole pore space."""

        phase = self._phases[0]
        X = np.zeros(self.dof_manager.num_dofs())
        # saturation is 1
        dof = self.dof_manager.dof_var([phase.saturation_var])
        X[dof] = 1.0

        self.dof_manager.distribute_variable(
            X, variables=[phase.saturation_var], to_iterate=True
        )
        if copy_to_state:
            self.dof_manager.distribute_variable(X, variables=[phase.saturation_var])

    def _2phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """Calculates the saturation value assuming phase molar fractions are given.
        In the case of 2 phases, the evaluation is straight forward.

        It holds:
            s_i = 1 / (1 + xi_j / (1 - x_j) * rho_i / rho_j) , i != j
        """
        # get reference to phases
        phase1 = self._phases[0]
        phase2 = self._phases[1]

        # get phase molar fraction values
        xi1 = phase1.fraction.evaluate(self.dof_manager).val
        xi2 = phase2.fraction.evaluate(self.dof_manager).val

        # get density values for given pressure and enthalpy
        rho1 = phase1.density(self._p, self._T).evaluate(self.dof_manager)
        if isinstance(rho1, pp.ad.Ad_array):
            rho1 = rho1.val
        rho2 = phase2.density(self._p, self._T).evaluate(self.dof_manager)
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

    def _multi_phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """Calculates the saturation value assuming phase molar fractions are given.
        Valid for compositions with at least 3 phases.
        In this case a linear system has to be solved for each multiphase cell

        It holds for all i = 1... m, where m is the number of phases:
            1 = sum_{j != i} (1 + rho_j / rho_i * xi_i / (1 - xi_i)) s_j
        """
        # get phases, phase molar fractions (xi) and densities (rho)
        phases = [phase for phase in self._phases]
        xi = [phase.fraction.evaluate(self.dof_manager).val for phase in phases]
        rho = list()
        for phase in phases:
            rho_p = phase.density(self._p, self._T).evaluate(self.dof_manager)
            if isinstance(rho_p, pp.ad.Ad_array):
                rho_p = rho_p.val
            rho.append(rho_p)

        mat_per_eq = list()

        # list of indicators per phase, where the phase is fully saturated
        saturated = list()
        # where one phase is saturated, the other vanish
        vanished = [np.zeros(self._nc, dtype=bool) for _ in phases]

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
        multiphase = np.logical_not(np.logical_or(saturated, vanished))

        # construct the matrix for saturation flash
        # first loop, per block row (equation per phase)
        for i in range(self.num_phases):
            mats = list()
            # second loop, per block column (block per phase per equation)
            for j in range(self.num_phases):
                # diagonal values are zero
                # This matrix is just a placeholder
                if i == j:
                    mats.append(sps.diags([np.zeros(self._nc)]))
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
        rhs = projection * np.ones(self._nc * self.num_phases)
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
            X[dof] = saturations[i * self._nc : (i + 1) * self._nc]
            var_names.append(phase.saturation_var)

        self.dof_manager.distribute_variable(X, variables=var_names, to_iterate=True)
        if copy_to_state:
            self.dof_manager.distribute_variable(X, variables=var_names)
