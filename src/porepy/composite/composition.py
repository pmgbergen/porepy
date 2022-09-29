"""Contains a class representing a multiphase multicomponent mixture (composition)."""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .component import VarLike

__all__ = ["Composition"]


class Composition:
    """Representation of a composition of multiple components (chemical substances).
    Performs thermodynamically consistent phase stability and equilibrium calculations.

    If a :class:`~porepy.ad.ADSystemManager` is passed at instantiation, the AD framework is
    is used to represent the physical state i.e., pressure, temperature/enthalpy.
    It is assumed that added components and phases are instantiated using the same AD system.

    If no AD system was passed, standalone functionality is guaranteed by simply using floats
    and accessing the fractional values of components and values directly.

    All components, phases and phase equilibrium equations must be added before
    initializing the composition.

    Whether specific enthalpy or temperature is temporarily a primary, depends on the chosen
    flash procedure.

    The primary variables are
        - pressure,
        - specific enthalpy of the mixture (depending on the flash procedure),
        - temperature of the mixture (depending on the flash procedure),
        - feed fractions per component.
    Primary Variables are assumed to be given and the values set **externally**.
    The equilibrium is performed for fixed p-T or fixed p-h,
    and additionally a fixed feed composition.

    Secondary variables are fractions, i.e.
        - molar phase fractions
        - molar component fractions in a phase
        - volumetric phase fractions (saturations)
    The values of these fractions are calculated by the composite framework, i.e. they do not
    have to be set externally, but are instantiated with zero values in respective classes.
    (this is especially important for the AD framework to be able to evaluate operators).

    While the molar fractions are the actual unknowns in the flash procedure, the saturation
    values can be computed once the equilibrium converges using a relation between molar and
    volumetric fractions for phases based on an averaging process for porous media.

    References to the secondary variables are stored in respective classes representing
    components and phases.

    The isenthalpic flash and isothermal flash procedure are implemented.
    The persistent variable approach is utilized based on the work of [1,2], and the references
    therein.

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


    Parameters:
        ad_system (optional): If given, this class will use the AD framework and the respective
            mixed-dimensional domain to represent all involved variables cell-wise in each
            subdomain.
    """

    def __init__(self, ad_system: Optional[pp.ad.ADSystemManager] = None) -> None:

        if ad_system is None:  # TODO
            raise NotImplementedError("Composition standalone model not yet available.")
        ### PUBLIC

        self.ad_system: Optional[pp.ad.ADSystemManager] = ad_system
        """The AD system passed at instantiation."""

        self.flash_history: List[Dict[str, Any]] = list()
        """Contains chronologically stored information about calculated flash procedures."""

        self.flash_tolerance: float = 1e-8
        """Convergence criterion for the flash algorithm."""

        self.max_iter_flash: int = 1000
        """Maximal number of iterations for the flash algorithms."""

        self.equilibrium_equations: Dict[str, Dict[str, pp.ad.Operator]] = dict()
        """Contains for each present component name (key) a sub-dictionary, which in return
        contains equilibrium equations per given equation name (key).

        """

        self.ph_subsystem: dict = dict()
        """A dictionary representing the subsystem for the p-h flash. Contains information on
        relevant variables and equations.

        """

        self.pT_subsystem: dict = dict()
        """A dictionary representing the subsystem for the p-T flash. Contains information on
        relevant variables and equations.

        """

        ### PRIVATE
        # primary variables
        self._p_var: str = "p"
        self._h_var: str = "h"
        self._T_var: str = "T"
        self._p: VarLike
        self._h: VarLike
        self._T: VarLike
        if ad_system:
            self._p = ad_system.create_variable(self._p_var, True)
            self._h = ad_system.create_variable(self._h_var, True)
            self._T = ad_system.create_variable(self._T_var, False)
        else:
            self._p = 101.325
            self._h = 0.0
            self._T = 0.0

        # composition
        self._components: List[pp.composite.Component] = list()
        self._phases: List[pp.composite.Phase] = list()
        # contains per component name (key) a list of phases in which this component is
        # modelled
        self._phases_of_component: Dict[str, List[pp.composite.Phase]] = dict()

        # other
        # maximal number of flash history entries (FiFo)
        self._max_history: int = 100
        # names of equations
        self._mass_conservation: str = "flash_mass"
        self._phase_fraction_unity: str = "flash_phase_unity"
        self._complementary: str = "flash_complementary"  # complementary conditions
        self._enthalpy_constraint: str = "flash_h_constraint"  # for p-h flash
        # complementary condition tuple TODO better solution...
        self._complementary_eq: dict = dict()

        # for standalone applications, flash results are stored in the composition class, not
        # in the AD variables
        self._feed_vals: Dict[str, float] = dict()  # val per component name
        self._saturation_vals: Dict[str, float] = dict()  # val per phase name
        self._phase_fraction_vals: Dict[str, float] = dict()  # val per phase name
        # dict per phase name, which in return contains vals per component name in that phase
        self._phase_composition_vals: Dict[str, Dict[str, float]] = dict()

    ### Thermodynamic State -------------------------------------------------------------------

    @property
    def p(self) -> VarLike:
        """Initialized with 1 atm (101.325 kPa).

        The values are assumed to represent values at equilibrium and are therefore constant
        during the flash.

        If no AD system is present, the pressure value can be set directly.
        If the AD framework is used, respective functionalities must be used to set values for
        the pressure (merged) variable.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kPa] = [kN / m^2]

        Parameters:
            value: a value to set the pressure for standalone applications.

        Returns:
            the primary variable pressure on the whole domain (cell-wise).
            Float if standalone.

        Raises:
            RuntimeError: if an attempt is made to set the value despite a present AD system.

        """
        return self._p

    @p.setter
    def p(self, value: float) -> None:
        if self.ad_system:
            raise RuntimeError(
                f"Cannot set the pressure when AD is used. "
                "Use respective functionalities of the AD system."
            )
        else:
            self._p = value

    @property
    def h(self) -> VarLike:
        """Initialized with 0.

        For the isenthalpic flash, the values are assumed to represent values at equilibrium.
        For the isothermal flash, the enthalpy changes based on the results (composition).

        If no AD system is present, the enthalpy value can be set directly.
        If the AD framework is used, respective functionalities must be used to set values for
        the enthalpy (merged) variable.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kJ / mol / K]

        Parameters:
            value: a value to set the enthalpy for standalone applications.

        Returns:
            the primary variable specific molar enthalpy on the whole domain (cell-wise).
            Float if standalone.

        Raises:
            RuntimeError: if an attempt is made to set the value despite a present AD system.

        """
        return self._h

    @h.setter
    def h(self, value: float) -> None:
        if self.ad_system:
            raise RuntimeError(
                f"Cannot set the enthalpy when AD is used. "
                "Use respective functionalities of the AD system."
            )
        else:
            self._h = value

    @property
    def T(self) -> VarLike:
        """Initialized with 0.

        For the isothermal flash, the values are assumed to represent values at equilibrium.
        For the isenthalpic flash, the temperature varies and depends on the enthalpy and the
        composition.

        If no AD system is present, the temperature value can be set directly.
        If the AD framework is used, respective functionalities must be used to set values for
        the temperature (merged) variable.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [K]

        Parameters:
            value: a value to set the temperature for standalone applications.

        Returns:
            the primary variable temperature on the whole domain (cell-wise).
            Float if standalone.

        Raises:
            RuntimeError: if an attempt is made to set the value despite a present AD system.

        """
        return self._T

    @T.setter
    def T(self, value: float) -> None:
        if self.ad_system:
            raise RuntimeError(
                f"Cannot set the temperature when AD is used. "
                "Use respective functionalities of the AD system."
            )
        else:
            self._T = value

    def density(self, prev_time: bool = False) -> VarLike:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [mol / REV]

        Parameters:
            prev_time: indicator to use values from the previous time step, if the AD framework
                is used.

        Returns:
            Returns the overall molar density of the composition
            given by the saturation-weighted sum of all phase densities.
            The phase densities are computed using the current temperature and pressure
            values.

        """
        # creating a list of saturation-weighted phase densities
        # If the value from the previous time step (STATE) is requested,
        # we do so using the functionality of the AD framework
        if self.ad_system:
            if prev_time:
                rho = [
                    phase.saturation.previous_timestep()
                    * phase.density(
                        self.p.previous_timestep(),
                        self.T.previous_timestep(),
                    )
                    for phase in self._phases
                ]
            else:
                rho = [
                    phase.saturation * phase.density(self.p, self.T)
                    for phase in self._phases
                ]
        # if AD is not used, we evaluate the density based on currently stored values
        else:
            rho = [
                self._saturation_vals[phase.name] * phase.density(self.p, self.T)
                for phase in self._phases
            ]
        # summing the elements of the list results in the mixture density
        return sum(rho)

    ### Composition Management ----------------------------------------------------------------

    @property
    def num_components(self) -> int:
        """Number of components in the composition."""
        return len(self._components)

    @property
    def num_phases(self) -> int:
        """Number of modelled (added) phases in the composition."""
        return len(self._phases)

    @property
    def num_equilibrium_equations(self) -> Dict[str, int]:
        """A dictionary containing the number of necessary phase equilibrium equations per
        component name (key).

        The equation has to be formulated with respect to the reference phase.

        The number is based on the number of phases in which a substance is anticipated i.e.,
        if substance c is present in ``n_p(c)`` phases, we need ``n_p(c) - 1``
        equilibrium equations, since one phase is chosen as a reference phase.

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
        Yields:
            components added to the composition.

        """
        for C in self._components:
            yield C

    @property
    def phases(self) -> Generator[pp.composite.Phase, None, None]:
        """
        Yields:
            phases modelled (added) to the composition.

        """
        for P in self._phases:
            yield P

    def add_component(
        self, component: Union[List[pp.composite.Component], pp.composite.Component]
    ) -> None:
        """Adds one or multiple components to the composition.

        All modelled components must be added before any phase is modelled/ added to this
        composition.

        All components, phases and phase equilibrium equations must be added before
        initializing the composition.

        Parameters:
            component: a component, or list of components, to be added to this mixture.

        Raises:
            RuntimeError: if the AD framework is used and the component was instantiated using
                a different AD system than the one used for this composition.

        """
        if isinstance(component, pp.composite.Component):
            component = [component]

        added_components = [comp.name for comp in self._components]

        for comp in component:

            if comp.name in added_components:
                # already added components are skipped
                continue

            # sanity check when using the AD framework
            if self.ad_system:
                if self.ad_system != comp.ad_system:
                    raise RuntimeError(
                        f"Component '{comp.name}' instantiated with a different AD system."
                    )
            # initiate zero feed fraction values for standalone application
            else:
                self._feed_vals.update({comp.name: 0.0})
            # add component and initiate dict for equilibrium equations
            self._components.append(comp)
            self.equilibrium_equations.update({comp.name: dict()})

    def add_phase(
        self, phases: Union[List[pp.composite.Phase], pp.composite.Phase]
    ) -> None:
        """Adds one or multiple phases which are anticipated in the current model.

        As of now, the modeler has to anticipate every phase which might appear or disappear.

        A phase is only allowed to contain components, which are known to the composition i.e.,
        have already been added using :meth:`add_component`.

        All components, phases and phase equilibrium equations must be added before
        initializing the composition.

        Note:
            The first phase added will be used as the reference phase (unitarity).
            All equilibrium equations must be formulated with respect to the reference phase.

        Parameters:
            phases: a phase, or a list of phases, anticipated in this mixture model.

        Raises:
            RuntimeError: if the AD framework is used and the phase was instantiated using
                a different AD system than the one used for this composition.
            RuntimeError: if the phase is 'empty', i.e. contains no components.
            ValueError: if the phase contains components unknown to this composition.
                Use :meth:`add_component` to add all components before adding any phases.

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

            # if 'empty' phases are passed, raise an error
            if phase.num_components == 0:
                raise RuntimeError("Phases without components are nonphysical.")

            # sanity check when using the AD framework
            if self.ad_system:
                if self.ad_system != phase.ad_system:
                    raise RuntimeError(
                        f"phase '{phase.name}' instantiated with a different AD system."
                    )
            # initiate zero feed fraction values for standalone application
            else:
                self._saturation_vals.update({phase.name: 0.0})
                self._phase_fraction_vals.update({phase.name: 0.0})
                self._phase_composition_vals.update({phase.name: dict()})

            # check if the components are known
            for component in phase:
                if component not in self._components:
                    raise ValueError(
                        f"Unknown component '{component.name}' in phase '{phase.name}'."
                    )
                # initiate zero fractions for standalone applications
                if not self.ad_system:
                    self._phase_composition_vals[phase.name].update(
                        {component.name: 0.0}
                    )

            # add to list of anticipated phases
            self._phases.append(phase)

        # resolving the composition to get a data structure yielding phases per component,
        # if the component is modelled in that phase
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

    def add_equilibrium_equation(
        self,
        component: pp.composite.Component,
        equation: pp.ad.Operator,
        equ_name: str,
    ) -> None:
        """Adds a phase equilibrium equation to the flash for when the AD framework is used.

        All components, phases and phase equilibrium equations must be added before
        initializing the composition.

        This is modularized and for now it is completely up to the modeler to assure
        a non-singular system of equations.

        For the number of necessary equations per component
        see :method:`num_equilibrium_equations`.

        Notes:
            - Make sure the equation is such, that the right-hand-side is zero and the passed
            operator represents the left-hand-side.
            - Make sure the name is unique and does not match any other equation.
              Unknown behavior else.
            - Make sure the equilibrium is formulated with respect to the reference phase.

        Parameters:
            component: component for which the equilibrium equation is intended.
                Must be a component already added to this composition.
            equation: a general AD operator representing the left-hand side of the equation.
            equ_name: a name for the equation, used as an unique identifier.

        Raises:
            ValueError: if the component is unknown to this composition.

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
            raise ValueError(
                f"Component '{component.name}' not present in composition."
            )

    def initialize(self) -> None:
        """Initializes the flash equations for this system, based on the added components,
        phases and equilibrium equations.

        This is the last step before a flash method should be called.
        All components, phases and equilibrium equations must be added before calling this
        method.

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

        # allocating place for the subsystem
        equations = dict()
        pT_subsystem: Dict[str, list] = self._get_subsystem_dict()
        ph_subsystem: Dict[str, list] = self._get_subsystem_dict()
        self._set_subsystem_vars(ph_subsystem, pT_subsystem)

        ### Mass conservation equations
        for component in self.components:
            name = f"{self._mass_conservation}_{component.name}"
            equation = self.get_mass_conservation_for(component)
            equations.update({name: equation})
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
        equ = self.get_phase_fraction_unity()
        equations.update({self._phase_fraction_unity: equ})
        pT_subsystem["equations"].append(self._phase_fraction_unity)
        ph_subsystem["equations"].append(self._phase_fraction_unity)

        ### enthalpy constraint for p-H flash
        equation = self.get_enthalpy_constraint()
        equations.update({self._enthalpy_constraint: equation})
        ph_subsystem["equations"].append(self._enthalpy_constraint)

        ### Semi-smooth complementary conditions per phase
        self._complementary_eq = dict()
        # the first complementary condition will be eliminated by the phase fraction unity
        for phase in self._phases[1:]:
            name = f"{self._complementary}_{phase.name}"
            condition = self.get_complementary_condition_for(phase)

            self._complementary_eq.update({name: condition})
            pT_subsystem["equations"].append(name)
            ph_subsystem["equations"].append(name)

        # adding equations to AD system
        if self.ad_system:
            for name, equ in equations.items():
                # TODO the ad-system has no clue about the complementary conditions
                # write a AD wrapper for the semi-smooth min
                self.ad_system.set_equation(name, equ)
        # storing references to the subsystems
        self.pT_subsystem = pT_subsystem
        self.ph_subsystem = ph_subsystem

    ### other ---------------------------------------------------------------------------------

    def print_last_flash(self) -> None:
        """Prints the result of the last flash calculation."""
        entry = self.flash_history[-1]
        msg = "\nProcedure: %s\n" % (str(entry["flash"]))
        msg += "SUCCESS: %s\n" % (str(entry["success"]))
        msg += "Method: %s\n" % (str(entry["method"]))
        msg += "Remarks: %s" % (str(entry["other"]))

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
        """Makes an entry in the flash history."""

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
            pT_subsystem["secondary_var_names"].append(component.fraction_var_name)
            ph_subsystem["secondary_vars"].append(component.fraction)
            ph_subsystem["secondary_var_names"].append(component.fraction_var_name)
        # saturations are always secondary vars
        for phase in self.phases:
            pT_subsystem["secondary_vars"].append(phase.saturation)
            pT_subsystem["secondary_var_names"].append(phase.saturation_var_name)
            ph_subsystem["secondary_vars"].append(phase.saturation)
            ph_subsystem["secondary_var_names"].append(phase.saturation_var_name)

        # primary vars which are same for both subsystems
        # phase fractions
        for phase in self.phases:
            pT_subsystem["primary_vars"].append(phase.fraction)
            pT_subsystem["primary_var_names"].append(phase.fraction_var_name)
            ph_subsystem["primary_vars"].append(phase.fraction)
            ph_subsystem["primary_var_names"].append(phase.fraction_var_name)
            # phase composition
            for component in phase:
                var = phase.component_fraction_of(component)
                var_name = phase.component_fraction_var_name(component)
                pT_subsystem["primary_vars"].append(var)
                pT_subsystem["primary_var_names"].append(var_name)
                ph_subsystem["primary_vars"].append(var)
                ph_subsystem["primary_var_names"].append(var_name)
        # for the p-h flash, T is an additional var
        ph_subsystem["primary_vars"].append(self._T)
        ph_subsystem["primary_var_names"].append(self._T_var)

    def print_x(self) -> None:
        X = self.ad_system.dof_manager.assemble_variable()
        print(X)
        X = self.ad_system.dof_manager.assemble_variable(from_iterate=True)
        print(X)

    ### Flash methods -------------------------------------------------------------------------

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
        if self.ad_system:
            success = self._Newton_min(self.pT_subsystem, copy_to_state, initial_guess)
        else:  # TODO
            success = False

        if success:
            self._post_process_fractions(copy_to_state)
        # if not successful, we re-normalize only the iterate
        else:
            self._post_process_fractions(False)
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
        if self.ad_system:
            success = self._Newton_min(self.ph_subsystem, copy_to_state, initial_guess)
        else:  # TODO
            success = False

        if success:
            self._post_process_fractions(copy_to_state)
        else:  # if not successful, we re-normalize only the iterate
            self._post_process_fractions(False)
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
        if self.num_phases == 1:
            self._single_phase_saturation_evaluation(copy_to_state)
        if self.num_phases == 2:
            self._2phase_saturation_evaluation(copy_to_state)
        elif self.num_phases >= 3:
            self._multi_phase_saturation_evaluation(copy_to_state)

    def evaluate_specific_enthalpy(self, copy_to_state: bool = True) -> None:
        """Based on current pressure, temperature and phase fractions, evaluates the
        specific molar enthalpy. Use with care, if the equilibrium problem is coupled with
        e.g., the flow.

        Parameters:
            copy_to_state: if an AD system is used, copies the values to the STATE of the
                AD variable, additionally to ITERATE.

        """
        # AD mode
        if self.ad_system:
            # obtain values by forward evaluation
            equ_ = list()
            for phase in self.phases:
                equ_.append(phase.fraction * phase.specific_enthalpy(self._p, self._T))
            equ = sum(equ_)

            # if no phase present (list empty) zero is returned and enthalpy is zero
            if equ == 0:
                h = np.zeros(self.ad_system.dof_manager.mdg.num_subdomain_cells())
            # else evaluate this operator
            elif isinstance(equ, pp.ad.Operator):
                h = equ.evaluate(self.ad_system.dof_manager).val
            else:
                raise RuntimeError("Something went terribly wrong.")
            # write values in local var form
            self.ad_system.set_var_values(self._h_var, h, copy_to_state)
        # standalone mode
        else:
            h = 0.0
            for phase in self.phases:
                h += self._phase_fraction_vals[phase.name] * phase.specific_enthalpy(
                    self.p, self.T
                )
            # storing value directly in standalone mode
            self.h = h

    ### Model equations -----------------------------------------------------------------------

    def get_mass_conservation_for(
        self, component: pp.composite.Component
    ) -> pp.ad.Operator:
        """Returns an equation representing the definition of the overall component fraction
        (mass conservation) for a component component.

        z_c - sum_phases y_e * x_ce = 0

        Returns: AD operator representing the left-hand side of the equation (rhs=0).

        """
        equation = None
        if self.ad_system:
            # zeta_c
            equation = component.fraction

            for phase in self._phases_of_component[component.name]:
                # - xi_e * chi_ce
                equation -= phase.fraction * phase.component_fraction_of(component)
        else:
            pass  # TODO

        return equation

    def get_phase_fraction_unity(self) -> pp.ad.Operator:
        """Returns an equation representing the phase fraction unity

        1 - sum_phases y_e = 0

        Returns: AD operator representing the left-hand side of the equation (rhs=0).

        """
        equation = None
        if self.ad_system:
            equation = pp.ad.Scalar(1.0)

            for phase in self.phases:
                equation -= phase.fraction
        else:
            pass  # TODO

        return equation

    def get_enthalpy_constraint(self) -> pp.ad.Operator:
        """Returns an equation representing the specific molar enthalpy of the composition,
        based on it's definition:

        h - sum_phases y_e * h_e(p,T) = 0

        Can be used to for the p-h flash as enthalpy constraint (T is an additional variable).

        Returns: AD operator representing the left-hand side of the equation (rhs=0).

        """
        equation = None
        if self.ad_system:
            equation = self.h
            for phase in self.phases:
                equation -= phase.fraction * phase.specific_enthalpy(self.p, self.T)
        else:
            pass  # TODO

        return equation

    def get_composition_unity_for(self, phase: pp.composite.Phase) -> pp.ad.Operator:
        """Returns an equation representing the unity if the composition for a given phase e:

        1 - sum_components chi_ce = 0

        Returns: AD operator representing the left-hand side of the equation (rhs=0).

        """
        equation = None
        if self.ad_system:
            equation = pp.ad.Scalar(1.0)

            for component in phase:
                equation -= phase.component_fraction_of(component)
        else:
            pass  # TODO

        return equation

    def get_complementary_condition_for(
        self, phase: pp.composite.Phase
    ) -> tuple[pp.ad.Operator, pp.ad.Operator]:
        """Returns the two complementary equations for a phase e:

        min{y_e, 1 - sum_components chi_ce} = 0

        Returns: tuple of AD operators representing the left-hand side of the equation (rhs=0).

        """
        if self.ad_system:
            return (phase.fraction, self.get_composition_unity_for(phase))
        else:
            pass  # TODO

    ### Flash methods -------------------------------------------------------------------------

    def _Newton_min(
        self, subsystem: dict, copy_to_state: bool, initial_guess: str
    ) -> bool:
        """Performs a semi-smooth newton (Newton-min), where the complementary conditions are
        the semi-smooth part.

        Parameters:
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
            prolongation = self.ad_system.dof_manager.projection_to(
                var_names
            ).transpose()

            for i in range(self.max_iter_flash):

                # solve iteration and add to ITERATE state additively
                dx = sps.linalg.spsolve(A, b)
                DX = prolongation * dx
                self.ad_system.dof_manager.distribute_variable(
                    DX,
                    variables=var_names,
                    additive=True,
                    to_iterate=True,
                )
                # counting necessary number of iterations
                iter_final = i + 1  # shift since range() starts with zero

                # assemble new matrix and residual
                A, b = self._assemble_semi_smooth_system(
                    equations, complementary_cond, vars
                )

                # in case of convergence
                if np.linalg.norm(b) <= self.flash_tolerance:

                    # setting STATE to newly found solution
                    if copy_to_state:
                        X = self.ad_system.dof_manager.assemble_variable(
                            variables=var_names, from_iterate=True
                        )
                        self.ad_system.dof_manager.distribute_variable(
                            X, variables=var_names
                        )

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
            variables=var_names,
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

        Parameters:
            equations: list of operators representing the smooth part of the system
            complementary_cond: a 2-tuple of AD operators representing the arguments for min
            vars: list of variables w.r.t. which the lin. equations should be assembled

        Returns:
            spmatrix: subgradient element w.r.t. to the given variables
            ndarray: residual vector for the current iterate state.

        """
        # assemble smooth subsystem
        A_s, b_s = self.ad_system.assemble_subsystem(equations, vars)

        # assemble non-smooth subsystem
        all_b_ns = list()
        all_A_ns = list()
        # projection to primary variables
        var_names = self.ad_system.get_var_names_from(vars)
        projection = self.ad_system.dof_manager.projection_to(var_names).transpose()

        for comp_cond in complementary_cond:
            op1, op2 = self._complementary_eq[comp_cond]

            b1 = op1.evaluate(self.ad_system.dof_manager)
            b2 = op2.evaluate(self.ad_system.dof_manager)
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

        # slice out relevant columns
        A_ns = sps.bmat(all_A_ns, format="csr") * projection
        # stack smooth and non-smooth part to global semi-smooth system
        A = sps.bmat([[A_s], [A_ns]], format="csr")
        b = np.hstack([b_s] + all_b_ns)

        return A, b

    def _set_initial_guess(
        self, initial_guess: str, guess_temperature: bool = False
    ) -> None:
        """Auxillary function to set the initial values for phase fractions, phase compositions
        and temperature, based on the chosen strategy.
        """

        # AD mode
        if self.ad_system:
            # shorten name space
            dm = self.ad_system.dof_manager
            nc = dm.mdg.num_subdomain_cells()

            if initial_guess == "iterate":
                pass  # DofManager does this by default
            elif initial_guess == "feed":
                # 'phase feed' is a temporary value, sum of feeds of present components
                # y_e = sum z_c if c in e
                all_phase_feeds = list()
                # setting the values for phase compositions per phase
                for phase in self.phases:
                    # feed fractions from components present in this phase
                    feed = [
                        self.ad_system.get_var_values(comp.fraction_var_name, False)
                        for comp in phase
                    ]

                    # this is not necessarily one if not all components are present in phase,
                    # but never zero because empty phases are not accepted
                    phase_feed = sum(feed)
                    all_phase_feeds.append(phase_feed)

                    for c, component in enumerate(phase):
                        fraction_in_phase = feed[c]
                        # re-normalize the fraction,
                        # dependent on number of components in this phase.
                        # component fractions have to sum up to 1.
                        fraction_in_phase = fraction_in_phase / phase_feed
                        # write
                        self.ad_system.set_var_values(
                            phase.component_fraction_var_name(component),
                            fraction_in_phase,
                        )

                # by re-normalizing the phase feeds,
                # we obtain an initial guess for the phase fraction.
                # phase fractions have to sum up to 1.
                phase_feed_sum = sum(all_phase_feeds)
                for e, phase in enumerate(self.phases):
                    phase_feed = all_phase_feeds[e]
                    phase_fraction = phase_feed / phase_feed_sum
                    self.ad_system.set_var_values(
                        phase.fraction_var_name, phase_fraction
                    )

            elif initial_guess == "uniform":
                # uniform values for phase fraction
                val_phases = 1.0 / self.num_phases
                for phase in self.phases:
                    self.ad_system.set_var_values(
                        phase.fraction_var_name, val_phases * np.ones(nc)
                    )
                    # uniform values for composition of this phase
                    val = 1.0 / phase.num_components
                    for component in phase:
                        self.ad_system.set_var_values(
                            phase.component_fraction_var_name(component),
                            val * np.ones(nc),
                        )
        # standalone
        else:
            if initial_guess == "iterate":
                # and there is no iterate for standalone applications
                pass
            elif initial_guess == "feed":
                # 'phase feed' is a temporary value, sum of feeds of present components
                # y_e = sum z_c if c in e
                all_phase_feeds = list()
                # setting the values for phase compositions per phase
                for phase in self.phases:
                    # feed fractions from components present in this phase
                    feed = [
                        self._phase_composition_vals[phase][component]
                        for component in phase
                    ]

                    # this is not necessarily one if not all components are present in phase,
                    # but never zero because empty phases are not accepted
                    phase_feed = sum(feed)
                    all_phase_feeds.append(phase_feed)

                    for c, component in enumerate(phase):
                        fraction_in_phase = feed[c]
                        # re-normalize the fraction,
                        # dependent on number of components in this phase.
                        # component fractions have to sum up to 1.
                        fraction_in_phase = fraction_in_phase / phase_feed
                        self._phase_composition_vals[phase][
                            component
                        ] = fraction_in_phase

                # by re-normalizing the phase feeds,
                # we obtain an initial guess for the phase fraction.
                # phase fractions have to sum up to 1.
                phase_feed_sum = sum(all_phase_feeds)
                for e, phase in enumerate(self.phases):
                    phase_feed = all_phase_feeds[e]
                    phase_fraction = phase_feed / phase_feed_sum
                    self._phase_fraction_vals[phase.name] = phase_fraction

            elif initial_guess == "uniform":
                # uniform values for phase fraction
                val_phases = 1.0 / self.num_phases
                for phase in self.phases:
                    self._phase_fraction_vals[phase.name] = val_phases

                    # uniform values for composition of this phase
                    val = 1.0 / phase.num_components
                    for component in phase:
                        self._phase_composition_vals[phase.name][component.name] = val
            else:
                raise ValueError(f"Unknown initial guess strategy '{initial_guess}'")

        if guess_temperature:
            # TODO implement
            pass

    def _post_process_fractions(self, copy_to_state: bool) -> None:
        """Re-normalizes phase compositions (restores unity) and removes numerical artifacts
        (values bound between 0 and 1).

        Phase compositions (fractions of components in that phase) are nonphysical if a
        phase is not present. The unified flash procedure yields nevertheless values, possibly
        violating the unity constraint. Respective fractions have to be re-normalized in a
        post-processing step.

        Also, removes artifacts outside the bound 0 and 1 for all molar fractions
        except feed fraction, which is **not** changed by the flash at all
        (the amount of matter is not supposed to change).

        Parameters:
            copy_to_state: if an AD system is present, copies the values to the STATE of the
                AD variable, additionally to ITERATE.

        """
        # Ad mode
        if self.ad_system:
            for phase in self.phases:
                # remove numerical artifacts
                phase_frac = self.ad_system.get_var_values(phase.fraction_var_name)
                phase_frac[phase_frac < 0.0] = 0.0
                phase_frac[phase_frac > 1.0] = 1.0
                self.ad_system.set_var_values(
                    phase.fraction_var_name, phase_frac, copy_to_state
                )
                # extracting phase composition
                phase_composition = list()
                for comp in phase:
                    comp_frac = self.ad_system.get_var_values(
                        phase.component_fraction_var_name(comp)
                    )
                    phase_composition.append(comp_frac)
                comp_sum = sum(phase_composition)

                # re-normalize phase composition.
                # DOFS where unity is already fulfilled remain unchanged.
                for c, comp in enumerate(phase):
                    comp_frac = phase_composition[c]
                    comp_frac = comp_frac / comp_sum
                    # remove numerical artifacts
                    comp_frac[comp_frac < 0.0] = 0.0
                    comp_frac[comp_frac > 1.0] = 1.0
                    self.ad_system.set_var_values(
                        phase.component_fraction_var_name(comp),
                        comp_frac,
                        copy_to_state,
                    )
        # standalone mode
        else:
            for phase in self.phases:
                # removing numerical artifacts
                phase_frac = self._phase_fraction_vals[phase.name]
                if phase_frac < 0.0:
                    phase_frac = 0.0
                elif phase_frac > 1.0:
                    phase_frac = 1.0
                self._phase_fraction_vals[phase.name] = phase_frac

                phase_composition = list()
                for comp in phase:
                    phase_composition.append(
                        self._phase_composition_vals[phase.name][comp.name]
                    )
                comp_sum = sum(phase_composition)

                # re-normalize phase composition.
                for c, comp in enumerate(phase):
                    comp_frac = phase_composition[c]
                    comp_frac = comp_frac / comp_sum
                    # remove numerical artifacts
                    if comp_frac < 0.0:
                        comp_frac = 0.0
                    elif comp_frac > 1.0:
                        comp_frac = 1
                    self._phase_composition_vals[phase.name][comp.name] = comp_frac

    def _single_phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """If only one phase is present, we assume it occupies the whole pore space."""
        phase = self._phases[0]
        if self.ad_system:
            values = np.ones(self.ad_system.dof_manager.mdg.num_subdomain_cells())
            self.ad_system.set_var_values(
                phase.saturation_var_name, values, copy_to_state
            )
        else:
            self._saturation_vals[phase.name] = 0.0

    def _2phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """Calculates the saturation value assuming phase molar fractions are given.
        In the case of 2 phases, the evaluation is straight forward.

        It holds:
            s_i = 1 / (1 + y_j / (1 - y_j) * rho_i / rho_j) , i != j

        """
        # get reference to phases
        phase1 = self._phases[0]
        phase2 = self._phases[1]

        # AD mode
        if self.ad_system:
            # shortening the name space
            dm = self.ad_system.dof_manager
            # get phase molar fraction values
            y1 = self.ad_system.get_var_values(phase1.fraction_var_name)
            y2 = self.ad_system.get_var_values(phase2.fraction_var_name)

            # get density values for given pressure and enthalpy
            rho1 = phase1.density(self.p, self.T).evaluate(dm)
            if isinstance(rho1, pp.ad.Ad_array):
                rho1 = rho1.val
            rho2 = phase2.density(self._p, self._T).evaluate(dm)
            if isinstance(rho2, pp.ad.Ad_array):
                rho2 = rho2.val

            # allocate saturations, size must be the same
            s1 = np.zeros(y1.size)
            s2 = np.zeros(y1.size)

            # TODO test sensitivity of this
            phase1_saturated = y1 == 1.0  # equal to phase2_vanished
            phase2_saturated = y2 == 1.0  # equal to phase1_vanished

            # calculate only non-saturated cells to avoid division by zero
            # set saturated or "vanishing" cells explicitly to 1., or 0. respectively
            idx = np.logical_not(phase2_saturated)
            y2_idx = y2[idx]
            rho1_idx = rho1[idx]
            rho2_idx = rho2[idx]
            s1[idx] = 1.0 / (1.0 + y2_idx / (1.0 - y2_idx) * rho1_idx / rho2_idx)
            s1[phase1_saturated] = 1.0
            s1[phase2_saturated] = 0.0

            idx = np.logical_not(phase1_saturated)
            y1_idx = y1[idx]
            rho1_idx = rho1[idx]
            rho2_idx = rho2[idx]
            s2[idx] = 1.0 / (1.0 + y1_idx / (1.0 - y1_idx) * rho2_idx / rho1_idx)
            s2[phase1_saturated] = 0.0
            s2[phase2_saturated] = 1.0

            # write values to AD system
            self.ad_system.set_var_values(phase1.saturation_var_name, s1, copy_to_state)
            self.ad_system.set_var_values(phase2.saturation_var_name, s2, copy_to_state)
        # standalone mode
        else:
            # get values for molar fractions and densities
            y1 = self._phase_fraction_vals[phase1.name]
            y2 = self._phase_fraction_vals[phase2.name]
            rho1 = phase1.density(self.p, self.T)
            rho2 = phase2.density(self.p, self.T)

            # TODO check sensitivity of this
            if y1 == 1.0:  # phase 2 vanished
                s1 = 1.0
                s2 = 0.0
            elif y1 == 0.0:  # phase 1 vanished
                s1 = 0.0
                s2 = 1.0
            else:  # both phases present
                s1 = 1.0 / (1.0 + y2 / (1.0 - y2) * rho1 / rho2)
                s2 = 1.0 - s1  # unity constraint
            # write values
            self._saturation_vals[phase1.name] = s1
            self._saturation_vals[phase2.name] = s2

    def _multi_phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """Calculates the saturation value assuming phase molar fractions are given.
        Valid for compositions with at least 3 phases.
        In this case a linear system has to be solved for each multiphase cell

        It holds for all i = 1... m, where m is the number of phases:
            1 = sum_{j != i} (1 + rho_j / rho_i * xi_i / (1 - xi_i)) s_j
        """
        # AD mode
        if self.ad_system:
            # shortening name space
            dm = self.ad_system.dof_manager
            nc = dm.mdg.num_subdomain_cells()
            # molar fractions per phase
            y = [
                self.ad_system.get_var_values(phase.saturation_var_name)
                for phase in self.phases
            ]
            # densities per phase
            rho = list()
            for phase in self.phases:
                rho_e = phase.density(self.p, self.T).evaluate(dm)
                if isinstance(rho_e, pp.ad.Ad_array):
                    rho_e = rho_e.val
                rho.append(rho_e)

            mat_per_eq = list()

            # list of indicators per phase, where the phase is fully saturated
            saturated = list()
            # where one phase is saturated, the other vanish
            vanished = [np.zeros(nc, dtype=bool) for _ in self.phases]

            for i in range(self.num_phases):
                # get the DOFS where one phase is fully saturated
                # TODO check sensitivity of this
                saturated_i = y[i] == 1.0
                saturated.append(saturated_i)

                # store information that other phases vanish at these DOFs
                for j in range(self.num_phases):
                    if j == i:
                        # a phase can not vanish and be saturated at the same time
                        continue
                    else:
                        # where phase i is saturated, phase j vanishes
                        # Use OR to accumulate the bools per i-loop without overwriting
                        vanished[j] = np.logical_or(vanished[j], saturated_i)

            # indicator which DOFs are saturated for the vector of stacked saturations
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
                        mats.append(sps.diags([np.zeros(nc)]))
                    # diagonals of blocks which are not on the main diagonal, are non-zero
                    else:
                        denominator = 1 - y[i]
                        # to avoid a division by zero error, we set it to one
                        # this is arbitrary, but respective matrix entries will be sliced out
                        # since they correspond to cells where one phase is saturated,
                        # i.e. the respective saturation is 1., the other 0.
                        denominator[denominator == 0.0] = 1.0
                        d = 1.0 + rho[j] / rho[i] * y[i] / denominator

                        mats.append(sps.diags([d]))

                # rectangular matrix per equation
                mat_per_eq.append(np.hstack(mats))

            # Stack matrices per equation on each other
            # This matrix corresponds to the vector of stacked saturations per phase
            mat = np.vstack(mat_per_eq)
            # TODO permute DOFS to get a block diagonal matrix. This one has a large band width
            mat = sps.csr_matrix(mat)

            # projection matrix to DOFs in multiphase region
            # start with identity in CSR format
            projection = sps.diags([np.ones(len(multiphase))]).tocsr()
            # slice image of canonical projection out of identity
            projection = projection[multiphase]
            projection_transposed = projection.transpose()

            # get sliced system
            rhs = projection * np.ones(nc * self.num_phases)
            mat = projection * mat * projection_transposed

            s = sps.linalg.spsolve(mat.tocsr(), rhs)

            # prolongate the values from the multiphase region to global DOFs
            saturations = projection_transposed * s
            # set values where phases are saturated or have vanished
            saturations[saturated] = 1.0
            saturations[vanished] = 0.0

            # distribute results to the saturation variables
            for i, phase in enumerate(self._phases):
                vals = saturations[i * nc : (i + 1) * nc]
                self.ad_system.set_var_values(
                    phase.saturation_var_name, vals, copy_to_state
                )
        # standalone mode
        else:
            y = [self._phase_fraction_vals[phase.name] for phase in self.phases]
            y = np.array(y)
            rho = [phase.density(self.p, self.T) for phase in self.phases]
            s = np.zeros(len(self._phases))

            # TODO check sensitivity
            if 1.0 in y:  # if one phase is saturated, all other vanish
                saturated = y.index(1.0)
                s[saturated] = 1.0
            else:
                # identify present phases
                present = y > 0.0
                y_present = y[present]
                n = len(y_present)

                mat = np.zeros((n, n))
                rhs = np.ones(n)

                for i in range(n):
                    for j in range(n):
                        if i == j:  # the diagonal of the matrix is zero
                            continue
                        else:
                            denominator = 1 - y[i]
                            mat[i, j] = 1.0 + rho[j] / rho[i] * y[i] / denominator
                # solve and insert the values for present phases
                s_present = np.linalg.solve(mat, rhs)
                s[present] = s_present
            # store values
            for i, phase in enumerate(self._phases):
                self._saturation_vals[phase.name] = s[i]

    ### Special methods -----------------------------------------------------------------------

    def __str__(self) -> str:
        """Returns string representation of the composition,
        with information about present components.

        """
        out = f"Composition with {self.num_components} components:"
        for component in self.components:
            out += f"\n{component.name}"
        out += f"\nand {self.num_phases} phases:"
        for phase in self.phases:
            out += f"\n{phase.name}"
        return out
