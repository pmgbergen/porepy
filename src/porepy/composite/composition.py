"""Contains a class representing a multiphase multicomponent mixture (composition)"""

from __future__ import annotations

import numbers
from typing import Any, Dict, Generator, List, Literal, Optional, Set, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from ._composite_utils import COMPUTATIONAL_VARIABLES

__all__ = ["Composition"]


class Composition:
    """Representation of a composition of multiple components (chemical substances).
    Performs thermodynamically consistent phase stability and equilibrium calculations.

    The composition is 'dynamic'. Meaning attributes like present phases depend on the last
    flash calculation, which was performed.
    They can be updated on a porepy.MixedDimensionalGrid by external models e.g., flow.
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
        - volumetric phase fractions (saturations)
        - molar component fractions in a phase

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

    References:
        [1] Lauser, A. et. al.:
            A new approach for phase transitions in miscible multi-phase flow in porous media
            DOI: 10.1016/j.advwatres.2011.04.021
        [2] Ben Gharbia, I. et. al.:
            An analysis of the unified formulation for the equilibrium problem of
            compositional multiphase mixture
            DOI: 10.1051/m2an/2021075

    IMPLEMENTATION NOTE:
        - Equilibrium calculations can be done cell-wise.
          I.e. the computations can be done smarter or parallelized. This is not yet exploited.
        - number of cells is assumed to be fixed and computed only once at instantiation.
        - Currently the first phase added will be used as the reference phase, keep this in 
          mind when assembling the composition. It might have numeric implications
        - Currently the last component added will be eliminated by unitarity. Keep also this in
          mind, for the same reason.

    """

    def __init__(self, md: pp.MixedDimensionalGrid) -> None:
        """
        Args:
            gb (:class:`~porepy.GridBucket`): A grid bucket representing the geometric domain.
                Equilibrium calculations are performed locally, i.e. per cell.

        """
        # public attributes
        self.md: pp.MixedDimensionalGrid = md
        self.dof_manager: pp.DofManager = pp.DofManager(md)
        self.eq_manager: pp.ad.EquationManager = pp.ad.EquationManager(
            md, self.dof_manager
        )
        # contains chronologically information about past flash procedures
        self.flash_history: List[Dict[str, Any]] = list()
        # convergence criterion for the flash algorithm
        self.flash_tolerance: float = 1e-8
        # maximal number of iterations for the flash algorithm
        self.max_iter_flash: int = 1000

        # private attributes
        # primary variables
        self._p: pp.ad.MergedVariable
        self._p_var: str = COMPUTATIONAL_VARIABLES["pressure"]
        self._h: pp.ad.MergedVariable
        self._h_var: str = COMPUTATIONAL_VARIABLES["enthalpy"]
        self._T: pp.ad.MergedVariable
        self._T_var: str = COMPUTATIONAL_VARIABLES["temperature"]

        # composition
        self._components: Set[pp.composite.Component] = set()
        self._phases: List[pp.composite.Phase] = list()

        # other
        # maximal number of flash history entries (FiFo)
        self._max_history: int = 100
        # this is set true once a proper feed composition was set
        self._feed_composition_set: bool = False
        # Number of cells. The equilibrium is computed cell-wise.
        self._nc = self.md.num_subdomain_cells()

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

        for comp in component:
            self._components.add(comp)

        self._feed_composition_set = False

    def set_feed_composition(self, feed: List[Union[numbers.Real, np.ndarray]]) -> None:
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

            dof = self.dof_manager.dof_var([component.fraction])
            X[dof] = fraction
            var_names.append(component.fraction_var)

        if not np.allclose(fraction_sum, 1.0):
            raise ValueError("Sum of feed fraction does not fulfill unity.")

        self.dof_manager.distribute_variable(X, variables=var_names, to_iterate=True)
        self.dof_manager.distribute_variable(X, variables=var_names)
        self._feed_composition_set = True

    def set_state(
        self,
        p: Union[numbers.Real, np.ndarray],
        T: Union[numbers.Real, np.ndarray],
        do_flash: bool = False,
    ) -> bool:
        """Sets the thermodynamic state of the composition in terms of pressure and temperature
        at equilibrium.

        Args:
            p (ArrayLike, number): Pressure
            T (ArrayLike, number): Temperature
            do_flash (bool): flag for performing the isothermal flash.
                Checks whether the feed composition is been set prior to that.
                (see :meth:`Composition.set_feed_composition`).

        Returns:
            bool: False, if the isothermal flash did not converge, True otherwise.

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

        if not do_flash:
            return True
        else:
            return self.isothermal_flash(copy_to_state=True)

    def print_last_flash(self) -> None:
        """Prints the result of the last flash calculation."""
        entry = self.flash_history[-1]
        msg = "\nProcedure: %s\n" % (str(entry["flash"]))
        msg += "SUCCESS: %s\n" % (str(entry["success"]))
        msg += "Method: %s\n" % (str(entry["method"]))
        msg += "Remarks: %s" % (str(entry["other"]))

    # -----------------------------------------------------------------------------------------
    ### Flash methods
    # -----------------------------------------------------------------------------------------

    def isothermal_flash(self, copy_to_state: bool = True) -> bool:
        """Isothermal flash procedure to determine the composition based on given
        temperature of the mixture, pressure and feed fraction per component.

        Args:
            copy_to_state (bool): Copies the values to the STATE of the AD variables,
                additionally to ITERATE.

        """

        pass

    def isenthalpic_flash(self, copy_to_state: bool = True) -> bool:
        """Isenthalpic flash procedure to determine the composition based on given
        specific enthalpy of the mixture, pressure and feed fractions per component.

        Args:
            copy_to_state (bool): Copies the values to the STATE of the AD variable,
                additionally to ITERATE.

        """
        pass

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
        h = self.specific_enthalpy_equation()
        h = h.evaluate(self.dof_manager).val
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

    def specific_enthalpy_equation(self) -> Union[pp.ad.Operator, Literal[0]]:
        """Returns an operator representing the specific molar enthalpy of the composition,
        based on it's definition:

        h = sum_phases phase.fraction * phase.specific_enthalpy(p,T)

        This is for a simple, p-T-based evaluation. Can be used for an initial guess or the
        final computation after the p-T-flash.

        """

        equ = list()
        for phase in self.phases:
            equ.append(phase.fraction * phase.specific_enthalpy(self._p, self._T))

        return sum(equ)

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
                "method:": method,
                "iterations:": iterations,
                "success:": success,
                "variables:": str(variables),
                "equations:": str(equations),
                "other:": str(kwargs),
            }
        )
        if len(self.flash_history) > self._max_history:
            self.flash_history.pop(0)

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
