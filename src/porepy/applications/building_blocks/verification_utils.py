"""Module containing a mixin class for reusing methods in verification setups."""
from __future__ import annotations

from typing import Callable

import numpy as np

import porepy as pp


class VerificationUtils:
    """Mixin class storing useful utility methods.

    The intended use is to mix this class with a utlilty class, specific to a
    verification/complete setup.

    """

    displacement: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Displacement variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """

    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of storing and scaling numerical values
    representing fluid-related quantities. Normally, this is set by an instance of
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """

    params: dict
    """Model parameters dictionary."""

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """

    solid: pp.SolidConstants
    """Solid constant object that takes care of storing and scaling numerical values
    representing solid-related quantities. Normally, this is set by an instance of
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    stress_keyword: str
    """Keyword for accessing the parameters of the mechanical subproblem."""

    time_dependent_bc_values_mechanics: Callable[[list[pp.Grid]], np.ndarray]
    """Values of the mechanical boundary conditions for a time-dependent problem.
    Normally set by a mixin instance of
    :class:`~porepy.models.poromechanics.BoundaryConditionsMechanicsTimeDependent`.

    """

    time_manager: pp.TimeManager
    """Time manager. Normally set by an instance of a subclass of
    :class:`porepy.models.solution_strategy.SolutionStrategy`.

    """

    units: pp.Units
    """Units object, containing the scaling of base magnitudes."""

    def face_displacement(self, sd: pp.Grid) -> np.ndarray:
        """Project the displacement vector onto the faces.

        Parameters:
            sd: subdomain grid.

        Returns:
            Numpy array containing the values of the displacement on the faces.

        Raises:
             Exception if the mixed-dimensional grid contains more that one subdomain
             or the dimension of the grid does not coincide with the maximum
             dimension of the mixed-dimensional grid.

        Note:
            This method should not be seen as a true trace of the displacement,
            since it only holds for certain choices of boundary conditions.

        """
        # Sanity check
        assert len(self.mdg.subdomains()) == 1 and sd.dim == self.mdg.dim_max()

        # Retrieve pressure and displacement
        u = self.displacement([sd])
        p = self.pressure([sd])

        # Discretization
        discr_mech = pp.ad.MpsaAd(self.stress_keyword, [sd])
        discr_poromech = pp.ad.BiotAd(self.stress_keyword, [sd])

        # Boundary conditions
        bc = pp.wrap_as_ad_array(self.time_dependent_bc_values_mechanics([sd]))

        # Compute the pseudo-trace of the displacement
        # Note that this is not the real trace, as this only holds for particular
        # choices of boundary condtions
        u_faces_ad = (
            discr_mech.bound_displacement_cell * u
            + discr_mech.bound_displacement_face * bc
            + discr_poromech.bound_pressure * p
        )

        # Parse numerical value and return the minimum and maximum value
        u_faces = u_faces_ad.evaluate(self.equation_system).val

        return u_faces

    def relative_l2_error(
        self,
        grid: pp.GridLike,
        true_array: np.ndarray,
        approx_array: np.ndarray,
        is_scalar: bool,
        is_cc: bool,
    ) -> pp.number:
        """Compute discrete relative L2-error as given in [1].

        Parameters:
            grid: Either a subdomain grid or a mortar grid.
            true_array: Array containing the true values of a given variable.
            approx_array: Array containing the approximate values of a given variable.
            is_scalar: Whether the variable is a scalar quantity. Use ``False`` for
                vector quantities. For example, ``is_scalar=True`` for pressure, whereas
                ``is_scalar=False`` for displacement.
            is_cc: Whether the variable is associated to cell centers. Use ``False``
                for variables associated to face centers. For example, ``is_cc=True``
                for pressures, whereas ``is_scalar=False`` for subdomain fluxes.

        Returns:
            Discrete relative L2-error between the true and approximated arrays.

        Raises:
            ValueError if a mortar grid is given and ``is_cc=False``.

        References:

            - [1] Nordbotten, J. M. (2016). Stable cell-centered finite volume
              discretization for Biot equations. SIAM Journal on Numerical Analysis,
              54(2), 942-968.

        """
        # Sanity check
        if isinstance(grid, pp.MortarGrid) and not is_cc:
            raise ValueError("Mortar variables can only be cell-centered.")

        # Obtain proper measure
        if is_cc:
            if is_scalar:
                meas = grid.cell_volumes
            else:
                meas = grid.cell_volumes.repeat(grid.dim)
        else:
            assert isinstance(grid, pp.Grid)
            if is_scalar:
                meas = grid.face_areas
            else:
                meas = grid.face_areas.repeat(grid.dim)

        # Compute error
        numerator = np.sqrt(np.sum(meas * np.abs(true_array - approx_array) ** 2))
        denominator = np.sqrt(np.sum(meas * np.abs(true_array) ** 2))

        return numerator / denominator
