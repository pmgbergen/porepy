"""Module containing a mixin class for reusing methods in verification setups."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

import porepy as pp
from porepy.models.constitutive_laws import LinearElasticMechanicalStress


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

    darcy_keyword: str
    """Keyword used to identify the Darcy flux discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`.

    """
    time_manager: pp.TimeManager
    """Time manager. Normally set by an instance of a subclass of
    :class:`porepy.models.solution_strategy.SolutionStrategy`.

    """

    units: pp.Units
    """Units object, containing the scaling of base magnitudes."""

    nd: int

    bc_type_mechanics: Callable[[pp.BoundaryGrid], np.ndarray]

    mechanical_stress: Callable[
        [pp.SubdomainsOrBoundaries], pp.ad.MixedDimensionalVariable
    ]

    _combine_boundary_operators: Callable[
        [
            Sequence[pp.Grid],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[pp.Grid], pp.BoundaryCondition],
            str,
            int,
        ],
        pp.ad.Operator,
    ]

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
        bc = LinearElasticMechanicalStress.combine_boundary_operators_mechanical_stress(
            self,  # type: ignore[arg-type]
            subdomains=[sd],
        )

        # Compute the pseudo-trace of the displacement
        # Note that this is not the real trace, as this only holds for particular
        # choices of boundary condtions
        u_faces_ad = (
            discr_mech.bound_displacement_cell() @ u
            + discr_mech.bound_displacement_face() @ bc
            + discr_poromech.bound_pressure(self.darcy_keyword) @ p
        )

        # Parse numerical value and return the minimum and maximum value
        u_faces = u_faces_ad.value(self.equation_system)
        assert isinstance(u_faces, np.ndarray)
        return u_faces
