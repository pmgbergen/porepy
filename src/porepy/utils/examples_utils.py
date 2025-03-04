"""Module containing a mixin class for reusing methods in verification setups."""

from __future__ import annotations

from typing import Callable

import numpy as np

import porepy as pp
from porepy.models.constitutive_laws import LinearElasticMechanicalStress


class VerificationUtils(pp.PorePyModel):
    """Mixin class storing useful utility methods.

    The intended use is to mix this class with a utlilty class, specific to a
    verification/complete model.

    """

    displacement: Callable[[pp.SubdomainsOrBoundaries], pp.ad.MixedDimensionalVariable]
    """Displacement variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """

    stress_keyword: str
    """Keyword for accessing the parameters of the mechanical subproblem."""

    darcy_keyword: str
    """Keyword used to identify the Darcy flux discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`.

    """

    bc_type_mechanics: Callable[[pp.BoundaryGrid], np.ndarray]

    mechanical_stress: Callable[
        [pp.SubdomainsOrBoundaries], pp.ad.MixedDimensionalVariable
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
        u_faces = self.equation_system.evaluate(u_faces_ad)
        assert isinstance(u_faces, np.ndarray)
        return u_faces
