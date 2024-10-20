"""
This module contains a code verification implementation for a manufactured solution for
the three-dimensional, compressible, single-phase flow with a single, fully embedded
vertical fracture in the middle of the domain.

The exact solution was obtained by extending the solution from the incompressible
case as given in Appendix D.2 from [1].

In particular, we have added a pressure-dependent density which obeys the following
constitutive relationship:

.. math::

    \\rho(p) = \rho_0 \\exp \\left[c_f (p - p_0) \\right],

where :math:`\\rho` and :math:`p` are the density and pressure, :math:`\\rho_0` and
:math:`p_0` are the density and pressure at reference states, and :math:`c_f` is the
fluid compressibility.

References:

    - [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu, F. A.
      (2022). A posteriori error estimates for hierarchical mixed-dimensional
      elliptic equations. Journal of Numerical Mathematics.

"""
from __future__ import annotations

import numpy as np
import sympy as sym

import porepy as pp
from tests.functional.setups.manu_flow_comp_2d_frac import (
    ManuCompFlowSetup2d,
    ManuCompSolutionStrategy2d,
)
from tests.functional.setups.manu_flow_incomp_frac_3d import (
    SingleEmbeddedVerticalPlaneFracture,
)


class ManuCompExactSolution3d:
    """Class containing the exact manufactured solution for the verification setup."""

    def __init__(self, setup):
        # Model setup
        self.setup = setup

        # Retrieve material constant from the setup
        rho_0 = self.setup.fluid.reference_component.density  # [kg * m^-3]  Reference fluid density
        p_0 = self.setup.fluid.reference_component.pressure  # [Pa] Reference fluid pressure
        c_f = self.setup.fluid.reference_component.compressibility  # [Pa^-1]  Fluid compressibility
        phi_0 = self.setup.solid.porosity  # [-] Reference porosity

        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Smoothness coefficient
        n = 1.5

        # Distance and bubble functions
        distance_fun = [
            ((x - 0.5) ** 2 + (y - 0.25) ** 2 + (z - 0.25) ** 2) ** 0.5,  # bottom front
            ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5,  # bottom middle
            ((x - 0.5) ** 2 + (y - 0.25) ** 2 + (z - 0.75) ** 2) ** 0.5,  # bottom back
            ((x - 0.5) ** 2 + (z - 0.25) ** 2) ** 0.5,  # front
            ((x - 0.5) ** 2) ** 0.5,  # middle
            ((x - 0.5) ** 2 + (z - 0.75) ** 2) ** 0.5,  # back
            ((x - 0.5) ** 2 + (y - 0.75) ** 2 + (z - 0.25) ** 2) ** 0.5,  # top front
            ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5,  # top middle
            ((x - 0.5) ** 2 + (y - 0.75) ** 2 + (z - 0.75) ** 2) ** 0.5,  # top back
        ]
        bubble_fun = (
            100 * (y - 0.25) ** 2 * (y - 0.75) ** 2 * (z - 0.25) ** 2 * (z - 0.75) ** 2
        )

        # Exact pressure in the matrix
        p_matrix = [
            t * (distance_fun[0] ** (1 + n)),  # bottom front
            t * (distance_fun[1] ** (1 + n)),  # bottom middle
            t * (distance_fun[2] ** (1 + n)),  # bottom back
            t * (distance_fun[3] ** (1 + n)),  # front
            t * (distance_fun[4] ** (1 + n) + bubble_fun * distance_fun[4]),  # middle
            t * (distance_fun[5] ** (1 + n)),  # back
            t * (distance_fun[6] ** (1 + n)),  # top front
            t * (distance_fun[7] ** (1 + n)),  # top middle
            t * (distance_fun[8] ** (1 + n)),  # top back
        ]

        # Exact Darcy flux in the matrix
        q_matrix = [
            [-sym.diff(p, x), -sym.diff(p, y), -sym.diff(p, z)] for p in p_matrix
        ]

        # Exact density in the matrix
        rho_matrix = [rho_0 * sym.exp(c_f * (p - p_0)) for p in p_matrix]

        # Exact mass flux in the matrix
        mf_matrix = [
            [rho * q[0], rho * q[1], rho * q[2]]
            for (rho, q) in zip(rho_matrix, q_matrix)
        ]

        # Exact divergence of the mass flux in the matrix
        div_mf_matrix = [
            sym.diff(mf[0], x) + sym.diff(mf[1], y) + sym.diff(mf[2], z)
            for mf in mf_matrix
        ]

        # Exact accumulation term in the matrix
        accum_matrix = [sym.diff(phi_0 * rho, t) for rho in rho_matrix]

        # Exact source term in the matrix
        f_matrix = [
            accum + div_mf for (accum, div_mf) in zip(accum_matrix, div_mf_matrix)
        ]

        # Exact flux on the interface (mortar fluxes)
        q_intf = t * bubble_fun

        # Exact density on the interface
        rho_intf = 1.0  # density evaluates to 1 at x = 0.5, 0.25 <= y <= 0.75

        # Exact mass flux on the interface
        mf_intf = rho_intf * q_intf

        # Exact pressure in the fracture
        p_frac = -t * bubble_fun

        # Exact Darcy flux in the fracture
        q_frac = [-sym.diff(p_frac, y), -sym.diff(p_frac, z)]

        # Exact density in the fracture
        rho_frac = rho_0 * sym.exp(c_f * (p_frac - p_0))

        # Exact mass flux in the fracture
        mf_frac = [rho_frac * q_frac[0], rho_frac * q_frac[1]]

        # Exact accumulation in the fracture
        accum_frac = sym.diff(phi_0 * rho_frac, t)

        # Exact divergence of the mass flux in the fracture
        div_mf_frac = sym.diff(mf_frac[0], y) + sym.diff(mf_frac[1], z)

        # Exact source in the fracture
        f_frac = accum_frac + div_mf_frac - 2 * mf_intf

        # Public attributes
        self.p_matrix = p_matrix
        self.q_matrix = q_matrix
        self.rho_matrix = rho_matrix
        self.mf_matrix = mf_matrix
        self.f_matrix = f_matrix

        self.p_frac = p_frac
        self.q_frac = q_frac
        self.rho_frac = rho_frac
        self.mf_frac = mf_frac
        self.f_frac = f_frac

        self.q_intf = q_intf
        self.rho_intf = rho_intf
        self.mf_intf = mf_intf

        # Private attributes
        self._bubble = bubble_fun

    def get_region_indices(
        self, where: str, on_boundary: bool = False
    ) -> list[np.ndarray]:
        """Get indices of the cells belonging to the different regions of the domain.

        Parameters:
            where: Use "cc" to evaluate at the cell centers and "fc" to evaluate at
                the face centers. Use "bg" to evaluate only at face centers related to
                the external domain boundary.

        Returns:
            List of length 9, containing the indices of the different regions of the
            domain.

        """
        # Sanity check
        assert where in ["cc", "fc", "bg"]

        # Retrieve coordinates
        sd = self.setup.mdg.subdomains()[0]
        if where == "cc":
            x = sd.cell_centers
        elif where == "fc":
            x = sd.face_centers
        else:
            bg = self.setup.mdg.subdomain_to_boundary_grid(sd)
            assert bg is not None
            x = bg.cell_centers

        # Get indices
        bottom_front = (x[1] < 0.25) & (x[2] < 0.25)
        bottom_middle = (x[1] < 0.25) & (x[2] > 0.25) & (x[2] < 0.75)
        bottom_back = (x[1] < 0.25) & (x[2] > 0.75)
        front = (x[1] > 0.25) & (x[1] < 0.75) & (x[2] < 0.25)
        middle = (x[1] >= 0.25) & (x[1] <= 0.75) & (x[2] >= 0.25) & (x[2] <= 0.75)
        back = (x[1] > 0.25) & (x[1] < 0.75) & (x[2] > 0.75)
        top_front = (x[1] > 0.75) & (x[2] < 0.25)
        top_middle = (x[1] > 0.75) & (x[2] > 0.25) & (x[2] < 0.75)
        top_back = (x[1] > 0.75) & (x[2] > 0.75)

        cell_idx = [
            bottom_front,
            bottom_middle,
            bottom_back,
            front,
            middle,
            back,
            top_front,
            top_middle,
            top_back,
        ]

        return cell_idx

    def matrix_pressure(self, sd_matrix: pp.Grid, time: pp.number) -> np.ndarray:
        """Evaluate exact matrix pressure [Pa] at the cell centers.

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the exact pressures at
            the cell centers at the given ``time``.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get list of cell indices
        cc = sd_matrix.cell_centers
        cell_idx = self.get_region_indices(where="cc")

        # Lambdify expression
        p_fun = [sym.lambdify((x, y, z, t), p, "numpy") for p in self.p_matrix]

        # Cell-centered pressures
        p_cc = np.zeros(sd_matrix.num_cells)
        for p, idx in zip(p_fun, cell_idx):
            p_cc += p(cc[0], cc[1], cc[2], time) * idx

        return p_cc

    def matrix_flux(self, sd_matrix: pp.Grid, time: pp.number) -> np.ndarray:
        """Evaluate exact matrix Darcy flux [m^3 * s^-1] at the face centers.

        Parameters:
            sd_matrix: Matrix grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd_matrix.num_faces, )`` containing the exact Darcy
            fluxes at the face centers at the given ``time``.

        Note:
            The returned fluxes are already scaled with ``sd_matrix.face_normals``.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get list of face indices
        fc = sd_matrix.face_centers
        face_idx = self.get_region_indices(where="fc")

        # Lambdify bubble function
        bubble_fun = sym.lambdify((y, z), self._bubble, "numpy")

        # Lambdify expression
        q_fun = [
            [
                sym.lambdify((x, y, z, t), q[0], "numpy"),
                sym.lambdify((x, y, z, t), q[1], "numpy"),
                sym.lambdify((x, y, z, t), q[2], "numpy"),
            ]
            for q in self.q_matrix
        ]

        # Computation of the fluxes in the middle region results in NaN on faces that
        # are outside the middle region. We therefore need to first evaluate the
        # middle region and then the other regions so that NaN faces outside the middle
        # region can be overwritten accordingly.
        fn = sd_matrix.face_normals
        q_fc = np.zeros(sd_matrix.num_faces)

        q_fun_sorted = q_fun.copy()
        q_fun_sorted.pop(4)
        q_fun_sorted.insert(0, q_fun[4])

        face_idx_sorted = face_idx.copy()
        face_idx_sorted.pop(4)
        face_idx_sorted.insert(0, face_idx[4])

        # Perform evaluations using the sorted list of exact Darcy velocities
        for q, idx in zip(q_fun_sorted, face_idx_sorted):
            q_fc[idx] = (
                q[0](fc[0][idx], fc[1][idx], fc[2][idx], time) * fn[0][idx]
                + q[1](fc[0][idx], fc[1][idx], fc[2][idx], time) * fn[1][idx]
                + q[2](fc[0][idx], fc[1][idx], fc[2][idx], time) * fn[2][idx]
            )

        # Now the only NaN faces should correspond to the internal boundaries, e.g.,
        # the ones at x = 0.5. To correct these values, we exploit the fact that
        # (rho_matrix * q_matrix) \dot n = (rho_intf * q_intf) holds in a continuous
        # sense. Furthermore, for our problem, rho_matrix = rho_intf = 1.0 at x = 0.5
        # and 0.25 <= y <= 0.75, 0.25 <= z <= 0.75 . Thus, the previous equality can
        # be simplified to q_matrix \dot n = q_intf on the
        # internal boundaries.

        # Here, we cannot use the face normals since we'll get wrong signs (not
        # entirely sure why). Instead, we multiply by the face area and the face sign.
        frac_faces = np.where(sd_matrix.tags["fracture_faces"])[0]
        q_fc[frac_faces] = (
            bubble_fun(fc[1][frac_faces], fc[2][frac_faces])
            * sd_matrix.face_areas[frac_faces]
            * sd_matrix.signs_and_cells_of_boundary_faces(frac_faces)[0]
        )

        return q_fc

    def matrix_source(self, sd_matrix: pp.Grid, time: pp.number) -> np.ndarray:
        """Compute exact integrated matrix source.

        Parameters:
            sd_matrix: Matrix grid.
            time: float

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the exact integrated
            sources at the given ``time``.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")
        cc = sd_matrix.cell_centers
        cell_idx = self.get_region_indices(where="cc")

        # Lambdify expression
        f_fun = [sym.lambdify((x, y, z, t), f, "numpy") for f in self.f_matrix]

        # Integrated cell-centered sources
        vol = sd_matrix.cell_volumes
        f_cc = np.zeros(sd_matrix.num_cells)
        for f, idx in zip(f_fun, cell_idx):
            f_cc += f(cc[0], cc[1], cc[2], time) * vol * idx

        return f_cc

    def fracture_pressure(self, sd_frac: pp.Grid, time: pp.number) -> np.ndarray:
        """Evaluate exact fracture pressure at the cell centers.

        Parameters:
            sd_frac: Fracture grid.
            time: time in seconds.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` containing the exact pressures at
            the cell centers at the given ``time``.

        """
        # Symbolic variable
        y, z, t = sym.symbols("y z t")

        # Cell centers
        cc = sd_frac.cell_centers

        # Lambdify expression
        p_fun = sym.lambdify((y, z, t), self.p_frac, "numpy")

        # Evaluate at the cell centers
        p_cc = p_fun(cc[1], cc[2], time)

        return p_cc

    def fracture_flux(self, sd_frac: pp.Grid, time: pp.number) -> np.ndarray:
        """Evaluate exact fracture Darcy flux at the face centers.

        Parameters:
            sd_frac: Fracture grid.
            time: float

        Returns:
            Array of ``shape=(sd_frac.num_faces, )`` containing the exact Darcy
            fluxes at the face centers at the given ``time``.

        Note:
            The returned fluxes are already scaled with ``sd_face.face_normals``.

        """
        # Symbolic variable
        y, z, t = sym.symbols("y z t")

        # Face centers and face normals
        fc = sd_frac.face_centers
        fn = sd_frac.face_normals

        # Lambdify expression
        q_fun = [sym.lambdify((y, z, t), q, "numpy") for q in self.q_frac]

        # Evaluate at the face centers and scale with face normals
        q_fc = (
            q_fun[0](fc[1], fc[2], time) * fn[1] + q_fun[1](fc[1], fc[2], time) * fn[2]
        )

        return q_fc

    def fracture_source(self, sd_frac: pp.Grid, time: pp.number) -> np.ndarray:
        """Compute exact integrated fracture source.

        Parameters:
            sd_frac: Fracture grid.
            time: time in seconds.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` containing the exact integrated
            sources for the given ``time``.

        """
        # Symbolic variable
        y, z, t = sym.symbols("y z t")

        # Cell centers and volumes
        cc = sd_frac.cell_centers
        vol = sd_frac.cell_volumes

        # Lambdify expression
        f_fun = sym.lambdify((y, z, t), self.f_frac, "numpy")

        # Evaluate and integrate
        f_cc = f_fun(cc[1], cc[2], time) * vol

        return f_cc

    def interface_flux(self, intf: pp.MortarGrid, time: pp.number) -> np.ndarray:
        """Compute exact mortar fluxes at the interface.

        Parameters:
            intf: Mortar grid.
            time: time in seconds.

        Returns:
            Array of ``shape=(intf.num_cells, )`` containing the exact mortar fluxes
            at the given ``time``.

        Note:
            The returned mortar fluxes are already scaled with ``intf.cell_volumes``.

        """
        # Symbolic variable
        y, z, t = sym.symbols("y z t")

        # Cell centers and volumes
        cc = intf.cell_centers
        vol = intf.cell_volumes
        # face_area = self.setup.mdg.subdomains()[1].face_areas[0]

        # Lambdify expression
        lmbda_fun = sym.lambdify((y, z, t), self.q_intf, "numpy")

        # Evaluate and integrate
        lmbda_cc = lmbda_fun(cc[1], cc[2], time) * vol

        return lmbda_cc

    def matrix_boundary_pressure(
        self,
        bg_matrix: pp.BoundaryGrid,
        time: pp.number,
    ) -> np.ndarray:
        """Exact pressure at the boundary faces.

        Parameters:
            bg_matrix: Boundary grid of the matrix.
            time: time in seconds.

        Returns:
            Array of ``shape=(sd_matrix.num_faces, )`` with the exact pressure values
            on the exterior boundary faces at the given ``time``.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get list of face indices
        fc = bg_matrix.cell_centers
        face_idx = self.get_region_indices(where="bg")

        # Lambdify expression
        p_fun = [sym.lambdify((x, y, z, t), p, "numpy") for p in self.p_matrix]

        # Boundary pressures
        p_bf = np.zeros(bg_matrix.num_cells)
        for p, idx in zip(p_fun, face_idx):
            p_bf += p(fc[0], fc[1], fc[2], time) * idx

        return p_bf


# -----> Solution strategy
class ManuCompSolutionStrategy3d(ManuCompSolutionStrategy2d):
    """Modified solution strategy for the verification setup."""

    exact_sol: ManuCompExactSolution3d
    """Exact solution object."""

    def __init__(self, params: dict):
        """Constructor of the class."""
        super().__init__(params)

        self.exact_sol: ManuCompExactSolution3d
        """Exact solution object."""

    def set_materials(self):
        super().set_materials()
        # Instantiate exact solution object
        self.exact_sol = ManuCompExactSolution3d(self)


# -----> Mixer
class ManuCompFlowSetup3d(  # type: ignore[misc]
    SingleEmbeddedVerticalPlaneFracture,
    ManuCompSolutionStrategy3d,
    ManuCompFlowSetup2d,
):
    """
    Mixer class for the 3d compressible flow setup with a single fracture.
    """
