"""
This module contains a code verification implementation using a manufactured solution
for the three-dimensional, incompressible, single phase flow with a single,
fully embedded vertical fracture in the middle of the domain.

Details regarding the manufactured solution can be found in Appendix D.2 from [1].

References:

    - [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu, F. A.
      (2022). A posteriori error estimates for hierarchical mixed-dimensional
      elliptic equations. Journal of Numerical Mathematics.

"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import sympy as sym

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from tests.functional.setups.manu_flow_incomp_frac_2d import (
    ManuIncompFlowModel2d,
    ManuIncompSolutionStrategy2d,
)

# PorePy typings
number = pp.number
grid = pp.GridLike


# -----> Exact solution
class ManuIncompExactSolution3d:
    """Class containing the exact manufactured solution for the verification model."""

    def __init__(self, model):
        self.model = model

        # Symbolic variables
        x, y, z = sym.symbols("x y z")

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
            1e6 * (y - 0.25) ** 2 * (y - 0.75) ** 2 * (z - 0.25) ** 2 * (z - 0.75) ** 2
        )

        # Exact pressure in the matrix
        p_matrix = [
            distance_fun[0] ** (1 + n),  # bottom front
            distance_fun[1] ** (1 + n),  # bottom middle
            distance_fun[2] ** (1 + n),  # bottom back
            distance_fun[3] ** (1 + n),  # front
            distance_fun[4] ** (1 + n) + bubble_fun * distance_fun[4],  # middle
            distance_fun[5] ** (1 + n),  # back
            distance_fun[6] ** (1 + n),  # top front
            distance_fun[7] ** (1 + n),  # top middle
            distance_fun[8] ** (1 + n),  # top back
        ]

        # Exact Darcy flux in the matrix
        q_matrix = [
            [-sym.diff(p, x), -sym.diff(p, y), -sym.diff(p, z)] for p in p_matrix
        ]

        # Exact divergence of the Darcy flux in the matrix
        f_matrix = [
            sym.diff(q[0], x) + sym.diff(q[1], y) + sym.diff(q[2], z) for q in q_matrix
        ]

        # Exact flux on the interface (mortar fluxes)
        q_intf = bubble_fun

        # Exact pressure in the fracture
        p_frac = -bubble_fun

        # Exact Darcy flux in the fracture
        q_frac = [-sym.diff(p_frac, y), -sym.diff(p_frac, z)]

        # Exact source in the fracture
        f_frac = sym.diff(q_frac[0], y) + sym.diff(q_frac[1], z) - 2 * q_intf

        # Public attributes
        self.p_matrix = p_matrix
        self.q_matrix = q_matrix
        self.f_matrix = f_matrix

        self.p_frac = p_frac
        self.q_frac = q_frac
        self.f_frac = f_frac

        self.q_intf = q_intf

        # Private attributes
        self._bubble = bubble_fun

    def get_region_indices(self, where: str) -> list[np.ndarray]:
        """Get indices of the cells belonging to the different regions of the domain.

        Parameters:
            where: Use "cc" to evaluate at the cell centers, "fc" to evaluate at
                the face centers and "bg" to evaluate at the face centers associated
                with the external boundary

        Returns:
            List of length 9, containing the indices of the different regions of the
            domain.

        """
        # Sanity check
        assert where in ["cc", "fc", "bg"]

        # Retrieve coordinates
        sd = self.model.mdg.subdomains()[0]
        if where == "cc":
            x = sd.cell_centers
        elif where == "fc":
            x = sd.face_centers
        else:
            bg = self.model.mdg.subdomain_to_boundary_grid(sd)
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

    def matrix_pressure(self, sd_matrix: pp.Grid) -> np.ndarray:
        """Evaluate exact matrix pressure [Pa] at the cell centers.

        Parameters:
            sd_matrix: Matrix grid.

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the exact pressures at
            the cell centers.

        """
        # Symbolic variables
        x, y, z = sym.symbols("x y z")

        # Get list of cell indices
        cc = sd_matrix.cell_centers
        cell_idx = self.get_region_indices(where="cc")

        # Lambdify expression
        p_fun = [sym.lambdify((x, y, z), p, "numpy") for p in self.p_matrix]

        # Cell-centered pressures
        p_cc = np.zeros(sd_matrix.num_cells)
        for p, idx in zip(p_fun, cell_idx):
            p_cc += p(cc[0], cc[1], cc[2]) * idx

        return p_cc

    def matrix_flux(self, sd_matrix: pp.Grid) -> np.ndarray:
        """Evaluate exact matrix Darcy flux [m^3 * s^-1] at the face centers.

        Parameters:
            sd_matrix: Matrix grid.

        Returns:
            Array of ``shape=(sd_matrix.num_faces, )`` containing the exact Darcy
            fluxes at the face centers.

        Note:
            The returned fluxes are already scaled with ``sd_matrix.face_normals``.

        """
        # Symbolic variables
        x, y, z = sym.symbols("x y z")

        # Get list of face indices
        fc = sd_matrix.face_centers
        face_idx = self.get_region_indices(where="fc")

        # Lambdify bubble function
        bubble_fun = sym.lambdify((y, z), self._bubble, "numpy")

        # Lambdify expression
        q_fun = [
            [
                sym.lambdify((x, y, z), q[0], "numpy"),
                sym.lambdify((x, y, z), q[1], "numpy"),
                sym.lambdify((x, y, z), q[2], "numpy"),
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
                q[0](fc[0][idx], fc[1][idx], fc[2][idx]) * fn[0][idx]
                + q[1](fc[0][idx], fc[1][idx], fc[2][idx]) * fn[1][idx]
                + q[2](fc[0][idx], fc[1][idx], fc[2][idx]) * fn[2][idx]
            )

        # Now the only NaN faces should correspond to the internal boundaries, e.g.,
        # the ones at x = 0.5. To correct these values, we exploit the fact that
        # (rho_matrix * q_matrix) \dot n = (rho_intf * q_intf) holds in a continuous
        # sense. Furthermore, for our problem, rho_matrix = rho_intf = 1.0 at x = 0.5
        # and 0.25 <= y <= 0.75, 0.25 <= z <= 0.75 . Thus, the previous equality can
        # be simplified to q_matrix \dot n = q_intf on the internal boundaries.

        # Here, we cannot use the face normals since we'll get wrong signs (not
        # entirely sure why). Instead, we multiply by the face area and the face sign.
        frac_faces = np.where(sd_matrix.tags["fracture_faces"])[0]
        q_fc[frac_faces] = (
            bubble_fun(fc[1][frac_faces], fc[2][frac_faces])
            * sd_matrix.face_areas[frac_faces]
            * sd_matrix.signs_and_cells_of_boundary_faces(frac_faces)[0]
        )

        return q_fc

    def matrix_source(self, sd_matrix: pp.Grid) -> np.ndarray:
        """Compute exact integrated matrix source.

        Parameters:
            sd_matrix: Matrix grid.

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the exact integrated
            sources.

        """
        # Symbolic variables
        x, y, z = sym.symbols("x y z")
        cc = sd_matrix.cell_centers
        cell_idx = self.get_region_indices(where="cc")

        # Lambdify expression
        f_fun = [sym.lambdify((x, y, z), f, "numpy") for f in self.f_matrix]

        # Integrated cell-centered sources
        vol = sd_matrix.cell_volumes
        f_cc = np.zeros(sd_matrix.num_cells)
        for f, idx in zip(f_fun, cell_idx):
            f_cc += f(cc[0], cc[1], cc[2]) * vol * idx

        return f_cc

    def fracture_pressure(self, sd_frac: pp.Grid) -> np.ndarray:
        """Evaluate exact fracture pressure at the cell centers.

        Parameters:
            sd_frac: Fracture grid.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` containing the exact pressures at
            the cell centers.

        """
        # Symbolic variable
        y, z = sym.symbols("y z")

        # Cell centers
        cc = sd_frac.cell_centers

        # Lambdify expression
        p_fun = sym.lambdify((y, z), self.p_frac, "numpy")

        # Evaluate at the cell centers
        p_cc = p_fun(cc[1], cc[2])

        return p_cc

    def fracture_flux(
        self,
        sd_frac: pp.Grid,
    ) -> np.ndarray:
        """Evaluate exact fracture Darcy flux at the face centers.

        Parameters:
            sd_frac: Fracture grid.

        Returns:
            Array of ``shape=(sd_frac.num_faces, )`` containing the exact Darcy
            fluxes at the face centers.

        Note:
            The returned fluxes are already scaled with ``sd_face.face_normals``.

        """
        # Symbolic variable
        y, z = sym.symbols("y z")

        # Face centers and face normals
        fc = sd_frac.face_centers
        fn = sd_frac.face_normals

        # Lambdify expression
        q_fun = [sym.lambdify((y, z), q, "numpy") for q in self.q_frac]

        # Evaluate at the face centers and scale with face normals
        q_fc = q_fun[0](fc[1], fc[2]) * fn[1] + q_fun[1](fc[1], fc[2]) * fn[2]

        return q_fc

    def fracture_source(self, sd_frac: pp.Grid) -> np.ndarray:
        """Compute exact integrated fracture source.

        Parameters:
            sd_frac: Fracture grid.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` containing the exact integrated
            sources.

        """
        # Symbolic variable
        y, z = sym.symbols("y z")

        # Cell centers and volumes
        cc = sd_frac.cell_centers
        vol = sd_frac.cell_volumes

        # Lambdify expression
        f_fun = sym.lambdify((y, z), self.f_frac, "numpy")

        # Evaluate and integrate
        f_cc = f_fun(cc[1], cc[2]) * vol

        return f_cc

    def interface_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        """Compute exact mortar fluxes at the interface.

        Parameters:
            intf: Mortar grid.

        Returns:
            Array of ``shape=(intf.num_cells, )`` containing the exact mortar fluxes.

        Note:
            The returned mortar fluxes are already scaled with ``intf.cell_volumes``.

        """
        # Symbolic variable
        y, z = sym.symbols("y z")

        # Cell centers and volumes
        cc = intf.cell_centers
        vol = intf.cell_volumes

        # Lambdify expression
        lmbda_fun = sym.lambdify((y, z), self.q_intf, "numpy")

        # Evaluate and "integrate"
        lmbda_cc = lmbda_fun(cc[1], cc[2]) * vol

        return lmbda_cc

    def boundary_values(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Exact pressure at the boundary faces.

        Parameters:
            bg: Matrix boundary grid.

        Returns:
            Array of ``shape=(bg.num_cells, )`` with the exact
            pressure values at the exterior boundary faces.

        """
        # Symbolic variables
        x, y, z = sym.symbols("x y z")

        # Get list of face indices
        fc = bg.cell_centers
        face_idx = self.get_region_indices(where="bg")

        # Lambdify expression
        p_fun = [sym.lambdify((x, y, z), p, "numpy") for p in self.p_matrix]

        # Boundary pressures
        p_bf = np.zeros(bg.num_cells)
        for p, idx in zip(p_fun, face_idx):
            p_bf += p(fc[0], fc[1], fc[2]) * idx

        return p_bf


# -----> Geometry
class SingleEmbeddedVerticalPlaneFracture:
    """Generate fracture network and mixed-dimensional grid."""

    params: dict
    """Simulation model parameters."""

    grid_type: Callable[[], Literal["cartesian", "simplex", "tensor_grid"]]
    """Type of grid."""

    def set_fractures(self) -> None:
        """Declare set of fractures.

        Note:
            The physical fracture is `physical_frac_0`. For simplices, fractures
            `ghost_frac_0` ... `ghost_frac_23` are constraints for the meshing process.

        """

        physical_frac_0 = pp.PlaneFracture(
            np.array(
                [
                    [0.50, 0.50, 0.50, 0.50],
                    [0.25, 0.25, 0.75, 0.75],
                    [0.25, 0.75, 0.75, 0.25],
                ]
            )
        )

        if self.grid_type() == "simplex":
            ghost_frac_0 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.75, 0.75, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )
            )
            ghost_frac_1 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.75, 0.75, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )
            )
            ghost_frac_2 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.75, 0.75, 0.25, 0.25],
                        [0.75, 0.75, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_3 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.75, 0.75, 0.25, 0.25],
                        [0.75, 0.75, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_4 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.50, 0.00, 0.00],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.75, 0.75, 0.25],
                    ]
                )
            )
            ghost_frac_5 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 1.00, 0.50, 0.50],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.75, 0.75, 0.25],
                    ]
                )
            )
            ghost_frac_6 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.50, 0.00, 0.00],
                        [0.75, 0.75, 0.75, 0.75],
                        [0.25, 0.75, 0.75, 0.25],
                    ]
                )
            )
            ghost_frac_7 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 1.00, 0.50, 0.50],
                        [0.75, 0.75, 0.75, 0.75],
                        [0.25, 0.75, 0.75, 0.25],
                    ]
                )
            )
            ghost_frac_8 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.75, 0.75, 1.00, 1.00],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )
            )
            ghost_frac_9 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.75, 0.75, 1.00, 1.00],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )
            )
            ghost_frac_10 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.75, 0.75, 1.00, 1.00],
                        [0.75, 0.75, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_11 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.75, 0.75, 1.00, 1.00],
                        [0.75, 0.75, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_12 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.25, 0.25, 0.00, 0.00],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )
            )
            ghost_frac_13 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.25, 0.25, 0.00, 0.00],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )
            )
            ghost_frac_14 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.25, 0.25, 0.00, 0.00],
                        [0.75, 0.75, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_15 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.25, 0.25, 0.00, 0.00],
                        [0.75, 0.75, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_16 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.75, 0.75, 0.75, 0.75],
                        [0.25, 0.25, 0.00, 0.00],
                    ]
                )
            )
            ghost_frac_17 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.75, 0.75, 0.75, 0.75],
                        [0.25, 0.25, 0.00, 0.00],
                    ]
                )
            )
            ghost_frac_18 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.75, 0.75, 0.75, 0.75],
                        [1.00, 1.00, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_19 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.75, 0.75, 0.75, 0.75],
                        [1.00, 1.00, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_20 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.00, 0.00],
                    ]
                )
            )
            ghost_frac_21 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.00, 0.00],
                    ]
                )
            )
            ghost_frac_22 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.25, 0.25, 0.25, 0.25],
                        [1.00, 1.00, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_23 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.25, 0.25, 0.25, 0.25],
                        [1.00, 1.00, 0.75, 0.75],
                    ]
                )
            )

            self._fractures = [
                physical_frac_0,
                ghost_frac_0,
                ghost_frac_1,
                ghost_frac_2,
                ghost_frac_3,
                ghost_frac_4,
                ghost_frac_5,
                ghost_frac_6,
                ghost_frac_7,
                ghost_frac_8,
                ghost_frac_9,
                ghost_frac_10,
                ghost_frac_11,
                ghost_frac_12,
                ghost_frac_13,
                ghost_frac_14,
                ghost_frac_15,
                ghost_frac_16,
                ghost_frac_17,
                ghost_frac_18,
                ghost_frac_19,
                ghost_frac_20,
                ghost_frac_21,
                ghost_frac_22,
                ghost_frac_23,
            ]

        elif self.grid_type() == "cartesian":
            self._fractures = [physical_frac_0]
        else:
            raise NotImplementedError()

    def set_domain(self) -> None:
        """Set domain"""
        self._domain = nd_cube_domain(3, 1.0)

    def meshing_arguments(self) -> dict[str, float]:
        """Define mesh arguments for meshing."""
        return self.params.get("meshing_arguments", {"cell_size": 0.125})

    def meshing_kwargs(self) -> dict:
        """Declare meshing constraints. Ignore ghost fractures 1 to 24."""
        kw_args = {}
        if self.grid_type() == "simplex":
            kw_args.update({"constraints": np.arange(1, 25)})
        return kw_args


# -----> Solution strategy
class ManuIncompSolutionStrategy3d(ManuIncompSolutionStrategy2d):
    """Modified solution strategy for the verification model."""

    exact_sol: ManuIncompExactSolution3d
    """Exact solution object."""

    def __init__(self, params: dict):
        """Constructor for the class."""

        super().__init__(params)

        self.exact_sol: ManuIncompExactSolution3d
        """Exact solution object."""

    def set_materials(self):
        """Set material constants for the verification model."""
        super().set_materials()
        # Instantiate exact solution object
        self.exact_sol = ManuIncompExactSolution3d(self)


# -----> Mixer
class ManuIncompFlowModel3d(  # type: ignore[misc]
    SingleEmbeddedVerticalPlaneFracture,
    ManuIncompSolutionStrategy3d,
    ManuIncompFlowModel2d,
):
    """
    Mixer class for the 3d incompressible flow model with a single fracture.
    """
