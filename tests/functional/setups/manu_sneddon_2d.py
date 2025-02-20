"""Model setup for the fracturized pressure problem using Sneddon's analytical solution.

References:

    - [1] Starfield, C.: Boundary Element Methods in Solid Mechanics (1983)
    - [2] Sneddon, I.N.: Fourier transforms (1951)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.geometry.distances import point_pointset


class SneddonExactSolution2d:
    """Class representing the analytical solution for the pressurized fracture problem."""

    def __init__(self, model: "ManuSneddonSetup2d"):
        self.p0 = model.params["p0"]
        self.theta_rad = model.params["theta_rad"]

        self.a = model.params["a"]
        self.shear_modulus = model.params["material_constants"]["solid"].shear_modulus
        self.poi = model.params["poi"]
        self.n = model.params["num_bem_segments"]
        self.length = model.domain_size

    def exact_sol_global(self, sd: pp.Grid) -> np.ndarray:
        """Compute the analytical solution for the pressurized fracture problem in
        question.

        Parameters:
            sd: Subdomain for which the analytical solution is to be computed.

        """

        h = 2 * self.a / self.n
        box_faces = sd.get_boundary_faces()
        u_bc = np.zeros((sd.dim, sd.num_faces))

        center = np.array([self.length / 2, self.length / 2, 0])
        bem_centers = self.get_bem_centers(h, center)
        eta = point_pointset(bem_centers, center)

        u_a = -self.analytical_displacements(eta)
        u_bc = self.displacement_bc_values_from_bem(
            sd, h / 2, box_faces, bem_centers, u_a
        )
        return u_bc

    def exact_sol_fracture(self, mdg: pp.MixedDimensionalGrid) -> np.ndarray:
        """Compute Sneddon's analytical solution for the pressurized crack
        problem in question.

        Parameters:
            mdg: Mixed-dimensional domain of the setup.

        Return:
            A numpy array with the analytical apertures.

        """

        ambient_dim = mdg.dim_max()
        g_1 = mdg.subdomains(dim=ambient_dim - 1)[0]
        fracture_center = np.array([self.length / 2, self.length / 2, 0])
        fracture_faces = g_1.cell_centers

        # See [2], p. 425, equ. 92
        eta = point_pointset(fracture_faces, fracture_center)
        apertures = self.analytical_displacements(eta)

        return apertures

    def get_bem_centers(self, h: float, center: np.ndarray) -> np.ndarray:
        """Compute coordinates of the centers of the BEM segments.

        Parameters:
            h: BEM segment length.
            center: Coordinates of the fracture center.

        Return:
            Array containing the BEM segment centers.

        """

        # See [1], p. 57 coordinate system (4.5.1)
        bem_centers = np.zeros((3, self.n))
        x_0 = center[0] - (self.a - 0.5 * h) * np.sin(self.theta_rad)
        y_0 = center[1] - (self.a - 0.5 * h) * np.cos(self.theta_rad)
        i = np.arange(self.n)
        bem_centers[0] = x_0 + i * h * np.sin(self.theta_rad)
        bem_centers[1] = y_0 + i * h * np.cos(self.theta_rad)

        return bem_centers

    def analytical_displacements(self, eta: np.ndarray) -> np.ndarray:
        """Compute Sneddon's analytical solution for the pressurized fracture displacement.

        Parameters:
            eta: Distances of fracture points to the fracture center.

        Return
            Array containing the analytical normal displacement jumps.

        """
        # See [2], p. 425 equ. 92
        cons = (1 - self.poi) / self.shear_modulus * self.p0 * self.a * 2
        return cons * np.sqrt(1 - np.power(eta / self.a, 2))

    def tranform_boundary_face_coordinates_bem(
        self, xc: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Translation and rotation transform of boundary face coordinates for the BEM.

        Parameters:
            xc: Coordinates of BEM segment centre.
            x: Coordinates of boundary faces.

        Return:
            Transformed coordinates.

        """
        alpha = np.pi / 2 - self.theta_rad
        x_bar = np.zeros_like(x)
        # See [1], p. 168 equ. 7.4.6
        x_bar[0, :] = (x[0, :] - xc[0]) * np.cos(alpha) + (x[1, :] - xc[1]) * np.sin(
            alpha
        )
        x_bar[1, :] = -(x[0, :] - xc[0]) * np.sin(alpha) + (x[1, :] - xc[1]) * np.cos(
            alpha
        )
        return x_bar

    def _get_bc_val_from_normal_displacement(
        self,
        sd: pp.Grid,
        bound_faces: np.ndarray,
        xf: np.ndarray,
        h: float,
        du: float,
    ) -> np.ndarray:
        """Computes semi-analytical displacement values on the boundary using the BEM for
        the Sneddon problem.

        Parameter
            sd: The matrix grid.
            bound_faces: Array of indices of boundary faces of ``sd``.
            xf: Coordinates of boundary faces.
            h: BEM segment length.
            du: Sneddon's analytical relative normal displacement.

        Return:
            Boundary values for the displacement.

        """
        # See [1], pages 57, 84-92, 168
        f2 = np.zeros(bound_faces.size)
        f3 = np.zeros(bound_faces.size)
        f4 = np.zeros(bound_faces.size)
        f5 = np.zeros(bound_faces.size)

        u = np.zeros((sd.dim, sd.num_faces))

        alpha = np.pi / 2 - self.theta_rad

        # See [1], equ. (7.4.5)
        m = 1 / (4 * np.pi * (1 - self.poi))
        f2[:] = m * (
            np.log(np.sqrt((xf[0, :] - h) ** 2 + xf[1] ** 2))
            - np.log(np.sqrt((xf[0, :] + h) ** 2 + xf[1] ** 2))
        )
        f3[:] = -m * (
            np.arctan2(xf[1, :], (xf[0, :] - h)) - np.arctan2(xf[1, :], (xf[0, :] + h))
        )

        # See [1], equations 5.5.3, 5.5.1, and p. 57, equ. 4.5.1
        f4[:] = m * (
            xf[1, :] / ((xf[0, :] - h) ** 2 + xf[1, :] ** 2)
            - xf[1, :] / ((xf[0, :] + h) ** 2 + xf[1, :] ** 2)
        )

        f5[:] = m * (
            (xf[0, :] - h) / ((xf[0, :] - h) ** 2 + xf[1, :] ** 2)
            - (xf[0, :] + h) / ((xf[0, :] + h) ** 2 + xf[1, :] ** 2)
        )

        u[0, bound_faces] = du * (
            -(1 - 2 * self.poi) * np.cos(alpha) * f2[:]
            - 2 * (1 - self.poi) * np.sin(alpha) * f3[:]
            - xf[1, :] * (np.cos(alpha) * f4[:] + np.sin(alpha) * f5[:])
        )
        u[1, bound_faces] = du * (
            -(1 - 2 * self.poi) * np.sin(alpha) * f2[:]
            + 2 * (1 - self.poi) * np.cos(alpha) * f3[:]
            - xf[1, :] * (np.sin(alpha) * f4[:] - np.cos(alpha) * f5[:])
        )

        return u

    def displacement_bc_values_from_bem(
        self,
        grid: pp.Grid,
        h: float,
        bound_faces: np.ndarray,
        bem_centers: np.ndarray,
        u_a: np.ndarray,
    ) -> np.ndarray:
        """Computes semi-analytical displacement values using the BEM for the Sneddon
        problem.

        Parameter
            grid: The matrix grid.
            h: BEM segment length.
            bound_faces: Array of indices of boundary faces of ``grid``.
            bem_centers: BEM segments centers.
            u_a: Sneddon's analytical relative normal displacement.

        Return:
            Semi-analytical boundary displacement values.

        """

        bc_val = np.zeros((grid.dim, grid.num_faces))
        bound_face_centers = grid.face_centers[:, bound_faces]

        for i in range(0, u_a.size):
            new_bound_face_centers = self.tranform_boundary_face_coordinates_bem(
                bem_centers[:, i], bound_face_centers
            )

            u_bound = self._get_bc_val_from_normal_displacement(
                grid, bound_faces, new_bound_face_centers, h, u_a[i]
            )

            bc_val += u_bound

        return bc_val


class ManuSneddonGeometry2d(SquareDomainOrthogonalFractures):
    """Square domain but with single line fracture."""

    def set_fractures(self):
        """Setting a single line fracture which runs through the domain center."""

        theta_rad = self.params["theta_rad"]
        a = self.params["a"]  # Half-length of the fracture
        height = length = self.params["domain_size"]
        frac_pts = self.compute_frac_pts(
            theta_rad=theta_rad, a=a, height=height, length=length
        )
        self._fractures = [pp.LineFracture(frac_pts)]

    @staticmethod
    def compute_frac_pts(
        theta_rad: float, a: float, height: float, length: float
    ) -> np.ndarray:
        """Assuming the fracture center is at the coordinate (height/2, length/2),
        compute the endpoints of a fracture given its orientation and fracture length.

        Parameters:
            theta_rad: Angle of the fracture in radians
            a: Half-length of the fracture.
            height: Height of the domain.
            length: Width of the domain.

        Returns:
            A 2x2 array where each column represents the coordinates of an end point of
            the fracture in 2D. The first column corresponds to one end point, and the
            second column corresponds to the other.

        """
        # Rotate the fracture with an angle theta_rad
        y_0 = height / 2 - a * np.cos(theta_rad)
        x_0 = length / 2 - a * np.sin(theta_rad)
        y_1 = height / 2 + a * np.cos(theta_rad)
        x_1 = length / 2 + a * np.sin(theta_rad)

        frac_pts = np.array([[x_0, y_0], [x_1, y_1]]).T
        return frac_pts


class ManuSneddonBoundaryConditions(pp.PorePyModel):
    exact_sol: SneddonExactSolution2d

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Set boundary condition type for the problem.

        Parameters:
            sd: Subdomain for which the boundary conditions are to be set.

        Returns:
            Boundary condition type for the problem.

        """
        bounds = self.domain_boundary_sides(sd)

        # Set the type of west and east boundaries to Dirichlet. North and south are
        # Neumann by default.
        bc = pp.BoundaryConditionVectorial(sd, bounds.all_bf, "dir")
        bc.internal_to_dirichlet(sd)

        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """This method sets the displacement boundary condition values for a given
        boundary grid using the Sneddon analytical solution through the BEM.

        Parameters:
            bg: The boundary grid for which the displacement boundary condition values
                are to be set.

        Returns:
            An array of displacement values on the boundary.

        """

        sd = bg.parent
        if sd.dim < 2:
            return np.zeros(self.nd * sd.num_faces)
        else:
            # Get the analytical solution for the displacement
            u_exact = self.exact_sol.exact_sol_global(sd)

            # Project the values to the grid
            return bg.projection(2) @ u_exact.ravel("F")


@dataclass
class ManuSneddonSaveData:
    """Data class for storing the error in the displacement field."""

    error_displacement: pp.number


class ManuSneddonDataSaving(pp.PorePyModel):
    """Class for saving the error in the displacement field."""

    exact_sol: SneddonExactSolution2d

    displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]

    def collect_data(self) -> ManuSneddonSaveData:
        """Collecting the error in the displacement field."""
        frac_sd = self.mdg.subdomains(dim=self.nd - 1)
        nd_vec_to_normal = self.normal_component(frac_sd)

        # Computing the numerical displacement jump along the fracture on the fracture.
        # cell centers.
        u_n = (nd_vec_to_normal @ self.displacement_jump(frac_sd)).value(
            self.equation_system
        )

        # Checking convergence specifically on the fracture.
        u_a = self.exact_sol.exact_sol_fracture(self.mdg)

        # Truncate the solution at the fracture cells close to the boundary.
        # How close is determined by the parameter eps (percent).
        eps = self.params["error_exclusion_zone_fracture_tips"]
        close_to_fracture_tips = self._index_fracture_cells_close_to_fracture_tips(eps)
        u_a[close_to_fracture_tips] = 0
        u_n[close_to_fracture_tips] = 0

        e = ConvergenceAnalysis.lp_error(
            frac_sd[0], u_a, u_n, is_scalar=False, is_cc=True, relative=True
        )

        collect_data = ManuSneddonSaveData(error_displacement=e)
        return collect_data

    def _index_fracture_cells_close_to_fracture_tips(self, eps: float) -> np.ndarray:
        """Compute the boolean mask array with `True` values for the fracture cells
        close to the fracture tips.

        Parameters:
            eps: The relative threshold to determine how close the fracture cells
                are to the fracture tips to be truncated, must be in `[0, 1]`.

        Returns:
            A boolean array of shape `(num_fracture_cells,)`.

        """
        frac_sd = self.mdg.subdomains(dim=self.nd - 1)
        assert len(frac_sd) == 1, "Must be only one fracture."

        # Find the domain center.
        xmax, xmin = self.domain.bounding_box["xmax"], self.domain.bounding_box["xmin"]
        ymax, ymin = self.domain.bounding_box["ymax"], self.domain.bounding_box["ymin"]
        xmean = (xmax - xmin) * 0.5
        ymean = (ymax - ymin) * 0.5

        # Compute distance from the domain center.
        x, y, _ = frac_sd[0].cell_centers
        distance_from_center = np.sqrt((x - xmean) ** 2 + (y - ymean) ** 2)

        # Compute the percentage of how close the cells are to the fracture tips.
        a = self.exact_sol.a  # Fracture half-length.
        relative_distance_from_center = (a - distance_from_center) / a
        return relative_distance_from_center < eps


class ManuSneddonConstitutiveLaws(pp.constitutive_laws.PressureStress):
    """Inherits the relevant pressure-stress relations, defines a constant pressure
    and sets the stress discretization to MPSA.

    Note that the latter is relevant because the inherited constitutive law
    uses the Biot discretization.

    """

    def pressure(self, domains: pp.SubdomainsOrBoundaries):
        """Defines a constant pressure given by ``params['p0']``.

        Parameters:
            subdomains: List of grids or boundary grids where the pressure is defined.

        Returns:
            A dense array broadcasting the constant pressure into a suitable array.

        """
        p = self.params["p0"] * np.ones(sum((grid.num_cells for grid in domains)))
        return pp.ad.DenseArray(p)

    def stress_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpsaAd:
        """Discretization class for the stress tensor.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            The MPSA discretization in Ad operator form.

        """
        return pp.ad.MpsaAd(self.stress_keyword, subdomains)


class ManuSneddonSetup2d(
    ManuSneddonGeometry2d,
    ManuSneddonDataSaving,
    ManuSneddonBoundaryConditions,
    ManuSneddonConstitutiveLaws,
    pp.MomentumBalance,
):
    """Complete Sneddon setup, including data collection and the analytical solution."""

    def __init__(self, params=None):
        super().__init__(params)
        self.exact_sol = SneddonExactSolution2d(self)
