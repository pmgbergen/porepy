from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.geometry.distances import point_pointset


def compute_eta(pointset_centers: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Compute the distance from fracture points to the fracture centre.

    Parameter:
        pointset_centers: Coordinates of fracture points.
        center: Coordinates of the fracture center.

    Return:
        Array of distances each point in pointset to the center.

    """
    return point_pointset(pointset_centers, center)


def get_bem_centers(
    a: float, h: float, n: int, theta: float, center: np.ndarray
) -> np.ndarray:
    """Compute coordinates of the centers of the BEM segments.

    Parameters:
        a: Half of the fracture length.
        h: BEM segment length.
        n: Number of BEM segments.
        theta: Orientation of the fracture.
        center: Coordinates of the fracture center.

    Return:
        Array containing the BEM segment centers.

    """

    # Coordinate system (4.5.1) in page 57 in in book
    # Crouch Starfield 1983 Boundary Element Methods in Solid Mechanics

    bem_centers = np.zeros((3, n))
    x_0 = center[0] - (a - 0.5 * h) * np.sin(theta)
    y_0 = center[1] - (a - 0.5 * h) * np.cos(theta)
    for i in range(0, n):
        bem_centers[0, i] = x_0 + i * h * np.sin(theta)
        bem_centers[1, i] = y_0 + i * h * np.cos(theta)

    return bem_centers


def analytical_displacements(
    a: float, eta: np.ndarray, p0: float, G: float, poi: float
) -> np.ndarray:
    """Compute Sneddon's analytical solution for the pressurized fracture displacement.

    References:
        Sneddon Fourier transforms 1951 page 425 eq 92

    Parameters:
        a: Half of the fracture length.
        eta: Distances of fracture points to the fracture center.
        p0: Constant, fixed pressure.
        G: Shear modulus.
        poi: Poisson ratio.

    Return
        Array containing the analytical normal displacement jumps.

    """
    cons = (1 - poi) / G * p0 * a * 2
    return cons * np.sqrt(1 - np.power(eta / a, 2))


def transform(xc: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """Translation and rotation transform of boundary face coordinates for the BEM.

    Parameters:
        xc: Coordinates of BEM segment centre.
        x: Coordinates of boundary faces.
        alpha: Fracture orientation.

    Return:
        Transformed coordinates.

    """
    x_bar = np.zeros_like(x)
    # Terms in (7.4.6) in
    # Crouch Starfield 1983 Boundary Element Methods in Solid Mechanics page 168
    x_bar[0, :] = (x[0, :] - xc[0]) * np.cos(alpha) + (x[1, :] - xc[1]) * np.sin(alpha)
    x_bar[1, :] = -(x[0, :] - xc[0]) * np.sin(alpha) + (x[1, :] - xc[1]) * np.cos(alpha)
    return x_bar


def get_bc_val(
    grid: pp.Grid,
    bound_faces: np.ndarray,
    xf: np.ndarray,
    h: float,
    poi: float,
    alpha: float,
    du: float,
) -> np.ndarray:
    """Computes semi-analytical displacement values on the boundary using the BEM for
    the Sneddon problem.

    Parameter
        grid: The matrix grid.
        bound_faces: Array of indices of boundary faces of ``grid``.
        xf: Coordinates of boundary faces.
        h: BEM segment length.
        poi: Poisson ratio.
        alpha: Fracture orientation.
        du: Sneddon's analytical relative normal displacement.

    Return:
        Boundary values for the displacement.

    """

    # Equations for f2,f3,f4.f5 can be found in book
    # Crouch Starfield 1983 Boundary Element Methods in Solid Mechanics pages 57, 84-92, 168
    f2 = np.zeros(bound_faces.size)
    f3 = np.zeros(bound_faces.size)
    f4 = np.zeros(bound_faces.size)
    f5 = np.zeros(bound_faces.size)

    u = np.zeros((grid.dim, grid.num_faces))

    # Constant term in (7.4.5)
    m = 1 / (4 * np.pi * (1 - poi))

    # Second term in (7.4.5)
    f2[:] = m * (
        np.log(np.sqrt((xf[0, :] - h) ** 2 + xf[1] ** 2))
        - np.log(np.sqrt((xf[0, :] + h) ** 2 + xf[1] ** 2))
    )

    # First term in (7.4.5)
    f3[:] = -m * (
        np.arctan2(xf[1, :], (xf[0, :] - h)) - np.arctan2(xf[1, :], (xf[0, :] + h))
    )

    # The following equalities can be found on page 91 where f3,f4 as equation (5.5.3) and ux, uy in (5.5.1)
    # Also this is in the coordinate system described in (4.5.1) at page 57, which essentially is a rotation.
    f4[:] = m * (
        xf[1, :] / ((xf[0, :] - h) ** 2 + xf[1, :] ** 2)
        - xf[1, :] / ((xf[0, :] + h) ** 2 + xf[1, :] ** 2)
    )

    f5[:] = m * (
        (xf[0, :] - h) / ((xf[0, :] - h) ** 2 + xf[1, :] ** 2)
        - (xf[0, :] + h) / ((xf[0, :] + h) ** 2 + xf[1, :] ** 2)
    )

    u[0, bound_faces] = du * (
        -(1 - 2 * poi) * np.cos(alpha) * f2[:]
        - 2 * (1 - poi) * np.sin(alpha) * f3[:]
        - xf[1, :] * (np.cos(alpha) * f4[:] + np.sin(alpha) * f5[:])
    )
    u[1, bound_faces] = du * (
        -(1 - 2 * poi) * np.sin(alpha) * f2[:]
        + 2 * (1 - poi) * np.cos(alpha) * f3[:]
        - xf[1, :] * (np.sin(alpha) * f4[:] - np.cos(alpha) * f5[:])
    )

    return u


def assign_bem(
    grid: pp.Grid,
    h: float,
    bound_faces: np.ndarray,
    alpha: float,
    bem_centers: np.ndarray,
    u_a: np.ndarray,
    poi: float,
) -> np.ndarray:
    """Computes semi-analytical displacement values using the BEM for the Sneddon
    problem.

    Parameter
        grid: The matrix grid.
        h: BEM segment length.
        bound_faces: Array of indices of boundary faces of ``grid``.
        alpha: Fracture orientation.
        bem_centers: BEM segments centers.
        u_a: Sneddon's analytical relative normal displacement.
        poi: Poisson ratio.

    Return:
        Semi-analytical boundary displacement values.

    """

    bc_val = np.zeros((grid.dim, grid.num_faces))

    alpha = np.pi / 2 - alpha

    bound_face_centers = grid.face_centers[:, bound_faces]

    for i in range(0, u_a.size):
        new_bound_face_centers = transform(bem_centers[:, i], bound_face_centers, alpha)

        u_bound = get_bc_val(
            grid, bound_faces, new_bound_face_centers, h, poi, alpha, u_a[i]
        )

        bc_val += u_bound

    return bc_val


class ManuSneddonExactSolution2d:
    """Class representing the analytical solution for the pressurized fracture problem."""

    def __init__(self, setup: dict):
        self.p0 = setup.get("p0")
        self.theta = setup.get("theta")

        self.a = setup.get("a")
        self.shear_modulus = setup.get("material_constants").get("solid").shear_modulus
        self.poi = setup.get("poi")
        self.length = setup.get("length")
        self.height = setup.get("height")

    def exact_sol_global(self, sd: pp.Grid) -> np.ndarray:
        """Compute the analytical solution for the pressurized fracture problem in
        question.

        Parameters:
            sd: Subdomain for which the analytical solution is to be computed.

        """

        n = 1000
        h = 2 * self.a / n
        box_faces = sd.get_boundary_faces()
        u_bc = np.zeros((sd.dim, sd.num_faces))

        center = np.array([self.length / 2, self.height / 2, 0])
        bem_centers = get_bem_centers(self.a, h, n, self.theta, center)
        eta = compute_eta(bem_centers, center)

        u_a = -analytical_displacements(
            self.a, eta, self.p0, self.shear_modulus, self.poi
        )
        u_bc = assign_bem(sd, h / 2, box_faces, self.theta, bem_centers, u_a, self.poi)
        return u_bc

    def exact_sol_fracture(
        self, mdg: pp.MixedDimensionalGrid
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Sneddon's analytical solution for the pressurized crack
        problem in question.

        References:
            Sneddon Fourier transforms 1951 page 425 eq 92

        Parameters:
            mdg: Mixed-dimensional domain of the setup.

        Return:
            A 2-tuple of numpy arrays, where the first vector contains the distances
            between fracture center to each fracture coordinate, and the second vector
            the respective analytical apertures.

        """
        ambient_dim = mdg.dim_max()
        g_1 = mdg.subdomains(dim=ambient_dim - 1)[0]
        fracture_center = np.array([self.length / 2, self.height / 2, 0])

        fracture_faces = g_1.cell_centers

        # compute distances from fracture centre with its corresponding apertures
        eta = compute_eta(fracture_faces, fracture_center)
        apertures = analytical_displacements(
            self.a, eta, self.p0, self.shear_modulus, self.poi
        )

        return eta, apertures


class ManuSneddonGeometry2d(pp.PorePyModel):
    def set_domain(self) -> None:
        """Defining a two-dimensional rectangle with parametrizable ``'length'`` and
        ``'height'`` in the model parameters."""

        self._domain = pp.Domain(
            bounding_box={
                "xmin": 0,
                "ymin": 0,
                "xmax": self.params["length"],
                "ymax": self.params["height"],
            }
        )

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        """Choosing the grid type for our domain."""
        return self.params.get("grid_type", "simplex")

    def set_fractures(self):
        """Setting a single line fracture in the domain using ``params['frac_pts']``."""

        points = self.params["frac_pts"]
        self._fractures = [pp.LineFracture(points)]


class ManuSneddonBoundaryConditions(pp.PorePyModel):
    exact_sol: ManuSneddonExactSolution2d

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
        frac_face = sd.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True

        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """This method sets the displacement boundary condition values for a given
        boundary grid using the Sneddon analytical solution through the Boundary
        Element Method (BEM).

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

    exact_sol: ManuSneddonExactSolution2d

    displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]

    def collect_data(self) -> ManuSneddonSaveData:
        """Collecting the error in the displacement field."""
        frac_sd = self.mdg.subdomains(dim=self.nd - 1)
        nd_vec_to_normal = self.normal_component(frac_sd)

        # Computing the numerical displacement jump along the fracture on the fracture
        # cell centers.
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(frac_sd)
        u_n = u_n.value(self.equation_system)

        # Checking convergence specifically on the fracture
        u_a = self.exact_sol.exact_sol_fracture(self.mdg)[1]

        e = ConvergenceAnalysis.lp_error(
            frac_sd[0], u_a, u_n, is_scalar=False, is_cc=True, relative=True
        )

        collect_data = ManuSneddonSaveData(error_displacement=e)
        return collect_data


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

    exact_sol: ManuSneddonExactSolution2d

    def set_materials(self):
        """Initiating additionally the exact solution."""
        super().set_materials()
        self.exact_sol = ManuSneddonExactSolution2d(self.params)
