import numpy as np
import porepy as pp
from porepy.viz.data_saving_model_mixin import VerificationDataSaving
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.models.protocol import PorePyModel

from dataclasses import dataclass

def compute_eta(pointset_centers: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Compute the distance fracture points to the fracture centre.

    Parameter::
    pointset: Array containing coordinates on the fracture
    center: fracture centre

    Return:
        Array of distances each point in pointset to the center

    """
    return pp.geometry.distances.point_pointset(pointset_centers, center)


def get_bem_centers(
    a: float, h: float, n: int, theta: float, center: np.ndarray
) -> np.ndarray:
    """
    Compute coordinates of the centers of the bem segments

    Parameters:
        a: half fracture length
        h: bem segment length
        n: number of bem segments
        theta: orientation of the fracture
        center: center of the fracture

    Return:
        Array of BEM centers
    """

    # Coordinate system (4.5.1) in page 57 in in book Crouch Starfield 1983 Boundary Element Methods in Solid Mechanics

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
    """
    Compute Sneddon's analytical solution for the pressurized fracture.

    Source: Sneddon Fourier transforms 1951 page 425 eq 92

    Parameter:
        a: half fracture length
        eta: distance from fracture centre
        p0: pressure
        G: shear modulus
        poi: poisson ratio

    Return
        List of analytical normal displacement jumps.
    """
    cons = (1 - poi) / G * p0 * a * 2
    return cons * np.sqrt(1 - np.power(eta / a, 2))


def transform(xc: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Translation and rotation coordinate transformation of boundary face coordinates for the BEM method

    Parameter
        xc: Coordinates of BEM segment centre
        x: Coordinates of boundary faces
        alpha: Fracture orientation
    Return:
        Transformed coordinates
    """
    x_bar = np.zeros_like(x)
    # Terms in (7.4.6) in Crouch Starfield 1983 Boundary Element Methods in Solid Mechanics page 168
    x_bar[0, :] = (x[0, :] - xc[0]) * np.cos(alpha) + (x[1, :] - xc[1]) * np.sin(alpha)
    x_bar[1, :] = -(x[0, :] - xc[0]) * np.sin(alpha) + (x[1, :] - xc[1]) * np.cos(alpha)
    return x_bar


def get_bc_val(
    g: pp.Grid,
    bound_faces: np.ndarray,
    xf: np.ndarray,
    h: float,
    poi: float,
    alpha: float,
    du: float,
) -> np.ndarray:
    """
    Compute semi-analytical displacement on the boundary using the BEM method for the Sneddon problem.

    Parameter
        g: Grid object
        bound_faces: Index lists for boundary faces
        xf: Coordinates of boundary faces
        h: BEM segment length
        poi: Poisson ratio
        alpha: Fracture orientation
        du: Sneddon's analytical relative normal displacement
    Return:
        Boundary values for the displacement
    """
    
    # Equations for f2,f3,f4.f5 can be found in book Crouch Starfield 1983 Boundary Element Methods in Solid Mechanics pages 57, 84-92, 168
    f2 = np.zeros(bound_faces.size)
    f3 = np.zeros(bound_faces.size)
    f4 = np.zeros(bound_faces.size)
    f5 = np.zeros(bound_faces.size)

    u = np.zeros((g.dim, g.num_faces))

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
    g: pp.Grid,
    h: float,
    bound_faces: np.ndarray,
    theta: float,
    bem_centers: np.ndarray,
    u_a: np.ndarray,
    poi: float,
) -> np.ndarray:
    """
    Compute analytical displacement using the BEM method for the Sneddon problem.

    Parameter
        g: Subdomain grid
        h: bem segment length
        bound_faces: boundary faces
        theta: fracture orientation
        bem_centers: bem segments centers
        u_a: Sneddon's analytical relative normal displacement
        poi: Poisson ratio

    Return:
        Semi-analytical boundary displacement values
    """

    bc_val = np.zeros((g.dim, g.num_faces))

    alpha = np.pi / 2 - theta

    bound_face_centers = g.face_centers[:, bound_faces]

    for i in range(0, u_a.size):

        new_bound_face_centers = transform(bem_centers[:, i], bound_face_centers, alpha)

        u_bound = get_bc_val(
            g, bound_faces, new_bound_face_centers, h, poi, alpha, u_a[i]
        )

        bc_val += u_bound

    return bc_val




class ManuExactSneddon2dSetup:
    """
    Class for setting up the analytical solution for the pressurized fracture problem.
    
    
    """
  
    def __init__(self, setup):
        # Initialize private variables from the setup dictionary
        self.p0 = setup.get("p0")
        self.theta = setup.get("theta")

        self.a = setup.get("a")
        self.shear_modulus  = setup.get("material_constants").get("solid").shear_modulus
        self.poi = setup.get("poi")
        self.length = setup.get("length")
        self.height = setup.get("height")
         
  
    def exact_sol_global(self, sd):
        """
        Compute the analytical solution for the pressurized fracture problem in question.
        
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
        
        
        u_a = -analytical_displacements(self.a, eta, self.p0, self.shear_modulus, self.poi)
        u_bc = assign_bem(sd, h / 2, box_faces, self.theta, bem_centers, u_a, self.poi)
        return u_bc
    

    def exact_sol_fracture( self,
    gb: pp.MixedDimensionalGrid) -> tuple:
        """
        Compute Sneddon's analytical solution for the pressurized crack
        problem in question.

        Source: Sneddon Fourier transforms 1951 page 425 eq 92

        Parameter
            gb: Grid object
            a: Half fracture length of fracture
            eta: Distance from fracture centre
            p0: Internal constant pressure inside the fracture
            G: Shear modulus
            poi: Poisson ratio
            height,length: Height and length of domain

        Return:
            A tuple containing two vectors: a list of distances from the fracture center to each fracture coordinate, and the corresponding analytical apertures.
        """
        ambient_dim = gb.dim_max()
        g_1 = gb.subdomains(dim=ambient_dim - 1)[0]
        fracture_center = np.array([self.length / 2, self.height / 2, 0])

        fracture_faces = g_1.cell_centers

        # compute distances from fracture centre with its corresponding apertures
        eta = compute_eta(fracture_faces, fracture_center)
        apertures = analytical_displacements(self.a, eta, self.p0, self.shear_modulus, self.poi)

        return eta, apertures





class ModifiedGeometry(PorePyModel):
    def set_domain(self) -> None:
        """Defining a two-dimensional square domain with sidelengths length and height."""

        self._domain = pp.Domain(
            bounding_box={
                "xmin": 0,
                "ymin": 0,
                "xmax": self.params["length"],
                "ymax": self.params["height"],
            }
        )

    def grid_type(self) -> str:
        """Choosing the grid type for our domain."""
        return self.params.get("grid_type", "simplex")

    def set_fractures(self):
        """Setting the fractures in the domain from params."""

        points = self.params["frac_pts"]
        self._fractures = [pp.LineFracture(points)]


class ModifiedBoundaryConditions(PorePyModel):
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
        """
        Setting displacement boundary condition values.
        This method sets the displacement boundary condition values for a given 
        boundary grid using the Sneddon analytical solution through the Boundary 
        Element Method (BEM).
        Parameters:
            The boundary grid for which the displacement boundary condition values 
            are to be set.
        Returns:
            An array of displacement boundary condition values.
        """

        sd = bg.parent
        if sd.dim < 2:
            return np.zeros(self.nd * sd.num_faces)

  
        # Get the analytical solution for the displacement 
        exact_sol = ManuExactSneddon2dSetup(self.params)
        u_exact = exact_sol.exact_sol_global(sd) 
        
        # Project the values to the grid
        return bg.projection(2) @ u_exact.ravel("F")


class PressureStressMixin(PorePyModel):
    def pressure(self, subdomains: pp.Grid):
        """Discretization of the stress tensor.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Discretization operator for the stress tensor.

        """
        # Return pressure in the fracture domain
        mat = self.params["p0"] * np.ones(
            sum(subdomain.num_cells for subdomain in subdomains)
        )
        return pp.ad.DenseArray(mat)

    def stress_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.BiotAd | pp.ad.MpsaAd:
        """Discretization of the stress tensor.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Discretization operator for the stress tensor.
        """
        return pp.ad.MpsaAd(self.stress_keyword, subdomains)


@dataclass
class SneddonData:
    """Data class for storing the error in the displacement field."""
    error_displacement: pp.number

class SneddonDataSaving(VerificationDataSaving):
    """Class for saving the error in the displacement field."""
    def collect_data(self):
        """Collecting the error in the displacement field.
        
        Returns: collected data dictionary
        """
        frac_sd = self.mdg.subdomains(dim=self.nd - 1)
        nd_vec_to_normal = self.normal_component(frac_sd)
    
        # Comuting the numerical displacement jump along the fracture on the fracture cell centers.
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(frac_sd)
        u_n = u_n.value(self.equation_system)
        
        exact_setup = ManuExactSneddon2dSetup(self.params)
        # Checking convergence specifically on the fracture
        u_a = exact_setup.exact_sol_fracture(self.mdg)[1]
    
        e = ConvergenceAnalysis.l2_error(
            frac_sd[0], u_a, u_n, is_scalar=False, is_cc=True, relative=True
        )
        
        collect_data = SneddonData(error_displacement=e)
        return collect_data


class MomentumBalanceGeometryBC(
    PressureStressMixin,
    ModifiedGeometry,
    SneddonDataSaving,
    ModifiedBoundaryConditions,
    pp.constitutive_laws.PressureStress,
    pp.MomentumBalance
):
    pass
    

