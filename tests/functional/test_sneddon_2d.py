
"""
This is a setup class for solving linear elasticity with contact between the fractures.
We do not consider any fluid, and solve only for the linear elasticity coupled to the contact.
"""

from matplotlib import pyplot as plt
import numpy as np
import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis

import math
from numba import config
import pytest


config.DISABLE_JIT = True


def compute_eta(pointset_centers: np.ndarray, point: np.ndarray):
    """
    Compute the distance of bem segments centers to the
    fracture centre.

    Parameter
    ---------
    pointset_centers: array containing centers of bem segments
    point: fracture centre, middle point of the square domain

    """
    return pp.geometry.distances.point_pointset(pointset_centers, point)


def get_bem_centers(a: float, h: float, n: int, theta: float, center: np.ndarray) -> np.ndarray:
    """
    Compute coordinates of the centers of the bem segments

    Parameter
    ---------
    a: half fracture length
    h: bem segment length
    n: number of bem segments
    theta: orientation of the fracture
    center: center of the fracture
    """
    
    # Coordinate system (4.5.1) in page 57 in in book Crouch Starfield 1983 Boundary Element Methods in Solid Mechanics

    bem_centers = np.zeros((3, n))
    x_0 = center[0] - (a - 0.5 * h) * np.sin(theta)
    y_0 = center[1] - (a - 0.5 * h) * np.cos(theta)
    for i in range(0, n):
        bem_centers[0, i] = x_0 + i * h * np.sin(theta)
        bem_centers[1, i] = y_0 + i * h * np.cos(theta)

    return bem_centers


def analytical_displacements(a: float, eta: np.ndarray, p0: float, G: float, poi: float ) -> np.ndarray:
    """
    Compute Sneddon's analytical solution for the pressurized crack
    problem in question.
    
    Source: Sneddon Fourier transforms 1951 page 425 eq 92 

    Parameter
    ---------
    a: half fracture length
    eta: distance from fracture centre
    p0: pressure
    G: shear modulus
    poi: poisson ratio
    """
    cons = (1 - poi) / G * p0 * a * 2
    return cons * np.sqrt(1 - np.power(eta / a, 2))


def transform(xc: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Coordinate transofrmation for the BEM method

    Parameter
    ---------
    xc: coordinates of BEM segment centre
    x: coordinates of boundary faces
    alpha: fracture orientation
    """
    x_bar = np.zeros_like(x)
    x_bar[0, :] = (x[0, :] - xc[0]) * np.cos(alpha) + (x[1, :] - xc[1]) * np.sin(alpha)
    x_bar[1, :] = -(x[0, :] - xc[0]) * np.sin(alpha) + (x[1, :] - xc[1]) * np.cos(alpha)
    return x_bar


def get_bc_val(g: pp.Grid, bound_faces, xf, h, poi, alpha, du):
    """
    Compute analytical displacement using the BEM method for the pressurized crack
    problem in question.
    


    Parameter
    ---------
    g: grid object
    bound_faces: boundary faces
    xf: coordinates of boundary faces
    h: bem segment length
    poi: Poisson ratio
    alpha: fracture orientation
    du: Sneddon's analytical relative normal displacement
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


def assign_bem(g: pp.Grid, h:float, bound_faces: np.ndarray, theta: float, bem_centers: np.ndarray, u_a: np.ndarray, poi: float) -> np.ndarray:
    """
    Compute analytical displacement using the BEM method for the pressurized crack
    problem in question.

    Parameter
    ---------
    g: Subdomain grid
    h: bem segment length
    bound_faces: boundary faces
    theta: fracture orientation
    bem_centers: bem segments centers
    u_a: Sneddon's analytical relative normal displacement
    poi: Poisson ratio
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



def run_analytical_displacements(gb: pp.Grid, a: float, p0: float, G: float, poi: float, height: float, length: float) -> tuple:
    """
    Compute Sneddon's analytical solution for the pressurized crack
    problem in question.
    
    Source: Sneddon Fourier transforms 1951 page 425 eq 92 

    Parameter
    ---------
    gb: Grid object
    a: half fracture length
    eta: distance from fracture centre
    p0: pressure
    G: shear modulus
    poi: poisson ratio
    height,length: Height and length of domain
    """
    ambient_dim = gb.dim_max()
    g_1 = gb.subdomains(dim=ambient_dim - 1)[0]
    fracture_center = np.array([length / 2, height / 2, 0])

    fracture_faces = g_1.cell_centers
    
    # compute distances from fracture centre with its corresponding apertures 
    eta = compute_eta(fracture_faces, fracture_center)   
    apertures = analytical_displacements(a, eta, p0,  G, poi)
    
    return eta, apertures


class ModifiedGeometry:
    def set_domain(self) -> None:
        """Defining a two-dimensional square domain with sidelength 2."""
        
        self._domain = pp.Domain(bounding_box={"xmin": 0, "ymin": 0, "xmax": self.params["length"], "ymax": self.params["height"]})

    def grid_type(self) -> str:
        """Choosing the grid type for our domain."""
        return self.params.get("grid_type", "simplex")

    def set_fractures(self):

        points = self.params["frac_pts"]
        self._fractures = [pp.LineFracture(points)]


class ModifiedBoundaryConditions:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Set boundary condition type for the problem."""
        bounds = self.domain_boundary_sides(sd)

        # Set the type of west and east boundaries to Dirichlet. North and south are
        # Neumann by default.
        bc = pp.BoundaryConditionVectorial(sd, bounds.all_bf, "dir")

        frac_face = sd.tags["fracture_faces"]

        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True

        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Setting displacement boundary condition values.

        This method returns an array of boundary condition values with the value 5t for
        western boundaries and ones for the eastern boundary.

        """
       
        a, p0, G, poi = self.params["a"], self.params["p0"], self.params["material_constants"]["solid"].shear_modulus, self.params["poi"]
        theta, length, height = self.params["theta"], self.params["length"], self.params["height"]
 
        sd = bg.parent
        # bg = self.mdg.subdomain_to_boundary_grid(sd) # technique to extract bg via sd

        if sd.dim < 2:
            return np.zeros(self.nd * sd.num_faces)
        box_faces = sd.get_boundary_faces()

        # Set the boundary values
        u_bc = np.zeros((sd.dim, sd.num_faces))

        # apply sneddon analytical solution through BEM method
        n = 1000
        h = 2 * a / n
        center = np.array([length / 2, height / 2, 0])
        bem_centers = get_bem_centers(a, h, n, theta, center)
        eta = compute_eta(bem_centers, center)
        
        u_a = -analytical_displacements(a, eta, p0, G, poi)

        u_bc = assign_bem(sd, h / 2, box_faces, theta, bem_centers, u_a, poi)

        return bg.projection(2) @ u_bc.ravel("F")


class PressureStressMixin:
    def pressure(self, subdomains: pp.Grid):
        # Return pressure in the fracture domain
        mat = self.params["p0"] * np.ones(sum(subdomain.num_cells for subdomain in subdomains))
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


class MomentumBalanceGeometryBC(
    PressureStressMixin,
    ModifiedGeometry,
    ModifiedBoundaryConditions,
    pp.constitutive_laws.PressureStress,
    pp.MomentumBalance,
):
    ...
    """Adding geometry and modified boundary conditions to the default model."""
    




def simulation(frac_pts, theta_rad, h, a=1.5, height=1, length=1, p0=1e-4, G=1, poi=0.25):
     
    lam = 2 * G * poi / (1 - 2 * poi)  # Lame parameter
    solid = pp.SolidConstants(shear_modulus = G, lame_lambda = lam)

    
    # Clean this up!! This is made so I can easily access the params in arbitary functions
    params = {
                   "meshing_arguments": {"cell_size": h},
                   "prepare_simulation": True,
                    "material_constants": {"solid": solid},
                    "frac_pts": frac_pts,
                    "theta": theta_rad,
                    "p0": p0, 
                    "a": a,
                    "poi": poi,
                    "length": length, #this info can be accessed via box, so remove these as well.
                    "height": height
                }

    model = MomentumBalanceGeometryBC(params)
    pp.run_time_dependent_model(model, params)
    
    frac_sd = model.mdg.subdomains(dim=model.nd - 1)
    nd_vec_to_normal = model.normal_component(frac_sd)

    u_n: pp.ad.Operator = nd_vec_to_normal @ model.displacement_jump(frac_sd)
    u_n = u_n.value(model.equation_system)
    
    # Checking convergence specifically on the fracture
    u_a = run_analytical_displacements(model.mdg, a, p0, G, poi, height, length)[1]
    e = ConvergenceAnalysis.l2_error(
                    frac_sd[0], u_a, u_n, is_scalar=False, is_cc=True, relative=True
                    )
    
    
    # Transform cell center points to distance from fracture center eta 
    fracture_center = np.array([length / 2, height / 2, 0]) 
    eta = compute_eta( frac_sd[0].cell_centers, fracture_center)   
    
    return e, eta, u_a, u_n


  
def compute_frac_pts(theta_rad,a, height, length):
    # Rotate the fracture with an angle theta_rad
    y_0 = height / 2 - a * np.cos(theta_rad)
    x_0 = length / 2 - a * np.sin(theta_rad)
    y_1 = height / 2 + a * np.cos(theta_rad)
    x_1 = length / 2 + a * np.sin(theta_rad)
    
    def vector_length(x0, y0, x1, y1):
        point1 = np.array([x0, y0])
        point2 = np.array([x1, y1])
        return np.linalg.norm(point2 - point1)
    
    print(f"Frac points, ({x_0, y_0} ), ({x_1, y_1} ) with half length {vector_length(x_0,y_0, x_1, y_1)/2}")
    frac_pts = np.array([[x_0, y_0], [x_1, y_1]]).T
    return frac_pts

def compute_eoc(h, e):
    # Compute the Error Order of Convergence (EOC)
    eoc = np.zeros(len(e) - 1)
    for i in range(len(e) - 1):
        eoc[i] = np.log(e[i+1] / e[i]) / np.log(h[i+1] / h[i])
    return eoc

    

def compute_convergence(h_list, theta_list, a=1, height=1, length=1):
    # Compute Error for big fracture length
    err = np.zeros(( len(theta_list), len(h_list)))  
    for k in range(0, len(theta_list)):
        theta_rad = math.radians(90 - theta_list[k])
        frac_pts = compute_frac_pts(theta_rad,a, height=height, length=length)   
        for i in range(0, len(h_list)):
            e, _, _, _ = simulation(frac_pts, theta_rad, h_list[i], a=a, height=height, length=length)
            err[k, i] = e
        print(f"a: {a}, (height,length) = ({height}, {length}) theta: {theta_list[k]}, EOC: {compute_eoc(h_list, err[k, :])}")
    
    return err


def test_sneddon_2d():
    
    # Simulation settings    
    num_refs = 4 # Number of refinements
    num_theta = 6 # Number of iteration of theta
    
    theta0 = 0 # Inital orientation of fracture
    delta_theta = 10 # The difference in theta between each refinement
    theta_list = theta0 + np.arange(0,num_theta)*delta_theta # List of all the fracture orientations we run simulation on
  
    # Mesh refinement
    nf0 = 6 
    nf = nf0 * 2**np.arange(0, num_refs)
    h_list = 2 * 1.5 / nf  # Mesh size for simulations


    # In this test we define the 
    height, length = 1,1
    
    # Here we define the half length of the fracture to be sufficientl big to avoid any fracture tip effects
    a = 1
   
    # Run convergence experiment
    err = compute_convergence(h_list, theta_list, a=a, height=height, length=length)
    
    # Computing average error of each theta
    avg_err = np.mean(err, axis=0) 

    # Perform linear regression on the log-log scale and compute EOC
    log_h = np.log(h_list)
    log_avg_err = np.log(avg_err)
    slope, _ = np.polyfit(log_h, log_avg_err, 1)
    assert slope > 1.0, f"Estimated EOC is {slope}, which is not greater than 1"

   
    