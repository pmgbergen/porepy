"""
Module containing a worker function to generate mixed-dimensional grids in an unified
way.
"""

from typing import Dict, Literal, Optional, Union
import numpy as np
import porepy as pp

import porepy.grids.standard_grids.utils as utils

def simplex_2d_grid():

    # Connected zig-zag fractures
    points = np.array([
        [0.1, 0.1],
        [0.5, 0.4],
        [0.3, 0.7],
        [0.6, 0.8],
        [0.65, 0.75],
        [0.9, 0.9]
    ]).T

    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]).T

    # Set mesh size close to the fracture
    mesh_args = {'mesh_size_frac': 0.2}

    # Define the domain
    domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}

    mdg = utils.make_mdg_2d_simplex(mesh_args, points, edges, domain)
    return mdg



def simplex_3d_grid():
    # Connected zig-zag fractures
    points = np.array([
        [0.1, 0.1],
        [0.5, 0.4],
        [0.3, 0.7],
        [0.6, 0.8],
        [0.65, 0.75],
        [0.9, 0.9]
    ]).T

    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]).T

    # Set mesh size close to the fracture
    mesh_args = {'mesh_size_frac': 0.2}

    # Define the domain
    domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}

    mdg = utils.make_mdg_2d_simplex(mesh_args, points, edges, domain)
    return mdg

def simplex_3d_grid():

    # define points
    x0 = 0.1
    x1 = 0.4
    x2 = 0.35
    x3 = 0.2

    y0 = 0.2
    y1 = 0.2
    y2 = 0.4
    y3 = 0.4

    z0 = 0.3
    z1 = 0.3
    z2 = 0.1
    z3 = 0.1

    x0_ = 0.2
    x1_ = 0.3
    x2_ = 0.25
    x3_ = 0.45

    y0_ = 0.3
    y1_ = 0.3
    y2_ = 0.4
    y3_ = 0.4

    z0_ = 0.2
    z1_ = 0.2
    z2_ = 0.5
    z3_ = 0.5

    # The fractures are specified by their vertices, stored in a numpy array
    f_1 = pp.PlaneFracture(np.array([[x0, x1, x2, x3], [y0, y1, y2, y3], [z0, z1, z2, z3]]))
    f_2 = pp.PlaneFracture(
        np.array([[x0_, x1_, x2_, x3_], [y0_, y1_, y2_, y3_], [z0_, z1_, z2_, z3_]]))

    # Define the domain
    domain = {'xmin': 0, 'xmax': 0.6, 'ymin': 0, 'ymax': 0.6, 'zmin': 0, 'zmax': 0.6}

    # Define a 3d FractureNetwork
    network = pp.FractureNetwork3d([f_1, f_2], domain=domain)

    # Defining mesh_args
    mesh_args = {'mesh_size_frac': 0.03, 'mesh_size_min': 0.02}

    # Generate mixed-dimensional mesh
    mdg = network.mesh(mesh_args)

def coord_cart_2d(self, phys_dims, dev, pos):
    xmax = phys_dims[0]
    ymax = phys_dims[1]

    x = np.array(pos)
    y = np.array([dev, ymax - dev])
    return np.array([x, y])


def cartersian_2d():

    # Two fractures at 1/3 and 2/3 of domain
    phys_dims = [20, 10]
    n_cells = [80,40]
    bounding_box_points = np.array([[0, phys_dims[0]],[0, phys_dims[1]]]);
    box = pp.geometry.bounding_box.from_points(bounding_box_points)
    f1 = coord_cart_2d(phys_dims = phys_dims, dev = 2,pos = [5,5])
    f2 = coord_cart_2d(phys_dims=phys_dims, dev=2, pos=[15, 15])
    mdg = pp.meshing.cart_grid(fracs = [f1,f2], physdims = phys_dims, nx = np.array(n_cells))
    return mdg

def coord_cart_3d(self, phys_dims, dev, pos):
    xmax = phys_dims[0]
    ymax = phys_dims[1]
    zmax = phys_dims[2]

    z = np.array([dev, dev, zmax - dev, zmax - dev])

    if pos[1] == 'x':
        x = np.ones(4) * pos[0]
        y = np.array([dev, ymax - dev, ymax - dev, dev])
    elif pos[1] == 'y':
        x = np.array([dev, xmax - dev, xmax - dev, dev])
        y = np.ones(4) * pos[0]
    return np.array([x, y, z])

def cartersian_3d():

    # Generate mixed-dimensional mesh
    phys_dims = [50,50,10]
    n_cells = [20,20,10]
    bounding_box_points = np.array([[0, phys_dims[0]],[0, phys_dims[1]],[0, phys_dims[2]]]);
    box = pp.geometry.bounding_box.from_points(bounding_box_points)

    frac1 = coord_cart_3d(phys_dims, 2,(25,'x'))
    frac2 = coord_cart_3d(phys_dims, 2,(25,'y'))
    frac3 = coord_cart_3d(phys_dims, 2,(23,'y'))
    frac4 = coord_cart_3d(phys_dims, 2,(27,'y'))
    mdg = pp.meshing.cart_grid(fracs = [frac1,frac2,frac3,frac4], physdims = phys_dims, nx = np.array(n_cells))
    return mdg


def create_mixed_dimensional_grid(fracture_network: Union[pp.FractureNetwork2d, pp.FractureNetwork3d],
grid_type: Literal["simplex", "cartesian", "tensor_grid"],
mesh_arguments: dict[str],
**kwargs) -> pp.MixedDimensionalGrid:

    # TODO: give a try on unify signatures

    # Assertion for FN type
    assert isinstance(fracture_network, pp.FractureNetwork2d) or isinstance(fracture_network, pp.FractureNetwork3d)

    # TODO: Collect examples for Tensor grids
    # 2d cases
    if isinstance(fracture_network, pp.FractureNetwork2d):
        if grid_type == "simplex":
            mdg = simplex_2d_grid()
        elif grid_type == "cartesian":
            mdg = cartersian_2d()

    # 3d cases
    if isinstance(fracture_network, pp.FractureNetwork2d):
        if grid_type == "simplex":
            mdg = simplex_3d_grid()
        elif grid_type == "cartesian":
            mdg = cartersian_3d()

    # TODO: write conditions for this case
    save = pp.Exporter(mdg, "mdg")
    save.write_vtu(mdg)
    return mdg
