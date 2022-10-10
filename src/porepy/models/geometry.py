"""Geometry definition for simulation setup.

"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)


class Geometry:
    """This class provides geometry related methods and information for a simulation model.

    """

    def __init__(self, params: Optional[Dict] = None):
        if params is None:
            self.params = {}
        else:
            self.params = params
        if params is None:
            params = {}
        default_params = {
        }

        default_params.update(params)
        self.params = default_params
        """Geometry parameter dictionary passed on init."""

        # Define attributes to be assigned later
        self.fracture_network: Union[pp.FractureNetwork2d, pp.FractureNetwork3d]
        """Representation of fracture network including intersections."""
        self.well_network: pp.WellNetwork3d
        """Well network."""
        self.mdg: pp.MixedDimensionalGrid
        """Mixed-dimensional grid."""
        self.box: dict
        """Box-shaped domain. FIXME: change to "domain"? """

        # Create fracture network and mixed-dimensional grid
        self.create_fracture_network()
        self.create_grid()
        self.nd: int = self.mdg.dim_max()
        # If fractures are present, it is advised to call
        pp.contact_conditions.set_projections(self.mdg)

    def create_fracture_network(self):
        """Assign fracture network class."""
        self.fracture_network = pp.FractureNetwork2d()


    def mesh_arguments(self):
        """Mesh arguments for md-grid creation.

        Returns:
            mesh_args: Dictionary of meshing arguments compatible with FractureNetwork.mesh()
                method.

        """
        mesh_args = dict()
        return mesh_args

    def create_md_grid(self) -> None:
        """Create the mixed-dimensional grid.

        A unit square grid with no fractures is assigned by default.

        The method assigns the following attributes to self:
            mdg (pp.MixedDimensionalGrid): The produced grid bucket.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """
        phys_dims = np.array([1, 1])
        n_cells = np.array([1, 1])
        self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
        g: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        g.compute_geometry()
        self.mdg = pp.meshing.subdomains_to_mdg([[g]])

    def create_geometry_operators(self):
        """Set geometry operators.

        The three operators set here are common to most standard problems. Extensions s.a.
        a vector version of the mortar projection and subdomain restriction/prolongation
        operators may be needed for specific problems.
        """
        subdomains = self.mdg.subdomains()
        interfaces = self.mdg.interfaces()
        self.subdomain_geometry = pp.ad.Geometry(subdomains, nd=self.nd, name="all subdomains")
        self.mortar_projection_scalar = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)

    ## Utility methods
    def _l2_norm_cell(self, g: pp.Grid, val: np.ndarray) -> float:
        """
        Compute the cell volume weighted norm of a vector-valued cell-wise quantity for
        a given grid.

        Parameters:
            g (pp.Grid): Grid
            val (np.array): Vector-valued function.

        Returns:
            double: The computed L2-norm.

        """
        nc = g.num_cells
        sz = val.size
        if nc == sz:
            nd = 1
        elif nc * g.dim == sz:
            nd = g.dim
        else:
            raise ValueError("Have not considered this type of unknown vector")

        norm = np.sqrt(np.reshape(val**2, (nd, nc), order="F") * g.cell_volumes)

        return np.sum(norm)

    def _nd_subdomain(self) -> pp.Grid:
        """Get the grid of the highest dimension. Assumes self.mdg is set.

        FIXME: Purge?"""
        return self.mdg.subdomains(dim=self.nd)[0]

    def domain_boundary_sides(
        self, g: pp.Grid
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Obtain indices of the faces of a grid that lie on each side of the domain
        boundaries. It is assumed the domain is box shaped.
        """
        tol = 1e-10
        box = self.box
        east = g.face_centers[0] > box["xmax"] - tol
        west = g.face_centers[0] < box["xmin"] + tol
        if self.mdg.dim_max() == 1:
            north = np.zeros(g.num_faces, dtype=bool)
            south = north.copy()
        else:
            north = g.face_centers[1] > box["ymax"] - tol
            south = g.face_centers[1] < box["ymin"] + tol
        if self.mdg.dim_max() < 3:
            top = np.zeros(g.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = g.face_centers[2] > box["zmax"] - tol
            bottom = g.face_centers[2] < box["zmin"] + tol
        all_bf = g.get_boundary_faces()
        return all_bf, east, west, north, south, top, bottom
