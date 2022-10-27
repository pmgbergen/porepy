"""Geometry definition for simulation setup.

"""
from __future__ import annotations

import logging
from typing import Tuple, Union

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)


class ModelGeometry:
    """This class provides geometry related methods and information for a simulation model."""

    # Define attributes to be assigned later
    fracture_network: Union[pp.FractureNetwork2d, pp.FractureNetwork3d]
    """Representation of fracture network including intersections."""
    well_network: pp.WellNetwork3d
    """Well network."""
    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""
    box: dict
    """Box-shaped domain. FIXME: change to "domain"? """
    nd: int
    """Ambient dimension."""

    def set_geometry(self) -> None:
        """Define geometry and create a mixed-dimensional grid."""
        # Create fracture network and mixed-dimensional grid
        self.set_fracture_network()
        self.set_md_grid()
        self.nd: int = self.mdg.dim_max()
        # If fractures are present, it is advised to call
        pp.contact_conditions.set_projections(self.mdg)

    def set_fracture_network(self) -> None:
        """Assign fracture network class."""
        self.fracture_network = pp.FractureNetwork2d()

    def mesh_arguments(self) -> dict:
        """Mesh arguments for md-grid creation.

        Returns:
            mesh_args: Dictionary of meshing arguments compatible with FractureNetwork.mesh()
                method.

        """
        mesh_args = dict()
        return mesh_args

    def set_md_grid(self) -> None:
        """Create the mixed-dimensional grid.

        A unit square grid with no fractures is assigned by default if self.fracture_network
        contains no fractures. Otherwise, the network's mesh method is used.

        The method assigns the following attributes to self:
            mdg (pp.MixedDimensionalGrid): The produced grid bucket.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """

        if self.fracture_network.num_frac() == 0:
            # Mono-dimensional grid by default
            phys_dims = np.array([1, 1])
            n_cells = np.array([1, 1])
            self.box = pp.geometry.bounding_box.from_points(
                np.array([[0, 0], phys_dims]).T
            )
            g: pp.Grid = pp.CartGrid(n_cells, phys_dims)
            g.compute_geometry()
            self.mdg = pp.meshing.subdomains_to_mdg([[g]])
        else:
            self.mdg = self.fracture_network.mesh(self.mesh_arguments())
            self.box = self.fracture_network.domain

    ## Utility methods
    def subdomains_to_interfaces(
        self, subdomains: list[pp.Grid], codims=[1]
    ) -> list[pp.MortarGrid]:
        """Unique list of all interfaces neighbouring any of the subdomains.
        FIXME: Sort
        """

        interfaces = list()
        for sd in subdomains:
            for intf in self.mdg.subdomain_to_interfaces(sd):
                if (
                    intf not in interfaces and intf.codim in codims
                ):  # could filter on codimension.
                    interfaces.append(intf)
        return interfaces

    def interfaces_to_subdomains(
        self, interfaces: list[pp.MortarGrid]
    ) -> list[pp.Grid]:
        """Unique list of all subdomains neighbouring any of the interfaces.
        FIXME: Sort
        """
        subdomains = list()
        for interface in interfaces:
            for sd in self.mdg.interface_to_subdomain_pair(interface):
                if sd not in subdomains:
                    subdomains.append(sd)
        return subdomains

    def subdomain_projections(self, dim: int):
        """Return the projection operators for all subdomains in md-grid.

        The projection operators restrict or prolong a dim-dimensional quantity from the full
        set of subdomains to any subset.
        Projection operators are constructed once and then stored. If you need to use
        projection operators based on a different set of subdomains, please construct
        them yourself. Alternatively, compose a projection from subset A to subset B as
            P_A_to_B = P_full_to_B * P_A_to_full.

        Parameters:
            dim: Dimension of the quantities to be projected.

        Returns:
            proj: Projection operator.
        """
        name = f"_subdomain_proj_{dim}"
        if hasattr(self, name):
            proj = getattr(self, name)
        else:
            proj = pp.ad.SubdomainProjections(self.mdg.subdomains(), dim)
            setattr(self, name, proj)
        return proj

    def _nd_subdomain(self) -> pp.Grid:
        """Get the grid of the highest dimension. Assumes self.mdg is set.

        FIXME: Purge?
        """
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
        if self.nd == 1:
            north = np.zeros(g.num_faces, dtype=bool)
            south = north.copy()
        else:
            north = g.face_centers[1] > box["ymax"] - tol
            south = g.face_centers[1] < box["ymin"] + tol
        if self.nd < 3:
            top = np.zeros(g.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = g.face_centers[2] > box["zmax"] - tol
            bottom = g.face_centers[2] < box["zmin"] + tol
        all_bf = g.get_boundary_faces()
        return all_bf, east, west, north, south, top, bottom
