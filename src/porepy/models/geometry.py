"""Geometry definition for simulation setup.

"""
from __future__ import annotations

import logging
from typing import Tuple, Union

import numpy as np
import scipy.sparse as sps

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

    def subdomains_to_interfaces(
        self, subdomains: list[pp.Grid], codims=[1]
    ) -> list[pp.MortarGrid]:
        """Interfaces neighbouring any of the subdomains.

        Args:
            subdomains (list[pp.Grid]): Subdomains for which to find interfaces.
            codims (list, optional): Codimension of interfaces to return. Defaults to [1],
            i.e. only interfaces between one dimension apart.

        Returns:
            list[pp.MortarGrid]: Unique, sorted list of interfaces.
        """
        interfaces = list()
        for sd in subdomains:
            for intf in self.mdg.subdomain_to_interfaces(sd):
                if intf not in interfaces and intf.codim in codims:
                    interfaces.append(intf)
        return self.mdg.sort_interfaces(interfaces)

    def interfaces_to_subdomains(
        self, interfaces: list[pp.MortarGrid]
    ) -> list[pp.Grid]:
        """Unique sorted list of all subdomains neighbouring any of the interfaces."""
        subdomains = list()
        for interface in interfaces:
            for sd in self.mdg.interface_to_subdomain_pair(interface):
                if sd not in subdomains:
                    subdomains.append(sd)
        return self.mdg.sort_subdomains(subdomains)

    def local_coordinates(self, subdomains: list[pp.Grid]) -> pp.ad.Matrix:
        """Ad wrapper around tangential_normal_projections for fractures.

        TODO: Extend to all subdomains.

        Parameters:
            subdomains: List of subdomains for which to compute the local coordinates.

        Returns:
            Local coordinates as a pp.ad.Matrix.

        """
        # For now, assert all subdomains are fractures, i.e. dim == nd - 1
        assert all([sd.dim == self.nd - 1 for sd in subdomains])
        local_coord_proj_list = [
            self.mdg.subdomain_data(sd)[
                "tangential_normal_projection"
            ].project_tangential_normal(sd.num_cells)
            for sd in subdomains
        ]
        return pp.ad.Matrix(sps.block_diag(local_coord_proj_list))

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

    # Local basis related methods
    def tangential_component(self, grids: list[pp.Grid]) -> pp.ad.Operator:
        """Compute the tangential component of a vector field.

        Parameters:
            grids: List of grids on which the vector field is defined.

        Returns:
            tangential: Operator extracting tangential component of the vector field and
            expressing it in tangential basis.
        """
        geom = pp.ad.Geometry(grids, self.nd)

        # We first need an inner product (or dot product), i.e. extract the tangential
        # component of the cell-wise vector v to be transformed. Then we want to express it in
        # the tangential basis. The two operations are combined in a single operator composed
        # right to left:
        # v will be hit by first e_i.T (row vector) and secondly t_i (column vector).
        op = sum(
            [geom.e_i(i, self.nd - 1) * geom.e_i(i, self.nd).T for i in range(self.nd)]
        )
        return op

    def normal_component(self, grids: list[pp.Grid]) -> pp.ad.Operator:
        """Compute the normal component of a vector field.

        Parameters:
            grids: List of grids on which the vector field is defined.

        Returns:
            normal: Operator extracting normal component of the vector field and
            expressing it in normal basis.
        """
        geometry = pp.ad.Geometry(grids, self.nd)
        e_n = geometry.e_i(self.nd - 1, self.nd)
        t_n = geometry.e_i(self.nd - 1, self.nd - 1)
        op = t_n * e_n.T
        return op
