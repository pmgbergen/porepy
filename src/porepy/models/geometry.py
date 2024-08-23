"""Geometry definition for simulation setup.

"""

from __future__ import annotations

import copy
from typing import Literal, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.models.protocol import PorePyModel


class ModelGeometry(PorePyModel):
    """This class provides geometry related methods and information for a simulation
    model."""

    _domain: pp.Domain

    fracture_network: pp.fracture_network
    """Representation of fracture network including intersections."""

    well_network: pp.WellNetwork3d
    """Well network."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid. Set by the method :meth:`set_md_grid`."""

    nd: int
    """Ambient dimension of the problem. Set by the method :meth:`set_geometry`"""

    def set_geometry(self) -> None:
        # Create the geometry through domain amd fracture set.
        self.set_domain()
        self.set_fractures()
        # Create a fracture network.
        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)
        # Create a mixed-dimensional grid.
        self.mdg = pp.create_mdg(
            self.grid_type(),
            self.meshing_arguments(),
            self.fracture_network,
            **self.meshing_kwargs(),
        )
        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fracture + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures.
            self.well_network.mesh(self.mdg)

    @property
    def domain(self) -> pp.Domain:
        return self._domain

    def set_domain(self) -> None:
        """Set domain of the problem.

        Defaults to a 2d unit square domain.
        Override this method to define a geometry with a different domain.

        """
        self._domain = nd_cube_domain(2, self.solid.convert_units(1.0, "m"))

    @property
    def fractures(self) -> Union[list[pp.LineFracture], list[pp.PlaneFracture]]:
        return self._fractures

    def set_fractures(self) -> None:
        """Set fractures in the fracture network.

        Override this method to define a geometry with fractures.

        """
        self._fractures: list = []

    def set_well_network(self) -> None:
        """Assign well network class."""
        self.well_network = pp.WellNetwork3d(domain=self._domain)

    def is_well(self, grid: pp.Grid | pp.MortarGrid) -> bool:
        if isinstance(grid, pp.Grid):
            return getattr(grid, "well_num", -1) >= 0
        elif isinstance(grid, pp.MortarGrid):
            return False
        else:
            raise ValueError("Unknown grid type.")

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict[str, float]:
        # Default value of 1/2, scaled by the length unit.
        cell_size = self.solid.convert_units(0.5, "m")
        default_meshing_args: dict[str, float] = {"cell_size": cell_size}
        # If meshing arguments are provided in the params, they should already be
        # scaled by the length unit.
        return self.params.get("meshing_arguments", default_meshing_args)

    def meshing_kwargs(self) -> dict:
        meshing_kwargs = self.params.get("meshing_kwargs", None)
        if meshing_kwargs is None:
            meshing_kwargs = {}
        return meshing_kwargs

    def subdomains_to_interfaces(
        self, subdomains: list[pp.Grid], codims: list[int]
    ) -> list[pp.MortarGrid]:
        # Initialize list of interfaces, build it up one subdomain at a time.
        interfaces: list[pp.MortarGrid] = []
        for sd in subdomains:
            for intf in self.mdg.subdomain_to_interfaces(sd):
                if intf not in interfaces and intf.codim in codims:
                    interfaces.append(intf)
        return self.mdg.sort_interfaces(interfaces)

    def interfaces_to_subdomains(
        self, interfaces: list[pp.MortarGrid]
    ) -> list[pp.Grid]:
        subdomains: list[pp.Grid] = []
        for interface in interfaces:
            for sd in self.mdg.interface_to_subdomain_pair(interface):
                if sd not in subdomains:
                    subdomains.append(sd)
        return self.mdg.sort_subdomains(subdomains)

    def subdomains_to_boundary_grids(
        self, subdomains: Sequence[pp.Grid]
    ) -> Sequence[pp.BoundaryGrid]:
        boundary_grids = [self.mdg.subdomain_to_boundary_grid(sd) for sd in subdomains]
        return [bg for bg in boundary_grids if bg is not None]

    def wrap_grid_attribute(
        self,
        grids: Sequence[pp.GridLike],
        attr: str,
        *,
        dim: int,
    ) -> pp.ad.DenseArray:
        # NOTE: The enforcement of keyword-only arguments, combined with this class
        # being used as a mixin with other classes (thus this function represented as a
        # Callable in the other classes) does not make mypy happy. The problem seems to
        # be that a method specified as Callable must be called exactly as the type
        # specification, thus when this method is called with arguments
        #
        #   self.wrap_grid_attribute(..., dim=some_integer, ...)
        #
        # mypy will react on the difference between the type specification (that did not
        # include the dim argument) and the actual call. We also tried adding the *
        # (indicating the start of keyword-only in the type specification), but while
        # this made mypy happy, it is not vald syntax. The only viable solution (save
        # from using typing protocols, which we really do not want to do, there are
        # enough classes and inheritance in the mixin combination as it is) seems to be
        # to add a # type: ignore[call-arg] comment where the method is called. By only
        # ignoring call-args problems, we limit the risk of silencing other errors that
        # mypy might find.

        if len(grids) > 0:
            # Check that all grids have the sought after attribute. We could have
            # avoided this loop by surrounding the getattr with a try-except, but this
            # would have given a more convoluted code.
            if not all(hasattr(g, attr) for g in grids):
                raise ValueError(f"Grids do not have attribute {attr}")
            # Check that the attribute is a ndarray
            if not all(isinstance(getattr(g, attr), np.ndarray) for g in grids):
                raise ValueError(f"Attribute {attr} is not a ndarray")

            # NOTE: We do not rule out the combination of subdomains and interfaces
            # in the same list. There should be no chance of errors here, and although
            # such a case seems to EK at the moment to be a bit of an edge case, there
            # is no reason to rule it out.

            if dim is None:
                # Default to all dimensions
                vals = np.hstack([getattr(g, attr).ravel("F") for g in grids])
            else:
                # Only include the first dim dimensions
                # We need to force the array to be 2d here, in case the dimension
                # argument is given for a non-vector attribute like cell_volumes.
                vals = np.hstack(
                    [np.atleast_2d(getattr(g, attr))[:dim].ravel("F") for g in grids]
                )
        else:
            # For an empty list of grids, return an empty matrix
            vals = np.zeros(0)

        array = pp.ad.DenseArray(vals)
        array.set_name(f"Array wrapping attribute {attr} on {len(grids)} grids.")
        return array

    def basis(self, grids: Sequence[pp.GridLike], dim: int) -> list[pp.ad.SparseArray]:
        # NOTE: See self.wrap_grid_attribute for comments on typing when this method
        # is used as a mixin, and the need to add type-ignore[call-arg] on use of this
        # method.

        # Collect the basis functions for each dimension
        basis: list[pp.ad.SparseArray] = []
        for i in range(dim):
            basis.append(self.e_i(grids, i=i, dim=dim))
        # Stack the basis functions horizontally
        return basis

    def e_i(
        self, grids: Sequence[pp.GridLike], *, i: int, dim: int
    ) -> pp.ad.SparseArray:
        # NOTE: See self.wrap_grid_attribute for comments on typing when this method
        # is used as a mixin, and the need to add type-ignore[call-arg] on use of this
        # method.

        # TODO: Should we expand this to grids not aligned with the coordinate axes, and
        # possibly unify with ``porepy.utils.projections.TangentialNormalProjection``?
        # This is not a priority for the moment, though.

        if dim is None:
            dim = self.nd

        # Sanity checks
        if i >= dim:
            raise ValueError("Basis function index out of range")

        # Construct a single vector, and later stack it to a matrix
        # Collect the basis functions for each dimension
        e_i = np.zeros((dim, 1))
        e_i[i] = 1
        # Expand to cell-wise column vectors.
        num_cells = sum([g.num_cells for g in grids])
        # Expand to a matrix.
        mat = sps.kron(sps.eye(num_cells), e_i)
        return pp.ad.SparseArray(mat)

    # Local basis related methods
    def tangential_component(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # We first need an inner product (or dot product), i.e. extract the tangential
        # component of the cell-wise vector v to be transformed. Then we want to express
        # it in the tangential basis. The two operations are combined in a single
        # operator composed right to left: v will be hit by first e_i.T (row vector) and
        # secondly t_i (column vector). Ignore mypy keyword argument error.
        op: pp.ad.Operator = pp.ad.sum_operator_list(
            [
                self.e_i(subdomains, i=i, dim=self.nd - 1)  # type: ignore[arg-type]
                @ self.e_i(subdomains, i=i, dim=self.nd).T
                for i in range(self.nd - 1)
            ]
        )
        op.set_name("tangential_component")
        return op

    def normal_component(self, subdomains: list[pp.Grid]) -> pp.ad.SparseArray:
        # Create the basis function for the normal component (which is known to be the
        # last component).
        e_n = self.e_i(subdomains, i=self.nd - 1, dim=self.nd)
        e_n.set_name("normal_component")
        return e_n.T

    def local_coordinates(self, subdomains: list[pp.Grid]) -> pp.ad.SparseArray:
        # TODO: If we ever implement a mapping to reference space for all subdomains,
        # the present method should be revisited.

        # For now, assert all subdomains are fractures, i.e. dim == nd - 1.
        # TODO: Extend to all subdomains, not only codimension 1?
        assert all([sd.dim == self.nd - 1 for sd in subdomains])
        if len(subdomains) > 0:
            # Compute the local coordinates for each subdomain. For this, we use the
            # preset tangential_normal_projection attribute of the subdomains.
            local_coord_proj_list = [
                self.mdg.subdomain_data(sd)[
                    "tangential_normal_projection"
                ].project_tangential_normal(sd.num_cells)
                for sd in subdomains
            ]
            local_coord_proj = sps.block_diag(local_coord_proj_list)
        else:
            # Also treat no subdomains
            local_coord_proj = sps.csr_matrix((0, 0))
        return pp.ad.SparseArray(local_coord_proj)

    def subdomain_projections(self, dim: int) -> pp.ad.SubdomainProjections:
        name = f"_subdomain_proj_of_dimension_{dim}"
        if hasattr(self, name):
            proj = getattr(self, name)
        else:
            proj = pp.ad.SubdomainProjections(self.mdg.subdomains(), dim)
            setattr(self, name, proj)
        return proj

    def domain_boundary_sides(
        self, domain: pp.Grid | pp.BoundaryGrid, tol: Optional[float] = 1e-10
    ) -> pp.domain.DomainSides:
        if isinstance(domain, pp.Grid):
            # bc_type_* methods ... require working with subdomains

            face_centers = domain.face_centers
            num_faces = domain.num_faces
            all_bf = domain.get_boundary_faces()
        elif isinstance(domain, pp.BoundaryGrid):
            # Cells of the boundary grid are faces of the parent subdomain.
            face_centers = domain.cell_centers
            num_faces = domain.num_cells
            all_bf = np.arange(num_faces)
        else:
            raise ValueError(
                "Domain must be either Grid or BoundaryGrid. Provided:", domain
            )

        # Get domain boundary sides
        box = copy.deepcopy(self.domain.bounding_box)

        east = np.abs(box["xmax"] - face_centers[0]) <= tol
        west = np.abs(box["xmin"] - face_centers[0]) <= tol
        if self.mdg.dim_max() == 1:
            north = np.zeros(num_faces, dtype=bool)
            south = north.copy()
        else:
            north = np.abs(box["ymax"] - face_centers[1]) <= tol
            south = np.abs(box["ymin"] - face_centers[1]) <= tol
        if self.mdg.dim_max() < 3:
            top = np.zeros(num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = np.abs(box["zmax"] - face_centers[2]) <= tol
            bottom = np.abs(box["zmin"] - face_centers[2]) <= tol

        # Create a namedtuple to store the arrays
        domain_sides = pp.domain.DomainSides(
            all_bf, east, west, north, south, top, bottom
        )

        return domain_sides

    def internal_boundary_normal_to_outwards(
        self,
        subdomains: list[pp.Grid],
        *,
        dim: int,
    ) -> pp.ad.Operator:
        if len(subdomains) == 0:
            # Special case if no interfaces.
            sign_flipper = pp.ad.SparseArray(sps.csr_matrix((0, 0)))
        else:
            # There is already a method to construct a switcher matrix in grid_utils,
            # so we use that. Loop over all subdomains, construct a local switcher
            # matrix and store it. The mixed-dimensional version can then be constructed
            # by block diagonal concatenation.
            # NOTE: While it is somewhat unfortunate to have the method in grid_utils,
            # since this leads to a more nested code, it also means we can use the
            # method outside the Ad framework. For the time being, we keep it.
            matrices = []
            for sd in subdomains:
                # Use the tagging of fracture surfaces to identify the faces on internal
                # boundaries.
                faces_on_fracture_surface = np.where(sd.tags["fracture_faces"])[0]
                switcher_int = pp.grid_utils.switch_sign_if_inwards_normal(
                    sd, dim, faces_on_fracture_surface
                )
                matrices.append(switcher_int)

            # Construct the block diagonal matrix.
            sign_flipper = pp.ad.SparseArray(sps.block_diag(matrices).tocsr())
        sign_flipper.set_name("Flip_normal_vectors")
        return sign_flipper

    def outwards_internal_boundary_normals(
        self,
        interfaces: list[pp.MortarGrid],
        *,
        unitary: bool,
    ) -> pp.ad.Operator:
        # NOTE: See self.wrap_grid_attribute for comments on typing when this method
        # is used as a mixin, and the need to add type-ignore[call-arg] on use of this
        # method.

        if len(interfaces) == 0:
            # Special case if no interfaces.
            return pp.ad.DenseArray(np.zeros(0))

        # Main ingredients: Normal vectors for primary subdomains for each interface,
        # and a switcher matrix to flip the sign if the normal vector points inwards.
        # The first is constructed herein, the second is a method of this class.

        # Since the normal vectors are stored on the primary subdomains, but are to be
        # computed on the interfaces, we need mortar projections.

        # Get hold of the primary subdomains, i.e. the higher-dimensional neighbors of
        # the interfaces.
        primary_subdomains: list[pp.Grid] = []
        for intf in interfaces:
            primary_subdomains.append(self.mdg.interface_to_subdomain_pair(intf)[0])

        # Projection operator between the subdomains and interfaces. The projection is
        # constructed to only consider the higher-dimensional subdomains.
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, primary_subdomains, interfaces, dim=self.nd
        )
        # Ignore mypy complaint about unexpected keyword arguments.
        primary_face_normals = self.wrap_grid_attribute(
            primary_subdomains, "face_normals", dim=self.nd  # type: ignore[call-arg]
        )
        # Account for sign of boundary face normals. This will give a matrix with a
        # shape equal to the total number of faces in all primary subdomains.
        # Ignore mypy complaint about unexpected keyword arguments.
        flip = self.internal_boundary_normal_to_outwards(
            primary_subdomains, dim=self.nd  # type: ignore[call-arg]
        )
        # Flip the normal vectors. Unravelled from the right: Restrict from faces on all
        # subdomains to the primary ones, multiply with the face normals, flip the
        # signs, and project back up to all subdomains.
        flipped_normals = flip @ primary_face_normals
        # Project to mortar grid, as a mapping from mortar to the subdomains and back
        # again. If we are to use cell_volumes from interfaces to normalize, projection
        # must logically be integration, not average. This also means that the normals
        # have length equal to cell_volume on mortar grids, by analogy to face_area for
        # subdomains.
        outwards_normals = mortar_projection.primary_to_mortar_int @ flipped_normals
        outwards_normals.set_name("outwards_internal_boundary_normals")

        # Normalize by face area if requested.
        if unitary:
            # 1 over cell volumes on the interfaces
            # Ignore mypy complaint about unexpected keyword arguments.
            cell_volumes_inv = pp.ad.Scalar(1) / self.wrap_grid_attribute(
                interfaces, "cell_volumes", dim=self.nd  # type: ignore[call-arg]
            )

            # Expand cell volumes to nd by multiplying from left by e_i and summing
            # over all dimensions.
            cell_volumes_inv_nd = pp.ad.sum_operator_list(
                [e @ cell_volumes_inv for e in self.basis(interfaces, self.nd)]
            )
            # Scale normals.
            outwards_normals = cell_volumes_inv_nd * outwards_normals
            outwards_normals.set_name("unitary_outwards_internal_boundary_normals")

        return outwards_normals
