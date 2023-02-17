"""
A module for representation and manipulations of fractures and fracture sets.

The model relies heavily on functions in the computational geometry library.

"""
from __future__ import annotations

import copy
import csv
import logging
import time
from typing import Optional, Union

import meshio
import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull

import porepy as pp
from porepy.fracs.gmsh_interface import GmshData3d, GmshWriter
from porepy.fracs.plane_fracture import PlaneFracture
from porepy.utils import setmembership, sort_points

from .gmsh_interface import Tags as GmshInterfaceTags

# Module-wide logger
logger = logging.getLogger(__name__)


class FractureNetwork3d(object):
    """Representation of a set of plane fractures in a three-dimensional domain.

    Collection of Fractures with geometrical information. Facilitates computation of
    intersections of the fracture. Also incorporates the bounding box of the domain.
    To ensure that all fractures lie within the box,
    call :meth:`impose_external_boundary()` _after_ all fractures have been specified.

    Parameters:
        fractures (list of Fracture, optional): Fractures that make up the network.
            Defaults to None, which will create a domain empty of fractures.
        domain (pp.Domain): Domain specification.
        tol (double, optional): Tolerance used in geometric computations. Defaults
            to 1e-8.
        run_checks (boolean, optional): Run consistency checks during the network
            processing. Can be considered a limited debug mode. Defaults to False.

    """

    def __init__(
        self,
        fractures: Optional[list[PlaneFracture]] = None,
        domain: Optional[pp.Domain] = None,
        tol: float = 1e-8,
        run_checks: bool = False,
    ) -> None:

        # Initialize fractures as an empy list
        self.fractures = []
        """All fractures forming the network."""

        if fractures is not None:
            for f in fractures:
                self.fractures.append(f)

        for i, f in enumerate(self.fractures):
            f.set_index(i)

        # Store intersection information as a dictionary. Keep track of
        # the intersecting fractures, the start and end point of the intersection line,
        # and whether the intersection is on the boundary of the fractures.
        # Note that a Y-type intersection between three fractures is represented
        # as three intersections.
        self.intersections: dict[str, np.ndarray] = {
            "first": np.array([], dtype=object),
            "second": np.array([], dtype=object),
            "start": np.zeros((3, 0)),
            "end": np.zeros((3, 0)),
            "bound_first": np.array([], dtype=bool),
            "bound_second": np.array([], dtype=bool),
        }
        """All known intersections in the network."""

        self.has_checked_intersections = False
        """If True, the intersection finder method has been run. Useful in meshing
        algorithms to avoid recomputing known information.

        """

        self.tol = tol
        """Geometric tolerance used in computations."""

        self.run_checks = run_checks
        """Wheter to run consistency checks during the network processing."""

        # Initialize with an empty domain unless given explicitly. Can be modified
        # later by a call to 'impose_external_boundary()'
        self.domain: pp.Domain | None = domain
        """Domain specification. See class:`~porepy.geometry.domain.Domain`."""

        # Initialize mesh size parameters as empty
        self.mesh_size_min = None
        """Mesh size parameter, minimum mesh size to be sent to gmsh. Set by
        insert_auxiliary_points().

        """

        self.mesh_size_frac = None
        """Mesh size parameter. Ideal mesh size, fed to gmsh. Set by
        insert_auxiliary_points().

        """

        self.mesh_size_bound = None
        """Mesh size parameter. Boundary mesh size, fed to gmsh. Set by
        insert_auxiliary_points().

        """

        # Assign an empty tag dictionary
        self.tags: dict[str, list[bool]] = {}
        """Tags used on Fractures and subdomain boundaries."""

        # No auxiliary points have been added
        self.auxiliary_points_added = False
        """Mesh size parameter. If True, extra points have been added to PlaneFracture
        geometry to facilitate mesh size tuning.

        """

        self.bounding_box_imposed = False
        """Whether a bounding box have been imposed to the set of fractures."""

        self.decomposition: dict
        """Splitting of network, accounting for fracture intersections etc. Necessary
        pre-processing before meshing. Added by :meth:`~split_intersections()`.

        """

    def add(self, f):
        """Add a fracture to the network.

        The fracture will be assigned a new index, higher than the maximum
        value currently found in the network.

        Parameters:
            f (Fracture): Fracture to be added.

        """
        ind = np.array([f.index for f in self.fractures])

        if ind.size > 0:
            f.set_index(np.max(ind) + 1)
        else:
            f.set_index(0)
        self.fractures.append(f)

    def copy(self):
        """Create deep copy of the network.

        The method will create a deep copy of all fractures, as well as the domain, of
        the network. Note that if the fractures have had extra points imposed as part
        of a meshing procedure, these will be included in the copied fractures.

        Returns:
            pp.FractureNetwork3d.

        """
        fracs = [f.copy() for f in self.fractures]

        domain = self.domain
        if domain is not None:
            # Get a deep copy of domain, but no need to do that if domain is None
            if domain.is_boxed:
                box = copy.deepcopy(domain.bounding_box)
                domain = pp.Domain(bounding_box=box)
            else:
                polytope = domain.polytope.copy()
                domain = pp.Domain(polytope=polytope)

        return FractureNetwork3d(fracs, domain, self.tol)

    def num_frac(self) -> int:
        """Get number of fractures in the network.

        Planes on the domain boundary are not included.

        Returns:
            int: Number of network fractures.
        """
        num = 0
        if not self.bounding_box_imposed:
            return len(self.fractures)

        # The boundary is set when the bounding box is imposed.
        for fi in range(len(self.fractures)):
            if not self.tags["boundary"][fi]:
                num += 1

        return num

    def mesh(
        self,
        mesh_args: dict[str, float],
        dfn: bool = False,
        file_name: Optional[str] = None,
        constraints: Optional[np.ndarray] = None,
        write_geo: bool = True,
        tags_to_transfer: Optional[list[str]] = None,
        finalize_gmsh: bool = True,
        clear_gmsh: bool = False,
        **kwargs,
    ) -> pp.MixedDimensionalGrid:
        """Mesh the fracture network, and generate a mixed-dimensional grid.

        The mesh itself is generated by Gmsh.

        Parameters:
            mesh_args (dict): Should contain fields ``mesh_size_frac``,
                ``mesh_size_min``, which represent the ideal mesh size at the fracture,
                and the minimum mesh size passed to gmsh. Can also contain
                ``mesh_size_bound``, which gives the far-field (boundary) mesh size.
            dfn (boolean, optional): If ``True``, a DFN mesh (of the network, but not
                the surrounding matrix) is created.
            file_name (str, optional): Name of file used to communicate with gmsh.
                Defaults to ``gmsh_frac_file``. The gmsh configuration file will be
                ``file_name.geo``, while the mesh is dumped to ``file_name.msh``.
            constraints (np.array): Index list of elements in the fracture list that
                should be treated as constraints in meshing, but not added as separate
                fracture grids (no splitting of nodes etc.). Useful to define subregions
                 of the domain (and assign e.g., sources, material properties, etc.).
            write_geo (bool, optional): If True (default), the gmsh configuration
                will be written to a ``.geo_unrolled`` file.
            tags_to_transfer (list of strings, optional): Tags, in ``self.tags`` to be
                transferred to the fracture grids. Provisional functionality.
            finalize_gmsh (boolean): If ``True`` (default), the port to Gmsh is closed
                when meshing is completed. On repeated invocations of Gmsh in the same
                Python session, a memory leak in Gmsh may cause reduced performance
                (written spring 2021). In these cases, it may be better to finalize gmsh
                externally to this class. See also ``clear_gmsh``.
            clear_gmsh (boolean, optional): If True, the geometry representation in gmsh
                is deleted when meshing is completed. This is of use only if
                ``finalize_gmsh`` is set to ``False``, in which case it may be desirable
                to delete the old geometry before adding a new one. Defaults to
                ``False``.

        Returns:
            MixedDimensionalGrid: Mixed-dimensional mesh.

        """
        if file_name is None:
            file_name = "gmsh_frac_file.msh"

        gmsh_repr = self.prepare_for_gmsh(
            mesh_args,
            dfn,
            constraints,
        )

        gmsh_writer = GmshWriter(gmsh_repr)

        if dfn:
            dim_meshing = 2
        else:
            dim_meshing = 3

        gmsh_writer.generate(
            file_name,
            dim_meshing,
            write_geo=write_geo,
            finalize=finalize_gmsh,
            clear_gmsh=clear_gmsh,
        )

        if dfn:
            subdomains = pp.fracs.simplex.triangle_grid_embedded(file_name)
        else:
            # Process the gmsh .msh output file, to make a list of grids
            subdomains = pp.fracs.simplex.tetrahedral_grid_from_gmsh(
                file_name, constraints
            )

        if tags_to_transfer:
            for id_g, g in enumerate(subdomains[1 - int(dfn)]):
                for key in tags_to_transfer:
                    if key not in g.tags:
                        g.tags[key] = self.tags[key][id_g]

        # Merge the grids into a mixed-dimensional MixedDimensionalGrid
        mdg = pp.meshing.subdomains_to_mdg(subdomains, **kwargs)
        return mdg

    def prepare_for_gmsh(self, mesh_args, dfn=False, constraints=None) -> GmshData3d:
        """Process network intersections and write a gmsh .geo configuration file,
        ready to be processed by gmsh.

        NOTE: Consider using the mesh() function instead to get a ready
        MixedDimensionalGrid.

        Parameters:
            mesh_args (dict): Should contain fields 'mesh_size_frac', 'mesh_size_min',
                which represent the ideal mesh size at the fracture, and the
                minimum mesh size passed to gmsh. Can also contain
                'mesh_size_bound', which gives the far-field (boundary) mesh
                size.
            dfn (boolean, optional): If True, a DFN mesh (of the network, but not
                the surrounding matrix) is created.
            constraints (np.array): Index list of elements in the fracture list that
                should be treated as constraints in meshing, but not added as separate
                fracture grids (no splitting of nodes etc.).

        Returns:
            str: Name of .geo file.

        """

        # The implementation in this function is fairly straightforward, all
        # technical difficulties are hidden in other functions.

        # Impose the boundary of the domain. This may cut and split fractures.
        # FIXME: When fractures are split or deleted, there will likely be
        # mapping errors between fracture indices.
        if not dfn and not self.bounding_box_imposed:
            self.impose_external_boundary(self.domain)

        if constraints is None:
            constraints = np.array([], dtype=int)

        # Find intersections between fractures
        if not self.has_checked_intersections:
            self.find_intersections()
        else:
            logger.info("Use existing intersections")

        if "mesh_size_frac" not in mesh_args.keys():
            raise ValueError("Meshing algorithm needs argument mesh_size_frac")
        if "mesh_size_min" not in mesh_args.keys():
            raise ValueError("Meshing algorithm needs argument mesh_size_min")

        # Insert auxiliary points for mesh size control. This is done after the
        # intersections are found, but before the fractures are collected into a
        # set of edges (see split_intersections).
        mesh_size_frac = mesh_args.get("mesh_size_frac", None)
        mesh_size_min = mesh_args.get("mesh_size_min", None)
        mesh_size_bound = mesh_args.get("mesh_size_bound", None)
        self._insert_auxiliary_points(mesh_size_frac, mesh_size_min, mesh_size_bound)

        # Process intersections to get a description of the geometry in non-
        # intersecting lines and polygons
        self.split_intersections()

        # Having found all intersections etc., the next step is to classify the
        # geometric objects before representing them in the data format expected by
        # the Gmsh interface.
        # The classification is somewhat complex, since, for certain applications, it is
        # necessary with a detailed description of different objects.

        # Extract geometrical information.
        p = self.decomposition["points"]
        edges = self.decomposition["edges"]
        poly = self._poly_2_segment()

        has_boundary = "boundary" in self.tags

        # Get preliminary set of tags for the edges. Also find which edges are
        # interior to all or only some edges
        edge_tags, not_boundary_edge, some_boundary_edge = self._classify_edges(
            poly, constraints
        )

        # All intersection lines and points on boundaries are non-physical in 3d.
        # I.e., they are assigned boundary conditions, but are not gridded. Hence:
        # Remove the points and edges at the boundary
        point_tags, edge_tags = self._on_domain_boundary(edges, edge_tags)

        # To ensure that polygons that are constraints, but not fractures, do not
        # trigger the generation of 1d fracture intersections, or 0d point grids, some
        # steps are needed.

        # Count the number of times a lines is defined as 'inside' a fracture, and
        # the number of times this is caused by a constraint
        in_frac_occurrences = np.zeros(edges.shape[1], dtype=int)
        in_frac_occurrences_by_constraints = np.zeros(edges.shape[1], dtype=int)

        for poly_ind, poly_edges in enumerate(self.decomposition["line_in_frac"]):
            for edge_ind in poly_edges:
                in_frac_occurrences[edge_ind] += 1
                if poly_ind in constraints:
                    in_frac_occurrences_by_constraints[edge_ind] += 1

        # Count the number of occurrences that are not caused by a constraint
        num_occ_not_by_constraints = (
            in_frac_occurrences - in_frac_occurrences_by_constraints
        )

        # If all but one occurrence of a line internal to a polygon is caused by
        # constraints, this is likely a fracture.
        auxiliary_line = num_occ_not_by_constraints == 1

        # .. However, the line may also be caused by a T- or L-intersection
        auxiliary_line[some_boundary_edge] = False
        # The edge tags for internal lines were set accordingly in self._classify_edges.
        # Update to auxiliary line if this was really what we had.
        edge_tags[auxiliary_line] = GmshInterfaceTags.AUXILIARY_LINE.value

        # .. and we're done with edges (until someone defines a new special case)
        # Next, find intersection points.

        # Count the number of times a point is referred to by an intersection
        # between two fractures. If this is more than one, the point should
        # have a 0-d grid assigned to it.
        # Here, we must account for lines that are internal, but not those that have
        # been found to be triggered by constraints.
        intersection_edge = np.logical_and(
            not_boundary_edge, np.logical_not(auxiliary_line)
        )

        # All intersection points should occur at least twice
        isect_p = edges[:2, intersection_edge].ravel()
        num_occ_isect_pt = np.bincount(isect_p)
        # All points that occur more than once could be intersection points
        prelim_intersection_candidate = np.where(num_occ_isect_pt > 1)[0]

        # We should also consider points that are both on a fracture tip and a domain
        # boundary line - these will be added to the fracture and boundary points below.
        domain_boundary_line = edge_tags == GmshInterfaceTags.DOMAIN_BOUNDARY_LINE.value
        fracture_tip_line = edge_tags == GmshInterfaceTags.FRACTURE_TIP.value
        domain_boundary_point = edges[:2, domain_boundary_line]
        fracture_tip_point = edges[:2, fracture_tip_line]
        domain_and_tip_point = np.intersect1d(domain_boundary_point, fracture_tip_point)

        intersection_point_candidates = np.hstack(
            (prelim_intersection_candidate, domain_and_tip_point)
        )

        # .. however, this is not enough: If a fracture and a constraint intersect at
        # a domain boundary, the candidate is not a true intersection point.
        # Loop over all candidates, find all polygons that have this as part of an edge,
        # and count the number of those polygons that are fractures (e.g. not boundary
        # or constraint). If there is more than one, this is indeed  a fracture
        # intersection and an intersection point grid should be assigned.
        # (Reason for more than one, not two, I believe is: Two fractures crossing will
        # not make a 0d point by themselves). Point must then come either from a
        # third fracture, or a constraint. In the latter case, we anyhow need the
        # intersection point as a grid to get dynamics between the intersection line
        # correctly represented.
        intersection_points = []
        # Points that are both on a boundary, and on a fracture. This may represent one
        # of a few cases: A fracture-constraint intersection on a boundary, and/or a
        # fracture crossing a domain surface boundary.
        fracture_and_boundary_points = []

        for pi in intersection_point_candidates:
            # edges of this point
            _, edge_ind = np.where(edges == pi)

            # Fractures of the edge
            frac_arr = np.array([], dtype=int)
            for e in edge_ind:
                frac_arr = np.append(frac_arr, self.decomposition["edges_2_frac"][e])
            # Uniquify.
            unique_fracs = np.unique(frac_arr)

            # If the domain has a boundary, classify the polygons of this point as
            # fracture, fracture or boundary (and tacitly, constraints)
            if has_boundary:
                is_boundary = np.array(self.tags["boundary"])[unique_fracs]
                is_frac = np.logical_not(
                    np.logical_or(np.in1d(unique_fracs, constraints), is_boundary)
                )
                is_frac_or_boundary = np.logical_not(
                    np.in1d(unique_fracs, constraints),
                )
            else:
                is_frac = np.logical_not(np.in1d(unique_fracs, constraints))
                is_frac_or_boundary = is_frac

            # If more than one fracture share this point, it is an intersection point.
            # See comment above on why not > 2.
            if is_frac.sum() > 1:
                intersection_points.append(pi)
            # If the point is on a fracture, and on a boundary surface, it will be
            # classified otherwise.
            if is_frac.sum() > 0 and (is_frac_or_boundary.sum() - is_frac.sum()) > 0:
                fracture_and_boundary_points.append(pi)

        # Finally, we have the full set of intersection points (take a good laugh when
        # finding this line in the next round of debugging).

        # Candidates that were not intersections. Here we do not consider the
        # boundary-tip combinations.
        fracture_constraint_intersection = np.setdiff1d(
            prelim_intersection_candidate, fracture_and_boundary_points
        )

        # Special tag for intersection between fracture and constraint.
        # These are not needed in the gmsh postprocessing (will not produce 0d grids),
        # but it can be useful to mark them for other purposes (EK: DFM upscaling)
        point_tags[
            fracture_constraint_intersection
        ] = GmshInterfaceTags.FRACTURE_CONSTRAINT_INTERSECTION_POINT.value

        point_tags[
            fracture_and_boundary_points
        ] = GmshInterfaceTags.FRACTURE_BOUNDARY_POINT.value

        # We're done! Hurrah!

        # Find points tagged as on the domain boundary
        boundary_points = np.where(
            point_tags == GmshInterfaceTags.DOMAIN_BOUNDARY_POINT.value
        )[0]

        fracture_boundary_points = np.where(
            point_tags == GmshInterfaceTags.FRACTURE_BOUNDARY_POINT.value
        )[0]

        # Intersections on the boundary should not have a 0d grid assigned
        true_intersection_points = np.setdiff1d(
            intersection_points, np.hstack((boundary_points, fracture_boundary_points))
        )

        edges = np.vstack((self.decomposition["edges"], edge_tags))

        # Obtain mesh size parameters
        mesh_size = self._determine_mesh_size(point_tags=point_tags)

        line_in_poly = self.decomposition["line_in_frac"]

        physical_points: dict[int, GmshInterfaceTags] = {}
        for pi in fracture_boundary_points:
            physical_points[pi] = GmshInterfaceTags.FRACTURE_BOUNDARY_POINT

        for pi in fracture_constraint_intersection:
            physical_points[
                pi
            ] = GmshInterfaceTags.FRACTURE_CONSTRAINT_INTERSECTION_POINT

        for pi in boundary_points:
            physical_points[pi] = GmshInterfaceTags.DOMAIN_BOUNDARY_POINT

        for pi in true_intersection_points:
            physical_points[pi] = GmshInterfaceTags.FRACTURE_INTERSECTION_POINT

        # Use separate structures to store tags and physical names for the polygons.
        # The former is used for feeding information into gmsh, while the latter is
        # used to tag information in the output from gmsh. They are kept as separate
        # variables since a user may want to modify the physical surfaces before
        # generating the .msh file.
        physical_surfaces: dict[int, GmshInterfaceTags] = dict()
        polygon_tags: dict[int, GmshInterfaceTags] = dict()

        for fi, _ in enumerate(self.fractures):
            # Translate from numerical tags to the GmshInterfaceTags system.
            if has_boundary and self.tags["boundary"][fi]:
                physical_surfaces[fi] = GmshInterfaceTags.DOMAIN_BOUNDARY_SURFACE
                polygon_tags[fi] = GmshInterfaceTags.DOMAIN_BOUNDARY_SURFACE
            elif fi in constraints:
                physical_surfaces[fi] = GmshInterfaceTags.AUXILIARY_PLANE
                polygon_tags[fi] = GmshInterfaceTags.AUXILIARY_PLANE
            else:
                physical_surfaces[fi] = GmshInterfaceTags.FRACTURE
                polygon_tags[fi] = GmshInterfaceTags.FRACTURE

        physical_lines: dict[int, GmshInterfaceTags] = dict()
        for ei, tag in enumerate(edges[2]):
            if tag in (
                GmshInterfaceTags.FRACTURE_TIP.value,
                GmshInterfaceTags.FRACTURE_INTERSECTION_LINE.value,
                GmshInterfaceTags.DOMAIN_BOUNDARY_LINE.value,
                GmshInterfaceTags.FRACTURE_BOUNDARY_LINE.value,
                GmshInterfaceTags.AUXILIARY_LINE.value,
            ):
                # Translate from the numerical value of the line to the tag in the
                # Gmsh interface.
                physical_lines[ei] = GmshInterfaceTags(tag)

        gmsh_repr = GmshData3d(
            dim=3,
            pts=p,
            lines=edges,
            mesh_size=mesh_size,
            polygons=poly,
            physical_surfaces=physical_surfaces,
            polygon_tags=polygon_tags,
            lines_in_surface=line_in_poly,
            physical_points=physical_points,
            physical_lines=physical_lines,
        )

        self.decomposition["edge_tags"] = edges[2]
        self.decomposition["domain_boundary_points"] = boundary_points
        self.decomposition["point_tags"] = point_tags

        return gmsh_repr

    def __getitem__(self, position):
        return self.fractures[position]

    def intersections_of_fracture(
        self, frac: Union[int, PlaneFracture]
    ) -> tuple[list[int], list[bool]]:
        """Get all known intersections for a fracture.

        If called before find_intersections(), the returned list will be empty.

        Parameters:
            frac: A fracture in the network

        Returns:
            np.array (Intersection): Array of intersections

        """
        fi: Optional[int]  # need this typing since frac.index might also be None
        if isinstance(frac, int):
            fi = frac
        else:
            fi = frac.index

        isects = []
        is_first = []
        for i in range(len(self.intersections["first"])):
            if self.intersections["first"][i].index == fi:
                isects.append(i)
                is_first.append(True)
            elif self.intersections["second"][i].index == fi:
                isects.append(i)
                is_first.append((False))

        return isects, is_first

    def find_intersections(self, use_orig_points=False):
        """
        Find intersections between fractures in terms of coordinates.

        The intersections are stored in the attribute self.Intersections.

        Handling of the intersections (splitting into non-intersecting
        polygons, paving the way for gridding) is taken care of by the function
        split_intersections().

        Note that find_intersections() should be invoked after external
        boundaries are imposed. If the reverse order is applied, intersections
        outside the domain may be identified, with unknown consequences for the
        reliability of the methods. If intersections outside the bounding box
        are of interest, these can be found by setting the parameter
        use_orig_points to True.

        Parameters:
            use_orig_points (boolean, optional): Whether to use the original
                fracture description in the search for intersections. Defaults
                to False. If True, all fractures will have their attribute p
                reset to their original value.

        """
        self.has_checked_intersections = True
        logger.info("Find intersection between fractures")
        start_time = time.time()

        # If desired, use the original points in the fracture intersection.
        # This will reset the field self._fractures.pts, and thus revoke
        # modifications due to boundaries etc.
        if use_orig_points:
            for f in self.fractures:
                f.pts = f.orig_pts

        # Intersections are found using a method in the comp_geom module, which requires
        # the fractures to be represented as a list of polygons.
        polys = [f.pts for f in self.fractures]

        # Obtain intersection points, indexes of intersection points for each fracture
        # information on whether the fracture is on the boundary, and pairs of fractures
        # that intersect.
        # We do not include point contacts here - that should be feasible, but the
        # implementation may require some effort (e.g. point contact in the interior of
        # a fracture would require embedding points in surfaces in the gmsh specification).
        isect, point_ind, bound_info, frac_pairs, *_ = pp.intersections.polygons_3d(
            polys, include_point_contact=False
        )

        # Loop over all pairs of intersection pairs, add the intersections to the
        # internal list.
        for pair in frac_pairs:
            # Indices of the relevant pairs.
            ind_0, ind_1 = pair
            # Find the common indices of intersection points (referring to isect)
            # and find where they are located in the array point_ind[ind_0]
            common_ind, i0 = pp.utils.setmembership.ismember_rows(
                point_ind[ind_1], point_ind[ind_0]
            )
            # Convert from boolean indices to indices
            common_ind = np.where(common_ind)[0]
            # There should be exactly two intersection points
            assert common_ind.size == 2

            # Do the same exercise with the second fracture
            common_ind_2, i1 = pp.utils.setmembership.ismember_rows(
                point_ind[ind_0], point_ind[ind_1]
            )
            common_ind_2 = np.where(common_ind_2)[0]
            assert common_ind_2.size == 2

            # Check if the intersection points form a boundary edge of each of the
            # fractures. The indexing is a bit involved, but is based on there being
            # two intersection points for each segment - thus the indices in i0 and i1
            # must be divided by two.
            on_bound_0 = bound_info[ind_0][np.floor(i0[0] / 2).astype(int)]
            on_bound_1 = bound_info[ind_1][np.floor(i1[0] / 2).astype(int)]

            # Add the intersection to the internal storage
            self._add_intersection(
                self.fractures[ind_0],
                self.fractures[ind_1],
                isect[:, point_ind[ind_1][common_ind[0]]],
                isect[:, point_ind[ind_1][common_ind[1]]],
                bound_first=on_bound_0,
                bound_second=on_bound_1,
            )
        logger.info(
            "Found %i intersections. Elapsed time: %.5f",
            len(self.intersections),
            time.time() - start_time,
        )

    def split_intersections(self):
        """
        Based on the fracture network, and their known intersections, decompose
        the fractures into non-intersecting sub-polygons. These can
        subsequently be exported to gmsh.

        The method will add an attribute decomposition to self.

        """

        logger.info("Split intersections")
        start_time = time.time()

        # First, collate all points and edges used to describe fracture
        # boundaries and intersections.
        all_p, edges, edges_2_frac, is_boundary_edge = self._point_and_edge_lists()

        # By now, all segments in the grid are defined by a unique set of
        # points and edges. The next task is to identify intersecting edges,
        # and split them.
        all_p, edges, edges_2_frac, is_boundary_edge = self._remove_edge_intersections(
            all_p, edges, edges_2_frac, is_boundary_edge
        )

        if self.run_checks:
            self._verify_fractures_in_plane(all_p, edges, edges_2_frac)

        # Store the full decomposition.
        self.decomposition = {
            "points": all_p,
            "edges": edges.astype("int"),
            "is_bound": is_boundary_edge,
            "edges_2_frac": edges_2_frac,
        }
        polygons = []
        line_in_frac = []
        for fi, _ in enumerate(self.fractures):
            ei = []
            ei_bound = []
            # Find the edges of this fracture, add to either internal or
            # external fracture list
            for i, e in enumerate(zip(edges_2_frac, is_boundary_edge)):
                # Check that the boundary information matches the fractures
                assert e[0].size == e[1].size
                hit = np.where(e[0] == fi)[0]
                if hit.size == 1:
                    if e[1][hit]:
                        ei_bound.append(i)
                    else:
                        ei.append(i)
                elif hit.size > 1:
                    raise ValueError("Non-unique fracture edge relation")
                else:
                    continue

            poly, _ = sort_points.sort_point_pairs(edges[:2, ei_bound])
            polygons.append(poly)
            line_in_frac.append(ei)

        self.decomposition["polygons"] = polygons
        self.decomposition["line_in_frac"] = line_in_frac

        logger.info(
            "Finished fracture splitting after %.5f seconds", time.time() - start_time
        )

    def _fracs_2_edges(self, edges_2_frac):
        """Invert the mapping between edges and fractures.

        Returns:
            List of list: For each fracture, index of all edges that points to
                it.
        """
        f2e = []
        for fi in range(len(self.fractures)):
            f_l = []
            for ei, e in enumerate(edges_2_frac):
                if fi in e:
                    f_l.append(ei)
            f2e.append(f_l)
        return f2e

    def _point_and_edge_lists(self):
        """
        Obtain lists of all points and connections necessary to describe
        fractures and their intersections.

        Returns:
            np.ndarray, 3xn: Unique coordinates of all points used to describe
            the fracture polygons, and their intersections.
            np.ndarray, 2xn_edge: Connections between points, formed either
                by a fracture boundary, or a fracture intersection.
            list: For each edge, index of all fractures that point to the
                edge.
            np.ndarray of bool (size=num_edges): A flag telling whether the
                edge is on the boundary of a fracture.

        """
        # The workflow is based on first collecting information for all fractures,
        # next get information for all intersections between fractures.

        logger.info("Compile list of points and edges")
        start_time = time.time()

        # Field for all points in the fracture description
        all_p = np.empty((3, 0))
        # All edges, either as fracture boundary, or fracture intersection
        edges = np.empty((2, 0))
        # For each edge, a list of all fractures pointing to the edge.
        edges_2_frac = []

        # Field to know if an edge is on the boundary of a fracture.
        # Not sure what to do with a T-type intersection here
        is_boundary_edge = []

        # First loop over all fractures. All edges are assumed to be new; we
        # will deal with coinciding points later.
        for fi, frac in enumerate(self.fractures):
            num_p = all_p.shape[1]
            num_p_loc = frac.pts.shape[1]
            all_p = np.hstack((all_p, frac.pts))

            loc_e = num_p + np.vstack(
                (np.arange(num_p_loc), (np.arange(num_p_loc) + 1) % num_p_loc)
            )
            edges = np.hstack((edges, loc_e))
            for i in range(num_p_loc):
                edges_2_frac.append([fi])
                is_boundary_edge.append([True])

        # Next, add points relating to the intersections between fractures.
        # Since the intersections are already defined as numpy arrays, this is
        # relatively straightforward.
        num_isect = self.intersections["start"].shape[1]
        num_p = all_p.shape[1]
        # Intersection points are added by first all starts, then all ends
        isect_pt = np.hstack((self.intersections["start"], self.intersections["end"]))
        # Intersection edges are offset by the number of fracture edges
        intersection_edges = num_p + np.vstack(
            (np.arange(num_isect), num_isect + np.arange(num_isect))
        )
        # Merge fields
        all_p = np.hstack((all_p, isect_pt))
        edges = np.hstack((edges, intersection_edges))

        # Mapping from intersections to their fractures
        isect_2_frac = [
            [
                self.intersections["first"][i].index,
                self.intersections["second"][i].index,
            ]
            for i in range(num_isect)
        ]
        edges_2_frac += isect_2_frac

        # Boolean for intersections being on fracture boundaries
        isect_is_boundary = [
            [
                self.intersections["bound_first"][i],
                self.intersections["bound_second"][i],
            ]
            for i in range(num_isect)
        ]
        is_boundary_edge += isect_is_boundary

        # Ensure that edges are integers
        edges = edges.astype("int")

        logger.info(
            "Points and edges done. Elapsed time %.5f", time.time() - start_time
        )
        # Uniquify the points and edges before returning.
        return self._uniquify_points_and_edges(
            all_p, edges, edges_2_frac, is_boundary_edge
        )

    def _uniquify_points_and_edges(self, all_p, edges, edges_2_frac, is_boundary_edge):

        start_time = time.time()
        logger.info(
            """Uniquify points and edges, starting with %i points, %i
                    edges""",
            all_p.shape[1],
            edges.shape[1],
        )

        # We now need to find points that occur in multiple places
        p_unique, _, all_2_unique_p = setmembership.uniquify_point_set(
            all_p, tol=self.tol * np.sqrt(3)
        )

        # Update edges to work with unique points
        edges = all_2_unique_p[edges]

        # Look for edges that share both nodes. These will be identical, and
        # will form either an L/Y-type intersection (shared boundary segment),
        # or a three fractures meeting in a line.
        # Do a sort of edges before looking for duplicates.
        e_unique, unique_ind_e, all_2_unique_e = setmembership.unique_columns_tol(
            np.sort(edges, axis=0)
        )

        # Update the edges_2_frac map to refer to the new edges
        edges_2_frac_new = e_unique.shape[1] * [np.empty(0, dtype=int)]
        is_boundary_edge_new = e_unique.shape[1] * [np.empty(0, dtype=int)]

        for old_i, new_i in enumerate(all_2_unique_e):
            edges_2_frac_new[new_i], ind = np.unique(
                np.hstack((edges_2_frac_new[new_i], edges_2_frac[old_i])),
                return_index=True,
            )
            tmp = np.hstack((is_boundary_edge_new[new_i], is_boundary_edge[old_i]))
            is_boundary_edge_new[new_i] = tmp[ind]

        edges_2_frac = edges_2_frac_new
        is_boundary_edge = is_boundary_edge_new

        # Represent edges by unique values
        edges = e_unique

        # The uniquification of points may lead to edges with identical start
        # and endpoint. Find and remove these.
        point_edges = np.where(np.squeeze(np.diff(edges, axis=0)) == 0)[0]
        edges = np.delete(edges, point_edges, axis=1)
        # FIXME: Delete, right?
        unique_ind_e = np.delete(unique_ind_e, point_edges)
        for ri in point_edges[::-1]:
            del edges_2_frac[ri]
            del is_boundary_edge[ri]

        # Ensure that the edge to fracture map to a list of numpy arrays.
        # Use unique so that the same edge only refers to an edge once.
        edges_2_frac = [
            np.unique(np.array(edges_2_frac[i])) for i in range(len(edges_2_frac))
        ]

        # Sanity check, the fractures should still be defined by points in a
        # plane.
        if self.run_checks:
            self._verify_fractures_in_plane(p_unique, edges, edges_2_frac)

        logger.info(
            """Uniquify complete. %i points, %i edges. Elapsed time
                    %.5f""",
            p_unique.shape[1],
            edges.shape[1],
            time.time() - start_time,
        )

        return p_unique, edges, edges_2_frac, is_boundary_edge

    def _remove_edge_intersections(self, all_p, edges, edges_2_frac, is_boundary_edge):
        """
        Remove crossings from the set of fracture intersections.

        Intersecting intersections (yes) are split, and new points are
        introduced.

        Parameters:
            all_p (np.ndarray, 3xn): Coordinates of all points used to describe
                the fracture polygons, and their intersections. Should be
                unique.
            edges (np.ndarray, 2xn): Connections between points, formed either
                by a fracture boundary, or a fracture intersection.
            edges_2_frac (list of np.arrays): One list item per edge, each item is an
                np.array with indices of all fractures sharing this edge.
            is_boundary_edge (list of np.arrays): One list item per edge. Each item is
                an np.array with values 0 or 1, according to whether the edge is on the
                boundary of a fracture. Ordering of the inner list corresponds to the
                ordering in edges_2_frac.

        Returns:
            The same fields, but updated so that all edges are
            non-intersecting.

        """
        logger.info("Remove edge intersections")
        start_time = time.time()

        # The algorithm loops over all fractures, pulls out edges associated
        # with the fracture, project to the local 2D plane, and look for
        # intersections there (direct search in 3D may also work, but this was
        # a simple option). When intersections are found, the global lists of
        # points and edges are updated.
        for fi in range(len(self.fractures)):

            logger.debug("Remove intersections from fracture %i", fi)

            # Identify the edges associated with this fracture
            # It would have been more convenient to use an inverse
            # relationship frac_2_edge here, but that would have made the
            # update for new edges (towards the end of this loop) more
            # cumbersome.

            # frac_2_edge is a list of arrays of varying size; we need to find
            # occurrences of fi within those lists. Looping is slow, so we expand
            # to a standard list (hstack below), and use rldecode to make a mapping
            # from the expanded list back to the original nested list.
            frac_ind_expanded = pp.matrix_operations.rldecode(
                np.arange(len(edges_2_frac)), np.array([e.size for e in edges_2_frac])
            )
            edges_loc_ind = frac_ind_expanded[np.hstack(edges_2_frac) == fi]
            edges_loc = np.vstack((edges[:, edges_loc_ind], edges_loc_ind))

            p_ind_loc = np.unique(edges_loc[:2])
            p_loc = all_p[:, p_ind_loc]

            p_2d, edges_2d, p_loc_c, rot = self._points_2_plane(
                p_loc, edges_loc, p_ind_loc
            )

            # Add a tag to trace the edges during splitting
            edges_2d[2] = edges_loc[2]

            # Obtain new points and edges, so that no edges on this fracture
            # are intersecting.
            # It seems necessary to increase the tolerance here somewhat to
            # obtain a more robust algorithm. Not sure about how to do this
            # consistent.
            p_new, edges_new, tags = pp.intersections.split_intersecting_segments_2d(
                p_2d, edges_2d, tol=self.tol
            )
            # Then, patch things up by converting new points to 3D,

            # From the design of the functions in cg, we know that new points
            # are attached to the end of the array. A more robust alternative
            # is to find unique points on a combined array of p_loc and p_new.
            p_add = p_new[:, p_ind_loc.size :]
            num_p_add = p_add.shape[1]

            # Add third coordinate, and map back to 3D
            p_add = np.vstack((p_add, np.zeros(num_p_add)))

            # Inverse of rotation matrix is the transpose, add cloud center
            # correction
            p_add_3d = rot.transpose().dot(p_add) + p_loc_c

            # The new points will be added at the end of the global point array
            # (if not, we would need to renumber all global edges).
            # Global index of added points
            ind_p_add = all_p.shape[1] + np.arange(num_p_add)
            # Global index of local points (new and added)
            p_ind_exp = np.hstack((p_ind_loc, ind_p_add))

            # Add the new points towards the end of the list.
            all_p = np.hstack((all_p, p_add_3d))

            new_all_p, _, ia = setmembership.uniquify_point_set(all_p, self.tol)

            # Handle case where the new point is already represented in the
            # global list of points.
            if new_all_p.shape[1] < all_p.shape[1]:
                all_p = new_all_p
                p_ind_exp = ia[p_ind_exp]

            # The ordering of the global edge list bears no significance. We
            # therefore plan to delete all edges (new and old), and add new
            # ones.

            # First add new edges.
            # All local edges in terms of global point indices
            edges_new_glob = p_ind_exp[edges_new[:2]]
            edges = np.hstack((edges, edges_new_glob))

            # Global indices of the local edges
            edges_loc_ind = np.unique(edges_loc_ind)

            # Append fields for edge-fracture map and boundary tags
            for ei in range(edges_new.shape[1]):
                # Find the global edge index. For most edges, this will be
                # correctly identified by edges_new[2], which tracks the
                # original edges under splitting. However, in cases of
                # overlapping segments, in which case the index of the one edge
                # may completely override the index of the other (this is
                # caused by the implementation of split_intersection_segments_2d.
                # We therefore compare the new edge to the old ones (before
                # splitting). If found, use the old information; if not, use
                # index as tracked by splitting.
                is_old, old_loc_ind = setmembership.ismember_rows(
                    edges_new_glob[:, ei].reshape((-1, 1)), edges[:2, edges_loc_ind]
                )
                if is_old[0]:
                    # The edge was preserved, with the same indices for start and end
                    # during splitting. Information on edge-to-fracs, and on whether this
                    # is a boundary edge is identical to that of the edge before the
                    # splitting routine. The global index is that of the local edges
                    # that had the same nodes.
                    glob_ei = [edges_loc_ind[old_loc_ind[0]]]
                else:
                    # This edge is formed by splitting of old edges. To recover all,
                    # exploit information from the splitting of segments.
                    cols_mapped_to_glob_ei = tags[1] == ei
                    glob_ei = tags[0][cols_mapped_to_glob_ei]

                # Update edge_2_frac and boundary information.
                e2f = np.array([], dtype=int)
                ib = np.array([], dtype=bool)
                for gi in glob_ei:
                    e2f = np.hstack((e2f, edges_2_frac[gi]))
                    ib = np.hstack((ib, is_boundary_edge[gi]))

                # There may de duplicates in e2f (if the size of glob_ei is larger
                # than 1), but these are removed in the below call to uniquify points
                # and edges.
                edges_2_frac.append(e2f)
                is_boundary_edge.append(ib)

            # Finally, purge the old edges
            edges = np.delete(edges, edges_loc_ind, axis=1)

            # We cannot delete more than one list element at a time. Delete by
            # index in decreasing order, so that we do not disturb the index
            # map.
            edges_loc_ind.sort()
            for ei in edges_loc_ind[::-1]:
                del edges_2_frac[ei]
                del is_boundary_edge[ei]
            # And we are done with this fracture. On to the next one.

        logger.info(
            "Done with intersection removal. Elapsed time %.5f",
            time.time() - start_time,
        )
        if self.run_checks:
            self._verify_fractures_in_plane(all_p, edges, edges_2_frac)

        return self._uniquify_points_and_edges(
            all_p, edges, edges_2_frac, is_boundary_edge
        )

    def fractures_of_points(self, pts):
        """
        For a given point, find all fractures that refer to it, either as
        vertex or as internal.

        Returns:
            list of int: indices of fractures, one list item per point.

        """
        fracs_of_points = []
        pts = np.atleast_1d(np.asarray(pts))
        for i in pts:
            fracs_loc = []

            # First identify edges that refer to the point
            edge_ind = np.argwhere(
                np.any(self.decomposition["edges"][:2] == i, axis=0)
            ).ravel("F")
            edges_loc = self.decomposition["edges"][:, edge_ind]
            # Loop over all polygons. If their edges are found in edges_loc,
            # store the corresponding fracture index
            for poly_ind, poly in enumerate(self.decomposition["polygons"]):
                ismem, _ = setmembership.ismember_rows(edges_loc, poly)
                if any(ismem):
                    fracs_loc.append(self.decomposition["polygon_frac"][poly_ind])
            fracs_of_points.append(list(np.unique(fracs_loc)))
        return fracs_of_points

    def close_points(self, dist):
        """
        In the set of points used to describe the fractures (after
        decomposition), find pairs that are closer than a certain distance.
        Intended use: Debugging.

        Parameters:
            dist (double): Threshold distance, all points closer than this will
                be reported.

        Returns:
            List of tuples: Each tuple contain indices of a set of close
                points, and the distance between the points. The list is not
                symmetric; if (a, b) is a member, (b, a) will not be.

        """
        c_points = []

        pt = self.decomposition["points"]
        for pi in range(pt.shape[1]):
            d = pp.distances.point_pointset(pt[:, pi], pt[:, pi + 1 :])
            ind = np.argwhere(d < dist).ravel("F")
            for i in ind:
                # Indices of close points, with an offset to compensate for
                # slicing of the point cloud.
                c_points.append((pi, i + pi + 1, d[i]))

        return c_points

    def _verify_fractures_in_plane(self, p, edges, edges_2_frac):
        """
        Essentially a debugging method that verify that the given set of
        points, edges and edge connections indeed form planes.

        This has turned out to be a common symptom of trouble.

        """
        for fi, _ in enumerate(self.fractures):

            # Identify the edges associated with this fracture
            edges_loc_ind = []
            for ei, e in enumerate(edges_2_frac):
                if np.any(e == fi):
                    edges_loc_ind.append(ei)

            edges_loc = edges[:, edges_loc_ind]
            p_ind_loc = np.unique(edges_loc)
            p_loc = p[:, p_ind_loc]

            # Run through points_2_plane, to check the assertions
            self._points_2_plane(p_loc, edges_loc, p_ind_loc)

    def _points_2_plane(self, p_loc, edges_loc, p_ind_loc):
        """
        Convenience method for rotating a point cloud into its own 2d-plane.
        """
        # Center point cloud around the origin
        p_loc_c = np.mean(p_loc, axis=1).reshape((-1, 1))
        p_loc -= p_loc_c

        # Project the points onto the local plane defined by the fracture
        rot = pp.map_geometry.project_plane_matrix(p_loc, tol=self.tol)
        p_2d = rot.dot(p_loc)

        extent = p_2d.max(axis=1) - p_2d.min(axis=1)
        lateral_extent = np.maximum(np.max(extent[2]), 1)

        assert extent[2] < lateral_extent * self.tol * 30

        # Dump third coordinate
        p_2d = p_2d[:2]

        # The edges must also be redefined to account for the (implicit)
        # local numbering of points
        edges_2d = np.empty_like(edges_loc)
        for ei in range(edges_loc.shape[1]):
            edges_2d[0, ei] = np.argwhere(p_ind_loc == edges_loc[0, ei])
            edges_2d[1, ei] = np.argwhere(p_ind_loc == edges_loc[1, ei])

        assert edges_2d[:2].max() < p_loc.shape[1]

        return p_2d, edges_2d, p_loc_c, rot

    def __repr__(self) -> str:
        s = (
            f"Three-dimensional fracture network with "
            f"{str(len(self.fractures))} plane fractures.\n"
        )
        if self.domain is not None:
            s += f"The domain is a {(str(self.domain)).lower()}"

        return s

    def _reindex_fractures(self):
        for fi, f in enumerate(self.fractures):
            f.index = fi

    def bounding_box(self):
        """Obtain bounding box for fracture network.

        The box is defined by the external boundary, if imposed, or if not by
        the maximal extent of the fractures in each direction.

        Returns:
            dictionary with fields 'xmin', 'xmax', 'ymin', 'ymax', 'zmin',
                'zmax'
        """
        min_coord = np.ones(3) * float("inf")
        max_coord = -np.ones(3) * float("inf")

        for f in self.fractures:
            min_coord = np.minimum(np.min(f.pts, axis=1), min_coord)
            max_coord = np.maximum(np.max(f.pts, axis=1), max_coord)

        return {
            "xmin": min_coord[0],
            "xmax": max_coord[0],
            "ymin": min_coord[1],
            "ymax": max_coord[1],
            "zmin": min_coord[2],
            "zmax": max_coord[2],
        }

    def impose_external_boundary(
        self,
        domain: Optional[pp.Domain] = None,
        keep_box: bool = True,
        area_threshold: float = 1e-4,
        clear_gmsh: bool = True,
        finalize_gmsh: bool = True,
    ) -> np.ndarray:
        """
        Set an external boundary for the fracture set.

        There are two permissible data formats for the domain boundary:
            1) A 3D box, described by its minimum and maximum coordinates.
            2) A list of polygons (each specified as a np.array, 3xn), that together
                form a closed polyhedron.

        If no bounding box is provided, a box will be fitted outside the fracture
        network.

        The fractures will be truncated to lay within the bounding
        box; that is, Fracture.pts will be modified. The original coordinates of
        the fracture boundary can still be recovered from the attribute
        Fracture.orig_points.

        Fractures that are completely outside the bounding box will be deleted
        from the fracture set.

        Parameters
        ----------
        domain (pp.Domain): See above for description.
        keep_box (bool, optional): If True (default), the bounding surfaces will be
            added to the end of the fracture list, and tagged as boundary.
        area_threshold (float): Lower threshold for how much of a fracture's area
            should be within the bounding box for the fracture to be preserved.
            Defaults to 1e-4
        finalize_gmsh and
        clear_gmsh
            are needed in the context of potentially non-convex fractures. The values
            used here should be the same as in a call to self.mesh(); default values
            should be sufficient for all regular usage.

        Returns
        -------
        np.array: Mapping from old to new fractures, referring to the fractures in
            self._fractures before and after imposing the external boundary.
            The mapping does not account for the boundary fractures added to the
            end of the fracture array (if keep_box) is True.

        Raises
        ------
        ValueError
            If the FractureNetwork contains no fractures and no domain was passed
            to this method.

        """
        if domain is None and not self.fractures:
            # Cannot automatically calculate external boundary for non-fractured grids.
            raise ValueError(
                "A domain must be supplied to constrain non-fractured media."
            )
        self.bounding_box_imposed = True

        if domain is not None:
            self.domain = domain
        else:
            # Compute a bounding box from the extension of the fractures.
            overlap = 0.15
            cmin = np.ones((3, 1)) * float("inf")
            cmax = -np.ones((3, 1)) * float("inf")
            for f in self.fractures:
                cmin = np.min(np.hstack((cmin, f.pts)), axis=1).reshape((-1, 1))
                cmax = np.max(np.hstack((cmax, f.pts)), axis=1).reshape((-1, 1))
            cmin = cmin[:, 0]
            cmax = cmax[:, 0]

            dx = overlap * (cmax - cmin)

            # If the fractures has no extension along one of the coordinate
            # (a single fracture aligned with one axis), the domain should
            # still have an extension.
            hit = np.where(dx < self.tol)[0]
            dx[hit] = overlap * np.max(dx)

            box = {
                "xmin": cmin[0] - dx[0],
                "xmax": cmax[0] + dx[0],
                "ymin": cmin[1] - dx[1],
                "ymax": cmax[1] + dx[1],
                "zmin": cmin[2] - dx[2],
                "zmax": cmax[2] + dx[2],
            }
            self.domain = pp.Domain(box)

        # Constrain the fractures to lie within the bounding polyhedron
        polys = [f.pts for f in self.fractures]

        constrained_polys, inds = pp.constrain_geometry.polygons_by_polyhedron(
            polys, self.domain.polytope
        )
        # Delete fractures that are not member of any constrained fracture
        old_frac_ind = np.arange(len(self.fractures))
        delete_frac = np.setdiff1d(old_frac_ind, inds)
        # Identify fractures that have been split
        if inds.size > 0:
            split_frac = np.where(np.bincount(inds) > 1)[0]
        else:
            split_frac = np.zeros(0, dtype=int)

        # Split fractures should be deleted
        delete_frac = np.hstack((delete_frac, split_frac))
        ind_map = np.delete(old_frac_ind, delete_frac)

        # Update the fractures with the new data format
        for poly, ind in zip(constrained_polys, inds):
            if ind not in split_frac:
                self.fractures[ind].pts = poly

        # Special handling of fractures that are split in two
        for fi in split_frac:
            hit = np.where(inds == fi)[0]

            for sub_i in hit:
                # The fractures may very well be non-convex at this point, so
                # the points should not be sorted.
                # Splitting of non-convex fractures into convex subparts is
                # handled below.
                new_frac = PlaneFracture(constrained_polys[sub_i], sort_points=False)
                self.add(new_frac)
                ind_map = np.hstack((ind_map, fi))

        # Now, we may modify the fractures for two reasons:
        # 1) Only a tiny part of the fracture is contained within the region,
        # 2) The constrained polygon in non-convex, and must be split into smaller parts
        #    to be compatible with the intersection identification.

        # Some bookkeeping is needed to track which fractures to track.
        current_ind_map = 0
        delete_from_ind_map = np.array([], dtype=int)
        # Loop over all fractures
        for fi, f in enumerate(self.fractures):
            # No need to check for fractures that will be deleted anyhow.
            if fi in delete_frac:
                continue

            # Compute the area of the constrained fractures, relative to the original
            # size.
            # Delete small fractures.
            # IMPLEMENTATION NOTE: Should we have an absolute threshold in addition to
            # the relative tolerance below?

            # Map the fracture to its natrual plane.
            # If the fracture is very small, we risk running into trouble here with
            # geometry checks that essentially detects almost coinciding points
            # (although the error message comes from a normal vector computation).
            # Therefore, we use very strict tolerances, and cross our fingers
            # everything is fine.
            rot = pp.map_geometry.project_plane_matrix(
                f.pts, tol=1e-12, check_planar=False
            )
            center_coord = f.center
            mapped_coord = rot.dot(f.pts - center_coord)
            mapped_orig = rot.dot(f.orig_pts - center_coord)

            # Construct convex hulls, use these to construct the areas
            # The area for 2d ConvexHull (scipy style) is represented by the attribute
            # volume.
            hull_now = ConvexHull(mapped_coord[:2].T)
            hull_orig = ConvexHull(mapped_orig[:2].T)

            if hull_now.volume / hull_orig.volume < area_threshold:
                # If the part of the fracture inside the box is very small, add the
                # fracture to the list to be deleted, and remove it from the index
                # mapping between all and preserved fractures.
                delete_frac = np.hstack((delete_frac, fi))
                # Take note that this item should be deleted from the index map
                delete_from_ind_map = np.hstack((delete_from_ind_map, current_ind_map))
                current_ind_map += 1
                # No need to consider this fracture more.
                continue

            # Next, look for non-convex fracture.
            if f.pts.shape[1] == 3:
                # triangles are convex.
                # Increase the index pointing to ind_map
                current_ind_map += 1
                continue

            # Non-convex polygons are identified by comparing two triangulations,
            # one standard Delaunay triangulation with no constraints on adhering to
            # the polygon boundaries, and a second which is guaranteed not to cross
            # the polygon boundaries. If the two have equal area, the polygon is
            # convex, and we need not do anything more. If not, it should be
            # subdivided into smaller subpolygons.

            # Shorthand notation
            p = mapped_coord
            num_p = p.shape[1]

            # Use the ear clipping triangulation to create a 'safe' triangulation
            # See Wikipedia for a description of the algorithm.

            # Indices used the access vertexes of the polygon. We will check if
            # a diagonal can be drawn between i0 and i2, so that the triangle
            # (i0, i1, i2) is inside the polygon and contains no other polygon
            # vertexes.
            i0 = 0
            i1 = 1
            i2 = 2

            # List of nodes removed (isolated by an introduced diagonal)
            removed = []
            # List of triangles introduced
            tris: list[list[int]] = []

            # Helper function to get i2 move to the next available node
            def _next_i2(i2):
                start = i2
                while True:
                    i2 += 1
                    if i2 == num_p:
                        i2 = 0
                    if i2 not in removed:
                        return i2

                    # If we have made it a full circle, something is wrong.
                    if start == i2:
                        raise ValueError

            # Rolled version of the polygon, and of the indices.
            # Needed for segment intersection checks
            p_rolled = np.roll(p, -1, axis=1)
            # Indices had to be rolled the other way, it turned out
            ind_rolled = np.roll(np.arange(num_p), 1)

            while True:
                # vertexes of the prospective subtriangle
                loc_poly = p[:, [i0, i1, i2]]

                # A prospective diagonal (i0, i2) can be ruled out for two reasons:
                # 1) It crosses a segment of the polygon
                # 2) It is completely outside the polygon
                # 3) The prospective subtriangle fully contains another point
                #    (thus other segments). This can happen for a non-convex quad.

                # Check for all three. There must be better ways of doing this, with
                # more knowledge of computational geometry, but the code is what it is.

                # Distance from the prospective diagonal to all boundary segments
                # of the polygon.
                dist, *_ = pp.distances.segment_segment_set(
                    p[:, i0], p[:, i2], p, p_rolled
                )

                # Mask away all boundary segments which has i0 or i2 as one of its
                # endpoints.
                mask = np.ones(num_p, dtype=bool)
                mask[np.unique([i0, i2, ind_rolled[i0], ind_rolled[i2]])] = False

                # Find center of prospective triangle, check if inside the polygon.
                center = loc_poly.mean(axis=1)
                center_in_poly = pp.geometry_property_checks.point_in_polygon(
                    p, center
                )[0]

                other_points_in_loc_poly = pp.geometry_property_checks.point_in_polygon(
                    loc_poly, p
                )

                if (
                    not center_in_poly
                    or (mask.any() and dist[mask].min() < self.tol)
                    or other_points_in_loc_poly.any()
                ):
                    # We cannot use this triangle - move on:
                    # i0 and i1 are moved one step up, i2 is moved to the next
                    # available vertex
                    i0 = i1
                    i1 = i2
                    i2 = _next_i2(i2)
                else:
                    # Introduce the diagonal (i0, i2).
                    tris.append([i0, i1, i2])
                    removed.append(i1)
                    # Move indices i1 and i2. i0 is fixed.
                    i1 = i2
                    i2 = _next_i2(i2)

                if len(removed) == num_p - 2:
                    # By now, we should have considered all possible candidates.
                    break

            # Make a grid out of this triangulation for easy volume computation.
            g_constrained = pp.TriangleGrid(p, np.array(tris).T)
            g_constrained.compute_geometry()

            # Make a grid with a standard Delaunay triangulation.
            g_full = pp.TriangleGrid(p[:2])
            g_full.compute_geometry()

            # Compare volumes of the two grids. If they are almost equal, the
            # polygon should be convex.
            if np.allclose(
                g_constrained.cell_volumes.sum(), g_full.cell_volumes.sum(), rtol=1e-5
            ):
                # Done with this fracture
                # Increase the index pointing to ind_map
                current_ind_map += 1
                continue

            # The polygon is not convex, and should be split into convex subparts.
            # We could use the good triangulation to that purpose, however, that is
            # likely to give badly shaped surfaces, and thus cells. Instead, use
            # the Hertel-Mehlhorn algorithm to construct larger convex polygons
            # by eliminating edges from the triangulation.
            # First, take note that the original fracture should be deleted.
            delete_frac = np.hstack((delete_frac, fi))
            delete_from_ind_map = np.hstack((delete_from_ind_map, current_ind_map))

            # Subset of triangles that are inside
            triangles = g_constrained.cell_nodes().indices.reshape((-1, 3))

            # Form all edges of the triangles
            all_edges = np.vstack(
                [
                    triangles[:, :2],
                    triangles[:, 1:],
                    triangles[:, [2, 0]],
                ]
            ).T
            all_edges.sort(axis=0)
            edges, _, b = pp.utils.setmembership.unique_columns_tol(all_edges)

            # Edges on the boundary
            essential_edge = np.where(np.bincount(b) == 1)[0]
            # Free edges are candidates for removal
            free_edges = np.setdiff1d(
                np.arange(edges.shape[1]), essential_edge
            ).tolist()

            # To keep track of angles in the polygon, we also need pairs of edges
            # with a common vertex.
            # Find the pairs looping over all vertexes, find all its
            # neighboring vertexes.
            main_vertex_list = []
            other_vertex_list: list[int] = []
            indptr_list = [0]

            for i in range(num_p):
                row, _ = np.where(triangles == i)
                other = np.setdiff1d(triangles[row].ravel(), i)
                num_other = other.size
                indptr_list.append(indptr_list[-1] + num_other)
                main_vertex_list += num_other * [i]
                other_vertex_list += other.tolist()

            edge_pairs = np.zeros((3, 0), dtype=int)
            other_vertex = np.array(other_vertex_list)
            indptr = np.array(indptr_list)
            main_vertex = np.array(main_vertex_list)
            # again loop over all the vertexes, sort the neighboring vertexes
            for vi in range(num_p):
                start = indptr[vi]
                end = indptr[vi + 1]
                inds = other_vertex[start:end]
                vecs = p[:, inds] - p[:, vi].reshape((-1, 1))

                angle = np.arctan2(vecs[1], vecs[0])
                sort_ind = np.argsort(angle)
                other_vertex[start:end] = inds[sort_ind]

            # Shift the vertexes to find pairs of neighboring edges
            other_vertex_shifted = np.roll(other_vertex, -1)
            other_vertex_shifted[indptr[1:] - 1] = other_vertex[indptr[:-1]]

            edge_pairs = np.vstack([main_vertex, other_vertex, other_vertex_shifted])

            # Find angles between all edge pairs.
            vec_1 = p[:2, edge_pairs[1]] - p[:2, edge_pairs[0]]
            vec_2 = p[:2, edge_pairs[2]] - p[:2, edge_pairs[0]]
            len_vec_1 = np.sqrt(np.sum(np.power(vec_1, 2), axis=0))
            len_vec_2 = np.sqrt(np.sum(np.power(vec_2, 2), axis=0))
            # The use of arccos will give an angle in (0, pi), that is, the
            # inner angle between the edge pairs (we know the pair is part of
            # a triangle).
            pair_angles = np.arccos(
                np.sum(vec_1 * vec_2, axis=0) / (len_vec_1 * len_vec_2)
            )

            # Helper function to combine edges (prepare for merge of the neighboring
            # edge pairs in both ends of the edge). Also compute angle in the
            # candidate new pair.
            def new_pairs_and_angles(ei):
                old_pairs = []
                new_pairs = []
                new_angles = []
                for e in edges[:, ei]:
                    loc_pair_ind = np.where(edge_pairs[0] == e)[0]
                    loc_pairs = edge_pairs[:, loc_pair_ind]

                    other = np.setdiff1d(edges[:, ei], e)
                    hit = np.logical_or(loc_pairs[1] == other, loc_pairs[2] == other)
                    new_pair = np.hstack(
                        [e, np.setdiff1d(loc_pairs[1:, hit].ravel(), other)]
                    )
                    new_pairs.append(new_pair)
                    old_pairs.append(loc_pair_ind[hit])
                    new_angles.append(np.sum(pair_angles[loc_pair_ind[hit]]))

                new_pairs = np.array([n for n in new_pairs]).T
                return old_pairs, new_pairs, np.array(new_angles)

            removed_edges = []
            # Loop over all free edges, try to merge them. The loop order will
            # impact the quality, but attempts to do a sophisticated ordering
            # failed spectacularly, so we loop by edge ordering.
            while len(free_edges) > 0:

                ind = free_edges.pop(0)
                # Get information about the perspective new edge pairs
                old_pair_ind, new_pairs, new_angles = new_pairs_and_angles(ind)
                # if the resulting pairs both are consistent with a convex
                # polygon, do the merge.
                if np.all(new_angles < np.pi):
                    edge_pairs = np.delete(edge_pairs, old_pair_ind, axis=1)
                    pair_angles = np.delete(pair_angles, old_pair_ind)
                    edge_pairs = np.hstack((edge_pairs, new_pairs))
                    pair_angles = np.hstack((pair_angles, new_angles))
                    # Take note that this edge should be removed from the triangulation.
                    removed_edges.append(ind)

            # Identify which triangles to merge by forming a graph of the edges to
            # remove, and then loop over connected subgraphs.
            graph = nx.Graph()

            # First add individual triangles to the graph
            for ti in range(triangles.shape[0]):
                graph.add_node(ti)
            # Next connections between the triangles corresponding to removed edges.
            for e in removed_edges:
                tri_ind = np.where(
                    np.sum(
                        np.logical_or(
                            triangles == edges[0, e], triangles == edges[1, e]
                        ).astype(int),
                        axis=1,
                    )
                    == 2
                )[0]
                graph.add_edge(tri_ind[0], tri_ind[1])

            for component in nx.connected_components(graph):
                # Extract subgraph of this cluster
                sg = graph.subgraph(component)
                tris = list(sg.nodes)
                # The node indices are given in a cyclic ordering (CW or CCW),
                # thus a linear ordering should be fine also for a subpolygon.
                verts = np.unique(triangles[tris])
                # To be sure, check the convexity of the polygon.
                self.add(PlaneFracture(f.pts[:, verts], check_convexity=False))
                ind_map = np.hstack((ind_map, fi))

            # Finally, increase pointer to ind_map array
            current_ind_map += 1

        # Delete fractures that have all points outside the bounding box
        # There may be some uncovered cases here, with a fracture barely
        # touching the box from the outside, but we leave that for now.
        for i in np.unique(delete_frac)[::-1]:
            del self.fractures[i]

        ind_map = np.delete(ind_map, delete_from_ind_map)

        # Final sanity check: All fractures should have at least three
        # points at the end of the manipulations
        for f in self.fractures:
            assert f.pts.shape[1] >= 3

        boundary_tags = self.tags.get("boundary", [False] * len(self.fractures))
        if keep_box:
            for pnt in self.domain.polytope:
                self.add(PlaneFracture(pnt))
                boundary_tags.append(True)
        self.tags["boundary"] = boundary_tags

        self._reindex_fractures()
        return ind_map

    def _classify_edges(self, polygon_edges, constraints):
        """
        Classify the edges into fracture boundary, intersection, or auxiliary.
        Also identify points on intersections between intersections (fractures
        of co-dimension 3)

        Parameters:
            polygon_edges (list of lists): For each polygon the global edge
                indices that forms the polygon boundary.
            constraints (list or np.array): Which polygons are constraints, and should
                not cause the generation of 1d grids.

        Returns:
            tag: Tag of the edges, using the values in GmshConstants. Note
                that auxiliary points will not be tagged (these are also
                ignored in gmsh_interface.GmshWriter).
            np.array, boolean: True for edges which are internal to at least one
                polygon.
            np.array, boolean: True for edges that are internal to at least one polygon,
                and on the boundary of another polygon.

        """
        edges = self.decomposition["edges"]
        is_bound = self.decomposition["is_bound"]
        num_edges = edges.shape[1]

        poly_2_frac = np.arange(len(self.decomposition["polygons"]))
        self.decomposition["polygon_frac"] = poly_2_frac

        # Construct a map from edges to polygons
        edge_2_poly = [[] for i in range(num_edges)]
        for pi, poly in enumerate(polygon_edges[0]):
            for ei in np.unique(poly):
                edge_2_poly[ei].append(poly_2_frac[pi])

        # Count the number of referrals to the edge from different polygons
        # Only do this if the polygon is not a constraint
        num_referrals = np.zeros(num_edges)
        for ei, ep in enumerate(edge_2_poly):
            ep_not_constraint = np.setdiff1d(ep, constraints)
            num_referrals[ei] = np.unique(ep_not_constraint).size

        # A 1-d grid will be inserted where there is more than one fracture
        # referring.
        has_1d_grid = np.where(num_referrals > 1)[0]

        num_is_bound = len(is_bound)
        tag = np.zeros(num_edges, dtype="int")
        # Find edges that are tagged as a boundary of all its fractures
        all_bound = [np.all(is_bound[i]) for i in range(num_is_bound)]

        # We also need to find edges that are on the boundary of some, but not all
        # the fractures
        any_bound = [np.any(is_bound[i]) for i in range(num_is_bound)]
        some_bound = np.logical_and(any_bound, np.logical_not(all_bound))

        bound_ind = np.where(all_bound)[0]
        # Remove boundary  that are referred to by more than fracture - this takes
        # care of L-type intersections
        bound_ind = np.setdiff1d(bound_ind, has_1d_grid)

        # Index of lines that should have a 1-d grid. This are all of the first
        # num-constraints, minus those on the boundary.
        # Note that edges with index > num_is_bound are known to be of the
        # auxiliary type. These will have tag zero; and treated in a special
        # manner by the interface to gmsh.
        not_boundary_ind = np.setdiff1d(np.arange(num_is_bound), bound_ind)
        tag[bound_ind] = GmshInterfaceTags.FRACTURE_TIP.value

        tag[not_boundary_ind] = GmshInterfaceTags.FRACTURE_INTERSECTION_LINE.value

        return tag, np.logical_not(all_bound), some_bound

    def _on_domain_boundary(self, edges, edge_tags):
        """
        Finds edges and points on boundary, to avoid that these
        are gridded.

        Returns:
            np.array: For all points in the decomposition, the value is 0 if the point
                is in the interior, constants.FRACTURE_TAG if the point is on a fracture
                that extends to the boundary, and constants.DOMAIN_BOUNDARY_TAG if the
                point is part of the boundary specification.
            np.array: For all edges in the decomposition, tags identifying the edge
                as on a fracture or boundary.

        """
        # Obtain current tags on fractures
        boundary_polygons = np.where(
            self.tags.get("boundary", [False] * len(self.fractures))
        )[0]

        # ... on the points...
        point_tags = GmshInterfaceTags.NEUTRAL.value * np.ones(
            self.decomposition["points"].shape[1], dtype=int
        )
        # and the mapping between fractures and edges.
        edges_2_frac = self.decomposition["edges_2_frac"]

        # Loop over edges, and the polygons to which the edge belongs
        for e, e2f in enumerate(edges_2_frac):

            # Check if the polygons are on the boundary
            edge_of_domain_boundary = np.in1d(e2f, boundary_polygons)

            if any(edge_of_domain_boundary):
                # If all associated polygons are boundary, this is simple
                if all(edge_of_domain_boundary):
                    # The point is not associated with a fracture extending to the
                    # boundary
                    edge_tags[e] = GmshInterfaceTags.DOMAIN_BOUNDARY_LINE.value
                    # The points of this edge are also associated with the boundary
                    point_tags[
                        edges[:, e]
                    ] = GmshInterfaceTags.DOMAIN_BOUNDARY_POINT.value
                else:
                    # The edge is associated with at least one fracture. Still, if it is
                    # also the edge of at least one boundary point, we will consider it
                    # a domain boundary edge
                    on_one_domain_edge = False
                    for pi in np.where(edge_of_domain_boundary)[0]:
                        if e not in self.decomposition["line_in_frac"][e2f[pi]]:
                            on_one_domain_edge = True
                            break

                    # The line is on the boundary
                    if on_one_domain_edge:
                        edge_tags[e] = GmshInterfaceTags.DOMAIN_BOUNDARY_LINE.value
                        point_tags[
                            edges[:, e]
                        ] = GmshInterfaceTags.DOMAIN_BOUNDARY_POINT.value
                    else:
                        # The edge is an intersection between a fracture and a boundary
                        # polygon
                        edge_tags[e] = GmshInterfaceTags.FRACTURE_BOUNDARY_LINE.value
                        point_tags[
                            edges[:, e]
                        ] = GmshInterfaceTags.FRACTURE_BOUNDARY_POINT.value
            else:
                # This is not an edge on the domain boundary, and the tag assigned in
                # in self._classify_edges() is still valid: It is either a fracture tip
                # or a fracture intersection line
                # The point tag is still neutral; it may be adjusted later depending on
                # how many intersection lines refer to the point
                continue

        return point_tags, edge_tags

    def _poly_2_segment(self):
        """
        Represent the polygons by the global edges, and determine if the lines
        must be reversed (locally) for the polygon to form a closed loop.

        Returns:
            list of np.array: Each list item gives the indices of the edges need to
                form a polygon (fracture, boundary) in the decomposition.
            list of np.array: For each polygon, and each edge of the polygon, True if
                the edge must be reversed (relative to its description in
                self.decomposition['edges']).

        """
        edges = self.decomposition["edges"]
        poly = self.decomposition["polygons"]

        poly_2_line = []
        line_reverse = []
        for p in poly:
            hit, ind = setmembership.ismember_rows(p, edges[:2], sort=False)
            hit_reverse, ind_reverse = setmembership.ismember_rows(
                p[::-1], edges[:2], sort=False
            )
            assert np.all(hit + hit_reverse == 1)

            line_ind = np.zeros(p.shape[1])

            hit_ind = np.where(hit)[0]
            hit_reverse_ind = np.where(hit_reverse)[0]
            line_ind[hit_ind] = ind
            line_ind[hit_reverse_ind] = ind_reverse

            poly_2_line.append(line_ind.astype("int"))
            line_reverse.append(hit_reverse)

        return poly_2_line, line_reverse

    def _determine_mesh_size(self, point_tags, **kwargs):
        """
        Set the preferred mesh size for geometrical points as specified by
        gmsh.

        Currently, the only option supported is to specify a single value for
        all fracture points, and one value for the boundary.

        See the gmsh manual for further details.

        """

        num_pts = self.decomposition["points"].shape[1]

        if self.mesh_size_min is None or self.mesh_size_frac is None:
            logger.info("Found no information on mesh sizes. Returning")
            return None

        p = self.decomposition["points"]
        num_pts = p.shape[1]
        # Compute distance between the points forming the network
        # This will also account for points close to segments on other fractures, as
        # an auxiliary point will already have been inserted there
        dist = pp.distances.pointset(p, max_diag=True)
        point_dist = np.min(dist, axis=1)
        logger.info(
            "Minimal distance between points encountered is " + str(np.min(dist))
        )
        # The mesh size should not be smaller than the prescribed minimum distance
        mesh_size_min = np.maximum(point_dist, self.mesh_size_min * np.ones(num_pts))
        # The mesh size should not be larger than the prescribed fracture mesh size.
        # If the boundary mesh size is also presribed, that will be enforced below.
        mesh_size = np.minimum(mesh_size_min, self.mesh_size_frac * np.ones(num_pts))

        if self.mesh_size_bound is not None:
            on_boundary = point_tags == GmshInterfaceTags.DOMAIN_BOUNDARY_POINT.value
            mesh_size_bound = np.minimum(
                mesh_size_min, self.mesh_size_bound * np.ones(num_pts)
            )
            mesh_size[on_boundary] = np.maximum(mesh_size, mesh_size_bound)[on_boundary]
        return mesh_size

    def _insert_auxiliary_points(
        self, mesh_size_frac=None, mesh_size_min=None, mesh_size_bound=None
    ):
        """Insert auxiliary points on fracture edges. Used to guide gmsh mesh
        size parameters.

        The function should only be called once to avoid insertion of multiple
        layers of extra points, this will likely kill gmsh.

        The function is motivated by similar functionality for 2d domains, but
        is considerably less mature.

        The function should be used in conjunction with _determine_mesh_size().
        The ultimate goal is to set the mesh size
        for geometrical points in Gmsh. To that end, this function inserts
        additional points on the fracture boundaries. The mesh size is then
        determined as the distance between all points in the fracture
        description.

        Parameters:
            mesh_size_frac: Ideal mesh size. Will be added to all points that
                are sufficiently far away from other points.
            mesh_size_min: Minimal mesh size; we will make no attempts to
                enforce even smaller mesh sizes upon Gmsh.
            mesh_size_bound (optional): Boundary mesh size. Will be added to the points

                defining the boundary, unless there are any fractures in the
                immediate vicinity influencing the size. In other words,
                mesh_size_bound is the boundary point equivalent of
                mesh_size_frac.
        """

        if self.auxiliary_points_added:
            logger.info("Auxiliary points already added. Returning.")
        else:
            self.auxiliary_points_added = True

        self.mesh_size_frac = mesh_size_frac
        self.mesh_size_min = mesh_size_min
        self.mesh_size_bound = mesh_size_bound

        # Auxiliary points may be added for two reasons: For fractures that do intersect
        # the points are added on the segments on the fracture boundaries, close to the
        # endpoints of the intersection segments. This is tailored to the way gmsh
        # creates grids, with meshing of 1d lines before 2d surfaces.
        # For non-intersecting fractures, auxiliary points may be added on the fracture
        # segments if they are sufficiently close.

        # FIXME: If a fracture is close to the interior of another fracture, this
        # will not be seen by the current algorithm. The solution should be to add
        # points to the fracture surface, but make sure that this is not decleared
        # physical. The operation should only be done for fracture pairs where no
        # auxiliary points were added because of close segments.

        def dist_p(a, b):
            # Helper function to get the distance from a set of points (a) to a single point
            # (b).
            if a.size == 3:
                a = a.reshape((-1, 1))
            b = b.reshape((-1, 1))
            return np.sqrt(np.sum(np.power(b - a, 2), axis=0))

        # Dictionary that for each fracture maps the index of all other fractures.
        intersecting_fracs: dict[list[int]] = {}
        # Loop over all fractures
        for fi, f in enumerate(self.fractures):

            # Do not insert auxiliary points on domain boundaries
            if "boundary" in self.tags.keys() and self.tags["boundary"][fi]:
                continue

            # First compare segments with intersections to this fracture
            nfp = f.pts.shape[1]

            # Keep track of which other fractures are intersecting - will be
            # needed later on
            intersections_this_fracture: list[int] = []

            # Loop over all intersections of the fracture
            isects, is_first_isect = self.intersections_of_fracture(f)

            for (i, is_first) in zip(isects, is_first_isect):

                # Register this intersection
                if is_first:
                    intersections_this_fracture.append(
                        self.intersections["second"][i].index
                    )
                else:
                    intersections_this_fracture.append(
                        self.intersections["first"][i].index
                    )

                # Find the point on the fracture segments that is closest to the
                # intersection point.
                # Note the use of vstack, combined with a transpose.
                isect_coord = np.vstack(
                    (self.intersections["start"][:, i], self.intersections["end"][:, i])
                ).T

                dist, cp = pp.distances.points_segments(
                    isect_coord, f.pts, np.roll(f.pts, -1, axis=1)
                )
                # Insert a (candidate) point only at the segment closest to the
                # intersection point. If the intersection line runs parallel
                # with a segment, this may be insufficient, but we will deal
                # with this if it turns out to be a problem.
                closest_segment = np.argmin(dist, axis=1)
                min_dist = dist[np.arange(2), closest_segment]

                # Sort the points on increasing point indices. This makes it easier to
                # keep track of where any new point should be added (variable
                # inserted_points below).
                sort_ind = np.argsort(closest_segment)
                closest_segment = closest_segment[sort_ind]
                min_dist = min_dist[sort_ind]

                inserted_points = 0

                for pi, (si, di) in enumerate(zip(closest_segment, min_dist)):
                    if di < mesh_size_frac and di > mesh_size_min:
                        # Distance between the closest point of the intersection segment
                        # and the points of this fracture.
                        d_1 = dist_p(cp[pi, si], f.pts[:, si])
                        d_2 = dist_p(cp[pi, si], f.pts[:, (si + 1) % nfp])
                        # If the intersection point is not very close to any of
                        # the points on the segment, we split the segment.
                        # NB: It is critical that this is done before a call to
                        # self.split_intersections(), or else the export to gmsh
                        # will go wrong.
                        if d_1 > mesh_size_min and d_2 > mesh_size_min:
                            f.pts = np.insert(
                                f.pts,
                                (si + 1 + inserted_points) % nfp,
                                cp[pi, si],
                                axis=1,
                            )

                            inserted_points += 1
                            nfp += 1
                            # Check if some of the intersections of the fracture should
                            # be split when the new point is added.
                            self._split_intersections_of_fracture(cp[pi, si])

            # Take note of the intersecting fractures
            intersecting_fracs[fi] = intersections_this_fracture

        # Next, insert points along segments that are close to other fractures,
        # which are not touching. The insertion is not symmetric, so that when
        # a point is inserted on one fracture, it is not inserted on the close
        # fracture which triggered the insertion (the corresponding point on the
        # second fracture may be inserted on later).

        # Precompute rolling of fracture points - this saves a bit of time.
        rolled_fracture_points = {
            f.index: np.roll(f.pts, -1, axis=1) for f in self.fractures
        }

        for fi, f in enumerate(self.fractures):

            # Do not insert auxiliary points on domain boundaries
            if "boundary" in self.tags.keys() and self.tags["boundary"][fi]:
                continue

            # Get the segments of all other fractures which do no intersect with
            # the main one.
            # First the indices
            other_fractures = []
            for of in self.fractures:
                if not (of.index in intersecting_fracs[fi] or of.index == f.index):
                    other_fractures.append(of)

            # If no other fractures were found, we go on.
            if len(other_fractures) == 0:
                continue

            # Next, arrays of start and end points. This allows us to use the
            # vectorized segment-segment distance computation.
            start_all = np.hstack([of.pts for of in other_fractures])
            end_all = np.hstack(
                [rolled_fracture_points[of.index] for of in other_fractures]
            )

            # Now, loop over all segments in the main fracture, look for close
            # segments on other fractures, and insert points along the main
            # segment if some are close enough. Points may be inserted if the
            # other fracture is closer than mesh_size_frac, however, the
            # auxiliary points should not be too dense along the main segment.
            # Specifically, points are only inserted if the distance form existing
            # points on the segment (endpoint or auxiliary) is larger than the
            # minimum distance. This implies that if several other fractures
            # have nearby closest points along the segment, only some of the
            # possible auxiliary points wil be inserted, however, the mesh
            # size in that part of the segment should still be fairly small.

            # Since the number of points in the fracture description changes by
            # insertion of auxiliary points, we use a counter and while loop
            # to loop over the segments.
            start_index = 0

            while start_index < f.pts.shape[1]:
                # Start and end of this segment. These should be vertexes in the
                # original fracture (prior to insertion of auxiliary points)
                seg_start = f.pts[:, start_index].reshape((-1, 1))
                # Modulus to finish the final segment of the fracture.
                seg_end = f.pts[:, (start_index + 1) % f.pts.shape[1]].reshape((-1, 1))

                # Find the distance from this segment to all other segments
                # (not intersecting).
                dist, cp_f, _ = pp.distances.segment_segment_set(
                    seg_start, seg_end, start_all, end_all
                )

                # Sort the points according to distances, so that we try to insert
                # the most needed auxiliary points first
                sort_ind = np.argsort(dist)
                dist = dist[sort_ind]
                cp_f = cp_f[:, sort_ind]

                # Chances are, many of the other segments will be closest to an end
                # of this segment. These can be disregarded (below).
                dist_segment_ends = np.minimum(
                    dist_p(cp_f, seg_start), dist_p(cp_f, seg_end)
                )
                not_closest_on_ends = dist_segment_ends > mesh_size_min

                # Also cut all points corresponding to segments that are further
                # away than the prescribed fracture mesh size
                not_far_away = dist < mesh_size_frac

                is_candidate = np.logical_and(not_closest_on_ends, not_far_away)

                if not np.any(is_candidate):
                    # If no candidate point was found for this edge, we can move on:
                    # Increase the start index to hit the next segment along the main
                    # fracture.
                    start_index += 1
                else:
                    # First pick out all condidate auxiliary points, then reduce
                    # to a unique subset (close candidate points may occur e.g.
                    # for fractures close to complex domain boundaries.)
                    candidates = cp_f[:, is_candidate]
                    (
                        unique_candidates,
                        _,
                        o2n,
                    ) = pp.utils.setmembership.uniquify_point_set(candidates, self.tol)

                    # Make arrays for points along the segment (originally the
                    # endpoints), and for the auxiliary points
                    segment_points = np.hstack((seg_start, seg_end))
                    new_points = np.zeros((3, 0))

                    # Loop over all candidate points (due to the sorting on
                    # distances, the candidates corresponds to increasing distance
                    # from the nearby fracture).
                    for point_counter in range(unique_candidates.shape[1]):
                        pt = unique_candidates[:, point_counter].reshape((-1, 1))

                        # Insert the candidate point if it is sufficiently far
                        # away from existing points along the line.
                        if np.min(dist_p(segment_points, pt)) > mesh_size_min:
                            # No sorting of points along the line, take care
                            # of this below.
                            segment_points = np.hstack((segment_points, pt))
                            # Register new point.
                            new_points = np.hstack((new_points, pt))

                    # Compute distance between new points and segment start
                    dist_from_start = dist_p(new_points, seg_start)

                    # Sort the new points according to distance. This will make
                    # the new sub-segments a line along the main segment. Then insert.
                    sorted_new = new_points[:, np.argsort(dist_from_start)]

                    f.pts = np.insert(f.pts, [start_index + 1], sorted_new, axis=1)
                    # Update precomputed dict of rolled fracture points
                    rolled_fracture_points[f.index] = np.roll(f.pts, -1, axis=-1)

                    # If the new points sit on top of intersections, these must be split
                    for pi in range(sorted_new.shape[1]):
                        self._split_intersections_of_fracture(
                            sorted_new[:, pi].reshape((-1, 1))
                        )

                    # Move the start index to the next segment, compensating for
                    # inserted auxiliary points.
                    start_index += 1 + sorted_new.shape[1]

    def _split_intersections_of_fracture(self, cp: np.ndarray) -> None:
        """Check if a point lies on intersections of a given fracture, and if so,
        split the intersection into two.

        The intended usage is when auxiliary points are added to a fracture's vertexes
        after the intersections have been identified.

        Parameters:
            cp (np.array, size 3): Coordinate of the added point. The intersection will
                be split if the point lies on the intersection.

        """
        # Intersections must be split for all fractures, not only those where the
        # fracture is involved, or else Y-type intersections may not be treated
        # correctly.

        if cp.size < 4:
            cp = cp.reshape((-1, 1))

        # Compute distance from the point to all other points
        isect_start = self.intersections["start"]
        isect_end = self.intersections["end"]
        d_isect, _ = pp.distances.points_segments(cp, isect_start, isect_end)

        # We are only interested in segments that are  close to the point
        close_to_segment = d_isect[0] < self.tol
        # If there are no close segments, there is nothing more to do.
        if not np.any(close_to_segment):
            return

        # If the close point is the endpoint of the intersection segment, we
        # should not split it
        dist_start = pp.distances.point_pointset(cp, isect_start)
        dist_end = pp.distances.point_pointset(cp, isect_end)
        close_to_endpoint = np.logical_or(dist_start < self.tol, dist_end < self.tol)

        # replace intersections that are cut in two pieces, both with non-zero length.
        to_replace = np.where(
            np.logical_and(close_to_segment, np.logical_not(close_to_endpoint))
        )[0]

        if to_replace.size == 0:
            return

        # Duplicate intersections, and split into two.
        first = self.intersections["first"][to_replace]
        second = self.intersections["second"][to_replace]

        bound_first = self.intersections["bound_first"][to_replace]
        bound_second = self.intersections["bound_second"][to_replace]

        start = self.intersections["start"][:, to_replace]
        end = self.intersections["end"][:, to_replace]
        if start.size < 4:
            start = start.reshape((-1, 1))
            end = end.reshape((-1, 1))

        new_first = np.tile(first, 2)
        new_second = np.tile(second, 2)
        new_bound_first = np.tile(bound_first, 2)
        new_bound_second = np.tile(bound_second, 2)

        new_start = np.hstack((start, end))
        new_end = np.tile(cp, 2 * to_replace.size)

        self._add_intersection(
            new_first, new_second, new_start, new_end, new_bound_first, new_bound_second
        )
        # Delete the old intersections.
        for key in self.intersections:
            self.intersections[key] = np.delete(
                self.intersections[key], to_replace, axis=-1
            )

    def _add_intersection(
        self,
        first: Union[PlaneFracture, np.ndarray],
        second: Union[PlaneFracture, np.ndarray],
        start: np.ndarray,
        end: np.ndarray,
        bound_first: Union[bool, np.ndarray],
        bound_second: Union[bool, np.ndarray],
    ) -> None:
        # Add one or several fracture pairs to the intersections.
        # If several fractures, these should be wrapped in a numpy array.
        self.intersections["first"] = np.hstack(
            (self.intersections["first"], first)  # type: ignore
        )
        self.intersections["second"] = np.hstack(
            (self.intersections["second"], second)  # type: ignore
        )

        if start.size < 4:
            start = start.reshape((-1, 1))
            end = end.reshape((-1, 1))

        self.intersections["start"] = np.hstack((self.intersections["start"], start))
        self.intersections["end"] = np.hstack((self.intersections["end"], end))
        self.intersections["bound_first"] = np.hstack(
            (self.intersections["bound_first"], bound_first)
        )
        self.intersections["bound_second"] = np.hstack(
            (self.intersections["bound_second"], bound_second)
        )

    def to_file(
        self,
        file_name: str,
        data: Optional[dict[str, Union[np.ndarray, list]]] = None,
        **kwargs,
    ) -> None:
        """
        Export the fracture network to file.

        The file format is given as an kwarg, by default vtu will be used. The writing
        is outsourced to meshio, thus the file format should be supported by that
        package.

        The fractures are treated as polygonal cells, with no special treatment
        of intersections.

        Fracture numbers are always exported (1-offset). In addition, it is
        possible to export additional data, as specified by the
        keyword-argument data.

        Parameters:
            file_name (str): Name of the target file.
            data (dictionary, optional): Data associated with the fractures.
                The values in the dictionary should be numpy arrays. 1d and 3d
                data is supported. Fracture numbers are always exported.

        Optional arguments in kwargs:
            binary (boolean): Use binary export format. Default to
                True.
            fracture_offset (int): Use to define the offset for a
                fracture id. Default to 1.
            folder_name (string): Path to save the file. Default to "./".
            extension (string): File extension. Default to ".vtu".

        See also the functions self.to_fab() and self.to_csv().

        """
        if data is None:
            data = {}

        binary: bool = kwargs.pop("binary", True)
        fracture_offset: int = kwargs.pop("fracture_offset", 1)
        extension: str = kwargs.pop("extension", ".vtu")
        folder_name: str = kwargs.pop("folder_name", "")

        if kwargs:
            msg = "Got unexpected keyword argument '{}'"
            raise TypeError(msg.format(kwargs.popitem()[0]))

        if not file_name.endswith(extension):
            file_name += extension

        # fracture points
        meshio_pts = np.empty((0, 3))

        # counter for the points
        pts_pos = 0

        # Data structure for cells in meshio format.
        meshio_cells = []
        # we operate fracture by fracture
        for fid, frac in enumerate(self.fractures):
            # In old meshio, polygonal cells are distinguished based on the
            # number of vertexes.
            # save the points of the fracture
            meshio_pts = np.vstack((meshio_pts, frac.pts.T))
            num_pts = frac.pts.shape[1]
            # Always represent the fracture as a polygon
            cell_type = "polygon"
            nodes = pts_pos + np.arange(num_pts)
            pts_pos += num_pts

            # In newer versions of meshio, make a list of 2-tuples with
            # (cell_type, list(node_numbers))
            meshio_cells.append((cell_type, [nodes.tolist()]))

        # The older version of meshio requires more post processing to work.
        # Cell data also requires different formats.

        data.update(
            {
                "fracture_number": [
                    [fracture_offset + i] for i in range(len(self.fractures))
                ]
            }
        )
        # The data is simply the data
        meshio_data = data

        meshio_grid_to_export = meshio.Mesh(
            meshio_pts, meshio_cells, cell_data=meshio_data
        )
        meshio.write(folder_name + file_name, meshio_grid_to_export, binary=binary)

    def to_csv(self, file_name, domain=None):
        """
        Save the 3d network on a csv file with comma , as separator.
        Note: the file is overwritten if present.
        The format is
        - if domain is given the first line describes the domain as a rectangle
          X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX
        - the other lines describe the N fractures as a list of points
          P0_X, P0_Y, P0_Z, ...,PN_X, PN_Y, PN_Z

        Parameters:
            file_name: name of the file
            domain: (optional) the bounding box of the problem
        """

        with open(file_name, "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")

            # if the domain (as bounding box) is defined save it
            if domain is not None:
                order = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
                csv_writer.writerow([domain.bounding_box[o] for o in order])

            # write all the fractures
            for f in self.fractures:
                csv_writer.writerow(f.pts.ravel(order="F"))

    def to_fab(self, file_name):
        """
        Save the 3d network on a fab file, as specified by FracMan.

        The filter is based on the .fab-files needed at the time of writing, and
        may not cover all options available.

        Parameters:
            file_name (str): File name.
        """
        # function to write a numpy matrix as string

        def to_file(p):
            return "\n\t\t".join(" ".join(map(str, x)) for x in p)

        with open(file_name, "w") as f:
            # write the first part of the file, some information are fake
            num_frac = len(self.fractures)
            num_nodes = np.sum([frac.pts.shape[1] for frac in self.fractures])
            f.write(
                "BEGIN FORMAT\n\tFormat = Ascii\n\tXAxis = East\n"
                + "\tScale = 100.0\n\tNo_Fractures = "
                + str(num_frac)
                + "\n"
                + "\tNo_TessFractures = 0\n\tNo_Nodes = "
                + str(num_nodes)
                + "\n"
                + "\tNo_Properties = 0\nEND FORMAT\n\n"
            )

            # start to write the fractures
            f.write("BEGIN FRACTURE\n")
            for frac_pos, frac in enumerate(self.fractures):
                f.write(
                    "\t" + str(frac_pos) + " " + str(frac.pts.shape[1]) + " 1\n\t\t"
                )
                p = np.concatenate(
                    (np.atleast_2d(np.arange(frac.pts.shape[1])), frac.pts), axis=0
                ).T
                f.write(to_file(p) + "\n")
                f.write("\t0 -1 -1 -1\n")
            f.write("END FRACTURE")

    def fracture_to_plane(self, frac_num):
        """Project fracture vertexes and intersection points to the natural
        plane of the fracture.

        Parameters:
            frac_num (int): Index of fracture.

        Returns:
            np.ndarray (2xn_pt): 2d coordinates of the fracture vertexes.
            np.ndarray (2xn_isect): 2d coordinates of fracture intersection
                points.
            np.ndarray: Index of intersecting fractures.
            np.ndarray, 3x3: Rotation matrix into the natural plane.
            np.ndarray, 3x1. 3d coordinates of the fracture center.

            The 3d coordinates of the fracture can be recovered by
                p_3d = cp + rot.T.dot(np.vstack((p_2d,
                                                 np.zeros(p_2d.shape[1]))))

        """
        isect = self.intersections_of_fracture(frac_num)

        frac = self.fractures[frac_num]
        cp = frac.center.reshape((-1, 1))

        rot = pp.map_geometry.project_plane_matrix(frac.pts)

        def rot_translate(pts):
            # Convenience method to translate and rotate a point.
            return rot.dot(pts - cp)

        p = rot_translate(frac.pts)
        assert np.max(np.abs(p[2])) < self.tol, (
            str(np.max(np.abs(p[2]))) + " " + str(self.tol)
        )
        p_2d = p[:2]

        # Intersection points, in 2d coordinates
        ip = np.empty((2, 0))

        other_frac = np.empty(0, dtype=int)

        for i in isect:
            if i.first.index == frac_num:
                other_frac = np.append(other_frac, i.second.index)
            else:
                other_frac = np.append(other_frac, i.first.index)

            tmp_p = rot_translate(i.coord)
            if tmp_p.shape[1] > 0:
                assert np.max(np.abs(tmp_p[2])) < self.tol
                ip = np.append(ip, tmp_p[:2], axis=1)

        return p_2d, ip, other_frac, rot, cp
