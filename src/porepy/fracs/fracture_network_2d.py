"""Module contains class for representing a fracture network in a 2d domain."""

from __future__ import annotations

import itertools
import gmsh
import copy
import csv
import logging
import time
from pathlib import Path
from typing import Optional
import multiprocessing

import gmsh
import meshio
import numpy as np

import porepy as pp
import porepy.fracs.simplex
from porepy.fracs import tools
from porepy.fracs.utils import linefractures_to_pts_edges, pts_edges_to_linefractures

from .gmsh_interface import GmshData2d, GmshWriter, PhysicalNames
from .gmsh_interface import Tags as GmshInterfaceTags

logger = logging.getLogger(__name__)

# Shortcut for the OpenCaste CAD kernel in gmsh.
fac = gmsh.model.occ


class FractureNetwork2d:
    """Representation of a set of line fractures in a 2D domain.

    The fractures are represented by line fracture objects (see
    :class:`~porepy.fracs.line_fracture.LineFracture`).

    Polyline fractures are currently not supported.

    There is no requirement or guarantee that the fractures are contained within the
    specified domain. The fractures can be cut to a given domain by the method
    :meth:`constrain_to_domain`.

    The domain can be a general non-convex polygon (see
    :class:`~porepy.geometry.domain.Domain`).

    Note:
        The class is mainly intended for representation and meshing of a fracture
        network. However, it also contains some utility functions. The balance between
        these components may change in the future, especially utility functions may be
        removed.

    Parameters:
        fractures: ``default=None``

            Line fractures that make up the network. Defaults to ``None``, which will
            create a domain without fractures. An empty ``fractures`` list is
            effectively treated as ``None``.
        domain: ``default=None``

            Domain specification. Can be box-shaped or a general (non-convex) polygon.
        tol:  ``default=1e-8``

            Tolerance used in geometric computations.

    """

    def __init__(
        self,
        fractures: Optional[list[pp.LineFracture]] = None,
        domain: Optional[pp.Domain] = None,
        tol: float = 1e-8,
    ) -> None:
        self._pts: np.ndarray
        """Start and endpoints of the fractures. Points can be shared by fractures."""

        self._edges: np.ndarray
        """The fractures as an array of start and end points, referring to ``_pts``.

        Additional rows are optional tags of the fractures. In the standard form, the
        third row (first row of tags) identifies the type of edges, referring to the
        numbering system in GmshInterfaceTags. The second row of tags keeps track of the
        numbering of the edges (referring to the original order of the edges) in
        geometry processing like intersection removal. Additional tags can be assigned
        by the user.

        """

        self.tol: float = tol
        """Tolerance used in geometric computations."""

        self.fractures: list[pp.LineFracture] = [] if fractures is None else fractures
        """List of line fractures."""

        # Save points and edges as private attributes
        if fractures is not None and len(fractures) > 0:
            self._pts, self._edges = linefractures_to_pts_edges(self.fractures, tol)
        else:
            self._pts = np.zeros((2, 0))
            self._edges = np.zeros((2, 0), dtype=int)

        self.domain: Optional[pp.Domain] = domain
        """Domain specification for the fracture network."""

        self.tags: dict[int | str, np.ndarray] = dict()
        """Tags for the fractures."""
        # Note that self.tags is for internal usage, while there is a separate system
        # used for the Gmsh interface, see self._edges. The whole thing works, but there
        # may be inconsistencies between the two systems.

        self.bounding_box_imposed: bool = False
        """Flag indicating whether the bounding box has been imposed."""

        self._decomposition: dict = dict()
        """Dictionary of geometric information obtained from the meshing process.

        This will include intersection points identified.
        """

        # Set the index of the fractures.
        for i, f in enumerate(self.fractures):
            f.set_index(i)

        # Logging
        if self.fractures is not None:
            logger.info("Generated empty fracture set")
        elif self._pts is not None and self._edges is not None:
            # Note: If the list of fracture is empty, we end up here.
            logger.info(f"Generated a fracture set with {self.num_frac()} fractures")
            if self._pts.size > 0:
                logger.debug(
                    f"Minimum point coordinates x: {self._pts[0].min():.2f}, \
                        y: {self._pts[1].min():.2f}",
                )
                logger.debug(
                    f"Maximum point coordinates x: {self._pts[0].max():.2f}, \
                        y: {self._pts[1].max():.2f}",
                )
        else:
            raise ValueError(
                "Specify both points and connections for a 2d fracture network."
            )
        if domain is not None:
            logger.info(f"Domain specification : {str(domain)}")

    def domain_to_gmsh_2D(self) -> int:
        """Export the rectangular domain to Gmsh using the OpenCASCADE kernel.

        This method creates a rectangle corresponding to the bounding box of the
        fracture network domain and adds it to the current Gmsh model. The OpenCASCADE
        CAD kernel is used for the geometry representation.

        Returns:
            The Gmsh tag ID of the created rectangle. This can be used to reference the
            rectangle in further Gmsh operations, such as meshing or boolean operations.

        Notes:
            * Ensure that `gmsh.initialize()` has been called before using this method,
                or call it in the method if starting a fresh Gmsh session.
            * The `gmsh.model.occ.synchronize()` call is required to update the model
                so that the rectangle can be used in subsequent operations.
            * This method currently only supports rectangular domains.

        """
        domain = self.domain
        if domain is None:
            return -1
        if domain.is_boxed:
            bb = self.domain.bounding_box
            xmin, xmax = bb["xmin"], bb["xmax"]
            ymin, ymax = bb["ymin"], bb["ymax"]

            # We assume that z is the zero coordinate when working in 2D, and thus the
            # third input to addRectangle is set to be 0:
            domain_tag = gmsh.model.occ.addRectangle(
                xmin, ymin, 0, xmax - xmin, ymax - ymin
            )
        else:
            # The domain is a general polygon.
            polygon = domain.polytope
            # Get the points of the polygon. We can do this by taking the first column
            # (first point) of each polygon in the list.
            pts = [poly[:, 0] for poly in polygon]
            # Add the points to gmsh.
            pt_tags = [gmsh.model.occ.addPoint(p[0], p[1], 0) for p in pts]
            # Close the list of points, represented as gmsh tags.
            pt_tags.append(pt_tags[0])
            # The lines of the polygon, the line loop, and the plane surface can all be
            # created, assuming that the lines are specificed in a consecutive manner.
            lines = [
                gmsh.model.occ.addLine(pt_tags[i], pt_tags[i + 1])
                for i in range(len(pts))
            ]
            line_loop = gmsh.model.occ.addCurveLoop(lines)
            domain_tag = gmsh.model.occ.addPlaneSurface([line_loop])

        return domain_tag

    def fractures_to_gmsh_2D(self) -> list[int]:
        """Take the tags of all fractures in the fracture network.

        By using the method for exporting a single fracture tag, we here collect the
        tags of all the fractures in the fracture network. The tags are returned as
        elements in a list.

        Returns:
            A list of integers which represent all fracture tags in the fracture
            network.

        NOTE:
            The method fracture_to_gmsh_2D() does not exist yet, nor is its name
            decided.

        """
        fracture_tags = [fracture.fracture_to_gmsh_2D() for fracture in self.fractures]

        return fracture_tags

    def mesh(
        self,
        mesh_args: dict[str, float],
        file_name: Optional[Path] = None,
        constraints: Optional[np.ndarray] = None,
        dfn: bool = False,
        tags_to_transfer: Optional[list[str]] = None,
        write_geo: bool = True,
        finalize_gmsh: bool = True,
        clear_gmsh: bool = False,
        **kwargs,
    ) -> pp.MixedDimensionalGrid:
        """Mesh the fracture network and generate a mixed-dimensional grid.

        Note that the mesh generation process is outsourced to gmsh.

        Returns:
            Mixed-dimensional grid for this fracture network.

        """
        if file_name is None:
            file_name = Path("gmsh_frac_file.msh")

        # No constraints if not available.
        if constraints is None:
            constraints = np.empty(0, dtype=int)
        else:
            constraints = np.atleast_1d(constraints)
            constraints.sort()
        assert isinstance(constraints, np.ndarray)

        gmsh.initialize()

        try:
            num_procs = multiprocessing.cpu_count() or 1
        except (NotImplementedError, AttributeError):
            num_procs = 1

        gmsh.option.setNumber("General.NumThreads", max(num_procs - 2, 1))
        nd = self.domain.dim

        # For the sake of a better overview, I use VSCode's regions to identify roughly
        # which method the new code corresponds to. This should make it easier to create
        # a logical split into methods later. Currently, the method is too long for my
        # taste.

        # region prepare_for_gmsh

        # Get gmsh tags of domain and fractures.
        # TODO Here, the question arises whether we should already enrich the tags with
        # dimensionality, e.g.,
        # gmsh_fractures = [(1, f) for f in self.fractures_to_gmsh_2D()]
        # Both are needed later, so not sure what is best. Having both seems
        # unnecessary.
        domain_tag = self.domain_to_gmsh_2D()
        fracture_tags = self.fractures_to_gmsh_2D()
        gmsh.model.occ.synchronize()

        # Identify fractures fully/partially outside the domain and remove/truncate
        # them.
        # NOTE This is done in such a way that the order of ``fracture_tags`` is
        # preserved. E.g., [0, 1, 2, 3, 4] -> [0, 2, 5, 4] if the fracture with gmsh tag
        # 1 is removed and fracture 3 is replaced with the truncated fracture 5.
        new_fractures = {}
        removed_fractures = []

        for ind, fracture_tag in enumerate(fracture_tags):
            # According to gmsh documentation (v4.14), the function intersect should be
            # able to identify fractures that do not intersect with the domain. The
            # expected result is that the map from the old fracture to the new one, that
            # is, the second return variable from the call to intersect is empty.
            # However, this does not seem to work unless the parameters removeTool and
            # removeObject are set to True (either both or one of them must be True, EK
            # is not sure exactly what counts). However, using these will remove the
            # fracture and/or the domain from the gmsh model, and even though we could
            # reintroduce them if it turns out that the fracture is indeed (partially)
            # within the domain, that will lead to a host of questions regarding
            # preserving tags etc. Instead, we therefore compute the distance between
            # the fracture and the domain, and if this is larger than tol (NOTE: the
            # sensitivity to this parameter is not thoroughly tested), the fracture will
            # be removed.
            distance = fac.getDistance(nd - 1, fracture_tag, nd, domain_tag)[0]
            if distance > self.tol:
                # The fracture is fully outside the domain. It will be deleted.
                removed_fractures.append(ind)
                continue

            # The fracture is either fully or partly inside the domain. We call
            # intersect to truncate the fracture if necessary.
            truncated_fracture, _ = fac.intersect(
                [(nd - 1, fracture_tag)],
                [(nd, domain_tag)],
                removeTool=False,
                removeObject=False,
            )
            if len(truncated_fracture) > 0 and truncated_fracture[0][1] != fracture_tag:
                # The fracture was partly outside the domain. It will be replaced.
                new_fractures[ind] = truncated_fracture[0]

        # Remove the fractures from the gmsh representation. Recursive is critical here,
        # or else the boundary of 'fracture' will continue to be present.
        for ind in removed_fractures:
            fac.remove([(nd - 1, fracture_tags[ind])], recursive=True)
        # Also update the constraints: Each fracture removal in effect shifts the
        # indices, but only for those with a higher index.
        for i in range(len(removed_fractures)):
            constraints = np.array(
                [
                    c - np.sum(removed_fractures < c)
                    for c in constraints
                    if c != removed_fractures[i]
                ]
            )

        # Remove fractures that were truncated from the gmsh representation and update
        # ``fractures`` with the tag of the truncated fracture.
        for old_fracture, new_fracture in new_fractures.items():
            fac.remove([(nd - 1, fracture_tags[old_fracture])], recursive=True)
            fracture_tags[old_fracture] = new_fracture[1]

        # Remove from fracture_tags those indices that are present in removed_fractures
        fracture_tags = [
            ft for i, ft in enumerate(fracture_tags) if i not in removed_fractures
        ]

        fac.synchronize()

        # Make gmsh calculate the intersections between fractures, using the domain as a
        # secondary object (the latter will by magic ensure that the fractures are
        # embedded in the domain, hence the mesh will conform to the fractures). The
        # removal statements here will replace the old (possibly intersecting) fractures
        # with new, split lines. Similarly, the removal of the domain (removeTool)
        # avoids the domain being present twice.
        _, isect_mapping = fac.fragment(
            [(nd - 1, ft) for ft in fracture_tags],
            [(nd, domain_tag)],
            removeObject=True,
            removeTool=True,
        )

        fac.synchronize()

        # During intersection removal, gmsh will add intersection points and replace the
        # fractures with non-intersecting polylines (example: Two fractures intersecting
        # as a cross become four fractures with a common point). We need to identify
        # these intersecting points.
        # Annoyingly, gmsh also reassigns tags during fragmentation. Hence we need to
        # find intersection points on our own, as is done below.
        intersection_points = []

        # Loop over the mappings from old fractures to new segments. The idea is to find
        # the boundary points of all segments, identify those that occur more than once
        # - these will be intersections - and store the tag that gmsh has assigned them.
        for fi, old_fracture in enumerate(isect_mapping):
            if len(old_fracture) > 0 and old_fracture[0][0] == nd:
                # It is unclear if processing the domain will make any harm, but there
                # is no need to take any chances. Skip it.
                continue

            if fi in constraints:
                # Constrained fractures are not to be considered for intersection
                # identification.
                continue

            all_boundary_points_of_segments = []
            for segment in old_fracture:
                all_boundary_points_of_segments += gmsh.model.get_boundary([segment])

            # Find the unique boundary points and obtain a mapping from the full set of
            # boundary points to the unique ones.
            unique_boundary_points, u2a_ind = np.unique(
                all_boundary_points_of_segments, axis=0, return_inverse=True
            )
            # Count the number of occurrences of each unique boundary point. Points that
            # occur more than once will be intersections.
            multiple_occs = np.where(np.bincount(u2a_ind) > 1)[0]
            # Store the gmsh representation of the intersection points.
            intersection_points += [unique_boundary_points[i] for i in multiple_occs]

        # Finally, we need to uniquify the intersection points, since the same point
        # will have been identified in at least two old fractures.
        if len(intersection_points) > 0:
            # Some extra gymnastics: intersection_points is a list of points at the
            # intersection between 1d lines, but Gmsh fragment makes no difference
            # between lines that we consider real fractures and constraints. Points that
            # form part of a constraints are not added to intersection_points, since we
            # skipped constraints in the previous loop. To identify those points that
            # are on the intersection between real fractures, we filter away points that
            # occur less than twice in intersection_points.

            candidate_intersection_points, unique_2_all = np.unique(
                # Take the second column, which contains the point tags. The first
                # column is the dimension, which is always 0 here.
                np.vstack(intersection_points)[:, 1],
                axis=0,
                return_inverse=True,
            )
            found_by_multiple_fractures = np.where(np.bincount(unique_2_all) > 1)[0]
            unique_intersection_points = candidate_intersection_points[
                found_by_multiple_fractures
            ]
        else:
            # No intersections, simply create an empty list.
            unique_intersection_points = np.array([], dtype=int)

        # Collect intersection points, fractures, and domain in physical groups in gmsh.

        # Intersection points can be dealt with right away.
        for i, pt in enumerate(unique_intersection_points):
            gmsh.model.addPhysicalGroup(
                nd - 2,
                [pt],
                -1,
                f"{PhysicalNames.FRACTURE_INTERSECTION_POINT.value}{i}",
            )

        fac.synchronize()

        # Since fractures may have been split at intersection points, we need to collect
        # all the segments (found in isect_mapping) into a single physical group.
        for i, line_group in enumerate(isect_mapping):
            all_lines = []
            for line in line_group:
                if line[0] == 1:
                    all_lines.append(line[1])
            if all_lines:
                if i in constraints:
                    gmsh.model.addPhysicalGroup(
                        nd - 1,
                        all_lines,
                        -1,
                        f"{PhysicalNames.AUXILIARY_LINE.value}{i}",
                    )
                else:
                    gmsh.model.addPhysicalGroup(
                        nd - 1, all_lines, -1, f"{PhysicalNames.FRACTURE.value}{i}"
                    )

        # It turns out that if fractures split the domain into disjoint parts, gmsh may
        # choose to redefine the domain as the sum of these parts. Therefore, we
        # redefine the domain tags here, using all volumes in the model.
        domain_tags = [entity[1] for entity in gmsh.model.get_entities(nd)]

        gmsh.model.addPhysicalGroup(
            nd, domain_tags, -1, f"{PhysicalNames.DOMAIN.value}"
        )

        fac.synchronize()

        # Set the mesh sizes after all geometry processing is done so that the
        # identification of objects is not disturbed by retagging of objects.
        self.set_mesh_size(mesh_args)
        gmsh.model.occ.synchronize()

        # region GmshWriter.generate
        if write_geo:
            gmsh.write(str(file_name.with_suffix(".geo_unrolled")))

        # Consider the dimension of the problem. Normally 2D, but if ``dfn`` is True 1D.
        ndim = nd - int(dfn)

        # Create a gmsh mesh.
        gmsh.model.mesh.generate(ndim)
        gmsh.write(str(file_name))

        # Create list of grids.
        if dfn:
            # FIXME The constraint weren't considered until here, so this will probably
            # not work when constraints is not None.
            subdomains = porepy.fracs.simplex.line_grid_from_gmsh(
                file_name, constraints=constraints
            )

        else:
            subdomains = porepy.fracs.simplex.triangle_grid_from_gmsh(
                file_name, constraints=constraints
            )

        if clear_gmsh:
            gmsh.clear()
        if finalize_gmsh:
            gmsh.finalize()

        # Assemble all subdomains in mixed-dimensional grid.
        return pp.meshing.subdomains_to_mdg(subdomains, **kwargs)

    def set_mesh_size(self, mesh_args: dict[str, float]) -> None:
        factory = gmsh.model.occ

        # Define a threshold for when to consider refining along fractures. This is a
        # heuristic value which should be reconsidered. Scaling with mesh size on
        # fractures is reasonable, but the factor 2 is arbitrary.
        THRESHOLD_REFINEMENT = mesh_args["mesh_size_frac"]

        # Now set the mesh sizes using gmsh fields. TODO: Give better names to ALPHA and
        # BETA and make them user parameters, IF the concept turns out to be useful and
        # extendable to 3d.

        # Alpha is a parameter controlling the mesh size in regions where refinement is
        # needed, e.g., if two fractures are close. In the immediate vicinity of such
        # regions, the mesh size is set to d/ALPHA, where d is the distance to the
        # object requiring refinement.
        ALPHA = 3
        # Beta is a parameter controlling the size of the transition region from fine
        # mesh to coarse mesh. The transition ends at a distance BETA*h_frac from the
        # object requiring refinement.
        BETA = 15

        h_frac = mesh_args["mesh_size_frac"]
        h_bound = mesh_args["mesh_size_bound"]
        # EK note to self: I am not entirely sure whether h_min should remain a user
        # parameter, or if we can get rid of it.
        h_min = mesh_args.get("mesh_size_min", h_frac / ALPHA)

        ### Get hold of lines representing fractures and boundaries.
        domain_entities = gmsh.model.get_entities(2)
        # TODO: If there is more than one domain entity (the domain is split into parts
        # by fractures), we need to pick out the outer boundary, that is, the ones which
        # only occurs once.
        boundaries = gmsh.model.get_boundary([(2, tag) for _, tag in domain_entities])
        fractures = [f for f in gmsh.model.getEntities(1) if f not in boundaries]

        line_tags = [tag for _, tag in gmsh.model.getEntities(1)]
        fracture_tags = [tag for _, tag in fractures]
        boundary_tags = [tag for _, tag in boundaries]

        mesh_size_points = {}

        def project_onto_line(pt, start_of_line, vector):
            vec_to_point = pt - start_of_line
            # Project the vector onto the line defined by the start point and the
            # direction vector, parametrized by t.
            return np.dot(vec_to_point, vector) / np.dot(vector, vector)

        for line_1, line_2 in itertools.combinations(line_tags, 2):
            if (line_1 in boundary_tags) and (line_2 in boundary_tags):
                # Both lines are on the boundary, so we do not need to consider their
                # mutual distance.
                continue

            distances = factory.getDistance(1, line_1, 1, line_2)

            if distances[0] < THRESHOLD_REFINEMENT:
                # Associate the closest points and distances with each line for later
                # processing. This will either be an intersection point or just a close
                # point.
                info_1 = mesh_size_points.get(line_1, [])
                info_1.append((distances[1:4], distances[0]))
                mesh_size_points[line_1] = info_1
                info_2 = mesh_size_points.get(line_2, [])
                info_2.append((distances[4:7], distances[0]))
                mesh_size_points[line_2] = info_2

                # We also need to check check all combinations of endpoints of ecah
                # fracture with the line on the other, to detect cases of (almost)
                # parallel lines, which may require fine meshing along the one or
                # both of the fractures.

                # For each of the endpoints of each the two lines, end_point_distance
                # will store the distance and the closest points on the other line.

                # The below loop extracts the following information regarding the two
                # lines:
                # - main_start_point: A closest point on one of the lines that is also
                #   an endpoint of that line. There will be one or two such points; we
                #   pick the first one we find.
                # - vectors: Direction vectors along each line. For the line containing
                #   main_start_point, the vector is oriented away from that point.
                # - main_line_ind: Index (0 or 1) of the line containing
                #   main_start_point.
                #

                vectors = []
                closest_points = [np.array(distances[1:4]), np.array(distances[4:7])]
                main_start_point = None
                end_point_coordinates = []
                main_line_ind = None

                # The loop variables are ind (index of the line, 0 or 1), line (the line
                # tag), and cp (the closest point on that line to the other line).
                for ind, (line, cp) in enumerate(zip([line_1, line_2], closest_points)):
                    other_line = line_2 if line == line_1 else line_1

                    # Get the endpoints of the line, in terms of gmsh tag-index pairs.
                    point_tags = gmsh.model.getBoundary(
                        [(1, line)], combined=False, oriented=False
                    )
                    # Get the coordinates of the endpoints.
                    end_point_coordinates += [
                        gmsh.model.get_bounding_box(*point_tags[i])[:3]
                        for i in range(2)
                    ]
                    # Vector along the line.
                    vectors.append(
                        np.array(
                            [
                                end_point_coordinates[-1][i]
                                - end_point_coordinates[-2][i]
                                for i in range(3)
                            ]
                        )
                    )
                    # Check if one of the endpoints is the closest point.
                    if main_start_point is None:
                        for i in range(2):
                            # Since end_point_coordinates contain endpoints of both
                            # lines, we need to index carefully.
                            d = np.linalg.norm(
                                cp - np.array(end_point_coordinates[-2 + i])
                            )
                            if d < self.tol:  # This should really be a gmsh tolerance.
                                main_start_point = np.array(
                                    end_point_coordinates[-2 + i]
                                )
                                main_line_ind = ind
                                if i == 1:
                                    # Ensure that the vector points away from the
                                    # closest point.
                                    vectors[-1] = -vectors[-1]

                if main_start_point is None:
                    assert False, "Logic error in closest point identification."

                # Identify the other line.
                other_line_ind = 1 - main_line_ind
                main_line_gmsh_ind = line_1 if main_line_ind == 0 else line_2
                other_line_gmsh_ind = line_2 if main_line_ind == 0 else line_1
                # Start point on the other line.
                other_start_point = end_point_coordinates[other_line_ind][0]
                # Vectors along the two lines.
                vector_other_line = vectors[other_line_ind]
                vector_main_line = vectors[main_line_ind]
                # Project the main_start_point onto the extension of the other line to
                # see if it falls within the segment defined by the other line.
                t_closest = project_onto_line(
                    main_start_point, np.array(other_start_point), vector_other_line
                )

                if t_closest < 0 or t_closest > 1:
                    # The closest point is outside the segment defined by the other
                    # line. We will not do further refinement along the lines in this
                    # case.
                    continue

                # Compute the angle between the two lines. Clip the cosine value to
                # avoid numerical issues.
                cos_angle = np.clip(
                    np.dot(
                        vector_main_line / np.linalg.norm(vector_main_line),
                        vector_other_line / np.linalg.norm(vector_other_line),
                    ),
                    -1.0,
                    1.0,
                )
                angle = np.arccos(cos_angle)
                # Force the angle to be in the right quadrant.
                if angle > np.pi / 2:
                    angle = np.pi - angle

                # The mesh size field centered at a point, and on lines, specifies that
                # the mesh size should increase from distances[0] / ALPHA to h_bound
                # over the interval distances[0] to BETA * h_frac. In the specification
                # of mesh_size_points above, we ensured that a mesh-size point is placed
                # at the closest point.
                #
                # The question then is whether to place additional points along the
                # lines, and if so, how to determine their locations. Intuitively, if
                # the lines are parallel, new points will be needed to ensure refinement
                # along the entire length of the lines (or at least the parts that are
                # close). If the lines are not orthogonal, most likely no additional
                # points are needed (though it may be possible to define h_frac,
                # h_bound, ALPHA, and BETA such that additional points are placed).
                #
                # It turns out that new points are needed only if the distance between
                # the lines when moving along the lines away from the closest point
                # increases faster than than the mesh size field increases. This leads
                # to the following condition for the angle between the lines:

                # Safeguarding. Should fix this, but that requires thinking about the
                # angles return from arctan2.
                assert h_bound >= h_frac

                # Angle of incline of the mesh size field.
                ANGLE_THRESHOLD = np.arctan2(
                    h_bound - distances[0] / ALPHA, BETA * h_frac - distances[0]
                )

                if angle > ANGLE_THRESHOLD:
                    # The lines are diverging fast enough that no further refinement is
                    # needed.
                    continue

                # The closest points on at least on of the lines must be an endpoint of
                # that line. Find it.

                if not (
                    angle < ANGLE_THRESHOLD or np.abs(angle - np.pi) < ANGLE_THRESHOLD
                ):
                    # The lines are not (almost) parallel, so we do not need to consider
                    # this any further.
                    continue

                # The lines are close enough to being parallel that there is a need for
                # refinement, not centered on the closest points only, but also along
                # (parts of) the main line. We *assume* here that it is sufficient to
                # prescribe new mesh size points along the main line, and that the other
                # line will be refined accordingly.
                if distances[0] < self.tol:
                    step_size = h_frac / ALPHA
                else:
                    step_size = max(h_min, distances[0])
                prev_point = main_start_point
                while True:
                    # Next candidate point along the main line. Use a normalized
                    # direction vector to let the step size be independent of the length
                    # of the line.
                    next_point = (
                        prev_point
                        + step_size
                        * vector_main_line
                        / np.linalg.norm(vector_main_line)
                    )
                    if np.linalg.norm(next_point - main_start_point) > np.linalg.norm(
                        vector_main_line
                    ):
                        # We have reached the end of the main line.
                        break
                    # Project the next point onto the other line to see if it falls
                    # within the segment defined by the other line.
                    t_next = project_onto_line(
                        next_point,
                        np.array(other_start_point),
                        vector_other_line,
                    )
                    if t_next < 0 or t_next > 1:
                        # The next point is outside the segment defined by the other
                        # line.
                        break
                    # Add the point temporarily to gmsh to compute the distance to the
                    # other line.
                    pi = factory.add_point(next_point[0], next_point[1], next_point[2])
                    d = gmsh.model.occ.get_distance(0, pi, 1, other_line_gmsh_ind)[0]
                    assert d >= distances[0], "Logic error in distance computation."
                    if d > THRESHOLD_REFINEMENT:
                        # Remove the point again.
                        factory.remove([(0, pi)])
                        # No further refinement needed.
                        break
                    else:
                        # Store the point and distance information for later processing.
                        info = mesh_size_points.get(main_line_gmsh_ind, [])
                        info.append((next_point, d))
                        mesh_size_points[main_line_gmsh_ind] = info
                    prev_point = next_point
                    step_size = d

        ## Start feeding the mesh size information to gmsh fields.
        gmsh_fields = []

        # Add fields for points close to other objects. The points are stored per line,
        # take one line at a time.
        for line, info in mesh_size_points.items():
            # Uniquify the point set (the same point may have been identified multiple
            # times).
            end_points = np.array(
                [
                    gmsh.model.occ.get_bounding_box(0, p[1])[:3]
                    for p in gmsh.model.get_boundary([(1, line)], combined=False)
                ]
            ).T
            length = np.linalg.norm(end_points[:, 1] - end_points[:, 0])
            tol = min(length, h_frac) / 2
            extra_points = np.array([d[0] for d in info]).T
            points, _, ind_map = pp.array_operations.uniquify_point_set(
                np.hstack((end_points, extra_points)), tol=tol
            )
            # Distance to other objects for each point, as computed previously. Assign
            # h_frac or h_bound to the endpoints, depending on whether the line is a
            # fracture or boundary line. We also assign h_frac, since no refinement is
            # needed just because this is an intersection point (if it is an
            # intersection with a bad angle, this should be picked up by a close point
            # on another line).
            h_end = h_bound if line in boundary_tags else h_frac

            other_object_distances_all = np.hstack(
                (
                    np.array([h_end, h_end]),
                    np.array([d[1] if d[1] > 0 else h_frac for d in info]),
                )
            )
            # Reduce to one distance per unique point, picking the minimum distance if
            # multiple distances were associated with the same geometric point.
            other_object_distances = []
            for i in range(points.shape[1]):
                inds = ind_map == i
                min_dist = np.min(other_object_distances_all[inds])
                other_object_distances.append(min_dist)
            other_object_distances = np.array(other_object_distances)

            # Mesh size information that relates to either endpoints or points close to
            # end points (which were filtered out) must be assigned to endpoints.

            if points.shape[1] > 0:
                # If there is more than one point in addition to the end points, we can
                # compute the point-point distances in pairs along this line.
                point_point_distances = pp.distances.pointset(points, max_diag=True)
                min_dist_point = np.min(point_point_distances, axis=0)
            else:
                # This is an isolated point. There is no reason to do refinement for
                # this line, though, if the same point is identified for other lines, it
                # may be added there. Note to self: A standard X-intersection with no
                # other lines in the vicinity will end up here.
                continue

            # The final distance to be used for mesh size calculation is the minimum of
            # the distance to other objects and the distance to other close points on
            # the same line.
            dist = np.minimum(other_object_distances, min_dist_point)
            for i, d in enumerate(dist):
                # Need set a lower bound on the mesh size to avoid zero distances, e.g.,
                # related to almost intersection points.
                if d > h_frac:
                    # No refinement needed at this point.
                    continue

                size = max(h_min, d) / ALPHA
                field = gmsh.model.mesh.field.add("Distance")
                pi = gmsh.model.occ.add_point(points[0, i], points[1, i], points[2, i])
                gmsh.model.mesh.field.setNumbers(field, "PointsList", [pi])
                threshold = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(threshold, "InField", field)

                # NOTE: If the definition of the threshold field is changed, the
                # computation of the critical angle for almost parallel lines must also
                # be updated. See the definition of variable 'angle_threshold' above.
                gmsh.model.mesh.field.setNumber(threshold, "DistMin", size)
                gmsh.model.mesh.field.setNumber(threshold, "SizeMin", size)
                gmsh.model.mesh.field.setNumber(threshold, "DistMax", BETA * h_frac)
                gmsh.model.mesh.field.setNumber(threshold, "SizeMax", h_bound)
                gmsh_fields.append(threshold)

        for line in fracture_tags:
            # Adapt the number of sampling points along the line to its length and the
            # fracture mesh size. There should be at least two points, though.
            end_points = gmsh.model.getBoundary([(1, line)], combined=False)
            length = gmsh.model.occ.get_distance(
                end_points[0][0], end_points[0][1], end_points[1][0], end_points[1][1]
            )[0]
            num_points = max(2, int(np.ceil(length / h_frac)) + 1)

            field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field, "CurvesList", [line])
            gmsh.model.mesh.field.setNumber(field, "Sampling", num_points)

            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", field)
            # Start coarsening immediately from zero distance.
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0)
            # TODO: This enforces at least ALPHA elements along each fracture. Do we want
            # this?
            gmsh.model.mesh.field.setNumber(
                threshold, "SizeMin", min(h_frac, length / ALPHA)
            )
            gmsh.model.mesh.field.setNumber(threshold, "DistMax", BETA * h_frac)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", h_bound)
            gmsh_fields.append(threshold)

        for line in boundary_tags:
            # Set mesh size along the boundary.
            field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field, "CurvesList", [line])
            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", field)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMin", h_bound)
            gmsh.model.mesh.field.setNumber(threshold, "DistMax", BETA * h_bound)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", h_bound)
            gmsh_fields.append(threshold)

        # Finally, as the background mesh, we take the minimum of all the created
        # fields.
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", gmsh_fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        # The background mesh incorporates all mesh size specifications. We turn off
        # other mesh size specifications.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    def mesh_old(
        self,
        mesh_args: dict[str, float],
        tol: Optional[float] = None,
        do_snap: bool = True,
        constraints: Optional[np.ndarray] = None,
        file_name: Optional[Path] = None,
        dfn: bool = False,
        tags_to_transfer: Optional[list[str]] = None,
        remove_small_fractures: bool = False,
        write_geo: bool = True,
        finalize_gmsh: bool = True,
        clear_gmsh: bool = False,
        **kwargs,
    ) -> pp.MixedDimensionalGrid:
        """Mesh the fracture network and generate a mixed-dimensional grid.

        Note that the mesh generation process is outsourced to gmsh.

        Parameters:
            mesh_args: Arguments passed on to mesh size control. It should contain,
                minimally, the keyword ``mesh_size_frac``. Other supported keywords are
                ``mesh_size_bound`` and ``mesh_size_min``.
            tol: ``default=None``

                Tolerance used for geometric computations. If not given,
                the tolerance of this fracture network will be used.
            do_snap: ``default=True``

                Whether to snap lines to avoid small segments.
            constraints: ``dtype=np.int32, default=None``

                Indices of fractures that should not generate lower-dimensional
                meshes, but only act as constraints in the meshing algorithm. Useful
                to define subregions of the domain (and assign, e.g., material
                properties, sources, etc.).

            file_name: ``default=None``

                Name of the output gmsh file(s). If not given, ``gmsh_frac_file``
                will be assigned.
            dfn: ``default=False``

                Whether the fracture network is of the DFN (Discrete Fracture
                Network) type. If ``True``, a DFN mesh, where only the network (and not
                the surrounding matrix) is created.
            tags_to_transfer: ``default=None``

                Tags of the fracture network that should be passed to the fracture
                grids.
            remove_small_fractures: ``default=False``

                DEPRECATED.
            write_geo: ``default=True``

                Whether to generate to write the Gmsh configuration file. If ``True``,
                the Gmsh configuration will be written to a ``.geo_unrolled`` file.
            finalize_gmsh: ``default=True``

                Whether to close the port to Gmsh when the meshing process is
                completed.

                On repeated invocations of Gmsh in the same Python session, a memory
                leak in Gmsh may cause reduced performance (written spring 2021). In
                these cases, it may be better to finalize gmsh externally
                to this class. See also ``clear_gmsh``.
            clear_gmsh: ``default=False``

                Whether to delete the geometry representation in Gmsh when the
                meshing process is completed.

                This is of use only if ``finalize_gmsh=False`, in which case it may be
                desirable to delete the old geometry before adding a new one.
            **kwargs:
                Passed on to the meshing function.

        Returns:
            Mixed-dimensional grid for this fracture network.

        """
        if file_name is None:
            file_name = Path("gmsh_frac_file.msh")

        # No constraints if not available.
        if constraints is None:
            constraints = np.empty(0, dtype=int)
        else:
            constraints = np.atleast_1d(constraints)
        assert isinstance(constraints, np.ndarray)

        gmsh_repr = self.prepare_for_gmsh(
            mesh_args, tol, do_snap, constraints, dfn, remove_small_fractures
        )
        gmsh_writer = GmshWriter(gmsh_repr)

        # Consider the dimension of the problem, normally 2d but if dfn is true 1d
        ndim = 2 - int(dfn)

        gmsh_writer.generate(
            file_name,
            ndim,
            write_geo=write_geo,
            finalize=finalize_gmsh,
            clear_gmsh=clear_gmsh,
        )

        if dfn:
            # Create list of grids.
            subdomains = porepy.fracs.simplex.line_grid_from_gmsh(
                file_name, constraints=constraints
            )

        else:
            # Create list of grids.
            subdomains = porepy.fracs.simplex.triangle_grid_from_gmsh(
                file_name, constraints=constraints
            )

        if tags_to_transfer:
            # Preserve tags for the fractures from the network. We assume a coherent
            # numeration between the network and the created grids.
            frac = np.setdiff1d(
                np.arange(self._edges.shape[1]), constraints, assume_unique=True
            )
            for idg, g in enumerate(subdomains[1 - int(dfn)]):
                for key in np.atleast_1d(tags_to_transfer):
                    if key not in g.tags:
                        g.tags[key] = self.tags[key][frac][idg]

        # Assemble in md grid.
        return pp.meshing.subdomains_to_mdg(subdomains, **kwargs)

    def prepare_for_gmsh(
        self,
        mesh_args: dict[str, float],
        tol: Optional[float] = None,
        do_snap: bool = True,
        constraints: Optional[np.ndarray] = None,
        dfn: bool = False,
        remove_small_fractures: bool = False,
    ) -> GmshData2d:
        """Process network intersections and write ``.geo`` configuration file.

        Note:
             Consider using
             :meth:`~porepy.fracs.fracture_network_2d.FractureNetwork2d.mesh` instead
             to get a usable mixed-dimensional grid.

        Parameters:
            mesh_args: Arguments passed on to mesh size control. It should contain at
                least the keyword ``mesh_size_frac``.
            tol: ``default=None``

                Tolerance used for geometric computations. If not given,
                the tolerance of this fracture network will be used.
            do_snap: ``default=True``

                Whether to snap lines to avoid small segments.
            constraints: ``dtype=np.int32, default=None``

                Indices of fractures that should not generate lower-dimensional
                meshes, but only act as constraints in the meshing algorithm.
            dfn: ``default=False``

                Whether the fracture network is of the DFN (Discrete Fracture
                Network) type. If ``True``, a DFN mesh, where only the network (and not
                the surrounding matrix) is created.
            remove_small_fractures: ``default=False``

                DEPRECATED.

        Returns:
            The data structure for gmsh in 2D.

        """

        if tol is None:
            tol = self.tol

        # No constraints if not available.
        if constraints is None:
            constraints = np.empty(0, dtype=int)
        else:
            constraints = np.atleast_1d(constraints)
        assert isinstance(constraints, np.ndarray)
        constraints = np.sort(constraints)

        p = self._pts
        e = self._edges

        num_edge_orig = e.shape[1]

        # Snap points to edges
        if do_snap and p is not None and p.size > 0:
            p, _ = self._snap_fracture_set(p, snap_tol=tol)

        self._pts = p

        if remove_small_fractures:
            # fractures smaller than the prescribed tolerance are removed
            s = "remove small fractures is deprecated."
            s += "The operation should be done manually before calling prepare_for_gmsh"
            raise TypeError(s)

        if not self.bounding_box_imposed:
            edges_kept, edges_deleted = self.impose_external_boundary(
                self.domain, add_domain_edges=not dfn
            )
            # Find edges of constraints to delete
            to_delete = np.where(np.isin(constraints, edges_deleted))[0]

            # Adjust constraint indices: Must be decreased for all deleted lines with
            # lower index, and increased for all lines with lower index that have
            # been split.
            adjustment = np.zeros(num_edge_orig, dtype=int)
            # Deleted edges give an index reduction of 1
            adjustment[edges_deleted] = -1

            # identify edges that have been split
            num_occ = np.bincount(edges_kept, minlength=adjustment.size)

            # Not sure what to do with split constraints; it should not be difficult,
            # but the current implementation does not cover it.
            assert np.all(num_occ[constraints] < 2)

            # Splitting of fractures give an increase of index corresponding to the
            # number of repeats. The clip avoids negative values for deleted edges,
            # these have been accounted for before. Maybe we could merge the two
            # adjustments.
            adjustment += np.clip(num_occ - 1, 0, None)

            # Do the real adjustment
            constraints += np.cumsum(adjustment)[constraints]

            # Delete constraints corresponding to deleted edges
            constraints = np.delete(constraints, to_delete)

            # FIXME: We do not keep track of indices of fractures and constraints
            #  before and after imposing the boundary.

        # The fractures should also be snapped to the boundary.
        if do_snap:
            self._snap_to_boundary(snap_tol=tol)

        # remove the edges that overlap the boundary
        to_delete = self._edges_overlapping_boundary(tol)
        self._edges = np.delete(self._edges, to_delete, axis=1)

        # if a non boundary edge is removed, orphan points may be present. Remove them
        new_pts_id = self._remove_orphan_pts()
        self._decomposition["domain_boundary_points"] = new_pts_id[
            self._decomposition["domain_boundary_points"]
        ]

        # uniquify the points
        self._pts, _, old_2_new = pp.array_operations.uniquify_point_set(
            self._pts, tol=self.tol
        )
        self._edges = old_2_new[self._edges]
        self._decomposition["domain_boundary_points"] = old_2_new[
            self._decomposition["domain_boundary_points"]
        ]

        # map the constraint index
        index_map = np.where(np.logical_not(to_delete))[0]
        mapped_constraints = np.arange(index_map.size)[np.isin(index_map, constraints)]

        # update the tags
        for key, value in self.tags.items():
            self.tags[key] = np.delete(value, to_delete)

        self._find_and_split_intersections(mapped_constraints)
        # Insert auxiliary points and determine mesh size.
        # _insert_auxiliary_points(..) does both.
        # _set_mesh_size_without_auxiliary_points() sets the mesh size
        # to the existing points. This is only done for DFNs, but could
        # also be used for any grid if that is desired.
        if not dfn:
            self._insert_auxiliary_points(**mesh_args)
        else:
            self._set_mesh_size_without_auxiliary_points(**mesh_args)

        # Transfer data to the format expected by the gmsh interface.

        # This requires some information processing and translation between data
        # formats: In the geometry processing undertaken up to this point,
        # it has been convenient to use numerical values for identifying the
        # different line types (fracture, constraint, boundary). For the Gmsh
        # processing, a string-based system is used, as this is more readable and
        # closer to the system employed in Gmsh. In practice, this requires
        # translating from GmshInterfaceTags values to ``names``.

        # In addition to this translation, the below code also does some interpretation
        # of the information obtained during geometry processing.

        decomp = self._decomposition

        edges = decomp["edges"]
        # Information about line types is found in the third row of edges
        edge_types = edges[2]

        # Process information about lines that should be tagged as physical by Gmsh.
        # These are fractures, domain boundaries and auxiliary (constraints).
        # phys_line_tags is a mapping from line index to the Tag.
        phys_line_tags: dict[int, GmshInterfaceTags] = {}

        for ei, tag in enumerate(edge_types):
            if tag in (
                GmshInterfaceTags.FRACTURE.value,
                GmshInterfaceTags.DOMAIN_BOUNDARY_LINE.value,
                GmshInterfaceTags.AUXILIARY_LINE.value,
            ):
                # Note: phys_line_tags contains the GmshInterfaceTags instead of
                # the numbers in edges[2].
                phys_line_tags[ei] = GmshInterfaceTags(tag)

        # Tag all points that have been defined as intersections between fractures.
        # phys_point_tags is a mapping from the point index to the tag.
        phys_point_tags: dict[int, GmshInterfaceTags] = {
            i: GmshInterfaceTags.FRACTURE_INTERSECTION_POINT
            for i in decomp["intersections"]
        }

        # Find points on the boundary, and mark these as physical points.
        point_on_boundary = edges[
            :2, edge_types == GmshInterfaceTags.DOMAIN_BOUNDARY_LINE.value
        ].ravel()
        phys_point_tags.update(
            {pi: GmshInterfaceTags.DOMAIN_BOUNDARY_POINT for pi in point_on_boundary}
        )

        # Find points that are both on the boundary and on a fracture. These have
        # a special tag, thus override the values set for normal boundary points.
        point_on_fracture = edges[
            :2, edge_types == GmshInterfaceTags.FRACTURE.value
        ].ravel()
        fracture_boundary_points = np.intersect1d(point_on_fracture, point_on_boundary)
        phys_point_tags.update(
            {
                pi: GmshInterfaceTags.FRACTURE_BOUNDARY_POINT
                for pi in fracture_boundary_points
            }
        )

        data = GmshData2d(
            pts=decomp["points"],
            mesh_size=decomp["mesh_size"],
            lines=edges,
            physical_points=phys_point_tags,
            physical_lines=phys_line_tags,
        )
        return data

    def _find_and_split_intersections(self, constraints: np.ndarray) -> None:
        """Unified description of points and lines for domain and fractures.

        Parameters:
            constraints: ``dtype=np.int32``

                Indices of fractures which should be considered meshing constraints,
                not as physical objects.

        """
        points = self._pts
        edges = self._edges

        if not np.all(np.diff(edges[:2], axis=0) != 0):
            raise ValueError("Found a point edge in splitting of edges")

        tags = np.zeros((2, edges.shape[1]), dtype=int)

        tags[0][np.logical_not(self.tags["boundary"])] = (
            GmshInterfaceTags.FRACTURE.value
        )
        tags[0][self.tags["boundary"]] = GmshInterfaceTags.DOMAIN_BOUNDARY_LINE.value
        tags[0][constraints] = GmshInterfaceTags.AUXILIARY_LINE.value

        tags[1] = np.arange(edges.shape[1])

        edges = np.vstack((edges, tags))

        # Ensure unique description of points
        pts_all, _, old_2_new = pp.array_operations.uniquify_point_set(
            points, tol=self.tol
        )
        edges[:2] = old_2_new[edges[:2]]
        to_remove = np.where(edges[0, :] == edges[1, :])[0]
        lines = np.delete(edges, to_remove, axis=1)

        self._decomposition["domain_boundary_points"] = old_2_new[
            self._decomposition["domain_boundary_points"]
        ]

        # In some cases the fractures and boundaries impose the same constraint
        # twice, although it is not clear why. Avoid this by uniquifying the lines.
        # This may disturb the line tags in lines[2], but we should not be dependent
        # on those.
        li = np.sort(lines[:2], axis=0)
        _, new_2_old, old_2_new = np.unique(
            li, axis=1, return_index=True, return_inverse=True
        )
        lines = lines[:, new_2_old]

        if not np.all(np.diff(lines[:2], axis=0) != 0):
            raise ValueError(
                "Found a point edge in splitting of edges after merging points"
            )

        # We split all fracture intersections so that the new lines do not
        # intersect, except possible at the end points
        logger.debug("Remove edge crossings")
        tm = time.time()

        pts_split, lines_split, *_ = pp.intersections.split_intersecting_segments_2d(
            pts_all, lines, tol=self.tol
        )
        logger.debug("Done. Elapsed time " + str(time.time() - tm))

        # Ensure unique description of points
        pts_split, _, old_2_new = pp.array_operations.uniquify_point_set(
            pts_split, tol=self.tol
        )
        lines_split[:2] = old_2_new[lines_split[:2]]
        # FIXME: Should the following two code lines operate on "split_lines"?
        to_remove = np.where(lines[0, :] == lines[1, :])[0]
        lines = np.delete(lines, to_remove, axis=1)

        self._decomposition["domain_boundary_points"] = old_2_new[
            self._decomposition["domain_boundary_points"]
        ]

        # Remove lines with the same start and end-point.
        # This can be caused by L-intersections, or possibly also if the two
        # endpoints are considered equal under tolerance tol.
        remove_line_ind = np.where(np.diff(lines_split[:2], axis=0)[0] == 0)[0]
        lines_split = np.delete(lines_split, remove_line_ind, axis=1)

        # We find the end points that are shared by more than one intersection
        intersections = self._find_intersection_points(lines_split)

        self._decomposition.update(
            {
                "points": pts_split,
                "edges": lines_split,
                "intersections": intersections,
                "domain": self.domain,
            }
        )

    def _find_intersection_points(self, lines: np.ndarray) -> np.ndarray:
        """Find intersection points for a set of lines.

        Parameters:
            lines: Lines defined as pairs of points.

        Returns:
            Numpy array containing the intersection between :attr:`lines`.

        """
        frac_id = np.ravel(lines[:2, lines[2] == GmshInterfaceTags.FRACTURE.value])
        _, frac_ia, frac_count = np.unique(frac_id, True, False, True)

        # In the case we have auxiliary points, do not create a 0d point in the case
        # where the point intersects a single fracture. In the case of multiple
        # fractures intersecting with an auxiliary point do consider the 0d.
        aux_id = np.logical_or(
            lines[2] == GmshInterfaceTags.AUXILIARY_LINE.value,
            lines[2] == GmshInterfaceTags.DOMAIN_BOUNDARY_LINE.value,
        )
        if np.any(aux_id):
            aux_id = np.ravel(lines[:2, aux_id])
            _, aux_ia, aux_count = np.unique(aux_id, True, False, True)

            # It can probably be done more efficiently, but currently we rarely use the
            # auxiliary points in 2d.
            for a in aux_id[aux_ia[aux_count > 1]]:
                # If a match is found decrease the frac_count only by one, this prevents
                # the multiple fracture case to be handled wrongly.
                frac_count[frac_id[frac_ia] == a] -= 1

        return frac_id[frac_ia[frac_count > 1]]

    def _insert_auxiliary_points(
        self,
        mesh_size_frac: Optional[float] = None,
        mesh_size_bound: Optional[float] = None,
        mesh_size_min: Optional[float] = None,
    ) -> None:
        """Insert auxiliary points.

        Parameters:
            mesh_size_frac: ``default=None``

                Mesh size close to the fractures.
            mesh_size_bound: ``default=None``

                Mesh size close to the external boundaries.
            mesh_size_min: ``default=None``

                Minimum allowable mesh size.

        """

        # Mesh size
        # Tag points at the domain corners
        logger.debug("Determine mesh size")
        tm = time.time()

        p = self._decomposition["points"]
        lines = self._decomposition["edges"]
        boundary_pt_ind = self._decomposition["domain_boundary_points"]

        mesh_size, pts_split, lines = tools.determine_mesh_size(
            p,
            lines,
            boundary_pt_ind,
            mesh_size_frac=mesh_size_frac,
            mesh_size_bound=mesh_size_bound,
            mesh_size_min=mesh_size_min,
        )

        logger.debug("Done. Elapsed time " + str(time.time() - tm))

        self._decomposition["points"] = pts_split
        self._decomposition["edges"] = lines
        self._decomposition["mesh_size"] = mesh_size

    def _set_mesh_size_without_auxiliary_points(
        self,
        mesh_size_frac: Optional[float] = None,
        mesh_size_bound: Optional[float] = None,
        mesh_size_min: Optional[float] = None,
    ) -> None:
        """Set the "vanilla" mesh size to points.

        No attempts at automatically determine the mesh size is done and no auxiliary
        points are inserted. Fracture points are given the :attr:`mesh_size_frac`
        mesh size and the domain boundary is given the :attr:`mesh_size_bound` mesh
        size. :attr:`mesh_size_min` is not used.

        Parameters:
            mesh_size_frac: ``default=None``

                Mesh size close to the fractures.
            mesh_size_bound: ``default=None``

                Mesh size close to the external boundaries.
            mesh_size_min: ``default=None``

                Minimum allowable mesh size.

        """
        # Gridding size
        # Tag points at the domain corners
        logger.debug("Determine mesh size")
        tm = time.time()

        boundary_pt_ind = self._decomposition["domain_boundary_points"]
        num_pts = self._decomposition["points"].shape[1]

        val = 1.0

        if mesh_size_frac is not None:
            val = mesh_size_frac
        # One value for each point to distinguish between val and val_bound.
        vals = val * np.ones(num_pts)
        if mesh_size_bound is not None:
            vals[boundary_pt_ind] = mesh_size_bound
        logger.debug("Done. Elapsed time " + str(time.time() - tm))
        self._decomposition["mesh_size"] = vals

    def impose_external_boundary(
        self,
        domain: Optional[pp.Domain] = None,
        add_domain_edges: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Constrain the fracture network to lie within a domain.

        Fractures outside the imposed domain will be deleted. The :attr:`domain` will
        be added to :attr:`_pts` and :attr:`_edges`, if ``add_domain_edges=True``.

        The domain boundary edges can be identified from :attr:`tags` using the key
        ``'boundary'``.

        Parameters:
            domain: ``default=None``

                Domain specification. If not provided, :attr:`domain` will be used.
                When the domain is given as a set of lines create the grid respecting
                the lines, we assume that these lines are ordered.
            add_domain_edges: ``default=True``

                Whether to include the boundary edges and points in the list of edges.

        Returns:
            Tuple with two elements.

                :obj:`numpy.ndarray`:

                    Array containing the indices of the fractures that have been kept.

                :obj:`numpy.ndarray`:

                    Array containing the indices of the fractures that have been
                    deleted, since they were outside the bounding box.

        """

        # Get min/max point of the domain. If no domain is given, a bounding box
        # based on `self.tol` will be imposed outside the fracture set
        if domain is None:
            # Sanity check
            if len(self.fractures) == 0:
                raise ValueError("No fractures given, domain cannot be imposed.")
            # Loop through the fracture list and retrieve the points
            x_pts = []
            y_pts = []
            for frac in self.fractures:
                # Append all start/end-points in the x-direction
                x_pts.append(frac.pts[0][0])
                x_pts.append(frac.pts[0][1])
                # Append all start/end-points in the y-direction
                y_pts.append(frac.pts[1][0])
                y_pts.append(frac.pts[1][1])
            # Get min/max points
            x_min = np.min(np.asarray(x_pts)) - 10 * self.tol
            x_max = np.max(np.asarray(x_pts)) + 10 * self.tol
            y_min = np.min(np.asarray(y_pts)) - 10 * self.tol
            y_max = np.max(np.asarray(y_pts)) + 10 * self.tol

            # Create the domain lines
            dom_p = np.array(
                [[x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max]]
            )

        elif domain.is_boxed:
            # If the domain is given, we know the min/max points
            x_min = domain.bounding_box["xmin"]
            x_max = domain.bounding_box["xmax"]
            y_min = domain.bounding_box["ymin"]
            y_max = domain.bounding_box["ymax"]

            # Create the domain lines
            dom_p = np.array(
                [[x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max]]
            )

        else:
            # Create the domain lines from the its polytopal representation
            # We suppose that the lines are ordered clockwise or counter-clockwise
            dom_p = np.hstack(domain.polytope)[:, ::2]

        # Define the lines of the domain boundary
        idx = np.arange(dom_p.shape[1])
        dom_lines = np.vstack((idx, np.roll(idx, -1)))

        # Constrain the edges to the domain
        p, e, edges_kept = pp.constrain_geometry.lines_by_polygon(
            dom_p, self._pts, self._edges
        )

        # Special case where an edge has one point on the boundary of the domain,
        # the other outside the domain. In this case the edge should be removed. The
        # edge will have been cut so that the endpoints coincide. Look for such edges
        _, _, n2o = pp.array_operations.uniquify_point_set(p, self.tol)
        reduced_edges = n2o[e]
        not_point_edge = np.diff(reduced_edges, axis=0).ravel() != 0

        # The point involved in point edges may be superfluous in the description of
        # the fracture network; this we will deal with later. For now, simply remove
        # the point edge.
        e = e[:, not_point_edge]
        edges_kept = edges_kept[not_point_edge]

        edges_deleted = np.setdiff1d(np.arange(self._edges.shape[1]), edges_kept)

        # Define boundary tags. Set False to all existing edges (after cutting those
        # outside the boundary).
        boundary_tags = self.tags.get("boundary", np.zeros(e.shape[1], dtype=bool))

        if add_domain_edges:
            num_p = p.shape[1]
            # Add the domain boundary edges and points
            self._edges = np.hstack((e, dom_lines + num_p))
            self._pts = np.hstack((p, dom_p))
            # preserve the tags
            for key, value in self.tags.items():
                self.tags[key] = np.hstack(
                    (
                        value[edges_kept],
                        GmshInterfaceTags.DOMAIN_BOUNDARY_LINE.value
                        * np.ones(dom_lines.shape[1], dtype=int),
                    )
                )

            # Define the new boundary tags
            new_boundary_tags = np.hstack(
                [boundary_tags, np.ones(dom_lines.shape[1], bool)]
            )
            self.tags["boundary"] = np.array(new_boundary_tags)

            self._decomposition["domain_boundary_points"] = num_p + np.arange(
                dom_p.shape[1], dtype=int
            )
        else:
            self.tags["boundary"] = boundary_tags
            self._decomposition["domain_boundary_points"] = np.empty(0, dtype=int)
            self._edges = e
            self._pts = p

        self.fractures = [f for fi, f in enumerate(self.fractures) if fi in edges_kept]

        self.bounding_box_imposed = True
        return edges_kept, edges_deleted

    def _snap_fracture_set(
        self,
        pts: np.ndarray,
        snap_tol: float,
        termination_tol: float = 1e-2,
        max_iter: int = 100,
    ) -> tuple[np.ndarray, bool]:
        """Snap vertexes of a set of fracture lines embedded in 2d domains.

        The aim is to remove small distances between lines and vertexes are removed.

        This is intended as a utility function to preprocess a fracture network
        before meshing. The function may change both connectivity and orientation of
        individual fractures in the network. Specifically, fractures that almost form
        a T-intersection (or L), may be connected, while X-intersections with very
        short branches may be truncated to T-intersections.

        The modification snaps vertexes to the closest point on the adjacent line.
        This will in general change the orientation of the fracture with the snapped
        vertex. The alternative, to prolong the fracture along its existing
        orientation, may result in very long fractures for almost intersecting lines.
        Depending on how the fractures are ordered, the same point may need to be
        snapped to a segment several times in an iterative process.

        The algorithm is *not* deterministic, in the sense that if the ordering of
        the fractures is permuted, the snapped fracture network will be slightly
        different.

        Parameters:
            pts: ``shape=(2, num_points)``

                Array of start and endpoints for fractures.
            snap_tol: Snapping tolerance. Distances below this will be snapped.
            termination_tol: ``default=1e-2``

                Minimum point movement needed for the iterations to continue.
            max_iter: ``default=100``

                Maximum number of iterations.

        Returns:
            Tuple with two elements.

            :obj:`numpy.ndarray`: ``shape=(2, num_points)``

                Copy of the point array, with modified point coordinates.

            :obj:`bool`:

                True if the iterations converged within allowed number of iterations.

        """
        pts_orig = pts.copy()
        edges = self._edges
        counter = 0
        while counter < max_iter:
            pn = pp.constrain_geometry.snap_points_to_segments(pts, edges, tol=snap_tol)
            diff = np.max(np.abs(pn - pts))
            logger.debug("Iteration " + str(counter) + ", max difference" + str(diff))
            pts = pn
            if diff < termination_tol:
                break
            counter += 1

        if counter < max_iter:
            logger.debug(
                "Fracture snapping converged after " + str(counter) + " iterations"
            )
            logger.debug("Maximum modification " + str(np.max(np.abs(pts - pts_orig))))
            return pts, True
        else:
            logger.warning("Fracture snapping failed to converge")
            logger.warning("Residual: " + str(diff))
            return pts, False

    def _snap_to_boundary(self, snap_tol: float) -> None:
        """Snap points to the domain boundary.

        Note:
            This function modifies ``self._pts``.

        Parameters:
            snap_tol: Tolerance. Internal points which are a distance ``d < snap_tol``
                away from the boundary will be snapped to the boundary.

        """
        is_bound = self.tags["boundary"]
        # interior edges
        interior_edges = self._edges[:2, np.logical_not(is_bound)]
        # Index of interior points
        interior_pt_ind = np.unique(interior_edges)
        # Snap only to boundary edges (snapping of fractures internally is another
        # operation, see self._snap_fracture_set()
        bound_edges = self._edges[:2, is_bound]

        # Use function to snap the points
        snapped_pts = pp.constrain_geometry.snap_points_to_segments(
            self._pts, bound_edges, snap_tol, p_to_snap=self._pts[:, interior_pt_ind]
        )

        # Replace the
        self._pts[:, interior_pt_ind] = snapped_pts

    def _edges_overlapping_boundary(self, tol: float) -> np.ndarray:
        """Obtain edges that overlap the exterior boundary.

        Intended usage is the removal of such edges in
        :meth:`~porepy.fracs.fracture_network_2d.FractureNetwork2d.prepare_for_gmsh`.

        Parameters:
            tol: Geometric tolerance to decide whether an edge is overlapping the
                exterior boundary.

        Returns:
            Boolean array of ``shape(self._edges.shape[1], )``. A ``True`` element
            implies that the edge is overlapping the exterior boundary.

        """

        # access array for boundary and internal edges
        is_bound = self.tags["boundary"]
        is_internal = np.logical_not(is_bound)

        # boundary edges by points
        start_bound_pts = self._pts[:, self._edges[0, is_bound]]
        end_bound_pts = self._pts[:, self._edges[1, is_bound]]

        overlap = np.zeros(self._edges.shape[1], dtype=bool)
        # loop on all the internal edges and check whether they should be removed
        for ind in np.where(is_internal)[0]:
            # define the start and end point of the current internal edge
            start = self._pts[:, self._edges[0, ind]]
            end = self._pts[:, self._edges[1, ind]]
            # check if the current internal edge is overlapping the boundary
            overlap[ind] = pp.distances.segment_overlap_segment_set(
                start, end, start_bound_pts, end_bound_pts, tol=tol
            )

        return overlap

    """
    End of methods related to meshing
    """

    def constrain_to_domain(
        self, domain: Optional[pp.Domain] = None
    ) -> FractureNetwork2d:
        """Constrain the fracture network to lay within a specified domain.

        Fractures that cross the boundary of the domain will be cut to lay within the
        boundary. Fractures that lay completely outside the domain will be dropped
        from the constrained description.

        Parameters:
            domain: ``default: None``

                Domain specification. If not given, :attr:`domain` will be used.

        Returns:
            Constrained, 2D fracture network.

        """
        if domain is None:
            domain = self.domain
        assert isinstance(domain, pp.Domain)

        p_domain = self._bounding_box_to_points(domain.bounding_box)

        p, e, _ = pp.constrain_geometry.lines_by_polygon(
            p_domain, self._pts, self._edges
        )
        fracs = pts_edges_to_linefractures(p, e)

        return FractureNetwork2d(fracs, domain, self.tol)

    def _bounding_box_to_points(self, box: dict[str, pp.number]) -> np.ndarray:
        """Helper function to convert a bounding box into a point set.

        TODO:
            Consider moving this method to :class:`~porepy.geometry.domain.Domain`.

        Parameters:
            box: Bounding box dictionary, containing the keywords ``xmin, xmax, ymin,
                ymax``.

        Returns:
            Array of ``shape=(2, 4)`` with the vertexes coordinates of the bounding box.

        """

        p00 = np.array([box["xmin"], box["ymin"]]).reshape((-1, 1))
        p10 = np.array([box["xmax"], box["ymin"]]).reshape((-1, 1))
        p11 = np.array([box["xmax"], box["ymax"]]).reshape((-1, 1))
        p01 = np.array([box["xmin"], box["ymax"]]).reshape((-1, 1))
        point_set = np.hstack((p00, p10, p11, p01))

        return point_set

    # Methods for copying fracture network
    def copy(self) -> FractureNetwork2d:
        """Create a deep copy of the fracture network.

        The method will create a deep copy of all fractures and of the domain.

        Note:
            If the fractures have had extra points imposed as part of a meshing
            procedure, these will be included in the copied fractures.

        See also:

            - :meth:`~snapped_copy`
            - :meth:`~copy_with_split_intersections`

        Returns:
            Deep copy of this fracture network.

        """
        if len(self.fractures) == 0:
            fractures_new = None
        else:
            fractures_new = copy.deepcopy(self.fractures)

        domain = self.domain
        if domain is not None:
            if domain.is_boxed:
                box = copy.deepcopy(domain.bounding_box)
                domain = pp.Domain(bounding_box=box)
            else:
                polytope = domain.polytope.copy()
                domain = pp.Domain(polytope=polytope)

        fn = FractureNetwork2d(fractures_new, domain, self.tol)
        fn.tags = self.tags.copy()

        return fn

    def snapped_copy(self, tol: float) -> FractureNetwork2d:
        """Create a snapped copy of this fracture network.

        The definition of points is modified so that short branches are removed, and
        almost-intersecting fractures become intersecting.

        See also:
            - :meth:`~copy`
            - :meth:`~copy_with_split_intersections`

        Parameters:
            tol: Threshold for geometric modifications. Points and segments closer
                than the threshold may be modified.

        Returns:
            A snapped copy of the fracture network with modified point coordinates.

        """
        # We will not modify the original fractures
        p = self._pts.copy()
        e = self._edges.copy()

        # Prolong
        p = pp.constrain_geometry.snap_points_to_segments(p, e, tol)
        fracs = pts_edges_to_linefractures(p, e)

        return FractureNetwork2d(fracs, self.domain, self.tol)

    def copy_with_split_intersections(
        self, tol: Optional[float] = None
    ) -> FractureNetwork2d:
        """Create a copy of this network with all fracture intersections removed.

        See also:

            - :meth:`~copy`
            - :meth:`~snapped_copy`

        Parameters:
            tol: ``default=None``

                Tolerance used in geometry computations when splitting fractures. If
                not given, :attr:`~tol`` will be used.

        Returns:
            New fracture network, where all intersection points are added so that
            the set only contains non-intersecting branches.

        """
        if tol is None:
            tol = self.tol

        # FIXME: tag_info may contain useful information if segments are intersecting.
        # Since the function called in general can return 3 or 4 values (but we know
        # it will return 4 here), we first store the returned values in a tuple, and
        # then unpack the tuple into the individual variables.
        result = pp.intersections.split_intersecting_segments_2d(
            self._pts, self._edges, tol=self.tol, return_argsort=True
        )
        assert len(result) == 4, "Unexpected number of return values"
        p, e, argsort, tag_info = result  # type: ignore
        # map the tags
        tags = {}
        for key, value in self.tags.items():
            tags[key] = value[argsort]

        fracs = pts_edges_to_linefractures(p, e)
        fn = FractureNetwork2d(fracs, self.domain, tol=tol)
        fn.tags = tags

        return fn

    # Utility functions below here

    def num_frac(self) -> int:
        """
        Returns:
            Number of fractures in this fracture network.

        """
        return self._edges.shape[1]

    def _remove_orphan_pts(self) -> np.ndarray:
        """Remove points that are not part of any edge.

        Note that the numerations are modified accordingly.

            Returns:
                Array of integers, containing the id of the new points.

        """

        pts_id = np.unique(self._edges)
        all_pts_id = np.arange(self._pts.shape[1])

        # determine the orphan points
        to_keep = np.ones(all_pts_id.size, dtype=bool)
        to_keep[np.setdiff1d(all_pts_id, pts_id, assume_unique=True)] = False

        # create the map between the old and new
        new_pts_id = -np.ones(all_pts_id.size, dtype=np.int32)
        new_pts_id[to_keep] = np.arange(pts_id.size)

        # update the edges numeration
        self._edges = new_pts_id[self._edges]
        # update the points
        self._pts = self._pts[:, pts_id]

        return new_pts_id

    def plot(self, **kwargs) -> None:
        """Plot the fracture network.

        The function passes this fracture set to
        :meth:`~porepy.viz.fracture_visualization.plot_fractures`

        Parameters:
            **kwargs: Keyword arguments to be passed on to
                :obj:`~matplotlib.pyplot.plot`.

        """
        pp.plot_fractures(self._pts, self._edges, domain=self.domain, **kwargs)

    def to_csv(self, file_name: Path, with_header: bool = True) -> None:
        """Save the 2D network on a CSV file with comma as separator.

        The format is ``FID, START_X, START_Y, END_X, END_Y``, where ``FID`` is the
        fracture ID, and ``START_X, ..., END_Y`` are the point coordinates.

        Warning:
            If ``file_name`` is already present, it will be overwritten without
            prompting any warning.

        Parameters:
            file_name: Name of the CSV file.
            with_header: ``default=True``

                Flag for writing headers for the five columns in the first row.

        """

        with open(file_name, "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            if with_header:
                header = ["# FID", "START_X", "START_Y", "END_X", "END_Y"]
                csv_writer.writerow(header)
            # write all the fractures
            for edge_id, edge in enumerate(self._edges.T):
                data = [edge_id]
                data.extend(self._pts[:, edge[0]])
                data.extend(self._pts[:, edge[1]])
                csv_writer.writerow(data)

    def to_file(
        self, file_name: Path, data: Optional[dict[str, np.ndarray]] = None, **kwargs
    ) -> None:
        """Export the fracture network to file.

        The file format is given as a ``kwargs``, by default ``vtu`` will be used.
        The writing is outsourced to meshio, thus the file format should be supported
        by that package.

        The fractures are treated as lines, with no special treatment of intersections.

        Fracture numbers are always exported (1-offset). In addition, it is possible
        to export additional data, as specified by the keyword-argument data.

        Parameters:
            file_name: Name of the target file.
            data: ``default=None``

             Data associated with the fractures. The values in the dictionary should
             be numpy arrays. 1d and 3d data is supported. Fracture numbers are
             always exported.

            **kwargs: The following arguments can be given:

                - ``'binary'`` (:obj:`bool`): ``default=True``

                    Whether to use binary export format.

                - ``'fracture_offset'`` (:obj:`int`): ``default=1``

                    Used to define the offset for a fracture id.

                - ``'folder_name'`` (:obj:`Path`): ``default=Path("")``

                    Path to save the file.

                - ``'extension'`` (:obj:`str`): ``default=".vtu"``

                    File extension.

        """
        if data is None:
            data = {}

        binary: bool = kwargs.pop("binary", True)
        fracture_offset: int = kwargs.pop("fracture_offset", 1)
        extension: str = kwargs.pop("extension", ".vtu")
        folder_name: Path = Path(kwargs.pop("folder_name", ""))

        if kwargs:
            msg = "Got unexpected keyword argument '{}'"
            raise TypeError(msg.format(kwargs.popitem()[0]))

        # Make sure the suffix is correct
        file_name = file_name.with_suffix(extension)

        # in 1d we have only one cell type
        cell_type = "line"

        # cell connectivity information
        meshio_cells = np.empty(1, dtype=object)
        meshio_cells[0] = meshio.CellBlock(cell_type, self._edges.T)

        # prepare the points
        meshio_pts = self._pts.T
        # make points 3d
        if meshio_pts.shape[1] == 2:
            meshio_pts = np.hstack((meshio_pts, np.zeros((meshio_pts.shape[0], 1))))

        # Cell-data to be exported is at least the fracture numbers
        meshio_cell_data = {}
        meshio_cell_data["fracture_number"] = [
            fracture_offset + np.arange(self._edges.shape[1])
        ]

        # process the
        for key, val in data.items():
            if val.ndim == 1:
                meshio_cell_data[key] = [val]
            elif val.ndim == 2:
                meshio_cell_data[key] = [val.T]

        meshio_grid_to_export = meshio.Mesh(
            meshio_pts, meshio_cells, cell_data=meshio_cell_data
        )
        path = folder_name / file_name
        meshio.write(path, meshio_grid_to_export, binary=binary)

    def __str__(self):
        s = (
            f"Two-dimensional fracture network with {str(self.num_frac())} line "
            f"fractures.\n"
        )
        if self.domain is not None:
            s += f"The domain is a {(str(self.domain)).lower()}"
        return s

    def __repr__(self):
        return self.__str__()
