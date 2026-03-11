"""Module contains class for representing a fracture network in a 2d domain."""

from __future__ import annotations

import copy
import csv
import itertools
import logging
import multiprocessing
import time
from pathlib import Path
from typing import Optional, cast

import gmsh
import meshio
import numpy as np
from matplotlib import pyplot as plt

import porepy as pp
import porepy.fracs.simplex
from porepy.fracs import tools

from .fracture_network import (
    FractureNetwork,
    GmshPointIdentifier,
    MeshSizeComputer,
    MeshSizeControlPointInserter,
)
from .gmsh_interface import PhysicalNames

logger = logging.getLogger(__name__)


class FractureNetwork2d(FractureNetwork):
    """Representation of a set of line fractures in a 2D domain.

    The fractures are represented by line fracture objects (see
    :class:`~porepy.fracs.line_fracture.LineFracture`).

    Polyline fractures are currently not supported.

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
        super().__init__(nd=2, domain=domain, tol=tol)

        self.fractures: list[pp.LineFracture] = []
        """List of fractures forming the network."""
        # Populate fracture list and assign indices.
        if fractures is not None:
            for index, f in enumerate(fractures):
                self.fractures.append(f)
                f.index = index

    def domain_to_gmsh(self) -> int:
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
            bb = domain.bounding_box
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
            # Now build the domain by first adding the lines of the boundary, then
            # define them to be a loop, and then define a surface inside that loop.
            lines = [
                # i + 1 is okay here, since pt_tags was augmented above.
                gmsh.model.occ.addLine(pt_tags[i], pt_tags[i + 1])
                for i in range(len(pts))
            ]
            line_loop = gmsh.model.occ.addCurveLoop(lines)
            domain_tag = gmsh.model.occ.addPlaneSurface([line_loop])

        return domain_tag

    def mesh(
        self,
        mesh_args: dict[str, float],
        file_name: Optional[Path] = None,
        constraints: Optional[np.ndarray] = None,
        dfn: bool = False,
        **kwargs,
    ) -> pp.MixedDimensionalGrid:
        """Generate a mixed-dimensional grid by meshing the fracture network.

        Parameters:
            mesh_args: Dictionary with mesh size parameters. See
                :class:`~porepy.fracs.fracture_network.MeshSizeComputer` for details.
            file_name: Path to the output Gmsh .msh file.
            constraints: Numpy array with indices of fractures to be treated as
                constraints during meshing. The indices refer to the ordering of
                fractures in the fracture network. If ``None``, no constraints are
                applied.
            dfn: If ``True``, a discrete fracture network (DFN) style meshing is
                performed, where only the fractures are meshed (no volume mesh is
                created).
            **kwargs: Additional keyword arguments passed to Gmsh.

        Returns:
            A :class:`~porepy.meshing.mixed_dimensional_grid.MixedDimensionalGrid`
            representing the meshed fracture network.

        """
        gmsh.initialize()
        # Prepare the mesh inputs. Also set some Gmsh options, see the method for
        # details.
        file_name, constraints = self._prepare_mesh_inputs(
            file_name, constraints, **kwargs
        )

        # Helper class to keep track of mesh size computations.
        mesh_size_computer = MeshSizeComputer(mesh_args)

        domain_tag = self.domain_to_gmsh()
        fracture_tags = self.fractures_to_gmsh()
        gmsh.model.occ.synchronize()
        # STEP 1: Insert mesh size control points on fractures and boundaries.
        mesh_size_points = self._insert_mesh_size_control_points(mesh_size_computer)
        gmsh.model.occ.synchronize()

        # STEP 2: Impose the domain boundary and process fracture intersections.
        (
            intersection_points,
            isect_mapping,
            constraints,
            gmsh_to_porepy_fracture_ind_map,
            mesh_size_points,
        ) = self._impose_boundary_process_intersections(
            fracture_tags, domain_tag, constraints, mesh_size_points
        )
        gmsh.model.occ.synchronize()

        # Write the .geo_unrolled file.
        gmsh.write(str(file_name.with_suffix(".geo_unrolled")))

        # STEP 3: Set the mesh sizes.
        self._set_background_mesh_field(
            self._set_mesh_size_fields(
                mesh_size_computer, mesh_size_points, restrict_to_fractures=True
            )
        )
        gmsh.model.occ.synchronize()

        # STEP 4: Set physical names.
        self._set_physical_names(
            intersection_points,
            isect_mapping,
            gmsh_to_porepy_fracture_ind_map,
            set(constraints),
        )

        # STEP 5: Create a gmsh mesh.
        gmsh.model.mesh.generate(1)
        if not dfn:
            # Remove the 1d mesh fields, set new ones, then generate the 2d mesh.
            for field in gmsh.model.mesh.field.list():
                gmsh.model.mesh.field.remove(field)

            self._set_background_mesh_field(
                self._set_mesh_size_fields(
                    mesh_size_computer, mesh_size_points, restrict_to_fractures=False
                )
            )
            gmsh.model.mesh.generate(2)

        # Delete the file 'file_name' if it exists, and write the new mesh to
        # 'file_name'. This seems to be necessary to run tests on GH actions.
        if file_name.exists():
            file_name.unlink()
        gmsh.write(str(file_name))

        # Report mesh quality metrics.
        if self._extra_meshing_args["plot_mesh_quality_metrics"]:
            self.mesh_quality_metrics()

        # STEP 6: Create list of grids and assemble mixed-dimensional grid.
        if dfn:
            subdomains = porepy.fracs.simplex.line_grid_from_gmsh(
                file_name, constraints=constraints
            )

        else:
            subdomains = porepy.fracs.simplex.triangle_grid_from_gmsh(
                file_name, constraints=constraints
            )
        gmsh.finalize()
        # Assemble all subdomains in mixed-dimensional grid.
        return pp.meshing.subdomains_to_mdg(subdomains, **kwargs)

    def _impose_boundary_process_intersections(
        self,
        fracture_tags: list[int],
        domain_tag: int,
        constraints: np.ndarray,
        mesh_size_points: dict[int, list[tuple[np.ndarray, float]]],
    ):
        """Impose the domain boundary and process fracture intersections.

        Helper function for mesh processing.

        Parameters:
            fracture_tags: List of gmsh tags representing the fractures in the network.
            domain_tag: Gmsh tag representing the domain.
            constraints: List of indices of fractures that are constraints. This refers
                to the index in the input fracture list, not the gmsh tag.
            mesh_size_points: Dictionary mapping gmsh fracture tags to lists of mesh
                size control points associated with that fracture.

        Returns:
            A tuple containing:
            - A list of gmsh tags representing intersection points.
            - isect_mapping updated so that fractures that have been split are
              identified as separate but related objects in the Gmsh representation.
            - Updated constraints array, still referring to input fracture indices, but
              with indices of removed fractures eliminated and indices shifted
              accordingly.
            - Updated inverse fracture tag map, mapping gmsh fracture tags to input
              fracture indices (with removed fractures eliminated).
            - Updated mesh_size_points dictionary.

        """
        # The method goes through the following stages:
        # 1. Identify fractures that are fully or partially outside the domain, and
        #    remove/truncate them accordingly. Update constraints and other data
        #    structures accordingly.
        # 2. Make gmsh compute intersections between fractures, using the domain as a
        #    secondary object to ensure embedding. Update data structures accordingly.
        # 3. Identify unique intersection points, excluding those on the domain
        #    boundary.
        # 4. Return the identified intersection points and updated data structures.
        #
        # A comparison with the 3d version of this method will show that though the
        # steps are similar, there are significant differences in implementation. The
        # two main causes for this are:
        # 1. The 3d version needs to consider planes and their intersections (which
        #    could be both lines and points), while the 2d version only deals with
        #    points. This makes the 2d version significantly simpler.
        # 2. There are differences in how Gmsh processes what is essentially the same
        #    operations in 2d and 3d, leading to different needs for book-keeping and
        #    data structure updates. Whether the true culprit here is Gmsh or EK is
        #    uncertain.

        # STEP 1: Identify fractures fully/partially outside the domain and
        # remove/truncate them.
        new_fractures = {}
        removed_fractures = []

        for ind, fracture_tag in enumerate(fracture_tags):
            if domain_tag < 0:
                # No domain is defined, so no fracture can be outside the domain. We
                # could have done this check outside the loop, but the performance
                # impact is negligible, so it is preferrable to avoid a new if-statement
                # with indentation.
                continue

            # According to gmsh documentation (v4.14), the function intersect should be
            # able to identify fractures that do not intersect with the domain. The
            # expected result is that the map from the old fracture to the new one, that
            # is, the second return variable from the call to intersect, is empty.
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
            distance = gmsh.model.occ.getDistance(
                self.nd - 1, fracture_tag, self.nd, domain_tag
            )[0]
            if distance > self._tol or self._entity_on_domain_boundary(
                1, [fracture_tag]
            ):
                # The fracture is either fully outside the domain or fully embedded on
                # the domain boundary. It will be deleted.
                removed_fractures.append(ind)
                continue

            # The fracture is either fully or partly inside the domain. We call
            # intersect to truncate the fracture if necessary.
            truncated_fracture, _ = gmsh.model.occ.intersect(
                [(self.nd - 1, fracture_tag)],
                [(self.nd, domain_tag)],
                removeTool=False,
                removeObject=False,
            )
            if len(truncated_fracture) > 0 and truncated_fracture[0][1] != fracture_tag:
                # The fracture was partly outside the domain. It will be replaced.
                new_fractures[ind] = truncated_fracture[0]

        # Remove the fractures from the gmsh representation. Recursive is critical here,
        # or else the boundary of 'fracture' will continue to be present.
        for ind in removed_fractures:
            gmsh.model.occ.remove([(self.nd - 1, fracture_tags[ind])], recursive=True)
        # Also update the constraints: Each fracture removal in effect shifts the
        # indices, but only for those whose index is higher than the removed index.
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
            gmsh.model.occ.remove(
                [(self.nd - 1, fracture_tags[old_fracture])], recursive=True
            )
            fracture_tags[old_fracture] = new_fracture[1]
        gmsh.model.occ.synchronize()

        # Remove from fracture_tags those indices that are present in removed_fractures.
        fracture_tags = [
            ft for i, ft in enumerate(fracture_tags) if i not in removed_fractures
        ]
        # Get tags of the domain boundaries.
        if domain_tag < 0:
            boundary_tags = []
        else:
            boundary_tags = [
                t for _, t in gmsh.model.get_boundary([(self.nd, domain_tag)])
            ]

        # Mapping from the new fracture tags (gmsh assigned) to the input fractures.
        gmsh_to_porepy_fracture_ind_map = {
            i: counter for counter, i in enumerate(fracture_tags)
        }

        # STEP 2: Make gmsh calculate the intersections between fractures, using the
        # domain as a secondary object (the latter will by magic ensure that the
        # fractures are embedded in the domain, hence the mesh will conform to the
        # fractures).
        line_tags_new = fracture_tags + boundary_tags
        isect_mapping = self._fragment_fractures(line_tags_new, domain_tag)

        # During intersection removal, gmsh will add intersection points and replace the
        # fractures with non-intersecting polylines (example: Two fractures intersecting
        # as a cross become four fractures with a common point). Furthermore, gmsh may
        # have retagged fractures, boundaries and other entities. To keep track of these
        # updates, the below for-loop takes action on three points:
        # 1. Update the keys (gmsh tags of fracture and boundary lines) for the mesh
        #    size control points.
        # 2. Update the inverse mapping from gmsh fracture tags to input fractures to
        #    work with the new gmsh fracture tags.
        # 3. Identify the boundary points of all fracture segments, as a pair of the
        #    gmsh indices of the points and the input fracture index. This will be used
        #    to identify intersection points later on.
        # Data structures to be filled.
        updated_mesh_size_points = {}
        updated_fracture_tag_map = {}
        boundary_points_fracture_indices = []
        for fi, old_fracture in enumerate(isect_mapping):
            if len(old_fracture) == 0:
                # EK is not sure when this happens, but it does occasionally. Skip it.
                continue

            if old_fracture[0][0] == self.nd:
                # This is the domain. Skip it.
                continue

            # Get hold of the gmsh tag used to represent this fracture before
            # intersection removal.
            old_gmsh_tag = line_tags_new[fi]
            if old_gmsh_tag in boundary_tags:
                # This is part of the boundary. Skip it.
                continue

            # This may be a constraint fracture, in which case there is no need to
            # work with intersection removal.
            frac_ind = gmsh_to_porepy_fracture_ind_map[old_gmsh_tag]

            for segment in old_fracture:
                if old_gmsh_tag in mesh_size_points:
                    # Update the mesh size points for the new segments.
                    updated_mesh_size_points[segment[1]] = mesh_size_points[
                        old_gmsh_tag
                    ]
                pt_index = gmsh.model.get_boundary([segment])

                if fi not in constraints:
                    # If this is not a constraint, collect the boundary points for
                    # intersection identification.
                    for pt in pt_index:
                        boundary_points_fracture_indices.append((pt[1], frac_ind))

                updated_fracture_tag_map[segment[1]] = frac_ind

        # The mesh size and fracture tag map can be updated by reassignment.
        mesh_size_points = updated_mesh_size_points
        gmsh_to_porepy_fracture_ind_map = updated_fracture_tag_map

        # STEP 3: Find the unique boundary points and obtain a mapping from the full set
        # of boundary points to the unique ones.
        unique_boundary_points = np.unique(boundary_points_fracture_indices, axis=0)

        # Finally, we need to uniquify the intersection points, since the same point
        # will have been identified in at least two old fractures.
        if unique_boundary_points.size > 0:
            # Count the number of occurrences of each unique boundary point. Points that
            # occur more than once will be intersections.
            all_intersection_points = np.where(
                np.bincount(unique_boundary_points[:, 0]) > 1
            )[0]

        else:
            # No intersections, simply create an empty list.
            all_intersection_points = np.array([], dtype=int)

        # Filter away those points that lie on the domain boundary.
        unique_intersection_points = [
            pt
            for pt in all_intersection_points
            if not self._entity_on_domain_boundary(0, [pt])
        ]
        return (
            unique_intersection_points,
            isect_mapping,
            constraints,
            gmsh_to_porepy_fracture_ind_map,
            mesh_size_points,
        )

    def _set_mesh_size_fields(
        self,
        mesh_size_computer: MeshSizeComputer,
        mesh_size_points: dict[int, list[tuple[np.ndarray, float]]],
        restrict_to_fractures: bool,
    ) -> list:
        """Given a mesh size computer and a set of mesh control points with distance
        information, assign mesh size fields in the Gmsh representation.

        Parameters:
            mesh_size_computer: Object that stores and processes mesh size information.
            mesh_size_points: Dictionary that maps points (identified by their gmsh
                tags) to the point coordinates and the distance to the nearest object.

        Returns:
            List of gmsh fields (to be used by Gmsh).

        """
        ### Get hold of lines representing fractures and boundaries.
        domain_entities = gmsh.model.get_entities(2)
        boundaries = gmsh.model.get_boundary(
            [(self.nd, tag) for _, tag in domain_entities]
        )

        line_tags = set(tag for _, tag in gmsh.model.getEntities(self.nd - 1))
        boundary_tags = set(tag for _, tag in boundaries)

        # Storage of the (gmsh tags for) mesh size fields.
        gmsh_fields = []
        # Object that maps from point coordinates to gmsh point indices.
        gmsh_point_finder = GmshPointIdentifier()

        # Ensure that the mesh size points are unique.
        self._uniquify_mesh_size_dictionary(mesh_size_points)
        # For fractures or boundaries with no information, we assign an empty list.
        mesh_size: dict[int, list[tuple[np.ndarray, float]]] = {
            tag: [] for tag in line_tags
        }
        mesh_size.update(mesh_size_points)

        for line, info in mesh_size.items():
            extra_points = (
                np.array([d[0] for d in info]).T if len(info) > 0 else np.empty((3, 0))
            )
            if extra_points.size == 0:
                # If there are no mesh size control points, we continue to the next
                # line.
                continue

            # Uniquify the points. As a threshold for point uniqueness we use half of
            # the minimum of the fracture mesh size and the fracture length.
            end_points = np.array(
                [
                    gmsh.model.occ.get_bounding_box(0, p[1])[:3]
                    for p in gmsh.model.get_boundary([(1, line)], combined=False)
                ]
            ).T
            length = np.linalg.norm(end_points[:, 1] - end_points[:, 0])
            tol = np.minimum(length, mesh_size_computer.h_fracture()) / 2
            points, _, ind_map = pp.array_operations.uniquify_point_set(
                extra_points, tol=tol
            )
            # Distance to other objects for each point, as computed previously. We
            # assign h_frac for intersections (d==0), since no refinement is needed just
            # because this is an intersection point (if it is an intersection with a bad
            # angle, this should be picked up by a close point on another line).

            # Ignore a typing error here; the type checker does not understand that the
            # numpy array will consist of floats.
            other_object_distances_all = np.hstack(  # type: ignore[call-overload]
                (
                    np.array(
                        [
                            d[1]
                            if d[1] > self._tol
                            else mesh_size_computer.h_fracture()
                            for d in info
                        ]
                    )
                )
            )
            # Reduce to one distance per unique point, picking the minimum distance if
            # multiple distances were associated with the same geometric point.
            other_object_distances = []
            for i in range(points.shape[1]):
                inds = ind_map == i
                min_dist = np.min(other_object_distances_all[inds])
                other_object_distances.append(min_dist)

            if points.size > 0:
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
            dist = np.minimum(np.asarray(other_object_distances), min_dist_point)

            # Assign mesh sizes based on the distances.
            gmsh_fields += self._assign_distance_based_mesh_size_field(
                line,
                points,
                dist,
                mesh_size_computer,
                gmsh_point_finder,
                line in boundary_tags,
                restrict_to_fractures,
            )

        # Assign uniform mesh size fields to all fractures and boundaries. This will
        # kick in on parts of fractures and boundaries where no close points were
        # identified.
        gmsh_fields += self._set_uniform_mesh_field(
            list(mesh_size.keys()),
            mesh_size_computer,
            boundary_tags,
            restrict_to_fractures,
        )

        return gmsh_fields

    def _set_physical_names(
        self,
        intersection_points: list[int],
        isect_mapping: list,
        gmsh_to_porepy_fracture_ind_map: dict,
        constraints: set,
    ):
        # Collect intersection points, fractures, and domain in physical groups in gmsh.
        # Intersection points can be dealt with right away.
        for i, pt in enumerate(intersection_points):
            gmsh.model.addPhysicalGroup(
                self.nd - 2,
                [pt],
                -1,
                f"{PhysicalNames.FRACTURE_INTERSECTION_POINT.value}{i}",
            )

        gmsh.model.occ.synchronize()

        # Since fractures may have been split at intersection points, we need to collect
        # all the segments (found in isect_mapping) into a single physical group.
        fracture_to_line = self._subfracture_to_fracture_mapping(
            isect_mapping, gmsh_to_porepy_fracture_ind_map
        )

        for fi, segments in fracture_to_line.items():
            if fi in constraints:
                gmsh.model.addPhysicalGroup(
                    self.nd - 1,
                    segments,
                    -1,
                    f"{PhysicalNames.AUXILIARY_LINE.value}{fi}",
                )
            else:
                gmsh.model.addPhysicalGroup(
                    self.nd - 1, segments, -1, f"{PhysicalNames.FRACTURE.value}{fi}"
                )

        if self.domain is not None:
            # It turns out that if fractures split the domain into disjoint parts, gmsh
            # may choose to redefine the domain as the sum of these parts. Therefore, we
            # redefine the domain tags here, using all volumes in the model.
            domain_tags = [entity[1] for entity in gmsh.model.get_entities(self.nd)]

            gmsh.model.addPhysicalGroup(
                self.nd, domain_tags, -1, f"{PhysicalNames.DOMAIN.value}"
            )

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
            fractures_new = []
        else:
            fractures_new = copy.deepcopy(self.fractures)

        fracs = [cast(pp.LineFracture, frac) for frac in fractures_new]

        domain = self.domain
        if domain is not None:
            if domain.is_boxed:
                box = copy.deepcopy(domain.bounding_box)
                domain = pp.Domain(bounding_box=box)
            else:
                polytope = domain.polytope.copy()
                domain = pp.Domain(polytope=polytope)

        fn = FractureNetwork2d(fracs, domain, self._tol)

        return fn

    # Utility functions below here

    def plot(self, **kwargs) -> None:
        """Plot the fracture network.

        The function passes this fracture set to
        :meth:`~porepy.viz.fracture_visualization.plot_fractures`

        Parameters:
            **kwargs: Keyword arguments to be passed on to
                :obj:`~matplotlib.pyplot.plot`.

        """
        fracs = [cast(pp.LineFracture, frac) for frac in self.fractures]
        pp.plot_fractures(
            *_linefractures_to_pts_and_edges(fracs), domain=self.domain, **kwargs
        )

    def to_csv(
        self,
        file_name: Path,
        write_header: bool = True,
    ) -> None:
        """Save the 2D network on a CSV file with comma as separator.

            The format is ``START_X, START_Y, END_X, END_Y``, where  ``START_X, ...,
            END_Y`` are the point coordinates.

        Warning:
            If ``file_name`` is already present, it will be overwritten without
            prompting any warning.

        Parameters:
            file_name: Name of the CSV file.
            write_header: ``default=True``

                Flag for writing headers for the five columns in the first row.

            domain: ``default=None``

                Domain specification.

        """
        fracs = [cast(pp.LineFracture, frac) for frac in self.fractures]
        pts, edges = _linefractures_to_pts_and_edges(fracs)

        # Delete the file 'csv_file' if it exists. This seems to be necessary to run
        # tests on GH actions.
        file_name = file_name.with_suffix(".csv")
        if file_name.exists():
            file_name.unlink()

        with open(file_name, "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            # write all the fractures
            if self.domain is not None:
                if write_header:
                    header = [
                        "# ",
                        "DOMAIN_XMIN",
                        "DOMAIN_YMIN",
                        "DOMAIN_XMAX",
                        "DOMAIN_YMAX",
                    ]
                    csv_writer.writerow(header)
                order = ["xmin", "ymin", "xmax", "ymax"]
                # Write the domain bounding box.
                csv_writer.writerow([self.domain.bounding_box[o] for o in order])
            if write_header:
                header = [
                    "# FRACTURE COORDINATES",
                    "START_X",
                    "START_Y",
                    "END_X",
                    "END_Y",
                ]
                csv_writer.writerow(header)
            for edge_id, edge in enumerate(edges.T):
                data = np.hstack([pts[:, edge[0]], pts[:, edge[1]]]).tolist()
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

        fracs = [cast(pp.LineFracture, frac) for frac in self.fractures]
        pts, edges = _linefractures_to_pts_and_edges(fracs)

        # cell connectivity information
        meshio_cells = np.empty(1, dtype=object)
        meshio_cells[0] = meshio.CellBlock(cell_type, edges.T)
        # prepare the points
        meshio_pts = pts.T
        # make points 3d
        if meshio_pts.shape[1] == 2:
            meshio_pts = np.hstack((meshio_pts, np.zeros((meshio_pts.shape[0], 1))))

        # Cell-data to be exported is at least the fracture numbers
        meshio_cell_data = {}
        meshio_cell_data["fracture_number"] = [
            fracture_offset + np.arange(edges.shape[1])
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


def _linefractures_to_pts_and_edges(
    fractures: list[pp.LineFracture], tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of line fractures into arrays of the corresponding points and
    edges.

    The function loops over the points of the individual fractures and checks if the
    point is the start/end point (up to the given tolerance) of a previously checked
    fracture. If yes, the edge index links to the existing point. If no, the point is
    added to the points array.

    Parameters:
        fractures: List of line fractures.
        tol: ``default=1e-8``

            Absolute tolerance to decide if start-/endpoints of two different fractures
            are equal. The comparison uses the max-norm over the difference in
            coordinates.

    Returns:
        A 2-tuple containing

        :obj:`~numpy.ndarray`: ``(shape=(2, num_points))``

            Coordinates of the start- and endpoints of the fractures.
        :obj:`~numpy.ndarray`: ``shape=(2 + num_tags, len(fractures)), dtype=int``

            An array containing column-wise (per fracture) the indices for the start-
            and endpoint in the first two rows.

            Note that one point in ``pts`` may be the start- and/or endpoint of multiple
            fractures.

            Additional rows are optional tags of the fractures. In the standard form,
            the third row (first row of tags) identifies the type of edges, referring
            to the numbering system in ``GmshInterfaceTags``. The second row of tags
            keeps track of the numbering of the edges (referring to the original
            order of the edges) in geometry processing like intersection removal.
            Additional tags can be assigned by the user.

        When an empty list of fractures is passed, both arrays have shape ``(2, 0)``.

    """
    pts_list: list[np.ndarray] = []
    edges_list: list[np.ndarray] = []

    # Iterate through the fractures and list all start-/endpoints and the corresponding
    # edge indices.
    for frac in fractures:
        pt_indices: list[int] = []
        for point in frac.points():
            # Check if the point is already start-/endpoint of another fracture.
            compare_points = [
                np.allclose(point.squeeze(), x, atol=tol) for x in pts_list
            ]
            if not any(compare_points):
                pts_list.append(point.squeeze())
                pt_indices.append(len(pts_list) - 1)
            else:
                pt_indices.append(compare_points.index(True))
        # Sanity check that two points indices were added.
        assert len(pt_indices) == 2
        # Combine with tags of the fracture and store the full edge in a list.
        edges_list.append(np.concatenate([np.array(pt_indices), frac.tags]))

    # Transform the lists to two ``np.ndarrays`` (``pts`` and ``edges``).
    if pts_list:
        # ``np.stack`` requires a nonempty list.
        pts = np.stack(pts_list, axis=-1)
    else:
        pts = np.zeros([2, 0])
    # Before creating the ``edges`` array, determine the maximum number of tags.
    # This determines the shape of the ``edges`` array.
    max_edge_dim = max((np.shape(edge)[0] for edge in edges_list), default=2)
    # Initialize the ``edges`` array with ``-1``. This value indicates that each edge
    # has no tags. Fill in the first two rows with the fracture start-/endpoints and
    # the rest of the rows with tags where they exist. All other tags keep their
    # initial value of ``-1``, which is equal to the tag not existing. This seemingly
    # complicated procedure is done to ensure that the ``edges`` array is not ragged.
    edges = np.full((max_edge_dim, len(fractures)), -1, dtype=np.int32)
    for row_index, edge in enumerate(edges_list):
        edges[: edge.shape[0], row_index] = edge

    return pts, edges
