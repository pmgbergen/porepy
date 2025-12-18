"""Module contains class for representing a fracture network in a 2d domain."""

from __future__ import annotations

import copy
import csv
import itertools
import logging
import multiprocessing
import time
from pathlib import Path
from typing import Optional

import gmsh
import meshio
import numpy as np
from matplotlib import pyplot as plt

import porepy as pp
import porepy.fracs.simplex
from porepy.fracs import tools
from porepy.fracs.utils import linefractures_to_pts_edges, pts_edges_to_linefractures

from .gmsh_interface import GmshData2d, GmshWriter, PhysicalNames
from .gmsh_interface import Tags as GmshInterfaceTags
from .fracture_network import (
    FractureNetwork,
    GmshPointIdentifier,
    MeshSizeComputer,
    MeshSizeControlPointInserter,
)

logger = logging.getLogger(__name__)

# Shortcut for the OpenCaste CAD kernel in gmsh.
fac = gmsh.model.occ


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
        super().__init__(nd=2, fractures=fractures, domain=domain, tol=tol)

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

    def fractures_to_gmsh(self) -> list[int]:
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
        mesh_size_computer = MeshSizeComputer(mesh_args)

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

        # TODO:
        # 1. Unified treatment of distance notions in mesh size control.

        # region prepare_for_gmsh

        # Get gmsh tags of domain and fractures.
        # TODO Here, the question arises whether we should already enrich the tags with
        # dimensionality, e.g.,
        # gmsh_fractures = [(1, f) for f in self.fractures_to_gmsh_2D()]
        # Both are needed later, so not sure what is best. Having both seems
        # unnecessary.
        domain_tag = self.domain_to_gmsh()
        fracture_tags = self.fractures_to_gmsh()
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
            if distance > self._tol or self._entity_on_domain_boundary(
                1, [fracture_tag]
            ):
                # The fracture is either fully outside the domain or fully embedded on
                # the domain boundary. It will be deleted.
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
        fac.synchronize()

        # Remove from fracture_tags those indices that are present in removed_fractures
        fracture_tags = [
            ft for i, ft in enumerate(fracture_tags) if i not in removed_fractures
        ]
        # Insert points to control mesh size for nearly intersecting lines.
        boundary_tags = [t for _, t in gmsh.model.get_boundary([(2, domain_tag)])]

        mesh_size_points = self._insert_mesh_size_control_points(mesh_size_computer)

        # Mapping from the new fracture tags (gmsh assigned) to the input fractures.
        inv_fracture_tag_map = {i: counter for counter, i in enumerate(fracture_tags)}

        # Make gmsh calculate the intersections between fractures, using the domain as a
        # secondary object (the latter will by magic ensure that the fractures are
        # embedded in the domain, hence the mesh will conform to the fractures). The
        # removal statements here will replace the old (possibly intersecting) fractures
        # with new, split lines. Similarly, the removal of the domain (removeTool)
        # avoids the domain being present twice.

        line_tags_new = fracture_tags + boundary_tags
        _, isect_mapping = fac.fragment(
            [(nd - 1, ft) for ft in line_tags_new],
            [(nd, domain_tag)],
            removeObject=True,
            removeTool=True,
        )

        fac.synchronize()

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

            if old_fracture[0][0] == nd:
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
            frac_ind = inv_fracture_tag_map[old_gmsh_tag]

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
        inv_fracture_tag_map = updated_fracture_tag_map

        # Find the unique boundary points and obtain a mapping from the full set of
        # boundary points to the unique ones.
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

        # Count the number of fracture objects that survived both the fragmentation and
        # the distance-based domain trimming.
        num_real_frac = len(set(inv_fracture_tag_map.values()))

        fracture_to_line = {}
        tmp_frac_line = []
        for i, line_group in enumerate(isect_mapping[:num_real_frac]):
            # A line_group here was formed after intersection removal. It may contain
            # either a full fracture, or be one of several segments forming a fracture.
            # In the latter case, the fracture was split into segments when mesh size
            # control points were added to the fracture.
            all_lines = []

            for line in line_group:
                if line[0] == 1:
                    all_lines.append(line[1])
                    tmp_frac_line.append(inv_fracture_tag_map[line[1]])
            if all_lines:
                frac_ind = inv_fracture_tag_map[all_lines[0]]
                fracs = fracture_to_line.get(frac_ind, [])
                fracs.extend(all_lines)
                fracture_to_line[frac_ind] = fracs

        # EK note to self: Failure of the following assertion implies if we have not
        # managed to track all split fractures (due to intersections or the presence of
        # boundary points). Not sure if we want to include it.
        #
        # assert len(set(tmp_frac_line)) == len(fracture_tags)

        for fi, segments in fracture_to_line.items():
            if fi in constraints:
                gmsh.model.addPhysicalGroup(
                    nd - 1,
                    segments,
                    -1,
                    f"{PhysicalNames.AUXILIARY_LINE.value}{fi}",
                )
            else:
                gmsh.model.addPhysicalGroup(
                    nd - 1, segments, -1, f"{PhysicalNames.FRACTURE.value}{fi}"
                )

        # It turns out that if fractures split the domain into disjoint parts, gmsh may
        # choose to redefine the domain as the sum of these parts. Therefore, we
        # redefine the domain tags here, using all volumes in the model.
        domain_tags = [entity[1] for entity in gmsh.model.get_entities(nd)]

        gmsh.model.addPhysicalGroup(
            nd, domain_tags, -1, f"{PhysicalNames.DOMAIN.value}"
        )

        fac.synchronize()
        if write_geo:
            gmsh.write(str(file_name.with_suffix(".geo_unrolled")))

        # Set the mesh sizes after all geometry processing is done so that the
        # identification of objects is not disturbed by retagging of objects.
        self._set_background_mesh_field(
            self._set_1d_mesh_size(mesh_size_computer, mesh_size_points)
        )
        gmsh.model.occ.synchronize()

        # region GmshWriter.generate

        # Consider the dimension of the problem. Normally 2D, but if ``dfn`` is True 1D.
        ndim = nd - int(dfn)

        # Create a gmsh mesh.
        gmsh.model.mesh.generate(1)
        if not dfn:
            # Remove the 1d mesh fields, set new ones, then generate the 2d mesh.
            for field in gmsh.model.mesh.field.list():
                gmsh.model.mesh.field.remove(field)
            self._set_2d_mesh_size(mesh_size_computer, mesh_size_points)
            gmsh.model.mesh.generate(2)

        gmsh.write(str(file_name))

        # Report mesh quality metrics.
        if False:
            self.mesh_quality_metrics()

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

    def _set_1d_mesh_size(
        self,
        mesh_size_computer: MeshSizeComputer,
        mesh_size_points: dict[int, list[tuple[np.ndarray, float]]],
        restrict_to_fractures: bool = True,
    ) -> None:
        ### Get hold of lines representing fractures and boundaries.
        domain_entities = gmsh.model.get_entities(2)
        # TODO: If there is more than one domain entity (the domain is split into parts
        # by fractures), we need to pick out the outer boundary, that is, the ones which
        # only occurs once.
        boundaries = gmsh.model.get_boundary([(2, tag) for _, tag in domain_entities])
        fractures = [f for f in gmsh.model.getEntities(1) if f not in boundaries]

        line_tags = [tag for _, tag in gmsh.model.getEntities(1)]
        fracture_tags = [tag for _, tag in fractures]
        boundary_tags = set(tag for _, tag in boundaries)

        gmsh_fields = []

        gmsh_point_finder = GmshPointIdentifier()

        if len(mesh_size_points) > 0:
            all_pts = []
            mesh_sizes = []
            line_item = []
            for line, info in mesh_size_points.items():
                for i, d in enumerate(info):
                    all_pts.append(d[0])
                    mesh_sizes.append(d[1])
                    line_item.append((line, i))
            all_pts = np.array(all_pts).T
            mesh_sizes = np.array(mesh_sizes)
            if all_pts.size > 0:
                _, ind_map, inv_map = pp.array_operations.uniquify_point_set(
                    all_pts, tol=self._tol
                )
                min_size = np.empty(ind_map.size, dtype=float)
                for i in range(ind_map.size):
                    inds = inv_map == i
                    min_size[i] = np.min(mesh_sizes[inds])

                # Map back to lines.
                for line_ind, pt_ind in enumerate(inv_map):
                    line = line_item[line_ind][0]
                    item = line_item[line_ind][1]
                    mesh_size_points[line][item] = (
                        mesh_size_points[line][item][0],
                        min_size[pt_ind],
                    )

        # For lines that with no extra
        mesh_size = {tag: [] for tag in line_tags}
        mesh_size.update(mesh_size_points)

        for line, info in mesh_size.items():
            # Uniquify the point set (the same point may have been identified multiple
            # times).
            end_points = np.array(
                [
                    gmsh.model.occ.get_bounding_box(0, p[1])[:3]
                    for p in gmsh.model.get_boundary([(1, line)], combined=False)
                ]
            ).T
            length = np.linalg.norm(end_points[:, 1] - end_points[:, 0])
            tol = min(length, mesh_size_computer.h_frac()) / 2
            extra_points = (
                np.array([d[0] for d in info]).T if len(info) > 0 else np.empty((3, 0))
            )

            points, _, ind_map = pp.array_operations.uniquify_point_set(
                np.hstack((end_points, extra_points)), tol=tol
            )
            # Distance to other objects for each point, as computed previously. Assign
            # h_frac or h_bound to the endpoints, depending on whether the line is a
            # fracture or boundary line. We also assign h_frac, since no refinement is
            # needed just because this is an intersection point (if it is an
            # intersection with a bad angle, this should be picked up by a close point
            # on another line).
            h_end = mesh_size_computer.h_end(line in boundary_tags)

            other_object_distances_all = np.hstack(
                (
                    np.array([h_end, h_end]),
                    np.array(
                        [
                            d[1] if d[1] > 0 else mesh_size_computer.h_frac()
                            for d in info
                        ]
                    ),
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
                assert False

            # The final distance to be used for mesh size calculation is the minimum of
            # the distance to other objects and the distance to other close points on
            # the same line.
            dist = np.minimum(other_object_distances, min_dist_point)

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
            mesh_size.keys(),
            mesh_size_computer,
            boundary_tags,
            restrict_to_fractures,
        )

        return gmsh_fields

    def _set_2d_mesh_size(
        self,
        mesh_size_computer: MeshSizeComputer,
        mesh_size_points: dict[tuple[int, int], float],
    ) -> None:
        factory = gmsh.model.occ

        gmsh_fields = self._set_1d_mesh_size(
            mesh_size_computer, mesh_size_points, restrict_to_fractures=False
        )

        # Finally, as the background mesh, we take the minimum of all the created
        # fields.
        self._set_background_mesh_field(gmsh_fields)

    def mesh_quality_metrics(self) -> None:
        """Visualize, and log elementwise mesh quality metrics using gmsh.

        The evaluated metrics include:
            - minDetJac / maxDetJac : Minimum and maximum determinant of the Jacobian.
            - minSJ                 : Minimum scaled Jacobian (element regularity).
            - minSICN / minSIGE     : Inverse condition numbers measuring element
                                      skewness.
            - gamma                 : Shape quality factor (close to 1 → good element).
            - innerRadius / outerRadius : Ratio of inscribed to circumscribed radii.
            - minIsotropy           : Degree of isotropy (1 → perfectly isotropic).
            - angleShape            : Angular distortion indicator.
            - minEdge / maxEdge     : Minimum and maximum edge lengths.
            - volume                : Element area or volume measure.

        """
        # Compute mesh quality metrics using gmsh.
        all_element_tags = gmsh.model.mesh.getElements(2)[1][0]
        quality_types = [
            "minDetJac",
            "maxDetJac",
            "minSJ",
            "minSICN",
            "minSIGE",
            "gamma",
            "innerRadius",
            "outerRadius",
            "minIsotropy",
            "angleShape",
            "minEdge",
            "maxEdge",
            "volume",
        ]
        results = {}
        for qtype in quality_types:
            try:
                qvalues = gmsh.model.mesh.getElementQualities(all_element_tags, qtype)
                if len(qvalues) > 0:
                    results[qtype] = qvalues
            except Exception as e:
                print(f"Skipping {qtype}: {e}")

        # Plot histogram of mesh quality metrics.
        n = len(results)
        cols = 5
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten()

        for ax, (qtype, qvalues) in zip(axes, results.items()):
            ax.hist(
                qvalues,
                bins=30,
                color="#4C72B0",
                edgecolor="white",
                alpha=0.8,
            )

            ax.set_title(qtype, fontsize=11, fontweight="bold")
            ax.set_xlabel("Value", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)

        # Hide unused subplots
        for ax in axes[len(results) :]:
            ax.axis("off")

        fig.suptitle("Mesh Quality Distributions", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

        # Log mesh quality metrics.
        for qtype, qvalues in results.items():
            if len(qvalues) > 0:
                logger.info(
                    f"{qtype:15s}: min = {qvalues.min():.4e}, max = {qvalues.max():.4e}"
                    + f"avg = {qvalues.mean():.4e}, std = {qvalues.std():.4e}"
                )
            else:
                logger.info(f"{qtype:15s}: (no values returned)")

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

        fn = FractureNetwork2d(fractures_new, domain, self._tol)

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
