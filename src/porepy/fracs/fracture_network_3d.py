"""A module for representation and manipulation of 3D fracture networks."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Optional, Union

import gmsh
import meshio
import numpy as np

import porepy as pp
from porepy.fracs.gmsh_interface import PhysicalNames

from .fracture_network import FractureNetwork, GmshPointIdentifier, MeshSizeComputer


class FractureNetwork3d(FractureNetwork):
    """Representation of a set of plane fractures in a three-dimensional domain.

    This is a collection of fractures with geometrical information. It facilitates
    computation of intersections of the fracture. Also, it incorporates the bounding
    box of the domain. To ensure that all fractures lie within the box,
    call :meth:`impose_external_boundary` *after* all fractures have been specified.

    Parameters:
        fractures: ``default=None``

            Plane fractures that make up the network. Defaults to ``None``, which will
            create a domain without fractures. An empty :attr:`~fractures` list is
            effectively treated as ``None``.
        domain: ``default=None``

            Domain specification. Can be box-shaped or a general (non-convex)
            polyhedron.
        tol:  ``default=1e-8``

            Tolerance used in geometric computations.

    """

    def __init__(
        self,
        fractures: Optional[list[pp.PlaneFracture | pp.EllipticFracture]] = None,
        domain: Optional[pp.Domain] = None,
        tol: float = 1e-8,
    ) -> None:
        super().__init__(nd=3, domain=domain, tol=tol)
        self.fractures: list[pp.PlaneFracture | pp.EllipticFracture] = []
        """List of fractures forming the network."""
        # Populate fracture list and assign indices.
        if fractures is not None:
            for index, f in enumerate(fractures):
                self.fractures.append(f)
                f.index = index

    def domain_to_gmsh(self) -> int:
        """Export the box domain to Gmsh using the OpenCASCADE kernel.

        This method creates a box corresponding to the bounding box of the
        fracture network domain and adds it to the current Gmsh model. The OpenCASCADE
        CAD kernel is used for the geometry representation.

        Returns:
            The Gmsh tag ID of the created box.

        """
        if self.domain is None:
            raise ValueError("No domain has been specified for this fracture network.")

        if self.domain.is_boxed:
            # Defining a box domain in Gmsh's OpenCASCADE kernel is straightforward.
            bb = self.domain.bounding_box

            xmin, xmax = bb["xmin"], bb["xmax"]
            ymin, ymax = bb["ymin"], bb["ymax"]
            zmin, zmax = bb["zmin"], bb["zmax"]

            domain_tag = gmsh.model.occ.addBox(
                xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin
            )
        else:
            # General polytopes require a more involved procedure: Listed backwards, we
            # need to define the surfaces of the polytope domain and define a volume
            # from these. Each surface is defined by a set of lines, which in turn are
            # defined by their endpoints. The challenge is that lines and points need to
            # be uniquely defined, so that if two surfaces in the polytope description
            # share a line, this must be represented by a single line in Gmsh, and the
            # surfaces' Gmsh representation must reference this single line.

            # For bookkeeping.
            polytope = self.domain.polytope
            polytope_sizes = [poly.shape[1] for poly in polytope]
            offsets = np.hstack(([0], np.cumsum(polytope_sizes)))

            # First find all unique points in the polytope description. Export them to
            # Gmsh.
            pts = np.hstack([poly for poly in polytope])
            unique_pts, _, unique_pt_map = pp.array_operations.uniquify_point_set(
                pts, self._tol
            )
            # These are the tags which Gmsh assigns to the points.
            pt_tags = [gmsh.model.occ.addPoint(*pt) for pt in unique_pts.T]

            # Define the lines making up the polygons in the polytope. Use offsets to
            # define the lines for a common set of point indices (i.e. the colummns in
            # variable pts above).
            polytope_lines = np.hstack(
                [
                    offsets[pi]
                    # The use of np.roll ensures that the last point connects to the
                    # first point.
                    + np.vstack((np.arange(size), np.roll(np.arange(size), -1)))
                    for pi, size in enumerate(polytope_sizes)
                ]
            )
            # Use the mapping to unique points to get the actual point indices
            # (referring to unique_pts above).
            lines_with_unique_points = unique_pt_map[polytope_lines]
            # Now we need to find the unique lines, which should be exported to gmsh,
            # and a mapping from all lines to the unique ones. Do a sort along axis 0,
            # so that lines that share nodes but have the opposite direction are
            # identified as the same line. Thankfully, the Gmsh OpenCascade kernel does
            # not mind if we define surfaces that contain lines with inconsistent
            # orientation, or else this would have been more complicated.
            unique_lines, line_mapping = np.unique(
                np.sort(lines_with_unique_points, axis=0), axis=1, return_inverse=True
            )
            # Now we can export the unique set of lines to Gmsh. We need to refer to the
            # points according to their Gmsh tags.
            line_tags = [
                gmsh.model.occ.addLine(
                    pt_tags[unique_lines[0, i]], pt_tags[unique_lines[1, i]]
                )
                for i in range(unique_lines.shape[1])
            ]
            # Gather the surfaces making up the polytope.
            surfaces = []

            for pi in range(len(polytope)):
                # Define the line loop for this polygon. We need to use the line mapping
                # to get referrals to the unique lines, and then the line tags to get
                # the Gmsh tags.
                line_loop = gmsh.model.occ.addCurveLoop(
                    [line_tags[i] for i in line_mapping[offsets[pi] : offsets[pi + 1]]]
                )
                surfaces.append(gmsh.model.occ.addPlaneSurface([line_loop]))
            # Finally, we need to create a surface loop for the polytope and then create
            # the volume.
            surf_loop = gmsh.model.occ.addSurfaceLoop(surfaces)
            domain_tag = gmsh.model.occ.addVolume([surf_loop])

        gmsh.model.occ.synchronize()
        return domain_tag

    def copy(self) -> FractureNetwork3d:
        """Create a deep copy of the fracture network.

        The method will create a deep copy of all fractures and of the domain.

        Note:
            If the fractures have had extra points imposed as part of a meshing
            procedure, these will be included in the copied fractures.

        Returns:
            Deep copy of this fracture network.

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

        return FractureNetwork3d(fracs, domain, self._tol)  # type: ignore[arg-type]

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

        if self.domain is not None:
            domain_tag = self.domain_to_gmsh()
            boundary_tags = [
                t
                for _, t in gmsh.model.get_boundary(
                    [(self.nd, domain_tag)], oriented=False
                )
            ]
        else:
            domain_tag = -1
            boundary_tags = []

        fracture_tags = self.fractures_to_gmsh()
        surface_tags = fracture_tags + boundary_tags
        gmsh.model.occ.synchronize()

        # STEP 1: Insert mesh size control points on fractures and boundaries.
        mesh_control_dict = self._insert_mesh_size_control_points(mesh_size_computer)
        gmsh.model.occ.synchronize()

        # Mapping from the gmsh fracture tags to the input fractures.
        gmsh_to_porepy_fracture_ind_map = {
            i: counter for counter, i in enumerate(fracture_tags)
        }

        surface_tags = fracture_tags + boundary_tags

        # STEP 2: Impose external boundary and process intersections.
        (
            intersection_points,
            intersection_lines,
            isect_mapping,
            num_parents_of_lines,
            constraints,
            intersection_line_parents,
            gmsh_to_porepy_fracture_ind_map,
        ) = self._impose_boundary_process_intersections(
            fracture_tags,
            domain_tag,
            constraints=constraints,
            gmsh_to_porepy_fracture_ind_map=gmsh_to_porepy_fracture_ind_map,
        )
        # Update mesh size control points after intersection processing.
        mesh_control_dict = self._update_mesh_size_control_point(
            mesh_control_dict, isect_mapping, surface_tags, boundary_tags
        )

        # STEP 3: Export physical entities to gmsh.
        fracture_to_surface = self._set_physical_names(
            intersection_points,
            intersection_lines,
            num_parents_of_lines,
            isect_mapping,
            gmsh_to_porepy_fracture_ind_map,
            constraints,
        )

        gmsh.model.occ.synchronize()

        gmsh.write(str(file_name.with_suffix(".geo_unrolled")))

        # STEP 4: Set the mesh sizes.
        self._set_background_mesh_field(
            self._set_mesh_size_fields(
                mesh_size_computer,
                mesh_control_dict,
                intersection_lines,
                intersection_line_parents,
                fracture_to_surface,
                restrict_to_fractures=True,
            )
        )
        gmsh.model.occ.synchronize()

        # STEP 5: Generate the mesh.
        gmsh.model.mesh.generate(self.nd - 1)
        if not dfn:
            # Remove the 1d mesh fields, set new ones, then generate the 2d mesh.
            for field in gmsh.model.mesh.field.list():
                gmsh.model.mesh.field.remove(field)
            # Set the new mesh size fields, but this time not restricting to fractures.
            self._set_background_mesh_field(
                self._set_mesh_size_fields(
                    mesh_size_computer,
                    mesh_control_dict,
                    intersection_lines,
                    intersection_line_parents,
                    fracture_to_surface,
                    restrict_to_fractures=False,
                )
            )
            gmsh.model.mesh.generate(self.nd)

        # Delete the file 'file_name' if it exists, and write the new mesh to
        # 'file_name'. This seems to be necessary to run tests on GH actions.
        if file_name.exists():
            file_name.unlink()

        gmsh.write(str(file_name))
        # Report mesh quality metrics.
        if self._extra_meshing_args["plot_mesh_quality_metrics"]:
            self.mesh_quality_metrics()

        # STEP 6: Process the gmsh .msh output file, to make first a list of grids, then
        # a MixedDimensionalGrid.
        if dfn:
            subdomains = pp.fracs.simplex.triangle_grid_embedded(file_name)
        else:
            subdomains = pp.fracs.simplex.tetrahedral_grid_from_gmsh(
                file_name, constraints
            )

        gmsh.finalize()
        # Merge the grids into a mixed-dimensional grid.
        mdg = pp.meshing.subdomains_to_mdg(subdomains, **kwargs)
        return mdg

    def _impose_boundary_process_intersections(
        self,
        fracture_tags: list[int],
        domain_tag: int,
        constraints: np.ndarray,
        gmsh_to_porepy_fracture_ind_map: dict[int, int],
    ) -> tuple[list, list, list, list, np.ndarray, list, dict]:
        """Impose the external boundary and process fracture intersections.

        This is the method where all the geometry processing happens. The method relies
        heavily on the OpenCASCADE kernel in Gmsh to perform the actual computations.

        Parameters:
            fracture_tags: List of gmsh tags representing the fractures in the network.
            domain_tag: Gmsh tag representing the domain.
            constraints: List of fracture indices (referring to input fracture
                numbering) that are to be treated as constraints.
            gmsh_to_porepy_fracture_ind_map: Mapping from gmsh fracture tags to input
                fracture indices.

        Returns:
            A tuple with the following elements:
            -   List with gmsh tags of intersection points.
            -   List with gmsh tags of intersection lines.
            -   List that maps input fracture indices to new surfaces after intersection
                processing.
            -   Numpy array with number of parent fractures for each intersection line.
            -   Updated numpy array of constraint fracture indices (still referring to
                the input fracture numbering).
            -   List of parent fracture indices for each intersection line.
            -   Updated mapping from gmsh fracture tags to input fracture indices.

        """
        nd = self.nd
        dim_fracture_tags = [(nd - 1, tag) for tag in fracture_tags]

        # Note: The method is called prior to splitting of fractures (which may also
        # split the domain), hence there will be a single domain object to fragment.
        isect_mapping = self._fragment_fractures(fracture_tags, domain_tag)

        # It turns out (...) that the fragmentation process may not eliminate parts of
        # fractures that lie outside the domain (this is contrary to EK's reading of the
        # Gmsh documentation, but so it goes). For now, we do a simple fix to eliminate
        # (parts of) fractures that are outside the domain: Identify the vertexes of
        # each fracture part, compute their distance to the domain. If any of these
        # distances is larger than the tolerance, we drop the fracture from further
        # consideration. There are surely cases where this simple approach fails, but it
        # will have to do for now.

        # The domain tags may have changed during fragmentation, so get the current
        # list.
        domain_tags = [t[1] for t in gmsh.model.get_entities(nd)]
        # Keep track of which fractures to keep.
        keep = np.ones(len(isect_mapping), dtype=bool)
        # Keep track of which fractures have had parts deleted. If all parts of a
        # fracture have been deleted, we need to update the constraint indices.
        part_of_fracture_deleted = []
        # Count the number of sub-fractures each fracture has been split into. Used to
        # figure out if the full fracture has been deleted as outside the domain, or
        # only parts of it; which again is used to update the constraint indices.
        num_orig_subfrac = []

        # Double loop: First over all fractures, then over all fragments of each
        # fracture. We kick out fragments where at least one vertex is outside the
        # domain (has a distance larger than tol). If all fragments of a fracture are
        # kicked out, we need to remove the fracture altogether, and update the
        # constraint indices accordingly.
        updated_fracture_tag_map: dict[int, int] = {}
        for fi, frac in enumerate(isect_mapping):
            if frac and frac[0][0] == 3:
                # This is the domain. Skip it.
                continue
            elif fi >= len(fracture_tags):
                # This is not a fracture, quite likely it is part of the boundary.
                # Skip it.
                continue
            frac_ind = fracture_tags[fi]
            loc_keep = np.ones(len(frac), dtype=bool)
            num_orig_subfrac.append(len(frac))

            for sfi, sub_frac in enumerate(frac):
                bounding_lines = gmsh.model.get_boundary([sub_frac])
                bounding_points = []
                for line in bounding_lines:
                    bounding_points += gmsh.model.get_boundary([line])

                if len(bounding_points) == 0:
                    # This is most likely a disc fracture, which has no bounding
                    # lines/points. Simply do a distance check between the fracture and
                    # the domain.
                    #
                    # If this assertion fails, this indicates that we have produced a
                    # non-disc domain without boundary lines/points. This case must be
                    # handled if it ever arises.
                    assert len(frac) == 1
                    distances = np.array(
                        [
                            gmsh.model.occ.get_distance(*sub_frac, self.nd, dtag)[0]
                            for dtag in domain_tags
                        ]
                    )
                    if np.all(distances > self._tol):
                        loc_keep[sfi] = False
                        part_of_fracture_deleted.append(
                            gmsh_to_porepy_fracture_ind_map[frac_ind]
                        )
                    else:
                        updated_fracture_tag_map[sub_frac[1]] = (
                            gmsh_to_porepy_fracture_ind_map[frac_ind]
                        )
                    continue

                # For each bounding point, compute the minimum distance to the different
                # parts of the domain (the domain may have been split in multiple parts
                # during fragmentation). Note to self: We cannot check the sub-surface
                # itself, since for fractures partially inside the domain, the
                # sub-surface that should be excluded will still be inside the domain.
                distances = np.zeros(len(bounding_points))
                if domain_tag > 0:
                    # Only do this if we have a domain. If not, all distances are zero,
                    # and the above if statement will not trigger.
                    for i, pt in enumerate(bounding_points):
                        distances[i] = min(
                            gmsh.model.occ.get_distance(*pt, nd, domain_tag)[0]
                            for domain_tag in domain_tags
                        )
                # If any bounding point is outside all parts of the domain, we drop this
                # sub-fracture.
                if np.any(distances > self._tol) or self._entity_on_domain_boundary(
                    0, [bp[1] for bp in bounding_points]
                ):
                    loc_keep[sfi] = False
                    # Take note that part of this fracture (mapping back to the input
                    # fracture index system) has been deleted.
                    part_of_fracture_deleted.append(
                        gmsh_to_porepy_fracture_ind_map[frac_ind]
                    )
                else:
                    if frac_ind in gmsh_to_porepy_fracture_ind_map:
                        updated_fracture_tag_map[sub_frac[1]] = (
                            gmsh_to_porepy_fracture_ind_map[frac_ind]
                        )
            # Keep only the sub-fractures that are within the domain.
            isect_mapping[fi] = [frac[i] for i in range(len(frac)) if loc_keep[i]]
            # If any sub-fracture is kept, we keep the fracture.
            keep[fi] = np.any(loc_keep)

        # Remove from the bookkeeping system fractures where all the sub-fractures were
        # outside the domain.
        isect_mapping = [isect_mapping[i] for i in range(len(keep)) if keep[i]]

        # Update the constraint indices to account for fully removed fractures.
        updated_constraints = []
        # If a fracture has been fully removed, we need to decrement the indices of
        # all following constraints.
        num_frac_deleted = 0
        for c in constraints:
            num_deleted_subfrac = part_of_fracture_deleted.count(c)
            if num_orig_subfrac[c] == num_deleted_subfrac:
                # The full fracture has been removed. It is not among the surviving
                # constraints, but we need to adjust the indices of the following ones.
                num_frac_deleted += 1
            else:
                # The fracture is still present, add it to the new constraints,
                # adjusting the index accordingly. Constraints are known to be sorted.
                updated_constraints.append(int(c) - num_frac_deleted)
        constraints = np.asarray(updated_constraints)
        gmsh_to_porepy_fracture_ind_map = updated_fracture_tag_map

        # Count the number of fracture objects that survived both the fragmentation and
        # the distance-based domain trimming.
        num_fracs = len(set(gmsh_to_porepy_fracture_ind_map.values()))

        # Now, identify intersection lines. Intersection lines are either on the
        # boundary or embedded in fractures. Make a list of each.
        bnd_lines = []
        embedded_lines = []
        # Keep track of which fracture each line stems from.
        fi_bnd = []
        fi_embedded = []

        # Loop over all identified fragments of the fractures, find their boundary and
        # embedded lines.
        for fi, gmsh_frac_ind in enumerate(isect_mapping[:num_fracs]):
            frac_ind = gmsh_to_porepy_fracture_ind_map[gmsh_frac_ind[0][1]]

            # NOTE: Constraints do not contribute to intersection lines. This implies
            # that all the counting of parent lines (for intersection points) below is
            # done without including constraints, hence we need not worry about
            # adjusting for constraints there.
            if frac_ind in constraints:
                continue

            # A fracture can be split into multiple sub-fractures if they are fully cut
            # by other fractures.
            for subfrac in gmsh_frac_ind:
                if subfrac[0] == 3:  # This is the domain.
                    continue
                # Get the boundary of the sub-fracture. It can contain both lines on the
                # boundary of the original fracture and lines on the boundary of
                # subfractures that were introduced because a fracture was cut in two.
                bnd = gmsh.model.get_boundary([subfrac], oriented=False)
                # Loop over the boundary.
                for line in bnd:
                    if line[0] == 1 and not self._entity_on_domain_boundary(
                        1, [line[1]]
                    ):
                        # This is a line, not a point (would be line[0] == 0).
                        bnd_lines.append(line[1])
                        # Keep track of the fracture index for each boundary line.
                        fi_bnd.append(frac_ind)

                # Also find lines that are embedded in this subfracture (this will be an
                # intersection line that does not cut subfrac in two).
                embedded = gmsh.model.mesh.get_embedded(*subfrac)
                for line in embedded:
                    if line[0] == 1 and not self._entity_on_domain_boundary(
                        1, [line[1]]
                    ):
                        embedded_lines.append(line[1])
                        # Also keep track of the fracture index for each embedded line.
                        fi_embedded.append(frac_ind)

        # For a boundary line to be an intersection, it must be shared by at least two
        # fractures.
        num_lines_occ = np.bincount(np.array(bnd_lines).astype(int))
        # Find the boundary lines occuring more than once.
        boundary_lines = np.where(num_lines_occ > 1)[0]
        all_lines = np.hstack((embedded_lines, boundary_lines))
        # Fracture intersection lines, to be added as physical lines.
        intersection_lines = np.unique(all_lines).astype(int)

        # Now, we need to find which intersection lines stem from the same set of
        # intersecting fractures (this can be two or more fractures). This requires some
        # juggling of indices.

        # For each intersection line, this will be a list of its parent fractures.
        line_parents = []
        for line in intersection_lines:
            # Find the set of parents, looking at both boundary and embedded lines (an
            # intersection can be on the boundary of fracture, but not the other).
            parent = np.hstack(
                (
                    np.asarray(fi_bnd)[np.where(np.array(bnd_lines) == line)[0]],
                    np.asarray(fi_embedded)[
                        np.where(np.abs(embedded_lines) == line)[0]
                    ],
                )
            ).astype(int)
            # Uniquify (thereby also sort) and turn to list.
            line_parents.append(np.unique(parent).tolist())

        # Now we need to find the unique parent sets. Since line_parents can have a
        # varying number of elements, we cannot just do a numpy unique, but instead need
        # to process each number of parents separately (if the number of parents
        # differs, clearly, the sets of parents must also differ).

        # Do a count.
        num_parents = np.array([len(lp) for lp in line_parents])
        # This will be the mapping from line indices to their parent set indices.
        parent_of_intersection_lines = np.full(num_parents.size, -1)

        # Counter over intersection lines. Linked to the physical tags that will be
        # associated with the intersection lines. Note that there is no requirement that
        # this is related to the physical tag of the parent fracture (to the degree we
        # care about such intersections, we go through the grid information generated by
        # gmsh).
        num_line_parent_counter = 0

        # Loop over all unique parent counts.
        for n in np.unique(num_parents):
            if n < 2:
                # At most one of the parents was not a constraint. This should not
                # produce a line.
                continue

            # Find all intersection lines that has this number of parents.
            inds = np.where(num_parents == n)[0]
            # Find the unique number of parents and the map from all intersection lines
            # with 'n' parents to the unique set.
            unique_parent, line = np.unique(
                [line_parents[i] for i in inds], axis=0, return_inverse=True
            )
            # Store the parent identification for this set of intersection lines.
            parent_of_intersection_lines[inds] = line + num_line_parent_counter
            # Increase the counter.
            num_line_parent_counter += unique_parent.shape[0]
        # Done with the intersection line processing.

        # Find intersection points: These by definition lie on the boundary of
        # intersection lines, so we loop over the latter, store their boundary points
        # and identify which points occur more than once.
        points_of_intersection_lines = []
        for li, line in enumerate(intersection_lines):
            if num_parents[li] < 2:
                # At most one of the parents was not a constraint. This line should not
                # produce a point.
                continue
            for bp in gmsh.model.get_boundary([(1, line)]):
                points_of_intersection_lines.append(bp[1])

        num_point_occ = np.bincount(points_of_intersection_lines)
        # Identify all points that occur more than once.
        all_intersection_points = np.where(num_point_occ > 1)[0]
        # Filter away those points that lie on the domain boundary.
        intersection_points = [
            pt
            for pt in all_intersection_points
            if not self._entity_on_domain_boundary(0, [pt])
        ]

        return (
            intersection_points,
            intersection_lines.tolist(),
            isect_mapping,
            num_parents.tolist(),
            constraints,
            line_parents,
            gmsh_to_porepy_fracture_ind_map,
        )

    def _update_mesh_size_control_point(
        self,
        mesh_control_dict: dict[int, list[tuple[np.ndarray, float]]],
        isect_mapping: list,
        surface_tags: list[int],
        boundary_tags: list[int],
    ) -> dict[int, list[tuple[np.ndarray, float]]]:
        """Update the mesh size control points after intersection.

        Parameters:
            mesh_control_dict: Mapping from gmsh tags of surfaces to mesh size control
                points before intersection removal.
            isect_mapping: Mapping from original fractures to new surfaces after
                intersection removal.
            surface_tags: List of gmsh tags of all surfaces (fractures and boundaries)
                after intersection removal.
            boundary_tags: List of gmsh tags of all boundary surfaces after
                intersection removal.

        """
        # Transfer mesh size points to the new segments after intersection removal.
        new_mesh_control_dict: dict[int, list[tuple[np.ndarray, float]]] = {}
        for fi, old_fracture in enumerate(isect_mapping):
            if len(old_fracture) > 0:
                if old_fracture[0][0] == self.nd:
                    # It is unclear if processing the domain will make any harm, but
                    # there is no need to take any chances. Skip it.
                    continue

            # Get hold of the gmsh tag used to represent this fracture before
            # intersection removal.
            old_gmsh_tag = surface_tags[fi]
            for sub_frac in old_fracture:
                if old_gmsh_tag in mesh_control_dict:
                    # Update the mesh size points for the new segments.
                    new_mesh_control_dict[sub_frac[1]] = mesh_control_dict[old_gmsh_tag]

        # We also need to transfer the mesh size points for the boundary surfaces.
        # Identification of these is not straightforward since, if we had included them
        # in the fragmentation (processing of intersections), gmsh would occasionally
        # have failed to identify intersections between boundary surfaces and fractures.
        # Therefore, we do an identification based on the geometry of the surfaces:
        bnd_tag = gmsh.model.get_boundary(
            gmsh.model.get_entities(self.nd), oriented=False
        )
        new_mesh_control_dict.update({t[1]: [] for t in bnd_tag})
        for bnd in boundary_tags:
            info = mesh_control_dict.get(bnd)
            if info is None or len(info) == 0:  # Check for None to please mypy.
                # No points were assigned to this boundary surface, so there is nothing
                # to transfer.
                continue
            points = np.array([p[0] for p in info]).T

            for new_bnd in bnd_tag:
                for i in range(points.shape[1]):
                    if gmsh.model.is_inside(*new_bnd, points[:, i]):
                        new_mesh_control_dict[new_bnd[1]].append(
                            (points[:, i], info[i][1])
                        )

        return new_mesh_control_dict

    def _set_mesh_size_fields(
        self,
        mesh_size_computer: MeshSizeComputer,
        mesh_size_points: dict[int, list[tuple[np.ndarray, float]]],
        intersection_lines: list[int],
        intersection_line_parents: list[list[int]],
        fracture_to_surface: dict[int, list[int]],
        restrict_to_fractures: bool,
    ) -> list:
        nd = 3

        ### Get hold of lines representing fractures and boundaries.
        domain_entities = gmsh.model.get_entities(nd)
        # Get the boundaries.
        boundaries = gmsh.model.get_boundary([(nd, tag) for _, tag in domain_entities])

        surface_tags = set(tag for _, tag in gmsh.model.get_entities(nd - 1))
        boundary_tags = set(tag for _, tag in boundaries)
        # It seems that, if there is more than one domain entity (the domain is split
        # into parts by fractures), the call to get_boundary will only return the outer
        # boundary, and not the internal boundaries between the domain parts. If this is
        # not the case, there will be duplicate boundaries in the list. Compare the list
        # to the set to detect this. If the assertion ever fails, we will need to be
        # more careful in the identification of boundaries, perhaps by sending in one
        # domain entity at a time to get_boundary and then postprocess the results.
        assert len(boundaries) == len(boundary_tags)

        # The list of gmsh fields created.
        gmsh_fields = []

        # All mesh size control points should have been inserted already, so we just
        # need to set up the fields and associate them with the points by the gmsh
        # indices. To that end, create a list of all physical point coordinates.
        gmsh_point_finder = GmshPointIdentifier()

        # The same point might have been inserted multiple times (e.g., if it lies at
        # the intersection of multiple fractures). We need to uniquify the point set,
        # and assign the minimum mesh size among all occurrences of the point.
        self._uniquify_mesh_size_dictionary(mesh_size_points)

        # For lines with no extra mesh size control points, assign an empty list.
        mesh_size: dict[int, list[tuple[np.ndarray, float]]] = {
            tag: [] for tag in surface_tags
        }
        mesh_size.update(mesh_size_points)

        def dist_other_lines(lines, this_line, default_size):
            """Compute distance from this_line to all other lines in lines."""
            other_lines = [l for l in lines if l != this_line]
            if len(other_lines) == 0:
                return 0

            distances = np.array(
                [
                    gmsh.model.occ.get_distance(1, this_line, 1, l)[0]
                    for l in other_lines
                ]
            )
            distances = distances[distances > self._tol]
            if len(distances) == 0:
                return default_size
            return float(np.min(distances))

        def dist_point_lines(lines, point):
            """Compute distance from point to all lines in lines."""
            distances = np.array(
                [gmsh.model.occ.get_distance(0, point, 1, l)[0] for l in lines]
            )
            return np.min(distances)

        for surface, info in mesh_size.items():
            # Boundary of the current surface.
            boundary_lines = [
                t
                for _, t in gmsh.model.get_boundary([(nd - 1, surface)], oriented=False)
            ]
            # Find all intersection lines that are part of this surface.
            surface_is_parent = np.zeros(len(intersection_lines), dtype=bool)
            for li, par in enumerate(intersection_line_parents):
                for fi in par:
                    if surface in fracture_to_surface.get(fi, []):
                        surface_is_parent[li] = True
                        break

            if surface_is_parent.size > 0:
                surface_lines = np.asarray(intersection_lines)[
                    surface_is_parent
                ].tolist()
            else:
                surface_lines = []
            # Distance to other objects for each point, as computed previously.
            h_end = mesh_size_computer.h_end(surface in boundary_tags)

            # Points on intersection lines. Since intersection of lines should result in
            # the line being split, the line points should also contain such
            # intersection points. Moreover, both lines and points should be embedded in
            # the surface during the fragmentation process, hence the mesh size set here
            # will be enforced during suface meshing.
            line_point_list = []
            line_dist = []
            for line in surface_lines:
                d = dist_other_lines(surface_lines + boundary_lines, line, h_end)
                line_dist += [d, d]
                bnd_pts = gmsh.model.get_boundary([(1, line)], oriented=False)

                for p in bnd_pts:
                    line_point_list.append(gmsh.model.occ.get_bounding_box(0, p[1])[:3])

            line_points = np.array(line_point_list).T
            if line_points.size == 0:
                line_points = np.empty((3, 0))
                line_dist = []

            # We need to detect lines that are close.

            control_points = (
                np.array([d[0] for d in info]).T if len(info) > 0 else np.empty((3, 0))
            )
            control_point_internal_distances = (
                np.array([d[1] for d in info]) if len(info) > 0 else np.empty((0,))
            )
            control_point_distance_to_lines = []
            for cp_d in info:
                pi = gmsh_point_finder.index(cp_d[0])
                d = dist_point_lines(surface_lines + boundary_lines, pi)
                if d < self._tol:
                    # For intersections, we assign the background mesh size for this
                    # surface.
                    control_point_distance_to_lines.append(h_end)
                else:
                    control_point_distance_to_lines.append(d)

            control_point_distance = np.minimum(
                control_point_internal_distances,
                np.asarray(control_point_distance_to_lines),
            )

            points, _, ind_map = pp.array_operations.uniquify_point_set(
                np.hstack((line_points, control_points)),
                tol=mesh_size_computer.h_min(),
            )

            other_object_distances_all = np.hstack(
                (
                    np.array([d if d > 0 else h_end for d in line_dist]),
                    np.array([d if d > 0 else h_end for d in control_point_distance]),
                )
            )
            # Reduce to one distance per unique point, picking the minimum distance if
            # multiple distances were associated with the same geometric point.
            other_object_distances = []
            for i in range(points.shape[1]):
                inds = ind_map == i
                min_dist = np.min(other_object_distances_all[inds])
                other_object_distances.append(min_dist)

            if points.shape[1] > 1:
                # If there is more than one point in addition to the end points, we can
                # compute the point-point distances in pairs along this line.
                point_point_distances = pp.distances.pointset(points, max_diag=True)
                min_dist_point = np.min(point_point_distances, axis=0)
            else:
                # If there is a single point, the point-point distance will return 0
                # even with the diag exclusion. In this case, we just set the distance
                # to a large value, so that the distance to other objects is the one
                # that is used.
                min_dist_point = 2 * np.asarray(other_object_distances) + 1

            # The final distance to be used for mesh size calculation is the minimum of
            # the distance to other objects and the distance to other close points on
            # the same line.
            dist = np.minimum(np.asarray(other_object_distances), min_dist_point)
            # Create the mesh size field for this surface.
            gmsh_fields += self._assign_distance_based_mesh_size_field(
                surface,
                points,
                dist,
                mesh_size_computer,
                gmsh_point_finder,
                surface in boundary_tags,
                restrict_to_fractures,
                surface_lines,
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
        intersection_lines: list[int],
        num_parents_of_lines: list[int],
        isect_mapping: list,
        gmsh_to_porepy_fracture_ind_map: dict[int, int],
        constraints: np.ndarray,
    ) -> dict[int, list[int]]:
        """Set physical names in gmsh for fractures and intersections.

        Parameters:
            intersection_points: Gmsh tags of intersection points.
            intersection_lines: Gmsh tags of intersection lines.
            num_parents_of_lines: Number of parent fractures for each intersection line.
            isect_mapping: Mapping from original fractures to new surfaces after
                intersection removal.
            gmsh_to_porepy_fracture_ind_map: Mapping from gmsh fracture tags to input
                fracture indices.
            constraints: List of fracture indices that are constraints.

        Returns:
            Mapping from input fracture indices to lists of gmsh surface tags
            representing the fractures.

        """

        # Intersection points.
        for i in intersection_points:
            gmsh.model.addPhysicalGroup(
                0, [i], -1, f"{PhysicalNames.FRACTURE_INTERSECTION_POINT.value}{i}"
            )

        # Intersection lines.
        for li, line in enumerate(intersection_lines):
            if num_parents_of_lines[li] < 2:
                continue

            gmsh.model.addPhysicalGroup(
                1,
                [int(line)],
                -1,
                f"{PhysicalNames.FRACTURE_INTERSECTION_LINE.value}{li}",
            )

        # Mapping from (sub) fractures to surfaces.
        fracture_to_surface = self._subfracture_to_fracture_mapping(
            isect_mapping, gmsh_to_porepy_fracture_ind_map
        )

        for i, frac in fracture_to_surface.items():
            if i in constraints:
                gmsh.model.addPhysicalGroup(
                    self.nd - 1, frac, -1, f"{PhysicalNames.AUXILIARY_PLANE.value}{i}"
                )

            else:
                gmsh.model.addPhysicalGroup(
                    self.nd - 1, frac, -1, f"{PhysicalNames.FRACTURE.value}{i}"
                )

        if self.domain is not None:
            # It turns out that if fractures split the domain into disjoint parts, gmsh
            # may choose to redefine the domain as the sum of these parts. Therefore, we
            # redefine the domain tags here, using all volumes in the model.
            domain_tags = [entity[1] for entity in gmsh.model.get_entities(self.nd)]
            gmsh.model.addPhysicalGroup(
                self.nd, domain_tags, -1, f"{PhysicalNames.DOMAIN.value}"
            )
        return fracture_to_surface

    def __repr__(self) -> str:
        s = (
            f"Three-dimensional fracture network with "
            f"{str(len(self.fractures))} plane fractures.\n"
        )
        if self.domain is not None:
            s += f"The domain is a {(str(self.domain)).lower()}"

        return s

    def to_file(
        self,
        file_name: Path,
        data: Optional[dict[str, Union[np.ndarray, list]]] = None,
        **kwargs,
    ) -> None:
        """Export the fracture network to a file.

        The file format is given as a keyword argument and by default ``vtu`` will is
        used. The writing is outsourced to meshio, thus the file format should be
        supported by that package.

        The fractures are treated as lines, with no special treatment of intersections.

        Fracture numbers are always exported (1-offset). In addition, it is possible
        to export additional data, as specified by the keyword-argument data.

        Parameters:
            file_name: Name of the target file.
            data: ``default=None``

             Data associated with the fractures. The values in the dictionary should
             be numpy arrays. 1D and 3D data is supported. Fracture numbers are
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

        Raises:
            NotImplementedError: If one of the fractures is elliptic.

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

        # fracture points
        meshio_pts = np.empty((0, 3))

        # counter for the points
        pts_pos = 0

        # Data structure for cells in meshio format.
        meshio_cells = []
        # we operate fracture by fracture
        for frac in self.fractures:
            if isinstance(frac, pp.EllipticFracture):
                raise NotImplementedError(
                    "Exporting elliptic fractures to file is not implemented."
                )

            # In old meshio, polygonal cells are distinguished based on the number of
            # vertexes. save the points of the fracture
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

        path = folder_name / file_name
        meshio.write(path, meshio_grid_to_export, binary=binary)

    def to_csv(
        self,
        file_name: Path,
        write_header: bool = True,
    ) -> None:
        """Save the 3D network on a CSV file with comma as separator.

        The format is as follows:

            - If :attr:`domain` is given ,the first line describes the domain as a
              cuboid ``X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX``.
            - The other lines describe the ``N`` fractures as a list of points
              ``P0_X, P0_Y, P0_Z, ..., PN_X, PN_Y, PN_Z``.

        Warning:
            If ``file_name`` is already present, it will be overwritten without
            prompting any warning.

        Parameters:
            file_name: Name of the CSV file.
            write_header: ``default=True``

                Flag for writing headers for the five columns in the first row.

        """
        file_name = file_name.with_suffix(".csv")

        # Delete the file 'csv_file' if it exists. This seems to be necessary to run
        # tests on GH actions.
        if file_name.exists():
            file_name.unlink()

        with open(file_name, "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            if write_header:
                csv_writer.writerow("# Fracture network exported from porepy.")
                csv_writer.writerow(
                    "# The first line may contain a 6-item bounding box for the domain"
                    " in the format X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX."
                )
                csv_writer.writerow(
                    "# The following lines contain the fractures, described "
                    "either as point sets or as elliptic shapes."
                )
            if self.domain is not None:
                order = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
                # Write the domain bounding box.
                csv_writer.writerow([self.domain.bounding_box[o] for o in order])

            # write all the fractures
            for f in self.fractures:
                if isinstance(f, pp.EllipticFracture):
                    data = [
                        f.center[0],
                        f.center[1],
                        f.center[2],
                        f.r1,
                        f.r2,
                        f.major_axis_angle,
                        f.strike_angle,
                        f.dip_angle,
                    ]
                    csv_writer.writerow(data)
                else:
                    csv_writer.writerow(f.pts.ravel(order="F"))
