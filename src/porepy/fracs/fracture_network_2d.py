"""
Module contains class for representing a fracture network in a 2d domain.
"""
from __future__ import annotations

import copy
import csv
import logging
import time
from typing import Optional, Tuple

import meshio
import numpy as np

import porepy as pp
import porepy.fracs.simplex
from porepy.fracs import tools
from porepy.fracs.line_fracture import LineFracture
from porepy.fracs.utils import linefractures_to_pts_edges, pts_edges_to_linefractures

from .gmsh_interface import GmshData2d, GmshWriter
from .gmsh_interface import Tags as GmshInterfaceTags

logger = logging.getLogger(__name__)


class FractureNetwork2d:
    """Representation of a set of line fracture in a two-dimensional domain.

    The fractures are represented by line fracture objects (see
    :class:`~porepy.fracs.line_fracture.LineFracture`. Poly-line fractures are
    currently not supported. There is no requirement or guarantee that the fractures
    are contained within the specified domain. The fractures can be cut to a given
    domain by the function constrain_to_domain().The domain can be a general
    non-convex polygon (see e.g., :class:`~porepy.geometry.domain.Domain`).

    Note:
        The class is mainly intended for representation and meshing of a fracture
        network, however, it also contains some utility functions. The balance
        between these components may change in the future, specifically, utility
        functions may be removed.

    Parameters:
        fractures: Fractures that make up the network. Defaults to None, which will
            create a domain wihtout fractures. An empty list can also be passed,
            but it will effectively treated as None.
        domain: An instance of :class:`~porepy.geometry.domain.Domain` representing
            the domain. Can be box-shaped or a general (non-convex) polygon.
        tol: Tolerance used in geometric computations.

    """

    def __init__(
        self,
        fractures: Optional[list[LineFracture]] = None,
        domain: Optional[pp.Domain] = None,
        tol: float = 1e-8,
    ) -> None:
        self._pts: np.ndarray
        """Start and endpoints of the fractures. Points can be shared by fractures."""

        self._edges: np.ndarray
        """The fractures as an array of start and end points, referring to ``_pts``

        Additional rows are optional tags of the fractures. In the standard form, the
        third row (first row of tags) identifies the type of edges, referring to the
        numbering system in GmshInterfaceTags. The second row of tags keeps track of the
        numbering of the edges (referring to the original order of the edges) in
        geometry processing like intersection removal. Additional tags can be assigned
        by the user.
        """

        self.tol = tol
        """Tolerance used in geometric computations."""

        self.fractures = [] if fractures is None else fractures
        """List of fractures.

        Internally transformed to points and edges.
        """
        if fractures is not None and len(fractures) > 0:
            self._pts, self._edges = linefractures_to_pts_edges(
                self.fractures, self.tol
            )
        else:
            self._pts = np.zeros((2, 0))
            self._edges = np.zeros((2, 0), dtype=int)

        self.domain: pp.Domain | None = domain
        """The domain for this fracture network."""

        self.tags: dict[int | str, np.ndarray] = dict()
        """Tags for the fractures."""
        # TODO: The current system of tags is a bit confusing, there is both self.tags
        # and the tags located in self.edges. The latter is used for the gmsh interface,
        # and there may be inconsistencies in the transfer of information between the
        # two systems.

        self.bounding_box_imposed: bool = False
        """Flag indicating whether the bounding box has been imposed."""

        self._decomposition: dict = dict()
        """Dictionary of geometric information obtained from the meshing process.

        This will include intersection points identified.
        """

        # -----> Logging
        if self.fractures is not None:
            logger.info("Generated empty fracture set")
        elif self._pts is not None and self._edges is not None:
            # Note: If the list of fracture is empty, we end up here.
            logger.info(f"Generated a fracture set with {self.num_frac()} fractures")
            if self._pts.size > 0:
                logger.info(
                    f"Minimum point coordinates x: {self._pts[0].min():.2f}, \
                        y: {self._pts[1].min():.2f}",
                )
                logger.info(
                    f"Maximum point coordinates x: {self._pts[0].max():.2f}, \
                        y: {self._pts[1].max():.2f}",
                )
        else:
            raise ValueError(
                "Specify both points and connections for a 2d fracture network."
            )
        if domain is not None:
            logger.info(f"Domain specification : {str(domain)}")

    def add_network(self, fs):
        """Add this fracture set to another one, and return a new set.

        The new set may contain non-unique points and edges.

        It is assumed that the domains, if specified, are on a dictionary form.

        WARNING: Tags, in FractureSet.edges[2:] are preserved. If the two sets have
        different set of tags, the necessary rows and columns are filled with the value
        ``-1``, which equals no tag.
        TODO: This function is not being used. Consider deleting it.

        Parameters:
            fs (FractureSet): Another set to be added

        Returns:
            New fracture set, containing all points and edges in both self and
                fs, and the union of the domains.

        """
        logger.info("Add fracture sets: ")
        logger.info(str(self))
        logger.info(str(fs))

        p = np.hstack((self._pts, fs._pts))
        e = np.hstack((self._edges[:2], fs._edges[:2] + self._pts.shape[1]))
        tags = {}
        # copy the tags of the first network
        for key, value in self.tags.items():
            fs_tag = fs.tags.get(key, [None] * fs._edges.shape[1])
            tags[key] = np.hstack((value, fs_tag))
        # copy the tags of the second network
        for key, value in fs.tags.items():
            if key not in tags:
                tags[key] = np.hstack(([None] * self._edges.shape[1], value))

        # Deal with tags
        # Create separate tag arrays for self and fs, with 0 rows if no tags exist
        if self._edges.shape[0] > 2:
            self_tags = self._edges[2:]
        else:
            self_tags = np.empty((0, self.num_frac()))
        if fs._edges.shape[0] > 2:
            fs_tags = fs._edges[2:]
        else:
            fs_tags = np.empty((0, fs.num_frac()))
        # Combine tags
        if self_tags.size > 0 or fs_tags.size > 0:
            n_self = self_tags.shape[0]
            n_fs = fs_tags.shape[0]
            if n_self < n_fs:
                extra_tags = np.full((n_fs - n_self, self.num_frac()), -1, dtype=int)
                self_tags = np.vstack((self_tags, extra_tags))
            elif n_self > n_fs:
                extra_tags = np.full((n_self - n_fs, fs.num_frac()), -1, dtype=int)
                fs_tags = np.vstack((fs_tags, extra_tags))
            tags = np.hstack((self_tags, fs_tags)).astype(int)
            e = np.vstack((e, tags))

        if self.domain is not None and fs.domain is not None:
            xmin = np.minimum(
                self.domain.bounding_box["xmin"], fs.domain.bounding_box["xmin"]
            )
            ymin = np.minimum(
                self.domain.bounding_box["ymin"], fs.domain.bounding_box["ymin"]
            )
            xmax = np.maximum(
                self.domain.bounding_box["xmax"], fs.domain.bounding_box["xmax"]
            )
            ymax = np.maximum(
                self.domain.bounding_box["ymax"], fs.domain.bounding_box["ymax"]
            )
            new_bounding_box = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
            domain = pp.Domain(new_bounding_box)
        elif self.domain is not None:
            domain = self.domain
        elif fs.domain is not None:
            domain = fs.domain
        else:
            domain = None

        fracs = pts_edges_to_linefractures(p, e)
        fn = FractureNetwork2d(fracs, domain, self.tol)
        fn.tags = tags
        return fn

    def mesh(
        self,
        mesh_args,
        tol: Optional[float] = None,
        do_snap: bool = True,
        constraints: Optional[np.ndarray] = None,
        file_name: Optional[str] = None,
        dfn: bool = False,
        tags_to_transfer: Optional[list[str]] = None,
        remove_small_fractures: bool = False,
        write_geo: bool = True,
        finalize_gmsh: bool = True,
        clear_gmsh: bool = False,
        **kwargs,
    ) -> pp.MixedDimensionalGrid:
        """Create a mixed-dimensional grid for this fracture network.

        Parameters:
            mesh_args: Arguments passed on to mesh size control.
            tol (double, optional): Tolerance used for geometric computations.
                Defaults to the tolerance of this network.
            do_snap (boolean, optional): Whether to snap lines to avoid small
                segments. Defaults to True.
            constraints (np.array of int): Index of network edges that should not
                generate lower-dimensional meshes, but only act as constraints in
                the meshing algorithm.
            dfn (boolean, optional): If True, a DFN mesh (of the network, but not
                the surrounding matrix) is created.
            tags_to_transfer (list of key, optional default None): The tags of
                the network are passed to the fracture grids.
            remove_small_fractures
                Whether to remove small fractures. FIXME: expand documentation.
            write_geo (bool, optional): If True (default), the gmsh configuration
                will be written to a .geo_unrolled file.
            finalize_gmsh (boolean): If True (default), the port to Gmsh is closed when
                meshing is completed. On repeated invocations of Gmsh in the same Python
                session, a memory leak in Gmsh may cause reduced performance (written
                spring 2021). In these cases, it may be better to finalize gmsh externally
                to this class. See also clear_gmsh.
            clear_gmsh (boolean, optional): If True, the geometry representation in gmsh
                is deleted when meshing is completed. This is of use only if finalize_gmsh
                is set to False, in which case it may be desirable to delete the old
                geometry before adding a new one. Defaults to False.

        Returns:
            MixedDimensionalGrid: Mixed-dimensional grid.

        """
        if file_name is None:
            file_name = "gmsh_frac_file.msh"
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
            # Create list of grids
            subdomains = porepy.fracs.simplex.line_grid_from_gmsh(
                file_name, constraints=constraints
            )

        else:
            # Create list of grids
            subdomains = porepy.fracs.simplex.triangle_grid_from_gmsh(
                file_name, constraints=constraints
            )

        if tags_to_transfer:
            # preserve tags for the fractures from the network
            # we are assuming a coherent numeration between the network
            # and the created grids
            frac = np.setdiff1d(
                np.arange(self._edges.shape[1]), constraints, assume_unique=True
            )
            for idg, g in enumerate(subdomains[1 - int(dfn)]):
                for key in np.atleast_1d(tags_to_transfer):
                    if key not in g.tags:
                        g.tags[key] = self.tags[key][frac][idg]

        # Assemble in grid bucket
        return pp.meshing.subdomains_to_mdg(subdomains, **kwargs)

    def prepare_for_gmsh(
        self,
        mesh_args: dict,
        tol: Optional[float] = None,
        do_snap: bool = True,
        constraints: Optional[np.ndarray] = None,
        dfn: bool = False,
        remove_small_fractures: bool = False,
    ) -> GmshData2d:
        """Process network intersections and write a gmsh .geo configuration file,
        ready to be processed by gmsh.

        NOTE: Consider using the mesh() function instead to get a ready
        MixedDimensionalGrid.

        Parameters:
            mesh_args: Arguments passed on to mesh size control.
            tol (double, optional): Tolerance used for geometric computations.
                Defaults to the tolerance of this network.
            do_snap (boolean, optional): Whether to snap lines to avoid small
                segments. Defaults to True.
            constraints (np.array of int): Index of network edges that should not
                generate lower-dimensional meshes, but only act as constraints in
                the meshing algorithm.
            dfn (boolean, optional): If True, a DFN mesh (of the network, but not
                the surrounding matrix) is created.
            remove_small_fractures: Whether to remove small fractures.
                FIXME: expand documentation.

        Returns:
            MixedDimensionalGrid: Mixed-dimensional mesh.

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
            to_delete = np.where(self.length() < tol)[0]
            self._edges = np.delete(self._edges, to_delete, axis=1)

            # remove also the fractures in the tags
            for key, value in self.tags.items():
                self.tags[key] = np.delete(value, to_delete)

        if not self.bounding_box_imposed:
            edges_kept, edges_deleted = self.impose_external_boundary(
                self.domain, add_domain_edges=not dfn
            )
            # Find edges of constraints to delete
            to_delete = np.where(np.isin(constraints, edges_deleted))[0]

            # Adjust constraint indices: Must be decreased for all deleted lines with
            # lower index, and increased for all lines with lower index that have been
            # split.
            adjustment = np.zeros(num_edge_orig, dtype=int)
            # Deleted edges give an index reduction of 1
            adjustment[edges_deleted] = -1

            # identify edges that have been split
            num_occ = np.bincount(edges_kept, minlength=adjustment.size)

            # Not sure what to do with split constraints; it should not be difficult,
            # but the current implementation does not cover it.
            assert np.all(num_occ[constraints] < 2)

            # Splitting of fractures give an increase of index corresponding to the number
            # of repeats. The clip avoids negative values for deleted edges, these have
            # been accounted for before. Maybe we could merge the two adjustments.
            adjustment += np.clip(num_occ - 1, 0, None)

            # Do the real adjustment
            constraints += np.cumsum(adjustment)[constraints]

            # Delete constraints corresponding to deleted edges
            constraints = np.delete(constraints, to_delete)

            # FIXME: We do not keep track of indices of fractures and constraints
            # before and after imposing the boundary.

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
        self._pts, _, old_2_new = pp.utils.setmembership.uniquify_point_set(
            self._pts, tol=self.tol
        )
        self._edges = old_2_new[self._edges]
        self._decomposition["domain_boundary_points"] = old_2_new[
            self._decomposition["domain_boundary_points"]
        ]

        # map the constraint index
        index_map = np.where(np.logical_not(to_delete))[0]
        mapped_constraints = np.arange(index_map.size)[np.in1d(index_map, constraints)]

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
        # formats: In the geometry processing undertaken up to this point, it has been
        # convenient to use numerical values for identifying the different line types
        # (fracture, constraint, boundary). For the Gmsh processing, a string-based
        # system is used, as this is more readable and closer to the system employed in
        # Gmsh. In practice, this requires translating from GmshInterfaceTags values to
        # names.
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

        FIXME: update documentation
        Parameters
        ----------
        constraints
            Indices of Fractures which should be considered meshing constraints,
            not as physical objects.

        Returns
        -------

        """
        points = self._pts
        edges = self._edges

        if not np.all(np.diff(edges[:2], axis=0) != 0):
            raise ValueError("Found a point edge in splitting of edges")

        tags = np.zeros((2, edges.shape[1]), dtype=int)

        tags[0][
            np.logical_not(self.tags["boundary"])
        ] = GmshInterfaceTags.FRACTURE.value
        tags[0][self.tags["boundary"]] = GmshInterfaceTags.DOMAIN_BOUNDARY_LINE.value
        tags[0][constraints] = GmshInterfaceTags.AUXILIARY_LINE.value

        tags[1] = np.arange(edges.shape[1])

        edges = np.vstack((edges, tags))

        # Ensure unique description of points
        pts_all, _, old_2_new = pp.utils.setmembership.uniquify_point_set(
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
        # This may disturb the line tags in lines[2], but we should not be
        # dependent on those.
        li = np.sort(lines[:2], axis=0)
        _, new_2_old, old_2_new = pp.utils.setmembership.unique_columns_tol(
            li, tol=self.tol
        )
        lines = lines[:, new_2_old]

        if not np.all(np.diff(lines[:2], axis=0) != 0):
            raise ValueError(
                "Found a point edge in splitting of edges after merging points"
            )

        # We split all fracture intersections so that the new lines do not
        # intersect, except possible at the end points
        logger.info("Remove edge crossings")
        tm = time.time()

        pts_split, lines_split, *_ = pp.intersections.split_intersecting_segments_2d(
            pts_all, lines, tol=self.tol
        )
        logger.info("Done. Elapsed time " + str(time.time() - tm))

        # Ensure unique description of points
        pts_split, _, old_2_new = pp.utils.setmembership.uniquify_point_set(
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

        # TODO: This operation may leave points that are not referenced by any
        # lines. We should probably delete these.

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

        frac_id = np.ravel(lines[:2, lines[2] == GmshInterfaceTags.FRACTURE.value])
        _, frac_ia, frac_count = np.unique(frac_id, True, False, True)

        # In the case we have auxiliary points remove do not create a 0d point in
        # case one intersects a single fracture. In the case of multiple fractures intersection
        # with an auxiliary point do consider the 0d.
        aux_id = np.logical_or(
            lines[2] == GmshInterfaceTags.AUXILIARY_LINE.value,
            lines[2] == GmshInterfaceTags.DOMAIN_BOUNDARY_LINE.value,
        )
        if np.any(aux_id):
            aux_id = np.ravel(lines[:2, aux_id])
            _, aux_ia, aux_count = np.unique(aux_id, True, False, True)

            # It can probably be done more efficiently, but currently we rarely use the
            # auxiliary points in 2d
            for a in aux_id[aux_ia[aux_count > 1]]:
                # if a match is found decrease the frac_count only by one, this prevents
                # the multiple fracture case to be handled wrongly
                frac_count[frac_id[frac_ia] == a] -= 1

        return frac_id[frac_ia[frac_count > 1]]

    def _insert_auxiliary_points(
        self,
        mesh_size_frac: Optional[float] = None,
        mesh_size_bound: Optional[float] = None,
        mesh_size_min: Optional[float] = None,
    ) -> None:
        # Mesh size
        # Tag points at the domain corners
        logger.info("Determine mesh size")
        tm = time.time()

        p = self._decomposition["points"]
        lines = self._decomposition["edges"]
        boundary_pt_ind = self._decomposition["domain_boundary_points"]

        mesh_size, pts_split, lines = tools.determine_mesh_size(
            p,
            boundary_pt_ind,
            lines,
            mesh_size_frac=mesh_size_frac,
            mesh_size_bound=mesh_size_bound,
            mesh_size_min=mesh_size_min,
        )

        logger.info("Done. Elapsed time " + str(time.time() - tm))

        self._decomposition["points"] = pts_split
        self._decomposition["edges"] = lines
        self._decomposition["mesh_size"] = mesh_size

    def _set_mesh_size_without_auxiliary_points(
        self,
        mesh_size_frac: Optional[float] = None,
        mesh_size_bound: Optional[float] = None,
        mesh_size_min: Optional[float] = None,
    ) -> None:
        """
        Set the "vanilla" mesh size to points. No attempts at automatically
        determine the mesh size is done and no auxiliary points are inserted.
        Fracture points are given the mesh_size_frac mesh size and the domain
        boundary is given the mesh_size_bound mesh size. mesh_size_min is unused.
        """
        # Gridding size
        # Tag points at the domain corners
        logger.info("Determine mesh size")
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
        logger.info("Done. Elapsed time " + str(time.time() - tm))
        self._decomposition["mesh_size"] = vals

    def impose_external_boundary(
        self,
        domain: Optional[pp.Domain] = None,
        add_domain_edges: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constrain the fracture network to lie within a domain.

        Fractures outside the imposed domain will be deleted.

        The domain will be added to self._pts and self.edges, if add_domain_edges is
        True.
        The domain boundary edges can be identified from self.tags['boundary'].

        Args:
            domain (dict or np.array, optional): Domain. See __init__ for description.
                if not provided, self.domain will be used.
            add_domain_edges(bool, optional): Include or not the boundary edges and _pts
            in the list of edges. Default value True.

        Returns:
            edges_deleted (np.array): Index of edges that were outside the bounding box
                and therefore deleted.

        """
        if domain is not None:
            # First create lines that define the domain
            x_min = domain.bounding_box["xmin"]
            x_max = domain.bounding_box["xmax"]
            y_min = domain.bounding_box["ymin"]
            y_max = domain.bounding_box["ymax"]
            dom_p: np.ndarray = np.array(
                [[x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max]]
            )
            dom_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]).T

        # Constrain the edges to the domain
        p, e, edges_kept = pp.constrain_geometry.lines_by_polygon(
            dom_p, self._pts, self._edges
        )

        # Special case where an edge has one point on the boundary of the domain,
        # the other outside the domain. In this case the edge should be removed.
        # The edge will have been cut so that the endpoints coincide. Look for
        # such edges
        _, _, n2o = pp.utils.setmembership.uniquify_point_set(p, self.tol)
        reduced_edges = n2o[e]
        not_point_edge = np.diff(reduced_edges, axis=0).ravel() != 0

        # The point involved in point edges may be superfluous in the description
        # of the fracture network; this we will deal with later. For now, simply
        # remove the point edge.
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

        self.bounding_box_imposed = True
        return edges_kept, edges_deleted

    def _snap_fracture_set(
        self,
        pts: np.ndarray,
        snap_tol: float,
        termination_tol: float = 1e-2,
        max_iter: int = 100,
    ) -> Tuple[np.ndarray, bool]:
        """Snap vertexes of a set of fracture lines embedded in 2D, so that small
        distances between lines and vertexes are removed.

        This is intended as a utility function to preprocess a fracture network
            before meshing. The function may change both connectivity and orientation
            of individual fractures in the network. Specifically, fractures that
            almost form a T-intersection (or L), may be connected, while
            X-intersections with very short branches may be truncated to T-intersections.

        The modification snaps vertexes to the closest point on the adjacent line.
            This will in general change the orientation of the fracture with the
            snapped vertex. The alternative, to prolong the fracture along its existing
            orientation, may result in very long fractures for almost intersecting
            lines. Depending on how the fractures are ordered, the same point may
            need to be snapped to a segment several times in an iterative process.

        The algorithm is *not* deterministic, in the sense that if the ordering of
            the fractures is permuted, the snapped fracture network will be slightly
            different.

        Parameters:
            pts (np.array, 2 x n_pts): Array of start and endpoints for fractures.
            snap_tol (double): Snapping tolerance. Distances below this will be
                snapped.
            termination_tol (double): Minimum point movement needed for the
                iterations to continue.
            max_iter (int, optional): Maximum number of iterations. Defaults to
                100.

        Returns:
            np.array (2 x n_pts): Copy of the point array, with modified point
                coordinates.
            boolean: True if the iterations converged within allowed number of
                iterations.

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
            logger.info(
                "Fracture snapping converged after " + str(counter) + " iterations"
            )
            logger.info("Maximum modification " + str(np.max(np.abs(pts - pts_orig))))
            return pts, True
        else:
            logger.warning("Fracture snapping failed to converge")
            logger.warning("Residual: " + str(diff))
            return pts, False

    def _snap_to_boundary(self, snap_tol: float):
        """Snap points to the domain boundary.

        The function modifies self._pts.
        Parameters
        ----------
        snap_tol
            Tolerance. Internal points which are a distance d<snap_tol
            away from the boundary will be snapped to the boundary.

        Returns
        -------

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

    def _edges_overlapping_boundary(self, tol: float):
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
    ---------------------------------
    """

    def _decompose_domain(self, domain, num_x, ny=None):
        x0 = domain["xmin"]
        dx = (domain["xmax"] - domain["xmin"]) / num_x

        if "ymin" in domain.keys() and "ymax" in domain.keys():
            y0 = domain["ymin"]
            dy = (domain["ymax"] - domain["ymin"]) / ny
            return x0, y0, dx, dy
        else:
            return x0, dx

    def constrain_to_domain(self, domain=None):
        """Constrain the fracture network to lay within a specified domain.

        Fractures that cross the boundary of the domain will be cut to lay
        within the boundary. Fractures that lay completely outside the domain
        will be dropped from the constrained description.

        TODO: Also return an index map from new to old fractures.

        Parameters:
            domain (dictionary, None): Domain specification, in the form of a
                dictionary with fields 'xmin', 'xmax', 'ymin', 'ymax'. If not
                provided, the domain of this object will be used.

        Returns:
            FractureNetwork2d: Initialized by the constrained fractures, and the
                specified domain.

        """
        if domain is None:
            domain = self.domain

        p_domain = self._bounding_box_to_points(domain.bounding_box)

        p, e, _ = pp.constrain_geometry.lines_by_polygon(
            p_domain, self._pts, self._edges
        )
        fracs = pts_edges_to_linefractures(p, e)

        return FractureNetwork2d(fracs, domain, self.tol)

    def _bounding_box_to_points(self, box: dict[str, pp.number]) -> np.ndarray:
        """Helper function to convert a bounding box in the form of a dictionary into
        a point set."""

        if isinstance(box, dict):
            p00 = np.array([box["xmin"], box["ymin"]]).reshape((-1, 1))
            p10 = np.array([box["xmax"], box["ymin"]]).reshape((-1, 1))
            p11 = np.array([box["xmax"], box["ymax"]]).reshape((-1, 1))
            p01 = np.array([box["xmin"], box["ymax"]]).reshape((-1, 1))
            point_set = np.hstack((p00, p10, p11, p01))
        else:  # not a good practice, but kept for legacy reasons
            point_set = box

        return point_set

    # Methods for copying fracture network
    def copy(self) -> "pp.FractureNetwork2d":
        """Create deep copy of the network.

        The method will create a deep copy of all fractures, as well as the domain, of
        the network. Note that if the fractures have had extra points imposed as part
        of a meshing procedure, these will be included in the copied fractures.

        Returns:
            pp.FractureNetwork2d.

        See also:
            self.snapped_copy(), self.copy_with_split_intersections()

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

    def snapped_copy(self, tol: float) -> "pp.FractureNetwork2d":
        """Modify point definition so that short branches are removed, and
        almost intersecting fractures become intersecting.

        Parameters:
            tol (double): Threshold for geometric modifications. Points and
                segments closer than the threshold may be modified.

        Returns:
            FractureNetwork2d: A new network with modified point coordinates.

        See also:
            self.copy(), self.copy_with_split_intersections()

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
    ) -> "pp.FractureNetwork2d":
        """Create a new FractureSet, with all fracture intersections removed

        Parameters:
            tol (optional): Tolerance used in geometry computations when
                splitting fractures. Defaults to the tolerance of this network.

        Returns:
            FractureSet: New set, where all intersection points are added so that
                the set only contains non-intersecting branches.

        See also:
            self.copy(), self.snapped_copy()

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

    # --------- Methods for analysis of the fracture set

    def as_graph(self, split_intersections=True):
        """Represent the fracture set as a graph, using the networkx data structure.

        By default, the fractures will first be split into non-intersecting branches.

        Parameters:
            split_intersections (boolean, optional): If True (default), the network
                is split into non-intersecting branches before invoking the graph
                representation.

        Returns:
            networkx.graph: Graph representation of the network, using the networkx
                data structure.
            FractureSet: This fracture set, split into non-intersecting branches.
                Only returned if split_intersections is True

        """
        if split_intersections:
            split_network = self.split_intersections()
            pts = split_network._pts
            edges = split_network._edges
        else:
            edges = self._edges
            pts = self._pts

        import networkx as nx

        G = nx.Graph()
        for pi in range(pts.shape[1]):
            G.add_node(pi, coordinate=pts[:, pi])

        for ei in range(edges.shape[1]):
            tags = {}
            for key, value in split_network.tags.items():
                tags[key] = value[ei]
            G.add_edge(edges[0, ei], edges[1, ei], labels=tags)

        if split_intersections:
            return G, split_network
        else:
            return G

    # --------- Utility functions below here

    def num_frac(self):
        """Return the number of fractures stored"""
        return self._edges.shape[1]

    def _remove_orphan_pts(self):
        """Remove points that are not part of any edge. Modify the numerations
        accordingly.

        """

        pts_id = np.unique(self._edges)
        all_pts_id = np.arange(self._pts.shape[1])

        # determine the orphan points
        to_keep = np.ones(all_pts_id.size, dtype=np.bool)
        to_keep[np.setdiff1d(all_pts_id, pts_id, assume_unique=True)] = False

        # create the map between the old and new
        new_pts_id = -np.ones(all_pts_id.size, dtype=np.int)
        new_pts_id[to_keep] = np.arange(pts_id.size)

        # update the edges numeration
        self._edges = new_pts_id[self._edges]
        # update the points
        self._pts = self._pts[:, pts_id]

        return new_pts_id

    def start_points(self, fi=None):
        """Get start points of all fractures, or a subset.

        Parameters:
            fi (np.array or int, optional): Index of the fractures for which the
                start point should be returned. Either a numpy array, or a single
                int. In case of multiple indices, the points are returned in the
                order specified in fi. If not specified, all start points will be
                returned.

        Returns:
            np.array, 2 x num_frac: Start coordinates of all fractures.

        """
        if fi is None:
            fi = np.arange(self.num_frac())

        p = self._pts[:, self._edges[0, fi]]
        # Always return a 2-d array
        if p.size == 2:
            p = p.reshape((-1, 1))
        return p

    def end_points(self, fi=None):
        """Get start points of all fractures, or a subset.

        Parameters:
            fi (np.array or int, optional): Index of the fractures for which the
                end point should be returned. Either a numpy array, or a single
                int. In case of multiple indices, the points are returned in the
                order specified in fi. If not specified, all end points will be
                returned.

        Returns:
            np.array, 2 x num_frac: End coordinates of all fractures.

        """
        if fi is None:
            fi = np.arange(self.num_frac())

        p = self._pts[:, self._edges[1, fi]]
        # Always return a 2-d array
        if p.size == 2:
            p = p.reshape((-1, 1))
        return p

    def get_points(self, fi=None):
        """Return start and end points for a specified fracture.

        Parameters:
            fi (np.array or int, optional): Index of the fractures for which the
                end point should be returned. Either a numpy array, or a single
                int. In case of multiple indices, the points are returned in the
                order specified in fi. If not specified, all end points will be
                returned.

        Returns:
            np.array, 2 x num_frac: End coordinates of all fractures.
            np.array, 2 x num_frac: End coordinates of all fractures.

        """
        return self.start_points(fi), self.end_points(fi)

    def length(self, fi=None):
        """
        Compute the total length of the fractures, based on the fracture id.
        The output array has length as unique(frac) and ordered from the lower index
        to the higher.

        Parameters:
            fi (np.array, or int): Index of fracture(s) where length should be
                computed. Refers to self.edges

        Return:
            np.array: Length of each fracture

        """
        if fi is None:
            fi = np.arange(self.num_frac())
        fi = np.asarray(fi)

        # compute the length for each segment
        norm = lambda e0, e1: np.linalg.norm(self._pts[:, e0] - self._pts[:, e1])
        length = np.array([norm(e[0], e[1]) for e in self._edges.T])

        # compute the total length based on the fracture id
        tot_l = lambda f: np.sum(length[np.isin(fi, f)])
        return np.array([tot_l(f) for f in np.unique(fi)])

    def orientation(self, fi=None):
        """Compute the angle of the fractures to the x-axis.

        Parameters:
            fi (np.array, or int): Index of fracture(s) where length should be
                computed. Refers to self.edges

        Return:
            angle: Orientation of each fracture, relative to the x-axis.
                Measured in radians, will be a number between 0 and pi.

        """
        if fi is None:
            fi = np.arange(self.num_frac())
        fi = np.asarray(fi)

        # compute the angle for each segment
        alpha = lambda e0, e1: np.arctan2(
            self._pts[1, e0] - self._pts[1, e1], self._pts[0, e0] - self._pts[0, e1]
        )
        a = np.array([alpha(e[0], e[1]) for e in self._edges.T])

        # compute the mean angle based on the fracture id
        mean_alpha = lambda f: np.mean(a[np.isin(fi, f)])
        mean_a = np.array([mean_alpha(f) for f in np.unique(fi)])

        # we want only angles in (0, pi)
        mask = mean_a < 0
        mean_a[mask] = np.pi - np.abs(mean_a[mask])
        mean_a[mean_a > np.pi] -= np.pi

        return mean_a

    def compute_center(self, p=None, edges=None):
        """Compute center points of a set of fractures.

        Parameters:
            p (np.array, 2 x n , optional): Points used to describe the fractures.

                defaults to the fractures in this set.
            edges (np.array, 2 x num_frac, optional): Indices, referring to _pts, of the start
                and end points of the fractures for which the centres should be computed.
                Defaults to the fractures of this set.

        Returns:
            np.array, 2 x num_frac: Coordinates of the centers of this fracture.

        """
        if p is None:
            p = self._pts
        if edges is None:
            edges = self._edges
        # first compute the fracture centres and then generate them
        avg = lambda e0, e1: 0.5 * (np.atleast_2d(p)[:, e0] + np.atleast_2d(p)[:, e1])
        pts_c = np.array([avg(e[0], e[1]) for e in edges.T]).T
        return pts_c

    def bounding_box_measure(self, bounding_box=None):
        """Get the measure (length, area) of a given box domain, specified by its
        extensions stored in a dictionary.

        The dimension of the domain is inferred from the dictionary fields.

        Parameters:
            bounding_box (dictionary, optional): Should contain keys 'xmin' and 'xmax'
                specifying the extension in the x-direction. If the domain is 2d,
                it should also have keys 'ymin' and 'ymax'. If no domain is specified
                the domain of this object will be used.

        Returns:
            double: Measure of the domain.

        """
        if bounding_box is None:
            bounding_box = self.domain.bounding_box
        if "ymin" and "ymax" in bounding_box.keys():
            return (bounding_box["xmax"] - bounding_box["xmin"]) * (
                bounding_box["ymax"] - bounding_box["ymin"]
            )
        else:
            return bounding_box["xmax"] - bounding_box["xmin"]

    def plot(self, **kwargs):
        """Plot the fracture set.

        The function passes this fracture set to PorePy plot_fractures

        Parameters:
            **kwargs: Keyword arguments to be passed on to matplotlib.

        """
        pp.plot_fractures(self._pts, self._edges, domain=self.domain, **kwargs)

    def to_csv(self, file_name: str, with_header=True):
        """
        Save the 2d network on a csv file with comma as separator.
        Note: the file is overwritten if present.
        The format is
        FID, START_X, START_Y, END_X, END_Y

        Parameters
        ----------
        file_name
            Name of the file.
        with_header
            Whether to write headers for the five columns in the first row.
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
        self, file_name: str, data: Optional[dict[str, np.ndarray]] = None, **kwargs
    ) -> None:
        """
        Export the fracture network to file.

        The file format is given as a kwarg, by default vtu will be used. The writing is
        outsourced to meshio, thus the file format should be supported by that package.

        The fractures are treated as lines, with no special treatment
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
        meshio.write(folder_name + file_name, meshio_grid_to_export, binary=binary)

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
