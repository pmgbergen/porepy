"""
Module contains class for representing a fracture network in a 2d domain.
"""
import copy
import csv
import logging
import time

import numpy as np

import porepy as pp
import porepy.fracs.simplex
from porepy.fracs import tools
from porepy.grids import constants
from porepy.grids.gmsh import gmsh_interface
from porepy.utils.setmembership import unique_columns_tol

# Imports of external packages that may not be present at the system. The
# module will work without any of these, but with limited functionalbility.
try:
    import vtk
    import vtk.util.numpy_support as vtk_np
except ImportError:
    import warnings

    warnings.warn(
        "VTK module is not available. Export of fracture network to\
    vtk will not work."
    )


logger = logging.getLogger(__name__)


class FractureNetwork2d(object):
    """Class representation of a set of fractures in a 2D domain.

    The fractures are represented by their endpoints. Poly-line fractures are
    currently not supported. There is no requirement or guarantee that the
    fractures are contained within the specified domain. The fractures can be
    cut to a given domain by the function constrain_to_domain().

    The domain can be a general non-convex polygon.

    IMPLEMENTATION NOTE: The class is mainly intended for representation and meshing of
    a fracture network, however, it also contains some utility functions. The balance
    between these components may change in the future, specifically, utility functions
    may be removed.

    Attributes:
        pts (np.array, 2 x num_pts): Start and endpoints of the fractures. Points
            can be shared by fractures.
        edges (np.array, (2 + num_tags) x num_fracs): The first two rows represent
            indices, refering to pts, of the start and end points of the fractures.
            Additional rows are optional tags of the fractures.
        domain (dictionary or np.ndarray): The domain in which the fracture set is
            defined. If dictionary, it should contain keys 'xmin', 'xmax', 'ymin',
            'ymax', each of which maps to a double giving the range of the domain.
            If np.array, it should be of size 2 x n, and given the vertexes of the.
            domain. The fractures need not lay inside the domain.
        num_frac (int): Number of fractures in the domain.
        tol (double): Tolerance used in geometric computations.
        tags (dict): Tags for fractures.
        decomposition (dict): Decomposition of the fracture network, used for export to
            gmsh, and available for later processing. Initially empty, is created by
            self.mesh().

    """

    def __init__(self, pts=None, edges=None, domain=None, tol=1e-8):
        """Define the frature set.

        Parameters:
            pts (np.array, 2 x n): Start and endpoints of the fractures. Points
            can be shared by fractures.
        edges (np.array, (2 + num_tags) x num_fracs): The first two rows represent
            indices, refering to pts, of the start and end points of the fractures.
            Additional rows are optional tags of the fractures.
        domain (dictionary or set of points): The domain in which the fracture set is
             defined. See self.attributes for description.
        tol (double, optional): Tolerance used in geometric computations. Defaults to
            1e-8.

        """

        if pts is None:
            self.pts = np.zeros((2, 0))
        else:
            self.pts = pts
        if edges is None:
            self.edges = np.zeros((2, 0), dtype=np.int)
        else:
            self.edges = edges
        self.domain = domain
        self.tol = tol

        self.num_frac = self.edges.shape[1]

        self.tags = {}
        self.bounding_box_imposed = False
        self.decomposition = {}

        if pts is None and edges is None:
            logger.info("Generated empty fracture set")
        else:
            logger.info("Generated a fracture set with %i fractures", self.num_frac)
            if pts.size > 0:
                logger.info(
                    "Minimum point coordinates x: %.2f, y: %.2f",
                    pts[0].min(),
                    pts[1].min(),
                )
                logger.info(
                    "Maximum point coordinates x: %.2f, y: %.2f",
                    pts[0].max(),
                    pts[1].max(),
                )
        if domain is not None:
            logger.info("Domain specification :" + str(domain))

    def copy(self):
        """Create deep copy of the network.

        The method will create a deep copy of all fractures, as well as the domain, of
        the network. Note that if the fractures have had extra points imposed as part
        of a meshing procedure, these will included in the copied fractures.

        Returns:
            pp.FractureNetwork3d.

        """
        p_new = np.copy(self.pts)
        edges_new = np.copy(self.edges)
        domain = self.domain
        if domain is not None:
            # Get a deep copy of domain, but no need to do that if domain is None
            domain = copy.deepcopy(domain)

        fn = FractureNetwork2d(p_new, edges_new, domain, self.tol)
        fn.tags = self.tags.copy()
        return fn

    def add_network(self, fs):
        """Add this fracture set to another one, and return a new set.

        The new set may contain non-unique points and edges.

        It is assumed that the domains, if specified, are on a dictionary form.

        WARNING: Tags, in FractureSet.edges[2:] are preserved. If the two sets have different
        set of tags, the necessary rows and columns are filled with what is essentially
        random values.

        Parameters:
            fs (FractureSet): Another set to be added

        Returns:
            New fracture set, containing all points and edges in both self and
                fs, and the union of the domains.

        """
        logger.info("Add fracture sets: ")
        logger.info(str(self))
        logger.info(str(fs))

        p = np.hstack((self.pts, fs.pts))
        e = np.hstack((self.edges[:2], fs.edges[:2] + self.pts.shape[1]))
        tags = {}
        # copy the tags of the first network
        for key, value in self.tags.items():
            fs_tag = fs.tags.get(key, [None] * fs.edges.shape[1])
            tags[key] = np.hstack((value, fs_tag))
        # copy the tags of the second network
        for key, value in fs.tags.items():
            if key not in tags:
                tags[key] = np.hstack(([None] * self.edges.shape[1], value))

        # Deal with tags
        # Create separate tag arrays for self and fs, with 0 rows if no tags exist
        if self.edges.shape[0] > 2:
            self_tags = self.edges[2:]
        else:
            self_tags = np.empty((0, self.num_frac))
        if fs.edges.shape[0] > 2:
            fs_tags = fs.edges[2:]
        else:
            fs_tags = np.empty((0, fs.num_frac))
        # Combine tags
        if self_tags.size > 0 or fs_tags.size > 0:
            n_self = self_tags.shape[0]
            n_fs = fs_tags.shape[0]
            if n_self < n_fs:
                extra_tags = np.empty((n_fs - n_self, self.num_frac), dtype=np.int)
                self_tags = np.vstack((self_tags, extra_tags))
            elif n_self > n_fs:
                extra_tags = np.empty((n_self - n_fs, fs.num_frac), dtype=np.int)
                fs_tags = np.vstack((fs_tags, extra_tags))
            tags = np.hstack((self_tags, fs_tags)).astype(np.int)
            e = np.vstack((e, tags))

        if self.domain is not None and fs.domain is not None:
            domain = {
                "xmin": np.minimum(self.domain["xmin"], fs.domain["xmin"]),
                "xmax": np.maximum(self.domain["xmax"], fs.domain["xmax"]),
                "ymin": np.minimum(self.domain["ymin"], fs.domain["ymin"]),
                "ymax": np.maximum(self.domain["ymax"], fs.domain["ymax"]),
            }
        elif self.domain is not None:
            domain = self.domain
        elif fs.domain is not None:
            domain = fs.domain
        else:
            domain = None

        fn = FractureNetwork2d(p, e, domain, self.tol)
        fn.tags = tags
        return fn

    def mesh(
        self,
        mesh_args,
        tol=None,
        do_snap=True,
        constraints=None,
        file_name=None,
        dfn=False,
        preserve_fracture_tags=None,
        **kwargs,
    ):
        """Create GridBucket (mixed-dimensional grid) for this fracture network.

        Parameters:
            mesh_args: Arguments passed on to mesh size control
            tol (double, optional): Tolerance used for geometric computations.
                Defaults to the tolerance of this network.
            do_snap (boolean, optional): Whether to snap lines to avoid small
                segments. Defults to True.
            constraints (np.array of int): Index of network edges that should not
                generate lower-dimensional meshes, but only act as constraints in
                the meshing algorithm.
            dfn (boolean, optional): If True, a DFN mesh (of the network, but not
                the surrounding matrix) is created.
            preserve_fracture_tags (list of key, optional default None): The tags of
                the network are passed to the fracture grids.

        Returns:
            GridBucket: Mixed-dimensional mesh.

        """
        in_file = self.prepare_for_gmsh(
            mesh_args, tol, do_snap, constraints, file_name, dfn
        )
        out_file = in_file[:-4] + ".msh"

        # Consider the dimension of the problem, normally 2d but if dfn is true 1d
        ndim = 2 - int(dfn)

        pp.grids.gmsh.gmsh_interface.run_gmsh(in_file, out_file, dim=ndim)

        if dfn:
            # Create list of grids
            grid_list = porepy.fracs.simplex.line_grid_from_gmsh(
                out_file, constraints=constraints
            )

        else:
            # Create list of grids
            grid_list = porepy.fracs.simplex.triangle_grid_from_gmsh(
                out_file, constraints=constraints
            )

        if preserve_fracture_tags:
            # preserve tags for the fractures from the network
            # we are assuming a coherent numeration between the network
            # and the created grids
            frac = np.setdiff1d(
                np.arange(self.edges.shape[1]), constraints, assume_unique=True
            )
            for idg, g in enumerate(grid_list[1 - int(dfn)]):
                for key in np.atleast_1d(preserve_fracture_tags):
                    if key not in g.tags:
                        g.tags[key] = self.tags[key][frac][idg]

        # Assemble in grid bucket
        return pp.meshing.grid_list_to_grid_bucket(grid_list, **kwargs)

    def prepare_for_gmsh(
        self,
        mesh_args,
        tol=None,
        do_snap=True,
        constraints=None,
        file_name=None,
        dfn=False,
    ):
        """Process network intersections and write a gmsh .geo configuration file,
        ready to be processed by gmsh.

        NOTE: Consider to use the mesh() function instead to get a ready GridBucket.

        Parameters:
            mesh_args: Arguments passed on to mesh size control
            tol (double, optional): Tolerance used for geometric computations.
                Defaults to the tolerance of this network.
            do_snap (boolean, optional): Whether to snap lines to avoid small
                segments. Defults to True.
            constraints (np.array of int): Index of network edges that should not
                generate lower-dimensional meshes, but only act as constraints in
                the meshing algorithm.
            dfn (boolean, optional): If True, a DFN mesh (of the network, but not
                the surrounding matrix) is created.

        Returns:
            GridBucket: Mixed-dimensional mesh.

        """

        if tol is None:
            tol = self.tol
        if constraints is None:
            constraints = np.empty(0, dtype=np.int)
        else:
            constraints = np.atleast_1d(constraints)
        constraints = np.sort(constraints)

        if file_name is None:
            file_name = "gmsh_frac_file"
        in_file = file_name + ".geo"

        p = self.pts
        e = self.edges

        # Snap points to edges
        if do_snap and p is not None and p.size > 0:
            p, _ = pp.frac_utils.snap_fracture_set_2d(p, self.edges, snap_tol=tol)

        self.pts = p

        if not self.bounding_box_imposed:
            edges_deleted = self.impose_external_boundary(
                self.domain, add_domain_edges=not dfn
            )

            # Find edges of constraints to delete
            to_delete = np.where(np.isin(constraints, edges_deleted))[0]

            # Adjust constraint indices for deleted edges
            adjustment = np.zeros(constraints.size, dtype=np.int)
            for e in edges_deleted:
                # All constraints with index above the deleted edge should be reduced
                adjustment[constraints > e] += 1

            constraints -= adjustment
            # Delete constraints corresponding to deleted edges
            constraints = np.delete(constraints, to_delete)

        # Consider the dimension of the problem, normally 2d but if dfn is true 1d
        ndim = 2 - int(dfn)

        self._find_and_split_intersections(constraints)
        self._insert_auxiliary_points(**mesh_args)
        self._to_gmsh(in_file, ndim=ndim)
        return in_file

    def _find_and_split_intersections(self, constraints):
        # Unified description of points and lines for domain, and fractures

        points = self.pts
        edges = self.edges

        if not np.all(np.diff(edges[:2], axis=0) != 0):
            raise ValueError("Found a point edge in splitting of edges")

        const = constants.GmshConstants()

        tags = np.zeros((2, edges.shape[1]), dtype=np.int)
        tags[0][np.logical_not(self.tags["boundary"])] = const.FRACTURE_TAG
        tags[0][self.tags["boundary"]] = const.DOMAIN_BOUNDARY_TAG
        tags[0][constraints] = const.AUXILIARY_TAG
        tags[1] = np.arange(edges.shape[1])

        edges = np.vstack((edges, tags))

        # Ensure unique description of points
        pts_all, _, old_2_new = unique_columns_tol(points, tol=self.tol)
        edges[:2] = old_2_new[edges[:2]]
        to_remove = np.where(edges[0, :] == edges[1, :])[0]
        lines = np.delete(edges, to_remove, axis=1)

        self.decomposition["domain_boundary_points"] = old_2_new[
            self.decomposition["domain_boundary_points"]
        ]

        # In some cases the fractures and boundaries impose the same constraint
        # twice, although it is not clear why. Avoid this by uniquifying the lines.
        # This may disturb the line tags in lines[2], but we should not be
        # dependent on those.
        li = np.sort(lines[:2], axis=0)
        _, new_2_old, old_2_new = unique_columns_tol(li, tol=self.tol)
        lines = lines[:, new_2_old]

        if not np.all(np.diff(lines[:2], axis=0) != 0):
            raise ValueError(
                "Found a point edge in splitting of edges after merging points"
            )

        # We split all fracture intersections so that the new lines do not
        # intersect, except possible at the end points
        logger.info("Remove edge crossings")
        tm = time.time()

        pts_split, lines_split = pp.intersections.split_intersecting_segments_2d(
            pts_all, lines, tol=self.tol
        )
        logger.info("Done. Elapsed time " + str(time.time() - tm))

        # Ensure unique description of points
        pts_split, _, old_2_new = unique_columns_tol(pts_split, tol=self.tol)
        lines_split[:2] = old_2_new[lines_split[:2]]
        to_remove = np.where(lines[0, :] == lines[1, :])[0]
        lines = np.delete(lines, to_remove, axis=1)

        self.decomposition["domain_boundary_points"] = old_2_new[
            self.decomposition["domain_boundary_points"]
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

        self.decomposition.update(
            {
                "points": pts_split,
                "edges": lines_split,
                "intersections": intersections,
                "domain": self.domain,
            }
        )

    def _find_intersection_points(self, lines):
        const = constants.GmshConstants()

        frac_id = np.ravel(lines[:2, lines[2] == const.FRACTURE_TAG])
        _, frac_ia, frac_count = np.unique(frac_id, True, False, True)

        # In the case we have auxiliary points remove do not create a 0d point in
        # case one intersects a single fracture. In the case of multiple fractures intersection
        # with an auxiliary point do consider the 0d.
        aux_id = np.logical_or(
            lines[2] == const.AUXILIARY_TAG, lines[2] == const.DOMAIN_BOUNDARY_TAG
        )
        if np.any(aux_id):
            aux_id = np.ravel(lines[:2, aux_id])
            _, aux_ia, aux_count = np.unique(aux_id, True, False, True)

            # probably it can be done more efficiently but currently we rarely use the
            # auxiliary points in 2d
            for a in aux_id[aux_ia[aux_count > 1]]:
                # if a match is found decrease the frac_count only by one, this prevent
                # the multiple fracture case to be handle wrongly
                frac_count[frac_id[frac_ia] == a] -= 1

        return frac_id[frac_ia[frac_count > 1]]

    def _insert_auxiliary_points(
        self, mesh_size_frac=None, mesh_size_bound=None, mesh_size_min=None
    ):
        # Gridding size
        # Tag points at the domain corners
        logger.info("Determine mesh size")
        tm = time.time()

        p = self.decomposition["points"]
        lines = self.decomposition["edges"]
        boundary_pt_ind = self.decomposition["domain_boundary_points"]

        mesh_size, pts_split, lines = tools.determine_mesh_size(
            p,
            boundary_pt_ind,
            lines,
            mesh_size_frac=mesh_size_frac,
            mesh_size_bound=mesh_size_bound,
            mesh_size_min=mesh_size_min,
        )

        logger.info("Done. Elapsed time " + str(time.time() - tm))

        self.decomposition["points"] = pts_split
        self.decomposition["edges"] = lines
        self.decomposition["mesh_size"] = mesh_size

    def impose_external_boundary(self, domain=None, add_domain_edges=True):
        """
        Constrain the fracture network to lie within a domain.

        Fractures outside the imposed domain will be deleted.

        The domain will be added to self.pts and self.edges, if add_domain_edges is True.
        The domain boundary edges can be identified from self.tags['boundary'].

        Args:
            domain (dict or np.array, optional): Domain. See __init__ for description.
                if not provided, self.domain will be used.
            add_domain_edges(bool, optional): Include or not the boundary edges and pts in
                the list of edges. Default value True.

        Returns:
            edges_deleted (np.array): Index of edges that were outside the bounding box
                and therefore deleted.

        """

        if isinstance(domain, dict):
            # First create lines that define the domain
            x_min = domain["xmin"]
            x_max = domain["xmax"]
            y_min = domain["ymin"]
            y_max = domain["ymax"]
            dom_p = np.array(
                [[x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max]]
            )
            dom_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]).T
        else:
            dom_p = domain
            tmp = np.arange(dom_p.shape[1])
            dom_lines = np.vstack((tmp, (tmp + 1) % dom_p.shape[1]))

        # Constrain the edges to the domain
        p, e, edges_kept = pp.constrain_geometry.lines_by_polygon(
            dom_p, self.pts, self.edges
        )

        edges_deleted = np.setdiff1d(np.arange(self.edges.shape[1]), edges_kept)

        # Define boundary tags. Set False to all existing edges (after cutting those
        # outside the boundary).
        boundary_tags = self.tags.get("boundary", [False] * e.shape[1])

        if add_domain_edges:
            num_p = p.shape[1]
            # Add the domain boundary edges and points
            self.edges = np.hstack((e, dom_lines + num_p))
            self.pts = np.hstack((p, dom_p))
            # preserve the tags
            for key, value in self.tags.items():
                self.tags[key] = np.hstack(
                    (value[edges_kept], [None] * dom_lines.shape[1])
                )

            # Define the new boundary tags
            new_boundary_tags = boundary_tags + dom_lines.shape[1] * [True]
            self.tags["boundary"] = np.array(new_boundary_tags)

            self.decomposition["domain_boundary_points"] = num_p + np.arange(
                dom_p.shape[1], dtype=np.int
            )
        else:
            self.tags["boundary"] = boundary_tags
            self.decomposition["domain_boundary_points"] = np.empty(0, dtype=np.int)

        self.bounding_box_imposed = True
        return edges_deleted

    def _to_gmsh(self, in_file, ndim):

        # Create a writer of gmsh .geo-files
        p = self.decomposition["points"]
        edges = self.decomposition["edges"]
        intersections = self.decomposition["intersections"]
        mesh_size = self.decomposition["mesh_size"]
        domain = self.decomposition["domain"]

        # Find points that are both on a domain boundary, and on a fracture.
        # These will be decleared Physical
        const = constants.GmshConstants()
        point_on_fracture = edges[:2, edges[2] == const.FRACTURE_TAG].ravel()
        point_on_boundary = edges[:2, edges[2] == const.DOMAIN_BOUNDARY_TAG].ravel()
        fracture_boundary_points = np.intersect1d(point_on_fracture, point_on_boundary)

        self.decomposition["fracture_boundary_points"] = fracture_boundary_points

        gw = gmsh_interface.GmshWriter(
            p,
            edges,
            domain=domain,
            mesh_size=mesh_size,
            intersection_points=intersections,
            domain_boundary_points=self.decomposition["domain_boundary_points"],
            fracture_and_boundary_points=fracture_boundary_points,
            nd=ndim,
        )
        gw.write_geo(in_file)

    ## end of methods related to meshing

    def _decompose_domain(self, domain, num_x, ny=None):
        x0 = domain["xmin"]
        dx = (domain["xmax"] - domain["xmin"]) / num_x

        if "ymin" in domain.keys() and "ymax" in domain.keys():
            y0 = domain["ymin"]
            dy = (domain["ymax"] - domain["ymin"]) / ny
            return x0, y0, dx, dy
        else:
            return x0, dx

    def snap(self, tol):
        """Modify point definition so that short branches are removed, and
        almost intersecting fractures become intersecting.

        Parameters:
            tol (double): Threshold for geometric modifications. Points and
                segments closer than the threshold may be modified.

        Returns:
            FractureNetwork2d: A new network with modified point coordinates.
        """

        # We will not modify the original fractures
        p = self.pts.copy()
        e = self.edges.copy()

        # Prolong
        p = pp.constrain_geometry.snap_points_to_segments(p, e, tol)

        return FractureNetwork2d(p, e, self.domain, self.tol)

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

        p_domain = self._domain_to_points(domain)

        p, e, _ = pp.constrain_geometry.lines_by_polygon(p_domain, self.pts, self.edges)

        return FractureNetwork2d(p, e, domain, self.tol)

    def _domain_to_points(self, domain):
        """Helper function to convert a domain specification in the form of
        a dictionary into a point set.

        If the domain is already a point set, nothing happens

        """
        if domain is None:
            domain = self.domain

        if isinstance(domain, dict):
            p00 = np.array([domain["xmin"], domain["ymin"]]).reshape((-1, 1))
            p10 = np.array([domain["xmax"], domain["ymin"]]).reshape((-1, 1))
            p11 = np.array([domain["xmax"], domain["ymax"]]).reshape((-1, 1))
            p01 = np.array([domain["xmin"], domain["ymax"]]).reshape((-1, 1))
            return np.hstack((p00, p10, p11, p01))

        else:
            return domain

    # --------- Methods for analysis of the fracture set

    def as_graph(self, split_intersections=True):
        """Represent the fracture set as a graph, using the networkx data structure.

        By default the fractures will first be split into non-intersecting branches.

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
            pts = split_network.pts
            edges = split_network.edges
        else:
            edges = self.edges
            pts = self.pts

        import networkx as nx

        G = nx.Graph()
        for pi in range(pts.shape[1]):
            G.add_node(pi, coordinate=pts[:, pi])

        for ei in range(edges.shape[1]):
            G.add_edge(edges[0, ei], edges[1, ei])

        if split_intersections:
            return G, split_network
        else:
            return G

    def split_intersections(self, tol=None):
        """Create a new FractureSet, with all fracture intersections removed

        Parameters:
            tol (optional): Tolerance used in geometry computations when
                splitting fractures. Defaults to the tolerance of this network.

        Returns:
            FractureSet: New set, where all intersection points are added so that
                the set only contains non-intersecting branches.

        """
        if tol is None:
            tol = self.tol

        p, e, argsort = pp.intersections.split_intersecting_segments_2d(
            self.pts, self.edges, tol=self.tol, return_argsort=True
        )
        # map the tags
        tags = {}
        for key, value in self.tags.items():
            tags[key] = value[argsort]

        fn = FractureNetwork2d(p, e, self.domain, tol=self.tol)
        fn.tags = tags

        return fn

    # --------- Utility functions below here

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
            fi = np.arange(self.num_frac)

        p = self.pts[:, self.edges[0, fi]]
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
            fi = np.arange(self.num_frac)

        p = self.pts[:, self.edges[1, fi]]
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
            fi = np.arange(self.num_frac)
        fi = np.asarray(fi)

        # compute the length for each segment
        norm = lambda e0, e1: np.linalg.norm(self.pts[:, e0] - self.pts[:, e1])
        length = np.array([norm(e[0], e[1]) for e in self.edges.T])

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
            fi = np.arange(self.num_frac)
        fi = np.asarray(fi)

        # compute the angle for each segment
        alpha = lambda e0, e1: np.arctan2(
            self.pts[1, e0] - self.pts[1, e1], self.pts[0, e0] - self.pts[0, e1]
        )
        a = np.array([alpha(e[0], e[1]) for e in self.edges.T])

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
            edges (np.array, 2 x num_frac, optional): Indices, refering to pts, of the start
                and end points of the fractures for which the centres should be computed.
                Defaults to the fractures of this set.

        Returns:
            np.array, 2 x num_frac: Coordinates of the centers of this fracture.

        """
        if p is None:
            p = self.pts
        if edges is None:
            edges = self.edges
        # first compute the fracture centres and then generate them
        avg = lambda e0, e1: 0.5 * (np.atleast_2d(p)[:, e0] + np.atleast_2d(p)[:, e1])
        pts_c = np.array([avg(e[0], e[1]) for e in edges.T]).T
        return pts_c

    def domain_measure(self, domain=None):
        """Get the measure (length, area) of a given box domain, specified by its
        extensions stored in a dictionary.

        The dimension of the domain is inferred from the dictionary fields.

        Parameters:
            domain (dictionary, optional): Should contain keys 'xmin' and 'xmax'
                specifying the extension in the x-direction. If the domain is 2d,
                it should also have keys 'ymin' and 'ymax'. If no domain is specified
                the domain of this object will be used.

        Returns:
            double: Measure of the domain.

        """
        if domain is None:
            domain = self.domain
        if "ymin" and "ymax" in domain.keys():
            return (domain["xmax"] - domain["xmin"]) * (domain["ymax"] - domain["ymin"])
        else:
            return domain["xmax"] - domain["xmin"]

    def plot(self, **kwargs):
        """Plot the fracture set.

        The function passes this fracture set to PorePy plot_fractures

        Parameters:
            **kwargs: Keyword arguments to be passed on to matplotlib.

        """
        pp.plot_fractures(self.pts, self.edges, domain=self.domain, **kwargs)

    def to_csv(self, file_name):
        """
        Save the 2d network on a csv file with comma , as separator.
        Note: the file is overwritten if present.
        The format is
        FID, START_X, START_Y, END_X, END_Y

        Parameters:
            file_name: name of the file
            domain: (optional) the bounding box of the problem
        """

        with open(file_name, "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            # write all the fractures
            for edge_id, edge in enumerate(self.edges.T):
                data = [edge_id]
                data.extend(self.pts[:, edge[0]])
                data.extend(self.pts[:, edge[1]])
                csv_writer.writerow(data)

    def to_vtk(self, file_name, data=None, binary=True):
        """
        Export the fracture network to vtk.

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
            binary (boolean, optional): Use binary export format. Defaults to
                True.

        """
        network_vtk = vtk.vtkUnstructuredGrid()

        point_counter = 0
        pts_vtk = vtk.vtkPoints()

        pts = self.pts
        # make points 3d
        if pts.shape[0] == 2:
            pts = np.vstack((pts, np.zeros(pts.shape[1])))

        for edge in self.edges.T:

            # Add local points
            pts_vtk.InsertNextPoint(*pts[:, edge[0]])
            pts_vtk.InsertNextPoint(*pts[:, edge[1]])

            # Indices of local points
            loc_pt_id = point_counter + np.arange(2)
            # Update offset
            point_counter += 2

            # Add bounding polygon
            frac_vtk = vtk.vtkIdList()
            [frac_vtk.InsertNextId(p) for p in loc_pt_id]
            # Close polygon
            frac_vtk.InsertNextId(loc_pt_id[0])

            network_vtk.InsertNextCell(vtk.VTK_POLYGON, frac_vtk)

        # Add the points
        network_vtk.SetPoints(pts_vtk)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(network_vtk)
        writer.SetFileName(file_name)

        if not binary:
            writer.SetDataModeToAscii()

        # Cell-data to be exported is at least the fracture numbers
        if data is None:
            data = {}
        # Use offset 1 for fracture numbers (should we rather do 0?)
        data["Fracture_Number"] = 1 + np.arange(self.edges.shape[1])

        for name, data in data.items():
            data_vtk = vtk_np.numpy_to_vtk(
                data.ravel(order="F"), deep=True, array_type=vtk.VTK_DOUBLE
            )
            data_vtk.SetName(name)
            data_vtk.SetNumberOfComponents(1 if data.ndim == 1 else 3)
            network_vtk.GetCellData().AddArray(data_vtk)

        writer.Update()

    def __str__(self):
        s = "Fracture set consisting of " + str(self.num_frac) + " fractures"
        if self.pts is not None:
            s += ", consisting of " + str(self.pts.shape[1]) + " points.\n"
        else:
            s += ".\n"
        if self.domain is not None:
            s += "Domain: "
            s += str(self.domain)
        return s

    def __repr__(self):
        return self.__str__()
