#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:24:49 2018

@author: Eirik Keilegavlens
"""
import numpy as np
import logging
import networkx as nx
import csv
import time

import porepy as pp
import porepy.fracs.simplex
from porepy.grids import constants
from porepy.grids.gmsh import gmsh_interface, mesh_2_grid
from porepy.fracs import fractures, tools
from porepy.utils.setmembership import unique_columns_tol, ismember_rows


logger = logging.getLogger(__name__)


class FractureNetwork2d(object):
    """ Class representation of a set of fractures in a 2D domain.

    The fractures are represented by their endpoints. Poly-line fractures are
    currently not supported. There is no requirement or guarantee that the
    fractures are contained within the specified domain. The fractures can be
    cut to a given domain by the function constrain_to_domain().

    Attributes:
        pts (np.array, 2 x num_pts): Start and endpoints of the fractures. Points
            can be shared by fractures.
        edges (np.array, (2 + num_tags) x num_fracs): The first two rows represent
            indices, refering to pts, of the start and end points of the fractures.
            Additional rows are optional tags of the fractures.
        domain (dictionary): The domain in which the fracture set is defined.
            Should contain keys 'xmin', 'xmax', 'ymin', 'ymax', each of which
            maps to a double giving the range of the domain. The fractures need
            not lay inside the domain.
        num_frac (int): Number of fractures in the domain.

    """

    def __init__(self, pts=None, edges=None, domain=None, tol=1e-8):
        """ Define the frature set.

        Parameters:
            pts (np.array, 2 x n): Start and endpoints of the fractures. Points
            can be shared by fractures.
        edges (np.array, (2 + num_tags) x num_fracs): The first two rows represent
            indices, refering to pts, of the start and end points of the fractures.
            Additional rows are optional tags of the fractures.
        domain (dictionary): The domain in which the fracture set is defined.
            Should contain keys 'xmin', 'xmax', 'ymin', 'ymax', each of which
            maps to a double giving the range of the domain.

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

    def add_network(self, fs):
        """ Add this fracture set to another one, and return a new set.

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

        return FractureNetwork2d(p, e, domain, self.tol)

    def mesh(
        self,
        mesh_args,
        tol=None,
        do_snap=True,
        constraints=None,
        file_name=None,
        **kwargs,
    ):
        """ Create GridBucket (mixed-dimensional grid) for this fracture network.

        Parameters:
            mesh_args: Arguments passed on to mesh size control
            tol (double, optional): Tolerance used for geometric computations.
                Defaults to the tolerance of this network.
            do_snap (boolean, optional): Whether to snap lines to avoid small
                segments. Defults to True.
            constraints (np.array of int): Index of network edges that should not
                generate lower-dimensional meshes, but only act as constraints in
                the meshing algorithm.

        Returns:
            GridBucket: Mixed-dimensional mesh.

        """

        if tol is None:
            tol = self.tol
        if constraints is None:
            constraints = np.empty(0, dtype=np.int)

        if file_name is None:
            file_name = "gmsh_frac_file"
        in_file = file_name + ".geo"
        out_file = file_name + ".msh"

        p = self.pts
        e = self.edges

        # Snap points to edges
        if do_snap and p is not None and p.size > 0:
            p, _ = pp.frac_utils.snap_fracture_set_2d(p, self.edges, snap_tol=tol)

        self.pts = p

        if not self.bounding_box_imposed:
            self.impose_external_boundary(self.domain)

        self._find_and_split_intersections(constraints)
        self._insert_auxiliary_points(**mesh_args)
        self._to_gmsh(in_file)

        gmsh_status = pp.grids.gmsh.gmsh_interface.run_gmsh(in_file, out_file, dims=2)
        if gmsh_status > 0:
            raise ValueError(f"Gmsh failed with status {gmsh_status}")

        logger.info("Gmsh completed with status " + str(gmsh_status))
        # Create list of grids
        grid_list = porepy.fracs.simplex.triangle_grid_from_gmsh(
            out_file, constraints=constraints
        )
        # Assemble in grid bucket
        gb = pp.meshing.grid_list_to_grid_bucket(grid_list, **kwargs)
        return gb

    def _find_and_split_intersections(self, constraints):
        # Unified description of points and lines for domain, and fractures

        points = self.pts
        edges = self.edges

        assert np.all(np.diff(edges[:2], axis=0) != 0)

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

        # In some cases the fractures and boundaries impose the same constraint
        # twice, although it is not clear why. Avoid this by uniquifying the lines.
        # This may disturb the line tags in lines[2], but we should not be
        # dependent on those.
        li = np.sort(lines[:2], axis=0)
        _, new_2_old, old_2_new = unique_columns_tol(li, tol=self.tol)
        lines = lines[:, new_2_old]

        assert np.all(np.diff(lines[:2], axis=0) != 0)

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

    def impose_external_boundary(self, domain=None):

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

        boundary_tags = self.tags.get("boundary", [False] * self.edges.shape[1])
        new_boundary_tags = boundary_tags + dom_lines.shape[1] * [True]

        num_p = self.pts.shape[1]
        self.pts = np.hstack((self.pts, dom_p))
        self.edges = np.hstack((self.edges, dom_lines + num_p))

        self.tags["boundary"] = new_boundary_tags

        self.bounding_box_imposed = True

        self.decomposition["domain_boundary_points"] = num_p + np.arange(
            dom_p.shape[1], dtype=np.int
        )

    def _to_gmsh(self, in_file):

        # Create a writer of gmsh .geo-files
        p = self.decomposition["points"]
        edges = self.decomposition["edges"]
        intersections = self.decomposition["intersections"]
        mesh_size = self.decomposition["mesh_size"]
        domain = self.decomposition["domain"]

        gw = gmsh_interface.GmshWriter(
            p,
            edges,
            domain=domain,
            mesh_size=mesh_size,
            intersection_points=intersections,
            domain_boundary_points=self.decomposition["domain_boundary_points"],
        )
        gw.write_geo(in_file)

    ## end of methods related to meshing

    def _decompose_domain(self, domain, nx, ny=None):
        x0 = domain["xmin"]
        dx = (domain["xmax"] - domain["xmin"]) / nx

        if "ymin" in domain.keys() and "ymax" in domain.keys():
            y0 = domain["ymin"]
            dy = (domain["ymax"] - domain["ymin"]) / ny
            return x0, y0, dx, dy
        else:
            return x0, dx

    def snap(self, tol):
        """ Modify point definition so that short branches are removed, and
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
        """ Constrain the fracture network to lay within a specified domain.

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

        p, e = pp.constrain_geometry.lines_by_polygon(p_domain, self.pts, self.edges)

        return FractureNetwork2d(p, e, domain, self.tol)

    def _domain_to_points(self, domain):
        """ Helper function to convert a domain specification in the form of
        a dictionary into a point set.
        """
        if domain is None:
            domain = self.domain

        p00 = np.array([domain["xmin"], domain["ymin"]]).reshape((-1, 1))
        p10 = np.array([domain["xmax"], domain["ymin"]]).reshape((-1, 1))
        p11 = np.array([domain["xmax"], domain["ymax"]]).reshape((-1, 1))
        p01 = np.array([domain["xmin"], domain["ymax"]]).reshape((-1, 1))
        return np.hstack((p00, p10, p11, p01))

    # --------- Methods for analysis of the fracture set

    def as_graph(self, split_intersections=True):
        """ Represent the fracture set as a graph, using the networkx data structure.

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
        """ Create a new FractureSet, with all fracture intersections removed

        Parameters:
            tol (optional): Tolerance used in geometry computations when
                splitting fractures. Defaults to the tolerance of this network.

        Returns:
            FractureSet: New set, where all intersection points are added so that
                the set only contains non-intersecting branches.

        """
        if tol is None:
            tol = self.tol

        p, e = pp.intersections.split_intersecting_segments_2d(
            self.pts, self.edges, tol=self.tol
        )
        return FractureNetwork2d(p, e, self.domain, tol=self.tol)

    # --------- Utility functions below here

    def start_points(self, fi=None):
        """ Get start points of all fractures, or a subset.

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
        """ Get start points of all fractures, or a subset.

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
        """ Return start and end points for a specified fracture.

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
        l = np.array([norm(e[0], e[1]) for e in self.edges.T])

        # compute the total length based on the fracture id
        tot_l = lambda f: np.sum(l[np.isin(fi, f)])
        return np.array([tot_l(f) for f in np.unique(fi)])

    def orientation(self, fi=None):
        """ Compute the angle of the fractures to the x-axis.

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
        """ Compute center points of a set of fractures.

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
        """ Get the measure (length, area) of a given box domain, specified by its
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
        """ Plot the fracture set.

        The function passes this fracture set to PorePy plot_fractures

        Parameters:
            **kwargs: Keyword arguments to be passed on to matplotlib.

        """
        pp.plot_fractures(self.domain, self.pts, self.edges, **kwargs)

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
