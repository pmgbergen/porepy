#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:54:36 2018

@author: Eirik Keilegavlens
"""
import numpy as np
import scipy
import scipy.stats as stats
import logging

from examples.papers.flow_upscaling import frac_gen
import porepy as pp


logger = logging.getLogger(__name__)


class FractureSet(object):
    """ Class representation of a set of fractures in a 2D domain.

    The fractures are represented by their endpoints. Poly-line fractures are
    currently not supported. There is no requirement or guarantee that the
    fractures are contained within the specified domain. The fractures can be
    cut to a given domain by the function constrain_to_domain().

    The main intended usage is to fit statistical distributions to the fractures,
    and use this to generate realizations based on this statistics. The statistical
    properties of the fracture set is characterized in terms of fracture position,
    length and angle.

    It is assumed that the fractures can meaningfully be represented by a single
    statistical distribution. To achieve this, it may be necessary to divide a
    fracture network into several sets, and fit them separately. As an example,
    a network where the fractures have one out of two orientations which are orthogonal
    to each other will not be meaningfully be represented as a single set.

    Attributes:
        pts (np.array, 2 x num_pts): Start and endpoints of the fractures. Points
            can be shared by fractures.
        edges (np.array, 2 x num_fracs): Indices, refering to pts, of the start
            and end points of the fractures.
        domain (dictionary): The domain in which the fracture set is defined.
            Should contain keys 'xmin', 'xmax', 'ymin', 'ymax', each of which
            maps to a double giving the range of the domain. The fractures need
            not lay inside the domain.
        num_frac (int): Number of fractures in the domain.

    """

    def __init__(self, pts=None, edges=None, domain=None):
        """ Define the frature set.

        Parameters:
            pts (np.array, 2 x n): Start and endpoints of the fractures. Points
            can be shared by fractures.
        edges (np.array, 2 x num_fracs): Indices, refering to pts, of the start
            and end points of the fractures.
        domain (dictionary): The domain in which the fracture set is defined.
            Should contain keys 'xmin', 'xmax', 'ymin', 'ymax', each of which
            maps to a double giving the range of the domain.

        """

        self.pts = pts
        self.edges = edges
        self.domain = domain

        if edges is not None:
            self.num_frac = self.edges.shape[1]
        else:
            self.num_frac = 0

        if pts is None and edges is None:
            logger.info('Generated empty fracture set')
        else:
            logger.info("Generated a fracture set with %i fractures", self.num_frac)
            if pts.size > 0:
                logger.info(
                    "Minimum point coordinates x: %.2f, y: %.2f", pts[0].min(), pts[1].min()
                )
                logger.info(
                    "Maximum point coordinates x: %.2f, y: %.2f", pts[0].max(), pts[1].max()
                )
        if domain is not None:
            logger.info("Domain specification :" + str(domain))

    def add(self, fs):
        """ Add this fracture set to another one, and return a new set.

        The new set may contain non-unique points and edges.

        Parameters:
            fs (FractureSet): Another set to be added

        Returns:
            New fracture set, containing all points and edges in both self and
                fs.
        """
        logger.info("Add fracture sets: ")
        logger.info(str(self))
        logger.info(str(fs))

        p = np.hstack((self.pts, fs.pts))
        e = np.hstack((self.edges, fs.edges + self.pts.shape[1]))

        domain = {
            "xmin": np.minimum(self.domain["xmin"], fs.domain["xmin"]),
            "xmax": np.maximum(self.domain["xmax"], fs.domain["xmax"]),
            "ymin": np.minimum(self.domain["ymin"], fs.domain["ymin"]),
            "ymax": np.maximum(self.domain["ymax"], fs.domain["ymax"]),
        }

        return FractureSet(p, e, domain)

    def mesh(self, tol, mesh_args, do_snap=True):

        p = self.pts
        e = self.edges

        if do_snap:
            p, _ = pp.frac_utils.snap_fracture_set_2d(p, e, snap_tol=tol)
        frac_dict = {'points': p, 'edges': e}
        gb = pp.meshing.simplex_grid(frac_dict, self.domain, tol=tol, **mesh_args)
        return gb


    def fit_distributions(self, **kwargs):
        """ Fit statistical distributions to describe the fracture set.

        The method will compute best fit distributions for fracture length,
        angle and position. These can later be used to generate realizations
        of other fracture network, using the current one as a base case.

        The length distribution can be either lognormal or exponential.

        The orientation is represented by a best fit of a von-Mises distribution.

        The fracture positions are represented by an intensity map, which
        divides the domain into subblocks and count the number of fracture
        centers per block.

        For more details, see the individual functions for fitting each of the
        distributions

        """
        logger.inf("Fit length, angle and intensity distribution")
        self.fit_length_distribution(**kwargs)
        self.fit_angle_distribution(**kwargs)
        self.fit_intensity_map(**kwargs)

    def fit_length_distribution(self, ks_size=100, p_val_min=0.05, **kwargs):
        """ Fit a statistical distribution to describe the length of the fractures.

        The best fit is sought between an exponential and lognormal representation.

        The resulting distribution is represented in an attribute dist_length.

        The function also evaluates the fitness of the chosen distribution by a
        Kolgomorov-Smirnov test.

        Parameters:
            ks_size (int, optional): The number of realizations used in the
                Kolmogorov-Smirnov test. Defaults to 100.
            p_val_min (double, optional): P-value used in Kolmogorev-Smirnov test
                for acceptance of the chosen distribution. Defaults to 0.05.

        """
        # fit the lenght distribution
        candidate_dist = np.array([stats.expon, stats.lognorm])

        # fit the possible lenght distributions
        l = self.length()
        dist_fit = np.array([d.fit(l, floc=0) for d in candidate_dist])

        # determine which is the best distribution with a Kolmogorov-Smirnov test
        ks = lambda d, p: stats.ks_2samp(l, d.rvs(*p, size=ks_size))[1]
        p_val = np.array([ks(d, p) for d, p in zip(candidate_dist, dist_fit)])
        best_fit = np.argmax(p_val)

        if p_val[best_fit] < p_val_min:
            raise ValueError("p-value not satisfactory for length fit")

        # collect the data
        dist_l = {
            "dist": candidate_dist[best_fit],
            "param": dist_fit[best_fit],
            "p_val": p_val[best_fit],
        }
        # Logging
        stat_string = ["exponential", "log-normal"]
        logger.info(
            "Fracture length represented by a %s distribution ", stat_string[best_fit]
        )
        s = "Fracture parameters: "
        for p in dist_fit[best_fit]:
            s += str(p) + ", "
        logger.info(s)
        logger.info("P-value for fitting: %.3f", p_val[best_fit])

        self.dist_length = dist_l

    def fit_angle_distribution(self, ks_size=100, p_val_min=0.05, **kwargs):

        """ Fit a statistical distribution to describe the length of the fractures.

        The best fit is sought between an exponential and lognormal representation.

        The resulting distribution is represented in an attribute dist_angle.

        The function also evaluates the fitness of the chosen distribution by a
        Kolgomorov-Smirnov test.

        Parameters:
            ks_size (int, optional): The number of realizations used in the
                Kolmogorov-Smirnov test. Defaults to 100.
            p_val_min (double, optional): P-value used in Kolmogorev-Smirnov test
                for acceptance of the chosen distribution. Defaults to 0.05.

        """
        dist = stats.vonmises
        a = self.angle()
        dist_fit = dist.fit(a, fscale=1)

        # check the goodness of the fit with Kolmogorov-Smirnov test
        p_val = stats.ks_2samp(a, dist.rvs(*dist_fit, size=ks_size))[1]

        if p_val < p_val_min:
            raise ValueError("p-value not satisfactory for angle fit")

        # logging
        logger.info("Fracture orientation represented by a von mises distribution ")
        s = "Fracture parameters: "
        for p in dist_fit:
            s += str(p) + ", "
        logger.info(s)
        logger.info("P-value for fitting: %.3f", p_val)

        # collect the data
        self.dist_angle = {"dist": dist, "param": dist_fit, "p_val": p_val}

    def fit_intensity_map(self, p=None, e=None, domain=None, nx=10, ny=10, **kwargs):
        """ Divide the domain into boxes, count the number of fracture centers
        contained within each box, and divide by the measure of the domain.

        The resulting intensity map is stored in an attribute intensity.

        Parameters:
            p (np.array, 2 x n, optional): Point coordinates of the fractures. Defaults to
                this set.
            e (np.array, 2 x n, optional): Connections between the coordinates. Defaults to
                this set.
            domain (dictionary, optional): Description of the simulation domain. Should
                contain fields xmin, xmax, ymin, ymax. Defaults to this set.
            nx, ny (int, optional): Number of boxes in x and y direction. Defaults
                to 10.

        Returns:
            np.array (nx x ny): Number of centers within each box, divided by the measure
                of the specified domain.

        """
        if p is None:
            p = self.pts
        if e is None:
            e = self.edges
        if domain is None:
            domain = self.domain

        p = np.atleast_2d(p)

        # Special treatment when the point array is empty
        if p.shape[1] == 0:
            if p.shape[0] == 1:
                return np.zeros(nx)
            else:  # p.shape[0] == 2
                return np.zeros((nx, ny))

        pc = self._compute_center(p, e)

        if p.shape[0] == 1:
            x0, dx = self._decompose_domain(domain, nx, ny)
            num_occ = np.zeros(nx)
            for i in range(nx):
                hit = np.logical_and.reduce(
                    [pc[0] > (x0 + i * dx), pc[0] <= (x0 + (i + 1) * dx)]
                )
                num_occ[i] = hit.sum()

            return num_occ.astype(np.int) / self.domain_measure(domain)

        elif p.shape[0] == 2:
            x0, y0, dx, dy = self._decompose_domain(domain, nx, ny)
            num_occ = np.zeros((nx, ny))
            # Can probably do this more vectorized, but for now, a for loop will suffice
            for i in range(nx):
                for j in range(ny):
                    hit = np.logical_and.reduce(
                        [
                            pc[0] > (x0 + i * dx),
                            pc[0] < (x0 + (i + 1) * dx),
                            pc[1] > (y0 + j * dy),
                            pc[1] < (y0 + (j + 1) * dy),
                        ]
                    )
                    num_occ[i, j] = hit.sum()

            return num_occ / self.domain_measure(domain)

        else:
            raise ValueError("Have not yet implemented 3D geometries")

        self.intensity = num_occ

    def set_length_distribution(self, dist, params):
        self.dist_length = {"dist": dist, "param": params}

    def set_angle_distribution(self, dist, params):
        self.dist_angle = {"dist": dist, "param": params}

    def set_intensity_map(self, box):
        self.intensity = box

    def _fracture_from_center_angle_length(self, p, angles, lengths):
        """ Generate fractures from a marked-point representation.

        Parameters:
            p (np.array, 2 x num_frac): Center points of the fractures.
            angles (np.array, num_frac): Angle from the x-axis of the fractures.
                Measured in radians.
            lengths (np.array, num_frac): Length of the fractures

        Returns:
            np.array (2 x 2 * num_frac): Start and endpoints of the fractures
            np.array (2 x num_frac): For each fracture, the start and endpoint,
                in terms of indices in the point array.

        """
        num_frac = lengths.size

        start = p + 0.5 * lengths * np.vstack((np.cos(angles), np.sin(angles)))
        end = p - 0.5 * lengths * np.vstack((np.cos(angles), np.sin(angles)))

        pts = np.hstack((start, end))

        e = np.vstack((np.arange(num_frac), num_frac + np.arange(num_frac)))
        return pts, e

    def _define_centers_by_boxes(self, domain, distribution="poisson"):
        """ Define center points of fractures, intended used in a marked point
        process.

        The domain is assumed decomposed into a set of boxes, and fracture points
        will be allocated within each box, according to the specified distribution
        and intensity.

        A tacit assumption is that the domain and intensity map corresponds to
        values used in and computed by count_center_point_densities. If this is
        not the case, scaling errors of the densities will arise. This should not
        be difficult to generalize, but there is no time right now.

        The implementation closely follows y Xu and Dowd:
            A new computer code for discrete fracture network modelling
            Computers and Geosciences, 2010

        Parameters:
            domain (dictionary): Description of the simulation domain. Should
                contain fields xmin, xmax, ymin, ymax.
            intensity (np.array, nx x ny): Intensity map, mean values for fracture
                density in each of the boxes the domain will be split into.
            distribution (str, default): Specify which distribution is followed.
                For now a placeholder value, only 'poisson' is allowed.

        Returns:
             np.array (2 x n): Coordinates of the fracture centers.

        Raises:
            ValueError if distribution does not equal poisson.

        """
        if distribution != "poisson":
            return ValueError("Only Poisson point processes have been implemented")

        # Intensity scaled to this domain
        intensity = self.intensity * self.domain_measure(domain)

        nx, ny = intensity.shape
        num_boxes = intensity.size

        max_intensity = intensity.max()

        x0, y0, dx, dy = self._decompose_domain(domain, nx, ny)

        # It is assumed that the intensities are computed relative to boxes of the
        # same size that are assigned in here
        area_of_box = 1

        pts = np.empty(num_boxes, dtype=np.object)

        # First generate the full set of points with maximum intensity
        counter = 0
        for i in range(nx):
            for j in range(ny):
                num_p_loc = stats.poisson(max_intensity * area_of_box).rvs(1)[0]
                p_loc = np.random.rand(2, num_p_loc)
                p_loc[0] = x0 + i * dx + p_loc[0] * dx
                p_loc[1] = y0 + j * dy + p_loc[1] * dy
                pts[counter] = p_loc
                counter += 1

        # Next, carry out a thinning process, which is really only necessary if the intensity is non-uniform
        # See Xu and Dowd Computers and Geosciences 2010, section 3.2 for a description
        counter = 0
        for i in range(nx):
            for j in range(ny):
                p_loc = pts[counter]
                threshold = np.random.rand(p_loc.shape[1])
                delete = np.where(intensity[i, j] / max_intensity < threshold)[0]
                pts[counter] = np.delete(p_loc, delete, axis=1)
                counter += 1

        return np.array(
            [pts[i][:, j] for i in range(pts.size) for j in range(pts[i].shape[1])]
        ).T

    def _decompose_domain(self, domain, nx, ny=None):
        x0 = domain["xmin"]
        dx = (domain["xmax"] - domain["xmin"]) / nx

        if "ymin" in domain.keys() and "ymax" in domain.keys():
            y0 = domain["ymin"]
            dy = (domain["ymax"] - domain["ymin"]) / ny
            return x0, y0, dx, dy
        else:
            return x0, dx

    def generate(self, domain=None, fit_distributions=True, **kwargs):
        """ Generate a realization of a fracture network from the statistical distributions
        represented in this object.

        The function relies on the statistical properties of the fracture set
        being known, in the form of attributes:

            dist_angle: Statistical distribution of orientations. Should be a dictionary
                with fields 'dist' and 'param'. Here, 'dist' should point to a
                scipy.stats.distribution, or another object with a function
                rvs to draw random variables, while 'param' points to the parameters
                passed on to dist.rvs.

            dist_length: Statistical distribution of length. Should be a dictionary
                with fields 'dist' and 'param'. Here, 'dist' should point to a
                scipy.stats.distribution, or another object with a function
                rvs to draw random variables, while 'param' points to the parameters
                passed on to dist.rvs.

            intensity (np.array): Frequency map of fracture centers in the domain.

        By default, these will be computed by this method. The attributes can
        also be set externally.

        Parameters:
            domain (dictionary, not in use): Future use will include a scaling of
                intensity to fit with another domain. For now, this field is not
                used.
            fit_distributions (boolean, optional): If True, compute the statistical
                properties of the network. Defaults to True.

        Returns:
            FractureSet: A new fracture set generated according to the statistical
                properties of this object.

        """
        if domain is None:
            domain = self.domain

        if fit_distributions:
            self.fit_distributions()

        # First define points
        p_center = self._define_centers_by_boxes(domain)
        # bookkeeping
        if p_center.size == 0:
            num_fracs = 0
        else:
            num_fracs = p_center.shape[1]

        # Then assign length and orientation
        angles = frac_gen.generate_from_distribution(num_fracs, self.dist_angle)
        lengths = frac_gen.generate_from_distribution(num_fracs, self.dist_length)

        p, e = self._fracture_from_center_angle_length(p_center, angles, lengths)

        return FractureSet(p, e, domain)

    # --------- Methods for manipulation of the fracture set geometry

    def snap(self, threshold):
        """ Modify point definition so that short branches are removed, and
        almost intersecting fractures become intersecting.

        Parameters:
            threshold (double): Threshold for geometric modifications. Points and
                segments closer than the threshold may be modified.

        Returns:
            FractureSet: A new FractureSet with modified point coordinates.
        """

        # We will not modify the original fractures
        p = self.pts.copy()
        e = self.edges.copy()

        # Prolong
        p = pp.cg.snap_points_to_segments(p, e, threshold)

        return FractureSet(p, e, self.domain)

    def branches(self):
        """ Split the fractures into branches.

        Returns:
            np.array (2 x npt): Start and endpoint of the fracture branches,
                that is, start and end points of fractures, as well as intersection
                points.
            np.array (3 x npt): Connections between points that form branches.
                The first two rows represent indices of the start and end points
                of the branches. The third gives the index of the fracture to which
                the branch belongs, referring to the ordering in self.num_frac

        """
        p = self.pts.copy()
        e = np.vstack((self.edges.copy(), np.arange(self.num_frac)))
        return pp.cg.remove_edge_crossings(p, e)


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
            FractureSet: Initialized by the constrained fractures, and the
                specified domain.

        """
        if domain is None:
            domain = self.domain

        p_domain = self._domain_to_points(domain)

        p, e = pp.cg.intersect_polygon_lines(p_domain, self.pts, self.edges)

        return FractureSet(p, e, domain)

    def _domain_to_points(self, domain):
        """ Helper function to convert a domain specification in the form of
        a dictionary into a point set.
        """
        if domain is None:
            domain = self.domain

        p00 = np.array([domain['xmin'], domain['ymin']]).reshape((-1, 1))
        p10 = np.array([domain['xmax'], domain['ymin']]).reshape((-1, 1))
        p11 = np.array([domain['xmax'], domain['ymax']]).reshape((-1, 1))
        p01 = np.array([domain['xmin'], domain['ymax']]).reshape((-1, 1))
        return np.hstack((p00, p10, p11, p01))

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

    def angle(self, fi=None):
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
        if 'ymin' and 'ymax' in domain.keys():
            return (domain['xmax'] - domain['xmin']) * (domain['ymax'] - domain['ymin'])
        else:
            return domain['xmax'] - domain['xmin']

    def plot(self, **kwargs):
        """ Plot the fracture set.

        The function passes this fracture set to PorePy plot_fractures

        Parameters:
            **kwargs: Keyword arguments to be passed on to matplotlib.

        """
        pp.plot_fractures(self.domain, self.pts, self.edges)

    def __str__(self):
        s = "Fracture set consisting of " + str(self.num_frac) + " fractures,"
        s += " consisting of " + str(self.pts.shape[1]) + " points.\n"
        s += "Domain: "
        s += str(self.domain)
        return s

    def __repr__(self):
        return self.__str__()


class ChildFractureSet(FractureSet):
    """ Fracture set that is defined based on its distance from a member of
    a parent family
    """

    def __init__(self, pts, edges, domain, parent):
        super(ChildFractureSet, self).__init__(pts, edges, domain)

        self.parent = parent

    def generate(self, parent_realiz, domain=None, y_separately=False):
        """ Generate a realization of a fracture network from the statistical distributions
        represented in this object.

        The function relies on the statistical properties of the fracture set
        being known, in the form of attributes:

            dist_angle: Statistical distribution of orientations. Should be a dictionary
                with fields 'dist' and 'param'. Here, 'dist' should point to a
                scipy.stats.distribution, or another object with a function
                rvs to draw random variables, while 'param' points to the parameters
                passed on to dist.rvs.

            dist_length: Statistical distribution of length. Should be a dictionary
                with fields 'dist' and 'param'. Here, 'dist' should point to a
                scipy.stats.distribution, or another object with a function
                rvs to draw random variables, while 'param' points to the parameters
                passed on to dist.rvs.

            dist_num_childern: Statistical distribution of orientations. Should be a
                scipy.stats.distribution, or another object with a function
                rvs to draw random variables.

            fraction_isolated, fraction_one_y: Fractions of the children that should
                on average be isolated and one-y. Should be doubles between
                0 and 1, and not sum to more than unity. The number of both-y
                fractures are 1 - (fraction_isolated + fraction_one_y)

            dist_from_parents: Statistical distribution that gives the distance from
                parent to isolated children, in the direction orthogonal to the parent.
                Should be a dictionary with fields 'dist' and 'param'. Here, 'dist' should
                point to a scipy.stats.distribution, or another object with a function
                rvs to draw random variables, while 'param' points to the parameters
                passed on to dist.rvs.

        These attributes should be set before the method is called.

        Parameters:
            parent_realiz (FractureSet): The parent of the new realization. This will
                possibly be the generated realization of the parent of this object.
            domain (dictionary, not in use): Future use will include a scaling of
                intensity to fit with another domain. For now, this field is not
                used, and the domain is taken as the same as for the original child set.

        Returns:
            FractureSet: A new fracture set generated according to the statistical
                properties of this object.

        """
        if domain is None:
            domain = self.domain

        num_parents = parent_realiz.edges.shape[1]

        # Arrays to store all points and fractures in the new realization
        all_p = np.empty((2, 0))
        all_edges = np.empty((2, 0))

        num_isolated = 0
        num_one_y = 0
        num_both_y = 0

        logger.info("Generate children for fracture set: \n" + str(parent_realiz))

        # Loop over all fractures in the parent realization. Decide on the
        # number of realizations.
        for pi in range(num_parents):
            # Decide on the number of children
            logging.debug("Parent fracture %i", pi)

            num_children = self._draw_num_children(parent_realiz, pi)
            logging.debug("Fracture has %i children", num_children)

            # If this fracture has no children, continue
            if num_children == 0:
                continue

            # Find the location of children points along the parent.
            # The interpretation of this point will differ, depending on whether
            # the child is chosen as isolated, one_y or both_y
            children_points = self._draw_children_along_parent(
                parent_realiz, pi, num_children
            )
            # For all children, decide type of child
            if y_separately:
                is_isolated, is_one_y, is_both_y = self._draw_children_type(
                    num_children, parent_realiz, pi, y_separately=y_separately
                )
                num_isolated += is_isolated.sum()
                num_one_y += is_one_y.sum()
                num_both_y += is_both_y.sum()

                logging.debug(
                    "Isolated children: %i, one y: %i, both y: %i",
                    is_isolated.sum(),
                    is_one_y.sum(),
                    is_both_y.sum(),
                )

            else:
                is_isolated, is_y = self._draw_children_type(
                    num_children, parent_realiz, pi, y_separately=y_separately
                )
                num_isolated += is_isolated.sum()
                num_one_y += is_y.sum()

                logging.debug(
                    "Isolated children: %i, one y: %i",
                    is_isolated.sum(),
                    is_y.sum(),
                )

            # Start and end point of parent
            start_parent, end_parent = parent_realiz.get_points(pi)

            # Generate isolated children
            p_i, edges_i = self._generate_isolated_fractures(
                children_points[:, is_isolated], start_parent, end_parent
            )
            # Store data
            num_pts = all_p.shape[1]
            all_p = np.hstack((all_p, p_i))
            edges_i += num_pts

            if y_separately:
                # Generate Y-fractures
                p_y, edges_y = self._generate_y_fractures(children_points[:, is_one_y])
                p_b_y, edges_b_y = self._generate_constrained_fractures(children_points[:, is_both_y], parent_realiz)

                # Assemble points
                all_p = np.hstack((all_p, p_y, p_b_y))

                # Adjust indices in point-fracture relation to account for previously
                # added objects
                edges_y += num_pts + p_i.shape[1]
                edges_b_y += num_pts + p_i.shape[1] + p_y.shape[1]

                all_edges = np.hstack((all_edges, edges_i, edges_y, edges_b_y)).astype(np.int)
            else:
                p_y, edges_y = self._generate_constrained_fractures(children_points[:, is_y], parent_realiz)
                # Assemble points
                all_p = np.hstack((all_p, p_y))

                # Adjust indices in point-fracture relation to account for previously
                # added objects
                edges_y += num_pts + p_i.shape[1]

                all_edges = np.hstack((all_edges, edges_i, edges_y)).astype(np.int)

        new_child = ChildFractureSet(all_p, all_edges, domain, parent_realiz)

        logger.info("Created new child, with properties: \n" + str(new_child))
        logging.debug(
            "Isolated children: %i, one y: %i, both y: %i",
            num_isolated,
            num_one_y,
            num_both_y,
        )

        return new_child

    def _draw_num_children(self, parent_realiz, pi):
        """ Draw the number of children for a fracture based on the statistical
        distribution.

        Parameters:
            parent_realiz (FractureSet): Fracture set for
            pi (int):

            These parameters are currently not in use. In the future, the number
            of children should scale with the length of the parent fracture.
        """
        return np.round(self.dist_num_children['dist'].rvs(1)[0] * parent_realiz.length()[pi]).astype(np.int)

    def _draw_children_along_parent(self, parent_realiz, pi, num_children):
        """ Define location of children along the lines of a parent fracture.

        The interpretation of the resulting coordinate depends on which type of
        fracture the child is: For an isolated node this will be the projection
        of the fracture center onto the parent. For y-nodes, the generated
        coordinate will be the end of the children that intersects with the
        parent.

        For the moment, the points are considered uniformly distributed along
        the parent fracture.

        Parameters:
            parent_realiz (FractureSet): Fracture set representing the parent
                of the realization being generated.
            pi (int): Index of the parent fracture these children will belong to.
            num_children (int): Number of children to be generated.

        Returns:
            np.array, 2 x num_children: Children points along the parent fracture.

        """

        # Start and end of the parent fracture
        start, end = parent_realiz.get_points(pi)

        dx = end - start

        p = start + np.random.rand(num_children) * dx
        if p.size == 2:
            p = p.reshape((-1, 1))
        return p

    def _draw_children_type(self, num_children, parent_realiz=None, pi=None, y_separately=False):
        """ Decide on which type of fracture is child is.

        The probabilities are proportional to the number of different fracture
        types in the original child (this object).

        Parameters:
            num_children: Number of fractures to generate
            parent_realiz (optional, defaults to None): Parent fracture set for this
                realization. Currently not used.
            pi (optional, int): Index of the current parent in this realization.
                Currently not used.

        Returns:
            np.array, boolean, length num_children: True for fractures that are
                to be isolated.
            np.array, boolean, length num_children: True for fractures that will
                have one T-node.
            np.array, boolean, length num_children: True for fractures that will
                have two T-nodes.

            Together, the return arrays should sum to the unit vector, that is,
            all fractures should be of one of the types.

        """
        rands = np.random.rand(num_children)
        is_isolated = rands < self.fraction_isolated
        rands -= self.fraction_isolated

        is_one_y = np.logical_and(
            np.logical_not(is_isolated), rands < self.fraction_one_y
        )

        is_both_y = np.logical_not(np.logical_or(is_isolated, is_one_y))

        if np.any(np.add.reduce((is_isolated, is_one_y, is_both_y)) != 1):
            # If we end up here, it is most likely a sign that the fractions
            # of different fracture types in the original set (this object)
            # do not sum to unity.
            raise ValueError("All fractures should be I, T or double T")

        if y_separately:
            return is_isolated, is_one_y, is_both_y
        else:
            return is_isolated, np.logical_or(is_one_y, is_both_y)

    def _generate_isolated_fractures(self, children_points, start_parent, end_parent):

        if children_points.size == 0:
            return np.empty((2, 0)), np.empty((2, 0))

        dx = end_parent - start_parent
        theta = np.arctan2(dx[1], dx[0])

        if children_points.ndim == 1:
            children_points = children_points.reshape((-1, 1))

        num_children = children_points.shape[1]

        dist_from_parent = frac_gen.generate_from_distribution(
            num_children, self.dist_from_parents
        )

        # Assign equal probability that the points are on each side of the parent
        side = 2 * (np.random.rand(num_children) > 0.5) - 1

        # Vector from the parent line to the new center points
        vec = np.vstack((-np.sin(theta), np.cos(theta))) * dist_from_parent

        children_center = children_points + side * vec

        child_angle = frac_gen.generate_from_distribution(num_children, self.dist_angle)
        child_length = frac_gen.generate_from_distribution(
            num_children, self.dist_length
        )

        p_start = children_center + 0.5 * child_length * np.vstack(
            (np.cos(child_angle), np.sin(child_angle))
        )
        p_end = children_center - 0.5 * child_length * np.vstack(
            (np.cos(child_angle), np.sin(child_angle))
        )

        p = np.hstack((p_start, p_end))
        edges = np.vstack(
            (np.arange(num_children), num_children + np.arange(num_children))
        )

        return p, edges

    def _generate_y_fractures(self, start, length_distribution=None):
        """ Generate fractures that originates in a parent fracture.

        Parameters:
            start (np.array, 2 x num_frac): Start point of the fractures. Will
                typically be located at a parent fracture.
            distribution (optional): Statistical distribution of fracture length.
                Used to define fracture length. If not provided, the attribute
                self.dist_length will be used.

        Returns:
            np.array (2 x 2*num_frac): Points that describe the generated fractures.
                The first num_frac points will be identical to start.
            np.array (2 x num_frac): Connection between the points. The first
                row correspond to start points, as provided in the input.

        """

        if length_distribution is None:
            length_distribution = self.dist_length

        if start.size == 0:
            return np.empty((2, 0)), np.empty((2, 0))

        if start.ndim == 1:
            start = start.reshape((-1, 1))

        num_children = start.shape[1]

        # Assign equal probability that the points are on each side of the parent
        side = 2 * (np.random.rand(num_children) > 0.5) - 1

        child_angle = frac_gen.generate_from_distribution(num_children, self.dist_angle)
        child_length = frac_gen.generate_from_distribution(
            num_children, length_distribution
        )

        # Vector from the parent line to the new center points
        vec = np.vstack((np.cos(child_angle), np.sin(child_angle))) * child_length

        end = start + side * vec

        p = np.hstack((start, end))
        edges = np.vstack(
            (np.arange(num_children), num_children + np.arange(num_children))
        )

        return p, edges

    def _generate_constrained_fractures(self, start, parent_realiz, constraints=None, y_separately=False):
        """
        """

        # Eventual return array for points
        p_found = np.empty((2, 0))

        # Special treatment if no fractures are generated
        if start.size == 0:
            # Create empty field for edges
            return p_found, np.empty((2, 0))

        if constraints is None:
            constraints = parent_realiz

        tol = 1e-4

        if y_separately:
            # Create fractures with the maximum allowed length for this distribution.
            # For this, we can use the function generate_y_fractures
            # The fractures will not be generated unless they cross a constraining
            # fracture, and the length will be adjusted accordingly
            p, edges = self._generate_y_fractures(start, self.dist_max_constrained_length)
        else:
            # Create the fracture with the standard length distribution
            p, edges = self._generate_y_fractures(start)

        num_children = edges.shape[1]

        start_parent, end_parent = parent_realiz.get_points()

        for ci in range(num_children):
            start = p[:, edges[0, ci]].reshape((-1, 1))
            end = p[:, edges[1, ci]].reshape((-1, 1))
            d, cp, cg_seg = pp.cg.dist_segment_segment_set(start, end, start_parent, end_parent)

            hit = np.where(d < tol)[0]
            if hit.size == 0:
                raise ValueError('Doubly constrained fractures should be constrained at its start point')
            elif hit.size == 1:
                if y_separately:
                    # The child failed to hit anything - this will not generate a
                    # constrained fracture
                    continue
                else:
                    # Attach this child as a singly-connected fracture
                    p_found = np.hstack((p_found, start, end))

            else:
                # The fracture has hit a constraint
                # Compute distance from all closest points to the start
                dist_start = np.sqrt(np.sum((start - cp[:, hit])**2, axis=0))
                # Find the first point along the line, away from the start
                first_constraint = np.argsort(dist_start)[1]
                p_found = np.hstack((p_found, start, cp[:, hit[first_constraint]].reshape((-1, 1))))

        # Finally define the edges, based on the fractures being ordered by
        # point pairs
        num_frac = p_found.shape[1] / 2
        e_found = np.vstack((2 * np.arange(num_frac), 1 + 2 * np.arange(num_frac)))

        return p_found, e_found

    def _fit_dist_from_parent_distribution(self, ks_size=100, p_val_min=0.05):
        """ For isolated fractures, fit a distribution for the distance from
        the child center to the parent fracture, orthogonal to the parent line.

        The function also evaluates the fitness of the chosen distribution by a
        Kolgomorov-Smirnov test.

        The function should be called after the field self.isolated_stats['center_distance']
        has been assigned, e.g. by calling self.compute_statistics()

        IMPLEMENTATION NOTE: The selection of appropriate distributions is a bit
        unclear. For the moment, we chose between uniform, lognormal and
        exponential distributions. More generally, this function can be made
        much more advanced, see for instance Xu and Dowd (Computers and
        Geosciences, 2010).

        Parameters:
            ks_size (int, optional): The number of realizations used in the
                Kolmogorov-Smirnov test. Defaults to 100.
            p_val_min (double, optional): P-value used in Kolmogorev-Smirnov test
                for acceptance of the chosen distribution. Defaults to 0.05.

        Returns:
            dictionary, with fields 'dist': The distribution with best fit.
                                    'param': Fitted parameters for the best
                                        ditribution.
                                    'p_val': P-value for the best distribution
                                        and parameters.

            If the fracture set contains no isolated fractures, and empty
            dictionary is returned.

        Raises:
            ValueError if none of the candidate distributions give a satisfactory
                fit.

        """
        data = self.isolated_stats["center_distance"]

        # Special case of no isolated fractures.
        if data.size == 0:
            return {}

        # Set of candidate distributions. This is somewhat arbitrary, better
        # options may exist
        candidate_dist = np.array([stats.uniform, stats.lognorm, stats.expon])
        # Fit each distribution
        dist_fit = np.array([d.fit(data, floc=0) for d in candidate_dist])

        # Inline function for Kolgomorov-Smirnov test
        ks = lambda d, p: stats.ks_2samp(data, d.rvs(*p, size=ks_size))[1]
        # Find the p-value for each of the candidate disributions, and their
        # fitted parameters
        p_val = np.array([ks(d, p) for d, p in zip(candidate_dist, dist_fit)])
        best_fit = np.argmax(p_val)

        if p_val[best_fit] < p_val_min:
            raise ValueError("p-value not satisfactory for length fit")

        self.dist_from_parents = {
            "dist": candidate_dist[best_fit],
            "param": dist_fit[best_fit],
            "pval": p_val[best_fit],
        }

    def _fit_num_children_distribution(self):
        """ Construct a Poisson distribution for the number of children per
        parent.

        Right now, it is not clear which data this should account for.

        The number of children should also account for the length of the
        parent fractures.
        """

        # Compute the intensity (dimensionless number of children)
        intensity = np.hstack(
            (self.isolated_stats["density"] + self.one_y_stats["density"] + self.both_y_stats['density'])
        ).astype(np.int)

        # For some reason, it seems scipy does not do parameter-fitting for
        # abstracting a set of data into a Poisson-distribution.
        # The below recipe is taken from
        #
        # https://stackoverflow.com/questions/25828184/fitting-to-poisson-histogram

        # Hand coded Poisson pdf
        def poisson(k, lamb):
            """poisson pdf, parameter lamb is the fit parameter"""
            return (lamb ** k / scipy.special.factorial(k)) * np.exp(-lamb)

        def negLogLikelihood(params, data):
            """ the negative log-Likelohood-Function"""
            lnl = -np.sum(np.log(poisson(data, params[0]) + 1e-5))
            return lnl

        # Use maximum likelihood fit. Use scipy optimize to find the best parameter
        result = scipy.optimize.minimize(
            negLogLikelihood,  # function to minimize
            x0=np.ones(1),  # start value
            args=(intensity,),  # additional arguments for function
            method="Powell",  # minimization method, see docs
        )
        ### End of code from stackoverflow

        # Define a Poisson distribution with the computed density function
        self.dist_num_children = stats.poisson(result.x)

    def fit_distributions(self, **kwargs):
        """ Compute statistical
        """

        # NOTE: Isolated nodes for the moment does not rule out that the child
        # intersects with a parent

        num_parents = self.parent.edges.shape[1]

        # Angle and length distribution as usual
        self.fit_angle_distribution(**kwargs)
        self.fit_length_distribution(**kwargs)

        node_types_self = analyze_intersections_of_sets(self, **kwargs)
        node_types_combined_self, node_types_combined_parent = analyze_intersections_of_sets(
            self, self.parent, **kwargs
        )

        # Find the number of Y-nodes that terminates in a parent node
        y_nodes_in_parent = (
            node_types_combined_self["y_nodes"] - node_types_self["y_nodes"]
        )

        # Fractures that ends in a parent fracture on both sides. If this is
        # a high number (whatever that means) relative to the total number of
        # fractures in this set, we may be better off by describing the set as
        # constrained
        both_y = np.where(y_nodes_in_parent == 2)[0]

        one_y = np.where(y_nodes_in_parent == 1)[0]

        isolated = np.where(node_types_combined_self["i_nodes"] == 2)[0]

        num_children = self.edges.shape[1]
        self.fraction_both_y = both_y.size / num_children
        self.fraction_one_y = one_y.size / num_children
        self.fraction_isolated = isolated.size / num_children

        self.isolated = isolated
        self.one_y = one_y
        self.both_y = both_y

        # Find the number of isolated fractures that cross a parent fracture.
        # Not sure how we will use this
        x_nodes_with_parent = (
            node_types_combined_self["x_nodes"] - node_types_self["x_nodes"]
        )
        intersections_of_isolated_nodes = x_nodes_with_parent[isolated]

        # Start and end points of the parent fractures

        # Treat isolated nodes
        if isolated.size > 0:
            density, center_distance = self._compute_line_density_isolated_nodes(
                isolated
            )
            self.isolated_stats = {
                "density": density / self.parent.length(),
                "center_distance": center_distance,
            }
            num_parents_with_isolated = np.sum(density > 0)
        else:
            num_parents_with_isolated = 0
            # The density is zero for all parent fratures.
            # Center-distance observations are empty.
            self.isolated_stats = {
                "density": np.zeros(num_parents),
                "center_distance": np.empty(0),
            }

        self._fit_dist_from_parent_distribution()

        ## fractures that have one Y-intersection with a parent
        # First, identify the parent-child relation
        if one_y.size > 0:
            density = self._compute_line_density_one_y_node(one_y)
            self.one_y_stats = {"density": density / self.parent.length()}
        else:
            # The density is zero for all parent fractures
            self.one_y_stats = {"density": np.zeros(num_parents)}

        if both_y.size > 0:
            # For the moment, we use the same computation as for one_y nodes
            # This will count all fractures twice. We try to compencate by
            # dividing the density by two, in effect saying that the fracture
            # has probability 0.5 to start from the fracture.
            # Hopefully that should not introduce bias.
            density = self._compute_line_density_one_y_node(both_y)
            self.both_y_stats = {"density": 0.5 * density / self.parent.length()}
        else:
            # The density is zero for all parent fractures
            self.both_y_stats = {"density": np.zeros(num_parents)}

        self._fit_num_children_distribution()

    def _compute_line_density_one_y_node(self, one_y):
        num_one_y = one_y.size

        start_parent, end_parent = self.parent.get_points()

        start_y, end_y = self.get_points(one_y)

        # Compute the distance from the start and end point of the children
        # to all parents
        # dist_start will here have dimensions num_children x num_parents
        # closest_pt_start has dimensions num_children x num_parents x dim (2)
        dist_start, closest_pt_start = pp.cg.dist_points_segments(
            start_y, start_parent, end_parent
        )
        dist_end, closest_pt_end = pp.cg.dist_points_segments(
            end_y, start_parent, end_parent
        )

        # For each child, identify which parent is the closest, and consider
        # only that distance and point
        closest_parent_start = np.argmin(dist_start, axis=1)
        dist_start = dist_start[np.arange(num_one_y), closest_parent_start]
        closest_pt_start = closest_pt_start[
            np.arange(num_one_y), closest_parent_start, :
        ].T
        # Then the end points
        closest_parent_end = np.argmin(dist_end, axis=1)
        dist_end = dist_end[np.arange(num_one_y), closest_parent_end]
        closest_pt_end = closest_pt_end[np.arange(num_one_y), closest_parent_end, :].T

        # Exactly one of the children end point should be on a parent
        # The tolerance used here is arbitrary.
        assert np.all(np.logical_or(dist_start < 1e-4, dist_end < 1e-4))

        start_closest = dist_start < dist_end

        num_parent = self.parent.num_frac
        num_occ_all = np.zeros(num_parent, dtype=np.object)

        for fi in range(num_parent):
            hit_start = np.logical_and(start_closest, closest_parent_start == fi)
            start_point_loc = closest_pt_start[:, hit_start]
            hit_end = np.logical_and(
                np.logical_not(start_closest), closest_parent_end == fi
            )
            end_point_loc = closest_pt_end[:, hit_end]
            p_loc = np.hstack((start_point_loc, end_point_loc))
            # Compute the number of points along the line.
            # Since we only ask for a single bin in the computation (nx=1),
            # we know there will be a single return value
            num_occ_all[fi] = self.compute_density_along_line(
                p_loc, start_parent[:, fi], end_parent[:, fi], nx=1
            )[0]

        return num_occ_all

    def _compute_line_density_isolated_nodes(self, isolated):
        # To ultimately describe the isolated fractures as a marked point
        # process, with stochastic location in terms of its distribution along
        # the fracture and perpendicular to it, we describe the distance from
        # the child center to its parent line.

        # There may be some issues with accumulation close to the end points
        # of the parent fracture; in particular if the orientation of the
        # child is far from perpendicular (meaning that a lot of it is outside
        # the 'span' of the parent), or if multiple parents are located nearby,
        # and we end up taking the distance to one that is not the natural
        # parent, whatever that means.
        # This all seems to confirm that ideall, a unique parent should be
        # identified for all children.

        # Start and end points of the parent fractures
        start_parent, end_parent = self.parent.get_points()

        center_of_isolated = 0.5 * (
            self.pts[:, self.edges[0, isolated]] + self.pts[:, self.edges[1, isolated]]
        )
        dist_isolated, closest_pt_isolated = pp.cg.dist_points_segments(
            center_of_isolated, start_parent, end_parent
        )

        # Minimum distance from center to a fracture
        num_isolated = isolated.size
        closest_parent_isolated = np.argmin(dist_isolated, axis=1)

        def dist_pt(a, b):
            return np.sqrt(np.sum((a - b) ** 2, axis=0))

        num_isolated = isolated.size

        # Distance from center of isolated node to the fracture (*not* its
        # prolongation). This will have some statistical distribution
        points_on_line = closest_pt_isolated[
            np.arange(num_isolated), closest_parent_isolated
        ].T
        pert_dist_isolated = dist_pt(center_of_isolated, points_on_line)

        num_occ_all = np.zeros(self.parent.edges.shape[1])

        # Loop over all parent fractures that are closest to some children.
        # Project the children onto the parent, compute a density map along
        # the parent.
        for counter, fi in enumerate(np.unique(closest_parent_isolated)):
            hit = np.where(closest_parent_isolated == fi)[0]
            p_loc = points_on_line[:, hit]
            num_occ_all[fi] = self.compute_density_along_line(
                p_loc, start_parent[:, fi], end_parent[:, fi], nx=1
            )

        return num_occ_all, pert_dist_isolated

    def _project_points_to_line(self, p, start, end):
        if p.ndim == 1:
            p = p.reshape((-1, 1))
        if start.ndim == 1:
            start = start.reshape((-1, 1))
        if end.ndim == 1:
            end = end.reshape((-1, 1))

        def _to_3d(pt):
            return np.vstack((pt, np.zeros(pt.shape[1])))

        p -= start
        end -= start
        theta = np.arctan2(end[1], end[0])

        assert np.abs(end[0] * np.sin(theta) + end[1] * np.cos(theta)) < 1e-5

        start_x = 0
        p_x = p[0] * np.cos(theta) - p[1] * np.sin(theta)
        end_x = end[0] * np.cos(theta) - end[1] * np.sin(theta)

        if end_x < start_x:
            domain_loc = {"xmin": end_x, "xmax": start_x}
        else:
            domain_loc = {"xmin": start_x, "xmax": end_x}

        # The density calculation computes the center of each fracture,
        # based on an assumption that the fracture consist of two points.
        # Make a line out of the points, with identical start and end points
        loc_edge = np.tile(np.arange(p_x.size), (2, 1))
        return p_x, loc_edge, domain_loc, theta

    def compute_density_along_line(self, p, start, end, **kwargs):

        p_x, loc_edge, domain_loc, _ = self._project_points_to_line(p, start, end)

        # Count the point density along this fracture.
        return frac_gen.count_center_point_densities(
            p_x, loc_edge, domain_loc, **kwargs
        )

    def snap(self, threshold):
        """ Modify point definition so that short branches are removed, and
        almost intersecting fractures become intersecting.

        Parameters:
            threshold (double): Threshold for geometric modifications. Points and
                segments closer than the threshold may be modified.

        Returns:
            FractureSet: A new ChildFractureSet with modified point coordinates.
        """
        # This function overwrites FractureSet.snap(), to ensure that the
        # returned fracture set also has a parent

        # We will not modify the original fractures
        p = self.pts.copy()
        e = self.edges.copy()

        # Prolong
        p = pp.cg.snap_points_to_segments(p, e, threshold)

        return ChildFractureSet(p, e, self.domain, self.parent)