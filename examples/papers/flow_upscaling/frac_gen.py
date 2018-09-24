import numpy as np
import scipy.stats as stats

def fit(pts, edges, frac, family, ks_size=100, p_val_min = 0.05):
    """
    Compute the distribution from a set of fracture families for the length and angle.

    Parameters:
    pts: list of points
    edges: list of edges as point ids
    frac: fracture identification number for each edge
    family: family identification number for each edge
    ks_size: (optional) sample size for the Kolmogorov-Simirnov test, default 100
    p_val_min: (optional) minimum p-value to validate the goodness of the fitting

    Return:
    dist_l: for each family distribution for the fracture length
    dist_a: for each family distribution for the fracture angle

    Note:
    1) so far this implementation does not take care of the family
    2) the angle should be divided in two categories, since we have conjugate fractures
    """

    # fit the lenght distribution
    dist = np.array([stats.expon, stats.lognorm])

    # fit the possible lenght distributions
    l = length(pts, edges, frac)
    dist_fit = np.array([d.fit(l, floc=0) for d in dist])

    # determine which is the best distribution with a Kolmogorov-Smirnov test
    ks = lambda d, p: stats.ks_2samp(l, d.rvs(*p, size=ks_size))[1]
    p_val = np.array([ks(d, p) for d, p in zip(dist, dist_fit)])
    best_fit = np.argmax(p_val)

    if p_val[best_fit] < p_val_min:
        raise ValueError("p-value not satisfactory for length fit")

    # collect the data
    dist_l = {"dist": dist[best_fit], "param": dist_fit[best_fit], "p_val": p_val[best_fit]}

    # start the computation for the angles
    dist = stats.vonmises
    a = angle(pts, edges, frac)
    dist_fit = dist.fit(a, fscale=1)

    # check the goodness of the fit with Kolmogorov-Smirnov test
    p_val = stats.ks_2samp(a, dist.rvs(*dist_fit, size=ks_size))[1]

    if p_val < p_val_min:
        raise ValueError("p-value not satisfactory for angle fit")

    # collect the data
    dist_a = {"dist": dist, "param": dist_fit, "p_val": p_val}

    return dist_l, dist_a

def generate(pts, edges, frac, dist_l, dist_a):

    num_frac = np.unique(frac).size
    # generate lenght and angle
    l = dist_l["dist"].rvs(*dist_l["param"], num_frac)
    a = dist_a["dist"].rvs(*dist_a["param"], num_frac)

    # first compute the fracture centres and then generate them
    avg = lambda e0, e1: 0.5*(pts[:, e0] + pts[:, e1])
    pts_c = np.array([avg(e[0], e[1]) for e in edges.T]).T

    # compute the mean centre based on the fracture id
    mean_c = lambda f: np.mean(pts_c[:, np.isin(frac, f)], axis=1)
    mean_c = np.array([mean_c(f) for f in np.unique(frac)]).T

    dist_c = stats.uniform.rvs
    c = dist_c(np.amin(mean_c, axis=1), np.amax(mean_c, axis=1), (num_frac, 2)).T

    # generate the new set of pts and edges
    pts_n = np.empty((2, l.size*2))
    delta = 0.5 * l * np.array([np.cos(a), np.sin(a)])
    for i in np.arange(num_frac):
        pts_n[:, 2*i] = c[:, i] + delta[:, i]
        pts_n[:, 2*i+1] = c[:, i] - delta[:, i]

    edges_n = np.array([2*np.arange(num_frac), 2*np.arange(1, num_frac+1)-1])

    return pts_n, edges_n

def length(pts, edges, frac):
    """
    Compute the total length of the fractures, based on the fracture id.
    The output array has length as unique(frac) and ordered from the lower index
    to the higher.

    Parameters:
    pts: list of points
    edges: list of edges as point ids
    frac: fracture identification number for each edge

    Return:
    length: total length for each fracture
    """

    # compute the length for each segment
    norm = lambda e0, e1: np.linalg.norm(pts[:, e0] - pts[:, e1])
    l = np.array([norm(e[0], e[1]) for e in edges.T])

    # compute the total length based on the fracture id
    tot_l = lambda f: np.sum(l[np.isin(frac, f)])
    return np.array([tot_l(f) for f in np.unique(frac)])

def angle(pts, edges, frac):
    """
    Compute the mean angle of the fractures, based on the fracture id.
    The output array has length as unique(frac) and ordered from the lower index
    to the higher.

    Parameters:
    pts: list of points
    edges: list of edges as point ids
    frac: fracture identification number for each edge

    Return:
    angle: mean angle for each fracture
    """

    # compute the angle for each segment
    alpha = lambda e0, e1: np.arctan2(pts[1, e0] - pts[1, e1], pts[0, e0] - pts[0, e1])
    a = np.array([alpha(e[0], e[1]) for e in edges.T])

    # compute the mean angle based on the fracture id
    mean_alpha = lambda f: np.mean(a[np.isin(frac, f)])
    mean_a = np.array([mean_alpha(f) for f in np.unique(frac)])

    # we want only angles in (0, pi)
    mask = mean_a < 0
    mean_a[mask] = np.pi - np.abs(mean_a[mask])
    mean_a[mean_a > np.pi] -= np.pi

    return mean_a


def count_center_point_densities(p, e, domain, nx=10, ny=10):
    """ Divide the domain into boxes, count the number of fracture centers
    contained within each box.

    Parameters:
        p (np.array, 2 x n): Point coordinates of the fractures
        e (np.array, 2 x n): Connections between the coordinates
        domain (dictionary): Description of the simulation domain. Should
            contain fields xmin, xmax, ymin, ymax.
        nx, ny (int, optional): Number of boxes in x and y direction. Defaults
            to 10.

    Returns:
        np.array (nx x ny): Number of centers within each box

    """

    pc = _compute_center(p, e)

    x0, y0, dx, dy = _decompose_domain(domain, nx, ny)


    num_occ = np.zeros((nx, ny))

    # Can probably do this more vectorized, but for now, a for loop will suffice
    for i in range(nx):
        for j in range(ny):
            hit = np.logical_and.reduce([pc[0] > (x0 + i * dx), pc[0] < (x0 + (i + 1) * dx),
                                         pc[1] > (y0 + j * dy), pc[1] < (y0 + (j + 1) * dy)])
            num_occ[i, j] = hit.sum()

    return num_occ

def define_centers_by_boxes(domain, intensity, distribution='poisson'):
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


    if distribution != 'poisson':
        return ValueError('Only Poisson point processes have been implemented')

    nx, ny = intensity.shape
    num_boxes = intensity.size

    max_intensity = intensity.max()

    x0, y0, dx, dy = _decompose_domain(domain, nx, ny)

    # It is assumed that the intensities are computed relative to boxes of the
    # same size that are assigned in here
    area_of_box=1

    pts = np.empty(num_boxes, dtype=np.object)

    # First generate the full set of points with maximum intensity
    counter = 0
    for i in range(nx):
        for j in range(ny):
            num_p_loc = stats.poisson(max_intensity * area_of_box).rvs(1)[0]
            p_loc = np.random.rand(2, num_p_loc)
            p_loc[0] = x0 + i * dx + p_loc[0] * dx
            p_loc[1] = y0 + i * dy + p_loc[1] * dy
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

    return np.array([pts[i][:, j] for i in range(pts.size) for j in range(pts[i].shape[1])]).T


def _compute_center(p, edges):
    # first compute the fracture centres and then generate them
    avg = lambda e0, e1: 0.5*(p[:, e0] + p[:, e1])
    pts_c = np.array([avg(e[0], e[1]) for e in edges.T]).T
    return pts_c

def _decompose_domain(domain, nx, ny):
    x0 = domain['xmin']
    y0 = domain['ymin']
    dx = (domain['xmax'] - domain['xmin']) / nx
    dy = (domain['ymax'] - domain['ymin']) / ny
    return x0, y0, dx, dy

