import warnings
import numpy as np

from porepy.fracs import meshing

#------------------------------------------------------------------------------#

def from_csv(f_name, mesh_kwargs, domain=None, pause=False, **kwargs):
    """
    Create the grid bucket from a set of fractures stored in a csv file and a
    domain. In the csv file, we assume the following structure:
    FID, START_X, START_Y, END_X, END_Y

    Where FID is the fracture id, START_X and START_Y are the abscissa and
    coordinate of the starting point, and END_X and END_Y are the abscissa and
    coordinate of the ending point.
    Note: the delimiter can be different.

    Parameters:
        f_name: the file name in CSV format
        mesh_kwargs: list of additional arguments for the meshing
        domain: rectangular domain, if not given the bounding-box is computed
        kwargs: list of argument for the numpy function genfromtxt

    Returns:
        gb: grid bucket associated to the configuration.
        domain: if the domain is not given as input parameter, the bounding box
        is returned.

    """
    pts, edges = fractures_from_csv(f_name, **kwargs)
    f_set = np.array([pts[:, e] for e in edges.T])

    # Define the domain as bounding-box if not defined
    if domain is None:
        overlap = kwargs.get('domain_overlap', 0)
        domain = _bounding_box(pts, overlap)
        return meshing.simplex_grid(f_set, domain, **mesh_kwargs), domain

    return meshing.simplex_grid(f_set, domain, **mesh_kwargs)

#------------------------------------------------------------------------------#

def fractures_from_csv(f_name, tagcols=None, **kwargs):
    """ Read csv file with fractures to obtain fracture description.

    Create the grid bucket from a set of fractures stored in a csv file and a
    domain. In the csv file, we assume the following structure:
    FID, START_X, START_Y, END_X, END_Y

    Where FID is the fracture id, START_X and START_Y are the abscissa and
    coordinate of the starting point, and END_X and END_Y are the abscissa and
    coordinate of the ending point.

    To change the delimiter from the default comma, use kwargs passed to
    np.genfromtxt.

    The csv file is assumed to have a header of 1 line. To change this number,
    use kwargs skip_header.

    Parameters:
        f_name (str): Path to csv file
        tagcols (array-like, int. Optional): Column index where fracture tags
            are stored. 0-offset. Defaults to no columns.
        **kwargs: keyword arguments passed on to np.genfromtxt.

    Returns:
        np.ndarray (2 x num_pts): Point coordinates used in the fracture
            description.
        np.ndarray (2+numtags x num_fracs): Fractures, described by their start
            and endpoints (first and second row). If tags are assigned to the
            fractures, these are stored in rows 2,...

    """
    npargs = {}
    # EK: Should these really be explicit keyword arguments?
    npargs['delimiter'] = kwargs.get('delimiter', ',')
    npargs['skip_header'] = kwargs.get('skip_header', 1)

    # Extract the data from the csv file
    data = np.genfromtxt(f_name, **npargs)
    if data.size == 0:
        return np.empty((2,0)), np.empty((2,0), dtype=np.int)
    data = np.atleast_2d(data)

    num_fracs = data.shape[0] if data.size > 0 else 0
    num_data = data.shape[1] if data.size > 0 else 0

    pt_cols = np.arange(1, num_data)
    if tagcols is not None:
        pt_cols = np.setdiff1d(pt_cols, tagcols)

    pts = data[:, pt_cols].reshape((-1, 2)).T

    # Let the edges correspond to the ordering of the fractures
    edges = np.vstack((np.arange(0, 2*num_fracs, 2),
                       np.arange(1, 2*num_fracs, 2)))
    if tagcols is not None:
        edges = np.vstack((edges, data[:, tagcols].T))

    return pts, edges.astype(np.int)

#------------------------------------------------------------------

def _bounding_box(pts, overlap=0):
    """ Obtain a bounding box for a point cloud.

    Parameters:
        pts: np.ndarray (nd x npt). Point cloud. nd should be 2 or 3
        overlap (double, defaults to 0): Extension of the bounding box outside
            the point cloud. Scaled with extent of the point cloud in the
            respective dimension.

    Returns:
        domain (dictionary): Containing keywords xmin, xmax, ymin, ymax, and
            possibly zmin and zmax (if nd == 3)

    """
    max_coord = pts.max(axis=1)
    min_coord = pts.min(axis=1)
    dx = max_coord - min_coord
    domain = {'xmin': min_coord[0] - dx[0] * overlap,
              'xmax': max_coord[0] + dx[0] * overlap,
              'ymin': min_coord[1] - dx[1] * overlap,
              'ymax': max_coord[1] + dx[1] * overlap}

    if max_coord.size == 3:
        domain['zmin'] = min_coord[2] - dx[2] * overlap
        domain['zmax'] = max_coord[2] + dx[2] * overlap
    return domain
