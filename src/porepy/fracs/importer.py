import warnings
import numpy as np

from porepy.fracs import meshing

#------------------------------------------------------------------------------#

def from_csv(f_name, mesh_kwargs, domain=None, pause=False,\
             return_domain=False, **kwargs):
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
        domain = bounding_box(pts, overlap)

    if return_domain:
        return meshing.simplex_grid(f_set, domain, **mesh_kwargs), domain
    else:
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

def from_fab(f_name):
    """ Read fractures from a .fab file, as specified by FracMan.

    The filter is based on the .fab-files available at the time of writing, and
    may not cover all options available.

    Parameters:
        f_name (str): Path to .fab file.

    Returns:
        fracs (list of np.ndarray): Each list element contains fracture
            vertexes as a nd x n_pt array.
        tess_fracs (list of np.ndarray): Each list element contains fracture
            cut by the domain boundary, represented by vertexes as a nd x n_pt
            array.
        tess_frac (np.ndarray): For each element in tess_frac, a +-1 defining
            which boundary the fracture is on.

    The function also reads in various other information of unknown usefulness,
    see implementation for details. This information is currently not returned.

    """

    def read_keyword(line):
        # Read a single keyword, on the form  key = val
        words = line.split('=')
        assert len(words) == 2
        key = words[0].strip()
        val = words[1].strip()
        return key, val

    def read_section(f, section_name):
        # Read a section of the file, surrounded by a BEGIN / END wrapping
        d = {}
        for line in f:
            if line.strip() == 'END ' + section_name.upper().strip():
                return d
            k, v = read_keyword(line)
            d[k] = v

    def read_fractures(f, is_tess=False):
        # Read the fracture
        fracs = []
        fracture_ids = []
        trans = []
        nd = 3
        for line in f:
            if not is_tess and line.strip() == 'END FRACTURE':
                return fracs, np.asarray(fracture_ids), np.asarray(trans)
            elif is_tess and line.strip() == 'END TESSFRACTURE':
                return fracs, np.asarray(fracture_ids), np.asarray(trans)
            if is_tess:
                ids, num_vert = line.split()
            else:
                ids, num_vert, t = line.split()
                trans.append(float(t))

            ids = int(ids)
            num_vert = int(num_vert)
            vert = np.zeros((num_vert, nd))
            for i in range(num_vert):
                data = f.readline().split()
                vert[i] = np.asarray(data[1:])

            # Transpose to nd x n_pt format
            vert = vert.T

            # Read line containing normal vector, but disregard result
            data = f.readline().split()
            if is_tess:
                trans.append(int(data[1]))
            fracs.append(vert)
            fracture_ids.append(ids)

    with open('DFN.fab', 'r') as f:
        for line in f:
            if line.strip() == 'BEGIN FORMAT':
                # Read the format section, but disregard the information for
                # now
                formats = read_section(f, 'FORMAT')
            elif line.strip() == 'BEGIN PROPERTIES':
                # Read in properties section, but disregard information
                props = read_section(f, 'PROPERTIES')
            elif line.strip() == 'BEGIN SETS':
                # Read set section, but disregard information.
                sets = read_section(f, 'SETS')
            elif line.strip() == 'BEGIN FRACTURE':
                # Read fractures
                fracs, frac_ids, trans = read_fractures(f, is_tess=False)
            elif line.strip() == 'BEGIN TESSFRACTURE':
                # Read tess_fractures
                tess_fracs, tess_frac_ids, tess_sgn = read_fractures(f, is_tess=True)
            elif line.strip()[:5] == 'BEGIN':
                # Check for keywords not yet implemented.
                raise ValueError('Unknown section type ' + line)

    return fracs, tess_fracs, tess_sgn


def bounding_box(pts, overlap=0):
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
