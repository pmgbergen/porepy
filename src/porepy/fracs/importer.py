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

    kwargs['delimiter'] = kwargs.get('delimiter', ',')
    kwargs['skip_header'] = kwargs.get('skip_header', 1)

    # Extract the data from the csv file
    data = np.atleast_2d(np.genfromtxt(f_name, **kwargs))

    num_fracs = data.shape[0] if data.size > 0 else 0
    #family = data[:, 0].astype('int')

    # Sort the points fracture-wise
    pts = data[:, 1:].reshape((-1, 2)).T

    # Let the edges correspond to the ordering of the fractures
    edges = np.vstack((np.arange(0, 2*num_fracs, 2),
                       np.arange(1, 2*num_fracs, 2)))

    f_set = np.array([pts[:, e] for e in edges.T])

    # Define the domain as bounding-box if not defined
    if domain is None:
        max_coord = pts.max(axis=1)
        min_coord = pts.min(axis=1)

        domain = {'xmin': min_coord[0], 'xmax': max_coord[0],
                  'ymin': min_coord[1], 'ymax': max_coord[1]}

        return meshing.simplex_grid(f_set, domain, **mesh_kwargs), domain

    return meshing.simplex_grid(f_set, domain, **mesh_kwargs)

#------------------------------------------------------------------------------#
