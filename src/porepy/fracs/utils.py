import numpy as np


def determine_mesh_size(num_pts, **kwargs):
    """
    Set the preferred mesh size for geometrical points as specified by
    gmsh.

    Currently, the only option supported is to specify a single value for
    all fracture points, and one value for the boundary.

    See the gmsh manual for further details.

    """
    mode = kwargs.get('mode', 'constant')

    if mode == 'constant':
        val = kwargs.get('value', None)
        bound_val = kwargs.get('bound_value', None)
        if val is not None:
            mesh_size = val * np.ones(num_pts)
        else:
            mesh_size = None
        if bound_val is not None:
            mesh_size_bound = bound_val
        else:
            mesh_size_bound = None
        return mesh_size, mesh_size_bound
    else:
        raise ValueError('Unknown mesh size mode ' + mode)
