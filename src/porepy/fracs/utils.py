import numpy as np


def determine_mesh_size(pts_split, lines_split, **kwargs):
    """
    Set the preferred mesh size for geometrical points as specified by
    gmsh.

    Currently, the only option supported is to specify a single value for
    all fracture points, and one value for the boundary.

    See the gmsh manual for further details.

    """
    mode = kwargs.get('mode', 'constant')
    num_pts = pts_split.shape[1]

    if mode == 'weighted':
        print( pts_split )
        print( lines_split )

        lines = lines_split[0:2, :]

        def dist(l):
            return np.linalg.norm(pts_split[:, l[0]] - pts_split[:, l[1]])
        lines_weight = np.array([dist(l) for l in lines.T])
        lines_weight = lines_weight / np.amax(lines_weight)

        pts_weight = np.tile(np.inf, num_pts)
        for i, line in enumerate(lines.T):
            pts_weight[line[0]] = np.amin([pts_weight[line[0]], lines_weight[i]])
            pts_weight[line[1]] = np.amin([pts_weight[line[1]], lines_weight[i]])

        print( pts_weight )

    #    pts_weight =
        print( lines_weight)
    else:
        pts_weight = np.ones(num_pts)

    if mode == 'constant' or mode == 'weighted':
        val = kwargs.get('value', None)
        bound_val = kwargs.get('bound_value', None)
#        weigh = kwargs.get('weight', np.ones(num_pts))
        bound_weigh = kwargs.get('bound_weight', 1)

        if val is not None:
            mesh_size = val * np.multiply(pts_weight, np.ones(num_pts))
        else:
            mesh_size = None
        if bound_val is not None:
            mesh_size_bound = bound_weigh * bound_val
        else:
            mesh_size_bound = None
        return mesh_size, mesh_size_bound
    else:
        raise ValueError('Unknown mesh size mode ' + mode)
