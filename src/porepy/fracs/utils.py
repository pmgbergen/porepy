import numpy as np
from porepy.utils import comp_geom as cg

#def determine_mesh_size(num_pts, **kwargs):
#        https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html#two-dimensional-spline-representation-procedural-bisplrep
def determine_mesh_size(pts, lines, **kwargs):
    """
    Set the preferred mesh size for geometrical points as specified by
    gmsh.

    Currently, the only option supported is to specify a single value for
    all fracture points, and one value for the boundary.

    See the gmsh manual for further details.

    """
    mode = kwargs.get('mode', 'constant')
    num_pts = pts.shape[1]

    if mode == 'weighted':

        val = kwargs.get('value', 1)
        bound_val = kwargs.get('bound_value', None)

        pts_id = lines[:2, :]
        dist = np.linalg.norm(pts[:, pts_id[0, :]] - pts[:, pts_id[1, :]],
                              axis=0)
        dist_pts = np.tile(np.inf, pts.shape[1])

        for i, pt_id in enumerate(pts_id.T):
            distances = np.array([dist_pts[pt_id], [dist[i]]*2, [val]*2])
            dist_pts[pt_id] = np.amin(distances, axis=0)

        num_pts = pts.shape[1]
        num_lines = lines.shape[1]
        pts_extra = np.empty((2, 0))
        dist_extra = np.empty(0)
        pts_id_extra = np.empty(0, dtype=np.int)

        # Loop on all the original points
        for pt_id, pt in enumerate(pts.T):
            # Loop on all the original lines
            for line in lines.T:
                start, end = pts[:, line[0]], pts[:, line[1]]
                dist, pt_int = cg.distance_point_segment(pt, start, end)
                if dist < val and not np.isclose(dist, 0): # add a factor to val
                    dist_pts[pt_id] = min(dist_pts[pt_id], dist)

                    dist_start = np.linalg.norm(pt_int - start)
                    dist_end = np.linalg.norm(pt_int - end)
                    if dist < dist_start and dist < dist_end:
                        dist_extra = np.r_[dist_extra, min(dist, val)]
                        pts_extra = np.c_[pts_extra, pt_int]
                        pts_id_extra = np.r_[pts_id_extra, line[3]]

        extra_ids, inv_index = np.unique(pts_id_extra, return_inverse=True)
        to_remove = np.empty(0, dtype=np.int)
        for idx, i in enumerate(extra_ids):
            mask = np.flatnonzero(inv_index == idx)
            if mask.size > 1:
                mesh_matrix = np.tile(dist_extra[mask], (mask.size, 1))
                pos_matrix = np.zeros((mask.size, mask.size))
                dist_matrix = np.ones((mask.size, mask.size))*np.inf

                for pt1_id_loc in np.arange(mask.size):
                    for pt2_id_loc in np.arange(pt1_id_loc+1, mask.size):
                        pt1_id = mask[pt1_id_loc]
                        pt2_id = mask[pt2_id_loc]
                        pt1 = pts_extra[:, pt1_id]
                        pt2 = pts_extra[:, pt2_id]
                        pts_id = np.array([pt1_id, pt2_id])
                        pts_id_loc = np.array([pt1_id_loc, pt2_id_loc])
                        pos_min = np.argmin(dist_extra[pts_id])
                        pos_max = np.argmax(dist_extra[pts_id])
                        dist_matrix[pts_id_loc[pos_min], pts_id_loc[pos_max]] = \
                                                       np.linalg.norm(pt1 - pt2)

                to_remove_loc = np.any(dist_matrix < mesh_matrix, axis=0)
                to_remove = np.r_[to_remove, mask[to_remove_loc]]

        pts_extra = np.delete(pts_extra, to_remove, axis=1)
        dist_extra = np.delete(dist_extra, to_remove)
        pts_id_extra = np.delete(pts_id_extra, to_remove)

        pts = np.c_[pts, pts_extra]
        dist_pts = np.r_[dist_pts, dist_extra]

        seg_ids = np.unique(lines[3, :])
        new_lines = np.empty((4,0), dtype=np.int)
        for seg_id in seg_ids:
            mask_bool = lines[3, :] == seg_id
            extra_mask_bool = pts_id_extra == seg_id
            if not np.any(extra_mask_bool):
                new_lines = np.c_[new_lines, lines[:, mask_bool]]
            else:
                pts_frac_id = np.hstack((lines[0:2, mask_bool].ravel(),
                                     np.flatnonzero(extra_mask_bool) + num_pts))
                pts_frac_id = np.unique(pts_frac_id)
                pts_frac = pts[:, pts_frac_id]
                pts_frac_id = pts_frac_id[cg.argsort_point_on_line(pts_frac)]
                pts_frac_id = np.vstack((pts_frac_id[:-1], pts_frac_id[1:]))
                other_info = np.tile(lines[2:, mask_bool][:, 0],
                                                    (pts_frac_id.shape[1], 1)).T
                new_lines = np.c_[new_lines, np.vstack((pts_frac_id, other_info))]


        relax = kwargs.get('relaxation', 0.8)
        lines = np.empty((4, 0), dtype=np.int)
        for seg in new_lines.T:
            mesh_size_pt1 = dist_pts[seg[0]]
            mesh_size_pt2 = dist_pts[seg[1]]
            dist = np.linalg.norm(pts[:, seg[0]] - pts[:, seg[1]])
            if (mesh_size_pt1 >= relax*val and mesh_size_pt2 >= relax*val) \
               or \
               (relax*dist <= 2*mesh_size_pt1 and relax*dist <= 2*mesh_size_pt2):
                lines = np.c_[lines, seg]
            else:
                pt_id = pts.shape[1]
                pts = np.c_[pts, 0.5*(pts[:, seg[0]] + pts[:, seg[1]])]
                mesh_size = np.amin([val, dist/2.])
                dist_pts = np.r_[dist_pts, mesh_size]
                lines = np.c_[lines, [seg[0], pt_id, seg[2], seg[3]],
                                     [pt_id, seg[1], seg[2], seg[3]]]


        return dist_pts, bound_val, pts, lines
#        return dist_pts, bound_val, pts, new_lines

    elif mode == 'constant':
        val = kwargs.get('value', None)
        bound_val = kwargs.get('bound_value', None)
        bound_weigh = kwargs.get('bound_weight', 1)

        if val is not None:
            mesh_size = val * np.ones(num_pts)
        else:
            mesh_size = None
        if bound_val is not None:
            mesh_size_bound = bound_weigh * bound_val
        else:
            mesh_size_bound = None

        return mesh_size, mesh_size_bound, pts, lines
    else:
        raise ValueError('Unknown mesh size mode ' + mode)
