import numpy as np

import porepy as pp
from porepy.numerics.fv import fvutils
from porepy.numerics.fv.fv_elliptic import FVElliptic


class MpfaUpscaling(FVElliptic):
    def __init__(self, keyword):
        super(MpfaUpscaling, self).__init__(keyword)

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells (pressure dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

    def discretize(self, g, data):
        """
        The data should contain a parameter class under the field "param".
        The following parameters will be accessed:
        get_tensor : SecondOrderTensor. Permeability defined cell-wise.
        get_bc : boundary conditions
        get_robin_weight : float. Weight for pressure in Robin condition
        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        """
        param = data["param"]
        #k = param.get_tensor(self)
        k = 1
        #bnd = param.get_bc(self)
        bnd = None

        flux, bound_flux, bound_pressure_cell, bound_pressure_face = \
            self._local_discr(g, k, bnd, data)

        data[self._key() + "flux"] = flux
        data[self._key() + "bound_flux"] = bound_flux
        data[self._key() + "bound_pressure_cell"] = bound_pressure_cell
        data[self._key() + "bound_pressure_face"] = bound_pressure_face

    def _local_discr(self, g, k, bnd, data):

        # data specific for the local mesh, the fractures are a dictionary
        # composed by a field "points" containing all the points and a field
        # "edges" containing all the fractures
        fracs = data["fractures"]
        mesh_args_loc = data["mesh_args_loc"]

        # map of faces connected to a face via its nodes
        face_faces = g.face_nodes.T*g.face_nodes
        # map of cells connected to a face via its nodes
        face_cells = g.face_nodes.T*g.cell_nodes()
        # all the nodes at the boundary of the grid
        b_nodes = g.get_all_boundary_nodes()

        # NOTE I'm assuming a 2d grid
        assert g.dim == 2

        tol_small = 1e-5

        for f in g.get_internal_faces():

            # nodes of the current face
            f_nodes = self._slice_out(f, g.face_nodes)
            f_nodes_coord = g.nodes[:, f_nodes]

            # we construct the local grid by considering the 1) patch, 2) the face and
            # the 3) local fractures

            # 1) get the patch composed by a list of ordered points
            pts_patch, is_cell_centers = self._patch(g, f, face_faces, face_cells, b_nodes)

            # 2) to make the grid conforming consider the current face as "subdomain"
            # TODO in principle also all the macro faces are constraints for the local grid
            subdom_edges = np.arange(f_nodes.size).reshape((g.dim, 1))
            subdom = {"points": f_nodes_coord[:g.dim], "edges": subdom_edges}

            # 3) consider the background fractures only limited to the current region
            fracs_int = self._patch_fracs(pts_patch, fracs)

            # construct the patch grid bucket
            gb_patch = self._patch_gb(fracs_int, pts_patch[:g.dim], subdom, mesh_args_loc)
            pp.plot_grid(gb_patch, alpha=0)

            # we need to identify the faces in the gb_patch that belong to the original face
            g_h = gb_patch.grids_of_dimension(g.dim)[0]

            # here I assume that we are in 2d
            dist, _ = pp.cg.dist_points_segments(g_h.face_centers, f_nodes_coord[:, 0], f_nodes_coord[:, 1])
            f_loc = np.where(np.logical_and(dist < tol_small, dist >=-tol_small))[0]

            # compute the sign with respect to f and normalize the result
            sign = np.sign(np.einsum('ij,i->j', g_h.face_normals[:, f_loc], g.face_normals[:, f]))

            dd

            # assign common data
            set_data(gb_patch, data)

            continue
            # need to compute the upscaled transmissibility
            # we loop on all the cell_centers and give the data
            for pt in np.where(is_cell_centers)[0]:
                # apply the boundary conditions for the current problem
                set_boundary_conditions(gb_patch, pts, pt, tol)

                # Discretize and solve the linear system
                p = sps.linalg.spsolve(*solver.matrix_rhs(gb_patch))

                # compute the discharge and fix the sign with f
                solver.split(gb_patch, "pressure", p)
                pp.fvutils.compute_discharges(gb_patch)
                discharge = sign * gb_patch.node_props(g_h, "discharge")[f_loc]



    def _patch(self, g, f, face_faces, face_cells, b_nodes):
        """
        Construct the local patch of the input face f
        """
        # extract the faces (removing self), cells, and nodes
        f_loc = self._slice_out(f, face_faces)
        f_loc = np.setdiff1d(f_loc, f)
        c_loc = self._slice_out(f, face_cells)
        n_loc = self._slice_out(f, g.face_nodes)

        # keep the face nodes if are on the boundary
        n_loc = n_loc[np.isin(n_loc, b_nodes, assume_unique=True)]

        # we need to construct the patch
        # collect from: faces, cells, node boundary (if)
        pts = np.hstack((g.cell_centers[:, c_loc], g.face_centers[:, f_loc], g.nodes[:, n_loc]))
        # TODO need a better name here
        is_cell_centers = np.hstack(([True]*c_loc.size, [False]*(f_loc.size+n_loc.size)))

        # sort the nodes by assuming star shaped region
        mask = pp.utils.sort_points.sort_point_plane(pts, g.face_centers[:, f])
        pts = pts[:, mask]
        is_cell_centers = is_cell_centers[mask]

        return pts, is_cell_centers

    def _patch_fracs(self, pts_patch, fracs):
        """
        Intersect the fractures to get only the one related to the current patch
        """
        # intersect the local fractures with the patch
        pts, edges = pp.cg.intersect_polygon_lines(pts_patch, \
                                                   fracs["points"], fracs["edges"])
        return {"points": pts, "edges": edges}

    def _patch_gb(self, fracs, patch, subdom, mesh_args_loc):
        """
        Construct the grid bucket for the current patch
        """
        return pp.fracs.meshing.simplex_grid(fracs, domain=patch, \
                                             subdomains=subdom, **mesh_args_loc)

    def _slice_out(self, f, m):
        return m.indices[m.indptr[f]: m.indptr[f+1]]

