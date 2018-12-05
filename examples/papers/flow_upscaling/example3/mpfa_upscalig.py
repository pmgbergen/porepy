import numpy as np
import scipy.sparse as sps

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
        bnd = param.get_bc(self)

        data_patch = data["local_problem"]

        flux, bound_flux, bound_pressure_cell, bound_pressure_face = \
            self._local_discr(g, k, bnd, data_patch)

        data[self._key() + "flux"] = flux
        data[self._key() + "bound_flux"] = bound_flux
        data[self._key() + "bound_pressure_cell"] = bound_pressure_cell
        data[self._key() + "bound_pressure_face"] = bound_pressure_face

    def _local_discr(self, g, k, bnd, data):

        # data specific for the local mesh, the fractures are a dictionary
        # composed by a field "points" containing all the points and a field
        # "edges" containing all the fractures
        fracs = data["fractures"]
        mesh_args = data["mesh_args"]

        # map of faces connected to a face via its nodes
        patch_face_faces = g.face_nodes.T*g.face_nodes
        # map of cells connected to a face via its nodes
        patch_face_cells = g.face_nodes.T*g.cell_nodes()

        node_faces = g.face_nodes.tocsr().T
        face_cells = g.cell_faces.tocsr().T

        # NOTE I'm assuming a 2d grid, I tryed to be general enough
        assert g.dim == 2

        # count the number of data that has to be saved in the flux matrix
        internal_faces = g.get_internal_faces()
        size = np.sum(patch_face_cells.indptr[internal_faces+1]-\
                      patch_face_cells.indptr[internal_faces])

        I = np.zeros(size)
        J = np.zeros(size)
        dataIJ = np.zeros(size)

        idx = 0
        for f in internal_faces:

            # nodes of the current face
            f_nodes = self._slice_out(f, g.face_nodes)
            f_nodes_coord = g.nodes[:, f_nodes]

            # we construct the local grid by considering the 1) patch, 2) the face and
            # the 3) local fractures

            # 1) get the patch composed by a list of ordered points
            patch_pts, bc_type, bc_corner = self._patch(g, f, bnd, patch_face_faces, patch_face_cells, node_faces, face_cells)

            # 2) to make the grid conforming consider the current face as "subdomain"
            # with auxiliary as tag
            subdom = {"points": f_nodes_coord[:g.dim],
                      "edges": np.arange(f_nodes.size).reshape((g.dim, 1))}

            # 3) consider the background fractures only limited to the current region
            patch_fracs = self._patch_fracs(patch_pts, fracs)

            # construct the patch grid bucket
            patch_gb = self._patch_gb(patch_fracs, patch_pts[:g.dim], subdom, mesh_args)
            #pp.plot_grid(patch_gb, alpha=0, info="f")

            # we need to identify the faces in the gb_patch that belong to the original face
            # this is used afterward to compute the discharge
            # since we have used the subdomain concept an auxiliary tag is provided by gmsh
            g_h = patch_gb.grids_of_dimension(g.dim)[0]
            f_loc = [v for k, v in g_h.tags.items() if "auxiliary_" in k][0]
            f_loc = np.where(f_loc)[0]

            # compute the sign with respect to f and normalize the result
            sign = np.sign(np.einsum('ij,i->j', g_h.face_normals[:, f_loc], g.face_normals[:, f]))

            # assign common data, defined by the user. Mostly it's the aperture and
            # permeability for all the objects
            self._patch_set_data(patch_gb, data)

            # need to compute the upscaled transmissibility
            # we loop on all the cell_centers and give the data
            for pt in np.where(bc_corner[0])[0]:

                # apply the boundary conditions for the current problem
                self._patch_set_bc(patch_gb, patch_pts, pt, bc_type, data["tol"])

                # solution operator, given the patch grid bucket return the
                # discharge computed
                discharge = np.sum(sign * data["compute_discharge"](patch_gb)[f_loc])

                print("discharge", discharge, bc_corner[:, pt])

                if bc_corner[1, pt] == 0:
                    dataIJ[idx] = discharge
                    I[idx] = f
                    J[idx] = bc_corner[2, pt]
                    idx += 1

        import pdb; pdb.set_trace()
        shape = (g.num_faces, g.num_cells)
        flux = sps.csr_matrix((dataIJ, (I, J)), shape=shape)

    def _patch(self, g, f, bnd, patch_face_faces, patch_face_cells, node_faces, face_cells):
        """
        Construct the local patch of the input face f
        """

        # all the faces at the boundary of the grid
        b_faces = g.tags["domain_boundary_faces"]

        # all the nodes at the boundary of the grid
        b_nodes = g.tags["domain_boundary_nodes"]

        # extract the faces (removing self), cells, and nodes
        f_loc = self._slice_out(f, patch_face_faces)
        f_loc = np.setdiff1d(f_loc, f)
        c_loc = self._slice_out(f, patch_face_cells)
        n_loc = self._slice_out(f, g.face_nodes)

        # keep the face nodes if are on the boundary
        n_loc = n_loc[b_nodes[n_loc]]

        # where to store the boundary ordered points of the patch
        pts = np.zeros((g.dim, f_loc.size + c_loc.size + n_loc.size))

        # second line 0 = cell, 1 = node. third line id
        bc_corner = -1 * np.ones((3, pts.shape[1]))
        bc_corner[0] = 0

        # devo introdurre una logica per passare le informazioni dei bordi, magari
        # appunto tramite gli eedges
        # it connects pts i to i+1
        bc_type = np.zeros((2, pts.shape[1]), dtype=np.object)
        bc_type[0] = "dir"
        bc_type[1] = False

        pts_pos = 0
        face = f_loc[0]
        faces = np.array([face])

        # save the type of boundary condition associate with the current
        # face
        if b_faces[face]:
            bc_tag = bnd.is_dir[face] * "dir" + \
                     bnd.is_neu[face] * "neu" + \
                     bnd.is_rob[face] * "rob"
            bc_type[:, pts_pos] = [bc_tag, True]

        while f_loc.size:

            # select the face
            mask = np.where(np.isin(faces, f_loc))[0][0]
            face = faces[mask]

            # we add the face in the list
            pts[:g.dim, pts_pos] = g.face_centers[:g.dim, face]
            f_loc = f_loc[f_loc != face]
            pts_pos += 1

            # check also if the face is a bounday face or it has
            # a node at the boundary
            nodes = self._slice_out(face, g.face_nodes)
            isin = np.isin(nodes, n_loc)

            # check if it is a boundary face, in case we need to consider
            # a node
            if b_faces[face] and np.any(isin):
                # identify the node
                mask = np.where(isin)[0][0]
                node = nodes[mask]
                n_loc = n_loc[n_loc != node]

                # save the node in the list of points
                pts[:g.dim, pts_pos] = g.nodes[:g.dim, node]

                # find the next connected faces at the boundary
                faces = self._slice_out(node, node_faces)
                faces = faces[b_faces[faces]]
                faces = faces[faces != face]

                # save the fact that it is a boundary corner
                bc_corner[:, pts_pos] = [1, 1, node]

                bc_tag = bnd.is_dir[face] * "dir" + \
                         bnd.is_neu[face] * "neu" + \
                         bnd.is_rob[face] * "rob"
                bc_type[:, pts_pos] = [bc_tag, True]

            else:
                # identify among the cells related to the current face
                # the single one that is also in the patch
                # select it and remove from the list of patch cells
                cells = self._slice_out(face, face_cells)
                mask = np.where(np.isin(cells, c_loc))[0][0]
                cell = cells[mask]
                c_loc = c_loc[c_loc != cell]

                # save the cell center in the list of points
                pts[:g.dim, pts_pos] = g.cell_centers[:g.dim, cell]

                # identify among the faces related to the current cell
                # the single one that is also in the patch
                # select it and remove from the list of patch faces
                faces = self._slice_out(cell, g.cell_faces)

                # save the fact that it is a boundary corner
                bc_corner[:, pts_pos] = [1, 0, cell]

            pts_pos += 1

        return pts, bc_type, bc_corner

    def _patch_fracs(self, pts_patch, fracs):
        """
        Intersect the fractures to get only the one related to the current patch
        """
        # intersect the local fractures with the patch
        pts, edges = pp.cg.intersect_polygon_lines(pts_patch, \
                                                   fracs["points"], fracs["edges"])
        return {"points": pts, "edges": edges}

    def _patch_gb(self, fracs, patch, subdom, mesh_args):
        """
        Construct the grid bucket for the current patch
        """
        return pp.fracs.meshing.simplex_grid(fracs, domain=patch, \
                                             subdomains=subdom, **mesh_args)

    def _patch_set_data(self, gb, data):
        # the data are user specific, in principle only the permeability and aperture
        # need to be inserted by the user and not a source term.
        for g, d in gb:
            d.update(data["node_data"](g, d, gb))

        for e, d in gb.edges():
            d.update(data["edge_data"](e, d, gb))

    def _patch_set_bc(self, gb, pts, pt, bc_type, tol):

        bd = np.array([[pt, (pt+1)%pts.shape[1], (pt+2)%pts.shape[1]],
                       [pt, (pt-1)%pts.shape[1], (pt-2)%pts.shape[1]]])

        dist = lambda i, j: np.linalg.norm(pts[:, bd[j, i]] - pts[:, bd[j, i+1]])
        length = np.array([[dist(i, j) for i in np.arange(2)] for j in np.arange(2)])
        sum_length = np.sum(length, axis=1)

        for g, d in gb:
            param = d["param"]

            # Boundaries
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if b_faces.size != 0:

                labels = ["dir"] * b_faces.size
                bc = pp.BoundaryCondition(g, b_faces, labels)
                param.set_bc("flow", bc)

                b_face_centers = g.face_centers[:, b_faces]
                bc_val = np.zeros(g.num_faces)
                for j in np.arange(2):
                    x_0 = np.array([0, length[j, 0]])
                    x_1 = np.array([length[j, 0], sum_length[j]])
                    y_0 = np.array([1, length[j, 0]/sum_length[j]])
                    y_1 = np.array([length[j, 0]/sum_length[j], 0])

                    for i in np.arange(2):
                        start = pts[:, bd[j, i]]
                        end = pts[:, bd[j, i+1]]

                        # detect all the points aligned with the segment
                        dist, _ = pp.cg.dist_points_segments(b_face_centers[:2], start, end)
                        mask = np.where(np.logical_and(dist < tol, dist >=-tol))[0]

                        # compute the distance
                        delta = np.tile(start, (mask.size, 1)).T - b_face_centers[:2, mask]

                        # define the boundary conditions
                        val = (y_1[i] - y_0[i])/(x_1[i] - x_0[i])
                        bc_val[b_faces[mask]] = np.linalg.norm(delta, axis=0)*val + y_0[i]

                param.set_bc_val("flow", bc_val)
            else:
                bc = pp.BoundaryCondition(g, np.empty(0), np.empty(0))
                param.set_bc("flow", bc)

            d["param"] = param

    def _slice_out(self, f, m):
        return m.indices[m.indptr[f]: m.indptr[f+1]]

