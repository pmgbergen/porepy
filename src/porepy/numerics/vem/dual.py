# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""

import warnings
import numpy as np
import scipy.sparse as sps

from porepy.params import tensor

from porepy.numerics.mixed_dim.solver import Solver, SolverMixDim
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.numerics.mixed_dim.abstract_coupling import AbstractCoupling

from porepy.utils import comp_geom as cg

#------------------------------------------------------------------------------#

class DualVEMMixDim(SolverMixDim):

    def __init__(self, physics='flow'):
        self.physics = physics

        self.discr = DualVEM(self.physics)
        self.coupling_conditions = DualCoupling(self.discr)

        self.solver = Coupler(self.discr, self.coupling_conditions)

    def extract_u(self, gb, up, u):
        for g, d in gb:
            d[u] = self.discr.extract_u(g, d[up])

    def extract_p(self, gb, up, p):
        for g, d in gb:
            d[p] = self.discr.extract_p(g, d[up])

    def project_u(self, gb, u, P0u):
        for g, d in gb:
            d[P0u] = self.discr.project_u(g, d[u], d)

    def check_conservation(self, gb, u, conservation):
        """
        Assert if the local conservation of mass is preserved for the grid
        bucket.
        Parameters
        ----------
        gb: grid bucket, or a subclass.
        u : string name of the velocity in the data associated to gb.
        conservation: string name for the conservation of mass.
        """
        for g, d in gb:
            d[conservation] = self.discr.check_conservation(g, d[u])

        # add to the lower dimensional grids the contribution from the higher
        # dimensional grids
        for e, data in gb.edges_props():
            g_l, g_h = gb.sorted_nodes_of_edge(e)

            cells_l, faces_h, _ = sps.find(data['face_cells'])
            faces, cells_h, sign = sps.find(g_h.cell_faces)
            ind = np.unique(faces, return_index=True)[1]
            sign = sign[ind][faces_h]

            conservation_l = gb.node_prop(g_l, conservation)
            u_h = sign*gb.node_prop(g_h, u)[faces_h]

            for c_l, u_f in zip(cells_l, u_h):
                conservation_l[c_l] -= u_f

        for g, d in gb:
            print(np.amax(np.abs(d[conservation])))

#------------------------------------------------------------------------------#

class DualVEMDFN(SolverMixDim):

    def __init__(self, dim_max, physics='flow'):
        # NOTE: There is no flow along the intersections of the fractures.

        self.physics = physics
        self.dim_max = dim_max
        self.discr = DualVEM(self.physics)

        self.coupling_conditions = DualCouplingDFN(self.__ndof__)

        kwargs = {"discr_ndof": self.__ndof__, "discr_fct": self.__matrix_rhs__}
        self.solver = Coupler(coupling = self.coupling_conditions, **kwargs)

    def extract_u(self, gb, up, u):
        for g, d in gb:
            if g.dim == self.dim_max:
                d[u] = self.discr.extract_u(g, d[up])
            else:
                d[u] = np.zeros(g.num_faces)

    def extract_p(self, gb, up, p):
        for g, d in gb:
            if g.dim == self.dim_max:
                d[p] = self.discr.extract_p(g, d[up])
            else:
                d[p] = d[up]

    def project_u(self, gb, u, P0u):
        for g, d in gb:
            if g.dim == self.dim_max:
                d[P0u] = self.discr.project_u(g, d[u], d)
            else:
                d[P0u] = np.zeros((3, g.num_cells))

    def __ndof__(self, g):
        # The highest dimensional problem has the standard number of dof
        # associated with the solver. For the lower dimensional problems, the
        # number of dof is the number of cells.
        if g.dim == self.dim_max:
            return self.discr.ndof(g)
        else:
            return g.num_cells

    def __matrix_rhs__(self, g, data):
        # The highest dimensional problem compute the matrix and rhs, the lower
        # dimensional problem and empty matrix. For the latter, the size of the
        # matrix is the number of cells.
        if g.dim == self.dim_max:
            return self.discr.matrix_rhs(g, data)
        else:
            ndof = self.__ndof__(g)
            return sps.csr_matrix((ndof, ndof)), np.zeros(ndof)

#------------------------------------------------------------------------------#

class DualVEM(Solver):

#------------------------------------------------------------------------------#

    def __init__(self, physics='flow'):
        self.physics = physics

#------------------------------------------------------------------------------#

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of faces (velocity dofs) plus the number of cells
        (pressure dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells + g.num_faces

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        """
        Return the matrix and righ-hand side for a discretization of a second
        order elliptic equation using dual virtual element method.
        The name of data in the input dictionary (data) are:
        perm : second_order_tensor
            Permeability defined cell-wise.
        source : array (self.g.num_cells)
            Scalar source term defined cell-wise. If not given a zero source
            term is assumed and a warning arised.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Return
        ------
        matrix: sparse csr (g.num_faces+g_num_cells, g.num_faces+g_num_cells)
            Saddle point matrix obtained from the discretization.
        rhs: array (g.num_faces+g_num_cells)
            Right-hand side which contains the boundary conditions and the scalar
            source term.

        Examples
        --------
        b_faces_neu = ... # id of the Neumann faces
        b_faces_dir = ... # id of the Dirichlet faces
        bnd = bc.BoundaryCondition(g, np.hstack((b_faces_dir, b_faces_neu)),
                                ['dir']*b_faces_dir.size + ['neu']*b_faces_neu.size)
        bnd_val = {'dir': fun_dir(g.face_centers[:, b_faces_dir]),
                   'neu': fun_neu(f.face_centers[:, b_faces_neu])}

        data = {'perm': perm, 'source': f, 'bc': bnd, 'bc_val': bnd_val}

        D, rhs = dual.matrix_rhs(g, data)
        up = sps.linalg.spsolve(D, rhs)
        u = dual.extract_u(g, up)
        p = dual.extract_p(g, up)
        P0u = dual.project_u(g, u, perm)

        """
        M, bc_weight = self.matrix(g, data, bc_weight=True)
        return M, self.rhs(g, data, bc_weight)

#------------------------------------------------------------------------------#

    def matrix(self, g, data, bc_weight=False):
        """
        Return the matrix for a discretization of a second order elliptic equation
        using dual virtual element method. See self.matrix_rhs for a detaild
        description.

        Additional parameter:
        --------------------
        bc_weight: to compute the infinity norm of the matrix and use it as a
            weight to impose the boundary conditions. Default True.

        Additional return:
        weight: if bc_weight is True return the weight computed.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            M = sps.dia_matrix(([1, 0], 0), (self.ndof(g), self.ndof(g)))
            if bc_weight:
                return M, 1
            return M

        # Retrieve the permeability, boundary conditions, and aperture
        # The aperture is needed in the hybrid-dimensional case, otherwise is
        # assumed unitary
        param = data['param']
        k = param.get_tensor(self)
        bc = param.get_bc(self)
        a = param.get_aperture()

        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        c_centers, f_normals, f_centers, _, _, _ = cg.map_grid(g)

        # In the virtual cell approach the cell diameters should involve the
        # apertures, however to keep consistency with the hybrid-dimensional
        # approach and with the related hypotheses we avoid.
        diams = g.cell_diameters()
        # Weight for the stabilization term
        weight = np.power(diams, 2-g.dim)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.sum(np.square(g.cell_faces.indptr[1:]-\
                                g.cell_faces.indptr[:-1]))
        I = np.empty(size, dtype=np.int)
        J = np.empty(size, dtype=np.int)
        dataIJ = np.empty(size)
        idx = 0

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its faces
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
            faces_loc = faces[loc]

            # Compute the H_div-mass local matrix
            A = self.massHdiv(a[c]*k.perm[0:g.dim, 0:g.dim, c], c_centers[:, c],
                              g.cell_volumes[c], f_centers[:, faces_loc],
                              f_normals[:, faces_loc], sign[loc],
                              diams[c], weight[c])[0]

            # Save values for Hdiv-mass local matrix in the global structure
            cols = np.tile(faces_loc, (faces_loc.size, 1))
            loc_idx = slice(idx, idx+cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        mass = sps.coo_matrix((dataIJ, (I, J)))
        div = -g.cell_faces.T
        M = sps.bmat([[mass, div.T],
                      [ div,  None]], format='csr')

        norm = sps.linalg.norm(mass, np.inf) if bc_weight else 1

        # assign the Neumann boundary conditions
        if bc and np.any(bc.is_neu):
            is_neu = np.hstack((bc.is_neu,
                                np.zeros(g.num_cells, dtype=np.bool)))
            M[is_neu, :] *= 0
            M[is_neu, is_neu] = norm

        if bc_weight:
            return M, norm
        return M

#------------------------------------------------------------------------------#

    def rhs(self, g, data, bc_weight=1):
        """
        Return the righ-hand side for a discretization of a second order elliptic
        equation using dual virtual element method. See self.matrix_rhs for a detaild
        description.

        Additional parameter:
        --------------------
        bc_weight: to use the infinity norm of the matrix to impose the
            boundary conditions. Default 1.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        param = data['param']
        f = param.get_source(self)

        if g.dim == 0:
            return np.hstack(([0], f))

        bc = param.get_bc(self)
        bc_val = param.get_bc_val(self)

        assert not bool(bc is None) != bool(bc_val is None)

        rhs = np.zeros(self.ndof(g))
        is_p = np.hstack((np.zeros(g.num_faces, dtype=np.bool),
                          np.ones(g.num_cells, dtype=np.bool)))

        rhs[is_p] = -f
        if bc is None:
            return rhs

        if np.any(bc.is_dir):
            is_dir = np.where(bc.is_dir)[0]
            faces, _, sign = sps.find(g.cell_faces)
            sign = sign[np.unique(faces, return_index=True)[1]]
            rhs[is_dir] += -sign[is_dir] * bc_val[is_dir]

        if np.any(bc.is_neu):
            is_neu = np.where(bc.is_neu)[0]
            rhs[is_neu] = bc_weight * bc_val[is_neu]

        return rhs

#------------------------------------------------------------------------------#

    def extract_u(self, g, up):
        """  Extract the velocity from a dual virtual element solution.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        up : array (g.num_faces+g.num_cells)
            Solution, stored as [velocity,pressure]

        Return
        ------
        u : array (g.num_faces)
            Velocity at each face.

        """
        # pylint: disable=invalid-name
        return up[:g.num_faces]

#------------------------------------------------------------------------------#

    def extract_p(self, g, up):
        """  Extract the pressure from a dual virtual element solution.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        up : array (g.num_faces+g.num_cells)
            Solution, stored as [velocity,pressure]

        Return
        ------
        p : array (g.num_cells)
            Pressure at each cell.

        """
        # pylint: disable=invalid-name
        return up[g.num_faces:]

#------------------------------------------------------------------------------#

    def project_u(self, g, u, data):
        """  Project the velocity computed with a dual vem solver to obtain a
        piecewise constant vector field, one triplet for each cell.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        u : array (g.num_faces) Velocity at each face.

        Return
        ------
        P0u : ndarray (3, g.num_faces) Velocity at each cell.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        if g.dim == 0:
            return np.zeros(3).reshape((3, 1))

        # The velocity field already has permeability effects incorporated,
        # thus we assign a unit permeability to be passed to self.massHdiv
        k = tensor.SecondOrder(g.dim, kxx=np.ones(g.num_cells))
        param = data['param']
        a = param.get_aperture()

        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        c_centers, f_normals, f_centers, R, dim, _ = cg.map_grid(g)

        # In the virtual cell approach the cell diameters should involve the
        # apertures, however to keep consistency with the hybrid-dimensional
        # approach and with the related hypotheses we avoid.
        diams = g.cell_diameters()

        P0u = np.zeros((3, g.num_cells))

        for c in np.arange(g.num_cells):
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
            faces_loc = faces[loc]

            Pi_s = self.massHdiv(a[c]*k.perm[0:g.dim, 0:g.dim, c], c_centers[:, c],
                                 g.cell_volumes[c], f_centers[:, faces_loc],
                                 f_normals[:, faces_loc], sign[loc],
                                 diams[c])[1]

            # extract the velocity for the current cell
            P0u[dim, c] = np.dot(Pi_s, u[faces_loc]) / diams[c] * a[c]
            P0u[:, c] = np.dot(R.T, P0u[:, c])

        return P0u

#------------------------------------------------------------------------------#

    def check_conservation(self, g, u):
        """
        Return the local conservation of mass in the cells.
        Parameters
        ----------
        g: grid, or a subclass.
        u : array (g.num_faces) velocity at each face.
        """
        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        conservation = np.empty(g.num_cells)
        for c in np.arange(g.num_cells):
            # For the current cell retrieve its faces
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
            conservation[c] = np.sum(u[faces[loc]]*sign[loc])

        return conservation

#------------------------------------------------------------------------------#

    def massHdiv(self, K, c_center, c_volume, f_centers, normals, sign, diam,
                 weight=0):
        """ Compute the local mass Hdiv matrix using the mixed vem approach.

        Parameters
        ----------
        K : ndarray (g.dim, g.dim)
            Permeability of the cell.
        c_center : array (g.dim)
            Cell center.
        c_volume : scalar
            Cell volume.
        f_centers : ndarray (g.dim, num_faces_of_cell)
            Center of the cell faces.
        normals : ndarray (g.dim, num_faces_of_cell)
            Normal of the cell faces weighted by the face areas.
        sign : array (num_faces_of_cell)
            +1 or -1 if the normal is inward or outward to the cell.
        diam : scalar
            Diameter of the cell.
        weight : scalar
            weight for the stabilization term. Optional, default = 0.

        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass Hdiv matrix.
        """
        # Allow short variable names in this function
        # pylint: disable=invalid-name

        dim = K.shape[0]
        mono = np.array([lambda pt, i=i: (pt[i] - c_center[i])/diam \
                                                       for i in np.arange(dim)])
        grad = np.eye(dim)/diam

        # local matrix D
        D = np.array([np.dot(normals.T, np.dot(K, g)) for g in grad]).T

        # local matrix G
        G = np.dot(grad, np.dot(K, grad.T))*c_volume

        # local matrix F
        F = np.array([s*m(f) for m in mono \
                      for s, f in zip(sign, f_centers.T)]).reshape((dim, -1))

        assert np.allclose(G, np.dot(F, D)), "G "+str(G)+" F*D "+str(np.dot(F,D))

        # local matrix Pi_s
        Pi_s = np.linalg.solve(G, F)
        I_Pi = np.eye(f_centers.shape[1]) - np.dot(D, Pi_s)

        # local Hdiv-mass matrix
        w = weight * np.linalg.norm(np.linalg.inv(K), np.inf)
        A = np.dot(Pi_s.T, np.dot(G, Pi_s)) + w * np.dot(I_Pi.T, I_Pi)

        return A, Pi_s

#------------------------------------------------------------------------------#

class DualCoupling(AbstractCoupling):

#------------------------------------------------------------------------------#

    def __init__(self, discr):
        self.discr_ndof = discr.ndof

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Construct the matrix (and right-hand side) for the coupling conditions.
        Note: the right-hand side is not implemented now.

        Parameters:
            g_h: grid of higher dimension
            g_l: grid of lower dimension
            data_h: dictionary which stores the data for the higher dimensional
                grid
            data_l: dictionary which stores the data for the lower dimensional
                grid
            data: dictionary which stores the data for the edges of the grid
                bucket

        Returns:
            cc: block matrix which store the contribution of the coupling
                condition. See the abstract coupling class for a more detailed
                description.
        """
        # pylint: disable=invalid-name

        # Normal permeability and aperture of the intersection
        k = 2*data_edge['kn'] # TODO: need to be handled in a different way
        aperture_h = data_h['param'].get_aperture()

        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions
        dof, cc = self.create_block_matrix(g_h, g_l)

        # Recover the information for the grid-grid mapping
        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])
        faces, cells_h, sign = sps.find(g_h.cell_faces)
        ind = np.unique(faces, return_index=True)[1]
        sign = sign[ind][faces_h]
        cells_h = cells_h[ind][faces_h]

        # Compute the off-diagonal terms
        dataIJ, I, J = sign, g_l.num_faces+cells_l, faces_h
        cc[1, 0] = sps.csr_matrix((dataIJ, (I, J)), (dof[1], dof[0]))
        cc[0, 1] = cc[1, 0].T

        # Compute the diagonal terms
        dataIJ = 1./(g_h.face_areas[faces_h] * aperture_h[cells_h] * k[cells_l])
        I, J = faces_h, faces_h
        cc[0, 0] = sps.csr_matrix((dataIJ, (I, J)), (dof[0], dof[0]))

        return cc

#------------------------------------------------------------------------------#

class DualCouplingDFN(AbstractCoupling):

#------------------------------------------------------------------------------#

    def __init__(self, discr_ndof):

        self.discr_ndof = discr_ndof

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Construct the matrix (and right-hand side) for the coupling conditions
        of a DFN. We use the Lagrange multiplier to impose continuity of the
        normal fluxes at the intersections.
        Note: the right-hand side is not implemented now.

        Parameters:
            g_h: grid of higher dimension
            g_l: grid of lower dimension
            data_h: Not used but kept for consistency
            data_l: Not used but kept for consistency
            data: Not used but kept for consistency

        Returns:
            cc: block matrix which store the contribution of the coupling
                condition. See the abstract coupling class for a more detailed
                description.
        """
        # pylint: disable=invalid-name

        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions
        dof, cc = self.create_block_matrix(g_h, g_l)

        # Recover the information for the grid-grid mapping
        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])
        faces, cells_h, sign = sps.find(g_h.cell_faces)
        ind = np.unique(faces, return_index=True)[1]
        sign = sign[ind][faces_h]

        # Compute the off-diagonal terms
        dataIJ, I, J = sign, cells_l, faces_h
        cc[1, 0] = sps.csr_matrix((dataIJ, (I, J)), (dof[1], dof[0]))
        cc[0, 1] = cc[1, 0].T

        return cc

#------------------------------------------------------------------------------#
