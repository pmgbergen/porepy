# -*- coding: utf-8 -*-
from typing import Callable, Dict

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.forward_mode import Ad_array
from porepy.numerics.ad.functions import heaviside
from porepy.numerics.ad.local_forward_mode import Local_Ad_array
from porepy.numerics.ad.operators import ApplicableOperator


class GeneralTpfaAd(pp.FVElliptic):
    """Discretize elliptic equations by a two-point flux approximation with
    arbitrarily weighted mobility.

    Attributes:

    keyword : str
        Which keyword is the solver intended flow. Will determine which data
        will be accessed (e.g. flow specific, or conductivity / heat-related).
        See Data class for more details. Defaults to flow.

    """

    def __init__(self, keyword):
        super().__init__(keyword)

    def discretize(self, g, data):
        """This is an excerpt discretize of tpfa.py but using merely constant
        transmissibility equal to 1.
        In addition, several grid connectivity related fields are stored for later use."""

        # Store for later evaluation of the flux - expect to run discretize before
        # evaluating flux
        self.data = data

        # Get the dictionaries for storage of data and discretization matrices
        parameter_dictionary: Dict = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: Dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]

        # Ambient dimension of the grid
        vector_source_dim: int = parameter_dictionary.get("ambient_dimension", g.dim)

        if g.dim == 0:
            # Short cut for 0d grids
            matrix_dictionary[self.flux_matrix_key] = sps.csr_matrix((0, g.num_cells))
            matrix_dictionary[self.bound_flux_matrix_key] = sps.csr_matrix((0, 0))
            matrix_dictionary[self.bound_pressure_cell_matrix_key] = sps.csr_matrix(
                (0, g.num_cells)
            )
            matrix_dictionary[self.bound_pressure_face_matrix_key] = sps.csr_matrix(
                (0, 0)
            )
            matrix_dictionary[self.vector_source_matrix_key] = sps.csr_matrix(
                (0, g.num_cells * max(vector_source_dim, 1))
            )
            matrix_dictionary[
                self.bound_pressure_vector_source_matrix_key
            ] = sps.csr_matrix((0, g.num_cells * max(vector_source_dim, 1)))
            return None

        # Extract parameters
        bnd: pp.BoundaryCondition() = parameter_dictionary["bc"]

        fi_g, ci_g, sgn_g = sps.find(g.cell_faces)

        # fi_g and ci_g now defines the geometric (grid) mapping from subfaces to cells.
        # The cell with index ci_g[i] has the face with index fi_g[i].
        # In addition to the geometric mappings, we need to add connections between
        # cells and faces over the periodic boundary.
        # The periodic boundary is defined by a mapping from left faces to right
        # faces:
        if hasattr(g, "periodic_face_map"):
            fi_left = g.periodic_face_map[0]
            fi_right = g.periodic_face_map[1]
        else:
            fi_left = np.array([], dtype=int)
            fi_right = np.array([], dtype=int)
        # We find the left(right)_face -> left(right)_cell mapping
        left_sfi, ci_left, left_sgn = sps.find(g.cell_faces[fi_left])
        right_sfi, ci_right, right_sgn = sps.find(g.cell_faces[fi_right])

        # Sort subface indices to not loose left to right periodic mapping
        # I.e., fi_left[i] maps to fi_right[i]
        I_left = np.argsort(left_sfi)
        I_right = np.argsort(right_sfi)
        if not (
            np.array_equal(left_sfi[I_left], np.arange(fi_left.size))
            and np.array_equal(right_sfi[I_right], np.arange(fi_right.size))
        ):
            raise RuntimeError("Could not find correct periodic boundary mapping")
        ci_left = ci_left[I_left]
        ci_right = ci_right[I_right]
        # Now, ci_left gives the cell indices of the left cells, and ci_right gives
        # the indices of the right cells. Further, fi_left gives the face indices of the
        # left faces that is periodic with the faces with indices fi_right. This means
        # that ci_left[i] is connected to ci_right[i] over the face fi_left (face of
        # ci_left[i]) and fi_right[i] (face of ci_right[i]).
        #
        # Next, we add connection between the left cells and right faces (and vice versa).
        # The flux over the periodic boundary face is defined equivalently to the
        # flux over an internal face: flux_left = T_left * (p_left - p_right).
        # The term T_left * p_left is already included in fi_g and ci_g, but we need
        # to add the second term T_left * (-p_right). Equivalently for flux_right.
        # f_mat and c_mat defines the indices of these entries in the flux matrix.
        fi_periodic = np.hstack((fi_g, fi_left, fi_right))
        ci_periodic = np.hstack((ci_g, ci_right, ci_left))
        sgn_periodic = np.hstack((sgn_g, -left_sgn, -right_sgn))

        # When calculating the subface transmissibilities, left cells should be mapped
        # to left faces, while right cells should be mapped to right faces.
        fi = np.hstack((fi_g, fi_right, fi_left))
        ci = np.hstack((ci_g, ci_right, ci_left))
        sgn = np.hstack((sgn_g, right_sgn, left_sgn))

        # Distance from face center to cell center
        fc_cc = g.face_centers[::, fi] - g.cell_centers[::, ci]

        # Create a linear operator corresponding to bincount -- also for later use.
        f_periodic_max = max(fi_periodic)
        c_periodic_max = len(fi_periodic)
        col = np.arange(c_periodic_max)
        row = fi_periodic[col]
        data = np.ones_like(row)
        bincount_fi_periodic = sps.csr_matrix(
            (data, (row, col)), shape=(f_periodic_max + 1, c_periodic_max)
        )

        # t does not encode the transsmissibility but will be only used for defininig
        # the cell to face map, and the scaling by the inverse of distances of cell centers.
        dist_face_cell = np.power(np.power(fc_cc, 2).sum(axis=0), 0.5)
        area_face_cell = np.power(np.power(g.face_normals[:, fi], 2).sum(axis=0), 0.5)
        t = (bincount_fi_periodic * (dist_face_cell / area_face_cell)) ** (-1)

        # Save values for use in recovery of boundary face pressures
        t_full = t.copy()

        # For primal-like discretizations like the TPFA, internal boundaries
        # are handled by assigning Neumann conditions.
        is_dir = np.logical_and(bnd.is_dir, np.logical_not(bnd.is_internal))
        is_neu = np.logical_or(bnd.is_neu, bnd.is_internal)

        # Move Neumann faces to Neumann transmissibility
        bndr_ind = g.get_all_boundary_faces()
        t_b = np.zeros(g.num_faces)
        t_b[is_dir] = -t[is_dir]
        t_b[is_neu] = 1
        t_b = t_b[bndr_ind]
        t[is_neu] = 0

        # Create flux matrix
        flux = sps.coo_matrix(
            (t[fi_periodic] * sgn_periodic, (fi_periodic, ci_periodic))
        ).tocsr()

        # Create boundary flux matrix
        bndr_sgn = (g.cell_faces[bndr_ind, :]).data
        sort_id = np.argsort(g.cell_faces[bndr_ind, :].indices)
        bndr_sgn = bndr_sgn[sort_id]
        bound_flux = sps.coo_matrix(
            (t_b * bndr_sgn, (bndr_ind, bndr_ind)), (g.num_faces, g.num_faces)
        ).tocsr()

        # Store the matrix in the right dictionary:
        matrix_dictionary[self.flux_matrix_key] = flux
        matrix_dictionary[self.bound_flux_matrix_key] = bound_flux

        # Next, construct operator to reconstruct pressure on boundaries
        # Fields for data storage
        v_cell = np.zeros(fi.size)
        v_face = np.zeros(g.num_faces)
        # On Dirichlet faces, simply recover boundary condition
        v_face[bnd.is_dir] = 1
        # On Neumann faces, the, use half-transmissibilities
        v_face[bnd.is_neu] = -1 / t_full[bnd.is_neu]
        v_cell[bnd.is_neu[fi]] = 1

        bound_pressure_cell = sps.coo_matrix(
            (v_cell, (fi, ci)), (g.num_faces, g.num_cells)
        ).tocsr()
        bound_pressure_face = sps.dia_matrix(
            (v_face, 0), (g.num_faces, g.num_faces)
        ).tocsr()
        matrix_dictionary[self.bound_pressure_cell_matrix_key] = bound_pressure_cell
        matrix_dictionary[self.bound_pressure_face_matrix_key] = bound_pressure_face

        # Discretization of vector source
        # e.g. gravity in Darcy's law
        # Use harmonic average of cell transmissibilities

        # The discretization involves the transmissibilities, multiplied with the
        # distance between cell and face centers, and with the sgn adjustment (or else)
        # the vector source will point in the wrong direction in certain cases.
        # See Starnoni et al 2020, WRR for details.
        vals = (t[fi_periodic] * fc_cc * sgn_periodic)[:vector_source_dim].ravel("f")

        # Rows and cols are given by fi / ci, expanded to account for the vector source
        # having multiple dimensions.
        rows = np.tile(fi_periodic, (vector_source_dim, 1)).ravel("f")
        cols = pp.fvutils.expand_indices_nd(ci_periodic, vector_source_dim)

        vector_source = sps.coo_matrix((vals, (rows, cols))).tocsr()

        matrix_dictionary[self.vector_source_matrix_key] = vector_source

        # Gravity contribution to pressure reconstruction
        # The pressure difference is computed as the dot product between the
        # vector source and the distance vector from cell to face centers.
        vals = np.zeros((vector_source_dim, fi.size))
        vals[:, bnd.is_neu[fi]] = fc_cc[:vector_source_dim, bnd.is_neu[fi]]
        bound_pressure_vector_source = sps.coo_matrix(
            (vals.ravel("f"), (rows, cols))
        ).tocsr()
        matrix_dictionary[
            self.bound_pressure_vector_source_matrix_key
        ] = bound_pressure_vector_source

        ################################################################
        # This is extra - needed for UpwindAd and HarmAvgAd
        ################################################################

        # Store some values for the later computation of fluxes
        self.fi_periodic = fi_periodic
        self.ci_periodic = ci_periodic
        self.sgn_periodic = sgn_periodic
        self.fi = fi
        self.ci = ci
        self.sgn = sgn
        self.fc_cc = fc_cc
        self.bincount_fi_periodic = bincount_fi_periodic

        # Needed for internal and external transmissibility computations
        self.is_dir = is_dir
        self.is_neu = is_neu

    def flux(self, face_transmissibility, potential, bc_data):
        """Return Ad object which implements the flux."""

        tpfaFlux = _TpfaFluxAd(self)
        return tpfaFlux(face_transmissibility, potential, bc_data)


class _TpfaFluxAd(ApplicableOperator):
    def __init__(self, tpfa):

        self._tpfa = tpfa
        self._set_tree()

    def __repr__(self) -> str:
        return "AD version of a TPFA flux."

    # TODO make bc = None default, and face_transmissibility = None (essentially idnetity)
    def apply(self, face_transmissibility, potential, bc):
        if isinstance(potential, Ad_array):
            return self._ad_apply(face_transmissibility, potential, bc)
        else:
            return self._non_ad_apply(face_transmissibility, potential, bc)

    # TODO merge methods _ad_apply and _non_ad_apply?
    def _ad_apply(self, face_transmissibility, potential, bc):
        tpfa = self._tpfa

        # Inner contribution
        matrix_dictionary = tpfa.data[pp.DISCRETIZATION_MATRICES][tpfa.keyword]
        flux_matrix = matrix_dictionary[tpfa.flux_matrix_key]
        bound_flux_matrix = matrix_dictionary[tpfa.bound_flux_matrix_key]

        # Compute flux and jacobian including bound flux data.

        # Determine value of face transmissibilities.
        if isinstance(face_transmissibility, Ad_array):
            face_transmissibility_val = face_transmissibility.val
        elif isinstance(face_transmissibility, np.ndarray):
            face_transmissibility_val = face_transmissibility
        else:
            raise RuntimeError("Type not implemented.")

        # How the flux and its jacobian are computed:
        # flux.val = diag(face_transmissibility.val) * flux_matrix * potential.val
        #          = diag(flux_matrix * potential.val) * face_transmissibility.val; hence:
        # Start assuming face_transmissibility is constant
        flux_val = (
            sps.diags(face_transmissibility_val).tocsc() * flux_matrix * potential.val
        )
        flux_jac = (
            sps.diags(face_transmissibility_val).tocsc() * flux_matrix * potential.jac
        )

        # Boundary contribution - require some care with neumann boundary conditions.
        # How the boundary flux and its jacobian is computed:
        # bc_value = diag(t_b) * bound_flux_matrix * bc = diag(bound_flux_matrix * bc) * t_b
        # For the jacobian use the latter formula.
        t_b_val = face_transmissibility_val

        # Enforce Neumann BCs by setting the transmissibility equal 1:
        is_neu = tpfa.is_neu
        t_b_val[is_neu] = 1

        # flux_val += sps.diags(t_b_val).tocsc() * bound_flux_matrix * bc # TODO rm?
        flux_val += t_b_val * (bound_flux_matrix * bc)

        # Now account in the Jacobian for nonlinear face_transmissibility
        if isinstance(face_transmissibility, Ad_array):
            flux_jac += (
                sps.diags(flux_matrix * potential.val) * face_transmissibility.jac
            )

            # Boundary contribution - require some care with neumann boundary conditions
            t_b_jac = face_transmissibility.jac

            # Enforce Neumann BCs by setting the transmissibility equal 1:
            neumann_rows = np.arange(len(t_b_val))[is_neu]
            pp.utils.sparse_mat.zero_rows(t_b_jac, neumann_rows)

            flux_jac += sps.diags(bound_flux_matrix * bc) * t_b_jac

        return Ad_array(flux_val, flux_jac)

    # TODO make bc = None default, and face_transmissibility = None (essentially idnetity)
    def _non_ad_apply(self, face_transmissibility, potential, bc):
        tpfa = self._tpfa

        # Inner contribution
        matrix_dictionary = tpfa.data[pp.DISCRETIZATION_MATRICES][tpfa.keyword]
        flux_matrix = matrix_dictionary[tpfa.flux_matrix_key]
        bound_flux_matrix = matrix_dictionary[tpfa.bound_flux_matrix_key]

        # Compute flux and jacobian including bound flux data.

        # Determine value of face transmissibilities.
        if isinstance(face_transmissibility, Ad_array):
            face_transmissibility_val = face_transmissibility.val
        elif isinstance(face_transmissibility, np.ndarray):
            face_transmissibility_val = face_transmissibility
        else:
            raise RuntimeError("Type not implemented.")

        # How the flux and its jacobian are computed:
        # flux.val = diag(face_transmissibility.val) * flux_matrix * potential.val
        #          = diag(flux_matrix * potential.val) * face_transmissibility.val; hence:
        # Start assuming face_transmissibility is constant
        flux_val = (
            sps.diags(face_transmissibility_val).tocsc() * flux_matrix * potential
        )

        # Boundary contribution - require some care with neumann boundary conditions.
        # How the boundary flux and its jacobian is computed:
        # bc_value = diag(t_b) * bound_flux_matrix * bc = diag(bound_flux_matrix * bc) * t_b
        # For the jacobian use the latter formula.
        t_b_val = face_transmissibility_val

        # Enforce Neumann BCs by setting the transmissibility equal 1:
        is_neu = tpfa.is_neu
        t_b_val[is_neu] = 1

        # flux_val += sps.diags(t_b_val).tocsc() * bound_flux_matrix * bc # TODO rm?
        flux_val += t_b_val * (bound_flux_matrix * bc)

        return flux_val


# TODO Actually only tpfa-unrelated info is retrieved from tpfa here.
class UpwindAd(ApplicableOperator):
    def __init__(self, g, tpfa, hs: Callable = heaviside):

        self._set_tree()
        self._g = g
        self._tpfa = tpfa
        self._heaviside = hs

        # Construct projection from cell-valued arrays to face-valued arrays with values to the
        # "left" and "right" of the face, here denoted by '0' and '1', respectively.
        cf_dense = g.cell_face_as_dense()
        cf_inner = [c >= 0 for c in cf_dense]

        row = [np.arange(g.num_faces)[cf_inner[i]] for i in range(0, 2)]
        col = [cf_dense[i][cf_inner[i]] for i in range(0, 2)]
        data = [np.ones_like(row[i]) for i in range(0, 2)]
        self._cf_inner = [
            sps.csr_matrix(
                (data[i], (row[i], col[i])),
                shape=(g.num_faces, g.num_cells),
                dtype=float,
            )
            for i in range(0, 2)
        ]

        # Store which 'left' and 'right' cells of all faces correspond to the Dirichlet
        # boundary.
        cf_boundary = np.logical_not(cf_inner)
        is_dir = tpfa.is_dir
        self._cf_is_dir = [np.logical_and(cf_boundary[i], is_dir) for i in range(0, 2)]

    def __repr__(self) -> str:
        return f"Upwind AD face operator"

    def apply(self, mobility_inner, direction_inner, mobility_bound, direction_bound):
        """Compute transmissibility via upwinding over faces. Use monotonicityexpr for
        deciding directionality.

        Idea: 'face value' = 'left cell value' * Heaviside('flux from left')
                           + 'right cell value' * Heaviside('flux from right').
        """

        # TODO only implemented for scalar relative permeabilities so far
        # TODO so far not for periodic bondary conditions.

        # Rename internal properties
        hs = self._heaviside
        cf_inner = self._cf_inner
        cf_is_dir = self._cf_is_dir

        # Determine direction-determining cell values to the left(0) and right(1) of each face.
        # Use Dirichlet boundary data where suitable.
        # Neglect Neumann boundaries since face transmissibilities at Neumann boundary data
        # anyhow does not play a role.
        # Determine the Jacobian manually - only in the interior of the domain.
        if isinstance(direction_inner, Ad_array):
            dir_f_val = [cf_inner[i] * direction_inner.val for i in range(0, 2)]
            for i in range(0, 2):
                dir_f_val[i][cf_is_dir[i]] = direction_bound[cf_is_dir[i]]
            dir_f_jac = [cf_inner[i] * direction_inner.jac for i in range(0, 2)]
            dir_f = [Ad_array(dir_f_val[i], dir_f_jac[i]) for i in range(0, 2)]
        else:
            dir_f = [cf_inner[i] * direction_inner for i in range(0, 2)]
            for i in range(0, 2):
                dir_f[i][cf_is_dir[i]] = direction_bound[cf_is_dir[i]]

        # Do the same for the mobility as for the direction-determining arrays.
        if isinstance(mobility_inner, Ad_array):
            mob_f_val = [cf_inner[i] * mobility_inner.val for i in range(0, 2)]
            for i in range(0, 2):
                mob_f_val[i][cf_is_dir[i]] = mobility_bound[cf_is_dir[i]]
            mob_f_jac = [cf_inner[i] * mobility_inner.jac for i in range(0, 2)]
            mob_f = [Ad_array(mob_f_val[i], mob_f_jac[i]) for i in range(0, 2)]

        else:
            mob_f = [cf_inner[i] * mobility_inner for i in range(0, 2)]
            for i in range(0, 2):
                mob_f[i][cf_is_dir[i]] = mobility_bound[cf_is_dir[i]]

        # Evaluate the Heaviside function of the "flux directions".
        hs_f_01 = hs(dir_f[0] - dir_f[1])
        hs_f_10 = hs(dir_f[1] - dir_f[0])

        # Determine the face mobility by utilizing the general idea (see above).
        face_mobility = mob_f[0] * hs_f_01 + mob_f[1] * hs_f_10

        return face_mobility


class FluxBasedUpwindAd(ApplicableOperator):
    def __init__(self, g, tpfa, hs: Callable = heaviside):

        self._set_tree()
        self._g = g
        self._tpfa = tpfa
        self._heaviside = hs

        # Construct projection from cell-valued arrays to face-valued arrays with values to the
        # "left" and "right" of the face, here denoted by '0' and '1', respectively.
        cf_dense = g.cell_face_as_dense()
        cf_inner = [c >= 0 for c in cf_dense]

        row = [np.arange(g.num_faces)[cf_inner[i]] for i in range(0, 2)]
        col = [cf_dense[i][cf_inner[i]] for i in range(0, 2)]
        data = [np.ones_like(row[i]) for i in range(0, 2)]
        self._cf_inner = [
            sps.csr_matrix(
                (data[i], (row[i], col[i])),
                shape=(g.num_faces, g.num_cells),
                dtype=float,
            )
            for i in range(0, 2)
        ]

        # Store which 'left' and 'right' cells of all faces correspond to the Dirichlet
        # boundary.
        cf_is_boundary = np.logical_not(cf_inner)
        self._cf_is_boundary = cf_is_boundary
        is_dir = tpfa.is_dir
        self._cf_is_dir = [
            np.logical_and(cf_is_boundary[i], is_dir) for i in range(0, 2)
        ]

    def __repr__(self) -> str:
        return f"Upwind AD face operator"

    def apply(self, mobility_inner, mobility_bound, face_flux):
        """Compute transmissibility via upwinding over faces. Use monotonicityexpr for
        deciding directionality.

        Idea: 'face value' = 'left cell value' * Heaviside('flux from left')
                           + 'right cell value' * Heaviside('flux from right').
        """

        # TODO only implemented for scalar relative permeabilities so far
        # TODO so far not for periodic bondary conditions.

        # Rename internal properties
        hs = self._heaviside
        cf_inner = self._cf_inner
        cf_is_boundary = self._cf_is_boundary

        # Determine direction-determining cell values to the left(0) and right(1) of each face.
        # Use Dirichlet boundary data where suitable.
        # Neglect Neumann boundaries since face transmissibilities at Neumann boundary data
        # anyhow does not play a role.
        # assert (face_flux, np.ndarray)  # TODO extend to Ad_arrays

        # Do the same for the mobility as for the direction-determining arrays.
        if isinstance(mobility_inner, Ad_array):
            raise RuntimeError("Not implemented.")
        else:
            mob_f = [cf_inner[i] * mobility_inner for i in range(0, 2)]
            for i in range(0, 2):
                mob_f[i][cf_is_boundary[i]] = mobility_bound[cf_is_boundary[i]]

        # Evaluate the Heaviside function of the "flux directions".
        hs_f_01 = hs(face_flux)
        hs_f_10 = hs(-face_flux)

        # Determine the face mobility by utilizing the general idea (see above).
        face_mobility = mob_f[0] * hs_f_01 + mob_f[1] * hs_f_10

        return face_mobility


class HarmAvgAd(ApplicableOperator):
    """ Object computing the harmonic average at faces given a cell-wise defined field."""

    def __init__(self, g, data, tpfa):
        # TODO iRegarding input parameter tpfa, actually only tpfa-unrelated info is
        # retrieved from tpfa here.

        self._set_tree()

        self._g = g
        self._data = data
        self._tpfa = tpfa

        # For later computation of face transmissibilities, create the linear operator
        # corresponding to bincount
        f_periodic_max = max(tpfa.fi_periodic)
        c_periodic_max = len(tpfa.fi_periodic)
        col = np.arange(c_periodic_max)
        row = tpfa.fi_periodic[col]
        data = np.ones_like(row)
        self.bincount_fi_periodic = sps.csr_matrix(
            (data, (row, col)), shape=(f_periodic_max + 1, c_periodic_max)
        )

    def __repr__(self) -> str:
        return "Harmonic average operator"

    def apply(self, cellwise_field):
        """ Compute transmissibility via harmonic averaging over faces."""

        if isinstance(cellwise_field, Local_Ad_array):
            return self._apply_ad(cellwise_field)
        elif isinstance(cellwise_field, np.ndarray):
            return self._apply_non_ad(cellwise_field)
        else:
            raise RuntimeError(
                "Harmonic average not implemented for this type of cellwise tensorfield."
            )

    def _apply_ad(self, cellwise_field) -> Ad_array:
        """ Compute transmissibility via harmonic averaging over faces."""

        # References to private variables
        data = self._data
        tpfa = self._tpfa

        if data.get("Aavatsmark_transmissibilities", False):
            raise RuntimeError(
                "AD version of Aavatsmark_transmissibilities not implemented."
            )

        # Get connectivity and grid based data
        ci = tpfa.ci
        ci_periodic = tpfa.ci_periodic
        fc_cc = tpfa.fc_cc
        dist_face_cell = np.power(np.power(fc_cc, 2).sum(axis=0), 0.5)

        # Consider two cases: scalar and tensor valued fields.

        # assert (cellwise_field.val, np.ndarray)
        # Case 1: Scalar valued fields.
        if len(cellwise_field.val.shape) == 1:
            t_cf_val = cellwise_field.val[ci]
            t_cf_jac = cellwise_field.jac[ci]
            t_cf_val /= dist_face_cell
            t_cf_jac /= dist_face_cell

        # Case 2: Tensor valued fields.
        elif len(cellwise_field.val.shape) == 3 and all(
            [cellwise_field.val.shape[i] == 3 for i in range(0, 2)]
        ):
            t_cf_tensor_val = cellwise_field.val[::, ::, ci]
            t_cf_tensor_jac = cellwise_field.jac[::, ::, ci]
            tn_cf_val = (t_cf_tensor_val * fc_cc).sum(axis=1)
            tn_cf_jac = (t_cf_tensor_jac * fc_cc).sum(axis=1)
            ntn_cf_val = (tn_cf_val * fc_cc).sum(axis=0)
            ntn_cf_jac = (tn_cf_jac * fc_cc).sum(axis=0)
            dist_face_cell_3 = np.power(dist_face_cell, 3)
            t_cf_val = np.divide(ntn_cf_val, dist_face_cell_3)
            t_cf_jac = np.divide(ntn_cf_jac, dist_face_cell_3)

        else:
            raise RuntimeError("Type of cell-wise field not supported.")

        # Continue with AD representation and utilize chain rule.
        t_face = Ad_array(t_cf_val, sps.diags(t_cf_jac).tocsc())

        # The final harmonic averaging using a linear operator representation of bincount.
        # TODO test!
        t_face = (self.bincount_fi_periodic * dist_face_cell) * (
            (self.bincount_fi_periodic * t_face ** (-1)) ** (-1)
        )

        # Project column space of t.jac onto the actual cell
        # TODO is there not a better way to create the projection matrix? By correct indexing?
        c = np.arange(len(ci_periodic))
        proj = sps.coo_matrix((np.ones_like(c), (c, ci_periodic))).tocsr()
        t_face.jac = t_face.jac * proj

        return t_face

    def _apply_non_ad(self, cellwise_field) -> np.ndarray:
        """ Compute transmissibility via harmonic averaging over faces."""

        # References to private variables
        data = self._data
        tpfa = self._tpfa
        if data.get("Aavatsmark_transmissibilities", False):
            raise RuntimeError(
                "AD version of Aavatsmark_transmissibilities not implemented."
            )

        # Get connectivity and grid based data
        ci = tpfa.ci
        fc_cc = tpfa.fc_cc
        dist_face_cell = np.power(np.power(fc_cc, 2).sum(axis=0), 0.5)

        # Consider two cases: scalar and tensor valued fields.

        # assert (cellwise_field, np.ndarray)
        # Case 1: Scalar valued fields.
        if len(cellwise_field.shape) == 1:
            t_cf_val = cellwise_field[ci]
            t_cf_val /= dist_face_cell

        # Case 2: Tensor valued fields.
        elif len(cellwise_field.shape) == 3 and all(
            [cellwise_field.shape[i] == 3 for i in range(0, 2)]
        ):
            t_cf_tensor_val = cellwise_field[::, ::, ci]
            tn_cf_val = (t_cf_tensor_val * fc_cc).sum(axis=1)
            ntn_cf_val = (tn_cf_val * fc_cc).sum(axis=0)
            dist_face_cell_3 = np.power(dist_face_cell, 3)
            t_cf_val = np.divide(ntn_cf_val, dist_face_cell_3)

        else:
            raise RuntimeError("Type of cell-wise field not supported.")

        # The final harmonic averaging using a linear operator representation of bincount.
        t_face = (self.bincount_fi_periodic * dist_face_cell) / (
            self.bincount_fi_periodic * t_cf_val ** (-1)
        )

        return t_face
