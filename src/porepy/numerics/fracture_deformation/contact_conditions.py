"""
For details on the conditions discretized herein, see

Berge et al., 2019: Finite volume discretization for poroelastic media with fractures
modeled by contact mechanics.

When solving contact problems, the sought fracture displacement (jumps) are defined
relative to an initial state. For transient problems, this initial state is the solution
at the previous time step. The state should be available in

    d[pp.STATE][self.mortar_displacement_variable],

and may usually be set to zero for stationary problems. The ColoumbContact
discretization operates on relative tangential jumps and absolute normal jumps.
See also contact_mechanics_interface_laws.py.

Signs of displacement jumps are reversed compared to Berge due to the PorePy definition
of the jump as [[ var ]] = var_k - var_j, which implies that positive normal jumps
correspond to fracture opening. Note that the fracture normal is in agreement with
Berge, i.e. it equals the outwards normal on the j side.

Option added to the Berge model:
Include a simple relationship between the gap and tangential displacements, i.e.

   g = g_0 + tan(dilation_angle) * || u_t ||,

with g_0 indicating initial gap distance. This only affects the normal relations when
fractures are in contact. The relation [u_n^{k+1}] = g of eqs. 30 and 31 becomes

   u_n^{k+1} - Dg^k \dot u_t^{k+1} = g^k - Dg \dot u_t^{k},

with Dg = dg/du_t. For the above g, we have Dg = tan(dilation_angle) * u_t / || u_t ||.
For the case u_t = 0, we extend the Jacobian to 0, i.e.
    dg/du_t(|| u_t || = 0) = 0.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)
module_sections = ["discretization", "numerics"]


class ColoumbContact:
    @pp.time_logger(sections=module_sections)
    def __init__(self, keyword: str, ambient_dimension: int, discr_h) -> None:
        self.keyword = keyword

        self.dim = ambient_dimension

        self.mortar_displacement_variable = "mortar_u"
        self.contact_variable = "contact_traction"

        self.traction_matrix_key = "traction_discretization"
        self.displacement_matrix_key = "displacement_discretization"
        self.rhs_matrix_key = "contact_rhs"

        self.discr_h = discr_h
        # Tolerance used to define numbers that effectively are zero.
        self.tol = 1e-10

    @pp.time_logger(sections=module_sections)
    def _key(self) -> str:
        return self.keyword + "_"

    @pp.time_logger(sections=module_sections)
    def _discretization_key(self) -> str:
        return self._key() + pp.DISCRETIZATION

    @pp.time_logger(sections=module_sections)
    def ndof(self, g) -> int:
        return g.num_cells * self.dim

    @pp.time_logger(sections=module_sections)
    def discretize(
        self, g_h: pp.Grid, g_l: pp.Grid, data_h: Dict, data_l: Dict, data_edge: Dict
    ) -> None:
        """Discretize the contact conditions using a semi-smooth Newton
        approach.

        The function relates the contact forces, represented on the
        lower-dimensional grid, to the jump in displacement between the two
        adjacent mortar grids. The function provides a (linearized)
        disrcetizaiton of the contact conditions, as described in Berge et al.

        The discertization is stated in the coordinate system defined by the
        projection operator associated with the surface. The contact forces
        should be interpreted as tangential and normal to this plane.

        Parameters in data_l:
            "friction_coefficient": float or np.ndarray (g_l.num_cells). A float
        is interpreted as a homogenous coefficient for all cells of the fracture.
            "c_num": float. Numerical parameter, defaults to 100. The sensitivity
        is currently unknown.

        Optional parameters: float or np.ndarray (g_l.num_cells), all default to 0:
            "initial_gap": The gap (minimum normal opening) in the undeformed state.
            "dilation_angle": Angle for dilation relation, see above.
            "cohesion": Threshold value for tangential traction.

        NOTES:
            Quantities stated in the global coordinate system (e.g.
        displacements on the adjacent mortar grids) must be projected to the
        local system, using the same projection operator, when paired with the
        produced discretization (that is, in the global assembly).

        Assumptions and other noteworthy aspects:  TODO: Rewrite this when the
        implementation is ready.
            * The contact surface is planar, so that all cells on the surface can
            be described by a single normal vector.
            * The contact forces are represented directly in the local
            coordinate system of the surface. The first self.dim - 1 elements
            of the contact vector are the tangential components of the first
            cell, then the normal component, then tangential of the second cell
            etc.

        """
        # CLARIFICATIONS NEEDED:
        #   1) Do projection and rotation commute on non-matching grids? The
        #   gut feel says yes, but I'm not sure.

        # Process input
        parameters_l = data_l[pp.PARAMETERS]

        # Numerical parameter, value and sensitivity is currently unknown.
        # The thesis of Hueeber is probably a good place to look for information.
        c_num = parameters_l[self.keyword].get(
            "contact_mechanics_numerical_parameter", 100
        )
        # Obtain the four cellwise parameters:
        # Mandatory friction coefficient relates normal and tangential forces.
        # The initial gap will usually be zero.
        # The gap value may be a function of tangential displacement.
        # We assume g(u_t) = - tan(dilation_angle) * || u_t ||
        # The cohesion represents a minimal force, independent of the normal force,
        # that must be overcome before the onset of sliding.
        cellwise_parameters = [
            "friction_coefficient",
            "initial_gap",
            "dilation_angle",
            "cohesion",
        ]

        defaults = [None, 0, 0, 0]
        vals = parameters_l.expand_scalars(
            g_l.num_cells, self.keyword, cellwise_parameters, defaults
        )
        friction_coefficient, initial_gap, dilation_angle, cohesion = (
            vals[0],
            vals[1],
            vals[2],
            vals[3],
        )

        mg = data_edge["mortar_grid"]

        # In an attempt to reduce the sensitivity of the numerical parameter on the
        # model parameters, we scale it with area and an order-of-magnitude estimate
        # for the elastic moduli.
        area = g_l.cell_volumes

        parameters_h = data_h[pp.PARAMETERS][self.discr_h.keyword]
        constit_h = parameters_h["fourth_order_tensor"]
        mean_constit = (
            mg.mortar_to_secondary_avg()
            * mg.primary_to_mortar_avg()
            * 0.5
            * np.abs(g_h.cell_faces * (constit_h.mu + constit_h.lmbda))
        )

        c_num_normal = c_num * mean_constit * area
        c_num_tangential = c_num * mean_constit * area

        # The tractions are scaled with area, so do the same with the cohesion.
        scaled_cohesion = cohesion * area

        # TODO: Implement a single method to get the normal vector with right sign
        # thus the right local coordinate system.

        # Pick the projection operator (defined elsewhere) for this surface.
        # IMPLEMENATION NOTE: It is paramount that this projection is used for all
        # operations relating to this surface, or else directions of normal vectors
        # will get confused.
        projection = data_l["tangential_normal_projection"]

        # The contact force is already computed in local coordinates
        contact_force = data_l[pp.STATE][pp.ITERATE][self.contact_variable]

        # Pick out the tangential and normal direction of the contact force.
        # The contact force of the first cell is in the first self.dim elements
        # of the vector, second cell has the next self.dim etc.
        # By design the tangential force is the first self.dim-1 components of
        # each cell, while the normal force is the last component.
        normal_indices = np.arange(self.dim - 1, contact_force.size, self.dim)
        tangential_indices = np.setdiff1d(np.arange(contact_force.size), normal_indices)
        contact_force_normal = contact_force[normal_indices]
        contact_force_tangential = contact_force[tangential_indices].reshape(
            (self.dim - 1, g_l.num_cells), order="F"
        )

        # The displacement jump (in global coordinates) is found by switching the
        # sign of the second mortar grid, and then sum the displacements on the
        # two sides (which is really a difference since one of the sides have
        # its sign switched).
        # The tangential displacements are relative to the initial state, which in the
        # transient case equals the previous time step.
        previous_displacement_iterate = data_edge[pp.STATE][pp.ITERATE][
            self.mortar_displacement_variable
        ]
        previous_displacement_time = data_edge[pp.STATE][
            self.mortar_displacement_variable
        ]
        displacement_jump_global_coord_iterate = (
            mg.mortar_to_secondary_avg(nd=self.dim)
            * mg.sign_of_mortar_sides(nd=self.dim)
            * previous_displacement_iterate
        )
        displacement_jump_global_coord_time = (
            mg.mortar_to_secondary_avg(nd=self.dim)
            * mg.sign_of_mortar_sides(nd=self.dim)
            * previous_displacement_time
        )
        # Rotated displacement jumps. These are in the local coordinates, on
        # the lower-dimensional grid. For the normal direction, we consider the absolute
        # displacement, not that relative to the initial state.
        displacement_jump_normal = projection.project_normal(g_l.num_cells) * (
            displacement_jump_global_coord_iterate
        )
        # The jump in the tangential direction is in g_l.dim columns, one per
        # dimension in the tangential direction. For the tangential direction, we
        # consider the relative displacement.
        displacement_jump_tangential = (
            projection.project_tangential(g_l.num_cells)
            * (
                displacement_jump_global_coord_iterate
                - displacement_jump_global_coord_time
            )
        ).reshape((self.dim - 1, g_l.num_cells), order="F")

        cumulative_tangential_jump = (
            projection.project_tangential(g_l.num_cells)
            * (displacement_jump_global_coord_iterate)
        ).reshape((self.dim - 1, g_l.num_cells), order="F")

        # Compute gap if it is a function of tangential jump, i.e.
        # g = g(u) + g_0 (careful with sign!)
        # Both dilation angle and g_0 default to zero (see above), implying g(u) = 0
        norm_displacement_jump_tangential = np.linalg.norm(
            cumulative_tangential_jump, axis=0
        )
        gap = initial_gap + np.tan(dilation_angle) * norm_displacement_jump_tangential

        # Compute dg/du_t = tan(dilation_angle) u_t / || u_t ||
        # Avoid dividing by zero if u_t = 0. In this case, we extend to the limit value
        # to zero, see module level explanation.
        ind = np.logical_not(
            np.isclose(cumulative_tangential_jump, 0, rtol=self.tol, atol=self.tol)
        )[0]
        d_gap = np.zeros((g_l.dim, g_l.num_cells))
        # Compute dg/du_t where u_t is nonzero
        tan = np.atleast_2d(np.tan(dilation_angle)[ind])
        d_gap[:, ind] = (
            tan
            * cumulative_tangential_jump[:, ind]
            / norm_displacement_jump_tangential[ind]
        )

        # The friction bound is computed from the previous state of the contact
        # force and normal component of the displacement jump.
        # Note that the displacement jump is rotated before adding to the contact force
        friction_bound = (
            friction_coefficient
            * np.clip(
                -contact_force_normal - c_num_normal * (displacement_jump_normal - gap),
                0,
                np.inf,
            )
            + scaled_cohesion
        )

        num_cells = friction_coefficient.size

        # Find contact and sliding region

        # Contact region is determined from the normal direction.
        penetration_bc = self._penetration(
            contact_force_normal, displacement_jump_normal, c_num_normal, gap
        )
        # Check criterion for sliding
        sliding_bc = self._sliding(
            contact_force_tangential,
            displacement_jump_tangential,
            friction_bound,
            c_num_tangential,
        )

        # Structures for storing the computed coefficients.
        displacement_weight = []  # Multiplies displacement jump
        traction_weight = []  # Multiplies the normal forces
        rhs = np.array([])  # Goes to the right hand side.

        # Zero vectors of the size of the tangential space and the full space,
        # respectively. These are needed to complement the discretization
        # coefficients to be determined below.
        zer = np.array([0] * (self.dim - 1))
        zer1 = np.array([0] * (self.dim))
        zer1[-1] = 1

        # Loop over all mortar cells, discretize according to the current state of
        # the contact
        # The loop computes three parameters:
        # L will eventually multiply the displacement jump, and be associated with
        #   the coefficient in a Robin boundary condition (using the terminology of
        #   the mpsa implementation)
        # r is the right hand side term

        for i in range(num_cells):
            if sliding_bc[i] & penetration_bc[i]:  # in contact and sliding
                # This is Eq (31) in Berge et al, including the regularization
                # described in (32) and onwards. The expressions are somewhat complex,
                # and are therefore moved to subfunctions.
                loc_displacement_tangential, r, v = self._sliding_coefficients(
                    contact_force_tangential[:, i],
                    displacement_jump_tangential[:, i],
                    friction_bound[i],
                    c_num_tangential[i],
                )

                # There is no interaction between displacement jumps in normal and
                # tangential direction. Opposite sign compared to Berge because
                # of jump conventions being opposite.
                L = -np.hstack((loc_displacement_tangential, np.atleast_2d(zer).T))
                normal_displacement = np.hstack((-d_gap[:, i], 1))
                loc_displacement_weight = np.vstack((L, normal_displacement))
                # Right hand side is computed from (24-25). In the normal
                # direction, a contribution from the previous iterate enters to cancel
                # the gap
                r_n = gap[i] - np.dot(d_gap[:, i], cumulative_tangential_jump[:, i].T)

                r_t = r + friction_bound[i] * v
                r = np.vstack((r_t, r_n))
                # Unit contribution from tangential force
                loc_traction_weight = np.eye(self.dim)
                # Zero weight on normal force
                loc_traction_weight[-1, -1] = 0
                # Contribution from normal force
                # NOTE: The sign is different from Berge (31); the paper is wrong
                loc_traction_weight[:-1, -1] = -friction_coefficient[i] * v.ravel()

            elif ~sliding_bc[i] & penetration_bc[i]:  # In contact and sticking
                # Weight for contact force computed according to (30) in Berge.
                # NOTE: There is a sign error in the paper, the coefficient for the
                # normal contact force should have a minus in front of it
                loc_traction_tangential = (
                    -friction_coefficient[i]  # The minus sign is correct
                    * displacement_jump_tangential[:, i].ravel("F")
                    / friction_bound[i]
                )
                # Unit coefficient for all displacement jumps
                loc_displacement_weight = np.eye(self.dim)
                # For non-constant gap, relate normal and tangential jumps
                loc_displacement_weight[-1, :-1] = -d_gap[:, i]

                # Tangential traction dependent on normal one
                loc_traction_weight = np.zeros((self.dim, self.dim))
                loc_traction_weight[:-1, -1] = loc_traction_tangential

                # The right hand side is the previous tangential jump, and the gap
                # value in the normal direction.
                r_t = displacement_jump_tangential[:, i]
                r_n = gap[i] - np.dot(d_gap[:, i], cumulative_tangential_jump[:, i].T)
                # assert np.isclose(r_n, initial_gap[i])
                r = np.hstack((r_t, r_n)).T

            elif ~penetration_bc[i]:  # not in contact
                # This is a free boundary, no conditions on displacement
                loc_displacement_weight = np.zeros((self.dim, self.dim))

                # Free boundary conditions on the forces.
                loc_traction_weight = np.eye(self.dim)
                r = np.zeros(self.dim)

            else:  # should never happen
                raise AssertionError("Should not get here")

            # Depending on the state of the system, the weights in the tangential direction may
            # become huge or tiny compared to the other equations. This will
            # impede convergence of an iterative solver for the linearized
            # system. As a partial remedy, rescale the condition to become
            # closer to unity.
            w_diag = np.diag(loc_displacement_weight) + np.diag(loc_traction_weight)
            W_inv = np.diag(1 / w_diag)
            loc_displacement_weight = W_inv.dot(loc_displacement_weight)
            loc_traction_weight = W_inv.dot(loc_traction_weight)
            r = r.ravel() / w_diag

            # Append to the list of global coefficients.
            displacement_weight.append(loc_displacement_weight)
            traction_weight.append(loc_traction_weight)
            rhs = np.hstack((rhs, r))

        num_blocks = len(traction_weight)
        data_traction = np.array(traction_weight).ravel(order="C")

        data_displacement = np.array(displacement_weight).ravel(order="C")

        data_l[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.traction_matrix_key
        ] = pp.utils.sparse_mat.csr_matrix_from_blocks(
            data_traction, self.dim, num_blocks
        )
        data_l[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.displacement_matrix_key
        ] = pp.utils.sparse_mat.csr_matrix_from_blocks(
            data_displacement, self.dim, num_blocks
        )
        data_l[pp.DISCRETIZATION_MATRICES][self.keyword][self.rhs_matrix_key] = rhs

        # Also store the contact state
        data_l[pp.STATE][pp.ITERATE]["penetration"] = penetration_bc
        data_l[pp.STATE][pp.ITERATE]["sliding"] = sliding_bc

    @pp.time_logger(sections=module_sections)
    def assemble_matrix_rhs(self, g: pp.Grid, data: Dict):
        # Generate matrix for the coupling. This can probably be generalized
        # once we have decided on a format for the general variables
        traction_coefficient = data[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.traction_matrix_key
        ]
        displacement_coefficient = data[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.displacement_matrix_key
        ]

        rhs = data[pp.DISCRETIZATION_MATRICES][self.keyword][self.rhs_matrix_key]

        return traction_coefficient, displacement_coefficient, rhs

    # Active and inactive boundary faces
    @pp.time_logger(sections=module_sections)
    def _sliding(self, Tt: np.ndarray, ut: np.ndarray, bf: np.ndarray, ct: np.ndarray):
        """Find faces where the frictional bound is exceeded, that is, the face is
        sliding.

        Arguments:
            Tt (np.array, nd-1 x num_cells): Tangential forces.
            ut (np.array, nd-1 x num_cells): Displacements jump velocity in tangential
                direction.
            bf (np.array, num_cells): Friction bound.
            ct (np.array, num_cells): Numerical parameter that relates displacement jump to
                tangential forces. See Huber et al for explanation.

        Returns:
            boolean, size num_faces: True if |-Tt + ct*ut| > bf for a face

        """
        # Use thresholding to not pick up faces that are just about sticking
        # Not sure about the sensitivity to the tolerance parameter here.
        return self._l2(-Tt - ct * ut) - bf > self.tol

    @pp.time_logger(sections=module_sections)
    def _penetration(
        self, Tn: np.ndarray, un: np.ndarray, cn: np.ndarray, gap: np.ndarray
    ) -> np.ndarray:
        """Find faces that are in contact.

        Arguments:
            Tn (np.array, num_cells): Normal forces.
            un (np.array, num_cells): Displament jump in normal direction.
            cn (np.array, num_cells): Numerical parameter that relates displacement jump to
                normal forces. See Huber et al for explanation.
            gap (np.array, num_cells): Value of gap function.

        Returns:
            boolean, size num_cells: True if |-Tu + cn*un| > 0 for a cell.

        """
        # Not sure about the sensitivity to the tolerance parameter here.
        return (-Tn - cn * (un - gap)) > self.tol

    #####
    ## Below here are different help function for calculating the Newton step
    #####

    @pp.time_logger(sections=module_sections)
    def _e(self, Tt: np.ndarray, cut: np.ndarray, bf: np.ndarray) -> np.ndarray:
        # Compute part of (32) in Berge et al.
        return bf / self._l2(-Tt - cut)

    @pp.time_logger(sections=module_sections)
    def _Q(self, Tt: np.ndarray, cut: np.ndarray, bf: np.ndarray) -> np.ndarray:
        # Implementation of the term Q involved in the calculation of (32) in Berge
        # et al.
        # This is the regularized Q
        numerator = -Tt.dot((-Tt - cut).T)

        # Regularization to avoid issues during the iterations to avoid dividing by
        # zero if the faces are not in contact durign iterations.
        denominator = max(bf, self._l2(-Tt)) * self._l2(-Tt - cut)

        return numerator / denominator

    @pp.time_logger(sections=module_sections)
    def _M(self, Tt: np.ndarray, cut: np.ndarray, bf: np.ndarray) -> np.ndarray:
        """Compute the coefficient M used in Eq. (32) in Berge et al."""
        Id = np.eye(Tt.shape[0])
        # M = e * (I - Q)
        return self._e(Tt, cut, bf) * (Id - self._Q(Tt, cut, bf))

    @pp.time_logger(sections=module_sections)
    def _hf(self, Tt: np.ndarray, cut: np.ndarray, bf: np.ndarray) -> np.ndarray:
        # This is the product e * Q * (-Tt + cut), used in computation of r in (32)
        return self._e(Tt, cut, bf) * self._Q(Tt, cut, bf).dot(-Tt - cut)

    @pp.time_logger(sections=module_sections)
    def _sliding_coefficients(
        self, Tt: np.ndarray, ut: np.ndarray, bf: np.ndarray, c: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the regularized versions of coefficients L, v and r, defined in
        Eq. (32) and section 3.2.1 in Berge et al. and used in Eq. (31).

        Arguments:
            Tt: Tangential forces. np array, one or two elements
            ut: Tangential displacement increment. Same size as Tt.
            bf: Friction bound for this mortar cell.
            c: Numerical parameter

        Returns:
            L: Weights for tangential displacement increment.
            v: Weights for normal traction.
            r: rhs contribution.

        """
        if Tt.ndim <= 1:
            Tt = np.atleast_2d(Tt).T
            ut = np.atleast_2d(ut).T

        cut = c * ut
        # Identity matrix
        Id = np.eye(Tt.shape[0])

        # Shortcut if the friction coefficient is effectively zero.
        # Numerical tolerance here is likely somewhat arbitrary.
        if bf <= self.tol:
            return (
                0 * Id,
                bf * np.ones((Id.shape[0], 1)),
                (-Tt - cut) / self._l2(-Tt - cut),
            )

        # Compute the coefficient M
        coeff_M = self._M(Tt, cut, bf)

        # Regularization during the iterations requires computations of parameters
        # alpha, beta, delta. In degenerate cases, use
        beta = 1
        # Avoid division by zero:
        l2_Tt = self._l2(-Tt)
        if l2_Tt > self.tol:
            alpha = -Tt.T.dot(-Tt - cut) / (l2_Tt * self._l2(-Tt - cut))
            # Parameter delta.
            # NOTE: The denominator bf is correct. The definition given in Berge is wrong.
            delta = min(l2_Tt / bf, 1)

            if alpha < 0:
                beta = 1 / (1 - alpha * delta)

        # The expression (I - beta * M)^-1
        # NOTE: In the definition of \tilde{L} in Berge, the inverse on the inner
        # paranthesis is missing.
        IdM_inv = np.linalg.inv(Id - beta * coeff_M)

        L = c * (IdM_inv - Id)
        r = -IdM_inv.dot(self._hf(Tt, cut, bf))
        v = IdM_inv.dot(-Tt - cut) / self._l2(-Tt - cut)

        return L, r, v

    @pp.time_logger(sections=module_sections)
    def _l2(self, x):
        x = np.atleast_2d(x)
        return np.sqrt(np.sum(x ** 2, axis=0))


@pp.time_logger(sections=module_sections)
def set_projections(
    gb: pp.GridBucket, edges: Optional[List[Tuple[pp.Grid, pp.Grid]]] = None
) -> None:
    """Define a local coordinate system, and projection matrices, for all
    grids of co-dimension 1.

    The function adds one item to the data dictionary of all GridBucket edges
    that neighbors a co-dimension 1 grid, defined as:
        key: tangential_normal_projection, value: pp.TangentialNormalProjection
            provides projection to the surface of the lower-dimensional grid

    Note that grids of co-dimension 2 and higher are ignored in this construction,
    as we do not plan to do contact mechanics on these objects.

    It is assumed that the surface is planar.

    """
    if edges is None:
        edges = [e for e, _ in gb.edges()]

    # Information on the vector normal to the surface is not available directly
    # from the surface grid (it could be constructed from the surface geometry,
    # which spans the tangential plane). We instead get the normal vector from
    # the adjacent higher dimensional grid.
    # We therefore access the grids via the edges of the mixed-dimensional grid.
    for e in edges:
        d_m = gb.edge_props(e)

        mg = d_m["mortar_grid"]
        # Only consider edges where the lower-dimensional neighbor is of co-dimension 1
        if not mg.dim == (gb.dim_max() - 1):
            continue

        # Neigboring grids
        g_l, g_h = gb.nodes_of_edge(e)

        # Find faces of the higher dimensional grid that coincide with the mortar
        # grid. Go via the primary to mortar projection
        # Convert matrix to csr, then the relevant face indices are found from
        # the (column) indices
        faces_on_surface = mg.primary_to_mortar_int().tocsr().indices

        # Find out whether the boundary faces have outwards pointing normal vectors
        # Negative sign implies that the normal vector points inwards.
        sgn, _ = g_h.signs_and_cells_of_boundary_faces(faces_on_surface)

        # Unit normal vector
        unit_normal = g_h.face_normals[: g_h.dim] / g_h.face_areas
        # Ensure all normal vectors on the relevant surface points outwards
        unit_normal[:, faces_on_surface] *= sgn

        # Now we need to pick out *one*  normal vector of the higher dimensional grid

        # which coincides with this mortar grid, so we kill off all entries for the
        # "other" side:
        unit_normal[:, mg._ind_face_on_other_side] = 0

        # Project to the mortar and then to the fracture
        outwards_unit_vector_mortar = mg.primary_to_mortar_int().dot(unit_normal.T).T
        normal_lower = mg.mortar_to_secondary_int().dot(outwards_unit_vector_mortar.T).T

        # NOTE: The normal vector is based on the first cell in the mortar grid,
        # and will be pointing from that cell towards the other side of the
        # mortar grid. This defines the positive direction in the normal direction.
        # Although a simpler implementation seems to be possible, going via the
        # first element in faces_on_surface, there is no guarantee that this will
        # give us a face on the positive (or negative) side, hence the more general
        # approach is preferred.
        #
        # NOTE: The basis for the tangential direction is determined by the
        # construction internally in TangentialNormalProjection.
        projection = pp.TangentialNormalProjection(normal_lower)

        d_l = gb.node_props(g_l)
        # Store the projection operator in the lower-dimensional data
        d_l["tangential_normal_projection"] = projection
