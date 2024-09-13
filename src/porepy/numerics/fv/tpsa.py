"""Implementation of the two-point stress approximation derived in

    Nordbotten and Keilegavlen, Two-point stress approximation: A simple and robust
    finite volume method for linearized (poro-) mechanics and Stokes flow,
    arXiv:2405.10390.

See in particular the apendix for the expressions of the discretization schemes.

In addition to the main discretization class, Tpsa, the module also contains various
helper classes that are really just containers for data needed for the discretization.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.grids.grid import Grid
from porepy.numerics.fv import fvutils
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data
from porepy.params.tensor import FourthOrderTensor


@dataclass
class _Numbering:
    """Helper class to store scalars and index arrays."""

    fi: np.ndarray
    """Face indices, with interior faces counted twice, once for each side."""
    ci: np.ndarray
    """Cell indices corresponding to the face indices in fi."""
    sgn: np.ndarray
    """Signs of the normal vector for the faces in fi."""
    fi_expanded: np.ndarray
    """Vector version of fi. Face 0 begets indices {0, 1} (2d) or {0, 1, 2} (3d)."""
    ci_expanded: np.ndarray
    """Vector version of ci. Cell 0 begets indices {0, 1} (2d) or {0, 1, 2} (3d)."""
    sgn_nd: np.ndarray
    """Vector version of sgn. All quantities are repeated nd times."""

    sgn_bf: np.ndarray


@dataclass
class _BoundaryFilters:
    """Helper class to store filters for applying various boundary conditions.

    The class stores only attributes needed in the implementation; it may well be that
    different choices in implementation would have needed different filters.
    """

    dir_pass_nd: sps.sparray
    """Filter that only lets through boundary faces with Dirichlet conditions."""
    dir_notpass: sps.sparray
    """Filter that removes Dirichlet conditions."""
    dir_notpass_nd: sps.sparray
    """Filter that removes boundary faces with Dirichlet conditions."""
    neu_pass_nd: sps.sparray
    """Filter that only lets through boundary faces with Neumann conditions."""
    neu_notpass_nd: sps.sparray
    """Filter that removes boundary faces with Neumann conditions."""
    neu_rob_pass_nd: sps.sparray
    """Filter that only lets through boundary faces with Neumann or Robin conditions."""
    rob_pass_nd: sps.sparray
    """Filter that only lets through boundary faces with Robin conditions."""


@dataclass
class _CellToFaceMaps:
    r"""Helper class to store maps from faces to cells.

    To see which maps are needed, confer the tpsa paper, specifically the quantity \Xi
    and its complement \tilde{\Xi}.

    All maps are from nd to nd unless specifically marked by the name.
    """

    c2f: sps.sparray
    """Map from cell to face quantities. Boundary conditions are ignored."""
    c2f_compl: sps.sparray
    """Complement map from cell to faces."""
    c2f_scalar_2_nd: sps.sparray
    """Map from scalar cell-wise quantities to vector face quantities."""
    c2f_compl_scalar_2_nd: sps.sparray
    """Complement map from scalar cell-wise quantities to vector face quantities."""
    b2f_rob: sps.sparray
    """Map from faces with Robin boundary conditions assigned to boundary faces. This
    works as a filter, but has non-binary weights set by the weight parameter in the
    Robin boundary condition.
    """
    b2f_rob_compl: sps.sparray
    """Complement of map from faces with Robin boundary conditions assigned to boundary
    faces. This works as a filter, but has non-binary weights set by the weight
    parameter in the Robin boundary condition.
    """


@dataclass
class _Distances:
    """Helper class to store various distance-related quantities needed in the
    discretization.

    As can be seen from the Tpsa paper, the discretization coefficients are often on the
    form mu/d, where mu is the shear modulus and d is a distance measure (this also
    holds for the Robin boundary condition), hence such quantities are stored here.
    """

    dist_fc_cc: np.ndarray
    """Distance from face to cell centers. Follows the fi/ci ordering defined in
    _Numbering."""
    mu_by_dist_fc_cc: np.ndarray
    """Shear modulus divided by the distance from face to cell centers. Follows the
    fi/ci ordering defined in _Numbering."""
    mu_by_dist_fc_cc_bound: np.ndarray
    """Shear modulus divided by the distance from face to cell centers, also accounting
    for Robin boundary conditions. Follows the fi/ci ordering defined in _Numbering."""
    inv_mu_by_dist_array: sps.dia_array
    """Diagonal array of the inverse of mu_by_dist_fc_cc_bound."""
    rob_weight: np.ndarray
    """Robin weights, extracted from the displacement boundary conditions. Included in
    this container since, in the description of Tpsa, the weights of the Robin condition
    is in part given the interpretation of a distance measure.
    """


class Tpsa:
    """Implementation of the two-point stress approximation derived in

        Nordbotten and Keilegavlen, Two-point stress approximation: A simple and robust
        finite volume method for linearized (poro-) mechanics and Stokes flow,
        arXiv:2405.10390.

    See in particular the apendix for the expressions of the discretization schemes.

    Example (intended to show basic input and output functionality):
        # Imports:
        >>> import numpy as np
        >>> import porepy as pp
        >>> import scipy.sparse as sps
        # Define geometry:
        >>> g = pp.CartGrid([3, 3])
        >>> g.compute_geometry()
        >>> nf, nc = g.num_faces, g.num_cells
        # Boundary condition - all Dirichlet:
        >>> bf = g.get_all_boundary_faces()
        >>> bc = pp.BoundaryConditionVectorial(g, bf, bf.size * ['dir'])
        # Elastic moduli (expressed by Lame parameters):
        >>> mu = 1 + np.arange(nc)
        >>> lmbda = np.ones(nc)
        >>> C = pp.FourthOrderTensor(mu, lmbda)
        # Gather information in a dict, also prepare for storing discretization
        # matrices:
        >>> key = 'mechanics'
        >>> data = {pp.PARAMETERS: {key: {'fourth_order_tensor': C, 'bc': bc}}}
        >>> data[pp.DISCRETIZATION_MATRICES] = {key: {}}
        # Construct discretization object, link it to the keyword
        >>> discr = pp.Tpsa(key)
        # Discretize:
        >>> discr.discretize(g, data)

        # This will produce discretization matrices stored in
        #    data[pp.DISCRETIZATION_MATRICES][key]
        # To assemble these and solve for the primary variables, do the following:

        # Deal with rotation variable being 1d if g.dim == 2, 3d if g.dim == 3
        >>> num_rot_face = nf if g.dim == 2 else 3 * nf
        >>> num_rot_cell = nc if g.dim == 2 else 3 * nc
        >>> div_scalar = pp.fvutils.scalar_divergence(g)
        >>> div_vector = pp.fvutils.vector_divergence(g)
        >>> div_rot = div_scalar if g.dim == 2 else div_vec

        >>> matrices = data[pp.DISCRETIZATION_MATRICES][key]

        # Assemble matrix of face quantities:
        >>> face_discr = sps.block_array(
                        [
                            [
                                matrices["stress"], matrices["stress_rotation"],
                                matrices["stress_total_pressure"],
                            ], [
                                matrices["rotation_displacement"],
                                matrices["rotation_diffusion"], sps.csr_array(
                                    (num_rot_face, nc)),
                            ], [
                                matrices["solid_mass_displacement"], sps.csr_array(
                                    (nf, num_rot_cell)),
                                matrices["solid_mass_total_pressure"],
                            ],
                        ],
                    )
        # Assemble divergence operator
        >>> div = sps.block_diag([div_vector, div_rot, div_scalar])
        # Matrix for imposing boundary conditions
        >>> rhs_matrix = sps.bmat(
                    [
                        [
                            matrices["bound_stress"],
                            sps.csr_array((nf * g.dim, num_rot_face)),
                        ], [
                            matrices["bound_rotation_displacement"],
                            sps.csr_matrix((num_rot_face, num_rot_face)),
                        ], [
                            matrices["bound_mass_displacement"], sps.csr_matrix((nf,
                            num_rot_face)),
                        ],
                    ]
                )
        # Boundary conditions for displacement is a vector of size nf * g.dim.
        # To keep track of indices and dimensions, it can be useful to do:
        >>> bc = np.zeros((g.dim, nf))
        # Set conditions in x-direction on boundary faces
        >>> bc[0, bf] = np.arange(bf.size)
        # Conditions in the y-direction are constant
        >>> bc[1, bf] = 42
        # Ravel. The vector contains the x-component of the first face, then the
        # y-component etc.
        >>> bc = bc.ravel('F')
        # Boundary conditions for rotation: Should be zero here, since there is no
        # rotation diffusion, thus the rotation is not an independent variable:
        >>> bc_rot = np.zeros(nf)
        # The full bc
        >>> bc_all = np.hstack((bc, bc_rot))
        # Convert to right hand side
        >>> b = - div @ rhs_matrix @ bc_all

        # Accumulation terms
        >>> accum = sps.block_diag(
                [
                    sps.csr_array((nc * g.dim, nc * g.dim)),
                    sps.dia_array((1.0 / mu, 0), shape=(num_rot_cell, num_rot_cell)),
                    sps.dia_array((1.0 / lmbda, 0), shape=(num_rot_cell, num_rot_cell)),
                ],
                format="csr",
            )
        # System matrix (the minus sign on accum corresponds to the sign in the paper)
        >>> A = div @ face_discr - accum
        # Solve
        >>> x = sps.linalg.spsolve(A, b)
        # Displacement variable:
        >>> u = x[:g.dim * nc]
        # X-component of the displacement
        >>> ux = u[::g.dim]
        # Rotation
        >>> r = x[g.dim * nc: g.dim * nc + num_rot_cell]
        # Solid pressure
        >>> p = x[g.dim * nc + num_rot_cell:]

        This example may be moved to a tutorial at a later point. Functionality to
        integrate Tpsa into the multiphysics models of PorePy is in the works.

        TODO: Migrate this example to a tutorial.

    """

    def __init__(self, keyword: str) -> None:

        self.keyword: str = keyword
        """Keyword used to identify the parameter dictionary."""

        self.stress_displacement_matrix_key: str = "stress"
        """Keyword used to identify the discretization matrix for the stress generated
        by the cell center displacements. Defaults to 'stress'.
        """
        self.stress_rotation_matrix_key: str = "stress_rotation"
        """Keyword used to identify the discretization matrix for the rotational stress
        generated by the cell center rotational stress. Defaults to
        'stress_rotation'."""
        self.stress_total_pressure_matrix_key: str = "stress_total_pressure"
        """Keyword used to identify the discretization matrix for the stress generated
        by the cell center solid pressure. Defaults to 'stress_total_pressure'."""
        self.rotation_displacement_matrix_key: str = "rotation_displacement"
        """Keyword used to identify the discretization matrix for the rotation
        generated by the cell center displacements. Defaults to
        'rotation_displacement'."""
        self.rotation_diffusion_matrix_key: str = "rotation_diffusion"
        """Keyword used to identify the discretization matrix for the rotational
        diffusion generated by the cell center rotational stress. Defaults to
        'rotation_diffusion'."""
        self.mass_total_pressure_matrix_key: str = "solid_mass_total_pressure"
        """Keyword used to identify the discretization matrix for the solid mass
        conservation generated by the cell center solid pressure. Defaults to
        'solid_mass_total_pressure'."""
        self.mass_displacement_matrix_key: str = "solid_mass_displacement"
        """Keyword used to identify the discretization matrix for the solid mass
        conservation generated by the cell center displacements. Defaults to
        'solid_mass_displacement'."""
        # Boundary conditions
        self.bound_stress_matrix_key: str = "bound_stress"
        """Keyword used to identify the discretization matrix for the boundary
        conditions for stress. Defaults to 'bound_stress'."""
        self.bound_rotation_displacement_matrix_key: str = "bound_rotation_displacement"
        """Keyword used to identify the discretization matrix for the boundary
        conditions for the displacement variable in the rotational flux. Defaults to
        'bound_rotation_displacement'.
        """
        self.bound_mass_displacement_matrix_key: str = "bound_mass_displacement"
        """Keyword used to identify the discretization matrix for the boundary
        conditions for the displacement variable in the solid mass flux. Defaults to
        'bound_mass_displacement'.
        """
        self.bound_rotation_diffusion_matrix_key: str = "bound_rotation_diffusion"
        """Keyword used to identify the discretization matrix for the boundary
        conditions for the rotation variable in the rotational flux. Defaults to
        'bound_rotation_diffusion'.
        """
        # Fields related to reconstruction of displacement on boundary faces
        self.bound_displacement_cell_matrix_key: str = "bound_displacement_cell"
        """Keyword used to identify the discretization matrix for the cell center
        displacement contribution to boundary displacement reconstrution. Defaults to
        'bound_displacement_cell'."""
        self.bound_displacement_face_matrix_key: str = "bound_displacement_face"
        """Keyword used to identify the discretization matrix for the contribtution from
        boundary condition of the displacement variable to the boundary displacement
        reconstrution. Defaults to 'bound_displacement_face'."""
        self.bound_displacement_rotation_cell_matrix_key: str = (
            "bound_displacement_rotation_cell"
        )
        """Keyword used to identify the discretization matrix for the cell center
        rotation contribution to boundary displacement reconstrution. Defaults to
        'bound_displacement_rotation_cell'."""
        self.bound_displacement_solid_pressure_cell_matrix_key: str = (
            "bound_displacement_solid_pressure_cell"
        )
        """Keyword used to identify the discretization matrix for the cell center
        solid pressure contribution to boundary displacement reconstrution. Defaults to
        'bound_displacement_solid_pressure_cell'."""

    def discretize(self, sd: Grid, data: dict) -> None:
        """Discretize linear elasticity equation using a two-point stress approximation
        (TPSA).

        Optionally, the discretization can include microrotations, in the form of a
        Cosserat material.

        The method constructs a set of discretization matrices for the balance of linear
        and angular momentum, as well as conservation of solid mass.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in ``data[pp.PARAMETERS][self.keyword]``.
            matrix_dictionary, for storage of discretization matrices.
                Stored in ``data[pp.DISCRETIZATION_MATRICES][self.keyword]``

        parameter_dictionary contains the entries:
            - fourth_order_tensor: ``class:~porepy.params.tensor.FourthOrderTensor``
                Stiffness tensor defined cell-wise. Note that the discretization will
                act directly on the Lame parameters ``FourthOrderTensor.mu``,
                ``FourthOrderTensor.lmbda``. That is, anisotropy encoded into the
                stiffness tensor will not be considered.

            - bc: ``class:~porepy.params.bc.BoundaryConditionVectorial``
                Boundary conditions for the displacement variable.
            - bc_rot: ``class:~porepy.params.bc.BoundaryConditionVectorial``
                Boundary condition for the rotation variable. Will only be considered if
                the Cosserat parameter is provided. Robin conditions are not
                implemnented and will raise an error.

            - cosserat_parameter (optional): np.ndarray giving the Cosserat parameter,
                which can be considered a parameter for diffusion of microrotations.
                Should have length equal to the number of cells. If not provided, the
                Cosserat parameter is set to zero.

        matrix_dictionary will be updated with the following entries:
            - stress: ``sps.csc_matrix (sd.dim * sd.num_faces, sd.dim * sd.num_cells)``
                Stress discretization, cell center contribution
            - bound_stress:
                ``sps.csc_matrix (sd.dim * sd.num_faces, sd.dim * sd.num_faces)``
                Stress discretization, face contribution.
            - rotation_displacement: ``sps.csc_matrix``
                     (sd.dim * sd.num_faces, sd.dim * sd.num_cells) (3d)``
                or ``(sd.dim * sd.num_faces, sd.num_cells) (2d)``
                Rotation generated by displacement discretization, cell center
                contribution.
            - bound_rotation_displacement: ``sps.csc_matrix``
                     ``(sd.dim * sd.num_faces, sd.dim * sd.num_faces) (3d)``
                or   ``(sd.dim * sd.num_faces, sd.num_faces) (2d)``
                Rotation generated by displacement discretization, boundary face
                contribution.
            - rotation_diffusion: ``sps.csc_matrix``
                     ``(sd.dim * sd.num_faces, sd.dim * sd.num_cells) (3d)``
                or   ``(sd.dim * sd.num_faces, sd.num_cells) (2d)``
                Rotation diffusion discretization, cell center contribution.
            - bound_rotation_diffusion: ``sps.csc_matrix``
                     ``(sd.dim * sd.num_faces, sd.dim * sd.num_faces) (3d)``
                or ``(sd.dim * sd.num_faces, sd.num_faces) (2d)``
                Rotation diffusion discretization, boundary face contribution.
            - stress_total_pressure: ``sps.csc_matrix (sd.dim * sd.num_faces,
                                                       sd.num_cells)``
                Stress generated by total pressure discretization.
            - bound_rotation_diffusion: ``sps.csc_matrix``
                     ``(sd.dim * sd.num_faces, sd.dim * sd.num_faces) (3d)
                or ``(sd.dim * sd.num_faces, sd.num_faces) (2d)``
                Rotation diffusion discretization, boundary face contribution
            - mass_total_pressure: ``sps.csc_matrix (sd.num_faces, sd.num_faces)``
                Solid mass movement generated by total pressure.
            - mass_displacement:
                ``sps.csc_matrix (sd.num_faces, sd.num_cells * sd.dim)``
                Solid mass movement generated by displacement. Cell center contribution.
            - bound_mass_displacement:
                ``sps.csc_matrix (sd.num_faces, sd.num_cells * sd.dim)``
                Solid mass movement generated by displacement. Boundary face
                contribution.
            - bound_displacement_cell:
                ``sps.csc_matrix (sd.dim * sd.num_faces, sd.dim * sd.num_cells)``
                Operator for reconstructing the displacement trace. Cell center
                displacement contribution.
            - bound_displacement_face:
                ``sps.csc_matrix (sd.dim * sd.num_faces, sd.dim * sd.num_faces)``
                Operator for reconstructing the displacement trace. Boundary condition
                contribution.
            - bound_displacement_rotation_cell: ``sps.csc_matrix``
                   ``(sd.dim * sd.num_faces, sd.dim * sd.num_cells)`` (3d)
                or ``(sd.dim * sd.num_faces, sd.num_cells)`` (2d)
                Operator for reconstructing the displacement trace. Cell center
                rotation contribution.
            - bound_displacement_solid_pressure_cell:
                    ``sps.csc_matrix (sd.dim * sd.num_faces, sd.num_cells)``
                Operator for reconstructing the displacement trace. Cell center solid
                pressure contribution.

        Raises:
            ValueError: If a scalar BoundaryCondition is given for the rotation variable
                in 3d (where rotation is a vector), or a BoundaryConditionVectorial is
                given for the rotation variable in 2d.
            NotImplementedError: If Robin boundary conditions are specified for the
                rotation variable.
            NotImplementedError: If the displacement variable has been assigned Robin
                conditions with a non-diagonal weight, or if the ``basis`` attribute of
                the displacement boundary condition is not a diagonal matrix.

        Parameters:
            sd: grid, or a subclass, with geometry fields computed.
            data: For entries, see above.

        """
        # Overview of implementation: The implementation closely follows the description
        # of Tpsa given in the arXiv paper (see module-level documentation).
        # Specifically, the discretization is derived in appendix 2 therein, with some
        # central definition provided in Section 1. The ingredients of the
        # discretization are:
        # 1. Quantities related to bookkeeping, such as face and cell indices.
        # 2. Distance measures (e.g., cell-face) sometimes scaled with the shear
        #    modulus.
        # 3. Mappings that compute averages of cell-quantities and map them to faces.
        # 4. Specical treatment of boundary conditions.
        #
        # From these quantities, matrices that represent the different components of the
        # discretization can be constructed. The actual discretization matrices are
        # specified in Equation (A2.24) in the Tpsa paper (referring to the version
        # found at https://arxiv.org/abs/2405.10390v2) for internal faces and (A2.28-31)
        # for boundary faces. Compared to the description in the paper, the
        # implementation uses a more explicit treatment of boundary conditions and
        # prefers more explicit naming of mathematical operators, but should otherwise
        # be equivalent.
        #
        # Specification of boundary conditions: In the paper, the boundary conditions
        # are given on the form
        #
        #       b(\sigma_k * n_k - g^u) = 2\mu (g^u - u_k).
        #
        # Here, b is considered a length scale and is also denoted \delta_k^j (see the
        # bottom of page 10 in the arXiv version of the paper, link above). The
        # superscript on g^u is used in the paper to mark the boundary condition for the
        # displacement variable (and not the rotation). It is understood that b = 0
        # gives a Dirichlet condition, b = infinity gives a Neumann condition, and other
        # values (positive and negative) form the Robin case. In PorePy, Dirichlet and
        # Neumann conditions are set separately, while the Robin parameter, \alpha, is
        # interpreted to be on the form
        #
        #       \sigma_k * n_k + \alpha u = g^u
        #
        # Thus, the PorePy \alpha corresponds to 2\mu / (\delta) in the paper.
        #
        # Implementation choices:
        # 1. To improve oversight of the implementation, it was decided to move
        #    computation of auxiliary quantities into static helper methods, and pass
        #    data around through helper dataclasses (namedtuples or dictionaries could
        #    also have been used). This is at times a bit awkward, but the alternative
        #    was a more chaotic implementation.
        # 2. Boundary conditions are implemented through a combination of filtering and
        #    average maps. This is in contrast to the tpsa paper, which presents a
        #    unified implementation of all types of boundary conditions based on the
        #    average maps. While this approach in EK's assessment would not be too
        #    difficult to follow, the more explicit treatment herein was preferred
        #    partly for ease of implementation and interpretation, and (honestly) partly
        #    for reasons of exhaustion on EK's part. For future reference, the most
        #    natural alternative approach is to introduce one weight each for Dirichlet
        #    and Neumann boundaries and let Robin be an average. The first step would be
        #    to verify that this does not introduce division by zero somewhere.
        #
        # Known shortcomings and known unknowns:
        # 1. The implementation of boundary conditions for the rotation variable covers
        #    only Neumann and Dirichlet conditions. Robin conditions should be feasible,
        #    but have not been prioritized.
        # 2. Robin conditions for the displacement variable are only implemented for a
        #    subset of the variations possible within PorePy. Errors are raised if the
        #    specified boundary condition falls outside the scope of the implementation.
        # 3. Non-homogeneous boundary conditions have generally not been thoroughly
        #    tested, mainly due to limited capacity. TODO: Such tests should be
        #    undertaken before the code is used for production purposes. EK's gut
        #    feeling in terms of the reliability of the different components is:
        #       * Internal faces (the main discretization) and homogeneous boundary
        #         conditions should be correct.
        #       * Non-zero Dirichlet and Neumann conditions are expected to be correct.
        #         Note however that this is mainly verified through unit tests that
        #         compare computed with numbers interpreted from the Tpsa paper for a
        #         specific geometry; see test_tpsa.py for details.
        #       * Non-zero Robin corrections should again be correct, with the same
        #         caveat as for Dirichlet and Neumann conditions. Since some additional
        #         interpretation is needed to translate Robin conditions between the
        #         formats in PorePy and the paper, the likelihood of bugs here is
        #         higher.
        #       * The functionality for boundary displacement recovery has not been
        #         tested. It should be correct for the easy case of Dirichlet
        #         conditions, quite likely also for Neumann conditions, while Robin
        #         should be considered a likely weak point.
        # 4. For the term 'mass_total_pressure', it is not clear how to interpret
        #    boundary conditions, and treatment of such conditions is therefore somewhat
        #    improvised. Plainly, EK does not know what to do here, but the current
        #    implementation seems to work. NOTE that this term acts as a numerical
        #    stabilization, and without it, oscillations may arise in the displacement
        #    variable. Thus, if oscillations are observed in the displacement close to
        #    the boundary, we may need to revisit this term.

        parameter_dictionary: dict[str, Any] = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]

        # Structure for bookkeeping.
        numbering = self._create_numbering(sd)
        nc, nf, nd = sd.num_cells, sd.num_faces, sd.dim

        # Fetch parameters for the mechanical behavior.
        stiffness: FourthOrderTensor = parameter_dictionary["fourth_order_tensor"]
        # The Cosserat parameter, if present. If this is None, the Cosserat parameter is
        # considered to be zero. In practice, we will set all Cosserat discretization
        # matrices to zero with no explicit computations.
        cosserat_values: np.ndarray | None = parameter_dictionary.get(
            "cosserat_parameter", None
        )

        # Boundary condition object. Use the keyword 'bc' here to be compatible with the
        # implementation in mpsa.py, although symmetry with the boundary conditions for
        # rotation seems to call for a keyword like 'bc_disp'.
        bnd_disp: pp.BoundaryConditionVectorial = parameter_dictionary["bc"]

        # Boundary conditions for the rotation variable. This should only be used if
        # the Cosserat parameter is non-zero. Since the rotation variable is scalar if
        # nd == 2 and vector if nd == 3, the type of boundary condition depends on the
        # dimension.
        bnd_rot: pp.BoundaryCondition | pp.BoundaryConditionVectorial = (
            parameter_dictionary.get("bc_rot", None)
        )

        # Check that the type of boundary condition is consistent with the dimension.
        # This is a bit awkward, since it requires an if-else on the client side, but
        # the alternative is to always use a vectorial boundary condition and make a
        # hack to interpret the vectorial condition as a scalar one for 2d problems.
        # Note that, if the Cosserat parameter is zero or not provided, all of this is
        # irrelevant.
        if nd == 2:
            if isinstance(bnd_rot, pp.BoundaryConditionVectorial):
                raise ValueError(
                    "Boundary conditions for rotations should be scalar if nd == 2"
                )
        elif nd == 3:
            if isinstance(bnd_rot, pp.BoundaryCondition):
                raise ValueError(
                    "Boundary conditions for rotations should be vectorial if nd == 3"
                )

        # Sanity check: If the Cosserat parameter is None, the boundary conditions for
        # the rotation variable are not relevant.
        if bnd_rot is not None and cosserat_values is None:
            warnings.warn(
                "Boundary conditions for rotations are only relevant if the Cosserat "
                "parameter is non-zero."
            )
        if bnd_rot is not None and np.sum(bnd_rot.is_rob) > 0:
            # The implementation should not be difficult, but has not been prioritized.
            raise NotImplementedError(
                "Robin conditions for rotations have not been implemented."
            )

        # Map the stiffness tensor to the face-wise ordering.
        mu = stiffness.mu[numbering.ci]
        if cosserat_values is not None:
            cosserat_parameter = cosserat_values[numbering.ci]

        # BoundaryConditionVectorial has an attribute 'basis', to be used with Robin
        # boundary conditions (see mpsa.py). This is not implemented for tpsa.
        if np.logical_or.reduce(
            (
                np.any(bnd_disp.basis[0, 1:, :] > 0),
                np.any(bnd_disp.basis[1, 0, :] > 0),
                np.any(bnd_disp.basis[1, 2:, :] > 0),
                np.any(bnd_disp.basis[2:, :2, :] > 0),
                np.any(bnd_disp.basis[0, 0, :] != 1),
                np.any(bnd_disp.basis[1, 1, :] != 1),
            )
        ):
            raise NotImplementedError(
                "Have not implemented Robin conditions with a non-trivial basis."
            )
        if nd == 3 and np.any(bnd_disp.basis[2, 2] != 1):
            raise NotImplementedError(
                "Have not implemented Robin conditions with a non-trivial basis."
            )
        # The Robin weight will be an nd x nd x nf array, with nd={2, 3}. For the
        # implementation to be valid in both cases, we use slices ([2:]) instead of
        # indexing ([2]), as the former will work also if the array has less than three
        # rows.
        if np.logical_or.reduce(
            (
                np.any(bnd_disp.robin_weight[0, 1:, :] > 0),
                np.any(bnd_disp.robin_weight[1, 0, :] > 0),
                np.any(bnd_disp.robin_weight[1, 2:, :] > 0),
                np.any(bnd_disp.robin_weight[2:, :2, :] > 0),
            )
        ):
            raise NotImplementedError(
                "Non-diagonal Robin weights have not been implemnted."
            )

        ###
        # Done with processing of parameters.

        # Construct filters that are used to isolate faces with different types of
        # boundary conditions assigned. We do this only for the displacement variable.
        # For the rotation variable (in the case of a non-zero Cosserat parameter) we
        # only need to deal with Dirichlet and Neumann conditions on a diffusion problem
        # (Robin conditions have not been implemented, see above), which is relatively
        # easy and handled on the fly.
        filters = self._create_filters(bnd_disp, numbering, sd)

        # Compute various distance measures, and also the discretization coefficients
        # for Robin boundary conditions. The suffix shear indicates that the
        # coefficients relate to the shear modulus.
        dist, t_shear_rob = self._compute_distances(sd, numbering, mu, bnd_disp)

        # Construct various maps from cell to face quantities.
        c2f_maps = self._create_cell_to_face_maps(
            sd, numbering, filters, bnd_disp, dist
        )

        #################
        # Start construction of discretization matrices.

        # Compute t_shear (e.g., stress related to the shear modulus) as the sum of the
        # contribution from the interior faces (computed here) and the Robin boundary
        # conditions (represented in t_shear_rob, computed previously). Since the Robin
        # conditions are already summed up, we use mu_by_dist_fc_cc, not
        # mu_by_dist_fc_cc_bound. The operation results in an nd x nf array, where each
        # row contains the discretization coefficients for one dimension of all the
        # faces (internal and boundary, though the boundary term will be modified in the
        # subsequent call to _vector_laplace_matrices).
        t_shear_nd = (
            2  # The factor 2 in Hook's law.
            * np.repeat(sd.face_areas, nd)  # Scaling with face areas.
            / (  # Sum up the contributions from the cells sharing the face. There will
                # be two cotnributions on internal faces, and one on boundary faces.
                np.bincount(
                    numbering.fi_expanded,  # Use the nd version of the face indices.
                    # The discretization coefficients are the shear modulus divided by
                    # the distance. The coefficient is the same in all directions
                    # (isotropy is assumed/imposed, this is in the nature of the tpsa
                    # approximation), thus we repeat the coefficient nd times.
                    weights=1.0 / np.repeat(dist.mu_by_dist_fc_cc, nd),
                )
                + t_shear_rob  # Contribution from Robin boundary conditions.
            )
        ).reshape(
            (nd, nf), order="F"
        )  #

        # Discretize the stress-displacement relation.
        stress, bound_stress = self._vector_laplace_matrices(
            sd, t_shear_nd, bnd_disp, numbering, c2f_maps.b2f_rob_compl
        )

        # Face normals.
        n = sd.face_normals

        # Diagonal representation of the face normal vectors.
        normal_vector_diag = sps.dia_array(
            (n[:nd].ravel("F"), 0), shape=(nf * nd, nf * nd)
        )
        # The stress generated by the total pressure is computed using the complement of
        # the average map scaled with the normal vector. The latter also gives the
        # correct scaling with the face area. Explicitly filter Neumann boundary
        # conditions, these do not contribute.
        stress_total_pressure = (
            filters.neu_notpass_nd @ normal_vector_diag @ c2f_maps.c2f_compl_scalar_2_nd
        )

        # The solid mass conservation equation is discretized by taking the average
        # displacement over the faces and scaling with the normal vector. To that end,
        # construct a sparse matrix that has one normal vector per row.
        normal_vector_nd = sps.csr_matrix(
            (
                n[:nd].ravel("F"),
                np.arange(nf * nd),
                np.arange(0, nf * nd + 1, nd),
            ),
            shape=(nf, nf * nd),
        )
        # The impact on the solid mass flux from the displacement is the matrix of
        # normal vectors multiplied with the average displacement over the faces. This
        # matrix will be empty on Dirichlet faces due to the filtering in
        # cell_to_face_average_nd.
        mass_displacement = normal_vector_nd @ c2f_maps.c2f

        # While there is no spatial operator that relates the total pressure to the
        # conservation of solid mass in the continuous equation, the TPSA discretization
        # naturally leads to a stabilization term, as computed below. This acts on
        # differences in the total pressure, and is scaled with the face area. Since the
        # solid pressure is a derived boundary quantity for which there is no boundary
        # condition set, it is unclear what to actually do with boundary terms. The
        # current implementation seems to work.
        #
        # Arithmetic average of shear modulus. No contribution from boundary conditions,
        # thus do not use mu_by_dist_fc_cc_bound.
        arithmetic_average_shear_modulus = np.bincount(
            numbering.fi,
            weights=dist.mu_by_dist_fc_cc,
            minlength=nf,
        )
        # Following the paper, we filter away Dirichlet boundary conditions.
        mass_total_pressure = -filters.dir_notpass @ (
            sps.dia_matrix(
                (sd.face_areas / (2 * arithmetic_average_shear_modulus), 0),
                shape=(nf, nf),
            )
            @ sd.cell_faces
        )

        # Take the harmonic average of the Cosserat parameter. For zero Cosserat
        # parameters, this involves a division by zero. This gives no actual problem,
        # but filtering would have been more elegant.
        if cosserat_values is not None:
            t_cosserat = sd.face_areas / np.bincount(
                numbering.fi,
                weights=1 / (cosserat_parameter / dist.dist_fc_cc),
                minlength=nf,
            )

        # A rotation in 2d has a single degree of freedom, while a 3d rotation has 3
        # degrees of freedom. This necessitates (or at least is most easily realized) by
        # a split into a 2d and a 3d code. In the below implementation, we refer to two
        # rotation matrices Rn_hat and Rn_bar, which are identical (and equal to the
        # matrix R used to describe Tpsa in the paper), but differ in 2d where Rn_hat
        # takes scalars to 2d vectors, while Rn_bar maps 2d vectors back to scalars.
        # These matrices are constructed in the below if-else; moreover, some additional
        # operations that differ in 2d and 3d are undertaken, including discretization
        # of the diffusion operators for the displacement and (if relevant) rotations.
        if nd == 3:
            # In this case, \hat{R}_k^n = \bar{R}_k^n is the 3x3 matrix given in the
            # Tpsa paper,
            #
            #    R^n = [[0, -n2, n0], [n2, 0, -n0], [-n1, n0, 0]]
            #
            # However, for efficient implementation we will use a utility function for
            # matrix construction available in PorePy which, it turns out, requires a
            # transpose in the inner array. Quite likely this could have been achieved
            # by a different order of raveling (see below), but the current approach
            # works.
            #
            # For reference, it is possible to use the following code to construct R_hat
            #
            # Rn_data = np.array([[z, -n[2], n[1]], [n[2], z, -n[0]], [-n[1], n[0], z]])
            # Rn_hat = sps.block_diag([Rn_data[:, :, i] for i in range(Rn.shape[2])])
            #
            # but this is much slower due to the block_diag construction.

            z = np.zeros(nf)
            Rn_data = np.array([[z, n[2], -n[1]], [-n[2], z, n[0]], [n[1], -n[0], z]])

            Rn_hat = pp.matrix_operations.csr_matrix_from_blocks(
                Rn_data.ravel("F"), nd, nf
            )
            Rn_bar = Rn_hat

            # Discretization of the stress generated by cell center rotations. No
            # contribution from Neumann boundaries.
            stress_rotation = -filters.neu_notpass_nd @ Rn_hat @ c2f_maps.c2f_compl

            # We know that the boundary condition for the rotation variable is a
            # vectorial condition.
            bnd_rot = cast(pp.BoundaryConditionVectorial, bnd_rot)

            if cosserat_values is not None:
                # Use the discretization of the vector Laplace problem. The
                # transmissibility will be the same in all directions.
                rotation_diffusion, bound_rotation_diffusion = (
                    self._vector_laplace_matrices(
                        sd,
                        np.tile(t_cosserat, (nd, 1)),
                        bnd_rot,
                        numbering,
                        c2f_maps.b2f_rob_compl,
                    )
                )
            else:
                # If the Cosserat parameter is zero, the diffusion operator is zero.
                rotation_diffusion = sps.csr_matrix((nf * nd, nc * nd))
                bound_rotation_diffusion = sps.csr_matrix((nf * nd, nf * nd))

        elif nd == 2:
            # In this case, \hat{R}_k^n and \bar{R}_k^n differ, and read, respectively
            #   \hat{R}_k^n = [[n2], [-n1]],
            #   \bar{R}_k^n = [-n2, n1].

            # Vector of normal vectors.
            normal_vector_data = np.array([n[1], -n[0]])

            # Mapping from average displacements over faces to rotations on the face.
            # Minus sign from definition of Rn_bar.
            Rn_bar = sps.csr_matrix(
                (
                    -normal_vector_data.ravel("F"),
                    np.arange(nf * nd),
                    np.arange(0, nd * nf + 1, nd),
                ),
                shape=(nf, nf * nd),
            )
            # Mapping from average rotations over faces to stresses.
            Rn_hat = sps.dia_matrix(
                (normal_vector_data.ravel("F"), 0), shape=(nf * nd, nf * nd)
            )
            # Discretization of the stress generated by cell center rotations.
            stress_rotation = (
                -filters.neu_notpass_nd @ Rn_hat @ c2f_maps.c2f_compl_scalar_2_nd
            )

            # Diffusion operator on the rotation if relevant.
            if cosserat_values is not None:
                # In 2d, the rotation is a scalar variable and we can treat this by
                # what is essentially a tpfa discretization.

                t_cosserat_bnd = np.zeros(nf)
                t_cosserat_bnd[bnd_rot.is_dir] = t_cosserat[bnd_rot.is_dir]
                # The boundary condition should simply be imposed.
                t_cosserat_bnd[bnd_rot.is_neu] = 1
                t_cosserat[bnd_rot.is_neu] = 0

                rotation_diffusion = -sps.coo_matrix(
                    (
                        t_cosserat[numbering.fi] * numbering.sgn,
                        (numbering.fi, numbering.ci),
                    ),
                    shape=(nf, nc),
                ).tocsr()

                bound_rotation_diffusion = sps.coo_matrix(
                    (
                        t_cosserat_bnd[numbering.fi] * numbering.sgn,
                        (numbering.fi, numbering.fi),
                    ),
                    shape=(nf, nf),
                ).tocsr()

            else:
                rotation_diffusion = sps.csr_matrix((nf, nc))
                bound_rotation_diffusion = sps.csr_matrix((nf, nf))

        # The rotation generated by the cell center displacements is computed from the
        # average displacement over the faces, multiplied by Rn_bar.
        rotation_displacement = -Rn_bar @ c2f_maps.c2f

        # This is the expression \delta_k^mu, valid for internal and boundary faces.
        # Note that the factor 1 / 2 is included in mu_by_dist_fc_cc_bound, see the
        # definition of that term for a discussion.
        inv_mu_face = sps.dia_matrix(
            (1.0 / dist.mu_by_dist_fc_cc_bound, 0), shape=(nf * nd, nf * nd)
        )
        # Boundary term for the rotation generated by displacement.
        bound_rotation_displacement = Rn_bar @ (
            filters.neu_rob_pass_nd @ inv_mu_face
            - filters.dir_pass_nd
            - c2f_maps.b2f_rob
        )

        # Boundary condition for the mass conservation equation.
        bound_mass_displacement = normal_vector_nd @ (
            filters.neu_rob_pass_nd @ inv_mu_face
            + filters.dir_pass_nd
            + c2f_maps.b2f_rob
        )

        # Fields related to reconstruction of displacements on the boundary. These are
        # not part of the main discretization, thus not explicitly described in the
        # paper, but can be considered derived quantities. The reconstruction depends on
        # the type of boundary condition assigned:
        # * For Dirichlet conditions, the displacement is known, and the reconstruction
        #   should fetch the boundary values.
        # * For Robin conditions, the displacement can be recovered from the boundary
        #   condition together with the discrete representations of the stress generated
        #   by cell center displacement, rotation, and solid pressure. The resulting
        #   expressions are consistent with the respective discretizations constructed
        #   above.
        # * For faces with Neumann conditions, the above discretizations do not provide
        #   discrete stresses, since the stress anyhow is imposed. For such faces, we
        #   compute stresses from the cell center variables and convert to
        #   displacements. This effectively implies that certain choices are made in the
        #   reconstruction, but it seems to be the approach most consistent with the
        #   overall discretization.

        # Common scaling.
        sgn_area_scaling = sps.dia_array(
            (np.repeat(numbering.sgn_bf / sd.face_areas, nd), 0),
            shape=(nd * nf, nd * nf),
        )

        # The cell displacement is used for both Robin and Neumann conditions. The
        # reconstruction can be seen as having two parts: The cell center value
        # (bound_displacement_cell) and the difference between face and cell, generated
        # by the different sources of stress and converted to a displacement through an
        # inversion of Hook's law (rotation and total pressure variables).
        bound_displacement_cell = filters.neu_rob_pass_nd @ c2f_maps.c2f

        # Contrbution from the face, that is the boundary condition. For Dirichlet
        # conditions this is a unit scaling. For Neumann and Robin, we invert the
        # imposed stress to a displacement via Hook's law (hence scaling by
        # inv_mu_face).
        bound_displacement_face = (
            filters.dir_pass_nd
            + sgn_area_scaling
            @ inv_mu_face
            @ (filters.neu_pass_nd + filters.rob_pass_nd @ c2f_maps.b2f_rob)
        )

        # The mapping from cell to face rotation is different in 2d and 3d. This is used
        # to construct a face rotation for Neumann faces (recall that these are simply
        # ignored in the construction of the rotational stress above).
        if nd == 2:
            face_rotation = c2f_maps.c2f_scalar_2_nd
        else:
            face_rotation = c2f_maps.c2f

        # Contributions from the cell center rotation.
        bound_displacement_rotation_cell = (
            sgn_area_scaling
            @ inv_mu_face
            @ (
                filters.rob_pass_nd @ stress_rotation
                - filters.neu_pass_nd @ Rn_hat @ face_rotation
            )
        )

        # Finally the contribution from the total pressure.
        bound_displacement_solid_pressure_cell = (
            sgn_area_scaling
            @ inv_mu_face
            @ (
                filters.rob_pass_nd @ stress_total_pressure
                + filters.neu_pass_nd @ normal_vector_diag @ c2f_maps.c2f_scalar_2_nd
            )
        )

        ## Store the computed fields.

        # Discretization matrices.
        matrix_dictionary[self.stress_displacement_matrix_key] = stress
        matrix_dictionary[self.stress_rotation_matrix_key] = stress_rotation
        matrix_dictionary[self.stress_total_pressure_matrix_key] = stress_total_pressure
        matrix_dictionary[self.rotation_displacement_matrix_key] = rotation_displacement
        matrix_dictionary[self.rotation_diffusion_matrix_key] = rotation_diffusion
        matrix_dictionary[self.mass_total_pressure_matrix_key] = mass_total_pressure
        matrix_dictionary[self.mass_displacement_matrix_key] = mass_displacement

        # Boundary conditions.
        matrix_dictionary[self.bound_stress_matrix_key] = bound_stress
        matrix_dictionary[self.bound_mass_displacement_matrix_key] = (
            bound_mass_displacement
        )
        matrix_dictionary[self.bound_rotation_diffusion_matrix_key] = (
            bound_rotation_diffusion
        )
        matrix_dictionary[self.bound_rotation_displacement_matrix_key] = (
            bound_rotation_displacement
        )

        matrix_dictionary[self.bound_displacement_cell_matrix_key] = (
            bound_displacement_cell
        )
        matrix_dictionary[self.bound_displacement_face_matrix_key] = (
            bound_displacement_face
        )
        matrix_dictionary[self.bound_displacement_rotation_cell_matrix_key] = (
            bound_displacement_rotation_cell
        )
        matrix_dictionary[self.bound_displacement_solid_pressure_cell_matrix_key] = (
            bound_displacement_solid_pressure_cell
        )

    @staticmethod
    def _create_filters(
        bnd_disp: pp.BoundaryConditionVectorial, numbering: _Numbering, sd: pp.Grid
    ) -> _BoundaryFilters:
        """Create filters to apply boundary conditions.

        We need only filters based on the displacement variable, since the rotation
        variable is handled directly in the discretization, see
        self._vector_laplace_matrices().

        Parameters:
            bnd_disp: Boundary conditions object for the displacement variable.
            numebring: Container of quantities needed for bookkeeping.
            sd: Grid.

        Returns:
            Container of filters for boundary conditions.

        """
        nf, nd = sd.num_faces, sd.dim

        is_dir = bnd_disp.is_dir.ravel("F")
        is_neu = bnd_disp.is_neu.ravel("F")
        is_rob = bnd_disp.is_rob.ravel("F")
        is_internal = np.logical_not(np.logical_or.reduce((is_dir, is_neu, is_rob)))

        dir_nd = sps.dia_matrix((is_dir.astype(int), 0), shape=(nf * nd, nf * nd))

        dir_notpass_nd = sps.dia_matrix(
            (np.logical_or.reduce((is_neu, is_rob, is_internal)).astype(int), 0),
            shape=(nf * nd, nf * nd),
        )
        neu_pass_nd = sps.dia_matrix((is_neu.astype(int), 0), shape=(nf * nd, nf * nd))

        neu_notpass_nd = sps.dia_matrix(
            (np.logical_or.reduce((is_dir, is_rob, is_internal)).astype(int), 0),
            shape=(nf * nd, nf * nd),
        )

        neu_rob_pass_nd = sps.dia_matrix(
            (np.logical_or(is_neu, is_rob).astype(int), 0), shape=(nf * nd, nf * nd)
        )
        rob_nd = sps.dia_array((is_rob.astype(int), 0), shape=(nf * nd, nf * nd))

        # We also need to deal with BCs on the numerical diffusion term for the solid
        # pressure. It is not fully clear what to do with this term on the boundary:
        # There is no boundary condition for the total pressure as this quantity is
        # derived from the displacement. The discretization scheme must however be
        # adjusted, so that it is zero on Dirichlet faces. The question is, what to do
        # with rolling boundary conditions, where a mixture of Dirichlet and Neumann
        # conditions are applied? For now, we pick the condition in the direction which
        # is closest to the normal vector of the face. While this should work nicely for
        # domains where the grid is aligned with the coordinate axis, it is more of a
        # question how this will work for rotated domains.
        max_ind = np.argmax(np.abs(sd.face_normals), axis=0)
        dir_scalar = bnd_disp.is_dir[max_ind, np.arange(nf)]
        dir_notpass = sps.dia_matrix(
            (np.logical_not(dir_scalar).astype(int), 0), shape=(nf, nf)
        )

        return _BoundaryFilters(
            dir_nd,
            dir_notpass,
            dir_notpass_nd,
            neu_pass_nd,
            neu_notpass_nd,
            neu_rob_pass_nd,
            rob_nd,
        )

    @staticmethod
    def _create_cell_to_face_maps(
        sd: pp.Grid,
        numbering: _Numbering,
        filters: _BoundaryFilters,
        bnd_disp: pp.BoundaryConditionVectorial,
        dist: _Distances,
    ):
        """Helper method to construct mappings from cells, and boundary conditions, to
        faces.

        Parameters:
            sd: Grid.
            numbering: Container of quantities needed for bookkeeping.
            filters: Necessary filters for imposing boundary conditions.
            bnd_disp: Boundary condition object for the displacement variable.
            dist: Container for distance measures.

        Returns:
            Container of maps from cells to faces.


        """
        nc, nf, nd = sd.num_cells, sd.num_faces, sd.dim

        # Handling of Dirichlet conditions.
        is_dir_nd = bnd_disp.is_dir
        is_dir = is_dir_nd.ravel("F")

        # Mapping from cell to face, with a weighting of mu / dist. Only interior faces.
        # Add a factor 2 to compensate for the factor 1/2 included through
        # multiplication by inv_mu_by_dist_array below; see the construction of
        # mu_by_dist_array for further comments.
        cell_to_face = sps.coo_array(
            ((2 * dist.mu_by_dist_fc_cc, (numbering.fi, numbering.ci))), shape=(nf, nc)
        ).tocsr()

        # Create the nd version, multiply with a scaling matrix to get an averaging map,
        # and filter away Dirichlet boundary conditions (these will be enforced
        # elsewhere in the code).
        c2f = (
            filters.dir_notpass_nd
            @ dist.inv_mu_by_dist_array
            @ sps.kron(cell_to_face, sps.eye(nd), format="csr")
        )
        # Complement map.
        c2f_compl = sps.csr_matrix(
            (1 - c2f.data, c2f.indices, c2f.indptr), shape=c2f.shape
        )

        # Create a mapping for Robin boundary values specifically (note the filter that
        # lets Robin conditions pass). Inspection of the part of the code where this map
        # is used will show that Dirichlet and Neumann conditions are treated separately
        # using filters (which are binary, we cannot do this with Robin since this is a
        # weighted map).
        b2f_rob = (
            filters.rob_pass_nd
            @ dist.inv_mu_by_dist_array
            @ sps.dia_array((dist.rob_weight.ravel("F"), 0), shape=(nf * nd, nf * nd))
        )
        # For the complement, we only need the diagnoal data.
        b2f_rob_compl = 1 - b2f_rob.diagonal()

        # Map from scalar cell quantities to nd face quantities (used e.g. for the solid
        # pressure).
        c2f_scalar_2_nd = dist.inv_mu_by_dist_array @ sps.kron(
            cell_to_face, sps.csr_array(np.ones((nd, 1))), format="csr"
        )
        # Specifically set a zero value for faces that have Dirichlet conditions, as
        # these will draw their values from the boundary condition. While this could
        # have been realized by multiplication with filters.dir_notpass_nd, this would
        # have left implicit (not represented) zeros at the boundaries, whereas we need
        # explicit zeros to construct the complement map below. Hence we enforce the
        # zeros by manipulating the data array.
        c2f_rows, *_ = sps.find(c2f_scalar_2_nd)
        c2f_rows_is_dir = np.isin(c2f_rows, np.where(is_dir))
        c2f_scalar_2_nd.data[c2f_rows_is_dir] = 0
        # Complement map.
        c2f_compl_scalar_2_nd = sps.csr_array(
            (
                1 - c2f_scalar_2_nd.data,
                c2f_scalar_2_nd.indices,
                c2f_scalar_2_nd.indptr,
            ),
            shape=c2f_scalar_2_nd.shape,
        )

        mappings = _CellToFaceMaps(
            c2f,
            c2f_compl,
            c2f_scalar_2_nd,
            c2f_compl_scalar_2_nd,
            b2f_rob,
            b2f_rob_compl,
        )
        return mappings

    @staticmethod
    def _compute_distances(
        sd: pp.Grid,
        numbering: _Numbering,
        mu: np.ndarray,
        bnd_disp: pp.BoundaryConditionVectorial,
    ):
        """Compute grid-related distance measures, including the distance weighted
        shear modulus.

        This is also where the discretization coefficients related to Robin boundary
        conditions are computed.

        Parameters:
            sd: Grid.
            numbering: Related to grid counting.
            mu: The shear modulus.
            bnd_disp: Boundary condition for the displacement variable.

        Returns:
            Tuple of two objects:

            _Dist:
                Various distance measures needed in the discretization.

            np.ndarray:
                Discretization coefficients for Robin boundary conditions.

        """
        nf, nd = sd.num_faces, sd.dim
        # Normal vectors in the face-wise ordering.
        n_fi = sd.face_normals[:, numbering.fi]
        # Switch signs where relevant.
        n_fi *= numbering.sgn

        # Get a vector from cell center to face center and project to the direction of
        # the face normal. Divide by the face area to get a unit vector in the normal
        # direction.
        fc_cc = (
            n_fi
            * (sd.face_centers[::, numbering.fi] - sd.cell_centers[::, numbering.ci])
            / sd.face_areas[numbering.fi]
        )
        # Get the length of the projected vector; take the absolute value to avoid
        # negative distances.
        dist_fc_cc = np.abs(np.sum(fc_cc, axis=0))

        # Construct mu_i / delta_k^i, and its nd version.
        mu_by_dist_fc_cc = mu / dist_fc_cc
        mu_by_dist_fc_cc_nd = np.repeat(mu_by_dist_fc_cc, nd)

        # Extract the diagonal of the Robin weight. Non-diagonal elements are ignored
        # here and specifically ruled out (with error messages) elsewhere in the code.
        rob_weight = np.vstack(
            (bnd_disp.robin_weight[0, 0], bnd_disp.robin_weight[1, 1])
        )
        if nd == 3:
            rob_weight = np.vstack((rob_weight, bnd_disp.robin_weight[2, 2]))

        # Nd version of the faces with Robin boundary conditions.
        rob_boundary_faces_expanded = pp.fvutils.expand_indices_nd(
            np.arange(nf), nd
        ).reshape((nd, nf), order="F")[bnd_disp.is_rob]
        rob_weights_boundary_faces = rob_weight[bnd_disp.is_rob]

        # This is the face-wise sum of the expressions mu/delta, also accounting for
        # boundary conditions. EK note to self: The factor 2 is motivated by Robin
        # boundary conditions: With the Robin weight represented, PorePy style, as
        # \alpha (rather than 2*\mu*\delta^-1 as is used in the paper), the averaging
        # operator for a face with a Robin condition becomes
        # (2*\mu*\delta^-1)/(2*\mu*\delta^-1 + \alpha) for the cell contribution; the
        # contribution from the boundary condition has \alpha in the nominator. The
        # factor 2 introduced here, and inherited in the diagonal scaling matrix
        # constructed next, must be compensated by a factor 2 in the nominator of the
        # averaging map, as is done elsewhere in the code.
        #
        # For reference, the reciprocal of this field is also the expression \delta_k^mu
        # (used in the paper to describe the discretization scheme).
        mu_by_dist_fc_cc_bound = np.bincount(
            np.hstack((numbering.fi_expanded, rob_boundary_faces_expanded)),
            weights=np.hstack((2 * mu_by_dist_fc_cc_nd, rob_weights_boundary_faces)),
        )

        # Create a diagonal matrix that can be used to scale face-to-cell maps to have
        # unit row sum (thus they become true averaging maps), both in the interior and
        # on faces with Robin boundary conditions.
        inv_mu_by_dist_array = sps.dia_array(
            (1 / mu_by_dist_fc_cc_bound, 0), shape=(nf * nd, nf * nd)
        )

        # Finally, since we have obtained the Robin weights and the associated indices
        # of the faces, we might as well compute the discretization coefficients for
        # this boundary condition here.
        t_shear_rob = np.bincount(
            rob_boundary_faces_expanded,
            weights=1.0 / rob_weights_boundary_faces,
            minlength=nd * nf,
        )

        dist = _Distances(
            dist_fc_cc,
            mu_by_dist_fc_cc,
            mu_by_dist_fc_cc_bound,
            inv_mu_by_dist_array,
            rob_weight,
        )

        return dist, t_shear_rob

    @staticmethod
    def _vector_laplace_matrices(
        sd: pp.Grid,
        trm_nd: np.ndarray,
        bnd: pp.BoundaryConditionVectorial,
        numbering: _Numbering,
        b2f_rob_compl: np.ndarray,
    ) -> tuple[sps.spmatrix, sps.spmatrix]:
        """Discritize the vector Laplacian by a two-point approximation.

        Parameters:
            sd: Grid.
            trm_nd: Transmissibility coefficients, as computed from the harmonic average
                of the shear modulus.
            bnd: Boundary conditions.
            numbering: Bookkeeping structure.
            b2f_rob_compl: Complement of the Robin boundary conditions.

        Returns:
            Tuple of two objects:

            sps.sparray:
                Discretization matrix for the vector Laplacian.

            sps.sparray:
                Discretization matrix for the boundary condition.

        """
        # The linear stress due to cell center displacements is computed from the
        # harmonic average of the shear modulus, scaled by the face areas. The
        # transmissibility is the same for each dimension, implying that the material is
        # in a sense isotropic.

        # Bookkeeping.
        nc, nf, nd = sd.num_cells, sd.num_faces, sd.dim

        # Get the types of boundary conditions.
        dir_faces = bnd.is_dir
        neu_faces = bnd.is_neu
        rob_faces = bnd.is_rob

        # Data structure for the discretization of the boundary conditions.
        trm_bnd = np.zeros((nd, nf))
        # On Dirichlet faces, the discretization coefficient of the boundary condition
        # is the same as that of the adjacent cell. The sign of the coefficient are
        # different for the cell and the boundary condition, this is introduced in the
        # construction of the discretization matrix.
        trm_bnd[dir_faces] = trm_nd[dir_faces]

        # On Neumann faces, the coefficient of the discretization itself is zero, as the
        # 'flux' through the boundary face is given by the boundary condition.
        trm_nd[neu_faces] = 0

        # The boundary condition should simply be imposed.
        # IMPLEMENTATION NOTE: Contrary to the tpfa implementation, the coefficients of
        # Neumann boundary conditions in tpsa are not multiplied with the sign of the
        # normal vector. This reflects that Neumann boundary values for mechanics are
        # set in terms of global coordinate directions, while for the flow/scalar
        # problem, the conditions are set with respect to the face-wise normal vector.
        trm_bnd[neu_faces] = 1

        trm_bnd[rob_faces] = (
            b2f_rob_compl.reshape((nd, nf), order="F")[rob_faces] + trm_nd[rob_faces]
        )

        # Discretization of the vector Laplacian. Regarding indexing, the ravel gives a
        # vector-sized array in linear ordering, which is shuffled to the (vector
        # version of the) face-wise ordering. The sign is set so that the stress is
        # positive in tension.
        discr = -sps.coo_matrix(
            (
                trm_nd.ravel("F")[numbering.fi_expanded] * numbering.sgn_nd,
                (numbering.fi_expanded, numbering.ci_expanded),
            ),
            shape=(nf * nd, nc * nd),
        ).tocsr()

        # Boundary condition.
        bound_discr = sps.coo_matrix(
            (
                trm_bnd.ravel("F")[numbering.fi_expanded] * numbering.sgn_nd,
                (numbering.fi_expanded, numbering.fi_expanded),
            ),
            shape=(nf * nd, nf * nd),
        ).tocsr()
        return discr, bound_discr

    @staticmethod
    def _create_numbering(sd: pp.Grid):
        """Helper method to generate a container of quantities needed for bookkeeping.

        Parameters:
            sd: Grid.

        Returns:
            A container of quantities needed for bookkeeping.

        """
        # Bookkeeping.
        nf, nd = sd.num_faces, sd.dim

        # The discretization matrices give generalized fluxes across faces in terms of
        # variables in the centers of adjacent cells. The discretization is based on a
        # two-point scheme, thus we need a mapping between cells and faces. The below
        # code generates a triplet of (face index, cell index, sign), where the sign
        # indicates the orientation of the face normal. Internal faces will occur twice,
        # with two different cell indices and opposite signs. Boundary faces will occur
        # only once.
        fi, ci, sgn = sparse_array_to_row_col_data(sd.cell_faces)

        # Expand face and cell indices to construct nd discretization matrices.
        fi_expanded = fvutils.expand_indices_nd(fi, nd)
        ci_expanded = fvutils.expand_indices_nd(ci, nd)
        # For vector quantities, we need fi repeated nd times, do this once and for all
        # here.
        sgn_nd = np.repeat(sgn, nd)

        bf = sd.get_all_boundary_faces()
        sgn_bf, _ = sd.signs_and_cells_of_boundary_faces(bf)
        sgn_vec = np.zeros(nf, dtype=int)
        sgn_vec[bf] = sgn_bf

        return _Numbering(fi, ci, sgn, fi_expanded, ci_expanded, sgn_nd, sgn_vec)
