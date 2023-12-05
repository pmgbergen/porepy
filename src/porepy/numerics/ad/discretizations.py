"""
For any discretization class compatible with PorePy, wrap_discretization associates
a discretization with all attributes of the class' attributes that end with
'_matrix_key'.


Example:
    # Generate grid
    >>> g = pp.CartGrid([2, 2])
    # Associate an Ad representation of an Mpfa method, aimed this grid
    >>> discr = MpfaAd(keyword='flow', grids=[g])
    # The flux discretization of Mpfa can now be accesed by
    >>> discr.flux
    # While the discretization of boundary conditions is available by
    >>> discr.bound_flux.

    The representation of different discretization objects can be combined with other
    Ad objects into an operator tree, using lazy evaluation.

    It is assumed that the actual action of discretization (creation of the
    discretization matrices) is performed before the operator tree is parsed.
"""
from __future__ import annotations

import abc
from typing import Callable, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.utils.porepy_types import discretization_type

from ._ad_utils import MergedOperator, wrap_discretization

__all__ = [
    "Discretization",
    "BiotAd",
    "MpsaAd",
    "GradPAd",
    "DivUAd",
    "BiotStabilizationAd",
    "MpfaAd",
    "TpfaAd",
    "UpwindAd",
    "UpwindCouplingAd",
    "DifferentiableFVAd",
]


class Discretization(abc.ABC):
    """General/utility methods for AD discretization classes.

    The init of the children classes below typically calls wrap_discretization
    and has arguments including subdomains or interfaces and keywords for parameter and
    possibly matrix storage.

    """

    def __init__(self) -> None:
        self._discretization: discretization_type
        """The discretization object, which is wrapped by this class."""
        self.mat_dict_key: str
        """Keyword for matrix storage in the data dictionary."""
        self.keyword: str
        """Keyword for parameter storage in the data dictionary."""

        self.subdomains: list[pp.Grid]
        """List of grids on which the discretization is defined."""
        self.interfaces: list[pp.MortarGrid]
        """List of interfaces on which the discretization is defined."""

        self._name: str

    def __repr__(self) -> str:
        s = f"""
        Ad discretization of type {self._name}. Defined on {len(self.subdomains)}
        subdomains and {len(self.interfaces)} interfaces.
        """
        return s

    def __str__(self) -> str:
        return f"{self._name}({self.keyword})"

    @property
    def name(self) -> str:
        """Name of the discretization."""
        return self._name


# Mechanics related discretizations


class BiotAd(Discretization):
    """Ad wrapper around the Biot discretization class.

    For description of the method, we refer to the standard Biot class.

    """

    def __init__(
        self, keyword: str, subdomains: list[pp.Grid], flow_keyword: str = "flow"
    ) -> None:
        self.subdomains = subdomains
        self._discretization = pp.Biot(keyword, flow_keyword)
        self._name = "BiotMpsa"

        self.keyword = keyword

        # Declare attributes, these will be initialized by the below call to the
        # discretization wrapper.

        self.stress: MergedOperator
        self.bound_stress: MergedOperator
        self.bound_displacement_cell: MergedOperator
        self.bound_displacement_face: MergedOperator

        self.div_u: MergedOperator
        self.bound_div_u: MergedOperator
        self.grad_p: MergedOperator
        self.stabilization: MergedOperator
        self.bound_pressure: MergedOperator

        wrap_discretization(
            obj=self,
            discr=self._discretization,
            subdomains=subdomains,
            mat_dict_key=self.keyword,
        )


class MpsaAd(Discretization):
    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = pp.Mpsa(keyword)
        self._name = "Mpsa"

        self.keyword = keyword

        # Declare attributes, these will be initialized by the below call to the
        # discretization wrapper.

        self.stress: MergedOperator
        self.bound_stress: MergedOperator
        self.bound_displacement_cell: MergedOperator
        self.bound_displacement_face: MergedOperator

        wrap_discretization(self, self._discretization, subdomains=subdomains)


class GradPAd(Discretization):
    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = pp.GradP(keyword)
        self._name = "GradP from Biot"
        self.keyword = keyword

        self.grad_p: MergedOperator

        wrap_discretization(self, self._discretization, subdomains=subdomains)


class DivUAd(Discretization):
    def __init__(
        self, keyword: str, subdomains: list[pp.Grid], mat_dict_keyword: str
    ) -> None:
        self.subdomains = subdomains
        self._discretization = pp.DivU(keyword, mat_dict_keyword)

        self._name = "DivU from Biot"
        self.keyword = mat_dict_keyword

        self.div_u: MergedOperator
        self.bound_div_u: MergedOperator

        wrap_discretization(
            self,
            self._discretization,
            subdomains=subdomains,
            mat_dict_key=mat_dict_keyword,
        )


class BiotStabilizationAd(Discretization):
    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = pp.BiotStabilization(keyword)
        self._name = "Biot stabilization term"
        self.keyword = keyword

        self.stabilization: MergedOperator

        wrap_discretization(self, self._discretization, subdomains=subdomains)


## Flow related


class MpfaAd(Discretization):
    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = pp.Mpfa(keyword)
        self._name = "Mpfa"
        self.keyword = keyword

        self.flux: MergedOperator
        self.bound_flux: MergedOperator
        self.bound_pressure_cell: MergedOperator
        self.bound_pressure_face: MergedOperator
        self.vector_source: MergedOperator
        self.bound_pressure_vector_source: MergedOperator

        wrap_discretization(self, self._discretization, subdomains=subdomains)


class TpfaAd(Discretization):
    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = pp.Tpfa(keyword)
        self._name = "Tpfa"
        self.keyword = keyword

        self.flux: MergedOperator
        self.bound_flux: MergedOperator
        self.bound_pressure_cell: MergedOperator
        self.bound_pressure_face: MergedOperator
        self.vector_source: MergedOperator
        self.bound_pressure_vector_source: MergedOperator

        wrap_discretization(self, self._discretization, subdomains=subdomains)


class UpwindAd(Discretization):
    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = pp.Upwind(keyword)
        self._name = "Upwind"
        self.keyword = keyword

        self.upwind: MergedOperator
        self.bound_transport_dir: MergedOperator
        self.bound_transport_neu: MergedOperator
        wrap_discretization(self, self._discretization, subdomains=subdomains)


class UpwindCouplingAd(Discretization):
    def __init__(self, keyword: str, interfaces: list[pp.MortarGrid]) -> None:
        self.interfaces = interfaces
        self._discretization = pp.UpwindCoupling(keyword)
        self._name = "Upwind coupling"
        self.keyword = keyword

        # UpwindCoupling also has discretization matrices for (inverse) trace.
        # These are not needed for Ad version since ad.Trace should be used instead
        self.mortar_discr: MergedOperator
        self.flux: MergedOperator
        self.upwind_primary: MergedOperator
        self.upwind_secondary: MergedOperator
        wrap_discretization(self, self._discretization, interfaces=interfaces)

    def __repr__(self) -> str:
        s = (
            f"Ad discretization of type {self._name}."
            f"Defined on {len(self.interfaces)} interfaces."
        )
        return s


class DifferentiableFVAd:
    r"""This class represents the application of the product and chain rule of the
    flux expression

    .. math:: q = T(k(u)) * p.

    Here, the transmissibility matrix T is a function of the cell permeability k, which
    again is a function of a primary variable u, while p is the potential (pressure).
    The chain rule applied to this expression reads

    .. math:: dq = p * dT/dk * dk/du * du + T * dp

    The transmissibility matrix can be computed from a Tpfa or Mpfa discretization, or
    (in principle) any other finite volume method. The derivative of the
    transmissibilities, dT/dk, is approximated with a two-point flux stencil.

    If vector sources are included, p should be replaced by (p - dist * vector_source),
    with dist the distance roughly corresponding to the inverse of the gradient.

    """

    def __init__(
        self,
        subdomains: list[pp.Grid],
        mdg: pp.MixedDimensionalGrid,
        base_discr: Union[pp.ad.MpfaAd, pp.ad.TpfaAd],
        equation_system: pp.ad.EquationSystem,
        permeability_function: Callable[[list[pp.Grid]], pp.ad.Operator],
        permeability_argument: pp.ad.Variable,
        potential: pp.ad.Variable,
        keyword: str,
    ) -> None:
        self.subdomains = subdomains
        """List of subdomains on which the discretization is defined."""
        self.mdg = mdg
        """Mixed-dimensional grid."""
        self._discretization = base_discr
        """Discretization used to compute the transmissibility matrix."""
        self.equation_system = equation_system
        """Equation system used to evaluate Ad operators."""
        self.keyword = keyword
        """Keyword used to identify the discretization in the data dictionary."""
        self._subdomain_projections = pp.ad.SubdomainProjections(self.subdomains)
        """Subdomain projections used between full set of subdomains and individual ones."""
        self._perm_function = permeability_function
        """Function returning permeability as an ad.DenseArray given the perm_argument."""
        self._perm_argument: pp.ad.Variable = permeability_argument
        """pp.ad.Variable representing the variable upon which perm_function depends."""
        self._potential: pp.ad.Variable = potential
        """pp.ad.Variable representation of potential for the flux law
        :math:`flux=-K\nabla(potential - vector_source)`."""

    def flux(
        self,
    ) -> pp.ad.Operator:
        """Flux from potential, BCs and vector sources.

        Returns:
            Operator representing the flux.

        """
        ad_func = pp.ad.Function(self._flux_function, "differentiated_flux")
        return ad_func(self._potential)

    def bound_pressure_face(
        self,
    ) -> pp.ad.Operator:
        """Boundary face contribution to pressure reconstruction.

        Returns:
            pp.ad.Operator representing the value and jacobian of bound_pressure_face,
            which is used to reconstruct the pressure trace on the boundary (see FV
            discretizations).

        """
        ad_func = pp.ad.Function(
            self._bound_pressure_face_function, "differentiated_bound_pressure_face"
        )
        return ad_func(self._perm_argument)

    def _flux_function(
        self,
        potential: pp.ad.AdArray,
    ) -> pp.ad.AdArray:
        """Flux from potential, BCs and vector sources.

        Parameters:
            perm_function: pp.ad.Function returning permeability given a list of
                subdomains.
            perm_argument: ad.DenseArray representing the variable upon which perm_function
                depends.
            potential: Potential for the flux law
                :math:`flux=-K\nabla(potential - vector_source)`.

        Returns:
            ad.DenseArray representing the flux.

        """
        # Implementation note:
        # When this function is called, potential should be an Ad operator (say, a
        # MixedDimensionalVariable representation of the pressure). During evaluation,
        # because of the way operator trees are evaluated, potential will be an
        # ad.DenseArray (it is closer to being an atomic variable, thus it will be
        # evaluated before this function).

        # The product rule applied to q = T(k(u)) * p gives
        #   dT/dk * dk/du * p + T * dp/dp.
        # The first part is rather involved and is handled inside self._transmissibility

        # Get hold of the underlying flux discretization.
        base_flux = self._discretization.flux.value(self.equation_system)
        assert isinstance(base_flux, sps.spmatrix)

        # The Jacobian matrix should have the same size as the base.
        flux_jac = sps.csr_matrix((base_flux.shape[0], self.equation_system.num_dofs()))

        # The differentiation of transmissibilities with respect to permeability is
        # implemented as a loop over all subdomains. It could be possible to gather all
        # grid information in arrays as a preprocessing step, but that seems not to be
        # worth the effort. However, to avoid evaluating the permeability function
        # multiple times, we precompute it here
        global_permeability = self._perm_function(self.subdomains).value_and_jacobian(
            self.equation_system
        )
        sd: pp.Grid
        for sd in self.subdomains:
            transmissibility_jac, _ = self._transmissibility(sd, global_permeability)

            params = self.mdg.subdomain_data(sd)[pp.PARAMETERS][self.keyword]
            # Potential for this grid
            cells_of_grid = self._subdomain_projections.cell_restriction([sd]).value(
                self.equation_system
            )
            potential_value = cells_of_grid * potential.val

            # Create matrix and multiply into Jacobian
            grad_p = sps.diags(
                pp.fvutils.scalar_divergence(sd).T * potential_value,
                shape=(sd.num_faces, sd.num_faces),
            )
            # One half of the product rule applied to (T grad p). The other half
            # is base_flux * potential.jac as added below this loop.
            grad_p_jac = grad_p * transmissibility_jac

            # Boundary conditions
            # Dirichlet values are weighted with the transmissibility on the
            # boundary face.

            # Create a copy of the Jacobian at the Dirichlet faces
            is_dir = params["bc"].is_dir
            is_neu = params["bc"].is_neu

            # See tpfa discretization for the following treatment of values
            sort_id = np.argsort(sd.cell_faces[is_dir, :].indices)
            bndr_sgn = (sd.cell_faces[is_dir, :]).data[sort_id]
            bc_values = np.zeros(sd.num_faces)
            # Sign of this term:
            # bndr_sgn is the gradient operation at the boundary.
            # The minus ensues from moving the term from rhs to lhs
            # Note: If you get confused by comparison to tpfa (where
            # the term is negative on the rhs), confer
            # fv_elliptic.assemble_rhs, where the sign is inverted,
            # and accept my condolences. IS
            bc_values[is_dir] = -bndr_sgn * params["bc_values"][is_dir]
            bc_value_mat = sps.diags(bc_values, shape=(sd.num_faces, sd.num_faces))

            # Dirichlet face contribution from boundary values:
            jac_bound = bc_value_mat * transmissibility_jac

            # Eliminate values from the grad_p Jacobian product on Neumann boundaries.
            pp.matrix_operations.zero_rows(grad_p_jac, np.where(is_neu)[0])

            # Prolong this Jacobian to the full set of faces and add.
            face_prolongation = self._subdomain_projections.face_prolongation(
                [sd]
            ).value(self.equation_system)
            flux_jac += face_prolongation * grad_p_jac
            flux_jac += face_prolongation * jac_bound

            if "vector_source" in params and sd.dim > 0:
                fi, ci, sgn, fc_cc = self._geometry_information(sd)
                vector_source = params["vector_source"]
                vector_source_dim = params.get("ambient_dimension", sd.dim)
                vals = (fc_cc * sgn)[:vector_source_dim].ravel("F")
                rows = np.tile(fi, (vector_source_dim, 1)).ravel("F")
                cols = pp.fvutils.expand_indices_nd(ci, vector_source_dim)
                vector_source_val = (
                    sps.coo_matrix((vals, (rows, cols))).tocsr() * vector_source
                )
                vector_source_mat = sps.diags(
                    vector_source_val,
                    shape=(sd.num_faces, sd.num_faces),
                )
                vector_source_jac = vector_source_mat * transmissibility_jac
                pp.matrix_operations.zero_rows(
                    vector_source_jac, np.where(params["bc"].is_neu)[0]
                )
                flux_jac += face_prolongation * vector_source_jac

        # Second part of product rule, applied to the potential. This is the standard
        # part of a Mpfa or Tpfa discretization
        flux_jac += base_flux * potential.jac

        # The value of the flux is the standard mpfa/tpfa expression.
        block_val = base_flux * potential.val

        flux = pp.ad.AdArray(block_val, flux_jac)
        return flux

    def _bound_pressure_face_function(
        self,
        perm_argument: pp.ad.AdArray,
    ) -> pp.ad.AdArray:
        """The actual implementation of the bound_pressure_face function.

        Returns:
            The flux, q, and its Jacobian matrix, where the latter accounts for
            dependencies in the transmissibilities on cell center permeabilities.

        """
        # Note on parameters: When this function is called, potential should be an Ad
        # operator (say, a MixedDimensionalVariable representation of the pressure).
        # During evaluation, because of the way operator trees are evaluated, potential
        # will be an ad.DenseArray (it is closer to being an atomic variable, thus it will be
        # evaluated before this function).

        # The product rule applied to q = T(k(u)) * p gives
        #   dT/dk * dk/du * p + T * dp/dp.
        # The first part is rather involved.

        # Get hold of the underlying flux discretization.
        base_bound_pressure_face = self._discretization.bound_pressure_face.value(
            self.equation_system
        )
        # NOTE: This assertion will most likely break, the expected type should be sps.spmatrix.
        # While the assertion could have been changed, this would lead to an error further below.
        # This reflects that the current implementation is incomplete, thus the code should only
        # be used with extreme caution. A reimplementation of the method with differentiable
        # Tpfa is under way, thus fixing the current implementation has not been prioritized.
        assert isinstance(base_bound_pressure_face, np.ndarray)
        # The Jacobian matrix should have the same size as the base.
        bound_pressure_face_jac = sps.csr_matrix(
            (base_bound_pressure_face.shape[0], self.equation_system.num_dofs())
        )

        projections = pp.ad.SubdomainProjections(self.subdomains)

        # The differentiation of transmissibilities with respect to permeability is
        # implemented as a loop over all subdomains. It could be possible to gather all
        # grid information in arrays as a preprocessing step, but that seems not to be
        # worth the effort. However, to avoid evaluating the permeability function
        # multiple times, we precompute it here
        global_permeability = self._perm_function(self.subdomains).value_and_jacobian(
            self.equation_system
        )

        sd: pp.Grid
        for sd in self.subdomains:
            params: dict = self.mdg.subdomain_data(sd)[pp.PARAMETERS][self.keyword]
            transmissibility_jac, inverse_sum_squared = self._transmissibility(
                sd, global_permeability
            )

            # On Dirichlet faces, the tpfa discretization simply recovers boundary
            # condition. Thus, no t-differentiation enters. On Neumann faces, however,
            # the tpfa contribution is v_face[bnd.is_neu] = -1 / t_full[bnd.is_neu]
            # Differentiate and multiply with transmissibility jacobian
            v_face = np.zeros(sd.num_faces)
            is_neu = params["bc"].is_neu

            v_face[is_neu] = inverse_sum_squared[is_neu]
            d_face2boundp_d_t = sps.diags(v_face, shape=(sd.num_faces, sd.num_faces))
            jac_bound_pressure_face = d_face2boundp_d_t * transmissibility_jac

            # Prolong this Jacobian to the full set of faces and add.
            face_prolongation = projections.face_prolongation([sd]).value(
                self.equation_system
            )
            bound_pressure_face_jac += face_prolongation * jac_bound_pressure_face

        bound_pressure = pp.ad.AdArray(
            base_bound_pressure_face, bound_pressure_face_jac
        )
        return bound_pressure

    def _geometry_information(
        self, g: pp.Grid
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Utility function to retrieve geometry information.

        Parameters:
            g: pp.Grid for which the information is extracted.

        Returns:
            Arrays representing face indices, cell indices, normal vector signs and
            vectors between face centers and cell centers. The former three have size
                    n = num_face_cell_pairs = g.num_face * 2 (two cells per face),
            while fc_cc.shape = (3, n).
        """
        fi, ci, sgn = sps.find(g.cell_faces)

        # Distance from face center to cell center
        fc_cc = g.face_centers[::, fi] - g.cell_centers[::, ci]
        return fi, ci, sgn, fc_cc

    def _transmissibility(
        self, g: pp.Grid, global_permeability: pp.ad.AdArray
    ) -> Union[sps.csr_matrix, np.ndarray]:
        """Compute Jacobian of a variable transmissibility.

        Parameters:
            g: pp.Grid for which we compute transmissibility
            global_permeability: Array representing the global (in mixed-dimensional
                sense) permeability

        Returns:
            Jacobian of the transmissibility for this grid.

        """
        # The first few lines are pasted from the standard Tpfa implementation
        fi, ci, sgn, fc_cc = self._geometry_information(g)
        sz = fi.size

        # Switch signs of face normals where relevant
        n = g.face_normals[:, fi]
        n *= sgn
        # This is really the expression n * K * dist(x_face, x_cell), but since we
        # assume the permeability is isotropic, we deal with this below.
        n_dist = n * fc_cc
        dist_face_cell = np.power(fc_cc, 2).sum(axis=0)

        # From here on, the code is specific for transmissibility differentiation.

        # The chain rule applied to T(k(u)) * p (where the k(u) dependency can be replaced
        # by other primary variables - Ad should take care of this) gives
        #
        #    dT/du * p + T dp =  * p + T dp
        #
        # Here, dT/du is in reality a third-order tensor, which we have represented
        # as a matrix (assuming isotropic permeability). For simplicity of
        # implementation, we move the gradient from dT/du to the p term. This
        # implies as far as EK/IS have deduced that each line of dT/du
        # (# faces x # global dofs) should be multiplied with the gradient at the
        # corresponding face.
        # The chain rule gives
        #
        #   dT/du = dT/dt dt/dk dk/du.
        #
        # k being an ad function, dk/du is available as .jac after evaluation.
        # dT/dt and dt/dk are computed below. The former requires differentiating T
        # w.r.t. half transmissibilities and collection of contributions from each
        # cell. For i in {left, right} cells, it reads
        #
        #   dT_face/dt_i = T_face ** 2 / t_i ** 2,
        #
        # where the tpfa face transmissibility is
        #
        #   T_face = 1 / (1/t_l + 1/t_r).
        #
        # For isotropic tpfa, dt/dk is areas (-weighted normal) divided by cell-face
        # distances.

        # Evaluate the permeability as a function of the current potential
        # The evaluation means we go from an Ad operator formulation to the forward
        # mode, working with ad.DenseArrays. We map the computed permeability to the
        # faces (distinguishing between the left and right sides of the face).
        cell_2_one_sided_face = sps.coo_matrix(
            (np.ones(sz), (np.arange(sz), ci)),
            shape=(sz, g.num_cells),
        ).tocsr()

        # Restrict the permeability to the current grid (if perm_function is
        #  specific to the grid, the restriction should be applied to the potential
        # rather than the permeability).
        # The evaluation means we go from an Ad operator formulation to the forward
        # mode, working with ad.DenseArrays.
        # Then, map the computed permeability to the faces (distinguishing between
        # the left and right sides of the face).
        cells_of_grid = self._subdomain_projections.cell_restriction([g]).value(
            self.equation_system
        )
        k_one_sided = cell_2_one_sided_face * cells_of_grid * global_permeability

        # Multiply the permeability (and its derivatives with respect to potential,
        # since k_one_sided is an ad.DenseArray) with area weighted normal vectors
        # divided by distance
        normals_over_distance = np.divide(n_dist.sum(axis=0), dist_face_cell)
        t_one_sided = (
            sps.dia_matrix((normals_over_distance, 0), shape=(sz, sz)) * k_one_sided
        )

        # Mapping which sums the right and left sides of the face.
        # Unlike in normal tpfa, the sign of the normal vector is disregarded.
        # This is made up for when multiplying by grad p.
        sum_cell_face_pair_to_face = sps.coo_matrix(
            (np.ones(sz), (fi, np.arange(sz))), shape=(g.num_faces, sz)
        ).tocsr()

        # Compute the two factors of dT_face/dt_i (see definition and explanation
        # above).
        inverse_sum_squared = (sum_cell_face_pair_to_face * (1 / t_one_sided.val)) ** 2

        face_transmissibility_squared = sps.dia_matrix(
            (1 / inverse_sum_squared, 0), shape=(g.num_faces, g.num_faces)
        )
        hf_vals = np.power(t_one_sided.val, -2)
        half_face_transmissibility_inv_squared = sps.coo_matrix(
            (hf_vals, (fi, np.arange(sz))), shape=(g.num_faces, sz)
        ).tocsr()
        # Face transmissibility differentiated w.r.t. half face transmissibility
        d_transmissibility_d_t = (
            face_transmissibility_squared * half_face_transmissibility_inv_squared
        )
        # Face half face transmissibility differentiated w.r.t. permeability
        d_t_d_k = sps.dia_matrix((normals_over_distance, 0), shape=(sz, sz))

        # Compose chain rule for T(t(K(u)). k_one_sided.jac is dK/du.
        transmissibility_jac = d_transmissibility_d_t * d_t_d_k * k_one_sided.jac
        return transmissibility_jac, inverse_sum_squared
