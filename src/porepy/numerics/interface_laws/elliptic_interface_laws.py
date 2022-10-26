"""
Coupling conditions between subdomains for elliptic equations.

Current content:
    Robin-type couplings, as described by Martin et al. 2005.
    Full continuity conditions between subdomains

"""
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp
import porepy.numerics.interface_laws.abstract_interface_law
from porepy.numerics.interface_laws.abstract_interface_law import AbstractInterfaceLaw


class RobinCoupling(AbstractInterfaceLaw):
    """A condition with resistance to flow between subdomains. Implementation
    of the model studied (though not originally proposed) by Martin et
    al 2005. The equation reads
        \lambda = -\int{\kappa_n [p_l - p_h +  a/2 g \cdot n]} dV,
    where the last gravity type term is zero by default (controlled by the vector_source
    parameter). That is, \lambda is an extensive quantity.

    The class can be used either as pure discretization, or also to do assembly of
    the local (to the interface / mortar grid) linear system. The latter case will
    be invoked if a global linear system is constructed by way of the assembler.

    """

    def __init__(
        self,
        keyword: str,
        discr_primary: Optional["pp.EllipticDiscretization"] = None,
        discr_secondary: Optional["pp.EllipticDiscretization"] = None,
        primary_keyword: Optional[str] = None,
    ) -> None:
        """Initialize Robin Coupling.

        Parameters:
            keyword (str): Keyword used to access parameters needed for this
                discretization in the data dictionary. Will also define where
                 discretization matrices are stored.
            discr_primary: Discretization on the higher-dimensional neighbor. Only
                needed when the RobinCoupling is used for local assembly.
            discr_secondary: Discretization on the lower-dimensional neighbor. Only
                needed when the RobinCoupling is used for local assembly. If not
                provided, it is assumed that primary and secondary discretizations
                are identical.
            primary_keyword: Parameter keyword for the discretization on the higher-
                dimensional neighbor, which the RobinCoupling is intended coupled to.
                Only needed when the object is not used for local assembly (that is,
                needed if Ad is used).
        """
        super().__init__(keyword)

        if discr_secondary is None:
            discr_secondary = discr_primary

        if primary_keyword is not None:
            self._primary_keyword = primary_keyword
        else:
            if discr_primary is None:
                raise ValueError(
                    "Either primary keyword or primary discretization must be specified"
                )
            else:
                self._primary_keyword = discr_primary.keyword

        self.discr_primary = discr_primary
        self.discr_secondary = discr_secondary

        # This interface law will have direct interface coupling to represent
        # the influence of the flux boundary condition of the secondary
        # interface on the pressure trace on the first interface.
        self.intf_coupling_via_high_dim = True
        # No coupling via lower-dimensional interfaces.
        self.intf_coupling_via_low_dim = False

        # Keys used to identify the discretization matrices of this discretization
        self.mortar_discr_matrix_key = "robin_mortar_discr"
        self.mortar_vector_source_matrix_key = "robin_vector_source_discr"

    def ndof(self, intf: pp.MortarGrid):
        return intf.num_cells

    def discretize(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        sd_data_primary: Dict,
        sd_data_secondary: Dict,
        intf_data: Dict,
    ):
        """Discretize the interface law and store the discretization in the
        edge data.

        Args:
            sd_primary: Grid of the primary domain.
            sd_secondary: Grid of the secondary domain.
            intf (pp.MortarGrid): Mortar grid on the interface between the subdomains.
            sd_data_primary: Data dictionary for the primary domain.
            sd_data_secondary: Data dictionary for the secondary domain.
            intf_data: Data dictionary for the edge between the domains.

        """
        matrix_dictionary_edge: Dict[str, sps.spmatrix] = intf_data[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]
        parameter_dictionary_edge: Dict = intf_data[pp.PARAMETERS][self.keyword]

        parameter_dictionary_h: Dict = sd_data_primary[pp.PARAMETERS][
            self._primary_keyword
        ]

        kn: Union[np.ndarray, float] = parameter_dictionary_edge["normal_diffusivity"]
        # If normal diffusivity is given as a constant, parse to np.array
        if not isinstance(kn, np.ndarray):
            kn *= np.ones(intf.num_cells)

        val: np.ndarray = intf.cell_volumes * kn
        matrix_dictionary_edge[self.mortar_discr_matrix_key] = -sps.diags(val)

        ## Vector source.
        # This contribution is last term of
        # lambda = -\int{\kappa_n [p_l - p_h +  a/2 g \cdot n]} dV,
        # where n is the outwards normal and the integral is taken over the mortar cell.
        # (Note: This assumes a P0 discretization of mortar fluxes).

        # Ambient dimension of the problem, as specified for the higher-dimensional
        # neighbor.
        # IMPLEMENTATION NOTE: The default value is needed to avoid that
        # ambient_dimension becomes a required parameter. If neither ambient dimension,
        # nor the actual vector_source is specified, there will be no problems (in the
        # assembly, a zero vector soucre of a size that fits with the discretization is
        # created). If a vector_source is specified, but the ambient dimension is not,
        # a dimension mismatch will result unless the ambient dimension implied by
        # the size of the vector source matches sd_primary.dim. This is okay for domains with
        # no subdomains with co-dimension more than 1, but will fail for fracture
        # intersections. The default value is thus the least bad option in this case.
        vector_source_dim: int = parameter_dictionary_h.get(
            "ambient_dimension", sd_primary.dim
        )
        # The ambient dimension cannot be less than the dimension of sd_primary.
        # If this is broken, we risk ending up with zero normal vectors below, so it is
        # better to break this off now
        if vector_source_dim < sd_primary.dim:
            raise ValueError(
                "Ambient dimension cannot be lower than the grid dimension"
            )

        # Construct the dot product between normals on fracture faces and the identity
        # matrix.

        # Find the mortar normal vectors by projection of the normal vectors in sd_primary
        normals_h = sd_primary.face_normals.copy()

        # projection matrix
        proj = intf.primary_to_mortar_avg()

        # Ensure that the normal vectors point out of sd_primary
        # Indices of faces neighboring this mortar grid
        _, fi_h, _ = sps.find(proj)
        # Switch direction of vectors if relevant
        sgn, _ = sd_primary.signs_and_cells_of_boundary_faces(fi_h)
        normals_h[:, fi_h] *= sgn

        # Project the normal vectors, we need to do some transposes to get this right
        normals_mortar = (proj * normals_h.T).T
        nrm = np.linalg.norm(normals_mortar, axis=0)
        # Sanity check
        assert np.nonzero(nrm)[0].size == nrm.size
        outwards_unit_mortar_normals = normals_mortar / nrm

        # We know that the ambient dimension for the vector source must be at least as
        # high as sd_primary, thus taking the first vector_source_dim components of the normal
        # vector should be fine.
        vals = outwards_unit_mortar_normals[:vector_source_dim].ravel("F")

        # The values in vals are sorted by the mortar cell index ordering (proj is a
        # csr matrix).
        ci_mortar = np.arange(intf.num_cells, dtype=int)

        # The mortar cell indices are expanded to account for the vector source
        # having multiple dimensions
        rows = np.tile(ci_mortar, (vector_source_dim, 1)).ravel("F")
        # Columns must account for the values being vector values.
        cols = pp.fvutils.expand_indices_nd(ci_mortar, vector_source_dim)

        # And we have the normal vectors
        mortar_normals = sps.coo_matrix((vals, (rows, cols))).tocsr()

        # On assembly, the outwards normals on the mortars will be multiplied by the
        # interface vector source.
        matrix_dictionary_edge[self.mortar_vector_source_matrix_key] = mortar_normals

    def assemble_matrix(
        self,
        sd_primary,
        sd_secondary,
        intf,
        sd_data_primary,
        sd_data_secondary,
        intf_data,
        matrix,
    ):
        """Assemble the discretization of the interface law, and its impact on
        the neighboring domains. Assemble only matrix, not rhs terms.

        Parameters:
            sd_primary: Grid on one neighboring subdomain.
            sd_secondary: Grid on the other neighboring subdomain.
            sd_data_primary: Data dictionary for the primary suddomain
            sd_data_secondary: Data dictionary for the secondary subdomain.
            intf_data: Data dictionary for the edge between the subdomains.
                If gravity is taken into consideration, the parameter sub-
                dictionary should contain something like a/2 * g, where
                g is the ambient_dimension x n_mortar_cells gravity vector
                as used in Starnoni et al 2020, typically with
                    g[ambient_dimension]= -G * rho.
            matrix: original discretization

        The discretization matrices must be included since they will be changed
        by the imposition of Neumann boundary conditions on the internal boundary
        in some numerical methods (Read: VEM, RT0).


        """

        return self._assemble(
            sd_primary,
            sd_secondary,
            intf,
            sd_data_primary,
            sd_data_secondary,
            intf_data,
            matrix,
            assemble_rhs=False,
        )

    def assemble_rhs(
        self,
        sd_primary,
        sd_secondary,
        intf,
        sd_data_primary,
        sd_data_secondary,
        intf_data,
        matrix,
    ):
        """Assemble the dicretization of the interface law, and its impact on
        the neighboring domains. Assemble only rhs terms, not the discretization.

        Parameters:
            sd_primary: Grid on one neighboring subdomain.
            sd_secondary: Grid on the other neighboring subdomain.
            sd_data_primary: Data dictionary for the primary suddomain
            sd_data_secondary: Data dictionary for the secondary subdomain.
            intf_data: Data dictionary for the edge between the subdomains.
                If gravity is taken into consideration, the parameter sub-
                dictionary should contain something like a/2 * g, where
                g is the ambient_dimension x n_mortar_cells gravity vector
                as used in Starnoni et al 2020, typically with
                    g[ambient_dimension]= -G * rho.
            matrix: original discretization

            The discretization matrices must be included since they will be
            changed by the imposition of Neumann boundary conditions on the
            internal boundary in some numerical methods (Read: VEM, RT0)

        """
        return self._assemble(
            sd_primary,
            sd_secondary,
            sd_data_primary,
            sd_data_secondary,
            intf_data,
            matrix,
            assemble_matrix=False,
        )

    def assemble_matrix_rhs(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        sd_data_primary: dict,
        sd_data_secondary: dict,
        intf_data: dict,
        matrix: sps.spmatrix,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.

        Parameters:
            sd_primary: Grid on one neighboring subdomain.
            sd_secondary: Grid on the other neighboring subdomain.
            sd_data_primary: Data dictionary for the primary suddomain
            sd_data_secondary: Data dictionary for the secondary subdomain.
            intf_data: Data dictionary for the edge between the subdomains.
                If gravity is taken into consideration, the parameter sub-
                dictionary should contain something like a/2 * g, where
                g is the ambient_dimension x n_mortar_cells gravity vector
                as used in Starnoni et al 2020, typically with
                    g[ambient_dimension]= -G * rho.
            matrix: original discretization

        The discretization matrices must be included since they will be changed
        by the imposition of Neumann boundary conditions on the internal boundary
        in some numerical methods (Read: VEM, RT0).

        The overall strategy is to first assemble the pressure contributions,
        then multiply by the mortar discretization (corresponding to -\kappa) and
        finally add the identity matrix for the interface flux (see equation
        description above).
        """
        return self._assemble(
            sd_primary,
            sd_secondary,
            intf,
            sd_data_primary,
            sd_data_secondary,
            intf_data,
            matrix,
        )

    def _assemble(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        sd_data_primary: dict,
        sd_data_secondary: dict,
        intf_data: dict,
        matrix: sps.spmatrix,
        assemble_matrix: bool = True,
        assemble_rhs: bool = True,
    ):
        """Actual implementation of assembly. May skip matrix and rhs if specified."""

        matrix_dictionary_edge: Dict[str, sps.spmatrix] = intf_data[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]
        diffusivity_discr = matrix_dictionary_edge[self.mortar_discr_matrix_key]
        parameter_dictionary_edge = intf_data[pp.PARAMETERS][self.keyword]

        primary_ind = 0
        secondary_ind = 1

        # Appease mypy
        assert self.discr_primary is not None
        assert self.discr_secondary is not None

        if assemble_rhs and not assemble_matrix:
            # We need not make the cc matrix to assemble local matrix contributions
            rhs = self._define_local_block_matrix(
                sd_primary,
                sd_secondary,
                intf,
                self.discr_primary,
                self.discr_secondary,
                matrix,
                create_matrix=False,
            )
            # We will occationally need the variable cc, but it need not have a value
            cc = None
        else:
            cc, rhs = self._define_local_block_matrix(
                sd_primary,
                sd_secondary,
                intf,
                self.discr_primary,
                self.discr_secondary,
                matrix,
            )
        assert isinstance(rhs, np.ndarray)

        # The convention, for now, is to put the higher dimensional information
        # in the first column and row in matrix, lower-dimensional in the second
        # and mortar variables in the third

        # Assembly of contribution from boundary pressure must be called even if only
        # matrix or rhs must be assembled.
        self.discr_primary.assemble_int_bound_pressure_trace(
            sd_primary, sd_data_primary, intf, intf_data, cc, matrix, rhs, primary_ind
        )
        if assemble_matrix:
            # Appease mypy
            assert cc is not None

            # Calls only for matrix assembly
            self.discr_primary.assemble_int_bound_flux(
                sd_primary,
                sd_data_primary,
                intf,
                intf_data,
                cc,
                matrix,
                rhs,
                primary_ind,
            )
            self.discr_secondary.assemble_int_bound_pressure_cell(
                sd_secondary,
                sd_data_secondary,
                intf,
                intf_data,
                cc,
                matrix,
                rhs,
                secondary_ind,
            )
            self.discr_secondary.assemble_int_bound_source(
                sd_secondary,
                sd_data_secondary,
                intf,
                intf_data,
                cc,
                matrix,
                rhs,
                secondary_ind,
            )

            for block in range(3):
                # Scale the pressure contributions in the mortar equation by
                # normal diffusivity.
                cc[2, block] = diffusivity_discr * cc[2, block]
            # Add mortar flux to the interface equation
            cc[2, 2] += sps.diags(np.ones(intf.num_cells))

            matrix += cc

            self.discr_primary.enforce_neumann_int_bound(
                sd_primary, intf, intf_data, matrix, primary_ind
            )

        if assemble_rhs:
            # Calls only for rhs assembly
            if "vector_source" in parameter_dictionary_edge:
                # Also assemble vector sources.
                # Discretization of the vector source term
                vector_source_discr: sps.spmatrix = matrix_dictionary_edge[
                    self.mortar_vector_source_matrix_key
                ]
                # The vector source, defaults to zero if not specified.
                vector_source: np.ndarray = parameter_dictionary_edge.get(
                    "vector_source"
                )
                if vector_source_discr.shape[1] != vector_source.size:
                    # If this happens chances are that either the ambient dimension was not set
                    # and thereby its default value was used. Another not unlikely reason is
                    # that the ambient dimension is set, but with a value that does not match
                    # the specified vector source.
                    raise ValueError(
                        """Mismatch in vector source dimensions.
                        Did you forget to specify the ambient dimension?"""
                    )

                rhs[2] = rhs[2] - vector_source_discr * vector_source

            rhs[2] = diffusivity_discr * rhs[2]

        if not assemble_matrix:
            return rhs
        elif not assemble_rhs:
            return matrix
        else:
            return matrix, rhs

    def assemble_intf_coupling_via_high_dim(
        self,
        sd: pp.Grid,
        sd_data: dict,
        intf_primary: pp.MortarGrid,
        sd_pair_primary: Tuple[pp.Grid, pp.Grid],
        intf_data_primary: dict,
        intf_secondary: pp.MortarGrid,
        sd_pair_secondary: Tuple[pp.Grid, pp.Grid],
        intf_data_secondary: dict,
        matrix,
        assemble_matrix: bool = True,
        assemble_rhs: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Represent the impact on a primary interface of the mortar (thus boundary)
        flux on a secondary interface.

        Parameters:
            sd (pp.Grid): Grid of the higher dimensional neighbor to the main interface.
            sd_data (dict): Data dictionary of the intermediate grid.
            intf_primary (tuple of grids): The grids of the primary edge
            intf_data_primary (dict): Data dictionary of the primary interface.
            intf_secondary (tuple of grids): The grids of the secondary edge.
            intf_data_secondary (dict): Data dictionary of the secondary interface.
            matrix: original discretization.

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the primary, secondary and mortar variable, respectively.
            np.array: Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the primary,
                secondary and mortar variable, respectively.

        """
        assert self.discr_primary is not None and self.discr_secondary is not None

        if assemble_matrix:
            cc, rhs = self._define_local_block_matrix_intf_coupling(
                sd, self.discr_primary, intf_primary, intf_secondary, matrix
            )
        else:
            rhs = self._define_local_block_matrix_intf_coupling(
                sd,
                self.discr_primary,
                intf_primary,
                intf_secondary,
                matrix,
                create_matrix=False,
            )

        matrix_dictionary_edge = intf_data_primary[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]

        if assemble_matrix:
            # Normally, the projections will be pressure from the primary (high-dim node)
            # to the primary mortar, and flux from secondary mortar to primary
            proj_pressure = intf_primary.primary_to_mortar_avg()
            proj_flux = intf_secondary.mortar_to_primary_int()

            # If the primary and / or secondary mortar is a boundary mortar grid, things
            # become more complex. This probably assumes that a FluxPressureContinuity
            # discretization is applied on the relevant mortar grid.
            if (
                sd_pair_primary[0].dim == sd_pair_primary[1].dim
                and sd_pair_primary[0] == sd
            ):
                proj_pressure = intf_primary.secondary_to_mortar_avg()
            if (
                sd_pair_secondary[0].dim == sd_pair_secondary[1].dim
                and sd_pair_secondary[0] == sd
            ):
                proj_flux = intf_secondary.mortar_to_secondary_int()
            # Assemble contribution between higher dimensions.
            self.discr_primary.assemble_int_bound_pressure_trace_between_interfaces(
                sd, sd_data, proj_pressure, proj_flux, cc, matrix, rhs
            )
            # Scale the equations (this will modify from K^-1 to K scaling if relevant)

            for block in range(3):
                # Scale the pressure blocks in the row of the primary mortar problem,
                # i.e. the row corresponding to the mortar variable to which the
                # contributions from the flux of block 2 is being added.
                # The secondary mortar will be treated somewhere else (handled by the
                # assembler).
                cc[1, block] = (
                    matrix_dictionary_edge[self.mortar_discr_matrix_key] * cc[1, block]
                )
        else:
            cc = None

        if assemble_rhs:
            rhs[1] = matrix_dictionary_edge[self.mortar_discr_matrix_key] * rhs[1]

        return cc, rhs

    def __repr__(self) -> str:
        return f"Interface coupling of Robin type, with keyword {self.keyword}"


class FluxPressureContinuity(RobinCoupling):
    """A condition for flux and pressure continuity between two domains. A particular
    attention is devoted in the case if these domanins are of equal
    dimension. This can be used to specify full continuity between fractures,
    two domains or a periodic boundary condition for a single domain. The faces
    coupled by flux and pressure condition must be specified by a MortarGrid on
    a graph edge.
    For each face we will impose
    v_m = lambda
    v_s = -lambda
    p_m - p_s = 0
    where subscript m and s is for primary and secondary, v is the flux, p the pressure,
    and lambda the mortar variable.

    """

    def __init__(
        self,
        keyword: str,
        discr_primary: pp.EllipticDiscretization,
        discr_secondary: Optional[pp.EllipticDiscretization] = None,
    ) -> None:
        self.keyword = keyword
        self.discr_primary: pp.EllipticDiscretization = discr_primary
        if discr_secondary is None:
            self.discr_secondary: pp.EllipticDiscretization = discr_primary
        else:
            self.discr_secondary = discr_secondary

        # This interface law will have direct interface coupling to represent
        # the influence of the flux boundary condition of the secondary
        # interface on the pressure trace on the first interface.
        self.intf_coupling_via_high_dim = False
        # No coupling via lower-dimensional interfaces.
        self.intf_coupling_via_low_dim = False

    def discretize(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        sd_data_primary: Dict,
        sd_data_secondary: Dict,
        intf_data: Dict,
    ) -> None:
        """Nothing really to do here

        Parameters:
            sd_primary: Grid of the primary domanin.
            sd_secondary: Grid of the secondary domain.
            sd_data_primary: Data dictionary for the primary domain.
            sd_data_secondary: Data dictionary for the secondary domain.
            intf_data: Data dictionary for the edge between the domains.

        """
        pass

    def assemble_rhs(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        sd_data_primary: Dict,
        sd_data_secondary: Dict,
        intf_data: Dict,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.

        Parameters:
            sd_primary: Grid on one neighboring subdomain.
            sd_secondary: Grid on the other neighboring subdomain.
            sd_data_primary: Data dictionary for the primary suddomain
            sd_data_secondary: Data dictionary for the secondary subdomain.
            intf_data: Data dictionary for the edge between the subdomains
            matrix_primary: original discretization for the primary subdomain

        """
        # IMPLEMENTATION NOTE: This function is aimed at computational savings in a case
        # where the same linear system must be solved with multiple right hand sides.
        # Compared to self.assemble_matrix_rhs(), the method short cuts parts of the
        # assembly that only give contributions to the matrix.

        primary_ind = 0
        secondary_ind = 1

        # Generate matrix for the coupling.
        rhs_primary = self._define_local_block_matrix(
            sd_primary,
            sd_secondary,
            intf,
            self.discr_primary,
            self.discr_secondary,
            matrix,
            create_matrix=False,
        )
        assert isinstance(rhs_primary, np.ndarray) and rhs_primary.size == 3

        # so just reconstruct everything.
        rhs_secondary = np.empty(3, dtype=object)
        rhs_secondary[primary_ind] = np.zeros_like(rhs_primary[primary_ind])
        rhs_secondary[secondary_ind] = np.zeros_like(rhs_primary[secondary_ind])
        rhs_secondary[2] = np.zeros_like(rhs_primary[2])

        # If primary and secondary is the same grid, they should contribute to the same
        # row and column. When the assembler assigns matrix[idx] it will only add
        # the secondary information because of duplicate indices (primary and secondary
        # is the same). We therefore write the both primary and secondary info to the
        # secondary index.
        if sd_primary == sd_secondary:
            primary_ind = 1
        else:
            primary_ind = 0

        # Compared to the full set of sub-assembly methods called in self.assemble_matrix_rhs,
        # we only get a contribution from the int_bound_pressure term. This must be
        # assembled for the primary, and, if this is a connection between grids of equal
        # dimension also the secondary side.
        cc_primary = None
        self.discr_primary.assemble_int_bound_pressure_trace_rhs(
            sd_primary,
            sd_data_primary,
            intf,
            intf_data,
            cc_primary,
            rhs_primary,
            primary_ind,
        )
        if sd_primary.dim == sd_secondary.dim:
            cc_secondary = None
            self.discr_secondary.assemble_int_bound_pressure_trace_rhs(
                sd_secondary,
                sd_data_secondary,
                intf,
                intf_data,
                cc_secondary,
                rhs_secondary,
                secondary_ind,
                use_secondary_proj=True,
            )
            # Sign change to enforce continuity.
            rhs_secondary[2] = -rhs_secondary[2]

        rhs = rhs_primary + rhs_secondary

        return rhs

    def assemble_matrix(
        self,
        sd_primary,
        sd_secondary,
        intf,
        sd_data_primary,
        sd_data_secondary,
        intf_data,
        matrix,
    ):
        """Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.

        Parameters:
            sd_primary: Grid on one neighboring subdomain.
            sd_secondary: Grid on the other neighboring subdomain.
            sd_data_primary: Data dictionary for the primary suddomain
            sd_data_secondary: Data dictionary for the secondary subdomain.
            intf_data: Data dictionary for the edge between the subdomains
            matrix_primary: original discretization for the primary subdomain
            matrix_secondary: original discretization for the secondary subdomain

        """
        A, _ = self.assemble_matrix_rhs(
            sd_primary,
            sd_secondary,
            intf,
            sd_data_primary,
            sd_data_secondary,
            intf_data,
            matrix,
        )
        return A

    def assemble_matrix_rhs(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        sd_data_primary: Dict,
        sd_data_secondary: Dict,
        intf_data: Dict,
        matrix: np.ndarray,
    ):
        """Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.

        Parameters:
            sd_primary: Grid on one neighboring subdomain.
            sd_secondary: Grid on the other neighboring subdomain.
            sd_data_primary: Data dictionary for the primary subdomain
            sd_data_secondary: Data dictionary for the secondary subdomain.
            intf_data: Data dictionary for the edge between the subdomains
            matrix_primary: original discretization for the primary subdomain

        """
        primary_ind = 0
        secondary_ind = 1

        # Generate matrix for the coupling.
        cc_primary, rhs_primary = self._define_local_block_matrix(
            sd_primary,
            sd_secondary,
            intf,
            self.discr_primary,
            self.discr_secondary,
            matrix,
        )

        cc_secondary = cc_primary.copy()

        # I got some problems with pointers when doing rhs_primary = rhs_secondary.copy()
        # so just reconstruct everything.
        rhs_secondary = np.empty(3, dtype=object)
        rhs_secondary[primary_ind] = np.zeros_like(rhs_primary[primary_ind])
        rhs_secondary[secondary_ind] = np.zeros_like(rhs_primary[secondary_ind])
        rhs_secondary[2] = np.zeros_like(rhs_primary[2])

        # The convention, for now, is to put the primary grid information
        # in the first column and row in matrix, secondary grid in the second
        # and mortar variables in the third
        # If primary and secondary is the same grid, they should contribute to the same
        # row and column. When the assembler assigns matrix[idx] it will only add
        # the secondary information because of duplicate indices (primary and secondary
        # is the same). We therefore write the both primary and secondary info to the
        # secondary index.
        if sd_primary == sd_secondary:
            primary_ind = 1
        else:
            primary_ind = 0

        self.discr_primary.assemble_int_bound_pressure_trace(
            sd_primary,
            sd_data_primary,
            intf,
            intf_data,
            cc_primary,
            matrix,
            rhs_primary,
            primary_ind,
        )
        self.discr_primary.assemble_int_bound_flux(
            sd_primary,
            sd_data_primary,
            intf,
            intf_data,
            cc_primary,
            matrix,
            rhs_primary,
            primary_ind,
        )

        if sd_primary.dim == sd_secondary.dim:
            # Consider this terms only if the grids are of the same dimension, by
            # imposing the same condition with a different sign, due to the normal
            self.discr_secondary.assemble_int_bound_pressure_trace(
                sd_secondary,
                sd_data_secondary,
                intf,
                intf_data,
                cc_secondary,
                matrix,
                rhs_secondary,
                secondary_ind,
                use_secondary_proj=True,
            )

            self.discr_secondary.assemble_int_bound_flux(
                sd_secondary,
                sd_data_secondary,
                intf,
                intf_data,
                cc_secondary,
                matrix,
                rhs_secondary,
                secondary_ind,
                use_secondary_proj=True,
            )
            # We now have to flip the sign of some of the matrices
            # First we flip the sign of the secondary flux because the mortar flux points
            # from the primary to the secondary, i.e., flux_s = -mortar_flux
            cc_secondary[secondary_ind, 2] = -cc_secondary[secondary_ind, 2]
            # Then we flip the sign for the pressure continuity since we have
            # We have that p_m - p_s = 0.
            cc_secondary[2, secondary_ind] = -cc_secondary[2, secondary_ind]
            rhs_secondary[2] = -rhs_secondary[2]
            # Note that cc_secondary[2, 2] is fliped twice, first for pressure continuity
        else:
            # Consider this terms only if the grids are of different dimension, by
            # imposing pressure trace continuity and conservation of the normal flux
            # through the lower dimensional object.
            self.discr_secondary.assemble_int_bound_pressure_cell(
                sd_secondary,
                sd_data_secondary,
                intf,
                intf_data,
                cc_secondary,
                matrix,
                rhs_secondary,
                secondary_ind,
            )

            self.discr_secondary.assemble_int_bound_source(
                sd_secondary,
                sd_data_secondary,
                intf,
                intf_data,
                cc_secondary,
                matrix,
                rhs_secondary,
                secondary_ind,
            )

        # Now, the matrix cc = cc_secondary + cc_primary expresses the flux and pressure
        # continuities over the mortars.
        # cc[0] -> flux_m = mortar_flux
        # cc[1] -> flux_s = -mortar_flux
        # cc[2] -> p_m - p_s = 0

        # Computational savings: Only add non-zero components.
        # Exception: The case with equal dimension of the two neighboring grids.
        if sd_primary.dim == sd_secondary.dim:
            matrix += cc_primary + cc_secondary
        else:
            matrix[0, 2] += cc_primary[0, 2]
            matrix[1, 2] += cc_secondary[1, 2]
            for col in range(3):
                matrix[2, col] += cc_primary[2, col] + cc_secondary[2, col]

        rhs = rhs_primary + rhs_secondary

        self.discr_primary.enforce_neumann_int_bound(
            sd_primary, intf, intf_data, matrix, primary_ind
        )

        # Consider this terms only if the grids are of the same dimension
        if sd_primary.dim == sd_secondary.dim:
            self.discr_secondary.enforce_neumann_int_bound(
                sd_secondary, intf, intf_data, matrix, secondary_ind
            )

        return matrix, rhs

    def __repr__(self) -> str:
        return f"Interface coupling with full pressure flux continuity. Keyword {self.keyword}"


class WellCoupling(AbstractInterfaceLaw):
    """Simple version of the classical Peaceman model relating well and fracture
    pressure.

    """

    def __init__(
        self,
        keyword: str,
        discr_primary: Optional["pp.EllipticDiscretization"] = None,
        discr_secondary: Optional["pp.EllipticDiscretization"] = None,
        primary_keyword: Optional[str] = None,
    ) -> None:
        """Initialize Robin Coupling.

        Parameters:
            keyword (str): Keyword used to access parameters needed for this
                discretization in the data dictionary. Will also define where
                 discretization matrices are stored.
            discr_primary: Discretization on the higher-dimensional neighbor. Only
                needed when the RobinCoupling is used for local assembly.
            discr_secondary: Discretization on the lower-dimensional neighbor. Only
                needed when the RobinCoupling is used for local assembly. If not
                provided, it is assumed that primary and secondary discretizations
                are identical.
            primary_keyword: Parameter keyword for the discretization on the higher-
                dimensional neighbor, which the RobinCoupling is intended coupled to.
                Only needed when the object is not used for local assembly (that is,
                needed if Ad is used).
        """
        super().__init__(keyword)

        if discr_secondary is None:
            discr_secondary = discr_primary

        if primary_keyword is not None:
            self._primary_keyword = primary_keyword
        else:
            if discr_primary is None:
                raise ValueError(
                    "Either primary keyword or primary discretization must be specified"
                )
            else:
                self._primary_keyword = discr_primary.keyword

        self.discr_primary = discr_primary
        self.discr_secondary = discr_secondary

        # Keys used to identify the discretization matrices of this discretization
        self.well_discr_matrix_key: str = "well_mortar_discr"
        self.well_vector_source_matrix_key: str = "well_vector_source_discr"

    def ndof(self, intf: pp.MortarGrid) -> int:
        return intf.num_cells

    def discretize(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        sd_data_primary: Dict,
        sd_data_secondary: Dict,
        intf_data: Dict,
    ) -> None:
        """Discretize the Peaceman interface law and store the discretization in the
        edge data.

        Parameters:
            sd_primary: Grid of the primary domanin.
            sd_secondary: Grid of the secondary domain.
            sd_data_primary: Data dictionary for the primary domain.
            sd_data_secondary: Data dictionary for the secondary domain.
            intf_data: Data dictionary for the edge between the domains.

        Implementational note: The computation of equivalent radius is highly simplified
        and ignores discretization and anisotropy effects. For more advanced alternatives,
        see the MRST book, https://www.cambridge.org/core/books/an-introduction-to-
        reservoir-simulation-using-matlabgnu-octave/F48C3D8C88A3F67E4D97D4E16970F894
        """
        matrix_dictionary_edge: Dict[str, sps.spmatrix] = intf_data[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]
        parameter_dictionary_edge: Dict = intf_data[pp.PARAMETERS][self.keyword]

        parameter_dictionary_h: Dict = sd_data_primary[pp.PARAMETERS][
            self._primary_keyword
        ]
        parameter_dictionary_l: Dict = sd_data_secondary[pp.PARAMETERS][self.keyword]

        # projection matrix
        proj_h = intf.primary_to_mortar_avg()
        proj_l = intf.secondary_to_mortar_avg()

        r_w = parameter_dictionary_l["well_radius"]
        skin_factor = parameter_dictionary_edge["skin_factor"]
        # Compute equivalent radius for Peaceman well model (see above note)
        r_e = 0.2 * np.power(sd_primary.cell_volumes, 1 / sd_primary.dim)
        # Compute effective permeability
        k: pp.SecondOrderTensor = parameter_dictionary_h["second_order_tensor"]
        if sd_primary.dim == 2:
            R = pp.map_geometry.project_plane_matrix(sd_primary.nodes)
            k = k.copy()
            k.values = np.tensordot(R.T, np.tensordot(R, k.values, (1, 0)), (0, 1))
            k.values = np.delete(k.values, (2), axis=0)
            k.values = np.delete(k.values, (2), axis=1)
        kx = k.values[0, 0]
        ky = k.values[1, 1]
        ke = np.sqrt(kx * ky)

        # Kh is ke * specific volume, but this is already captured by k
        Kh = proj_h * ke * intf.cell_volumes
        WI = sps.diags(
            2 * np.pi * Kh / (np.log(proj_h * r_e / r_w) + skin_factor), format="csr"
        )
        matrix_dictionary_edge[self.well_discr_matrix_key] = -WI

        ## Vector source.
        # This contribution is last term of
        # lambda = -\int{\kappa_n [p_l - p_h +  a/2 g \cdot n]} dV,
        # where n is the outwards normal and the integral is taken over the mortar cell.
        # (Note: This assumes a P0 discretization of mortar fluxes).

        # Ambient dimension of the problem, as specified for the higher-dimensional
        # neighbor.
        # IMPLEMENTATION NOTE: The default value is needed to avoid that
        # ambient_dimension becomes a required parameter. If neither ambient dimension,
        # nor the actual vector_source is specified, there will be no problems (in the
        # assembly, a zero vector source of a size that fits with the discretization is
        # created). If a vector_source is specified, but the ambient dimension is not,
        # a dimension mismatch will result unless the ambient dimension implied by
        # the size of the vector source matches sd_primary.dim. This is okay for domains with
        # no subdomains with co-dimension more than 1, but will fail for fracture
        # intersections. The default value is thus the least bad option in this case.
        vector_source_dim: int = parameter_dictionary_h.get(
            "ambient_dimension", sd_primary.dim
        )
        # The ambient dimension cannot be less than the dimension of sd_primary.
        # If this is broken, we risk ending up with zero normal vectors below, so it is
        # better to break this off now
        if vector_source_dim < sd_primary.dim:
            raise ValueError(
                "Ambient dimension cannot be lower than the grid dimension"
            )

        # Construct the dot product between vectors connecting fracture and well
        # cell centers and the identity matrix.

        # Project the vectors, we need to do some transposes to get this right.
        # Note that in the codim 2 case, proj_h maps from sd_primary cells, not faces.
        vectors_mortar = (
            proj_h * sd_primary.cell_centers.T - proj_l * sd_secondary.cell_centers.T
        ).T

        # The values in vals are sorted by the mortar cell index ordering (proj is a
        # csr matrix).
        ci_mortar = np.arange(intf.num_cells, dtype=int)

        # The mortar cell indices are expanded to account for the vector source
        # having multiple dimensions
        rows = np.tile(ci_mortar, (vector_source_dim, 1)).ravel("F")
        # Columns must account for the values being vector values.
        cols = pp.fvutils.expand_indices_nd(ci_mortar, vector_source_dim)
        vals = vectors_mortar[:vector_source_dim].ravel("F")
        # And we have the normal vectors
        expanded_vectors = sps.coo_matrix((vals, (rows, cols))).tocsr()

        # On assembly, the outwards normals on the mortars will be multiplied by the
        # interface vector source.
        matrix_dictionary_edge[self.well_vector_source_matrix_key] = expanded_vectors

    def assemble_matrix_rhs(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        sd_data_primary: Dict,
        sd_data_secondary: Dict,
        intf_data: Dict,
        matrix: sps.spmatrix,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def __repr__(self) -> str:
        return f"Interface coupling of Well type, with keyword {self.keyword}"
