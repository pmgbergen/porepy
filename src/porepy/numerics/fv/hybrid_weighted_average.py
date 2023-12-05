import numpy as np
import scipy as sp
import porepy as pp
from typing import Callable, Tuple, Union
from . import hybrid_upwind_utils as hu_utils
import copy
import pdb


from porepy.numerics.discretization import Discretization


def myprint(var):
    print("\n" + var + " = ", eval(var))


def ndof(g: pp.Grid) -> int:
    """- hardcoded for two-phase"""
    return 2 * g.num_cells


def total_flux_internal(
    sd: pp.Grid,
    mixture: pp.Mixture,
    pressure: pp.ad.AdArray,
    gravity_value: float,
    left_restriction: sp.sparse.spmatrix,
    right_restriction: sp.sparse.spmatrix,
    transmissibility_internal_tpfa: np.ndarray,
    ad: bool,
    dynamic_viscosity: float,
    dim_max: int,
    mobility: Callable,
    permeability: Callable,
) -> pp.ad.AdArray:
    """ """

    def gamma_value(permeability):
        """ """
        alpha = 1.0  # as in the paper 2022

        kr0 = permeability(saturation=1)  # TODO: is it right?

        def second_derivative(permeability, val):
            """sorry, I'm lazy..."""
            h = 1e-4
            return (
                permeability(val + h) - 2 * permeability(val) + permeability(val - h)
            ) / (h**2)

        dd_kr_max = np.nanmax(
            second_derivative(permeability, np.linspace(0, 1, 10))
        )  # TODO: improve it...

        gamma_val = alpha / kr0 * dd_kr_max
        return gamma_val

    def g_ref_faces(
        mixture: pp.Mixture,
        pressure: pp.ad.AdArray,
        z: np.ndarray,
        gravity_value: float,
        left_restriction: sp.sparse.spmatrix,
        right_restriction: sp.sparse.spmatrix,
    ):
        """
        - harcoded for two phases
        """
        density_faces_0 = hu_utils.density_internal_faces(
            mixture.get_phase(0).saturation,
            mixture.get_phase(0).mass_density(pressure),
            left_restriction,
            right_restriction,
        )
        density_faces_1 = hu_utils.density_internal_faces(
            mixture.get_phase(1).saturation,
            mixture.get_phase(1).mass_density(pressure),
            left_restriction,
            right_restriction,
        )

        density_max = pp.ad.maximum(density_faces_0, density_faces_1)

        g_ref = (
            density_max * gravity_value * (left_restriction @ z - right_restriction @ z)
        )
        return g_ref

    def delta_potential_faces(
        pressure,
        saturation,
        density,
        z,
        gravity_value,
        left_restriction,
        right_restriction,
    ):
        """ """
        density_internal_faces = hu_utils.density_internal_faces(
            saturation, density, left_restriction, right_restriction
        )
        g_internal_faces = hu_utils.g_internal_faces(
            z,
            density_internal_faces,
            gravity_value,
            left_restriction,
            right_restriction,
        )
        delta_pot = (
            left_restriction @ pressure - right_restriction @ pressure
        ) - g_internal_faces

        # print(
        #     np.concatenate(
        #         (
        #             (
        #                 left_restriction @ pressure - right_restriction @ pressure
        #             ).val.reshape(1, -1),
        #             np.arange(17).reshape(1, -1),
        #         ),
        #         axis=0,
        #     ).T
        # )

        # print(
        #     np.concatenate(
        #         (g_internal_faces.val.reshape(1, -1), np.arange(17).reshape(1, -1)),
        #         axis=0,
        #     ).T
        # )

        # print(
        #     np.concatenate(
        #         (delta_pot.val.reshape(1, -1), np.arange(17).reshape(1, -1)), axis=0
        #     ).T
        # )

        return delta_pot

    def beta_faces(
        pressure,
        saturation,
        density,
        z,
        gravity_value,
        gamma_val,
        g_ref_faces,
        left_restriction,
        right_restriction,
        ad,
    ):
        """ """
        c_faces_ref = 0  # no capillary pressure

        delta_pot_faces = delta_potential_faces(
            pressure,
            saturation,
            density,
            z,
            gravity_value,
            left_restriction,
            right_restriction,
        )
        tmp = gamma_val / (
            pp.ad.abs(g_ref_faces) + c_faces_ref + 1e-8
        )  # added epsilon to avoid division by zero

        if ad:
            tmp = -pp.ad.functions.maximum(-tmp, -1e6)
            beta_faces = 0.5 + 1 / np.pi * pp.ad.arctan(tmp * delta_pot_faces)
        else:
            tmp = np.minimum(np.real(tmp), 1e6) + np.imag(tmp) * 1j  # TODO: improve it
            beta_faces = 0.5 + 1 / np.pi * np.arctan(tmp * delta_pot_faces)

        # print(
        #     "beta_faces = ",
        #     np.concatenate(
        #         (beta_faces.val.reshape(1, -1), np.arange(17).reshape(1, -1)), axis=0
        #     ).T,
        # )

        # pdb.set_trace()

        return beta_faces

    def lambda_WA_faces(beta_faces, mobility, left_restriction, right_restriction):
        """ """
        lambda_WA = beta_faces * (left_restriction @ mobility) + (1 - beta_faces) * (
            right_restriction @ mobility
        )
        return lambda_WA

    # total flux computation:

    # 0D shortcut:
    if sd.dim == 0:
        total_flux = [None] * mixture.num_phases
        for m in np.arange(mixture.num_phases):
            total_flux[m] = pp.ad.AdArray(
                np.empty((0)), sp.sparse.csr_matrix((0, pressure.jac.shape[1]))
            )

        return total_flux

    z = -sd.cell_centers[
        dim_max - 1
    ]  # zed is reversed to conform to the notation in paper 2022
    # pp assumes that a 2D problem lies in xy plane. I assume that g is alinged to the last axes. Therefore I need to know dim_max and I call last dimension coordinate z

    g_ref_faces = g_ref_faces(
        mixture, pressure, z, gravity_value, left_restriction, right_restriction
    )
    gamma_val = gamma_value(permeability)

    total_flux = [None] * mixture.num_phases

    for m in np.arange(mixture.num_phases):
        saturation_m = mixture.get_phase(m).saturation
        density_m = mixture.get_phase(m).mass_density(pressure)
        bet_faces = beta_faces(
            pressure,
            saturation_m,
            density_m,
            z,
            gravity_value,
            gamma_val,
            g_ref_faces,
            left_restriction,
            right_restriction,
            ad,
        )
        mob = mobility(saturation_m, dynamic_viscosity)  # TODO: improve it

        lam_WA_faces = lambda_WA_faces(
            bet_faces, mob, left_restriction, right_restriction
        )

        delta_pot_faces = delta_potential_faces(
            pressure,
            saturation_m,
            density_m,
            z,
            gravity_value,
            left_restriction,
            right_restriction,
        )

        total_flux[m] = (
            lam_WA_faces * delta_pot_faces
        ) * transmissibility_internal_tpfa

    return total_flux


def rho_total_flux_internal(
    sd: pp.Grid,
    mixture,
    pressure,
    gravity_value,
    left_restriction,
    right_restriction,
    transmissibility_internal_tpfa,
    ad,
    dynamic_viscosity,
    dim_max,
    mobility,
    permeability,
):
    """ """
    qt = total_flux_internal(
        sd,
        mixture,
        pressure,
        gravity_value,
        left_restriction,
        right_restriction,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
        dim_max,
        mobility,
        permeability,
    )

    rho_qt = [None, None]
    for m in np.arange(mixture.num_phases):
        saturation_m = mixture.get_phase(m).saturation
        density_m = mixture.get_phase(m).mass_density(pressure)
        rho_m = hu_utils.density_internal_faces(
            saturation_m, density_m, left_restriction, right_restriction
        )

        rho_qt[m] = rho_m * qt[m]

    rho_qt = rho_qt[0] + rho_qt[1]
    return rho_qt


def rho_total_flux(
    sd: pp.Grid,
    mixture,
    pressure,
    gravity_value,
    left_restriction,
    right_restriction,
    expansion_matrix,
    transmissibility_internal_tpfa,
    ad,
    dynamic_viscosity,
    dim_max,
    mobility,
    permeability,
):
    # 0D shortcut:
    if sd.dim == 0:
        # rho_qt = pp.ad.AdArray(np.array([0]), 0*pressure.jac[0])
        rho_qt = pp.ad.AdArray(
            np.empty((0)), sp.sparse.csr_matrix((0, pressure.jac.shape[1]))
        )
        return rho_qt

    rho_qt = expansion_matrix @ rho_total_flux_internal(
        sd,
        mixture,
        pressure,
        gravity_value,
        left_restriction,
        right_restriction,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
        dim_max,
        mobility,
        permeability,
    )

    return rho_qt


'''
    def no_name(self, sd: pp.Grid, data: dict) -> None:
        """ """

        mixture = self.mixture
        pressure = self.pressure
        gravity_value = self.gravity_value
        hu_utils.compute_transmissibility_tmp(
            sd, data
        )  # TODO: move it in prepare_simualtion
        transmissibility_internal_tpfa = hu_utils.utils.get_transmissibility_tpfa(
            sd, data
        )
        left_restriction, right_restriction = hu_utils.restriction_matrices_left_right(
            sd
        )
        ad = True
        dynamic_viscosity = 1  ### TODO: constitutive_law.ConstantViscosity
        # I gess: dynamic_viscosity = self.fluid.viscosity

        rho_qt = self.rho_total_flux(
            sd,
            mixture,
            pressure,
            gravity_value,
            left_restriction,
            right_restriction,
            transmissibility_internal_tpfa,
            ad,
            dynamic_viscosity,
        )

    ################################################################################################################
    # not used/to delete:
    #################################################################################################################

    @staticmethod
    def compute_jacobian_qt_ad(
        sd, data, mixture, pressure, gravity_value, ad, dynamic_viscosity
    ):
        """
        attention: here you don't need ell bcs the primary variable has been identified before computing the flux.
                    for the other jac computation you need ell
        """
        L, R = HybridUpwind.restriction_matrices_left_right(sd)
        HybridUpwind.compute_transmissibility_tpfa(sd, data)
        _, transmissibility_internal_tpfa = HybridUpwind.get_transmissibility_tpfa(
            sd, data
        )

        rho_qt_internal = HybridUpwind.rho_total_flux(
            sd,
            mixture,
            pressure,
            gravity_value,
            L,
            R,
            transmissibility_internal_tpfa,
            ad,
            dynamic_viscosity,
        )

        pp_div = pp.fvutils.scalar_divergence(sd)

        expansion = HybridUpwind.expansion_matrix(sd)
        rho_qt = expansion @ rho_qt_internal

        flux_cell_no_bc = pp_div @ rho_qt

        return flux_cell_no_bc.val, flux_cell_no_bc.jac.A

    @staticmethod
    def compute_jacobian_qt_complex(
        sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
    ):
        """attention: different logic wrt finite diff to add eps
        attention: here you need ell to know what is the primary variable. in ad you don't
        """

        L, R = HybridUpwind.restriction_matrices_left_right(sd)
        HybridUpwind.compute_transmissibility_tpfa(sd, data)
        _, transmissibility_internal_tpfa = HybridUpwind.get_transmissibility_tpfa(
            sd, data
        )

        jacobian = np.zeros((sd.num_cells, 2 * sd.num_cells), dtype=np.complex128)
        eps = 1e-20j

        pp_div = pp.fvutils.scalar_divergence(sd)

        pressure_eps = copy.deepcopy(pressure)  # useless
        for i in np.arange(sd.num_cells):  # TODO: ...
            pressure_eps[i] += eps  ### ...

            rho_qt_internal = HybridUpwind.rho_total_flux(
                sd,
                mixture,
                pressure_eps,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )
            rho_qt = np.zeros(sd.num_faces, dtype=np.complex128)  ### ...
            rho_qt[sd.get_internal_faces()] = rho_qt_internal

            flux_cell_no_bc = pp_div @ rho_qt
            jacobian[:, i] = np.imag(flux_cell_no_bc) / np.imag(eps)

            pressure_eps[i] -= eps  ### ...

        saturation_ell = mixture.get_phase(ell).saturation
        saturation_ell_eps = copy.deepcopy(saturation_ell)  # useless

        if ell == 0:  # remove it... it is tmp
            m = 1
        else:
            m = 0
        saturation_m = mixture.get_phase(m).saturation
        saturation_m_eps = copy.deepcopy(saturation_m)

        for i in np.arange(sd.num_cells):  # TODO: ...
            saturation_ell_eps[i] += eps  ###
            mixture.get_phase(ell)._s = saturation_ell_eps

            saturation_m_eps[i] -= eps  # the constraint is here...
            mixture.get_phase(m)._s = saturation_m_eps

            rho_qt_internal = HybridUpwind.rho_total_flux(
                sd,
                mixture,
                pressure,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )
            rho_qt = np.zeros(sd.num_faces, dtype=np.complex128)  ### ...
            rho_qt[sd.get_internal_faces()] = rho_qt_internal

            flux_cell_no_bc = pp_div @ rho_qt
            jacobian[:, sd.num_cells + i] = np.imag(flux_cell_no_bc) / np.imag(
                eps
            )  # pay attention: jacobian[:,sd.num_cells+i]

            saturation_ell_eps[i] -= eps  ###...
            saturation_m_eps[i] += eps  ###...

        return jacobian

    @staticmethod
    def compute_jacobian_qt_finite_diff(
        sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
    ):
        """attention: different logic wrt complex step to add eps"""

        L, R = HybridUpwind.restriction_matrices_left_right(sd)
        HybridUpwind.compute_transmissibility_tpfa(sd, data)
        _, transmissibility_internal_tpfa = HybridUpwind.get_transmissibility_tpfa(
            sd, data
        )

        jacobian = np.zeros((sd.num_cells, 2 * sd.num_cells), dtype=np.complex128)
        eps = 1e-5

        pp_div = pp.fvutils.scalar_divergence(sd)

        rho_qt_internal = HybridUpwind.rho_total_flux(
            sd,
            mixture,
            pressure,
            gravity_value,
            L,
            R,
            transmissibility_internal_tpfa,
            ad,
            dynamic_viscosity,
        )
        rho_qt = np.zeros(sd.num_faces, dtype=np.complex128)  ### ...
        rho_qt[sd.get_internal_faces()] = rho_qt_internal

        flux_cell_no_bc = pp_div @ rho_qt

        for i in np.arange(sd.num_cells):  # TODO: ...
            pressure_eps = copy.deepcopy(pressure)  ### ...
            pressure_eps[i] += eps  ### ...

            rho_qt_internal_eps = HybridUpwind.rho_total_flux(
                sd,
                mixture,
                pressure_eps,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )
            rho_qt_eps = np.zeros(sd.num_faces, dtype=np.complex128)  ### ...
            rho_qt_eps[sd.get_internal_faces()] = rho_qt_internal_eps

            flux_cell_no_bc_eps = pp_div @ rho_qt_eps
            jacobian[:, i] = (flux_cell_no_bc_eps - flux_cell_no_bc) / eps

        saturation_ell = mixture.get_phase(ell).saturation
        saturation_ell_eps = copy.deepcopy(saturation_ell)  ### TODO: ...

        if ell == 0:  # remove it... it is tmp
            m = 1
        else:
            m = 0
        saturation_m = mixture.get_phase(m).saturation
        saturation_m_eps = copy.deepcopy(saturation_m)

        for i in np.arange(sd.num_cells):  # TODO: ...
            saturation_ell_eps[i] += eps  ### ...
            mixture.get_phase(ell)._s = saturation_ell_eps

            saturation_m_eps[i] -= eps  # the constraint is here...
            mixture.get_phase(m)._s = saturation_m_eps

            rho_qt_internal_eps = HybridUpwind.rho_total_flux(
                sd,
                mixture,
                pressure,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )
            rho_qt_eps = np.zeros(sd.num_faces, dtype=np.complex128)  ### ...
            rho_qt_eps[sd.get_internal_faces()] = rho_qt_internal_eps

            flux_cell_no_bc_eps = pp_div @ rho_qt_eps
            jacobian[:, sd.num_cells + i] = (
                flux_cell_no_bc_eps - flux_cell_no_bc
            ) / eps  # pay attention: jacobian[:,sd.num_cells+i]

            saturation_ell_eps[i] -= eps  ### ...
            saturation_m_eps[i] += eps  ### ...
        # mixture.get_phase(ell)._s = saturation_ell  # otherwise the last component keeps the eps

        return jacobian

    def assemble_matrix_rhs(
        self, g: pp.Grid, data
    ) -> Union[sp.sparse.spmatrix, np.ndarray]:
        """not implemented because not gonna use it"""

        print("\ninside assemble_matrix_rhs")
'''
