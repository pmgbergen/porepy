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
    """TODO: is it right?"""
    return 2 * g.num_cells


def total_flux_internal(
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
):
    """ """

    def gamma_value():
        """ """
        alpha = 1  # as in the paper 2022

        kr0 = pp.rel_perm_brooks_corey(saturation=1)  # TODO: is it right?
        dd_kr_max = pp.second_derivative(
            np.arange(0, 1, 10)
        ).max()  # TODO: improve it...

        gamma_val = alpha / kr0 * dd_kr_max
        # return pp.ad.Scalar(gamma_val, name="gamma value")
        return gamma_val

    def g_ref_faces(
        mixture, pressure, z, gravity_value, left_restriction, right_restriction
    ):
        """
        harcoded for two phases
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

        density_max = pp.ad.maximum(
            density_faces_0, density_faces_1
        )  # 1 Giune 2023: np.maximum(density_faces_0, density_faces_1), eh? che è successo? non puoi usare np.maximum, ovv...
        # density_max = maximum_operators(density_faces_0, density_faces_1)

        # # OLD:
        # g_ref = (
        #     density_max
        #     * gravity_value
        #     * (left_restriction * z - right_restriction * z)
        # )

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
        ) - g_internal_faces  # TODO: parenthesis... are they really necessary?
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
            g_ref_faces + c_faces_ref + 1e-8
        )  # added epsilon to avoid division by zero

        if ad:
            tmp = -pp.ad.functions.maximum(-tmp, -1e6)
            beta_faces = 0.5 + 1 / np.pi * pp.ad.arctan(tmp * delta_pot_faces)
            # beta_faces = pp.ad.Scalar(0.5, "zero point five") + pp.ad.Scalar(
            #     1 / np.pi, name="one over pi"
            # ) * arctan_operators(
            #     tmp * delta_pot_faces
            # )  # TODO: right?

        else:
            tmp = np.minimum(np.real(tmp), 1e6) + np.imag(tmp) * 1j  # TODO: improve it
            beta_faces = 0.5 + 1 / np.pi * np.arctan(tmp * delta_pot_faces)

        return beta_faces

    def lambda_WA_faces(beta_faces, mobility, left_restriction, right_restriction):
        """ """
        # OLD:
        lambda_WA = beta_faces * (left_restriction @ mobility) + (1 - beta_faces) * (
            right_restriction @ mobility
        )

        # NEW:
        # lambda_WA = beta_faces * (left_restriction @ mobility) + (
        #     pp.ad.Scalar(1, name="one") - beta_faces
        # ) * (right_restriction @ mobility)
        return lambda_WA

    # total flux computation:

    z = -sd.cell_centers[
        dim_max - 1
    ]  # zed is reversed to conform to the notation in paper 2022
    # pp assumes that a 2D problem lies in xy plane. I assume that g is alinged to the last axes. Therefore I need to know dim_max and I call last dimension coordinate z

    # # TODO:
    # dim_max = self.sd.dim_max()
    # z = -self.wrap_grid_attribute(
    #     sd.subdomains(), "cell_centers", dim_max
    # )  # it sould be ok...x

    # print("\n\n check z: -------")
    # pdb.set_trace()
    # TODO: projection to be added. # NO! you simply dont have to

    g_ref_faces = g_ref_faces(
        mixture, pressure, z, gravity_value, left_restriction, right_restriction
    )
    gamma_val = gamma_value()

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
        mobility = pp.mobility(saturation_m, dynamic_viscosity)  # TODO: improve it

        lam_WA_faces = lambda_WA_faces(
            bet_faces, mobility, left_restriction, right_restriction
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
):
    


    # print(sd)
    # print(mixture)
    # print(mixture.get_phase(0).saturation.val)
    # print(mixture.get_phase(1).saturation.val)
    # print(pressure)
    # print(gravity_value)
    # print(left_restriction)
    # print(right_restriction)
    # print(expansion_matrix)
    # print(transmissibility_internal_tpfa)
    # print(ad)
    # print(dynamic_viscosity)
    # print(dim_max)

    # pdb.set_trace()


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
