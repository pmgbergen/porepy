import numpy as np
import scipy as sp
import porepy as pp
import copy
import pdb


from porepy.numerics.discretization import Discretization


def myprint(var):
    print("\n" + var + " = ", eval(var))


class Hu:  
    """
    procedurally paradigm is adopted
    """

    def __init__(self, keyword: str = "transport") -> None:
        """ """
        self.keyword = keyword

    @staticmethod
    def expansion_matrix(sd):
        """
        from internal faces set to all faces set
        TODO: change the name
        """
        data = np.ones(sd.get_internal_faces().shape[0])
        rows = sd.get_internal_faces()
        cols = np.arange(sd.get_internal_faces().shape[0])

        expansion = sp.sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(sd.num_faces, sd.get_internal_faces().shape[0]),
        )
        return expansion

    @staticmethod
    def restriction_matrices_left_right(sd):
        """
        remark: with ad you have to use matrices instead of working with indices
        TODO: PAY ATTENTION: there are two logical operation in the same function (improve it):
              get internal set of faces and compute left and right restriction of internal set
        TODO: this function was essentially copied from email. Improve it if possible.

        var_left = left_restriction @ var
        var_left.shape = (len(sd.get_internal_faces()),)
        var.shape = (sd.num_faces,)
        """

        internal_faces = sd.get_internal_faces()

        cell_left_internal_id = sd.cell_face_as_dense()[
            0, internal_faces
        ]  # left cells id of the internal faces subset
        cell_right_internal_id = sd.cell_face_as_dense()[
            1, internal_faces
        ]  # right cells id of the internal faces subset

        data_l = np.ones(cell_left_internal_id.size)
        rows_l = np.arange(cell_left_internal_id.size)
        cols_l = cell_left_internal_id

        left_restriction = sp.sparse.coo_matrix(
            (data_l, (rows_l, cols_l)), shape=(cell_left_internal_id.size, sd.num_cells)
        )

        data_r = np.ones(cell_right_internal_id.size)
        rows_r = np.arange(cell_right_internal_id.size)
        cols_r = cell_right_internal_id

        right_restriction = sp.sparse.coo_matrix(
            (data_r, (rows_r, cols_r)), shape=(cell_left_internal_id.size, sd.num_cells)
        )

        return left_restriction, right_restriction

    @staticmethod  
    def density_internal_faces(
        saturation, density, left_restriction, right_restriction
    ):
        """ """
        s_rho = saturation * density

        density_internal_faces = (
            left_restriction @ s_rho + right_restriction @ s_rho
        ) / (
            left_restriction @ saturation + right_restriction @ saturation + 1e-10
        )  # added epsilon to avoid division by zero
        return density_internal_faces

    @staticmethod
    def g_internal_faces(
        z, density_faces, gravity_value, left_restriction, right_restriction
    ):
        """ """
        g_faces = (
            density_faces
            * gravity_value
            * (left_restriction * z - right_restriction * z)
        )
        return g_faces

    @staticmethod
    def compute_transmissibility_tpfa(sd, data, keyword="flow"): 
        """ """
        discr = pp.Tpfa(keyword)
        discr.discretize(sd, data)

    @staticmethod  
    def get_transmissibility_tpfa(sd, data, keyword="flow"):  
        """ """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]
        div_transmissibility = matrix_dictionary["flux"]  # TODO: "flux"
        transmissibility = matrix_dictionary["transmissibility"]
        transmissibility_internal = transmissibility[sd.get_internal_faces()]

        # TODO: you can always use the matrix, so do it. remark: for mpfa you can only use the matrix...

        return transmissibility, transmissibility_internal  

    @staticmethod
    def var_upwinded_faces(var, upwind_directions, left_restriction, right_restriction):
        """
        - works for both ad and non ad
        - var defined at cell centers
        - remark: no derivative wrt upwind direction => we use only the val if ad or real part if complex step
        """
        var_left = left_restriction @ var
        var_right = right_restriction @ var

        if isinstance(upwind_directions, pp.ad.AdArray):
            upwind_directions = upwind_directions.val  

        upwind_left = np.maximum(
            0, np.heaviside(np.real(upwind_directions), 1)
        )  # attention: I'm using only the real part
        upwind_right = (
            np.ones(upwind_directions.shape) - upwind_left
        )  # what's not left is right (here!)

        upwind_left_matrix = sp.sparse.diags(
            upwind_left
        )  
        upwind_right_matrix = sp.sparse.diags(upwind_right)

        var_upwinded = upwind_left_matrix @ var_left + upwind_right_matrix @ var_right

        return var_upwinded


    @staticmethod  
    def total_flux(
        sd: pp.Grid,
        mixture,
        pressure,
        gravity_value,
        left_restriction,
        right_restriction,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity
    ):

        def gamma_value():
            """ """
            alpha = 1  # as in the paper 2022

            kr0 = pp.rel_perm_brooks_corey(saturation=1)  # TODO: is it right?
            dd_kr_max = pp.second_derivative(
                np.arange(0, 1, 10)
            ).max()  # TODO: improve it...

            gamma_val = alpha / kr0 * dd_kr_max
            return gamma_val

        def g_ref_faces(
            mixture, pressure, z, gravity_value, left_restriction, right_restriction
        ):
            """
            harcoded for two phases
            """
            density_faces_0 = Hu.density_internal_faces(
                mixture.get_phase(0).saturation,
                mixture.get_phase(0).mass_density(pressure),
                left_restriction,
                right_restriction,
            )
            density_faces_1 = Hu.density_internal_faces(
                mixture.get_phase(1).saturation,
                mixture.get_phase(1).mass_density(pressure),
                left_restriction,
                right_restriction,
            )

            density_max = pp.ad.functions.maximum(density_faces_0, density_faces_1)

            g_ref = (
                density_max
                * gravity_value
                * (left_restriction * z - right_restriction * z)
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
            density_internal_faces = Hu.density_internal_faces(
                saturation, density, left_restriction, right_restriction
            )
            g_internal_faces = Hu.g_internal_faces(
                z,
                density_internal_faces,
                gravity_value,
                left_restriction,
                right_restriction,
            )
            delta_pot = (
                left_restriction @ pressure
                - right_restriction @ pressure
                - g_internal_faces
            )
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
                beta_faces = 0.5 + 1 / np.pi * pp.ad.functions.arctan(
                    tmp * delta_pot_faces
                )

            else:
                tmp = (
                    np.minimum(np.real(tmp), 1e6) + np.imag(tmp) * 1j
                )  # TODO: improve it
                beta_faces = 0.5 + 1 / np.pi * np.arctan(tmp * delta_pot_faces)

            return beta_faces

        def lambda_WA_faces(beta_faces, mobility, left_restriction, right_restriction):
            """ """
            lambda_WA = beta_faces * (left_restriction @ mobility) + (
                1 - beta_faces
            ) * (right_restriction @ mobility)
            return lambda_WA

        # total flux computation:
        z = -sd.cell_centers[
            sd.dim - 1
        ]  # TODO: this is wrong, works only in 2D ### zed is reversed to conform to the notation in paper 2022 
        g_ref_faces = g_ref_faces(
            mixture, pressure, z, gravity_value, left_restriction, right_restriction
        )  
        gamma_val = gamma_value()

        total_flux = [None]*mixture.num_phases

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
            mobility = pp.mobility(saturation_m, dynamic_viscosity)  # TODO: impreve it

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

    @staticmethod
    def rho_total_flux(
        sd: pp.Grid,
        mixture,
        pressure,
        gravity_value,
        left_restriction,
        right_restriction,
        transmissibility_internal_tpfa,
        ad, dynamic_viscosity
    ):
        """ """
        qt = Hu.total_flux(
            sd,
            mixture,
            pressure,
            gravity_value,
            left_restriction,
            right_restriction,
            transmissibility_internal_tpfa,
            ad, dynamic_viscosity
        )

        rho_qt = [None, None]
        for m in np.arange(mixture.num_phases):
            saturation_m = mixture.get_phase(m).saturation
            density_m = mixture.get_phase(m).mass_density(pressure)
            rho_m = Hu.density_internal_faces(
                saturation_m, density_m, left_restriction, right_restriction
            )

            rho_qt[m] = rho_m * qt[m]

        rho_qt = rho_qt[0] + rho_qt[1]
        return rho_qt

    @staticmethod  
    def flux_V(
        sd, mixture, ell, total_flux_internal, left_restriction, right_restriction, ad, dynamic_viscosity
    ):  
        """ """

        def mobility_V_faces(
            saturation, total_flux_internal, left_restriction, right_restriction, dynamic_viscosity
        ):
            """ """

            mobility_upwinded = Hu.var_upwinded_faces(
                pp.mobility(saturation, dynamic_viscosity),
                total_flux_internal,
                left_restriction,
                right_restriction,
            )
            return mobility_upwinded

        def mobility_tot_V_faces(
            saturation_list,
            total_flux_internal,
            left_restriction,
            right_restriction,
            ad,
        ):
            """ """

            if ad:
                mobility_tot = pp.ad.AdArray(
                    np.zeros(left_restriction.shape[0]),
                    0 * sp.sparse.eye(left_restriction.shape[0], 2 * sd.num_cells),
                )  # TODO: find a smart way to initialize ad vars
            else:
                mobility_tot = np.zeros(
                    left_restriction.shape[0], dtype=np.complex128
                )  # TODO: improve it, left_restriction.shape[0] = len(sd.get_internal_faces())

            for m in np.arange(mixture.num_phases):
                mobility_tot += mobility_V_faces(
                    saturation_list[m],
                    total_flux_internal,
                    left_restriction,
                    right_restriction, dynamic_viscosity
                )
            return mobility_tot

        # V (viscous/convective) flux computation:
        saturation_list = [None] * mixture.num_phases
        for phase_id in np.arange(mixture.num_phases):
            saturation_list[phase_id] = mixture.get_phase(phase_id).saturation

        mob_V = mobility_V_faces(
            saturation_list[ell],
            total_flux_internal,
            left_restriction,
            right_restriction, dynamic_viscosity
        )  
        mob_tot_V = mobility_tot_V_faces(
            saturation_list,
            total_flux_internal,
            left_restriction,
            right_restriction,
            ad,
        )
        V_internal = mob_V / mob_tot_V * total_flux_internal

        return V_internal

    @staticmethod
    def rho_flux_V(
        sd,
        mixture,
        ell,
        pressure,
        total_flux_internal,
        left_restriction,
        right_restriction,
        ad, dynamic_viscosity
    ):
        """ """
        V = Hu.flux_V(
            sd,
            mixture,
            ell,
            total_flux_internal,
            left_restriction,
            right_restriction,
            ad, dynamic_viscosity
        )
        density = mixture.get_phase(ell).mass_density(pressure)
        rho_upwinded = Hu.var_upwinded_faces(
            density, V, left_restriction, right_restriction
        )
        rho_V_internal = rho_upwinded * V

        expansion = Hu.expansion_matrix(sd)
        rho_V = expansion @ rho_V_internal

        return rho_V

    @staticmethod  
    def flux_G(
        sd,
        mixture,
        ell,
        pressure,
        gravity_value,
        left_restriction,
        right_restriction,
        transmissibility_internal_tpfa,
        ad, dynamic_viscosity
    ):
        """
        TODO: consider the idea to move omega outside the flux_G, if you don't see why => it is already in the right place.
        """

        def omega(
            num_phases, ell, mobilities, g, left_restriction, right_restriction, ad
        ):
            """
            TODO: i run into some issues with pp.ad.functions.heaviside
            """
            if ad:
                omega_ell = pp.ad.AdArray(
                    np.zeros(left_restriction.shape[0]),
                    0 * sp.sparse.eye(left_restriction.shape[0], 2 * sd.num_cells),
                ) # TODO: find a better way to initialize arrays

                for m in np.arange(num_phases):
                    omega_ell += (
                        (left_restriction @ mobilities[m])
                        * pp.ad.functions.heaviside(-g[m] + g[ell])
                        + (right_restriction @ mobilities[m])
                        * pp.ad.functions.heaviside(g[m] - g[ell])
                    ) * (g[m] - g[ell])

            else:
                omega_ell = np.zeros(left_restriction.shape[0], dtype=np.complex128)

                for m in np.arange(num_phases):
                    omega_ell += (
                        (left_restriction @ mobilities[m]) * (g[m] < g[ell])
                        + (right_restriction @ mobilities[m]) * (g[m] > g[ell])
                    ) * (g[m] - g[ell])

            return omega_ell

        def mobility_G_faces(saturation, omega_ell, left_restriction, right_restriction, dynamic_viscosity):
            """ """
            mobility_upwinded = Hu.var_upwinded_faces(
                pp.mobility(saturation, dynamic_viscosity),
                omega_ell,
                left_restriction,
                right_restriction,
            )
            return mobility_upwinded  

        def mobility_tot_G_faces(
            num_phases,
            saturation_list,
            omega_ell,
            left_restriction,
            right_restriction,
        ):
            """ """
            if ad:
                mobility_tot_G = pp.ad.AdArray(
                    np.zeros(left_restriction.shape[0]),
                    0 * sp.sparse.eye(left_restriction.shape[0], 2 * sd.num_cells))
            else:
                mobility_tot_G = np.zeros(
                    left_restriction.shape[0], dtype=np.complex128)


            for m in np.arange(num_phases):  # m = phase_id
                mobility_tot_G += mobility_G_faces(
                    saturation_list[m],
                    omega_ell,
                    left_restriction,
                    right_restriction,dynamic_viscosity
                )

            return mobility_tot_G

        # flux G computation:
        z = -sd.cell_centers[
            sd.dim - 1
        ]  # TODO: this is wrong, works only in 2D ### zed is reversed to conform to paper 2022 notation

        saturation_list = [None] * mixture.num_phases
        g_list = [None] * mixture.num_phases
        mobility_list = [None] * mixture.num_phases
        omega_list = [None] * mixture.num_phases

        for phase_id in np.arange(mixture.num_phases):
            saturation = mixture.get_phase(phase_id).saturation  # ell and m ref paper
            saturation_list[phase_id] = saturation  # TODO: find a better solution
            rho = mixture.get_phase(phase_id).mass_density(pressure)
            rho = Hu.density_internal_faces(
                saturation, rho, left_restriction, right_restriction
            )  # TODO: rho used twice
            g_list[phase_id] = Hu.g_internal_faces(
                z, rho, gravity_value, left_restriction, right_restriction
            )  # TODO: g_ell and g_m are computed twice, one in G and one in omega
            mobility_list[phase_id] = pp.mobility(saturation, dynamic_viscosity)

        for phase_id in np.arange(mixture.num_phases):
            omega_list[phase_id] = omega(
                mixture.num_phases,
                phase_id,
                mobility_list,
                g_list,
                left_restriction,
                right_restriction,
                ad,
            )

        mob_tot_G = mobility_tot_G_faces(
            mixture.num_phases,
            saturation_list,
            omega_list[ell],
            left_restriction,
            right_restriction,
        )

        if ad:
            G_internal = pp.ad.AdArray(
                np.zeros(left_restriction.shape[0]),
                0 * sp.sparse.eye(left_restriction.shape[0], 2 * sd.num_cells),
            )  # TODO: find a smart way to initialize ad vars
        else:
            G_internal = np.zeros(
                left_restriction.shape[0], dtype=np.complex128
            )  # TODO: improve it

        for m in np.arange(mixture.num_phases):
            mob_G_ell = mobility_G_faces(
                saturation_list[ell],
                omega_list[ell],
                left_restriction,
                right_restriction, dynamic_viscosity
            )  # yes, you can move it outside the loop
            mob_G_m = mobility_G_faces(
                saturation_list[m],
                omega_list[m],
                left_restriction,
                right_restriction, dynamic_viscosity
            )
            G_internal += mob_G_ell * mob_G_m / mob_tot_G * (g_list[m] - g_list[ell])

        G_internal *= transmissibility_internal_tpfa
        return G_internal

    @staticmethod
    def rho_flux_G(
        sd,
        mixture,
        ell,
        pressure,
        gravity_value,
        left_restriction,
        right_restriction,
        transmissibility_internal_tpfa,
        ad, dynamic_viscosity
    ):
        """ """
        G = Hu.flux_G(
            sd,
            mixture,
            ell,
            pressure,
            gravity_value,
            left_restriction,
            right_restriction,
            transmissibility_internal_tpfa,
            ad, dynamic_viscosity
        )
        density = mixture.get_phase(ell).mass_density(pressure)
        rho_upwinded = Hu.var_upwinded_faces(
            density, G, left_restriction, right_restriction
        )
        rho_G_internal = rho_upwinded * G

        expansion = Hu.expansion_matrix(sd)
        rho_G = expansion @ rho_G_internal

        return rho_G

    @staticmethod  
    def boundary_conditions_tmp(sd: pp.Grid, bc_val):
        """
        TODO: bc for pressure eq and for mass flux. For now, flux = 0, so I use this method for both

        TODO: bc_val is outwards flux wrt domain, not integrated. => if bc_val negative => entering mass
            change it(?) in according to pp convention
        
            TODO: not 100% sure about the sign, check it

            returns the rhs, in the right size and div included
        """
        # kind of div -abs(sg.cell_faces()).T@sd.face_areas()
        return -abs(sd.cell_faces).T @ (sd.face_areas * bc_val)

    @staticmethod
    def compute_jacobian_qt_ad(sd, data, mixture, pressure, gravity_value, ad, dynamic_viscosity):
        """
        attention: here you don't need ell bcs the primary variable has been identified before computing the flux.
                    for the other jac computation you need ell
        """
        L, R = Hu.restriction_matrices_left_right(sd)
        Hu.compute_transmissibility_tpfa(sd, data)
        _, transmissibility_internal_tpfa = Hu.get_transmissibility_tpfa(sd, data)

        rho_qt_internal = Hu.rho_total_flux(
            sd,
            mixture,
            pressure,
            gravity_value,
            L,
            R,
            transmissibility_internal_tpfa,
            ad, dynamic_viscosity
        )

        pp_div = pp.fvutils.scalar_divergence(sd)

        expansion = Hu.expansion_matrix(sd)
        rho_qt = expansion @ rho_qt_internal

        flux_cell_no_bc = pp_div @ rho_qt

        return flux_cell_no_bc.val, flux_cell_no_bc.jac.A

    @staticmethod
    def compute_jacobian_V_G_ad(sd, data, mixture, ell, pressure, gravity_value, ad, dynamic_viscosity):
        """
        this is a tmp function and conceptually wrong. I momentary need  it to test the jac
        """
        L, R = Hu.restriction_matrices_left_right(sd)
        Hu.compute_transmissibility_tpfa(sd, data)
        _, transmissibility_internal_tpfa = Hu.get_transmissibility_tpfa(sd, data)

        qt_internal = Hu.total_flux(
            sd,
            mixture,
            pressure,
            gravity_value,
            L,
            R,
            transmissibility_internal_tpfa,
            ad, dynamic_viscosity
        )
        qt_internal = qt_internal[0] + qt_internal[1]

        rho_V = Hu.rho_flux_V(sd, mixture, ell, pressure, qt_internal, L, R, ad, dynamic_viscosity)

        rho_G = Hu.rho_flux_G(
            sd,
            mixture,
            ell,
            pressure,
            gravity_value,
            L,
            R,
            transmissibility_internal_tpfa,
            ad, dynamic_viscosity
        )

        pp_div = pp.fvutils.scalar_divergence(sd)

        rho_V_cell_no_bc = pp_div @ rho_V
        rho_G_cell_no_bc = pp_div @ rho_G

        flux_cell_no_bc = rho_V_cell_no_bc + rho_G_cell_no_bc

        return flux_cell_no_bc.val, flux_cell_no_bc.jac.A

    @staticmethod
    def compute_jacobian_qt_complex(
        sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
    ):
        """attention: different logic wrt finite diff to add eps
        attention: here you need ell to know what is the primary variable. in ad you don't
        """

        L, R = Hu.restriction_matrices_left_right(sd)
        Hu.compute_transmissibility_tpfa(sd, data)
        _, transmissibility_internal_tpfa = Hu.get_transmissibility_tpfa(sd, data)

        jacobian = np.zeros((sd.num_cells, 2 * sd.num_cells), dtype=np.complex128)
        eps = 1e-20j

        pp_div = pp.fvutils.scalar_divergence(sd)

        pressure_eps = copy.deepcopy(pressure)  # useless
        for i in np.arange(sd.num_cells):  # TODO: ...
            pressure_eps[i] += eps  ### ...

            rho_qt_internal = Hu.rho_total_flux(
                sd,
                mixture,
                pressure_eps,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad, dynamic_viscosity
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

            rho_qt_internal = Hu.rho_total_flux(
                sd,
                mixture,
                pressure,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad, dynamic_viscosity
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
    def compute_jacobian_V_G_complex(
        sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
    ):
        """attention: different logic wrt finite diff to add eps"""

        L, R = Hu.restriction_matrices_left_right(sd)
        Hu.compute_transmissibility_tpfa(sd, data)
        _, transmissibility_internal_tpfa = Hu.get_transmissibility_tpfa(sd, data)

        jacobian = np.zeros((sd.num_cells, 2 * sd.num_cells), dtype=np.complex128)
        eps = 1e-20j

        pp_div = pp.fvutils.scalar_divergence(sd)

        pressure_eps = copy.deepcopy(pressure)  # useless
        for i in np.arange(sd.num_cells):  # TODO: ...
            pressure_eps[i] += eps  ### ...

            qt = Hu.total_flux(
                sd,
                mixture,
                pressure_eps,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad, dynamic_viscosity
            )
            qt = qt[0] + qt[1]

            V = Hu.rho_flux_V(
                sd,
                mixture,
                ell,
                pressure_eps,
                qt,
                L,
                R,
                ad, dynamic_viscosity
            )

            G = Hu.rho_flux_G(
                sd,
                mixture,
                ell,
                pressure_eps,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad, dynamic_viscosity
            )

            F = V + G

            flux_cell_no_bc = pp_div @ F  # move it
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
            saturation_ell_eps[i] += eps  ### ...
            mixture.get_phase(ell)._s = saturation_ell_eps

            saturation_m_eps[i] -= eps  # the constraint is here...
            mixture.get_phase(m)._s = saturation_m_eps

            qt = Hu.total_flux(
                sd,
                mixture,
                pressure,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad, dynamic_viscosity
            )
            qt = qt[0] + qt[1]

            V = Hu.rho_flux_V(sd, mixture, ell, pressure, qt, L, R, ad, dynamic_viscosity)
            G = Hu.rho_flux_G(
                sd,
                mixture,
                ell,
                pressure,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad, dynamic_viscosity
            )
            F = V + G

            flux_cell_no_bc = pp_div @ F
            jacobian[:, sd.num_cells + i] = np.imag(flux_cell_no_bc) / np.imag(
                eps
            )  # pay attention: jacobian[:,sd.num_cells+i]

            saturation_ell_eps[i] -= eps  ### ...
            saturation_m_eps[i] += eps  ### ...

        return jacobian

    @staticmethod
    def compute_jacobian_qt_finite_diff(
        sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
    ):
        """attention: different logic wrt complex step to add eps"""

        L, R = Hu.restriction_matrices_left_right(sd)
        Hu.compute_transmissibility_tpfa(sd, data)
        _, transmissibility_internal_tpfa = Hu.get_transmissibility_tpfa(sd, data)

        jacobian = np.zeros((sd.num_cells, 2 * sd.num_cells), dtype=np.complex128)
        eps = 1e-5

        pp_div = pp.fvutils.scalar_divergence(sd)

        rho_qt_internal = Hu.rho_total_flux(
            sd,
            mixture,
            pressure,
            gravity_value,
            L,
            R,
            transmissibility_internal_tpfa,
            ad, dynamic_viscosity
        )
        rho_qt = np.zeros(sd.num_faces, dtype=np.complex128)  ### ...
        rho_qt[sd.get_internal_faces()] = rho_qt_internal

        flux_cell_no_bc = pp_div @ rho_qt

        for i in np.arange(sd.num_cells):  # TODO: ...
            pressure_eps = copy.deepcopy(pressure)  ### ...
            pressure_eps[i] += eps  ### ...

            rho_qt_internal_eps = Hu.rho_total_flux(
                sd,
                mixture,
                pressure_eps,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad, dynamic_viscosity
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

            rho_qt_internal_eps = Hu.rho_total_flux(
                sd,
                mixture,
                pressure,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad, dynamic_viscosity
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

    @staticmethod
    def compute_jacobian_V_G_finite_diff(
        sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
    ):
        """attention: different logic wrt complex step to add eps"""

        L, R = Hu.restriction_matrices_left_right(sd)
        Hu.compute_transmissibility_tpfa(sd, data)
        _, transmissibility_internal_tpfa = Hu.get_transmissibility_tpfa(sd, data)

        jacobian = np.zeros((sd.num_cells, 2 * sd.num_cells), dtype=np.complex128)
        eps = 1e-5

        pp_div = pp.fvutils.scalar_divergence(sd)

        qt = Hu.total_flux(
            sd,
            mixture,
            pressure,
            gravity_value,
            L,
            R,
            transmissibility_internal_tpfa,
            ad, dynamic_viscosity
        )
        qt = qt[0] + qt[1]

        V = Hu.rho_flux_V(
            sd,
            mixture,
            ell,
            pressure,
            qt,
            L,
            R,
            ad, dynamic_viscosity
        )
        G = Hu.rho_flux_G(
            sd,
            mixture,
            ell,
            pressure,
            gravity_value,
            L,
            R,
            transmissibility_internal_tpfa,
            ad, dynamic_viscosity
        )
        F = V + G
        flux_cell_no_bc = pp_div @ F

        for i in np.arange(sd.num_cells):  # TODO: ...
            pressure_eps = copy.deepcopy(pressure)  ### ...
            pressure_eps[i] += eps  ### ...

            qt_eps = Hu.total_flux(
                sd,
                mixture,
                pressure_eps,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad, dynamic_viscosity
            )
            qt_eps = qt_eps[0] + qt_eps[1]

            V_eps = Hu.rho_flux_V(
                sd,
                mixture,
                ell,
                pressure_eps,
                qt_eps,
                L,
                R,
                ad, dynamic_viscosity
            )
            G_eps = Hu.rho_flux_G(
                sd,
                mixture,
                ell,
                pressure_eps,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad, dynamic_viscosity
            )
            F_eps = V_eps + G_eps

            flux_cell_no_bc_eps = pp_div @ F_eps
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

            qt_eps = Hu.total_flux(
                sd,
                mixture,
                pressure,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad, dynamic_viscosity
            )
            qt_eps = qt_eps[0] + qt_eps[1]

            V_eps = Hu.rho_flux_V(
                sd,
                mixture,
                ell,
                pressure,
                qt_eps,
                L,
                R,
                ad, dynamic_viscosity
            )
            G_eps = Hu.rho_flux_G(
                sd,
                mixture,
                ell,
                pressure,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad, dynamic_viscosity
            )
            F_eps = V_eps + G_eps

            flux_cell_no_bc_eps = pp_div @ F_eps
            jacobian[:, sd.num_cells + i] = (
                flux_cell_no_bc_eps - flux_cell_no_bc
            ) / eps  # pay attention: jacobian[:,sd.num_cells+i]

            saturation_ell_eps[i] -= eps  ### ...
            saturation_m_eps[i] += eps  ### ...

        return jacobian

    @staticmethod
    def assemble_matrix_rhs_ad(
        sd, data, mixture, pressure, ell, gravity_value, bc_val, ad, dynamic_viscosity
    ):
        """this function is conceptually wrong"""
        flux_qt, jacobian_qt = Hu.compute_jacobian_qt_ad(
            sd, data, mixture, pressure, gravity_value, ad, dynamic_viscosity
        )
        flux_V_G, jacobian_V_G = Hu.compute_jacobian_V_G_ad(
            sd, data, mixture, ell, pressure, gravity_value, ad, dynamic_viscosity
        )

        F = np.zeros(2 * sd.num_cells)
        F[0 : sd.num_cells] = flux_qt
        F[sd.num_cells : 2 * sd.num_cells] = flux_V_G

        JF = np.zeros((2 * sd.num_cells, 2 * sd.num_cells))
        JF[0 : sd.num_cells, 0 : 2 * sd.num_cells] = jacobian_qt
        JF[sd.num_cells : 2 * sd.num_cells, 0 : 2 * sd.num_cells] = jacobian_V_G

        b = Hu.boundary_conditions_tmp(sd, bc_val)

        return F, JF, b

    @staticmethod
    def assemble_matrix_rhs_complex(
        sd, data, mixture, pressure, ell, gravity_value, bc_val, ad, dynamic_viscosity
    ):
        """ """
        jacobian_qt = Hu.compute_jacobian_qt_complex(
            sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
        )
        jacobian_V_G = Hu.compute_jacobian_V_G_complex(
            sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
        )

        assert np.sum(np.imag(jacobian_qt)) == 0
        assert np.sum(np.imag(jacobian_V_G)) == 0

        jacobian_qt = np.real(jacobian_qt)
        jacobian_V_G = np.real(jacobian_V_G)

        A = np.zeros((2 * sd.num_cells, 2 * sd.num_cells))
        A[0 : sd.num_cells, 0 : 2 * sd.num_cells] = jacobian_qt
        A[sd.num_cells : 2 * sd.num_cells, 0 : 2 * sd.num_cells] = jacobian_V_G

        b = Hu.boundary_conditions_tmp(sd, bc_val)
        return A, b

    @staticmethod
    def assemble_matrix_rhs_tmp_finite_diff(
        sd, data, mixture, pressure, ell, gravity_value, bc_val, ad, dynamic_viscosity
    ):
        """ """
        jacobian_qt = Hu.compute_jacobian_qt_finite_diff(
            sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
        )
        jacobian_V_G = Hu.compute_jacobian_V_G_finite_diff(
            sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
        )

        assert np.sum(np.imag(jacobian_qt)) == 0
        assert np.sum(np.imag(jacobian_V_G)) == 0

        jacobian_qt = np.real(jacobian_qt)
        jacobian_V_G = np.real(jacobian_V_G)

        A = np.zeros((2 * sd.num_cells, 2 * sd.num_cells))
        A[0 : sd.num_cells, 0 : 2 * sd.num_cells] = jacobian_qt
        A[sd.num_cells : 2 * sd.num_cells, 0 : 2 * sd.num_cells] = jacobian_V_G

        b = Hu.boundary_conditions_tmp(sd, bc_val)
        return A, b


if __name__ == "__main__":
    print("\n\n See test_hu.py \n\n")
