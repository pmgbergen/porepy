import numpy as np
import porepy as pp

import os
import sys
import pdb

os.system("clear")

"""
old tests regarding the hybrid upwind before its integration into the model 

"""


def myprint(var):
    print("\n" + var + " = ", eval(var))


print("\n\nTESTS: -------------------\n")


def test_expansion_matrix():
    """ """
    nx = 2
    ny = 2
    sd = pp.CartGrid(np.array([nx, ny]))
    sd.compute_geometry()
    # pp.plot_grid(sd, vector_value=0.2 * sd.face_normals, info="cf", alpha=0)

    hu = pp.HybridUpwind()

    E = hu.expansion_matrix(sd)
    var_faces = np.array([1, 2, 3, 4])
    var = E @ var_faces

    assert np.all(var == np.array([0, 1, 0, 0, 2, 0, 0, 0, 3, 4, 0, 0]))


def test_restriction_matrices():
    """ """
    nx = 2
    ny = 2
    sd = pp.CartGrid(np.array([nx, ny]))
    sd.compute_geometry()
    # pp.plot_grid(sd, vector_value=0.2*sd.face_normals, info='cf', alpha=0)

    hu = pp.HybridUpwind()
    L, R = hu.restriction_matrices_left_right(sd)

    var = np.arange(sd.num_cells)

    var_left = L * var
    var_right = R * var

    assert np.all(var_left == np.array([0, 2, 0, 1]))
    assert np.all(var_right == np.array([1, 3, 2, 3]))

    var_left = L @ var
    var_right = R @ var

    assert np.all(var_left == np.array([0, 2, 0, 1]))
    assert np.all(var_right == np.array([1, 3, 2, 3]))


class ConstantDensityPhase(pp.Phase):
    """ """

    def mass_density(self, p):
        """ """
        if isinstance(p, pp.ad.AdArray):
            rho = self._rho0 * pp.ad.AdArray(
                np.ones(p.val.shape), 0 * p.jac
            )  # TODO: is it right?
        else:
            rho = self._rho0 * np.ones(p.shape)
        return rho


def test_total_flux_no_gravity(ad):
    """
    TODO: fix the mobility model
    """
    nx = 1
    ny = 2
    sd = pp.CartGrid(np.array([nx, ny]))
    sd.compute_geometry()
    # pp.plot_grid(sd, vector_value=0.2*sd.face_normals, info='cf', alpha=0)

    data = pp.initialize_default_data(grid=sd, data={}, parameter_type="flow")

    wetting_phase = ConstantDensityPhase(rho0=1)
    non_wetting_phase = ConstantDensityPhase(rho0=0.5)

    if ad:
        pressure_val = np.array([1, 0])  # s.t. gradient = 1 and delta_pot = 1

        wetting_saturation_val = np.ones(sd.num_cells)
        non_wetting_saturation_val = (
            np.ones(sd.num_cells) - wetting_saturation_val
        )  # the constraint is applied here. TODO: not clear... improve it

        primary_vars = pp.ad.initAdArrays([pressure_val, wetting_saturation_val])

        pressure = primary_vars[0]
        wetting_phase._s = primary_vars[1]
        non_wetting_phase._s = 1 - primary_vars[1]

    else:
        pressure = np.array([1, 0])  # s.t. gradient = 1 and delta_pot = 1

        wetting_phase._s = np.ones(sd.num_cells, dtype=np.complex128)
        non_wetting_phase._s = np.ones(sd.num_cells) - wetting_phase._s

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    gravity_value = 0

    hu = pp.HybridUpwind()
    L, R = hu.restriction_matrices_left_right(sd)
    hu.compute_transmissibility_tpfa(sd, data)
    _, transmissibility_internal_tpfa = hu.get_transmissibility_tpfa(sd, data)

    dynamic_viscosity = 1

    qt_internal = hu.total_flux(
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
    qt_internal = qt_internal[0] + qt_internal[1]

    # doing the math, lambda_wa has = 1 so total flux is 1

    if ad:
        assert qt_internal.val == np.array([1])

    else:
        assert np.imag(qt_internal) == 0
        qt_internal = np.real(qt_internal)
        assert qt_internal == np.array([1])


def test_total_flux_no_pressure(ad):
    """ """
    nx = 1
    ny = 2
    sd = pp.CartGrid(np.array([nx, ny]))
    sd.compute_geometry()

    data = pp.initialize_default_data(grid=sd, data={}, parameter_type="flow")

    wetting_phase = ConstantDensityPhase(rho0=1)
    non_wetting_phase = ConstantDensityPhase(rho0=0.5)

    if ad:
        pressure_val = np.array([0, 0])

        wetting_saturation_val = np.ones(sd.num_cells)

        primary_vars = pp.ad.initAdArrays([pressure_val, wetting_saturation_val])

        pressure = primary_vars[0]
        wetting_phase._s = primary_vars[1]
        non_wetting_phase._s = (
            1 - primary_vars[1]
        )  # the constraint is applied here. TODO: not clear... improve it

    else:
        pressure = np.array([0, 0])

        wetting_phase._s = np.ones(sd.num_cells, dtype=np.complex128)
        non_wetting_phase._s = np.ones(sd.num_cells) - wetting_phase._s

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    gravity_value = 1

    hu = pp.HybridUpwind()
    L, R = hu.restriction_matrices_left_right(sd)
    hu.compute_transmissibility_tpfa(sd, data)
    _, transmissibility_internal_tpfa = hu.get_transmissibility_tpfa(sd, data)

    dynamic_viscosity = 1
    qt_internal = hu.total_flux(
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
    qt_internal = qt_internal[0] + qt_internal[1]

    # doing the math, lambda_wa has = 1 so total flux is 1

    if ad:
        assert np.isclose(qt_internal.val, np.array([-1.0]), rtol=0, atol=1e-10)
    else:
        assert np.imag(qt_internal) == 0
        qt_internal = np.real(qt_internal)
        assert np.isclose(
            qt_internal, np.array([-1.0]), rtol=0, atol=1e-10
        )  # there is an epsilon in density_internal_faces which make the result slightly different from 1


def test_total_flux_null(ad):
    """ """
    nx = 1
    ny = 2
    sd = pp.CartGrid(np.array([nx, ny]))
    sd.compute_geometry()

    data = pp.initialize_default_data(grid=sd, data={}, parameter_type="flow")

    wetting_phase = ConstantDensityPhase(rho0=1)
    non_wetting_phase = ConstantDensityPhase(rho0=0.5)

    if ad:
        pressure_val = np.array([1, 0])

        wetting_saturation_val = np.ones(sd.num_cells)

        primary_vars = pp.ad.initAdArrays([pressure_val, wetting_saturation_val])

        pressure = primary_vars[0]
        wetting_phase._s = primary_vars[1]
        non_wetting_phase._s = (
            1 - primary_vars[1]
        )  # the constraint is applied here. TODO: not clear... improve it

    else:
        pressure = np.array([1, 0])

        wetting_phase._s = np.ones(sd.num_cells, dtype=np.complex128)
        non_wetting_phase._s = np.ones(sd.num_cells) - wetting_phase._s

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    gravity_value = 1

    hu = pp.HybridUpwind()
    L, R = hu.restriction_matrices_left_right(sd)
    hu.compute_transmissibility_tpfa(sd, data)
    _, transmissibility_internal_tpfa = hu.get_transmissibility_tpfa(sd, data)

    dynamic_viscosity = 1

    qt_internal = hu.total_flux(
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
    qt_internal = qt_internal[0] + qt_internal[1]

    if ad:
        assert np.isclose(qt_internal.val, np.array([0.0]), rtol=0, atol=1e-10)
    else:
        assert np.imag(qt_internal) == 0
        qt_internal = np.real(qt_internal)
        assert np.isclose(
            qt_internal, np.array([0.0]), rtol=0, atol=1e-10
        )  # there is an epsilon in density_internal_faces which make the result slightly different from 1


def test_total_flux_jac():
    """
    TODO: redo this test, no analitical solutoin
    Remark: same jacobians with wetting_saturation = np.array([0.7, 0.9]). I dont know the analytical solution
    """
    nx = 1
    ny = 2
    sd = pp.CartGrid(np.array([nx, ny]))
    sd.compute_geometry()
    # pp.plot_grid(sd, vector_value=0.2 * sd.face_normals, info="cf", alpha=0)

    data = pp.initialize_default_data(grid=sd, data={}, parameter_type="flow")

    wetting_phase = ConstantDensityPhase(rho0=1)
    non_wetting_phase = ConstantDensityPhase(rho0=0.5)

    gravity_value = 1

    # ad: ------------------
    pressure_val = np.array([1, 0])

    wetting_saturation_val = np.ones(sd.num_cells)
    # wetting_saturation_val = np.array([0.7, 0.9])

    primary_vars = pp.ad.initAdArrays([pressure_val, wetting_saturation_val])

    pressure = primary_vars[0]
    wetting_phase._s = primary_vars[1]
    non_wetting_phase._s = (
        1 - primary_vars[1]
    )  # the constraint is applied here. TODO: not clear... improve it

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    dynamic_viscosity = 1

    hu = pp.HybridUpwind()
    ad = True
    _, qt_jacobian_ad = hu.compute_jacobian_qt_ad(
        sd, data, mixture, pressure, gravity_value, ad, dynamic_viscosity
    )

    # complex step: ----------------------
    pressure = np.array([1, 0], dtype=np.complex128)

    wetting_phase._s = np.ones(sd.num_cells, dtype=np.complex128)
    # wetting_phase._s = np.array([0.7, 0.9], dtype=np.complex128)
    non_wetting_phase._s = np.ones(sd.num_cells) - wetting_phase._s

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    ell = 0  ### TODO: this is a mess... it will be naturalluy solved removing finite differences and complex step
    ad = False
    qt_jacobian_complex = hu.compute_jacobian_qt_complex(
        sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
    )

    # finite difference: ----------------------
    qt_jacobian_finite_diff = hu.compute_jacobian_qt_finite_diff(
        sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
    )

    assert np.all(np.imag(qt_jacobian_complex) == 0)
    qt_jacobian_complex = np.real(qt_jacobian_complex)

    assert np.all(np.imag(qt_jacobian_finite_diff) == 0)
    qt_jacobian_finite_diff = np.real(qt_jacobian_finite_diff)

    qt_jacobian_exact = np.array(
        [[1, -1, 0, 0], [-1, 1, 0, 0]]
    )  # remeber that delta_Phi = 0, lamnda = 1 or 0, pressure is in beta,  delta = i-j and first row is about i and second about j
    # TODO: please, check the analytical expression another time

    assert np.all(
        np.isclose(
            qt_jacobian_ad - qt_jacobian_exact,
            np.zeros(qt_jacobian_ad.shape),
            rtol=0,
            atol=1e-10,
        )
    )  # TODO: EH? why did you compare the difference with zero?
    assert np.all(
        np.isclose(
            qt_jacobian_complex - qt_jacobian_exact,
            np.zeros(qt_jacobian_ad.shape),
            rtol=0,
            atol=1e-10,
        )
    )
    assert np.all(
        np.isclose(
            qt_jacobian_finite_diff - qt_jacobian_exact,
            np.zeros(qt_jacobian_ad.shape),
            rtol=0,
            atol=1e-5,
        )
    )


def test_flux_V():
    """
    only ad

    """
    ad = True

    nx = 1
    ny = 2
    sd = pp.CartGrid(np.array([nx, ny]))
    sd.compute_geometry()
    # pp.plot_grid(sd, vector_value=0.2 * sd.face_normals, info="cf", alpha=0)

    data = pp.initialize_default_data(grid=sd, data={}, parameter_type="flow")

    wetting_phase = ConstantDensityPhase(rho0=1)
    non_wetting_phase = ConstantDensityPhase(rho0=0.5)

    pressure_val = np.array([1, 0])

    wetting_saturation_val = np.ones(sd.num_cells)

    primary_vars = pp.ad.initAdArrays([pressure_val, wetting_saturation_val])

    pressure = primary_vars[0]
    wetting_phase._s = primary_vars[1]
    non_wetting_phase._s = (
        1 - primary_vars[1]
    )  # the constraint is applied here. TODO: not clear... improve it

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    gravity_value = 0

    hu = pp.HybridUpwind()
    L, R = hu.restriction_matrices_left_right(sd)
    hu.compute_transmissibility_tpfa(sd, data)
    _, transmissibility_internal_tpfa = hu.get_transmissibility_tpfa(sd, data)

    dynamic_viscosity = 1

    qt_internal = hu.total_flux(
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
    qt_internal = qt_internal[0] + qt_internal[1]

    ell = 0

    rho_V = hu.rho_flux_V(
        sd, mixture, ell, pressure, qt_internal, L, R, ad, dynamic_viscosity
    )

    assert np.all(rho_V.val == np.array([0, 0, 0, 0, 0, 1, 0]))


def test_flux_G():
    """
    only ad

    """
    ad = True

    nx = 1
    ny = 2
    sd = pp.CartGrid(np.array([nx, ny]))
    sd.compute_geometry()

    data = pp.initialize_default_data(grid=sd, data={}, parameter_type="flow")

    wetting_phase = ConstantDensityPhase(rho0=1)
    non_wetting_phase = ConstantDensityPhase(rho0=0.5)

    pressure_val = np.array([0, 0])

    wetting_saturation_val = np.ones(sd.num_cells)

    primary_vars = pp.ad.initAdArrays([pressure_val, wetting_saturation_val])

    pressure = primary_vars[0]
    wetting_phase._s = primary_vars[1]
    non_wetting_phase._s = (
        1 - primary_vars[1]
    )  # the constraint is applied here. TODO: not clear... improve it

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    gravity_value = 1

    hu = pp.HybridUpwind()
    L, R = hu.restriction_matrices_left_right(sd)
    hu.compute_transmissibility_tpfa(sd, data)
    _, transmissibility_internal_tpfa = hu.get_transmissibility_tpfa(sd, data)

    ell = 0
    dynamic_viscosity = 1

    rho_G = hu.rho_flux_G(
        sd,
        mixture,
        ell,
        pressure,
        gravity_value,
        L,
        R,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
    )
    assert np.all(
        rho_G.val == np.array([0, 0, 0, 0, 0, 0, 0])
    )  # you have only one phase so G = 0


def test_flux_V_G():
    """
    TODO
    """


def test_upwind_direction_V_G():
    """ """
    nx = 1
    ny = y_max = 2
    sd = pp.CartGrid(np.array([nx, ny]))
    sd.compute_geometry()
    # pp.plot_grid(sd, vector_value=0.2 * sd.face_normals, info="cf", alpha=0)

    data = pp.initialize_default_data(grid=sd, data={}, parameter_type="flow")

    wetting_phase = ConstantDensityPhase(rho0=1)
    non_wetting_phase = ConstantDensityPhase(rho0=2)

    pressure_val = np.array(
        [0, 0]
    )  # not required... find a way to remove it withoud errors
    wetting_saturation_val = 0 * np.ones(sd.num_cells)
    wetting_saturation_val[np.where(sd.cell_centers[1] >= y_max / 2)] = 1.0
    primary_vars = pp.ad.initAdArrays([pressure_val, wetting_saturation_val])

    pressure = primary_vars[0]
    wetting_phase._s = primary_vars[1]
    non_wetting_phase._s = (
        1 - primary_vars[1]
    )  # the constraint is applied here. TODO: not clear... improve it

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    hu = pp.HybridUpwind()
    L, R = hu.restriction_matrices_left_right(sd)
    hu.compute_transmissibility_tpfa(sd, data)
    _, transmissibility_internal_tpfa = hu.get_transmissibility_tpfa(sd, data)

    ad = True

    dynamic_viscosity = 1

    # V: ------------------------------------------------
    qt_internal = np.array([1])

    ell = 0
    V = hu.flux_V(sd, mixture, ell, qt_internal, L, R, ad, dynamic_viscosity)

    assert V.val == np.array([0])

    ell = 1
    V = hu.flux_V(sd, mixture, ell, qt_internal, L, R, ad, dynamic_viscosity)
    assert V.val == np.array([1])

    qt_internal = np.array([-1])

    ell = 0
    V = hu.flux_V(sd, mixture, ell, qt_internal, L, R, ad, dynamic_viscosity)
    assert V.val == np.array([-1])

    ell = 1
    V = hu.flux_V(sd, mixture, ell, qt_internal, L, R, ad, dynamic_viscosity)
    assert V.val == np.array([0])

    # G: ----------------------------------------------------------
    pressure = np.array([1, 1])
    gravity_value = 1

    ell = 0
    G = hu.flux_G(
        sd,
        mixture,
        ell,
        pressure,
        gravity_value,
        L,
        R,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
    )

    assert G.val == np.array([0])

    ell = 1
    G = hu.flux_G(
        sd,
        mixture,
        ell,
        pressure,
        gravity_value,
        L,
        R,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
    )

    assert G.val == np.array([0])

    gravity_value = -1
    ell = 0
    G = hu.flux_G(
        sd,
        mixture,
        ell,
        pressure,
        gravity_value,
        L,
        R,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
    )

    assert np.isclose(
        G.val, np.array([-1]), rtol=0, atol=1e-9
    )  # 1e-9? why? isn't it a bit high?

    ell = 1
    G = hu.flux_G(
        sd,
        mixture,
        ell,
        pressure,
        gravity_value,
        L,
        R,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
    )

    assert np.isclose(
        G.val, np.array([1]), rtol=0, atol=1e-9
    )  # pay attention: G is the volumetric flux, not the mass flux, thus np.array([1])


def test_upwind_direction_V_G_3x3():
    """pay attention to the analytical computations"""
    nx = 3
    ny = y_max = 3
    sd = pp.CartGrid(np.array([nx, ny]))
    sd.compute_geometry()
    # pp.plot_grid(sd, vector_value=0.2 * sd.face_normals, info="cf", alpha=0)

    data = pp.initialize_default_data(grid=sd, data={}, parameter_type="flow")

    wetting_phase = ConstantDensityPhase(rho0=1)
    non_wetting_phase = ConstantDensityPhase(rho0=2)

    pressure_val = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    )  # not required... find a way to remove it withoud errors
    wetting_saturation_val = 0 * np.ones(sd.num_cells)
    wetting_saturation_val[np.where(sd.cell_centers[1] >= y_max / 2)] = 1.0
    primary_vars = pp.ad.initAdArrays([pressure_val, wetting_saturation_val])

    pressure = primary_vars[0]
    wetting_phase._s = primary_vars[1]
    non_wetting_phase._s = (
        1 - primary_vars[1]
    )  # the constraint is applied here. TODO: not clear... improve it

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    #
    #

    # pp.plot_grid(
    #     sd,
    #     mixture.get_phase(0).saturation.val,
    #     title="saturation 0, density = "
    #     + str(mixture.get_phase(0).mass_density(pressure).val),
    # )
    # pp.plot_grid(
    #     sd,
    #     mixture.get_phase(1).saturation.val,
    #     title="saturation 1, density = "
    #     + str(mixture.get_phase(1).mass_density(pressure).val),
    # )

    #
    #

    hu = pp.HybridUpwind()
    L, R = hu.restriction_matrices_left_right(sd)
    hu.compute_transmissibility_tpfa(sd, data)
    _, transmissibility_internal_tpfa = hu.get_transmissibility_tpfa(sd, data)

    ad = True

    dynamic_viscosity = 1

    # V: ------------------------------------------------
    qt_internal = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    ell = 0
    V = hu.flux_V(sd, mixture, ell, qt_internal, L, R, ad, dynamic_viscosity)

    assert np.all(V.val == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]))

    ell = 1
    V = hu.flux_V(sd, mixture, ell, qt_internal, L, R, ad, dynamic_viscosity)
    assert np.all(V.val == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]))

    qt_internal = np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1])

    ell = 0
    V = hu.flux_V(sd, mixture, ell, qt_internal, L, R, ad, dynamic_viscosity)
    assert np.all(V.val == np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1]))

    ell = 1
    V = hu.flux_V(sd, mixture, ell, qt_internal, L, R, ad, dynamic_viscosity)
    assert np.all(V.val == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    # G: ----------------------------------------------------------
    pressure = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    gravity_value = 1

    ell = 0
    G = hu.flux_G(
        sd,
        mixture,
        ell,
        pressure,
        gravity_value,
        L,
        R,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
    )

    assert np.all(G.val == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    ell = 1
    G = hu.flux_G(
        sd,
        mixture,
        ell,
        pressure,
        gravity_value,
        L,
        R,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
    )

    assert np.all(G.val == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    gravity_value = -1
    ell = 0
    G = hu.flux_G(
        sd,
        mixture,
        ell,
        pressure,
        gravity_value,
        L,
        R,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
    )

    assert np.all(
        np.isclose(
            G.val, np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0]), rtol=0, atol=1e-9
        )
    )  # 1e-9? why? isn't it a bit high?

    ell = 1
    G = hu.flux_G(
        sd,
        mixture,
        ell,
        pressure,
        gravity_value,
        L,
        R,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
    )

    assert np.all(
        np.isclose(
            G.val, np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]), rtol=0, atol=1e-9
        )
    )  # pay attention: G is the volumetric flux, not the mass flux, thus np.array([1])


def test_full_jacobian():
    """
    Remark: in this test the total flux is null, which is the tricky case becouse the upwind direction is not clearly defined.
    Actually, becouse of epsilons added to avoid division by zero qt is not exactly null.
    I forced it to be exactly = 0 in compute_jacobian_V_G_ad, compute_jacobian_V_G_complex, and compute_jacobian_V_G_finite_diff
    and nothing happend, only the jacobian computed with finite diff was slighlty affected, the others weren't.
    I haven't tried to modified the upwind direction for G, I force only qt to be equal to zero.

    same jacobians with wetting_saturation = np.array([0.7, 0.9])
    """
    nx = 1
    ny = 2
    sd = pp.CartGrid(np.array([nx, ny]))
    sd.compute_geometry()

    data = pp.initialize_default_data(grid=sd, data={}, parameter_type="flow")

    wetting_phase = ConstantDensityPhase(rho0=1)
    non_wetting_phase = ConstantDensityPhase(rho0=0.5)

    gravity_value = 1
    ell = 0
    bc_val = np.zeros(sd.num_faces)

    # ad: ------------------
    pressure_val = np.array([1, 0])
    wetting_saturation_val = np.ones(sd.num_cells)
    # wetting_saturation_val = np.array([0.7, 0.9])
    primary_vars = pp.ad.initAdArrays([pressure_val, wetting_saturation_val])

    pressure = primary_vars[0]
    wetting_phase._s = primary_vars[1]
    non_wetting_phase._s = (
        1 - primary_vars[1]
    )  # the constraint is applied here. TODO: not clear... improve it

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    hu = pp.HybridUpwind()
    dynamic_viscosity = 1

    ad = True
    _, A_ad, _ = hu.assemble_matrix_rhs_ad(
        sd, data, mixture, pressure, ell, gravity_value, bc_val, ad, dynamic_viscosity
    )

    # complex: ---------------------------
    pressure = np.array([1, 0], dtype=np.complex128)

    wetting_phase._s = np.ones(sd.num_cells, dtype=np.complex128)
    # wetting_phase._s = np.array([0.7, 0.9], dtype=np.complex128)
    non_wetting_phase._s = np.ones(sd.num_cells) - wetting_phase._s

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    ad = False
    A_complex, _ = hu.assemble_matrix_rhs_complex(
        sd, data, mixture, pressure, ell, gravity_value, bc_val, ad, dynamic_viscosity
    )

    # finite differnce: -------------------------
    A_finite_diff, _ = hu.assemble_matrix_rhs_tmp_finite_diff(
        sd, data, mixture, pressure, ell, gravity_value, bc_val, ad, dynamic_viscosity
    )

    A_exact = np.array([[1, -1, 0, 0], [-1, 1, 0, 0], [1, -1, 0, 0], [-1, 1, 0, 0]])

    assert np.all(
        np.isclose(A_ad, A_exact, rtol=0, atol=1e-10)
    )  # TODO: not sure why the zero elements are not exactly zero
    assert np.all(np.isclose(A_complex, A_exact, rtol=0, atol=1e-10))
    assert np.all(np.isclose(A_finite_diff, A_exact, rtol=0, atol=1e-5))


ad = True
test_expansion_matrix()
test_restriction_matrices()
test_total_flux_no_gravity(ad)
test_total_flux_no_pressure(ad)
test_total_flux_null(ad)
test_total_flux_jac()
test_flux_V()
test_flux_G()
test_full_jacobian()
test_upwind_direction_V_G()
test_upwind_direction_V_G_3x3()
print("ufficial tests passed")

print("\n\n dont forget that these tests do not check everything\n\n")


###################################################################################################################
#
###################################################################################################################

# solve one iteration: ---------------------------------------------------------------------

do_this_section = False

np.set_printoptions(precision=2, linewidth=150)


def unstationary_term(
    sd,
    mixture,
    pressure,
    ell,
    porosity,
):
    """tmp
    TODO: find a better name
    """
    volumes = sd.cell_volumes
    rho = mixture.get_phase(ell).mass_density(pressure)
    saturation = mixture.get_phase(ell).saturation
    # lack_of_a_better_name = volumes * porosity * rho * saturation # you wish....
    lack_of_a_better_name = rho * saturation * (volumes * porosity)

    return lack_of_a_better_name


def newton_tmp(
    sd,
    data,
    mixture,
    ell,
    gravity_value,
    bc_val,
    initial_guess,
    timestep,
    dynamic_viscosity,
):
    """
    - Strongly hardcoded
    - null bc
    - equation order: pressure eq, mass balance
    - primary vars order: pressure, saturation
    - initial_guess = solution at previous timestep
    - it is a mess but it is temporary...
    """
    hu = pp.HybridUpwind()

    toll = 1e-12  # no comment...

    if ell == 0:  # sorry...
        m = 1
    else:  # ell == 1
        m = 0

    ad = True

    u_old = initial_guess
    length = u_old[0].val.shape[0]

    # unstationary term pressure eq:
    pressure = u_old[0]
    mixture.get_phase(ell)._s = u_old[1]
    mixture.get_phase(m)._s = 1 - u_old[1]
    unstationary_0 = unstationary_term(sd, mixture, pressure, 0, porosity)
    unstationary_1 = unstationary_term(sd, mixture, pressure, 1, porosity)
    unstationary_0_1 = unstationary_0 + unstationary_1
    rhs_unst_pressure = unstationary_0_1.val

    # unstationary term mass balance:
    unstationary_ell = unstationary_term(sd, mixture, pressure, ell, porosity)
    rhs_unst_mass = unstationary_ell.val

    rhs = np.zeros(2 * sd.num_cells)
    rhs[0 : sd.num_cells] = rhs_unst_pressure
    rhs[sd.num_cells : 2 * sd.num_cells] = rhs_unst_mass

    err = toll + 1
    iteration = 0
    while err > toll:
        print("\n\n iteration = ", iteration)
        pressure = u_old[0]  ###
        mixture.get_phase(ell)._s = u_old[1]
        mixture.get_phase(m)._s = 1 - u_old[1]

        # print("pressure = ", pressure.val)
        # print("mixture.get_phase(ell)._s = ", mixture.get_phase(ell)._s.val)
        # print("mixture.get_phase(m)._s = ", mixture.get_phase(m)._s.val)

        F, JF, _ = hu.assemble_matrix_rhs_ad(
            sd,
            data,
            mixture,
            pressure,
            ell,
            gravity_value,
            bc_val,
            ad,
            dynamic_viscosity,
        )

        # unstationary term pressure eq:
        unstationary_0 = unstationary_term(sd, mixture, pressure, 0, porosity)
        unstationary_1 = unstationary_term(sd, mixture, pressure, 1, porosity)
        unstationary_0_1 = unstationary_0 + unstationary_1
        val_unst_pressure = unstationary_0_1.val
        jac_unst_pressure = unstationary_0_1.jac.A

        # unstationary term mass balance:
        unstationary_ell = unstationary_term(sd, mixture, pressure, ell, porosity)
        val_unst_mass = unstationary_ell.val
        jac_unst_mass = unstationary_ell.jac.A

        val_unstationary = np.zeros(2 * sd.num_cells)
        val_unstationary[0 : sd.num_cells] = val_unst_pressure
        val_unstationary[sd.num_cells : 2 * sd.num_cells] = val_unst_mass

        unstationary_jacobian = np.zeros((2 * sd.num_cells, 2 * sd.num_cells))
        unstationary_jacobian[0 : sd.num_cells, :] = jac_unst_pressure
        unstationary_jacobian[sd.num_cells : 2 * sd.num_cells, :] = jac_unst_mass

        full_J = timestep * JF + unstationary_jacobian

        # print("\n JF = ", JF)
        # print("np.linalg.cond(JF)= ", np.linalg.cond(JF))
        # print("\n full_J = ", full_J)
        # print("np.linalg.cond(full_J)= ", np.linalg.cond(full_J))

        # residual = np.linalg.solve(JF, F)
        residual = np.linalg.solve(full_J, timestep * F + val_unstationary - rhs)
        u_new = pp.ad.initAdArrays([u_old[0].val, u_old[1].val])
        u_new[0].val -= residual[0:length]  # sorry...
        u_new[1].val -= residual[length : 2 * length]
        err = np.linalg.norm(residual, ord=2)
        u_old = u_new
        iteration += 1

        print("err = ", err)

    return u_new


if do_this_section:
    x_max = nx = 10
    y_max = ny = 10
    sd = pp.CartGrid(np.array([nx, ny]))
    sd = pp.StructuredTriangleGrid(np.array([nx, ny]))
    sd.compute_geometry()

    data = pp.initialize_default_data(grid=sd, data={}, parameter_type="flow")
    wetting_phase = pp.Phase(rho0=1)  # pp.Phase(rho0=1000)
    non_wetting_phase = pp.Phase(rho0=1.5)  # pp.Phase(rho0=1500)

    use_ad = False
    pressure_val = np.ones(sd.num_cells)  # 1e5 * np.ones(sd.num_cells)
    # pressure_val = 2 - 1 * sd.cell_centers[1] / y_max  # the initial guess counts! (?)
    # pressure_val[np.where(sd.cell_centers[1] >= y_max / 2)] = 0.5  # 5e4

    non_wetting_saturation_val = 0.0 * np.ones(sd.num_cells)
    non_wetting_saturation_val[np.where(sd.cell_centers[1] >= y_max / 2)] = 1.0

    wetting_saturation_val = np.ones(sd.num_cells) - non_wetting_saturation_val

    primary_vars = pp.ad.initAdArrays([pressure_val, non_wetting_saturation_val])
    pressure = primary_vars[0]  # useless for newton. TODO: remove it
    wetting_phase._s = 1 - primary_vars[1]  # useless for newton. TODO: remove it
    non_wetting_phase._s = primary_vars[1]  # useless for newton. TODO: remove it

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    pp.plot_grid(sd, pressure_val, title="pressure", info="c", alpha=0.5)
    pp.plot_grid(
        sd,
        mixture.get_phase(0).saturation.val,
        title="saturation 0, density = "
        + str(mixture.get_phase(0).mass_density(pressure).val),
    )
    pp.plot_grid(
        sd,
        mixture.get_phase(1).saturation.val,
        title="saturation 1, density = "
        + str(mixture.get_phase(1).mass_density(pressure).val),
    )

    # let's try newton: --------------------------------------------------
    gravity_value = 1  # 9.81
    ell = 1  # 0 = wetting phase, 1 = non-wetting phase
    bc_val = np.zeros(
        sd.num_faces
    )  # TODO: sd.num_faces().shape? or sd.get_all_boundary_faces().shape?
    initial_guess = primary_vars
    timestep = 0.1
    porosity = 0.5 * np.ones(sd.num_cells)
    dynamic_viscosity = 1

    time_list = np.arange(0, 20, step=timestep)
    for time in time_list:
        sol = newton_tmp(
            sd,
            data,
            mixture,
            ell,
            gravity_value,
            bc_val,
            initial_guess,
            timestep,
            dynamic_viscosity,
        )

        if np.mod(time, 0.5) == 0:
            # let's plot the solution: ----------------
            print("\n\n")
            print("time = ", time)
            np.set_printoptions(precision=6)
            print("ell = ", ell)
            print(
                "the primary saturation is about the non wetting, i.e., the heavier and the mass balance is wrt that phase\n\n"
            )
            print("primary varibles: ")
            print("pressure = ", sol[0].val)
            print("saturation ell =", sol[1].val)

            # pp.plot_grid(sd, sol[0].val, title="pressure", info="c", alpha=0.5)

            print(
                "deisity in this plot = ",
                mixture.get_phase(ell).mass_density(pressure).val,
                "non wetting, i.e., heavy",
            )
            pp.plot_grid(
                sd,
                sol[1].val,
                title="saturation " + str(ell) + " = " + str(sol[1].val),
            )

            if ell == 0:
                m = 1
            else:
                m = 0
            # print(
            #     "deisity in this plot = ",
            #     mixture.get_phase(m).mass_density(pressure).val,
            #     "wetting, i.e., light",
            # )
            # pp.plot_grid(
            #     sd,
            #     1 - sol[1].val,
            #     title="saturation " + str(m) + " = " + str(1 - sol[1].val),
            # )
        initial_guess = sol
