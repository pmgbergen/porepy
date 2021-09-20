"""
Testing of classical face averageing schemes within the AD framework:
    Arithmetic
    (Lazy) Flux-based upwind
    TODO: Potential-based upwind
    TODO: Harmonic
"""

import porepy as pp
import numpy as np
import pytest

from porepy.numerics.fv.generaltpfaad import FluxBasedUpwindAD, ArithmeticAverageAd
from porepy.numerics.ad.grid_operators import DirBC

def assign_parameters_2d(g, d, param_key, bound_case="neu"):
    """
    Assign parameters for a toy 2d problem

    Parameters
    ----------
    g : grid
    d : data dictionary
    param_key : problem keyword, e.g., "flow"
    bound_case : type of boundary case
        The options are: "neu" for all external boundaries set to no-flux,
        "dir_left", "dir_right", "dir_bottom", and "dir_top" for Dirichlet
        boundary conditions at those sides.

    Returns
    -------
    None.

    """

    k = pp.SecondOrderTensor(kxx=np.ones(g.num_cells))

    top = np.where(np.abs(g.face_centers[1] - 1) < 1e-5)[0]
    bottom = np.where(np.abs(g.face_centers[1]) < 1e-5)[0]
    left = np.where(np.abs(g.face_centers[0]) < 1e-5)[0]
    right = np.where(np.abs(g.face_centers[0] - 1) < 1e-5)[0]

    left_dir = np.array([12, 8])
    right_dir = np.array([12, 8])
    top_dir = np.array([8, 12, 6])
    bottom_dir = np.array([8, 12, 6])

    bc_values = np.zeros(g.num_faces)
    if bound_case == "neu":
        bc = pp.BoundaryCondition(g)
    elif bound_case == "dir_left":
        bc_faces = left
        bc_type = bc_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_values[left] = left_dir
    elif bound_case == "dir_right":
        bc_faces = right
        bc_type = bc_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_values[right] = right_dir
    elif bound_case == "dir_top":
        bc_faces = top
        bc_type = bc_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_values[top] = top_dir
    elif bound_case == "dir_bottom":
        bc_faces = bottom
        bc_type = bc_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_values[bottom] = bottom_dir

    specified_parameters = {"second_order_tensor": k, "bc": bc, "bc_values": bc_values}

    pp.initialize_data(g, d, param_key, specified_parameters)


def test_input_arithmetic():
    """
    Test if an error is raised when a wrong object type is passed
    """

    # Make 2d grid
    gb = pp.meshing.cart_grid([], nx=[3, 2], physdims=[1, 1])
    g = gb.grids_of_dimension(2)[0]
    d = gb.node_props(g)

    # Primary variable
    param_key = "flow"
    pressure_var = "pressure"
    d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}

    # Initiliaze data
    pp.set_state(d)
    pp.set_iterate(d)
    d[pp.STATE][pressure_var] = np.array([30, 100, 4, 2, 10, 50])
    d[pp.STATE][pp.ITERATE][pressure_var] = d[pp.STATE][pressure_var].copy()
    assign_parameters_2d(g, d, param_key, bound_case="neu")

    # AD variables and manager
    grid_list = [g]
    dof_manager = pp.DofManager(gb)
    equation_manager = pp.ad.EquationManager(gb, dof_manager)
    p = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])

    # Perform arithmetic average
    bound_ad = pp.ad.BoundaryCondition(param_key, grids=grid_list)
    dir_bound_ad = DirBC(bound_ad, grid_list)
    arithmetic_avg = ArithmeticAverageAd(g, d, param_key)

    with pytest.raises(TypeError):
        ar_faces_ad = arithmetic_avg(p, dir_bound_ad)
        ar_faces_eval = pp.ad.Expression(ar_faces_ad, dof_manager)
        ar_faces_eval.to_ad(gb=gb)


def test_input_upwind():
    """
    Test if an error is raised when a wrong object type is passed
    """
    
    # Make 2d grid
    gb = pp.meshing.cart_grid([], nx=[3, 2], physdims=[1, 1])
    g = gb.grids_of_dimension(2)[0]
    d = gb.node_props(g)
    
    # Primary variable
    param_key = "flow"
    pressure_var = "p"
    d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}
    
    # Initiliaze data
    pp.set_state(d)
    pp.set_iterate(d)
    d[pp.STATE][pressure_var] = np.array([30, 100, 4, 2, 10, 50])
    d[pp.STATE][pp.ITERATE][pressure_var] = d[pp.STATE][pressure_var].copy()
    assign_parameters_2d(g, d, param_key, bound_case="neu")
    
    # AD variables and manager
    grid_list = [g]
    dof_manager = pp.DofManager(gb)
    equation_manager = pp.ad.EquationManager(gb, dof_manager)
    p = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])
    
    #Assign parameters and perform upwinding
    mpfa_ad = pp.ad.MpfaAd(param_key, grid_list)
    
    bound_ad = pp.ad.BoundaryCondition(param_key, grids=grid_list)
    dir_bound_ad = DirBC(bound_ad, grid_list)
    flux_1p_ad_active = mpfa_ad.flux * p + mpfa_ad.bound_flux * bound_ad
    flux_1p_ad_inactive = (
        mpfa_ad.flux * p.previous_iteration() + mpfa_ad.bound_flux * bound_ad
    )
    upwind = FluxBasedUpwindAD(g, d, param_key)
    
    # Pass only pressure as active variable
    with pytest.raises(TypeError):
        upw_faces_ad = upwind(p, dir_bound_ad, flux_1p_ad_inactive)
        upw_faces_eval = pp.ad.Expression(upw_faces_ad, dof_manager)
        upw_faces_eval.discretize(gb=gb)
        upw_faces_eval.to_ad(gb)
    
    # Pass only flux as active variable
    with pytest.raises(TypeError):
        upw_faces_ad = upwind(p.previous_iteration(), dir_bound_ad, flux_1p_ad_active)
        upw_faces_eval = pp.ad.Expression(upw_faces_ad, dof_manager)
        upw_faces_eval.discretize(gb=gb)
        upw_faces_eval.to_ad(gb)
    
    # Pass both, pressure and fluxes as active variables
    with pytest.raises(TypeError):
        upw_faces_ad = upwind(p, dir_bound_ad, flux_1p_ad_active)
        upw_faces_eval = pp.ad.Expression(upw_faces_ad, dof_manager)
        upw_faces_eval.discretize(gb=gb)
        upw_faces_eval.to_ad(gb)
    

def test_arithmetic_2d():
    """
    Tests face arithmetic average for several combinations of boundary
    conditions
    """

    # Make 2d grid
    gb = pp.meshing.cart_grid([], nx=[3, 2], physdims=[1, 1])
    g = gb.grids_of_dimension(2)[0]
    d = gb.node_props(g)

    # Primary variable
    param_key = "flow"
    pressure_var = "pressure"
    d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}

    # Initiliaze data
    pp.set_state(d)
    pp.set_iterate(d)
    d[pp.STATE][pressure_var] = np.array([30, 100, 4, 2, 10, 50])
    d[pp.STATE][pp.ITERATE][pressure_var] = d[pp.STATE][pressure_var].copy()

    # AD variables and manager
    grid_list = [g]
    dof_manager = pp.DofManager(gb)
    equation_manager = pp.ad.EquationManager(gb, dof_manager)
    p = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])

    # Bound cases
    cases = ["neu", "dir_left", "dir_right", "dir_bottom", "dir_top"]

    # True values
    true_neu_arithmetic = np.array(
        [
            1,
            65,
            52,
            1,
            1,
            6,
            30,
            1,
            1,
            1,
            1,
            16,
            55,
            27,
            1,
            1,
            1,
        ]
    )

    true_left_arithmetic = np.array(
        [
            21,
            65,
            52,
            1,
            5,
            6,
            30,
            1,
            1,
            1,
            1,
            16,
            55,
            27,
            1,
            1,
            1,
        ]
    )

    true_right_arithmetic = np.array(
        [
            1,
            65,
            52,
            8,
            1,
            6,
            30,
            29,
            1,
            1,
            1,
            16,
            55,
            27,
            1,
            1,
            1,
        ]
    )

    true_bottom_arithmetic = np.array(
        [
            1,
            65,
            52,
            1,
            1,
            6,
            30,
            1,
            19,
            56,
            5,
            16,
            55,
            27,
            1,
            1,
            1,
        ]
    )

    true_top_arithmetic = np.array(
        [
            1,
            65,
            52,
            1,
            1,
            6,
            30,
            1,
            1,
            1,
            1,
            16,
            55,
            27,
            5,
            11,
            28,
        ]
    )

    true_values = [
        true_neu_arithmetic,
        true_left_arithmetic,
        true_right_arithmetic,
        true_bottom_arithmetic,
        true_top_arithmetic,
    ]

    for case, true_value in zip(cases, true_values):
        assign_parameters_2d(g, d, param_key, bound_case=case)
        bound_ad = pp.ad.BoundaryCondition(param_key, grids=grid_list)
        dir_bound_ad = DirBC(bound_ad, grid_list)
        arithmetic_avg = ArithmeticAverageAd(g, d, param_key)
        ar_faces_ad = arithmetic_avg(p.previous_iteration(), dir_bound_ad)
        ar_faces_eval = pp.ad.Expression(ar_faces_ad, dof_manager)
        ar_faces_num = ar_faces_eval.to_ad(gb)
        ar_faces = ar_faces_num.diagonal()
        np.testing.assert_array_equal(ar_faces, true_value)

def test_upwind_2d():
    """
    Tests face upwind for several combinations of boundary conditions
    """

    # Make 2d grid
    gb = pp.meshing.cart_grid([], nx=[3, 2], physdims=[1, 1])
    g = gb.grids_of_dimension(2)[0]
    d = gb.node_props(g)

    # Primary variable
    param_key = "flow"
    pressure_var = "pressure"
    d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}

    # Initiliaze data
    pp.set_state(d)
    pp.set_iterate(d)
    d[pp.STATE][pressure_var] = np.array([30, 100, 4, 2, 10, 50])
    d[pp.STATE][pp.ITERATE][pressure_var] = d[pp.STATE][pressure_var].copy()

    # AD variables and manager
    grid_list = [g]
    dof_manager = pp.DofManager(gb)
    equation_manager = pp.ad.EquationManager(gb, dof_manager)
    p = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])

    # Bound cases
    cases = ["neu", "dir_left", "dir_right", "dir_bottom", "dir_top"]

    # True values
    true_neu_upwind = np.array(
        [
            1,
            100,
            100,
            1,
            1,
            10,
            50,
            1,
            1,
            1,
            1,
            30,
            100,
            50,
            1,
            1,
            1,
        ]
    )

    true_left_upwind = np.array(
        [
            30,
            100,
            100,
            1,
            8,
            10,
            50,
            1,
            1,
            1,
            1,
            30,
            100,
            50,
            1,
            1,
            1,
        ]
    )

    true_right_upwind = np.array(
        [
            1,
            100,
            100,
            12,
            1,
            10,
            50,
            50,
            1,
            1,
            1,
            30,
            100,
            50,
            1,
            1,
            1,
        ]
    )

    true_bottom_upwind = np.array(
        [
            1,
            100,
            100,
            1,
            1,
            10,
            50,
            1,
            30,
            100,
            6,
            30,
            100,
            50,
            1,
            1,
            1,
        ]
    )

    true_top_upwind = np.array(
        [
            1,
            100,
            100,
            1,
            1,
            10,
            50,
            1,
            1,
            1,
            1,
            30,
            100,
            50,
            8,
            12,
            50,
        ]
    )

    true_values = [
        true_neu_upwind,
        true_left_upwind,
        true_right_upwind,
        true_bottom_upwind,
        true_top_upwind,
    ]

    for case, true_value in zip(cases, true_values):
        assign_parameters_2d(g, d, param_key, bound_case=case)
        mpfa_ad = pp.ad.MpfaAd(param_key, grid_list)
        bound_ad = pp.ad.BoundaryCondition(param_key, grids=grid_list)
        dir_bound_ad = DirBC(bound_ad, grid_list)
        flux_1p_ad = (
            mpfa_ad.flux * p.previous_iteration() + mpfa_ad.bound_flux * bound_ad
        )
        upwind = FluxBasedUpwindAD(g, d, param_key)
        upw_faces_ad = upwind(p.previous_iteration(), dir_bound_ad, flux_1p_ad)
        upw_faces_eval = pp.ad.Expression(upw_faces_ad, dof_manager)
        upw_faces_eval.discretize(gb=gb)
        upw_faces_num = upw_faces_eval.to_ad(gb)
        upw_faces = upw_faces_num.diagonal()
        np.testing.assert_array_equal(upw_faces, true_value)
        
