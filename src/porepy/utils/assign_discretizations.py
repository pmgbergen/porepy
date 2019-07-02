"""
Convenience methods for assigning discretization methods for some common coupled
problems.

"""
import porepy as pp
from porepy.utils.derived_discretizations import implicit_euler as IE_discretizations


def contact_mechanics_discretizations(setup):
    """
    Setup should have a gb field, and the following names specified:
       Parameter keys:
           mechanics_parameter_key
           friction_parameter_key
       Variables:
           displacement_variable - higher-dimensional displacements
           mortar_displacement_variable - displacement on the internal boundary
           contact_variable - represents traction on the fracture
    """
    gb = setup.gb
    ambient_dim = gb.dim_max()
    # Define discretization
    # For the Nd domain we solve linear elasticity with mpsa.
    mpsa = pp.Mpsa(setup.mechanics_parameter_key)
    empty_discr = pp.VoidDiscretization(
        setup.friction_parameter_key, ndof_cell=ambient_dim
    )

    # Define discretization parameters
    for g, d in gb:
        if g.dim == ambient_dim:
            d[pp.PRIMARY_VARIABLES] = {
                setup.displacement_variable: {"cells": ambient_dim}
            }
            d[pp.DISCRETIZATION] = {setup.displacement_variable: {"mpsa": mpsa}}
        elif g.dim == ambient_dim - 1:
            d[pp.PRIMARY_VARIABLES] = {setup.contact_variable: {"cells": ambient_dim}}
            d[pp.DISCRETIZATION] = {setup.contact_variable: {"empty": empty_discr}}
        else:
            d[pp.PRIMARY_VARIABLES] = {}

    # And define a Robin condition on the mortar grid
    coloumb = pp.ColoumbContact(setup.friction_parameter_key, ambient_dim)
    contact_discr = pp.PrimalContactCoupling(
        setup.friction_parameter_key, mpsa, coloumb
    )

    for e, d in gb.edges():
        g_l, g_h = gb.nodes_of_edge(e)

        if g_h.dim == ambient_dim:
            d[pp.PRIMARY_VARIABLES] = {
                setup.mortar_displacement_variable: {"cells": ambient_dim}
            }

            d[pp.COUPLING_DISCRETIZATION] = {
                setup.friction_coupling_term: {
                    g_h: (setup.displacement_variable, "mpsa"),
                    g_l: (setup.contact_variable, "empty"),
                    (g_h, g_l): (setup.mortar_displacement_variable, contact_discr),
                }
            }
        else:
            d[pp.PRIMARY_VARIABLES] = {}


def contact_mechanics_and_biot_discretizations(setup, subtract_fracture_pressure=True):
    """
    Assign the discretizations for fracture deformation with a coupled scalar (pressure)
    in both dimensions. No fracture intersections are allowed (for now).

    Setup should have a gb field, and the following names specified:
       Parameter keys:
           mechanics_parameter_key
           scalar_parameter_key
       Variables:
           displacement_variable - higher-dimensional displacements
           mortar_displacement_variable - displacement on the internal boundary
           contact_traction_variable - represents traction on the fracture
           scalar_variable - scalar (pressure) in both dimensions
           mortar_scalar_variable - darcy flux
    subtract_fracture_pressure (bool): Whether or not to subtract the fracture pressure
        contribution to the fracture force balance equation. This is needed for the
        pressure case, where the forces on the fracture surfaces are the sum of the
        contact force and the pressure force. It is not, however, needed for TM
        simulations, where there is no force from the fracture temperature.
    """
    gb = setup.gb
    ambient_dim = gb.dim_max()
    key_s, key_m = setup.scalar_parameter_key, setup.mechanics_parameter_key
    var_s, var_d = setup.scalar_variable, setup.displacement_variable
    # Define discretization
    # For the Nd domain we solve linear elasticity with mpsa.
    mpsa = pp.Mpsa(key_m)
    empty_discr = pp.VoidDiscretization(key_m, ndof_cell=ambient_dim)
    # Scalar discretizations (all dimensions)
    diff_disc_s = IE_discretizations.ImplicitMpfa(key_s)
    mass_disc_s = IE_discretizations.ImplicitMassMatrix(key_s, var_s)
    source_disc_s = pp.ScalarSource(key_s)
    # Coupling discretizations
    # All dimensions
    div_u_disc = pp.DivU(
        key_m, key_s, variable=var_d, mortar_variable=setup.mortar_displacement_variable
    )
    # Nd
    grad_p_disc = pp.GradP(key_m)
    stabilization_disc_s = pp.BiotStabilization(key_s, var_s)

    # Assign node discretizations
    for g, d in gb:
        if g.dim == ambient_dim:
            d[pp.PRIMARY_VARIABLES] = {
                var_d: {"cells": ambient_dim},
                var_s: {"cells": 1},
            }
            d[pp.DISCRETIZATION] = {
                var_d: {"mpsa": mpsa},
                var_s: {
                    "diffusion": diff_disc_s,
                    "mass": mass_disc_s,
                    "stabilization": stabilization_disc_s,
                    "source": source_disc_s,
                },
                var_d + "_" + var_s: {"grad_p": grad_p_disc},
                var_s + "_" + var_d: {"div_u": div_u_disc},
            }
        elif g.dim == ambient_dim - 1:
            d[pp.PRIMARY_VARIABLES] = {
                setup.contact_traction_variable: {"cells": ambient_dim},
                var_s: {"cells": 1},
            }
            d[pp.DISCRETIZATION] = {
                setup.contact_traction_variable: {"empty": empty_discr},
                var_s: {
                    "diffusion": diff_disc_s,
                    "mass": mass_disc_s,
                    "source": source_disc_s,
                },
            }
        else:
            d[pp.PRIMARY_VARIABLES] = {}

    # Define edge discretizations for the mortar grid
    contact_law = pp.ColoumbContact(setup.mechanics_parameter_key, ambient_dim)
    contact_discr = pp.PrimalContactCoupling(
        setup.mechanics_parameter_key, mpsa, contact_law
    )
    # Account for the mortar displacements effect on scalar balance in the
    #   matrix, as an internal boundary contribution,
    #   fracture, aperture changes appear as a source contribution.
    div_u_coupling = pp.DivUCoupling(
        setup.displacement_variable, div_u_disc, div_u_disc
    )
    # Account for the pressure contributions to the force balance on the fracture
    # (see contact_discr).
    # This discretization needs the keyword used to store the grad p discretization:
    grad_p_key = key_m
    matrix_scalar_to_force_balance = pp.MatrixScalarToForceBalance(
        grad_p_key, mass_disc_s, mass_disc_s
    )
    if subtract_fracture_pressure:
        fracture_scalar_to_force_balance = pp.FractureScalarToForceBalance(
            mass_disc_s, mass_disc_s
        )
    for e, d in gb.edges():
        g_l, g_h = gb.nodes_of_edge(e)

        if g_h.dim == ambient_dim:
            d[pp.PRIMARY_VARIABLES] = {
                setup.mortar_displacement_variable: {"cells": ambient_dim},
                setup.mortar_scalar_variable: {"cells": 1},
            }

            d[pp.COUPLING_DISCRETIZATION] = {
                setup.friction_coupling_term: {
                    g_h: (var_d, "mpsa"),
                    g_l: (setup.contact_traction_variable, "empty"),
                    (g_h, g_l): (setup.mortar_displacement_variable, contact_discr),
                },
                setup.scalar_coupling_term: {
                    g_h: (var_s, "diffusion"),
                    g_l: (var_s, "diffusion"),
                    e: (
                        setup.mortar_scalar_variable,
                        pp.RobinCoupling(key_s, diff_disc_s),
                    ),
                },
                "div_u_coupling": {
                    g_h: (
                        var_s,
                        "mass",
                    ),  # This is really the div_u, but this is not implemented
                    g_l: (var_s, "mass"),
                    e: (setup.mortar_displacement_variable, div_u_coupling),
                },
                "matrix_scalar_to_force_balance": {
                    g_h: (var_s, "mass"),
                    g_l: (var_s, "mass"),
                    e: (
                        setup.mortar_displacement_variable,
                        matrix_scalar_to_force_balance,
                    ),
                },
            }
            if subtract_fracture_pressure:
                d[pp.COUPLING_DISCRETIZATION].update(
                    {
                        "matrix_scalar_to_force_balance": {
                            g_h: (var_s, "mass"),
                            g_l: (var_s, "mass"),
                            e: (
                                setup.mortar_displacement_variable,
                                fracture_scalar_to_force_balance,
                            ),
                        }
                    }
                )
        else:
            raise ValueError(
                "assign_discretizations assumes no fracture intersections."
            )
