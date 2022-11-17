"""Tests for model variables.

"""
from inspect import signature

import pytest
"""Utility methods for setting up models for testing."""


import numpy as np

import porepy as pp
from porepy.models import fluid_mass_balance as fmb
from porepy.models import force_balance as fb


class FracGeom(pp.ModelGeometry):
    def set_fracture_network(self) -> None:
        p = np.array([[0, 1], [0.5, 0.5]])
        e = np.array([[0], [1]])
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        self.fracture_network = pp.FractureNetwork2d(p, e, domain)

    def mesh_arguments(self) -> dict:
        return {"mesh_size_frac": 0.5, "mesh_size_bound": 0.5}


class IncompressibleCombined(
    FracGeom,
    fmb.MassBalanceEquations,
    fmb.ConstitutiveEquationsIncompressibleFlow,
    fmb.VariablesSinglePhaseFlow,
    fmb.SolutionStrategyIncompressibleFlow,
    pp.DataSavingMixin,
):
    """Demonstration of how to combine in a class which can be used with
    pp.run_stationary_problem (once cleanup has been done).
    """

    pass


class MassBalanceCombined(
    FracGeom,
    fmb.MassBalanceEquations,
    fmb.ConstitutiveEquationsCompressibleFlow,
    fmb.VariablesSinglePhaseFlow,
    fmb.SolutionStrategyCompressibleFlow,
    pp.DataSavingMixin,
):
    ...


class ForceBalanceCombined(
    FracGeom,
    fb.ForceBalanceEquations,
    fb.ConstitutiveEquationsForceBalance,
    fb.VariablesForceBalance,
    fb.SolutionStrategyForceBalance,
    pp.DataSavingMixin,
):
    """Combine components needed for force balance simulation."""

    pass


def model(type: str) -> ForceBalanceCombined:
    """Setup for tests."""
    # Suppress output for tests
    params = {"suppress_export": True}

    # Choose model and create setup object
    if type == "mass_balance":
        ob = MassBalanceCombined(params)
    elif type == "force_balance":
        ob = ForceBalanceCombined(params)
    else:
        raise ValueError("Unknown type")

    # Prepare the simulation
    # (create gridsm, variables, equations, discretize, etc.)
    ob.prepare_simulation()
    return ob




@pytest.mark.parametrize(
    "model_type,equation_name,domain_inds",
    [
        ("mass_balance", "mass_balance_equation", []),
        ("mass_balance", "interface_darcy_flux_equation", []),
        ("force_balance", "matrix_force_balance_equation", [0]),
        ("force_balance", "interface_force_balance_equation", [0]),
        ("force_balance", "normal_fracture_deformation_equation", [1]),
        ("force_balance", "tangential_fracture_deformation_equation", [1]),
    ],
)
def test_parse_equations(model_type, equation_name, domain_inds):
    """Test that equation parsing works as expected."""
    setup = model(model_type)
    # TODO: Add EquationManager method for retrieving equation domains
    method = getattr(setup, equation_name)
    sig = signature(method)
    assert len(sig.parameters) == 1
    if "subdomains" in sig.parameters:
        domains = setup.mdg.subdomains()
    elif "interfaces" in sig.parameters:
        domains = setup.mdg.interfaces()

    # Pick out the relevant domains
    if len(domain_inds) > 0:
        domains = [domains[i] for i in domain_inds if i <= len(domains)]
    setup.equation_system.assemble_subsystem({equation_name: domains})

test_parse_equations("force_balance", "normal_fracture_deformation_equation", [1])