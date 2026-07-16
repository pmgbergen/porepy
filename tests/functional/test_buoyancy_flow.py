"""
Tests for the N-phase, N-component buoyancy-driven flow model.

This file verifies mass and energy conservation and the reciprocity of buoyancy
fluxes in an immiscible flow simulation under gravity.

It covers two multicomponent fluid systems:
- N = 2: Two phases (aqueous liquid, gas) and two components (e.g., H₂O, CH₄).
- N = 3: Three phases (aqueous liquid, oleic liquid, gas) and
three components (e.g., H₂O, CO₂, CH₄).

Simulations are run in 2D and 3D for several conservation tolerances, and
the observed conservation is checked to be of the expected order. After each
time step the following are tested:
1. Reciprocal buoyancy fluxes: Component buoyancy fluxes are equal and opposite.
2. Mass conservation: The change in the total volume of independent phases over
   the simulation time remains within the specified tolerance, demonstrating a
   mass-conservative discretization of the buoyancy term.
3. Energy conservation: The change in total fluid energy over the simulation
   time remains within the specified tolerance, demonstrating an energy-conservative
   discretization of the energy convective buoyancy terms.
"""

from typing import Literal

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.models import add_mixin


def _global_unique_ad_nodes(equations) -> int:
    """Count unique operator nodes across *all* equations with one shared visited set.

    Unlike summing :meth:`Operator.inspect` per equation, this deduplicates subtrees that
    are shared *between* equations (e.g. a density or mobility reused by the mass, energy
    and buoyancy equations because the constitutive laws use ``@cached_method``). It is
    the truest measure of the assembled graph's size.
    """
    visited: set[int] = set()
    stack = list(equations.values())
    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.extend(node.children)
    return len(visited)


def _report_ad_graph_size(model: pp.PorePyModel, label: str) -> dict[str, int]:
    """Report the size of the assembled AD operator graph.

    Walks every equation registered on the model's equation system with
    :meth:`porepy.numerics.ad.operators.Operator.inspect` (which is DAG-aware: a
    subexpression shared across parents is counted once). Reports, per equation, the
    sum over equations, and the *global* unique-node count (subtrees shared between
    equations counted once), so the effect of graph-size reductions (e.g. sharing
    subtrees via ``@cached_method``) is visible while running the test.

    Parameters:
        model: A prepared PorePy model (``prepare_simulation`` already called).
        label: A short identifier for the model configuration, used in the printout.

    Returns:
        A dict with the per-equation node sum (``"total_nodes"``), the global unique-node
        count (``"unique_nodes"``) and equation count (``"num_equations"``).

    """
    equations = model.equation_system.equations
    total_nodes = 0
    print(f"\n=== AD graph size [{label}] ===")
    for name, eq in equations.items():
        stats = eq.inspect(verbose=False)
        total_nodes += stats["total_nodes"]
        print(
            f"  {name}: {stats['total_nodes']} nodes, "
            f"depth {stats['max_depth']}, {len(stats['variables'])} variables"
        )
    unique_nodes = _global_unique_ad_nodes(equations)
    print(
        f"  --> {len(equations)} equations, {total_nodes} nodes summed per equation, "
        f"{unique_nodes} globally-unique AD nodes"
    )
    return {
        "total_nodes": total_nodes,
        "unique_nodes": unique_nodes,
        "num_equations": len(equations),
    }
from tests.functional.setups.buoyancy_flow_model import (
    ModelGeometry2D,
    ModelGeometry3D,
    ModelMDGeometry2D,
    ModelMDGeometry3D,
    NullMeanPressureLinearSolver,
    NullSpaceDriftCriterion,
    buoyancy_flow_model,
    to_Mega,
)

# Parameterization list for both tests: (number of phases/components, dimension, order).
Parameterization = [
    (2, 2, 4),
    (2, 3, 4),
    (3, 2, 4),
    (3, 3, 4),
]


def _run_buoyancy_model(
    n_phases: int,
    dim: Literal[2, 3],
    expected_order_loss: int,
    md: bool = False,
    fractional_flow: bool = True,
) -> None:
    """Run buoyancy flow simulation for given parameters."""

    # The residual tolerance for Newton must be *tighter* than the conservation target:
    # the conservation loss checked below is bounded by the Newton residual, and it
    # accumulates over the time steps (and grows with the vigour of the buoyant
    # overturning). Converging one decade below ``expected_order_loss`` keeps the residual
    # from polluting the conservation-order checks.
    residual_tolerance = 10.0 ** (-(expected_order_loss + 1))
    day = 86400
    if md:
        tf = 0.5 * day
        dt = 0.25 * day
        geometry2d = ModelMDGeometry2D
        geometry3d = ModelMDGeometry3D
    else:
        tf = 2.0 * day
        dt = 1.0 * day
        geometry2d = ModelGeometry2D
        geometry3d = ModelGeometry3D

    solid_constants = pp.SolidConstants(
        permeability=1.0e-14,
        porosity=0.1,
        thermal_conductivity=2.0 * to_Mega,
        density=2500.0,
        specific_heat_capacity=1000.0 * to_Mega,
    )
    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
        iter_max=50,
        print_info=True,
    )
    model_params = {
        # fractional_flow=True -> total mass mobility is in the Darcy permeability tensor
        # (CompositionalFractionalFlowTemplate); False -> standard formulation with an
        # explicit total-mobility factor in the buoyancy term (CompositionalFlowTemplate).
        "fractional_flow": fractional_flow,
        "enable_buoyancy_effects": True,
        "buoyancy_upwinding": "hybrid",
        "material_constants": {"solid": solid_constants},
        "time_manager": time_manager,
        "expected_order_loss": expected_order_loss,
        # The Newton tolerance, exposed to the model so the converged-state checks can
        # verify the residual's null-space component (total-mass drift) actually met it.
        "residual_tolerance": residual_tolerance,
    }
    # Build the model with the fractional_flow-selected template, then mix in geometry.
    geometry_class = geometry2d if dim == 2 else geometry3d
    model_class = buoyancy_flow_model(n_phases, fractional_flow)
    model_class = add_mixin(geometry_class, model_class)
    model = model_class(model_params)
    # Use a Lebesgue metric for the residual convergence criterion, since this will
    # strictly bound the residual error in the mass conservation equations.
    solver_params = {
        "nl_convergence_criteria": {
            "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                tol=residual_tolerance, metric=pp.EquationBasedLebesgueMetric(model)
            ),
            # The metric above bounds the residual -- a mass RATE. The conservation
            # checks accumulate MASS: dt scales a metric-converged rate residual by ~1e5,
            # and the total-mass drift is a null-space component the linear solve cannot
            # correct (it decays only quadratically). Converge it explicitly, in the
            # dt-scaled volume-normalized units the conservation checks measure.
            "null_drift": NullSpaceDriftCriterion(model, tol=residual_tolerance),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=50),
        },
    }

    # The closed all-Neumann domain leaves a singular constant-pressure mode: the Newton
    # solver must use the null-mean-bordered direct solve (the gauge constraint), NOT the
    # default Pardiso solve. On this branch the linear solver is an object passed to the
    # NewtonSolver, so the gauge is wired here instead of overriding
    # ``model.solve_linear_system``.
    nonlinear_solver = pp.solvers.NewtonSolver(
        params=solver_params,
        linear_solver=NullMeanPressureLinearSolver(),
    )

    # Constructing the runner prepares the simulation (sets equations), so the AD graph
    # is available for inspection before the (slow) run.
    runner = pp.ModelRunner(model, solver_params, nonlinear_solver=nonlinear_solver)
    _report_ad_graph_size( model, f"{model_class.__name__} dim={dim} md={md}")
    runner.run()

@pytest.mark.parametrize("fractional_flow", [True])
@pytest.mark.parametrize("n_phases, dim, expected_order_loss", Parameterization)
@pytest.mark.parametrize("md", [False,True])  # False skipped to limit computational cost.
def test_buoyancy_model(
    n_phases, dim: Literal[2, 3], expected_order_loss, md, fractional_flow
):
    """Test buoyancy-driven flow model (FD)."""
    _run_buoyancy_model(
        n_phases, dim, expected_order_loss, md=md, fractional_flow=fractional_flow
    )
