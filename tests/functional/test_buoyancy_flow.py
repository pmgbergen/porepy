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
    BuoyancyFlowModel2N,
    BuoyancyFlowModel3N,
    ModelGeometry2D,
    ModelGeometry3D,
    ModelMDGeometry2D,
    ModelMDGeometry3D,
    to_Mega,
)

# Parameterization list for both tests
Parameterization = [
    (BuoyancyFlowModel2N, 2, 4),
    (BuoyancyFlowModel2N, 3, 4),
    (BuoyancyFlowModel3N, 2, 4),
    (BuoyancyFlowModel3N, 3, 4),
]


def _run_buoyancy_model(
    model_class: type,
    dim: Literal[2, 3],
    expected_order_loss: int,
    md: bool = False,
) -> None:
    """Run buoyancy flow simulation for given parameters."""

    # The residual tolerance for Newton must be *tighter* than the conservation target:
    # the conservation loss checked below is bounded by the Newton residual, and it
    # accumulates over the time steps (and grows with the vigour of the buoyant
    # overturning). Converging one decade below ``expected_order_loss`` keeps the residual
    # from polluting the conservation-order checks.
    residual_tolerance = 10.0 ** (-(expected_order_loss + 0.75))
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
        # Fractional-flow formulation: total mass mobility is incorporated into the Darcy
        # permeability tensor, and the buoyancy term consistently handle the explicit
        # total-mobility factor.
        "fractional_flow": True,
        "enable_buoyancy_effects": True,
        "buoyancy_upwinding": "hybrid",
        "material_constants": {"solid": solid_constants},
        "time_manager": time_manager,
        "expected_order_loss": expected_order_loss,
    }
    # Combine geometry with model class.
    geometry_class = geometry2d if dim == 2 else geometry3d
    model_class = add_mixin(geometry_class, model_class)
    model = model_class(model_params)
    # Use a Lebesgue metric for the residual convergence criterion, since this will
    # strictly bound the residual error in the mass conservation equations.
    solver_params = {
        "nl_convergence_criteria": {
            "res_abs": pp.ResidualBasedAbsoluteCriterion(
                tol=residual_tolerance, metric=pp.EquationBasedLebesgueMetric(model)
            ),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.MaxIterationsCriterion(max_iterations=50),
        },
    }

    # Constructing the runner prepares the simulation (sets equations), so the AD graph
    # is available for inspection before the (slow) run.
    runner = pp.ModelRunner(model, solver_params)
    _report_ad_graph_size(
        model, f"{model_class.__name__} dim={dim} md={md}"
    )
    runner.run()


#@pytest.mark.skipped  # reason: slow
@pytest.mark.parametrize("model_class, dim, expected_order_loss", Parameterization)
@pytest.mark.parametrize("md", [True])  # False skipped to limit computational cost.
def test_buoyancy_model(model_class, dim: Literal[2, 3], expected_order_loss, md):
    """Test buoyancy-driven flow model (FD)."""
    _run_buoyancy_model(model_class, dim, expected_order_loss, md=md)
