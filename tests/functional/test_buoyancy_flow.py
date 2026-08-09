"""N-phase, N-component buoyancy-driven flow under gravity: conservation and
hybrid-upwinding monotonicity.

The domain is closed (no-flow Neumann on all boundaries), so pressure is determined
only up to an additive constant, fixed by a null-mean-pressure gauge, Sum(p_matrix) = 0,
imposed by NullMeanPressureSolve / NullMeanPressureLinearSolver. Fluids: N = 2 (liquid,
gas; H2O, CH4) and N = 3 (liquid, oil, gas; H2O, CO2, CH4), in 2D and 3D.

Conservation -- test_buoyancy_model, on each accepted time step:
  (i)   reciprocity: paired component buoyancy fluxes are equal and opposite (sum to 0);
  (ii)  |Δ total independent-phase volume| ≤ tol      (mass, to the expected order);
  (iii) |Δ total fluid energy| ≤ tol                  (energy, to the expected order);
  (iv)  the null-mean-pressure residual stays ≤ null_mean_res_tolerance
        (NullSpaceCriterion).

Monotonicity -- test_buoyancy_flux_monotonicity and
test_buoyancy_interface_flux_monotonicity: the component buoyancy flux is monotone in
the swept cell's overall composition -- the hybrid-upwinding property of Hamon &
Tchelepi (SIAM J. Numer. Anal. 54(3), 2016), on a subdomain internal face and across
each matrix-fracture mortar side.
"""

from typing import Literal

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.models import add_mixin
from tests.functional.setups.buoyancy_flow_model import (
    ModelGeometry2D,
    ModelGeometry3D,
    ModelMDGeometry2D,
    ModelMDGeometry3D,
    NullMeanPressureLinearSolver,
    NullSpaceCriterion,
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

    # Newton must converge one decade below the conservation target so the
    # residual does not pollute the conservation-order checks.
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
    # Per-step null-mean-pressure residual budget: the conservation threshold
    # 10^-(order-1) split over n_steps, with a factor-2 margin.
    n_steps = round(tf / dt)
    null_mean_res_tolerance = 10.0 ** (-(expected_order_loss - 1)) / (2 * n_steps)

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
        # True: total mobility in the Darcy tensor (fractional flow); False:
        # explicit total-mobility factor (standard formulation).
        "fractional_flow": fractional_flow,
        "enable_buoyancy_effects": True,
        "material_constants": {"solid": solid_constants},
        "time_manager": time_manager,
        "expected_order_loss": expected_order_loss,
        # Tolerances exposed to the model for the converged-state checks.
        "residual_tolerance": residual_tolerance,
        "null_mean_res_tolerance": null_mean_res_tolerance,
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
            # The residual is a rate; the null-mean-pressure criterion bounds what
            # the conservation checks accumulate per step.
            "null_mean_res": NullSpaceCriterion(model, tol=null_mean_res_tolerance),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=50),
        },
    }

    # The closed domain leaves a singular pressure mode; the gauge-fixing solver
    # handles it.
    nonlinear_solver = pp.solvers.NewtonSolver(
        params=solver_params,
        linear_solver=NullMeanPressureLinearSolver(),
    )

    runner = pp.ModelRunner(model, solver_params, nonlinear_solver=nonlinear_solver)
    runner.run()


@pytest.mark.skipped  # reason: slow
@pytest.mark.parametrize("fractional_flow", [True, False])
@pytest.mark.parametrize("n_phases, dim, expected_order_loss", Parameterization)
@pytest.mark.parametrize("md", [True])  # False skipped to limit computational cost.
def test_buoyancy_model(
    n_phases, dim: Literal[2, 3], expected_order_loss, md, fractional_flow
):
    """Test buoyancy-driven flow model (FD)."""
    _run_buoyancy_model(
        n_phases, dim, expected_order_loss, md=md, fractional_flow=fractional_flow
    )


class _TwoCellColumn(ModelGeometry2D):
    """1 x 2 vertical Cartesian stack: two cells, one internal face across gravity."""

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmax": self.units.convert_units(1.0, "m"),
                "ymax": self.units.convert_units(2.0, "m"),
            }
        )

    def meshing_arguments(self) -> dict:
        return {
            "cell_size_x": self.units.convert_units(1.0, "m"),
            "cell_size_y": self.units.convert_units(1.0, "m"),
        }


class _CompositionSweepIC(pp.PorePyModel):
    """Impose the swept component's overall fraction in one column cell (rest fixed).

    ``initial_condition`` propagates the imposed composition to consistent phase
    saturations through the model's saturation map, so each sweep point is
    flash-consistent.
    """

    _swept_component: str = "CH4"
    _swept_cell: int = 0
    _swept_value: float = 0.3
    _reference_fraction: float = 0.2

    def ic_values_overall_fraction(self, component, sd):
        if component.name == "H2O":
            return (
                1.0
                - self.ic_values_overall_fraction(self.fluid.components[1], sd)
                - self.ic_values_overall_fraction(self.fluid.components[2], sd)
            )
        vals = np.full(sd.num_cells, self._reference_fraction)
        if component.name == self._swept_component:
            vals[self._swept_cell] = self._swept_value
        return vals


@pytest.fixture(scope="module", params=[False, True], ids=["hu", "hu_mwp"])
def subdomain_sweep_model(request):
    """Two-cell buoyancy model built once per formulation, shared across ``swept_cell``
    and the composition sweep.

    The mesh and equations depend only on ``fractional_flow``; the test re-sets the IC
    (and the flash-derived quantities, required for the state-dependent HU-mwp flux) per
    sweep point, which is bit-identical to rebuilding the model and collapses N builds
    into one.
    """
    fractional_flow = request.param
    base = add_mixin(
        _TwoCellColumn, buoyancy_flow_model(3, fractional_flow=fractional_flow)
    )

    class Model(_CompositionSweepIC, base):
        pass

    solid_constants = pp.SolidConstants(
        permeability=1.0e-14,
        porosity=0.1,
        thermal_conductivity=2.0 * to_Mega,
        density=2500.0,
        specific_heat_capacity=1000.0 * to_Mega,
    )
    model = Model(
        {
            "fractional_flow": fractional_flow,
            "enable_buoyancy_effects": True,
            "material_constants": {"solid": solid_constants},
            "time_manager": pp.TimeManager(
                schedule=[0.0, 86400.0],
                dt_init=86400.0,
                constant_dt=True,
                iter_max=50,
                print_info=False,
            ),
            "expected_order_loss": 3,
            "residual_tolerance": 1e-4,
            "null_mean_res_tolerance": 1e-4,
        }
    )
    model.prepare_simulation()
    return model


@pytest.mark.parametrize("swept_cell", [0, 1])
# CI: CH4 only; C5H12 -> ``skipped`` (--run-skipped, nightly).
# |components| 2->1 halves the per-PR suite.
@pytest.mark.parametrize(
    "swept_component", ["CH4", pytest.param("C5H12", marks=pytest.mark.skipped)]
)
def test_buoyancy_flux_monotonicity(
    subdomain_sweep_model, swept_component: str, swept_cell: int
):
    """Buoyant component flux is monotone in the swept cell's overall composition.

    Flux monotonicity w.r.t. its own saturation is the hybrid-upwinding property of Hamon
    and Tchelepi (SIAM J. Numer. Anal. 54(3), 2016); both matrix cells are swept and both
    flux formulations (total-mass, CFF) are covered.
    """
    model = subdomain_sweep_model
    model._swept_component = swept_component
    model._swept_cell = swept_cell
    subdomains = model.mdg.subdomains()
    internal_face = subdomains[0].get_internal_faces()[0]
    component = next(c for c in model.fluid.components if c.name == swept_component)

    z_sweep = np.linspace(0.05, 1.0, 11)
    fluxes = []
    for z in z_sweep:
        model._swept_value = float(z)
        model.initial_condition()  # flash-consistent IC state for this z
        model.update_all_boundary_conditions()
        model.update_derived_quantities()  # flash refresh for the HU-mwp flux
        model.before_nonlinear_iteration()  # activate the hybrid-upwind directions
        flux = model.component_buoyancy(component, subdomains).value(
            model.equation_system
        )
        fluxes.append(flux[internal_face])

    fluxes = np.array(fluxes)
    scale = np.max(np.abs(fluxes))
    # The sweep must genuinely drive the flux, else monotonicity is vacuous.
    assert scale > 1e-8, "buoyant flux is ~zero over the sweep; test is vacuous"
    diffs = np.diff(fluxes)
    atol = 1e-9 * scale
    monotone = np.all(diffs >= -atol) or np.all(diffs <= atol)
    assert monotone, (
        f"component buoyancy flux for {swept_component} is not monotone in its overall "
        f"composition (a phase-potential-upwinding kink): diffs={diffs}"
    )


class _MDColumnFracture(ModelGeometry2D):
    """[0,1] x [0,2] matrix column (1 x 2) split by a single horizontal fracture at y=1.

    Yields two matrix cells (cell 0 = bottom at y=0.5, cell 1 = top at y=1.5), one
    fracture cell, and one interface whose 2-cell mortar has a "below" side (bottom cell
    <-> fracture) and an "above" side (top cell <-> fracture).
    """

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmax": self.units.convert_units(1.0, "m"),
                "ymax": self.units.convert_units(2.0, "m"),
            }
        )

    def set_fractures(self) -> None:
        points = np.array([[0.0, 1.0], [1.0, 1.0]]).T
        self._fractures = pp.frac_utils.pts_edges_to_linefractures(
            points, np.array([[0, 1]]).T
        )

    def meshing_arguments(self) -> dict:
        return {"cell_size": self.units.convert_units(1.0, "m")}


class _MDCompositionSweepIC(pp.PorePyModel):
    """Impose the swept component's overall fraction in one target MD-column cell.

    Targets: "bottom"/"top" (matrix cells) or "fracture" (fracture cell). The model's
    initial_condition propagates the composition to consistent saturations on every
    subdomain via its own map, so each sweep point is a flash-consistent state.
    """

    _swept_component: str = "CH4"
    _swept_target: str = "bottom"
    _swept_value: float = 0.3
    _reference_fraction: float = 0.2

    def ic_values_overall_fraction(self, component, sd):
        if component.name == "H2O":
            return (
                1.0
                - self.ic_values_overall_fraction(self.fluid.components[1], sd)
                - self.ic_values_overall_fraction(self.fluid.components[2], sd)
            )
        vals = np.full(sd.num_cells, self._reference_fraction)
        if component.name == self._swept_component:
            if sd.dim == 2:  # matrix: cell 0 = bottom, cell 1 = top
                matrix_index = {"bottom": 0, "top": 1}.get(self._swept_target)
                if matrix_index is not None:
                    vals[matrix_index] = self._swept_value
            elif sd.dim == 1 and self._swept_target == "fracture":
                vals[0] = self._swept_value
        return vals


def _component_interface_flux(model, component):
    """The per-mortar-cell component buoyancy flux (no public accessor exists): the same pair
    sum as ``component_buoyancy_jump`` but taken on the interface, before projection onto the
    fracture subdomain."""
    subdomains = model.mdg.subdomains()
    terms = []
    for phase in model.fluid.phases:
        for gamma, delta in model.phase_pairs_for(phase):
            chi = model._advected_partial_fraction(component, gamma, subdomains)
            terms.append(model._interface_pair_coupling(chi, gamma, delta, subdomains))
    return pp.ad.sum_operator_list(terms)


def _below_above_mortar_indices(model):
    """Mortar-cell indices of the (below, above) sides: below couples the lower-y matrix cell."""
    interface = model.mdg.interfaces()[0]
    matrix, _ = model.mdg.interface_to_subdomain_pair(interface)
    y_of_cell = matrix.cell_centers[1]
    primary_to_mortar = interface.primary_to_mortar_int().tocsr()
    mortar_y = []
    for mortar_cell in range(interface.num_cells):
        faces = primary_to_mortar.getrow(mortar_cell).indices
        cells = np.unique(np.abs(matrix.cell_faces)[faces].nonzero()[1])
        mortar_y.append(y_of_cell[cells].mean())
    mortar_y = np.array(mortar_y)
    return int(np.argmin(mortar_y)), int(np.argmax(mortar_y))


# Each interface side is checked against both cells it couples: the "below" side against the
# bottom matrix cell and the fracture, the "above" side against the top matrix cell and the
# fracture.
_MD_INTERFACE_CHECKS = [
    ("below", "bottom"),
    ("below", "fracture"),
    ("above", "fracture"),
    ("above", "top"),
]


@pytest.fixture(scope="module", params=[False, True], ids=["hu", "hu_mwp"])
def interface_sweep_model(request):
    """MD column-with-fracture buoyancy model built once per formulation, shared across the
    interface-side checks and the composition sweep. Same reuse rationale as
    :func:`subdomain_sweep_model`.
    """
    fractional_flow = request.param
    base = add_mixin(
        _MDColumnFracture, buoyancy_flow_model(3, fractional_flow=fractional_flow)
    )

    class Model(_MDCompositionSweepIC, base):
        pass

    solid_constants = pp.SolidConstants(
        permeability=1.0e-14,
        porosity=0.1,
        thermal_conductivity=2.0 * to_Mega,
        density=2500.0,
        specific_heat_capacity=1000.0 * to_Mega,
    )
    model = Model(
        {
            "fractional_flow": fractional_flow,
            "enable_buoyancy_effects": True,
            "material_constants": {"solid": solid_constants},
            "time_manager": pp.TimeManager(
                schedule=[0.0, 86400.0],
                dt_init=86400.0,
                constant_dt=True,
                iter_max=50,
                print_info=False,
            ),
            "expected_order_loss": 3,
            "residual_tolerance": 1e-4,
            "null_mean_res_tolerance": 1e-4,
        }
    )
    model.prepare_simulation()
    return model


@pytest.mark.parametrize("side, swept_target", _MD_INTERFACE_CHECKS)
# CI: CH4 only; C5H12 -> ``skipped`` (--run-skipped, nightly).
# |components| 2->1 halves the per-PR suite.
@pytest.mark.parametrize(
    "swept_component", ["CH4", pytest.param("C5H12", marks=pytest.mark.skipped)]
)
def test_buoyancy_interface_flux_monotonicity(
    interface_sweep_model, swept_component: str, side: str, swept_target: str
):
    """Buoyant component flux across each matrix-fracture mortar side is monotone in the
    composition of each cell it couples.

    Mixed-dimensional analogue of the hybrid-upwinding flux monotonicity of Hamon and Tchelepi
    (SIAM J. Numer. Anal. 54(3), 2016); the "below" side couples the bottom matrix cell and the
    fracture, the "above" side the top cell and the fracture. Both flux formulations are covered.
    """
    model = interface_sweep_model
    model._swept_component = swept_component
    model._swept_target = swept_target
    below, above = _below_above_mortar_indices(model)
    component = next(c for c in model.fluid.components if c.name == swept_component)

    z_sweep = np.linspace(0.05, 1.0, 11)
    fluxes = []
    for z in z_sweep:
        model._swept_value = float(z)
        model.initial_condition()  # flash-consistent IC state for this z
        model.update_all_boundary_conditions()
        model.update_derived_quantities()  # flash refresh for the HU-mwp flux
        model.before_nonlinear_iteration()  # activate the hybrid-upwind directions
        mortar_flux = model.equation_system.evaluate(
            _component_interface_flux(model, component)
        )
        fluxes.append(mortar_flux[below if side == "below" else above])

    fluxes = np.array(fluxes)
    scale = np.max(np.abs(fluxes))
    assert scale > 1e-8, (
        "interface buoyant flux is ~zero over the sweep; test is vacuous"
    )
    diffs = np.diff(fluxes)
    atol = 1e-9 * scale
    monotone = np.all(diffs >= -atol) or np.all(diffs <= atol)
    assert monotone, (
        f"{side}-side interface buoyancy flux for {swept_component} is not monotone in the "
        f"{swept_target}-cell composition (a phase-potential-upwinding kink): diffs={diffs}"
    )
