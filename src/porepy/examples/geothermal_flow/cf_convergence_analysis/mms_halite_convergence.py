"""Driver for the convergence study with the manufactured-solution test."""

from copy import deepcopy
import numpy as np
from pathlib import Path

import porepy as pp
from porepy.numerics.nonlinear.line_search import LineSearchNewtonSolver
from porepy.applications.convergence_analysis import ConvergenceAnalysis

from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_halite_constants import MMS_HALITE
from porepy.examples.geothermal_flow.cf_convergence_analysis.flow_halite_mms import HaliteMMSModel
from porepy.examples.geothermal_flow.cf_convergence_analysis.plot_convergence_util import PlotConvergenceUtil
from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_results_io import (
    save_results_json, save_ensemble_json, subdomain_geometry,
)
from porepy.examples.geothermal_flow.cf_convergence_analysis.plot_mms_fields import plot_exact_vs_computed

use_ensemble_of_mesh = False
show_ensemble_members = False
use_schur_technique = False
use_line_search = False
use_anderson = False
use_preconditioner = False
use_anderson_acceleration = False
appleyard_chop = False
appleyard_chop_value = 0.2
anderson_gate = 5.0

NONLINEAR_TOLERANCE = 1.0e-8
"""float: Tolerance on both the nonlinear increment and the residual."""

PREDICTED_OOC = {
    "matrix_pressure": 2, "fracture_pressure": 2,
    "matrix_enthalpy": 1, "fracture_enthalpy": 1,
    "matrix_z_NaCl": 1, "fracture_z_NaCl": 1,
    "interface_darcy_flux": 1,
    "interface_component_flux_NaCl": 1, 
    "interface_advective_energy_flux": 1,
    "interface_conductive_energy_flux": 1,
    "interface_energy_flux": 1,
}

class ParamsAwareConvergenceAnalysis(ConvergenceAnalysis):
    
    def run_analysis(self) -> list:
        convergence_results = []
        for level in range(self.levels):
            params = deepcopy(self.model_params[level])
            model = self.model_class(deepcopy(params))
            if not model._is_time_dependent():
                pp.run_stationary_model(model, params)
                setattr(model.results[-1], "cell_diameter", model.mdg.diameter())
            else:
                pp.run_time_dependent_model(model, params)   # <-- pass params
                setattr(model.results[-1], "cell_diameter", model.mdg.diameter())
                setattr(model.results[-1], "dt", model.time_manager.dt)
            
            setattr(model.results[-1], "geometry", subdomain_geometry(model.mdg))
            convergence_results.append(model.results[-1])
        return convergence_results


def model_parameters(grid_type="cartesian", levels=3, base_cell_size=0.125, dt=0.5, final_time=1.0):
    """Model parameters of the manufactured-solution test.

    Parameters:
        grid_type: ``"cartesian"`` or ``"simplex"``.
        levels: Number of refinement levels.
        base_cell_size: Cell size of the coarsest level.
        dt: Time step of the coarsest level.
        final_time: End time.

    Returns:
        The model parameter dictionary.

    """
    constants = MMS_HALITE
    tolerance = NONLINEAR_TOLERANCE
    return {
        "convergence": False,
        "levels": levels,
        "debug_mms_residual": True,
        "mms_constants": constants,
        "grid_type": grid_type,
        "meshing_arguments": {"cell_size": base_cell_size},
        "time_manager": pp.TimeManager([0.0, final_time], dt, True),
        "material_constants": {"solid": pp.SolidConstants(
            residual_aperture=constants.a_res, permeability=constants.K0,
            normal_permeability=constants.K_k, porosity=constants.phi0,
            thermal_conductivity=constants.kappa_r, density=constants.rho_r,
            specific_heat_capacity=constants.cp_r,
        )},
        "reference_variable_values": pp.ReferenceVariableValues(
            pressure=constants.p0, temperature=constants.T0,
        ),
        "fractional_flow": False,
        "eliminate_reference_phase": True,
        "eliminate_reference_component": True,
        "prepare_simulation": False,
        "enable_buoyancy_effects": False,
        "max_iterations": 200,
        "flag_failure_as_diverged": False,
        "nl_convergence_tol": tolerance,
        "nl_convergence_tol_res": tolerance,
        "linear_solver": "pypardiso",
        "nonlinear_solver": LineSearchNewtonSolver,
        "global_line_search": use_line_search,
        "local_line_search": False,
        "folder_name": "mms_halite_output",
    }


def error_column_names(results):
    """Names of the reported error quantities.

    Parameters:
        results: Per-level result objects.

    Returns:
        The quantity names, those with a predicted order first.

    """
    available = {
        name.removeprefix("error_")
        for name in vars(results[0])
        if name.startswith("error_")
    }
    predicted = [name for name in PREDICTED_OOC if name in available]
    return predicted + sorted(available - set(predicted))


def report_errors(results):
    """Print every error at every level, with the observed rate over each interval.

    Parameters:
        results: Per-level result objects, coarsest first.

    """
    names = error_column_names(results)
    diameters = [getattr(r, "cell_diameter", float("nan")) for r in results]
    levels = len(results)

    print("\nErrors at every level, with the rate over each interval")
    print(f"{'':34s}" + "".join(f"{'h=' + format(d, '.4g'):>14s}" for d in diameters))
    header = f"{'quantity':34s}" + "".join(f"{'level ' + str(i):>14s}" for i in range(levels))
    header += "  |" + "".join(f"{format(i, 'd') + '->' + format(i + 1, 'd'):>9s}"
                              for i in range(levels - 1))
    print(header)
    print("-" * len(header))
    for name in names:
        values = [getattr(r, "error_" + name) for r in results]
        row = f"{name:34s}" + "".join(f"{v:14.4e}" for v in values)
        row += "  |"
        for i in range(levels - 1):
            # Use the measured cell diameters rather than assuming exact halving.
            ratio = np.log(diameters[i] / diameters[i + 1])
            rate = np.log(values[i] / values[i + 1]) / ratio if ratio and ratio > 0 and values[i + 1] > 0 else float("nan")
            row += f"{rate:9.2f}"
        print(row)


def report_convergence(analysis, results):
    """Print the observed orders, over all levels and the last two, against the predicted ones.

    Parameters:
        analysis: Convergence analysis that produced the results.
        results: Per-level result objects.

    """
    report_errors(results)

    observed = analysis.order_of_convergence(results)
    asymptotic = analysis.order_of_convergence(results, data_range=slice(-2, None))
    print(f"\n{'quantity':34s} {'predicted':>10s} {'observed':>10s} {'asymptotic':>12s}")
    for name in PREDICTED_OOC:
        key = "ooc_" + name
        print(f"{name:34s} {PREDICTED_OOC[name]:10d} {observed.get(key, float('nan')):10.3f} "
              f"{asymptotic.get(key, float('nan')):12.3f}")
    extra = sorted(set(observed) - {"ooc_" + n for n in PREDICTED_OOC})
    if extra:
        print("\nreported without a pre-recorded prediction:")
        for key in extra:
            print(f"{key.removeprefix('ooc_'):34s} {'--':>10s} {observed[key]:10.3f} "
                  f"{asymptotic.get(key, float('nan')):12.3f}")

model_params = model_parameters( # This is giving the right rate!!
    grid_type="cartesian", levels=3, base_cell_size=0.125, dt= 0.5)

model_params.update(convergence=True, debug_mms_residual=True)

def main(params=None):
    """Run the convergence study.

    Parameters:
        params: Complete model parameters; ``model_params`` if None.

    """
    params = deepcopy(model_params if params is None else params)
    constants = params["mms_constants"]
    print("Predicted orders (fixed before running):", PREDICTED_OOC)
    print(f"S1 = {constants.S1}  "
          f"({'aperture constant in time' if constants.S1 == 0 else 'aperture feedback active'})")
    print("Increment and residual targets:", params["nl_convergence_tol"])

    if params["convergence"]:
        if use_ensemble_of_mesh:

            from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_mesh_ensemble import (
            EnsembleConvergenceAnalysis, plot_ensemble_members)
            params.update(prepare_simulation=True)
            params["mesh_ensemble"] = {
                "jitters": { # mesh_size_perturbations
                    0: tuple(np.linspace(-0.3, 0.3, 25)),
                    1: tuple(np.linspace(-0.3, 0.3, 25)),
                    2: (0.0, -0.16, 0.16, -0.12, 0.12, -0.08, 0.08, -0.04, 0.04)
                }
            }

            analysis = EnsembleConvergenceAnalysis(
                HaliteMMSModel, params, levels=params["levels"],
                spatial_refinement_rate=2, temporal_refinement_rate=4
            )
            results = analysis.run_analysis()              # averaged, one per level
            report_convergence(analysis, results)          # unchanged
            analysis.report_ensemble()
            print(analysis.bootstrap_order_of_convergence("interface_conductive_energy_flux"))

            out = Path(params["folder_name"])
            grid_type = params["grid_type"]
            save_ensemble_json(
                analysis, results, out / f"mms_results_ensemble_{grid_type}.json",
                meta={"grid_type": grid_type, "levels": params["levels"],
                      "base_cell_size": params["meshing_arguments"]["cell_size"],
                      "S1": constants.S1, "predicted_ooc": PREDICTED_OOC,
                      "jitters": params["mesh_ensemble"]["jitters"], "average": "arithmetic"},
            )

            fig, (_, ax) = PlotConvergenceUtil(results).plot()
            if show_ensemble_members:
                plot_ensemble_members(ax, analysis.ensemble_results,
                                      "interface_conductive_energy_flux", color="C3")
            fig.savefig(out / f"convergence_ensemble_{grid_type}.pdf", bbox_inches="tight")

        else:
            params.update(prepare_simulation=True)
            analysis = ParamsAwareConvergenceAnalysis(
                HaliteMMSModel, params, levels=params["levels"],
                spatial_refinement_rate=2, temporal_refinement_rate=4,
            )
            results = analysis.run_analysis()
            report_convergence(analysis, results)
            out = Path(params["folder_name"])
            grid_type = params["grid_type"]
            save_results_json(
                results, out / f"mms_results_{grid_type}.json",
                meta={"grid_type": grid_type, "levels": params["levels"],
                      "base_cell_size": params["meshing_arguments"]["cell_size"],
                      "S1": constants.S1, "predicted_ooc": PREDICTED_OOC},
            )
            if grid_type.startswith("cartesian"):
                PlotConvergenceUtil(results).plot(
                    save_path=out / f"convergence_{grid_type}.png",
                    xlabel=None,
                    title=None,
                    primary_title=None,
                    interface_title=None,
                    show_legend=False,
                    **PlotConvergenceUtil.SLIDE_STYLE,
                )
            else:
                PlotConvergenceUtil(results).plot(
                    save_path=out / f"convergence_{grid_type}.png",
                    **PlotConvergenceUtil.SLIDE_STYLE,
                )

            plot_exact_vs_computed(results, out / f"exact_vs_computed_{grid_type}.png")

    else:
        model = HaliteMMSModel(params)
        model.prepare_simulation()
        pp.run_time_dependent_model(model, params)
        result = model.results[-1]
        for name in ("darcy_flux", "component_flux_NaCl", "energy_flux"):
            values = result.values[f"interface_{name}"]
            print(f"R1 {name}: max|numerical|={max(abs(values['approximate'])):.8g}")
        print("Per-side interface errors:", result.interface_sides)


if __name__ == "__main__":
    main()
