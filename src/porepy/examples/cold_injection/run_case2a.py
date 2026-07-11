"""2D, 2-phase water flow through horizontal fracture domain with temporal aperture
jump.

Isothermal model using a global pT-formulation and the possibility to do nonlinear
vT-preconditioning.

"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from functools import partial
from typing import Callable, cast

# os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np

import porepy as pp
from porepy.examples.cold_injection.config import (
    get_default_convergence_criteria,
    get_default_params,
    get_rpc,
    set_schur_complement,
)
from porepy.examples.cold_injection.geometry import HorizontalFractureAndPointWells2D
from porepy.examples.cold_injection.model import (
    ColdInjectionMixins,
    FluidPoreInteraction,
    NoFluxRediscretization,
)
from porepy.models.compositional_flow_with_equilibrium import (
    IsothermalCFLEModelTemplate,
)

JUMP_TIME = 25 * pp.DAY
T_BEFORE_JUMP = JUMP_TIME - pp.HOUR
T_END_DAYS = 50
time_schedule = [i * pp.DAY for i in range(T_END_DAYS)]
dt_init = pp.DAY * 0.5
dt_min = pp.SECOND

newton_tol_res = 1e-7
newton_tol_res_isofug = 1e-2
newton_tol_inc = 1.0
max_iterations = 25
iter_range = (15, max_iterations)


def modify_schedule(old_schedule: list[float]) -> list[float]:
    t = np.array(old_schedule).copy()
    t_before: list[float] = t[t < JUMP_TIME].tolist()
    t_after: list[float] = t[t > JUMP_TIME].tolist()
    if t_before[-1] < T_BEFORE_JUMP:
        t_before += [T_BEFORE_JUMP]
    if t_after[0] > JUMP_TIME + pp.HOUR:
        t_before += (
            np.arange(JUMP_TIME, JUMP_TIME + pp.HOUR, pp.MINUTE).tolist()
            + np.arange(JUMP_TIME + pp.HOUR, t_after[0], pp.HOUR).tolist()
        )
    return t_before + t_after


model_params, solver_params = get_default_params()

# model_params["linear_solver"] = "pypardiso"  # scipy_sparse default
# model_params["times_to_export"] = time_schedule

eos_params = [1e-4, 1e-2, 1e-3, 10.0]
model_params["flash_params"]["gen_arg_params"] = eos_params
model_params["flash_params"]["phase_property_params"] = eos_params
model_params["phase_property_params"] = eos_params
model_params["flash_params"]["global_iteration_stride"] = None
model_params["flash_params"]["solver_params"]["atol_res"] = 1e-5
model_params["flash_params"]["solver_params"]["max_iterations"] = 25

model_params["equilibrium_specification"] = (
    pp.compositional.FlashSpec.pT,
    "persistent-variables",
)
model_params["flash_params"]["compile_args"] = (
    pp.compositional.FlashSpec.pT,
    pp.compositional.FlashSpec.vT,
)

solver_params["atol_objective"] = newton_tol_res
solver_params["newton_chop"] = None
solver_params["appleyard_chop"] = 0.3
solver_params["pressure_clip"] = (0.9, 1.1)  # (0.8, 1.2)
solver_params["volume_clip"] = (0.9, 1.1)  # (0.8, 1.2)
model_params["use_logp_nonlinear_rpc"] = False

solver_params["do_armijo_line_search"] = False
solver_params["armijo_line_search_weight"] = 0.9
solver_params["armijo_line_search_incline"] = 1e-4
solver_params["armijo_line_search_max_iterations"] = 20
solver_params["armijo_stop_after_residual_reaches"] = 1e-5

solver_params["do_ntrdc"] = True
solver_params["ntrdc_scale_with_inf"] = True
solver_params["ntrdc_return_nan"] = False
solver_params["ntrdc_eta_3"] = 0.5
solver_params["ntrdc_eta_2"] = 0.1
solver_params["ntrdc_delta_tol"] = 1e-7

solver_params["in_physical_space"] = True


class Case2DataCollection(pp.PorePyModel):
    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def __init__(self, params=None):
        super().__init__(params)

        self._p_before_drop: None | np.ndarray = None
        self._T_before_drop: None | np.ndarray = None
        self._y_vanished: bool = False
        self._p_transient_over: bool = False
        self._T_transient_over: bool = False
        self._transient_over: bool = False

    def after_nonlinear_convergence(self):
        frac = self.mdg.subdomains(dim=1)
        gasphase = [
            p
            for p in self.fluid.phases
            if p.state == pp.compositional.PhysicalState.gas
        ][0]
        yG = gasphase.fraction(frac)
        sG = gasphase.saturation(frac)
        cell_volumes = self.wrap_grid_attribute(
            frac, "cell_volumes", dim=1
        ) * self.specific_volume(frac)

        frac_volume = float(self.equation_system.evaluate(cell_volumes).sum())

        yG_avg = (
            float(self.equation_system.evaluate(yG * cell_volumes).sum()) / frac_volume
        )
        sG_avg = (
            float(self.equation_system.evaluate(sG * cell_volumes).sum()) / frac_volume
        )

        self.nonlinear_solver_statistics.log_custom_data(gas_in_frac=yG_avg)
        self.nonlinear_solver_statistics.log_custom_data(sat_in_frac=sG_avg)

        subdomains = self.mdg.subdomains()
        # Calculate pressure drop at jump time.
        t = self.time_manager.time
        if t >= JUMP_TIME - dt_min / 10 and not self._transient_over:
            p_now = self.pressure(subdomains)
            p_now_vals = cast(np.ndarray, self.equation_system.evaluate(p_now))

            l2_norm = pp.ad.Function(partial(pp.ad.l2_norm, 1), "l2_norm")
            diff = lambda x: np.sqrt(
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(
                            l2_norm(x) * l2_norm(x),
                            subdomains,
                            1,
                        )
                    )
                )
            )

            # Safe p-values before a-jump.
            if self._p_before_drop is None:
                self._p_before_drop = cast(
                    np.ndarray, self.equation_system.evaluate(p_now.previous_timestep())
                )

            p_factor = 1e-6  # Convert to MPa
            delta_p_l2 = diff(
                (p_now - pp.ad.DenseArray(self._p_before_drop)) * pp.ad.Scalar(p_factor)
            )
            delta_p_max = np.abs((p_now_vals - self._p_before_drop) * p_factor).max()

            self.nonlinear_solver_statistics.log_custom_data(
                to_global=True,
                append=True,
                transient_t=t,
            )
            self.nonlinear_solver_statistics.log_custom_data(
                to_global=True,
                append=True,
                delta_p_l2_transient=delta_p_l2,
            )
            self.nonlinear_solver_statistics.log_custom_data(
                to_global=True,
                append=True,
                delta_p_max_transient=delta_p_max,
            )

            yG_vals = cast(np.ndarray, self.equation_system.evaluate(yG))
            if np.all(np.abs(yG_vals) <= 1e-10) and not self._y_vanished:
                self._y_vanished = True
                self.nonlinear_solver_statistics.log_custom_data(
                    to_global=True, gas_disappears_time=t
                )

            # if np.linalg.norm(self._p_before_drop - p_now_vals) <= 1e-2:
            # NOTE l2 norm gives the longest transient period, folloed by eucledian 2
            # norm, and max(abs())
            # np.allclose(p_now_vals, self._p_before_drop) gives the shortest.
            if delta_p_l2 < 1 and not self._p_transient_over:
                self._p_transient_over = True
                self.nonlinear_solver_statistics.log_custom_data(
                    to_global=True, p_transient_end_time=t
                )

            self._transient_over = self._y_vanished and self._p_transient_over

            # Register T drop analogous to p.
            if isinstance(self, pp.energy_balance.VariablesEnergyBalance):
                T_now = self.temperature(subdomains)
                T_now_vals = cast(np.ndarray, self.equation_system.evaluate(T_now))

                if self._T_before_drop is None:
                    self._T_before_drop = cast(
                        np.ndarray,
                        self.equation_system.evaluate(T_now.previous_timestep()),
                    )

                delta_T_l2 = diff(T_now - pp.ad.DenseArray(self._T_before_drop))
                delta_T_max = np.abs(T_now_vals - self._T_before_drop).max()

                self.nonlinear_solver_statistics.log_custom_data(
                    to_global=True,
                    append=True,
                    delta_T_l2_transient=delta_T_l2,
                )
                self.nonlinear_solver_statistics.log_custom_data(
                    to_global=True,
                    append=True,
                    delta_T_max_transient=delta_T_max,
                )

                if delta_T_l2 < 1 and not self._T_transient_over:
                    self._T_transient_over = True
                    self.nonlinear_solver_statistics.log_custom_data(
                        to_global=True, T_transient_end_time=t
                    )

                self._transient_over = self._transient_over and self._T_transient_over

            if self._transient_over:
                self.nonlinear_solver_statistics.log_custom_data(
                    to_global=True, transient_end_time=t
                )

        return super().after_nonlinear_convergence()


def get_case2_argparser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="Cold CO2 injection run script")
    parser.add_argument(
        "-a",
        "--aperture",
        nargs=1,
        default=1.0,
        type=float,
        help="Aperture jump factor (float > 1).",
    )
    parser.add_argument(
        "-e",
        "--extensive",
        action="store_true",
        help="Eliminate the extensive state variables",
    )
    parser.add_argument(
        "-p",
        "--precondition",
        action="store_true",
        help="Deactivate isochoric preconditioning.",
    )

    return parser


def resolve_args(
    args: argparse.Namespace,
) -> tuple[list[tuple[float, float]], bool, bool]:
    if args.aperture:
        if isinstance(args.aperture, list):
            ajump = args.aperture[0]
        else:
            ajump = args.aperture
        ajump = float(ajump)
        assert ajump >= 1, f"Expecting aperture jump factor >1. Got {ajump}."
        if ajump > 1:
            schedule = [(JUMP_TIME, ajump)]
        else:
            schedule = []
    else:
        schedule = []

    if args.extensive:
        e_prim = False
    else:
        e_prim = True

    if args.precondition:
        npc = False
    else:
        npc = True

    return schedule, e_prim, npc


class ModelClass(  # type:ignore
    Case2DataCollection,
    FluidPoreInteraction,
    NoFluxRediscretization,
    HorizontalFractureAndPointWells2D,
    ColdInjectionMixins,
    IsothermalCFLEModelTemplate,
):
    pass


model_params["create_fluid_volume_variable"] = False


ModelClass._COMPONENT_NAMES = ["H2O"]
ModelClass._IDEAL_COMPONENTS = [pp.compositional.ideal.IdealH2O]
# NOTE water density in mol / m^3 at 15 MPa and 300 K using Peng-Robinson.
ModelClass._TOTAL_INJECTED_MASS = 10 * 47134.59273520758 / (60 * 60)
ModelClass._p_INIT = 10e6
ModelClass._p_OUT = 10e6
ModelClass._p_BC = 10e6
ModelClass._T_INIT = 450.0
ModelClass._T_IN = 450.0
ModelClass._T_BC = 450.0
ModelClass._z_INIT = {"H2O": 1.0}
ModelClass._z_IN = {"H2O": 1.0}


if __name__ == "__main__":
    parser = get_case2_argparser("CI Case 2a.")
    APERTURE_JUMP_SCHEDULE, E_PRIMARY, ISOCHORIC_NPC = resolve_args(parser.parse_args())
    E_PRIMARY = False

    # NOTE for debugging
    # APERTURE_JUMP_SCHEDULE = [(JUMP_TIME, 10)]
    # ISOCHORIC_NPC = True

    ajump: float | None
    if APERTURE_JUMP_SCHEDULE:
        ajump = APERTURE_JUMP_SCHEDULE[0][1]
        time_schedule = modify_schedule(time_schedule)
    else:
        ajump = None

    ModelClass._APERTURE_FACTOR_AFTER_TIME = APERTURE_JUMP_SCHEDULE
    model_params["time_manager"] = pp.TimeManager(
        schedule=time_schedule,
        dt_init=dt_init,
        dt_min_max=(dt_min, np.max(np.diff(np.array(time_schedule)))),
        iter_max=max_iterations,
        iter_optimal_range=iter_range,
        iter_relax_factors=(0.75, 1.5),
        recomp_factor=0.5,
        recomp_max=10,
        print_info=True,
        atol=5e-15,
    )

    if ISOCHORIC_NPC:
        ModelClass._ISOCHORIC_NPC_SPEC = pp.compositional.FlashSpec.vT

    timestamp = datetime.today().strftime("%d%B%Y_%H-%M-%S")
    sub_folder = (
        "CI_CASE2A/"
        f"{timestamp}"
        f"_AJUMP_{ajump}"
        f"_ICHOR_{bool(ISOCHORIC_NPC)}"
        # f"_STRIDE_{model_params["flash_params"]["global_iteration_stride"]}"
        f"_EPRIM_{bool(E_PRIMARY)}"
    )
    model_params["folder_name"] = f"visualization/{sub_folder}"

    print(
        f"\nStarting simulation : {sub_folder}\n"
        f"Aperture jump: {ajump}\n"
        f"Extensives primary: {E_PRIMARY}\n"
        f"Do isochoric NPC: {ISOCHORIC_NPC}\n"
    )

    model = ModelClass(model_params)  # type:ignore[abstract]

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy").setLevel(logging.DEBUG)
    t_0 = time.time()
    model.prepare_simulation()
    prep_sim_time = time.time() - t_0
    logging.getLogger("porepy").setLevel(logging.INFO)

    model.params["linear_right_preconditioner"] = get_rpc(model)  # type:ignore

    # Defining sub system for Schur complement reduction.
    set_schur_complement(model, use_extensives=E_PRIMARY)  # type:ignore[arg-type]
    solver_params.update(
        get_default_convergence_criteria(
            model, max_iterations, newton_tol_res, newton_tol_inc, newton_tol_res_isofug
        )
    )

    t_0 = time.time()
    pp.run_time_dependent_model(model, solver_params)
    sim_time = time.time() - t_0

    print(f"Simulation prepared after {str(timedelta(seconds=prep_sim_time))}")
    print(f"Simulation finished after {str(timedelta(seconds=sim_time))}")
