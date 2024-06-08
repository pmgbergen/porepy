"""This module implements a multiphase-multicomponent flow model with phase change
using the Soereide model to define the fluid mixture.

Note:
    It uses the numba-compiled version of the Peng-Robinson EoS and a numba-compiled
    unified flash.

    Import and compilation take some time.

References:
    [1] Ingolf Søreide, Curtis H. Whitson,
        Peng-Robinson predictions for hydrocarbons, CO2, N2, and H2 S with pure water
        and NaCI brine,
        Fluid Phase Equilibria,
        Volume 77,
        1992,
        https://doi.org/10.1016/0378-3812(92)85105-H

"""

from __future__ import annotations

import logging
import os
import pathlib
import time

# os.environ["NUMBA_DISABLE_JIT"] = "1"
compile_time = 0.0

logging.basicConfig(level=logging.INFO)
logging.getLogger("porepy").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from typing import Sequence, cast

import numpy as np

import porepy as pp

t_0 = time.time()
import porepy.compositional as ppc
import porepy.compositional.peng_robinson as ppcpr

compile_time += time.time() - t_0
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.applications.md_grids.domains import nd_cube_domain

logger = logging.getLogger(__name__)


class SoereideMixture:
    """Model fluid using the Soereide mixture, a Peng-Robinson based EoS for
    NaCl brine with CO2, H2S and N2."""

    def get_components(self) -> Sequence[ppc.Component]:
        chems = ["H2O", "CO2"]
        species = ppc.load_species(chems)
        components = [
            ppcpr.H2O.from_species(species[0]),
            ppcpr.H2S.from_species(species[1]),
        ]
        return components

    def get_phase_configuration(
        self, components: Sequence[ppc.Component]
    ) -> Sequence[tuple[ppc.EoSCompiler, int, str]]:
        eos = ppcpr.PengRobinsonCompiler(components)
        return [(eos, 0, "L"), (eos, 1, "G")]


class CompiledFlash(ppc.FlashMixin):
    """Sets the compiled flash as the flash class, consistent with the EoS."""

    flash_params = {
        "mode": "parallel",
    }

    def set_up_flasher(self) -> None:
        eos = self.fluid_mixture.reference_phase.eos
        eos = cast(ppcpr.PengRobinsonCompiler, eos)

        flash = ppc.CompiledUnifiedFlash(self.fluid_mixture, eos)

        # Compiling the flash and the EoS
        eos.compile()
        flash.compile()

        # Configuring solver for the requested equilibrium type
        flash.tolerance = 1e-8
        flash.max_iter = 150
        flash.heavy_ball_momentum = False
        flash.armijo_parameters["rho"] = 0.99
        flash.armijo_parameters["kappa"] = 0.4
        flash.armijo_parameters["max_iter"] = (
            50 if self.equilibrium_type == "p-T" else 30
        )
        flash.npipm_parameters["u1"] = 10.0
        flash.npipm_parameters["u2"] = (
            10.0  # if self.equilibrium_type == "p-T" else 1.0
        )
        flash.npipm_parameters["eta"] = 0.5
        flash.initialization_parameters["N1"] = 3
        flash.initialization_parameters["N2"] = 1
        flash.initialization_parameters["N3"] = 5
        flash.initialization_parameters["eps"] = flash.tolerance

        # Setting the attribute of the mixin
        self.flash = flash

    def get_fluid_state(
        self, subdomains: Sequence[pp.Grid], state: np.ndarray | None = None
    ) -> ppc.FluidState:
        """Method to pre-process the evaluated fractions. Normalizes the extended
        fractions where they violate the unity constraint."""
        fluid_state = super().get_fluid_state(subdomains, state)

        # sanity checks
        p = self.equation_system.get_variable_values(
            [self.pressure_variable], iterate_index=0
        )
        T = self.equation_system.get_variable_values(
            [self.temperature_variable], iterate_index=0
        )
        if np.any(p <= 0.0):
            raise ValueError("Pressure diverged to negative values.")
        if np.any(T <= 0):
            raise ValueError("Temperature diverged to negative values.")

        unity_tolerance = 0.05

        sat_sum = np.sum(fluid_state.sat, axis=0)
        if np.any(sat_sum > 1 + unity_tolerance):
            raise ValueError("Saturations violate unity constraint")
        y_sum = np.sum(fluid_state.y, axis=0)
        idx = fluid_state.y < 0
        if np.any(idx):
            fluid_state.y[idx] = 0.0
        idx = fluid_state.y > 1.0
        if np.any(idx):
            fluid_state.y[idx] = 1.0
        y_sum = np.sum(fluid_state.y, axis=0)
        idx = y_sum > 1.0
        if np.any(idx):
            fluid_state.y[:, idx] = ppc.normalize_rows(fluid_state.y[:, idx].T).T

        # if np.any(y_sum > 1 + unity_tolerance):
        #     raise ValueError("Phase fractions violate unity constraint")
        # if np.any(fluid_state.y < 0 - unity_tolerance) or np.any(
        #     fluid_state.y > 1 + unity_tolerance
        # ):
        #     raise ValueError("Phase fractions out of bound.")

        for phase in fluid_state.phases:
            x_sum = phase.x.sum(axis=0)
            idx = x_sum > 1.0
            if np.any(idx):
                phase.x[:, idx] = ppc.normalize_rows(phase.x[:, idx].T).T
            idx = phase.x < 0
            if np.any(idx):
                phase.x[idx] = 0.0
            idx = phase.x > 1.0
            if np.any(idx):
                phase.x[idx] = 1.0
            # if np.any(x_sum > 1 + unity_tolerance):
            #     raise ValueError(
            #         f"Extended fractions in phase {j} violate unity constraint."
            #     )
            # if np.any(phase.x < 0 - unity_tolerance) or np.any(
            #     phase.x > 1 + unity_tolerance
            # ):
            #     raise ValueError(f"Extended fractions in phase {j} out of bound.")

        return fluid_state

    def postprocess_failures(
        self, fluid_state: ppc.FluidState, success: np.ndarray
    ) -> ppc.FluidState:
        """A post-processing where the flash is again attempted where not succesful.

        But the new attempt does not use iterate values as initial guesses, but computes
        the flash from scratch.

        """
        failure = success > 0
        if np.any(failure):
            sds = self.mdg.subdomains()
            logger.warning(
                f"Flash from iterate state failed in {failure.sum()} cases."
                + " Re-starting with computation of initial guess."
            )
            z = [
                comp.fraction(sds).value(self.equation_system)[failure]
                for comp in self.fluid_mixture.components
            ]
            p = self.pressure(sds).value(self.equation_system)[failure]
            h = self.enthalpy(sds).value(self.equation_system)[failure]
            T = self.temperature(sds).value(self.equation_system)[failure]

            logger.info(f"Failed at\nz: {z}\np: {p}\nT: {T}\nh: {h}")
            # no initial guess, and this model uses only p-h flash.
            flash_kwargs = {
                "z": z,
                "p": p,
                "parameters": self.flash_params,
            }

            if self.equilibrium_type == "p-h":
                flash_kwargs["h"] = h
            elif self.equilibrium_type == "p-T":
                flash_kwargs["T"] = T

            sub_state, sub_success, _ = self.flash.flash(**flash_kwargs)

            # treat max iter reached as success, and hope for the best in the PDE iter
            sub_success[sub_success == 1] = 0
            # update parent state with sub state values
            success[failure] = sub_success
            fluid_state.T[failure] = sub_state.T
            fluid_state.h[failure] = sub_state.h
            fluid_state.rho[failure] = sub_state.rho

            for j in range(len(fluid_state.phases)):
                fluid_state.sat[j][failure] = sub_state.sat[j]
                fluid_state.y[j][failure] = sub_state.y[j]

                fluid_state.phases[j].x[:, failure] = sub_state.phases[j].x

                fluid_state.phases[j].rho[failure] = sub_state.phases[j].rho
                fluid_state.phases[j].h[failure] = sub_state.phases[j].h
                fluid_state.phases[j].mu[failure] = sub_state.phases[j].mu
                fluid_state.phases[j].kappa[failure] = sub_state.phases[j].kappa

                fluid_state.phases[j].drho[:, failure] = sub_state.phases[j].drho
                fluid_state.phases[j].dh[:, failure] = sub_state.phases[j].dh
                fluid_state.phases[j].dmu[:, failure] = sub_state.phases[j].dmu
                fluid_state.phases[j].dkappa[:, failure] = sub_state.phases[j].dkappa

                fluid_state.phases[j].phis[:, failure] = sub_state.phases[j].phis
                fluid_state.phases[j].dphis[:, :, failure] = sub_state.phases[j].dphis

        # Parent method performs a check that everything is successful.
        return super().postprocess_failures(fluid_state, success)


class ModelGeometry:
    def set_domain(self) -> None:
        size = self.solid.convert_units(1, "m")
        self._domain = nd_cube_domain(2, size)

    # def set_fractures(self) -> None:
    #     """Setting a diagonal fracture"""
    #     frac_1_points = self.solid.convert_units(
    #         np.array([[0.2, 1.8], [0.2, 1.8]]), "m"
    #     )
    #     frac_1 = pp.LineFracture(frac_1_points)
    #     self._fractures = [frac_1]

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.05, "m")
        cell_size_fracture = self.solid.convert_units(0.05, "m")
        mesh_args: dict[str, float] = {
            "cell_size": cell_size,
            "cell_size_fracture": cell_size_fracture,
        }
        return mesh_args


class InitialConditions:
    """Define initial pressure, temperature and compositions."""

    _p_IN: float
    _p_OUT: float

    _p_INIT: float = 12e6
    _T_INIT: float = 540.0
    _z_INIT: dict[str, float] = {"H2O": 0.995, "CO2": 0.005}

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        # f = lambda x: self._p_IN + x * (self._p_OUT - self._p_IN)
        # vals = np.array(list(map(f, sd.cell_centers[0])))
        # return vals
        return np.ones(sd.num_cells) * self._p_INIT

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        return np.ones(sd.num_cells) * self._T_INIT

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        return np.ones(sd.num_cells) * self._z_INIT[component.name]


class BoundaryConditions:
    """Boundary conditions defining a ``left to right`` flow in the matrix (2D)

    Mass flux:

    - No flux conditions on top and bottom
    - Dirichlet data (pressure, temperatyre and composition) on left and right faces

    Heat flux:

    - No flux on left, top, right
    - heated bottom side (given by temperature as a Dirichlet-type BC)

    Trivial Neumann conditions for fractures.

    """

    mdg: pp.MixedDimensionalGrid

    has_time_dependent_boundary_equilibrium = False
    """Constant BC for primary variables, hence constant BC for all other."""

    _p_INIT: float
    _T_INIT: float
    _z_INIT: dict[str, float]

    _p_IN: float = 15e6
    _p_OUT: float = InitialConditions._p_INIT

    _T_IN: float = InitialConditions._T_INIT
    _T_OUT: float = InitialConditions._T_INIT
    _T_HEATED: float = 630.0

    _z_IN: dict[str, float] = {
        "H2O": InitialConditions._z_INIT["H2O"] - 0.105,
        "CO2": InitialConditions._z_INIT["CO2"] + 0.105,
    }
    _z_OUT: dict[str, float] = {
        "H2O": InitialConditions._z_INIT["H2O"],
        "CO2": InitialConditions._z_INIT["CO2"],
    }

    def _inlet_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define inlet."""
        sides = self.domain_boundary_sides(sd)

        inlet = np.zeros(sd.num_faces, dtype=bool)
        inlet[sides.west] = True
        inlet &= sd.face_centers[1] > 0.2
        inlet &= sd.face_centers[1] < 0.5

        return inlet

    def _outlet_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define outlet."""

        sides = self.domain_boundary_sides(sd)

        outlet = np.zeros(sd.num_faces, dtype=bool)
        outlet[sides.east] = True
        outlet &= sd.face_centers[1] > 0.75
        outlet &= sd.face_centers[1] < 0.99

        return outlet

    def _heated_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define heated boundary with D-type conditions for conductive flux."""
        sides = self.domain_boundary_sides(sd)

        heated = np.zeros(sd.num_faces, dtype=bool)
        heated[sides.south] = True
        heated &= sd.face_centers[0] > 0.2
        heated &= sd.face_centers[0] < 0.8

        return heated

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # Setting only conditions on matrix
        if sd.dim == 2:

            inout = self._inlet_faces(sd) | self._outlet_faces(sd)
            # Define boundary condition on all boundary faces.
            return pp.BoundaryCondition(sd, inout, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if sd.dim == 2:
            inout = self._inlet_faces(sd) | self._outlet_faces(sd)
            heated = self._heated_faces(sd) | inout
            return pp.BoundaryCondition(sd, heated, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:

        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * self._p_INIT

            inlet = self._inlet_faces(sd)
            outlet = self._outlet_faces(sd)

            # vals[sides.west] = 10e6
            vals[inlet] = self._p_IN
            vals[outlet] = self._p_OUT

            vals = vals[sides.all_bf]
        else:
            vals = np.zeros(boundary_grid.num_cells)

        return vals

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:

        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * self._T_INIT

            inlet = self._inlet_faces(sd)
            outlet = self._outlet_faces(sd)
            heated = self._heated_faces(sd)

            vals[inlet] = self._T_IN
            vals[outlet] = self._T_OUT
            vals[heated] = self._T_HEATED

            vals = vals[sides.all_bf]
        else:
            vals = np.zeros(boundary_grid.num_cells)

        return vals

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:

        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * self._z_INIT[component.name]

            inlet = self._inlet_faces(sd)
            outlet = self._outlet_faces(sd)

            vals[inlet] = self._z_IN[component.name]
            vals[outlet] = self._z_OUT[component.name]

            vals = vals[sides.all_bf]
        else:
            vals = np.zeros(boundary_grid.num_cells)

        return vals


class GeothermalFlow(
    ModelGeometry,
    SoereideMixture,
    CompiledFlash,
    InitialConditions,
    BoundaryConditions,
    cfle.CFLEModelMixin_ph,
):
    """Geothermal flow using a fluid defined by the Soereide model and the compiled
    flash."""

    def after_nonlinear_failure(self):
        self.exporter.write_pvd()
        super().after_nonlinear_failure()


days = 365
t_scale = 1e-5
T_end = 40 * days * t_scale
dt_init = 1 * days * t_scale / 2
max_iterations = 80
newton_tol = 1e-6
newton_tol_increment = newton_tol

time_manager = pp.TimeManager(
    schedule=[0, T_end],
    dt_init=dt_init,
    dt_min_max=(0.1 * dt_init, 2 * dt_init),
    iter_max=max_iterations,
    iter_optimal_range=(2, 10),
    iter_relax_factors=(0.9, 1.1),
    recomp_factor=0.1,
    recomp_max=5,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {
        "permeability": 1e-8,
        "porosity": 0.2,
        "thermal_conductivity": 1000.0,
        "specific_heat_capacity": 500.0,
    }
)
material_constants = {"solid": solid_constants}

restart_options = {
    "restart": True,
    "is_mdg_pvd": True,
    "pdv_file": pathlib.Path("./visualization/data.pvd"),
}

# Model setup:
# eliminate reference phase fractions  and reference component.
params = {
    "material_constants": material_constants,
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "normalize_state_constraints": True,
    "use_semismooth_complementarity": True,
    "reduce_linear_system_q": False,
    "time_manager": time_manager,
    "max_iterations": max_iterations,
    "nl_convergence_tol": newton_tol_increment,
    "nl_convergence_tol_res": newton_tol,
    "prepare_simulation": False,
    "progressbars": True,
    # 'restart_options': restart_options,  # NOTE no yet fully integrated in porepy
}
model = GeothermalFlow(params)

model.equilibrium_type = "p-h"

t_0 = time.time()
model.prepare_simulation()
prep_sim_time = time.time() - t_0
compile_time += prep_sim_time
t_0 = time.time()
pp.run_time_dependent_model(model, params)
sim_time = time.time() - t_0
print(f"Finished prepare_simulation in {prep_sim_time} seconds.")
print(f"Finished simulation in {sim_time} seconds.")

N = len(model._stats)
tx = np.linspace(0, T_end, N, endpoint=True)

# values per time step
num_iter = list()
# residuals before flash, max, avg, min
res_bf_max = list()
res_bf_med = list()
res_bf_min = list()
# residuals after flash, max avg min
res_af_max = list()
res_af_med = list()
res_af_min = list()
# residuals at assembly, max avg min
res_aa_max = list()
res_aa_med = list()
res_aa_min = list()
# averages over iterations per time step
time_assembly_avg = list()
time_linsolve_avg = list()
time_flash_avg = list()

for t in range(N):
    stats = model._stats[t]
    ni = len(stats)
    num_iter.append(ni)

    time_assembly = list()
    time_linsolve = list()
    time_flash = list()
    res_bf = list()
    res_af = list()
    res_aa = list()

    for i in range(ni):
        si = stats[i]
        if "time_assembly" in si:
            time_assembly.append(si["time_assembly"])
        if "time_linsolve" in si:
            time_linsolve.append(si["time_linsolve"])
        if "time_flash" in si:
            if i == 0 and t == 0:
                # adding this time to compile time, because first call of p-h flash
                compile_time += si["time_flash"]
            else:
                time_flash.append(si["time_flash"])
        if "residual_before_flash" in si:
            res_bf.append(si["residual_before_flash"])
        if "residual_after_flash" in si:
            res_af.append(si["residual_after_flash"])
        if "residual_at_assembly" in si:
            res_aa.append(si["residual_at_assembly"])

    time_assembly_avg.append(np.average(time_assembly))
    time_linsolve_avg.append(np.average(time_linsolve))
    time_flash_avg.append(np.average(time_flash))

    if len(res_bf) > 0:
        res_bf_max.append(np.max(res_bf))
        res_bf_med.append(np.median(res_bf))
        res_bf_min.append(np.min(res_bf))
    if len(res_af) > 0:
        res_af_max.append(np.max(res_af))
        res_af_med.append(np.median(res_af))
        res_af_min.append(np.min(res_af))
    if len(res_aa) > 0:
        res_aa_max.append(np.max(res_aa))
        res_aa_med.append(np.median(res_aa))
        res_aa_min.append(np.min(res_aa))

print(f"Estimated compile time: {compile_time} (s)")

num_iter = np.array(num_iter)
res_bf_max = np.array(res_bf_max)
res_bf_med = np.array(res_bf_med)
res_bf_min = np.array(res_bf_min)
res_af_max = np.array(res_af_max)
res_af_med = np.array(res_af_med)
res_af_min = np.array(res_af_min)
res_aa_max = np.array(res_aa_max)
res_aa_med = np.array(res_aa_med)
res_aa_min = np.array(res_aa_min)
time_assembly_avg = np.array(time_assembly_avg)
time_linsolve_avg = np.array(time_linsolve_avg)
time_flash_avg = np.array(time_flash_avg)

num_iter.tofile(pathlib.Path("./visualization/num_iter.csv", sep=";"))
res_bf_max.tofile(pathlib.Path("./visualization/res_bf_max.csv", sep=";"))
res_bf_med.tofile(pathlib.Path("./visualization/res_bf_med.csv", sep=";"))
res_bf_min.tofile(pathlib.Path("./visualization/res_bf_min.csv", sep=";"))
res_af_max.tofile(pathlib.Path("./visualization/res_af_max.csv", sep=";"))
res_af_med.tofile(pathlib.Path("./visualization/res_af_med.csv", sep=";"))
res_af_min.tofile(pathlib.Path("./visualization/res_af_min.csv", sep=";"))
res_aa_max.tofile(pathlib.Path("./visualization/res_aa_max.csv", sep=";"))
res_aa_med.tofile(pathlib.Path("./visualization/res_aa_med.csv", sep=";"))
res_aa_min.tofile(pathlib.Path("./visualization/res_aa_min.csv", sep=";"))
time_assembly_avg.tofile(pathlib.Path("./visualization/time_assembly_avg.csv", sep=";"))
time_linsolve_avg.tofile(pathlib.Path("./visualization/time_linsolve_avg.csv", sep=";"))
time_flash_avg.tofile(pathlib.Path("./visualization/time_flash_avg.csv", sep=";"))

font_size = 20
plt.rc("font", size=font_size)  # controls default text size
plt.rc("axes", titlesize=font_size)  # fontsize of the title
plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=font_size)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=font_size)  # fontsize of the y tick labels
plt.rc("legend", fontsize=17)  # fontsize of the legend

# region Num iter and time
fig = plt.figure(figsize=(2 * 8.0, 8.0))
axis = fig.add_subplot(1, 2, 1)
axis.set_box_aspect(1)
axis.set_xlabel("Simulation time")
axis.set_ylabel("Number of iterations")
axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
axis.yaxis.set_major_locator(MaxNLocator(integer=True))

img = axis.plot(tx, num_iter, linestyle="solid", color="black", linewidth=3)
axis.set_ylim((1, np.max(num_iter)))

axis = fig.add_subplot(1, 2, 2)
axis.set_box_aspect(1)
axis.set_xlabel("Simulation time")
axis.set_ylabel("Avg. computational time")
axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
axis.set_yscale("log")

img_1 = axis.plot(tx, time_flash_avg, linestyle="solid", color="red", linewidth=3)
img_2 = axis.plot(tx, time_assembly_avg, linestyle="solid", color="blue", linewidth=3)
img_3 = axis.plot(tx, time_linsolve_avg, linestyle="solid", color="black", linewidth=3)

axis.legend(
    img_1 + img_2 + img_3,
    ["cell-wise flash", "assembly", "lin-solve"],
    loc="upper right",
    markerscale=3,
)

fig.tight_layout(pad=0.05, h_pad=0.05, w_pad=1)
fig.savefig(
    str(pathlib.Path("./visualization/iter_and_time.png").resolve()),
    format="png",
    dpi=400,
)

# endregion

# region Whole plot residual

fig = plt.figure(figsize=(2 * 8.0, 8.0))
axis = fig.add_subplot(1, 1, 1)
axis.set_box_aspect(0.5)
axis.set_xlabel("Simulation time")
axis.set_ylabel("l2-norm residual")
axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
axis.set_yscale("log")

img_1 = axis.plot(tx, res_bf_med, linestyle="solid", color="blue", linewidth=3)
img_1f = axis.fill_between(tx, res_bf_max, res_bf_min, alpha=0.2, color="blue")
img_2 = axis.plot(tx, res_af_med, linestyle="solid", color="red", linewidth=3)
img_2f = axis.fill_between(tx, res_af_max, res_af_min, alpha=0.2, color="red")
img_3 = axis.plot(tx, res_aa_med, linestyle="solid", color="black", linewidth=3)
img_3f = axis.fill_between(
    tx, res_aa_max, res_aa_min, alpha=0.2, color="black", facecolor="grey", hatch="/"
)
r = axis.get_ylim()
axis.set_ylim((r[0] * 1e-1, r[1]))

axis.legend(
    img_1 + [img_1f] + img_2 + [img_2f] + img_3 + [img_3f],
    [
        "med. before flash",
        "range bf",
        "med. after flash",
        "range af",
        "med. after re-discr.",
        "range ad",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
    ncol=3,
    fancybox=True,
    shadow=True,
    markerscale=3,
)
axis.hlines(
    y=newton_tol, xmin=0, xmax=T_end, linewidth=3, color="black", linestyles="dashed"
)
axis.text(T_end, newton_tol, "tol")
# endregion

# region broken axis residual

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(2 * 8., 8.))
# fig.subplots_adjust(hspace=0.05)  # adjust space between Axes
# ax2.set_xlabel("Simulation time")
# ax1.set_ylabel("l2-norm residual")
# ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax1.set_yscale('log')
# ax2.set_yscale('log')

# # plot the same data on both Axes
# img_1 = ax1.plot(
#     tx, np.array(res_bf_avg), linestyle="solid", color="blue", linewidth=3
# )
# img_1f = ax1.fill_between(
#     tx, np.array(res_bf_max), np.array(res_bf_min), alpha=0.2, color='blue'
# )
# img_2 = ax1.plot(
#     tx, np.array(res_af_avg), linestyle="solid", color="red", linewidth=3
# )
# img_2f = ax1.fill_between(
#     tx, np.array(res_af_max), np.array(res_af_min), alpha=0.2, color='red'
# )
# img_3 = ax1.plot(
#     tx, np.array(res_aa_avg), linestyle="solid", color="black", linewidth=3
# )
# img_3f = ax1.fill_between(
#     tx, np.array(res_aa_max), np.array(res_aa_min), alpha=0.2, color='black', facecolor='grey', hatch='/'
# )
# ax1.legend(
#     img_1 + [img_1f] + img_2 + [img_2f] + img_3 + [img_3f],
#     ['avg before flash', 'range bf', 'avg after flash', 'range af', 'avg at assembly', 'range aa'],
#     loc='upper center',
#     bbox_to_anchor=(0.5, 1.1),
#     ncol=3,
#     fancybox=True,
#     shadow=True,
#     markerscale=3
# )

# img_1 = ax2.plot(
#     tx, np.array(res_bf_avg), linestyle="solid", color="blue", linewidth=3
# )
# img_1f = ax2.fill_between(
#     tx, np.array(res_bf_max), np.array(res_bf_min), alpha=0.2, color='blue'
# )
# img_2 = ax2.plot(
#     tx, np.array(res_af_avg), linestyle="solid", color="red", linewidth=3
# )
# img_2f = ax2.fill_between(
#     tx, np.array(res_af_max), np.array(res_af_min), alpha=0.2, color='red'
# )
# img_3 = ax2.plot(
#     tx, np.array(res_aa_avg), linestyle="solid", color="black", linewidth=3
# )
# img_3f = ax2.fill_between(
#     tx, np.array(res_aa_max), np.array(res_aa_min), alpha=0.2, color='black', facecolor='grey', hatch='/'
# )

# # zoom-in / limit the view to different portions of the data
# ax1.set_ylim((10e2, 40e6))
# ax2.set_ylim((newton_tol, 10e-4))

# # hide the spines between ax and ax2
# ax1.spines.bottom.set_visible(False)
# ax2.spines.top.set_visible(False)
# ax1.xaxis.tick_top()
# ax1.tick_params(labeltop=False)  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()

# d = .5  # proportion of vertical to horizontal extent of the slanted line
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
#               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
# ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
# ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# endregion

fig.tight_layout(pad=0.05, w_pad=1)
fig.savefig(
    str(pathlib.Path("./visualization/residuals.png").resolve()),
    format="png",
    dpi=400,
)
