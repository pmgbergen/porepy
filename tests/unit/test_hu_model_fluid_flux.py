import numpy as np
import scipy as sp
import porepy as pp

from typing import Callable, Optional, Type
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays

import test_hu_model
from test_hu_model import FunctionRhoG, FunctionRhoV, FunctionTotalFlux


import os
import sys
import pdb

os.system("clear")

"""
OLD TEST

- I check that the interface (noun "interface" missing everywhere) fluid flux of the non-present phase is actually zero
- test after one, or more, timesteps. Density is constant. It shouldn't conferge but id does, sometimes..?
"""


class SolutionStrategyPressureMassTest(test_hu_model.SolutionStrategyPressureMass):
    def initial_condition(self) -> None:
        """ """
        val = np.zeros(self.equation_system.num_dofs())
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                val,
                time_step_index=time_step_index,
            )

        for iterate_index in self.iterate_indices:
            self.equation_system.set_variable_values(val, iterate_index=iterate_index)

        # from email:
        for sd in self.mdg.subdomains():
            saturation_variable = (
                self.mixture.mixture_for_subdomain(self.equation_system, sd)
                .get_phase(0)
                .saturation_operator([sd])
            )

            if sd.dim == 2:
                saturation_values = self.saturation_values_2d
            else:
                saturation_values = self.saturation_values_1d

            self.equation_system.set_variable_values(
                saturation_values,
                variables=[saturation_variable],
                time_step_index=0,
                iterate_index=0,
            )

            pressure_variable = self.pressure([sd])

            if sd.dim == 2:
                pressure_values = np.ones(sd.num_cells)
            else:  # sd.dim == 1
                pressure_values = np.zeros(sd.num_cells)

            self.equation_system.set_variable_values(
                pressure_values,
                variables=[pressure_variable],
                time_step_index=0,
                iterate_index=0,
            )

        for sd, data in self.mdg.subdomains(return_data=True):
            # aperture = np.ones(sd.num_cells) * np.power(
            #     7e-2, self.mdg.dim_max() - sd.dim
            # )

            pp.initialize_data(
                sd,
                data,
                self.ppu_keyword,
                {
                    "darcy_flux_phase_0": np.zeros(sd.num_faces),
                    "darcy_flux_phase_1": np.zeros(sd.num_faces),
                    # "aperture": aperture,
                },
            )
        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                intf,
                data,
                self.ppu_keyword,
                {
                    "interface_mortar_flux_phase_0": np.zeros(intf.num_cells),
                    "interface_mortar_flux_phase_1": np.zeros(intf.num_cells),
                },
            )

        for sd in self.mdg.subdomains():
            gigi = (
                self.mixture.mixture_for_subdomain(self.equation_system, sd)
                .get_phase(0)
                .saturation
            )
            print("saturation = ", gigi.val)
            pp.plot_grid(
                sd,
                gigi.val,
                alpha=0.5,
                info="c",
                title="saturation " + str(self.ell),
            )

    def eb_after_timestep(self):
        """ """
        # TEST SECTION: ---------------------------------------------------------

        # # open the fracture:
        # for sd in self.mdg.subdomains():
        #     if sd.dim == 2:
        #         internal_nodes_id = sd.get_internal_nodes()
        #         sd.nodes[1][[8, 10]] += 0.1
        #         sd.nodes[1][[9, 11]] -= 0.1

        # self.mdg.compute_geometry()

        pp.plot_grid(self.mdg, info="cf", alpha=0)

        for sd in self.mdg.subdomains():  # useless, but may be useful, so keep it
            if sd.dim == 2:
                gigi = (
                    self.mixture.mixture_for_subdomain(self.equation_system, sd)
                    .get_phase(0)
                    .saturation
                )
                print("saturation = ", gigi.val)
                pp.plot_grid(
                    sd,
                    gigi.val,
                    alpha=0.5,
                    info="c",
                    title="saturation " + str(self.ell),
                )

        (
            _,
            _,
            _,
            flux_intf_phase_0_p,
            flux_intf_phase_1_p,
            _,
            source_phase_0,
            source_phase_1,
            _,
        ) = self.eq_fcn_pressure(self.mdg.subdomains())

        flux_intf_phase_0_p = flux_intf_phase_0_p.evaluate(self.equation_system).val
        flux_intf_phase_1_p = flux_intf_phase_1_p.evaluate(self.equation_system).val

        (
            _,
            _,
            _,
            flux_intf_phase_0_m,
            flux_intf_phase_1_m,
            source_phase_0,
            source_phase_1,
        ) = self.eq_fcn_mass(self.mdg.subdomains())

        flux_intf_phase_0_m = flux_intf_phase_0_m.evaluate(self.equation_system).val
        flux_intf_phase_1_m = flux_intf_phase_1_m.evaluate(self.equation_system).val

        if self.case == 1:
            print("case 1")
            print("\nflux_intf_phase_0_p = ", flux_intf_phase_0_p)
            print("\nflux_intf_phase_0_m = ", flux_intf_phase_0_m)

            pdb.set_trace()

            # saturation phase 0 = 0
            assert np.any(
                np.isclose(flux_intf_phase_0_p, 0 * flux_intf_phase_0_p)
            )  # these are fluxes of phase 1, whose saturation is 0 everywhere

            assert np.any(np.isclose(flux_intf_phase_0_m, 0 * flux_intf_phase_0_m))

        if self.case == 2:
            print("case 2")
            print("\nflux_intf_phase_1_p = ", flux_intf_phase_1_p)
            print("\nflux_intf_phase_1_m = ", flux_intf_phase_1_m)

            pdb.set_trace()

            # saturation phase 1 = 0
            assert np.any(
                np.isclose(flux_intf_phase_1_p, 0 * flux_intf_phase_1_p)
            )  # these are fluxes of phase 1, whose saturation is 0 everywhere

            assert np.any(np.isclose(flux_intf_phase_1_m, 0 * flux_intf_phase_1_m))

        print("\n\n TEST PASSED ------------------- ")
        pdb.set_trace()


class MyModelGeometryTest(pp.ModelGeometry):
    def set_domain(self) -> None:
        """ """
        self.size = 1.0 / self.units.m
        box = {"xmin": 0, "xmax": 5, "ymin": 0, "ymax": 2 * self.size}
        self._domain = pp.Domain(box)

    def set_fractures(self) -> None:
        """ """
        frac1 = pp.LineFracture(np.array([[1, 4], [1, 1]]))
        self._fractures: list = [frac1]

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {"cell_size": 0.99}
        return self.params.get("meshing_arguments", default_meshing_args)


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


class PartialFinalModelTest(
    test_hu_model.PrimaryVariables,
    test_hu_model.Equations,
    test_hu_model.ConstitutiveLawPressureMass,
    test_hu_model.BoundaryConditionsPressureMass,
    SolutionStrategyPressureMassTest,
    MyModelGeometryTest,
    pp.DataSavingMixin,
):
    """ """


class FinalModel(PartialFinalModelTest):  # I'm sorry...
    def __init__(self, mixture, params: Optional[dict] = None):
        super().__init__(params)
        self.mixture = mixture
        self.ell = 0  # 0 = wetting, 1 = non-wetting
        self.gravity_value = 1  # pp.GRAVITY_ACCELERATION
        self.dynamic_viscosity = 1  # TODO: it is hardoced everywhere, you know...

        self.case = None
        self.saturation_values_2d = None
        self.saturation_values_1d = None


fluid_constants = pp.FluidConstants({})
solid_constants = pp.SolidConstants(
    {
        "porosity": 0.25,
        "permeability": 1,
        "normal_permeability": 1e3,
        "residual_aperture": 0.1,
    }
)

material_constants = {"fluid": fluid_constants, "solid": solid_constants}
time_manager = pp.TimeManager(
    schedule=[0, 100],
    dt_init=1e0,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

params = {
    "material_constants": material_constants,
    "max_iterations": 100,
    "nl_convergence_tol": 1e-10,
    "nl_divergence_tol": 1e5,
    "time_manager": time_manager,
}

# wetting_phase = ConstantDensityPhase(rho0=1)
# non_wetting_phase = ConstantDensityPhase(rho0=0.5)
wetting_phase = pp.composite.phase.Phase(rho0=1)
non_wetting_phase = pp.composite.phase.Phase(rho0=0.5)

mixture = pp.Mixture()
mixture.add([wetting_phase, non_wetting_phase])

model = FinalModel(mixture, params)

case = 1
model.case = case

if case == 1:
    model.saturation_values_2d = 0 * np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    model.saturation_values_1d = 0 * np.array([1.0, 1, 1])
if case == 2:
    model.saturation_values_2d = 1 * np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    model.saturation_values_1d = 1 * np.array([1.0, 1, 1])

pp.run_time_dependent_model(model, params)
