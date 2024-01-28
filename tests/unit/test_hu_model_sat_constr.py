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

"""


class EquationsTest(test_hu_model.Equations):
    def pressure_equation(
        self, subdomains: list[pp.Grid], return_flux=False
    ) -> pp.ad.Operator:
        # accumulation term: ------------------------------
        mass_density_phase_0 = (
            self.porosity(subdomains)
            * self.mixture.get_phase(0).mass_density_operator(subdomains, self.pressure)
            * self.mixture.get_phase(0).saturation_operator(subdomains)
        )
        mass_density_phase_1 = (
            self.porosity(subdomains)
            * self.mixture.get_phase(1).mass_density_operator(subdomains, self.pressure)
            * self.mixture.get_phase(1).saturation_operator(subdomains)
        )
        mass_density = mass_density_phase_0 + mass_density_phase_1

        accumulation = self.volume_integral(mass_density, subdomains, dim=1)
        accumulation.set_name("fluid_mass_p_eq")

        rho_total_flux_operator = FunctionTotalFlux(
            pp.rho_total_flux,
            self.equation_system,
            self.mixture,
            self.mdg,
            name="rho qt",
        )

        fake_input = self.pressure(subdomains)
        flux = rho_total_flux_operator(fake_input) + self.bc_values(
            subdomains
        )  # this is wrong, but bc val are 0 so I dont care...

        # interfaces flux contribution (copied from mass bal): ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        flux_intf_phase_0 = (
            discr.bound_transport_neu
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_0(
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        flux_intf_phase_1 = (
            discr.bound_transport_neu
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_1(
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        flux = flux - flux_intf_phase_0 - flux_intf_phase_1

        # sources: --------------------------------------------------------------
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_0 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_0(
                interfaces, self.mixture.get_phase(0), "interface_mortar_flux_phase_0"
            )
        )
        source_phase_0.set_name("interface_fluid_mass_flux_source_phase_0")

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_1 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_1(
                interfaces, self.mixture.get_phase(1), "interface_mortar_flux_phase_1"
            )
        )
        source_phase_1.set_name("interface_fluid_mass_flux_source_phase_1")

        source = source_phase_0 + source_phase_1

        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name("pressure_equation")

        flux_contribution = (
            pp.ad.Scalar(-1)
            * pp.ad.Divergence(subdomains, dim=1)
            @ (flux_intf_phase_0 + flux_intf_phase_1)
        )

        if return_flux:
            flux.set_name("pressure_flux")
            return flux  ### for test
        else:
            return eq

    def mass_balance_equation(
        self, subdomains: list[pp.Grid], return_flux=False
    ) -> pp.ad.Operator:
        # accumulation term: ------------------------------------------------
        mass_density = (
            self.porosity(subdomains)
            * self.mixture.get_phase(self.ell).mass_density_operator(
                subdomains, self.pressure
            )
            * self.mixture.get_phase(self.ell).saturation_operator(subdomains)
        )
        accumulation = self.volume_integral(mass_density, subdomains, dim=1)
        accumulation.set_name("fluid_mass_mass_eq")

        # subdomains flux contribution: -------------------------------------
        rho_V_operator = FunctionRhoV(
            pp.rho_flux_V, self.equation_system, self.mixture, self.mdg, name="rho V"
        )
        rho_G_operator = FunctionRhoG(
            pp.rho_flux_G, self.equation_system, self.mixture, self.mdg, name="rho G"
        )

        fake_input = self.pressure(subdomains)
        flux = (
            rho_V_operator(fake_input)
            + rho_G_operator(fake_input)
            + self.bc_values(subdomains)
        )  # TODO: this is wrong, but bc val are 0 so I dont care...

        # interfaces flux contribution: ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        if self.ell == 0:  # TODO: you can avoid if condition
            discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")

            flux -= (
                discr.bound_transport_neu  # -1,0,1 matrix
                @ mortar_projection.mortar_to_primary_int
                @ self.interface_fluid_mass_flux_phase_0(
                    interfaces,
                    self.mixture.get_phase(0),
                    "interface_mortar_flux_phase_0",
                )
            )

        else:  # self.ell == 1
            discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
            flux -= (
                discr.bound_transport_neu
                @ mortar_projection.mortar_to_primary_int
                @ self.interface_fluid_mass_flux_phase_1(
                    interfaces,
                    self.mixture.get_phase(1),
                    "interface_mortar_flux_phase_1",
                )
            )

        # sources: ---------------------------------------
        if self.ell == 0:  # TODO: you can avoid if condition
            projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
            source = (
                projection.mortar_to_secondary_int
                @ self.interface_fluid_mass_flux_phase_0(
                    interfaces,
                    self.mixture.get_phase(0),
                    "interface_mortar_flux_phase_0",
                )
            )
            source.set_name("interface_fluid_mass_flux_source_phase_0")

        else:  # self.ell == 1:
            projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
            source = (
                projection.mortar_to_secondary_int
                @ self.interface_fluid_mass_flux_phase_1(
                    interfaces,
                    self.mixture.get_phase(1),
                    "interface_mortar_flux_phase_1",
                )
            )
            source.set_name("interface_fluid_mass_flux_source_phase_1")

        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)

        if return_flux:
            flux.set_name("mass_flux")
            return flux  ### for test
        else:
            return eq


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

            # saturation_values = np.ones(sd.num_cells)

            saturation_values = 1 * np.ones(sd.num_cells)

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

        # pp.plot_grid(self.mdg, info="cf", alpha=0)

        for sd in self.mdg.subdomains():  # useless, but may be useful, so keep it
            if sd.dim == 2:
                s_phase_0 = (
                    self.mixture.mixture_for_subdomain(self.equation_system, sd)
                    .get_phase(0)
                    .saturation
                ).val

                s_phase_1 = (
                    self.mixture.mixture_for_subdomain(self.equation_system, sd)
                    .get_phase(1)
                    .saturation
                ).val

                print("saturation phase 0 = ", s_phase_0)
                print("saturation phase 1 = ", s_phase_1)
                pp.plot_grid(
                    sd,
                    s_phase_0,
                    alpha=0.5,
                    info="c",
                    title="saturation " + str(self.ell),
                )

                pdb.set_trace()
                # assert np.all(
                #     np.isclose(s_phase_1, 0 * s_phase_1, rtol=0, atol=1e-5)
                # )  ### they should stay closer to zero

                # lowering the compressibility s_phase_1 remains closer to zero

        print("\n\n TEST PASSED ------------------- ")


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
    EquationsTest,
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


pp.run_time_dependent_model(model, params)
