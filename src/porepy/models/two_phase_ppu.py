import numpy as np
import scipy as sp
import porepy as pp

from typing import Callable, Optional, Type, Literal, Sequence, Union
from types import NoneType
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays

import porepy.models.two_phase_hu as two_phase_hu

import os
import sys
import pdb

os.system("clear")


def myprint(var):
    print("\n" + var + " = ", eval(var))


"""
"""


class EquationsPPU(two_phase_hu.Equations):
    """I dont see the point of splitting this class more than this, I prefer commented titles instead of mixin classes"""

    # PRESSURE EQUATION: -------------------------------------------------------------------------------------------------

    def eq_fcn_pressure(self, subdomains):  # I suck in python. I need this for tests.
        # accumulation term: ------------------------------
        mass_density_phase_0 = self.mixture.get_phase(0).mass_density_operator(
            subdomains, self.pressure
        ) * self.mixture.get_phase(0).saturation_operator(subdomains)

        mass_density_phase_1 = self.mixture.get_phase(1).mass_density_operator(
            subdomains, self.pressure
        ) * self.mixture.get_phase(1).saturation_operator(subdomains)

        mass_density = self.porosity(subdomains) * (
            mass_density_phase_0 + mass_density_phase_1
        )

        accumulation = self.volume_integral(mass_density, subdomains, dim=1)
        accumulation.set_name("fluid_mass_p_eq")

        # subdomains flux: -----------------------------------------------------------
        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        mob_rho_phase_0 = self.mixture.get_phase(0).mass_density_operator(
            subdomains, self.pressure
        ) * self.mobility_operator(
            subdomains,
            self.mixture.get_phase(0).saturation_operator,
            self.dynamic_viscosity,
        )

        darcy_flux_phase_0 = self.darcy_flux_phase_0(
            subdomains, self.mixture.get_phase(0)
        )
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        flux_phase_0: pp.ad.Operator = darcy_flux_phase_0 * (
            discr.upwind @ mob_rho_phase_0
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        mob_rho_phase_1 = self.mixture.get_phase(1).mass_density_operator(
            subdomains, self.pressure
        ) * self.mobility_operator(
            subdomains,
            self.mixture.get_phase(1).saturation_operator,
            self.dynamic_viscosity,
        )

        darcy_flux_phase_1 = self.darcy_flux_phase_1(
            subdomains, self.mixture.get_phase(1)
        )
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        flux_phase_1: pp.ad.Operator = darcy_flux_phase_1 * (
            discr.upwind @ mob_rho_phase_1
        )

        flux_tot = flux_phase_0 + flux_phase_1

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
                self.interface_mortar_flux_phase_0(interfaces),
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
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        discr_phase_0 = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        discr_phase_1 = self.ppu_discretization(subdomains, "darcy_flux_phase_1")

        flux = (
            flux_tot
            - flux_intf_phase_0
            - flux_intf_phase_1
            - discr_phase_0.bound_transport_neu @ self.bc_neu_phase_0(subdomains)
            - discr_phase_1.bound_transport_neu @ self.bc_neu_phase_1(subdomains)
        )

        # sources: --------------------------------------------------------------
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_0 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )
        source_phase_0.set_name("interface_fluid_mass_flux_source_phase_0")

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_1 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )
        source_phase_1.set_name("interface_fluid_mass_flux_source_phase_1")

        source = source_phase_0 + source_phase_1

        eq = self.balance_equation_variable_dt(
            subdomains, accumulation, flux, source, dim=1
        )  # * pp.ad.Scalar(1e6)
        eq.set_name("pressure_equation")

        return (
            eq,
            accumulation,
            flux_tot,
            flux_intf_phase_0,
            flux_intf_phase_1,
            flux,
            source_phase_0,
            source_phase_1,
            source,
        )

    def pressure_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        eq, _, _, _, _, _, _, _, _ = self.eq_fcn_pressure(subdomains)
        return eq

    # MASS BALANCE: ----------------------------------------------------------------------------------------------------------------

    def eq_fcn_mass(self, subdomains):  # I suck in python. I need this for tests.
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
        if self.ell == 0:
            discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
            mob_rho = self.mixture.get_phase(0).mass_density_operator(
                subdomains, self.pressure
            ) * self.mobility_operator(
                subdomains,
                self.mixture.get_phase(0).saturation_operator,
                self.dynamic_viscosity,
            )

            darcy_flux_phase_0 = self.darcy_flux_phase_0(
                subdomains, self.mixture.get_phase(0)
            )
            interfaces = self.subdomains_to_interfaces(subdomains, [1])
            mortar_projection = pp.ad.MortarProjections(
                self.mdg, subdomains, interfaces, dim=1
            )

            flux_phase_0: pp.ad.Operator = darcy_flux_phase_0 * (discr.upwind @ mob_rho)

            flux_no_intf = flux_phase_0  # just to keep the structure of hu code

        else:  # self.ell == 1
            discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
            mob_rho = self.mixture.get_phase(1).mass_density_operator(
                subdomains, self.pressure
            ) * self.mobility_operator(
                subdomains,
                self.mixture.get_phase(1).saturation_operator,
                self.dynamic_viscosity,
            )

            darcy_flux_phase_1 = self.darcy_flux_phase_1(
                subdomains, self.mixture.get_phase(1)
            )
            interfaces = self.subdomains_to_interfaces(subdomains, [1])
            mortar_projection = pp.ad.MortarProjections(
                self.mdg, subdomains, interfaces, dim=1
            )

            flux_phase_1: pp.ad.Operator = darcy_flux_phase_1 * (discr.upwind @ mob_rho)

            flux_no_intf = flux_phase_1

        # interfaces flux contribution: ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        flux_intf_phase_0 = (
            discr.bound_transport_neu  # -1,0,1 matrix
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )  # sorry, I need to test it

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        flux_intf_phase_1 = (
            discr.bound_transport_neu
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        discr_phase_0 = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        discr_phase_1 = self.ppu_discretization(subdomains, "darcy_flux_phase_1")

        if self.ell == 0:
            flux = (
                flux_phase_0
                - flux_intf_phase_0
                - discr_phase_0.bound_transport_neu @ self.bc_neu_phase_0(subdomains)
            )

        else:  # self.ell == 1
            flux = (
                flux_phase_1
                - flux_intf_phase_1
                - discr_phase_1.bound_transport_neu @ self.bc_neu_phase_1(subdomains)
            )

        flux.set_name("ppu_flux")

        # sources: ---------------------------------------
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_0 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )  # sorry, I need to output everything for the tests

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_1 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )  # sorry, I need to output everything for the tests

        if self.ell == 0:
            source = source_phase_0  # I need this for the tests
            source.set_name("interface_fluid_mass_flux_source_phase_0")

        else:  # self.ell == 1:
            source = source_phase_1
            source.set_name("interface_fluid_mass_flux_source_phase_1")

        eq = self.balance_equation_variable_dt(
            subdomains, accumulation, flux, source, dim=1
        )  # * pp.ad.Scalar(1e6)
        eq.set_name("mass_balance_equation")

        return (
            eq,
            accumulation,
            None,
            None,
            flux_no_intf,
            flux_intf_phase_0,
            flux_intf_phase_1,
            source_phase_0,
            source_phase_1,
        )

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        eq, _, _, _, _, _, _, _, _ = self.eq_fcn_mass(subdomains)

        return eq

    def ppu_discretization(
        self, subdomains: list[pp.Grid], flux_array_key
    ) -> pp.ad.UpwindAd:
        """
        flux_array_key =  either darcy_flux_phase_0 or darcy_flux_phase_1
        """
        return pp.ad.UpwindAd(
            self.ppu_keyword + "_" + flux_array_key, subdomains, flux_array_key
        )

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """ """
        return pp.ad.MpfaAd(self.darcy_keyword, subdomains)

    def darcy_flux_phase_0(self, subdomains: list[pp.Grid], phase) -> pp.ad.Operator:
        """
        - TODO: why the heck is there a phase as input
        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains
        )

        mass_density = phase.mass_density_operator(subdomains, self.pressure)

        rho = (
            self.vector_expansion(subdomains) @ mass_density
        )  # there are a lot of extra computations in vector_expansion

        flux: pp.ad.Operator = (
            discr.flux @ self.pressure(subdomains)
            + discr.bound_flux
            @ (
                self.bc_values_darcy(subdomains)
                + projection.mortar_to_primary_int
                @ self.interface_mortar_flux_phase_0(interfaces)
            )
            - discr.vector_source
            @ (
                rho  ### VECTOR source, not scalar source
                * self.vector_source(subdomains)
            )  # NOTE: mass_density not upwinded
        )

        flux.set_name("darcy_flux_phase_0")

        return flux

    def darcy_flux_phase_1(self, subdomains: list[pp.Grid], phase) -> pp.ad.Operator:
        """ """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains
        )

        mass_density = phase.mass_density_operator(subdomains, self.pressure)
        rho = self.vector_expansion(subdomains) @ mass_density

        flux: pp.ad.Operator = (
            discr.flux @ self.pressure(subdomains)
            + discr.bound_flux
            @ (
                self.bc_values_darcy(subdomains)
                + projection.mortar_to_primary_int
                @ self.interface_mortar_flux_phase_1(interfaces)
            )
            - discr.vector_source
            @ (rho * self.vector_source(subdomains))  # NOTE: mass_density not upwinded
        )

        flux.set_name("darcy_flux_phase_1")

        return flux


class SolutionStrategyPressureMassPPU(two_phase_hu.SolutionStrategyPressureMass):
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

        for sd in self.mdg.subdomains():
            saturation_variable = (
                self.mixture.mixture_for_subdomain(sd)
                .get_phase(self.ell)
                .saturation_operator([sd])
            )

            saturation_values = 0.0 * np.ones(sd.num_cells)
            saturation_values[
                np.where(sd.cell_centers[1] >= self.ymax / 2)
            ] = 1.0  # TODO: HARDCODED for 2D

            if sd.dim == 1:
                saturation_values = 0.8 * np.ones(sd.num_cells)

            self.equation_system.set_variable_values(
                saturation_values,
                variables=[saturation_variable],
                time_step_index=0,
                iterate_index=0,
            )

            pressure_variable = self.pressure([sd])

            pressure_values = (
                2 - 1 * sd.cell_centers[1] / self.ymax
            ) / self.p_0  # TODO: hardcoded for 2D

            self.equation_system.set_variable_values(
                pressure_values,
                variables=[pressure_variable],
                time_step_index=0,
                iterate_index=0,
            )

        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.ppu_keyword + "_" + "darcy_flux_phase_0",
                {
                    "darcy_flux_phase_0": np.zeros(sd.num_faces),
                },
            )

            pp.initialize_data(
                sd,
                data,
                self.ppu_keyword + "_" + "darcy_flux_phase_1",
                {
                    "darcy_flux_phase_1": np.zeros(sd.num_faces),
                },
            )

        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                intf,
                data,
                self.ppu_keyword + "_" + "interface_mortar_flux_phase_0",
                {
                    "interface_mortar_flux_phase_0": np.zeros(intf.num_cells),
                },
            )

            pp.initialize_data(
                intf,
                data,
                self.ppu_keyword + "_" + "interface_mortar_flux_phase_1",
                {
                    "interface_mortar_flux_phase_1": np.zeros(intf.num_cells),
                },
            )

    def set_discretization_parameters(self) -> None:
        """ """

        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.darcy_keyword,
                {
                    "bc": self.bc_type_darcy(sd),
                    "second_order_tensor": self.intrinsic_permeability_tensor(sd),
                    "ambient_dimension": self.nd,
                },
            )

            pp.initialize_data(
                sd,
                data,
                self.ppu_keyword + "_" + "darcy_flux_phase_0",
                {
                    "bc": self.bc_type_darcy(sd),
                },
            )

            pp.initialize_data(
                sd,
                data,
                self.ppu_keyword + "_" + "darcy_flux_phase_1",
                {
                    "bc": self.bc_type_darcy(sd),
                },
            )

        for intf, intf_data in self.mdg.interfaces(return_data=True, codim=1):
            pp.initialize_data(
                intf,
                intf_data,
                self.darcy_keyword,
                {
                    "ambient_dimension": self.nd,
                },
            )

    def before_nonlinear_iteration(self):
        """ """

        for sd, data in self.mdg.subdomains(return_data=True):
            vals_0 = (
                self.darcy_flux_phase_0([sd], self.mixture.get_phase(0))
                .evaluate(self.equation_system)
                .val
            )
            data[pp.PARAMETERS][self.ppu_keyword + "_" + "darcy_flux_phase_0"].update(
                {"darcy_flux_phase_0": vals_0}
            )

            vals_1 = (
                self.darcy_flux_phase_1([sd], self.mixture.get_phase(1))
                .evaluate(self.equation_system)
                .val
            )
            data[pp.PARAMETERS][self.ppu_keyword + "_" + "darcy_flux_phase_1"].update(
                {"darcy_flux_phase_1": vals_1}
            )

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            vals = (
                self.interface_mortar_flux_phase_0([intf])
                .evaluate(self.equation_system)
                .val
            )
            data[pp.PARAMETERS][
                self.ppu_keyword + "_" + "interface_mortar_flux_phase_0"
            ].update({"interface_mortar_flux_phase_0": vals})

            vals = (
                self.interface_mortar_flux_phase_1([intf])
                .evaluate(self.equation_system)
                .val
            )
            data[pp.PARAMETERS][
                self.ppu_keyword + "_" + "interface_mortar_flux_phase_1"
            ].update({"interface_mortar_flux_phase_1": vals})

        self.set_discretization_parameters()
        self.rediscretize()

    def flip_flop(self):
        """ """
        for sd in self.mdg.subdomains():
            if sd.dim == 2:
                darcy_phase_0 = (
                    self.darcy_flux_phase_0([sd], self.mixture.get_phase(0))
                    .evaluate(self.equation_system)
                    .val
                )
                darcy_phase_1 = (
                    self.darcy_flux_phase_1([sd], self.mixture.get_phase(1))
                    .evaluate(self.equation_system)
                    .val
                )

                sign_darcy_phase_0 = np.sign(darcy_phase_0)
                sign_darcy_phase_1 = np.sign(darcy_phase_1)

                if type(self.sign_darcy_phase_0_prev) == NoneType:
                    self.sign_darcy_phase_0_prev = sign_darcy_phase_0

                if type(self.sign_darcy_phase_1_prev) == NoneType:
                    self.sign_darcy_phase_1_prev = sign_darcy_phase_1

                number_flips_darcy_phase_0 = np.sum(
                    np.not_equal(self.sign_darcy_phase_0_prev, sign_darcy_phase_0)
                )

                number_flips_darcy_phase_1 = np.sum(
                    np.not_equal(self.sign_darcy_phase_1_prev, sign_darcy_phase_1)
                )

        return np.array([sign_darcy_phase_0, sign_darcy_phase_1]), np.array(
            [number_flips_darcy_phase_0, number_flips_darcy_phase_1]
        )

    def eb_after_timestep(self):
        """used for plots, tests, and more"""

        self.compute_mass()


class PartialFinalModel(
    two_phase_hu.PrimaryVariables,
    EquationsPPU,
    two_phase_hu.ConstitutiveLawPressureMass,
    two_phase_hu.BoundaryConditionsPressureMass,
    SolutionStrategyPressureMassPPU,
    two_phase_hu.MyModelGeometry,
    pp.DataSavingMixin,
):
    """ """


if __name__ == "__main__":
    # scaling:
    # very bad logic, improve it...
    L_0 = 1
    gravity_0 = 1
    dynamic_viscosity_0 = 1
    rho_0 = 1  # |rho_phase_0-rho_phase_1|
    p_0 = 1
    Ka_0 = 1
    u_0 = Ka_0 * p_0 / (dynamic_viscosity_0 * L_0)
    t_0 = L_0 / u_0

    gravity_number = Ka_0 * rho_0 * gravity_0 / (dynamic_viscosity_0 * u_0)

    print("\nSCALING: ======================================")
    print("u_0 = ", u_0)
    print("t_0 = ", u_0)
    print("gravity_number = ", gravity_number)
    print(
        "pay attention: gravity number is not influenced by Ka_0 and dynamic_viscosity_0"
    )
    print("=========================================\n")

    fluid_constants = pp.FluidConstants({})
    solid_constants = pp.SolidConstants(
        {
            "porosity": 0.25,
            "intrinsic_permeability": 1 / Ka_0,
            "normal_permeability": 0.1 / Ka_0,  # this does NOT include the aperture
            "residual_aperture": 0.1 / Ka_0,
        }
    )

    material_constants = {"fluid": fluid_constants, "solid": solid_constants}

    time_manager = two_phase_hu.TimeManagerPP(
        schedule=np.array([0, 50]) / t_0,
        dt_init=1e0 / t_0,
        dt_min_max=np.array([1e-5, 1e0]) / t_0,
        constant_dt=False,
        iter_max=20,
        print_info=True,
    )

    params = {
        "material_constants": material_constants,
        "max_iterations": 10,
        "nl_convergence_tol": 1e-6,
        "nl_divergence_tol": 1e5,
        "time_manager": time_manager,
    }

    wetting_phase = pp.composite.phase.Phase(
        rho0=1 / rho_0, p0=p_0, beta=1e-10
    )  # TODO: improve that p0, different menaing wrt rho0
    non_wetting_phase = pp.composite.phase.Phase(rho0=0.5 / rho_0, p0=p_0, beta=1e-10)

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    class FinalModel(PartialFinalModel):
        def __init__(self, params: Optional[dict] = None):
            super().__init__(params)

            # scaling values: (not all of them are actually used inside model)
            self.L_0 = L_0
            self.gravity_0 = gravity_0
            self.dynamic_viscosity_0 = dynamic_viscosity_0
            self.rho_0 = rho_0
            self.p_0 = p_0
            self.Ka_0 = Ka_0
            self.t_0 = t_0

            self.mixture = mixture
            self.ell = 0
            self.gravity_value = 1 / self.gravity_0
            self.dynamic_viscosity = 1 / self.dynamic_viscosity_0

            self.xmin = 0 / self.L_0
            self.xmax = 1 / self.L_0
            self.ymin = 0 / self.L_0
            self.ymax = 1 / self.L_0

            self.relative_permeability = (
                pp.tobedefined.relative_permeability.rel_perm_quadratic
            )
            self.mobility = pp.tobedefined.mobility.mobility(self.relative_permeability)
            self.mobility_operator = pp.tobedefined.mobility.mobility_operator(
                self.mobility
            )

            self.sign_darcy_phase_0_prev = None
            self.sign_darcy_phase_1_prev = None

            self.output_file_name = "./OUTPUT_NEWTON_INFO_PPU"
            self.mass_output_file_name = "./MASS_OVER_TIME_PPU"
            self.flips_file_name = "./FLIPS_PPU"

    model = FinalModel(params)

    pp.run_time_dependent_model(model, params)
