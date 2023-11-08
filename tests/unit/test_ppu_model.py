import numpy as np
import scipy as sp
import porepy as pp

from typing import Callable, Optional, Type, Literal, Sequence, Union
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays

import test_hu_model

import os
import sys
import pdb

os.system("clear")


def myprint(var):
    print("\n" + var + " = ", eval(var))


"""
- copied from test_hu_model.py
"""


class EquationsPPU(test_hu_model.Equations):
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
        """ """
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
                * self.vector_source(subdomains, material="fluid")
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
            @ (
                rho * self.vector_source(subdomains, material="fluid")
            )  # NOTE: mass_density not upwinded
        )

        flux.set_name("darcy_flux_phase_1")

        return flux


class SolutionStrategyPressureMassPPU(test_hu_model.SolutionStrategyPressureMass):
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
            )  # TODO: hardcoded for 2D

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

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """ """

        self._nonlinear_iteration += 1
        self.equation_system.shift_iterate_values()

        self.equation_system.set_variable_values(
            values=solution_vector, additive=True, iterate_index=0
        )

        # TEST HERE:

        # subdomains = self.mdg.subdomains()

        # discr_phase_0 = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        # discr_phase_1 = self.ppu_discretization(subdomains, "darcy_flux_phase_1")

        # up_matrix_0 = discr_phase_0.upwind.evaluate(self.equation_system)
        # up_matrix_1 = discr_phase_1.upwind.evaluate(self.equation_system)

        # print("\nup_matrix_0 = ", up_matrix_0)
        # print("up_matrix_1 = ", up_matrix_1)

        # pdb.set_trace()


class PartialFinalModel(
    test_hu_model.PrimaryVariables,
    EquationsPPU,
    test_hu_model.ConstitutiveLawPressureMass,
    test_hu_model.BoundaryConditionsPressureMass,
    SolutionStrategyPressureMassPPU,
    test_hu_model.MyModelGeometry,
    pp.DataSavingMixin,
):
    """ """


if __name__ == "__main__":
    mixture = test_hu_model.mixture

    class FinalModel(PartialFinalModel):
        def __init__(self, params: Optional[dict] = None):
            super().__init__(params)
            self.mixture = mixture
            self.ell = 0
            self.gravity_value = 1
            self.dynamic_viscosity = 1

            self.xmin = 0
            self.xmax = 1 * 1
            self.ymin = 0
            self.ymax = 1 * 1

            self.relative_permeability = (
                pp.tobedefined.relative_permeability.rel_perm_quadratic
            )
            self.mobility = pp.tobedefined.mobility.mobility(self.relative_permeability)
            self.mobility_operator = pp.tobedefined.mobility.mobility_operator(
                self.mobility
            )

    params = test_hu_model.params
    model = FinalModel(params)

    pp.run_time_dependent_model(model, params)
