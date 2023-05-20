import numpy as np
import scipy as sp
import porepy as pp

import os
import sys
import pdb

os.system("clear")


def myprint(var):
    print("\n" + var + " = ", eval(var))


###################################################################################################################
# mass and energy model:
###################################################################################################################
fluid_constants = pp.FluidConstants({"compressibility": 114})
material_constants = {"fluid": fluid_constants}
params = {"material_constants": material_constants}

model = pp.mass_and_energy_balance.MassAndEnergyBalance(params)


pp.run_time_dependent_model(model, params)


print(model.fluid.constants)

print("\n\n\n\n")

for i in dir(model):
    print(i)

pdb.set_trace()


class FunctionTotalFlux(pp.ad.operator_functions.AbstractFunction):
    """ """

    def __init__(self, func: Callable, name: str, array_compatible: bool = True):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False

    def get_args_from_model(self):
        """to be overriden"""

        return

    def get_args_from_sd_data_dictionary(self, data):
        """ """

        mixture = 1  # TODO: you know...
        pressure = data["for_hu"]["pressure_AdArray"]  # SOMETHING SIMILAR FOR SATURAION
        gravity_value = (
            self.gravtiy_value
        )  # TODO: there is little chance it will work...

        left_restriction, right_restriction = data["for_hu"]["restriction_matrices"]
        transmissibility_internal_tpfa = data["for_hu"][
            "transmissibility_internal_tpfa"
        ]
        ad = True  # Sorry
        dynamic_viscosity = 1  # Sorry

        return [
            mixture,
            pressure,
            gravity_value,
            left_restriction,
            right_restriction,
            transmissibility_internal_tpfa,
            ad,
            dynamic_viscosity,
        ]

    def get_values(self, *args: AdArray) -> np.ndarray:
        """ """
        rhs_list = []
        for subdomain, data in self.mdg.subdomain(return_data=True):
            args = [subdomain]
            args_dict = self.get_args_from_sd_data_dictionary(data)
            args.append(args_dict)

            rhs_list.append(self.func(*args).val)

        return np.hstack(rhs_list)

    def get_jacobian(self, *args: AdArray) -> np.ndarray:
        """ """
        jac_list = []
        for subdomain, data in self.mdg.subdomain(return_data=True):
            args = [subdomain]
            args_dict = self.get_args_from_sd_data_dictionary(data)
            args.append(args_dict)

            jac_list.append(self.func(*args).jac)

        return sp.sparse.block_diag(jac_list)


class FunctionRhoV(pp.ad.operator_functions.AbstractFunction):
    """ """

    def __init__(self, func: Callable, name: str, array_compatible: bool = True):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False

    def get_args_from_model(self):
        """to be overriden"""

        return

    def get_args_from_sd_data_dictionary(self, data):
        """ """
        mixture = 1  # TODO
        ell = data["for_hu"][
            "ell"
        ]  # YES, PUT IT INTO THE DICTIONARY. TODO: create proper function in the model
        pressure = data["for_hu"]["pressure_AdArray"]
        total_flux_internal = data["for_hu"]["total_flux_internal"]
        left_restriction, right_restriction = data["for_hu"]["restriction_matrices"]
        ad = True  # Sorry
        dynamic_viscosity = 1  # Sorry
        return [
            mixture,
            ell,
            pressure,
            total_flux_internal,
            left_restriction,
            right_restriction,
            ad,
            dynamic_viscosity,
        ]

    def get_values(self, *args: AdArray) -> np.ndarray:
        """ """
        rhs_list = []
        for subdomain, data in self.mdg.subdomain(return_data=True):
            args = [subdomain]
            args_dict = self.get_args_from_sd_data_dictionary(data)
            args.append(args_dict)

            rhs_list.append(self.func(*args).val)
            # TODO: not your they are actually rhs, check the sign
            # TODO: find a smart way to not repeat the hu call for val and jac
        return np.hstack(rhs_list)

    def get_jacobian(self, *args: AdArray) -> np.ndarray:
        """ """
        jac_list = []
        for subdomain, data in self.mdg.subdomain(return_data=True):
            args = [subdomain]
            args_dict = self.get_args_from_sd_data_dictionary(data)
            args.append(args_dict)

            jac_list.append(self.func(*args).jac)

        return sp.sparse.block_diag(jac_list)


class FunctionRhoG(pp.ad.operator_functions.AbstractFunction):
    """ """

    def __init__(self, func: Callable, name: str, array_compatible: bool = True):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False

    def get_args_from_model(self):
        """to be overriden"""

        return

    def get_args_from_sd_data_dictionary(self, data):
        """ """
        mixture = 1  # TODO
        ell = data["for_hu"]["ell"]
        pressure = data["for_hu"]["pressure_AdArray"]
        gravity_value = self.gravity_value  # TODO: you know...
        left_restriction, right_restriction = data["for_hu"]["restriction_matrices"]
        ad = True  # Sorry
        dynamic_viscosity = 1  # Sorry
        return [
            mixture,
            ell,
            pressure,
            gravity_value,
            left_restriction,
            right_restriction,
            ad,
            dynamic_viscosity,
        ]

    def get_values(self, *args: AdArray) -> np.ndarray:
        """ """
        rhs_list = []
        for subdomain, data in self.mdg.subdomain(return_data=True):
            args = [subdomain]
            args_dict = self.get_args_from_sd_data_dictionary(data)
            args.append(args_dict)

            rhs_list.append(self.func(*args).val)
        return np.hstack(rhs_list)

    def get_jacobian(self, *args: AdArray) -> np.ndarray:
        """ """
        jac_list = []
        for subdomain, data in self.mdg.subdomain(return_data=True):
            args = [subdomain]
            args_dict = self.get_args_from_sd_data_dictionary(data)
            args.append(args_dict)

            jac_list.append(self.func(*args).jac)

        return sp.sparse.block_diag(jac_list)


class VariablesTwoPhaseFlow(pp.VariableMixin):
    """ """

    def create_variables(self) -> None:
        """ """
        self.equation_system.create_variables(
            self.pressure_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "Pa"},
        )

        self.equation_system.create_variables(  ####
            self.saturation_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": ""},
        )

        self.equation_system.create_variables(
            self.interface_darcy_flux_variable,
            interfaces=self.mdg.interfaces(codim=1),
            tags={"si_units": f"m^{self.nd} * Pa"},
        )

    def pressure(self, subdomains: list[pp.Grid]) -> pp.ad.MixedDimensionalVariable:
        p = self.equation_system.md_variable(self.pressure_variable, subdomains)
        return p

    def saturation(self, subdomains: list[pp.Grid]) -> pp.ad.MixedDimensionalVariable:
        s = self.equation_system.md_variable(self.saturation_variable, subdomains)
        return s

    def interface_darcy_flux(  # TODO: do we want mortar fluxes?
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """ """
        flux = self.equation_system.md_variable(
            self.interface_darcy_flux_variable, interfaces
        )
        return flux

    def reference_pressure(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        p_ref = self.fluid.pressure()
        size = sum([sd.num_cells for sd in subdomains])
        return pp.wrap_as_ad_array(p_ref, size, name="reference_pressure")


class BoundaryConditionsTwoPhaseFlow:
    """TODO: really not clear what i'm supposed to do
    don't i need just a function that returns the bc_val used in the constitutive law of hybrid upwind
    """

    domain_boundary_sides: Callable[
        [pp.Grid],
        pp.domain.DomainSides,
    ]

    fluid: pp.FluidConstants

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """ """
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def bc_type_mobrho(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """ """
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def bc_values_darcy(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """ """
        num_faces = sum([sd.num_faces for sd in subdomains])
        return pp.wrap_as_ad_array(0, num_faces, "bc_values_darcy")

    def bc_values_mobrho(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """ """
        bc_values: list[np.ndarray] = []

        for sd in subdomains:
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            vals = np.zeros(sd.num_faces)
            vals[boundary_faces] = self.fluid.density() / self.fluid.viscosity()
            bc_values.append(vals)

        bc_values_array: pp.ad.DenseArray = pp.wrap_as_ad_array(  # type: ignore
            np.hstack(bc_values), name="bc_values_mobility"
        )
        return bc_values_array


class TwoPhaseMassBalEq(pp.fluid_mass_balance.MassBalanceEquations):

    """TODO: roughly everywhere: not clear where to add ell"""

    def set_equations(self):
        """"""
        subdomains = self.mdg.subdomains()
        codim_1_interfaces = self.mdg.interfaces(codim=1)
        codim_2_interfaces = self.mdg.interfaces(codim=2)
        sd_eq = self.mass_balance_equation(subdomains)
        intf_eq = self.interface_darcy_flux_equation(codim_1_interfaces)
        # well_eq = self.well_flux_equation(codim_2_interfaces)
        self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})
        self.equation_system.set_equation(intf_eq, codim_1_interfaces, {"cells": 1})
        # self.equation_system.set_equation(well_eq, codim_2_interfaces, {"cells": 1})

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        accumulation = self.fluid_mass(subdomains)
        flux = self.fluid_flux(subdomains)
        source = self.fluid_source(subdomains)
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name("mass_balance_equation")
        return eq

    def fluid_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        mass_density = (
            self.fluid_density(subdomains) * self.saturation * self.porosity(subdomains)
        )
        mass = self.volume_integral(mass_density, subdomains, dim=1)
        mass.set_name("fluid_mass")
        return mass

    def fluid_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """

        bc_values = self.bc_values_phase_1(subdomains)  # TODO: this is tmp

        rho_flux_V_operator = FunctionRhoV(
            pp.hybrid_upwind.rho_flux_V, name="rho flux V"
        )
        rho_flux_G_operator = FunctionRhoG(
            pp.hybrid_upwind.rho_flux_G, name="rho flux G"
        )

        rho_V_G = rho_flux_V_operator + rho_flux_G_operator

        rho_V_G.set_name("flux rho V G")
        return rho_V_G

    def interface_flux_equation(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Interface flux equation.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the interface flux equation.

        """
        return self.interface_darcy_flux_equation(interfaces)

    def interface_fluid_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Interface fluid flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the interface fluid flux.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_mobility_discretization(interfaces)
        mob_rho = (
            self.mobility(subdomains) * SATURATION * self.fluid_density(subdomains)
        )

        flux: pp.ad.Operator = self.interface_advective_flux(interfaces, mob_rho, discr)
        flux.set_name("interface_fluid_flux")
        return flux


class SolutionStrategyMassBalance(
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow
):
    """ """

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        # Variables
        self.pressure_variable: str = "pressure"
        """Name of the pressure variable."""

        self.interface_darcy_flux_variable: str = "interface_darcy_flux"
        """Name of the primary variable representing the Darcy flux on interfaces of
        codimension one."""

        self.well_flux_variable: str = "well_flux"
        """Name of the primary variable representing the well flux on interfaces of
        codimension two."""

        # Discretization
        self.darcy_keyword: str = "flow"
        """Keyword for Darcy flux term.

        Used to access discretization parameters and store discretization matrices.

        """
        self.mobility_keyword: str = "mobility"
        """Keyword for mobility factor.

        Used to access discretization parameters and store discretization matrices.

        """

    def initial_condition(self) -> None:
        """TODO"""

    def set_discretization_parameters(self) -> None:
        """TODO"""
        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.darcy_keyword,
                {
                    "bc": self.bc_type_darcy(sd),
                    "second_order_tensor": self.permeability_tensor(sd),
                    "ambient_dimension": self.nd,
                },
            )
            pp.initialize_data(
                sd,
                data,
                self.mobility_keyword,
                {
                    "bc": self.bc_type_mobrho(sd),
                },
            )

        # Assign diffusivity in the normal direction of the fractures.
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
        """
        Evaluate Darcy flux for each subdomain and interface and store in the data
        dictionary for use in upstream weighting.

        """

        for sd, data in self.mdg.subdomains(return_data=True):
            mixture = 1  # TODO: you know
            pressure = self.pressure([sd]).evaluate(self.equation_system)
            left_restriction, right_restriction = data["for_hu"][
                "restriction_matrices"
            ]  # they will be created in a preceding step, in prepare_simulation
            transmissibility_internal_tpfa = data["for_hu"][
                "transmissibility_internal_tpfa"
            ]  # idem...
            ad = True
            dynamic_viscosity = 1  # Sorry
            total_flux_internal = pp.hybrid_weighted_average.total_flux(
                sd,
                mixture,
                pressure,
                self.gravity_value,
                left_restriction,
                right_restriction,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )
            data["for_hu"]["total_flux_internal"] = (
                total_flux_internal[0].val + total_flux_internal[1].val
            )
            data["for_hu"]["pressure_AdArray"] = self.pressure([sd]).evaluate(
                self.equation_system
            )
            # TODO: something similar for the saturation

            # from original before_nonlinear_iteration:
            vals = self.darcy_flux([sd]).evaluate(self.equation_system).val
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            vals = self.interface_darcy_flux([intf]).evaluate(self.equation_system).val
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})

        for intf, data in self.mdg.interfaces(return_data=True, codim=2):
            vals = self.well_flux([intf]).evaluate(self.equation_system).val
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})

        super().before_nonlinear_iteration()

    def set_nonlinear_discretizations(self) -> None:
        """TODO"""
        super().set_nonlinear_discretizations()
        self.add_nonlinear_discretization(
            self.mobility_discretization(self.mdg.subdomains()).upwind,
        )
        self.add_nonlinear_discretization(
            self.interface_mobility_discretization(self.mdg.interfaces()).flux,
        )
        # TODO


class EquationsPressureMass(
    PressureEquation,
    MassBalance,
):
    def set_equations(self):
        PressureEquation.set_equations(self)
        MassBalance.set_equations(self)


class VariablesPressureMass(
    PressureEquation,
    MassBalance,
):
    def create_variables(self) -> None:
        PressureEquation.create_variables(self)
        MassBalance.create_variables(self)


class ConstitutiveLawPressureMass(
    pp.constitutive_laws.FluidDensityFromPressure,
    pp.constitutive_laws.ConstantSolidDensity,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.FluidMobility,
    pp.constitutive_laws.ConstantPorosity,
    pp.constitutive_laws.ConstantPermeability,
    pp.constitutive_laws.ConstantViscosity,
):
    pass


class BoundaryConditionsPressureMass(
    BoundaryConditionsPressure,
    BoundaryConditionsTwoPhaseFlow,
):
    pass


class SolutionStrategyPressureMass(
    SolutionStrategyPressure,
    SolutionStrategy,
):
    pass


class MyModelGeometry(pp.ModelGeometry):
    def set_domain(self) -> None:
        """ """
        size = 1 / self.units.m
        self._domain = nd_cube_domain(2, size)


wetting_phase = ConstantDensityPhase(rho0=1)
non_wetting_phase = ConstantDensityPhase(rho0=0.5)

mixture = pp.Mixture()
mixture.add([wetting_phase, non_wetting_phase])


class FinalModel(
    EquationsPressureMass,
    VariablesPressureMass,
    ConstitutiveLawPressureMass,
    BoundaryConditionsPressureMass,
    SolutionStrategyPressureMass,
    MyModelGeometry,
    pp.DataSavingMixin,
):
    def __init__(self, mixture):
        self.mixture = mixture
        self.ell = 0  # 0 = wetting, 1 = non-wetting
        self.gravity_value = pp.GRAVITY_ACCELERATION  # TODO: ...


# no, you can add attributes through the params of the model. They become attributes in NewtonSolver, right?


fluid_constants = pp.FluidConstants(
    {
        "compressibility": 1,
        "density": 1,  # TODO: is this the reference density?
        "normal_thermal_conductivity": 1,
        "pressure": 0,
        "specific_heat_capacity": 1,
        "temperature": 0,
        "thermal_conductivity": 1,
        "thermal_expansion": 0,
        "viscosity": 1,
    }
)
material_constants = {"fluid": fluid_constants}
params = {"material_constants": material_constants}


model = FinalModel(mixture, params)
