import numpy as np
from typing import Callable, Optional, Type, Literal, Sequence, Union
import porepy as pp
import test_hu_model
import test_ppu_model
import test_1_paper_hu


class SolutionStrategyTest1(test_ppu_model.SolutionStrategyPressureMassPPU):
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

    def after_nonlinear_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ):
        """ """

        if self._is_nonlinear_problem():
            print("\n===========================================")
            print(
                "Nonlinear iterations did not converge. I'm going to reduce the time step"
            )
            print("===========================================")

            prev_sol = self.equation_system.get_variable_values(time_step_index=0)

            self.equation_system.set_variable_values(
                values=prev_sol, additive=False, iterate_index=0
            )

            (
                eq,
                accumulation,
                rho_V_operator,
                rho_G_operator,
                flux_V_G,
                flux_intf_phase_0,
                flux_intf_phase_1,
                source_phase_0,
                source_phase_1,
            ) = self.eq_fcn_mass(self.mdg.subdomains())

        else:
            raise ValueError("Tried solving singular matrix for the linear problem.")


class PartialFinalModel(
    test_hu_model.PrimaryVariables,
    test_ppu_model.EquationsPPU,
    test_hu_model.ConstitutiveLawPressureMass,
    test_hu_model.BoundaryConditionsPressureMass,
    SolutionStrategyTest1,
    test_1_paper_hu.GeometryTest1,
    pp.DataSavingMixin,
):
    """ """


fluid_constants = pp.FluidConstants({})
solid_constants = pp.SolidConstants(
    {
        "porosity": 0.25,
        "intrinsic_permeability": 1,
        "normal_permeability": 1,
        "residual_aperture": 0.1,
    }
)

material_constants = {"fluid": fluid_constants, "solid": solid_constants}

time_manager = test_1_paper_hu.time_manager

params = test_1_paper_hu.params
mixture = test_1_paper_hu.mixture

if __name__ == "__main__":

    class FinalModel(PartialFinalModel):
        def __init__(self, params: Optional[dict] = None):
            super().__init__(params)
            self.mixture = mixture
            self.ell = 0
            self.gravity_value = 1
            self.dynamic_viscosity = 1

            self.xmin = 0
            self.xmax = 1
            self.ymin = 0
            self.ymax = 1

            self.relative_permeability = (
                pp.tobedefined.relative_permeability.rel_perm_quadratic
            )
            self.mobility = pp.tobedefined.mobility.mobility(self.relative_permeability)
            self.mobility_operator = pp.tobedefined.mobility.mobility_operator(
                self.mobility
            )

    model = FinalModel(params)

    pp.run_time_dependent_model(model, params)
