import numpy as np
from typing import Callable, Optional, Type, Literal, Sequence, Union
import porepy as pp
import pdb
import test_hu_model


class SolutionStrategyTest1(test_hu_model.SolutionStrategyPressureMass):
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
                self.ppu_keyword,
                {
                    "darcy_flux_phase_0": np.zeros(sd.num_faces),
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

        else:
            raise ValueError("Tried solving singular matrix for the linear problem.")


class GeometryTest1(pp.ModelGeometry):
    def set_geometry(self) -> None:
        """ """

        self.set_domain()
        self.set_fractures()

        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)

        self.mdg = pp.create_mdg(
            "cartesian",
            self.meshing_arguments(),
            self.fracture_network,
            **self.meshing_kwargs(),
        )
        self.nd: int = self.mdg.dim_max()

        pp.set_local_coordinate_projections(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            assert isinstance(self.fracture_network, pp.FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            self.well_network.mesh(self.mdg)

    def set_domain(self) -> None:
        """ """
        self.size = 1 / self.units.m

        # unstructred unit square:
        bounding_box = {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
        }
        self._domain = pp.Domain(bounding_box=bounding_box)

    def set_fractures(self) -> None:
        """ """
        self._fractures: list = []

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": 0.05,
            "cell_size_fracture": 0.05,
        }
        return self.params.get("meshing_arguments", default_meshing_args)


class PartialFinalModel(
    test_hu_model.PrimaryVariables,
    test_hu_model.Equations,
    test_hu_model.ConstitutiveLawPressureMass,
    test_hu_model.BoundaryConditionsPressureMass,
    SolutionStrategyTest1,
    GeometryTest1,
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


time_manager = test_hu_model.TimeManagerPP(
    schedule=[0, 50],
    dt_init=5e-1,
    dt_min_max=[1e-1, 5e-1],
    constant_dt=False,
    recomp_factor=0.5,
    recomp_max=10,
    iter_max=10,
    print_info=True,
)

params = {
    "material_constants": material_constants,
    "max_iterations": 20,
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e0,
    "time_manager": time_manager,
}

wetting_phase = pp.composite.phase.Phase(rho0=1, beta=1e-10)
non_wetting_phase = pp.composite.phase.Phase(rho0=0.5, beta=1e-10)

mixture = pp.Mixture()
mixture.add([wetting_phase, non_wetting_phase])

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

            self.output_file_name = "./OUTPUT_NEWTON_INFO"

    model = FinalModel(params)

    pp.run_time_dependent_model(model, params)
