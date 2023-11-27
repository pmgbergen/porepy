import scipy as sp
import numpy as np
from typing import Callable, Optional, Type, Literal, Sequence, Union
import porepy as pp
import os
import pdb
import porepy.models.two_phase_hu as two_phase_hu
import porepy.models.two_phase_ppu as two_phase_ppu

"""
- permeable k = 10, impermeable k = 0.1
- non conforming displacement_max = -0.1, only for permeable fracture
- linear solver is the bottleneck for small mesh size

"""


class SolutionStrategyTest1(two_phase_ppu.SolutionStrategyPressureMassPPU):
    def prepare_simulation(self) -> None:
        """ """
        self.clean_working_directory()

        self.set_geometry(mdg_ref=True)
        self.set_geometry(mdg_ref=False)

        self.initialize_data_saving()

        self.set_materials()

        self.set_equation_system_manager()

        self.add_equation_system_to_phases()
        self.mixture.apply_constraint(self.ell)

        self.create_variables()

        self.initial_condition()
        self.compute_mass()

        self.reset_state_from_file()
        self.set_equations()

        self.set_discretization_parameters()
        self.discretize()

        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()

        self.save_data_time_step()  # it is in pp.viz.data_saving_model_mixin

        self.computations_for_hu()

    def clean_working_directory(self):
        """ """
        os.system("rm " + self.output_file_name)
        os.system("rm " + self.mass_output_file_name)
        # os.system("rm -r " + self.root_path)
        # os.system("mkdir " + self.root_path)

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
            # saturation_values[np.where(sd.cell_centers[1] >= self.ymax / 2)] = 1.0
            saturation_values = 1.0 - 1 * sd.cell_centers[1] / self.ymax

            # if sd.dim == 1:
            #     saturation_values = 0.5 * np.ones(sd.num_cells)

            self.equation_system.set_variable_values(
                saturation_values,
                variables=[saturation_variable],
                time_step_index=0,
                iterate_index=0,
            )

            pressure_variable = self.pressure([sd])

            pressure_values = (2 - 1 * sd.cell_centers[1] / self.ymax) / self.p_0

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

    def after_simulation(self) -> None:
        """
        old = coarse
        new = fine, ref
        goal: have variables on fine grid

        hardcoded for two grids only
        """

        tol = 1e-6

        sd_id = 0  # not used, just remember that it might be useful to order the grids without using operators

        for sd_ref, sd in zip(self.mdg_ref.subdomains(), self.mdg.subdomains()):
            if sd_ref.dim == 2:
                mapping_matrix_2d = pp.match_grids.match_2d(
                    sd_ref, sd, tol, "averaged"
                )  # this takes long time!
                sp.sparse.save_npz(
                    self.root_path + "mapping_matrix_2d_" + str(self.cell_size),
                    mapping_matrix_2d,
                )
                np.save(
                    self.root_path + "volumes_2d_" + str(self.cell_size),
                    sd.cell_volumes,
                )
            if sd_ref.dim == 1:
                # mapping_matrix_1d = pp.match_grids.match_1d(new_sd, old_sd, tol, "averaged")
                mapping_matrix_1d = pp.match_grids.match_1d(sd_ref, sd, tol, "averaged")
                sp.sparse.save_npz(
                    self.root_path + "mapping_matrix_1d_" + str(self.cell_size),
                    mapping_matrix_1d,
                )

                np.savetxt(
                    self.root_path + "volumes_1d_" + str(self.cell_size),
                    sd.cell_volumes,
                )

            sd_id += 1

        pressure = (
            self.pressure(self.mdg.subdomains()).evaluate(self.equation_system).val
        )
        saturation = (
            self.mixture.get_phase(0)
            .saturation_operator(self.mdg.subdomains())
            .evaluate(self.equation_system)
        ).val
        mortar_phase_0 = (
            self.interface_mortar_flux_phase_0(self.mdg.interfaces())
            .evaluate(self.equation_system)
            .val
        )
        mortar_phase_1 = (
            self.interface_mortar_flux_phase_1(self.mdg.interfaces())
            .evaluate(self.equation_system)
            .val
        )

        np.save(self.root_path + "pressure_" + str(self.cell_size), pressure)
        np.save(self.root_path + "saturation_" + str(self.cell_size), saturation)
        np.savetxt(
            self.root_path + "mortar_phase_0_" + str(self.cell_size), mortar_phase_0
        )
        np.savetxt(
            self.root_path + "mortar_phase_1_" + str(self.cell_size), mortar_phase_1
        )

        np.savetxt(
            self.root_path + "variable_num_dofs_" + str(self.cell_size),
            self.equation_system._variable_num_dofs,
        )


class GeometryConvergence(pp.ModelGeometry):
    def set_geometry(self, mdg_ref=False) -> None:
        """ """

        self.set_domain()
        self.set_fractures()

        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)

        if mdg_ref:
            self.mdg_ref = pp.create_mdg(
                "simplex",
                self.meshing_arguments_ref(),
                self.fracture_network,
                **self.meshing_kwargs(),
            )

        self.mdg = pp.create_mdg(
            "simplex",
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
        self.size = 1

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
        frac1 = pp.LineFracture(
            np.array([[self.x_bottom, self.x_top], [self.ymin, self.ymax]])
        )

        tangent = np.array(
            [np.sin(self.tilt_angle), np.cos(self.tilt_angle)]
        )  # sorry, I cant use the dedicated function because it requires the grid that i dont have yet
        displ_x = self.displacement_max * tangent[0]
        displ_y = self.displacement_max * tangent[1]

        self.x_intersection = self.xmean - displ_x
        self.y_intersection = self.ymean - displ_y

        frac_constr_1 = pp.LineFracture(
            np.array(
                [
                    [self.x_intersection, self.xmax],
                    [
                        self.y_intersection,
                        self.y_intersection,
                    ],
                ]
            )
        )

        frac_constr_2 = pp.LineFracture(
            np.array([[self.xmin, self.xmean], [self.ymean, self.ymean]])
        )

        self._fractures: list = [frac1, frac_constr_1, frac_constr_2]

    def meshing_arguments_ref(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": self.cell_size_ref / self.L_0,
            "cell_size_fracture": self.cell_size_ref / self.L_0,
        }
        return self.params.get("meshing_arguments", default_meshing_args)

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": self.cell_size / self.L_0,
            "cell_size_fracture": self.cell_size / self.L_0,
        }
        return self.params.get("meshing_arguments", default_meshing_args)


class PartialFinalModel(
    two_phase_hu.PrimaryVariables,
    two_phase_ppu.EquationsPPU,
    two_phase_hu.ConstitutiveLawPressureMass,
    two_phase_hu.BoundaryConditionsPressureMass,
    SolutionStrategyTest1,
    GeometryConvergence,
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

    Kn = 0.1
    solid_constants = pp.SolidConstants(
        {
            "porosity": 0.25,
            "intrinsic_permeability": 1.0 / Ka_0,
            "normal_permeability": Kn / Ka_0,
            "residual_aperture": 0.1 / L_0,
        }
    )

    material_constants = {"fluid": fluid_constants, "solid": solid_constants}

    wetting_phase = pp.composite.phase.Phase(rho0=1 / rho_0, p0=p_0, beta=1e-10)
    non_wetting_phase = pp.composite.phase.Phase(rho0=0.5 / rho_0, p0=p_0, beta=1e-10)

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    class FinalModel(PartialFinalModel):
        def __init__(self, params: Optional[dict] = None):
            super().__init__(params)

            self.mdg_ref = None  # fine mesh
            self.mdg = None  # coarse mesh
            self.cell_size = None
            self.cell_size_ref = None

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
            self.gravity_value = 1.0 / self.gravity_0
            self.dynamic_viscosity = 1.0 / self.dynamic_viscosity_0

            self.xmin = 0.0 / self.L_0
            self.xmax = 1.0 / self.L_0
            self.ymin = 0.0 / self.L_0
            self.ymax = 1.0 / self.L_0

            self.xmean = (self.xmax - self.xmin) / 2
            self.ymean = (self.ymax - self.ymin) / 2

            self.tilt_angle = 30 * np.pi / 180
            self.x_bottom = (
                self.xmean
                - (self.ymax - self.ymin)
                / 2
                * np.sin(self.tilt_angle)
                / np.cos(self.tilt_angle)
                / self.L_0
            )
            self.x_top = (
                self.xmean
                + (self.ymax - self.ymin)
                / 2
                * np.sin(self.tilt_angle)
                / np.cos(self.tilt_angle)
                / self.L_0
            )
            self.displacement_max = (
                -0.0
            )  # it doesnt work with a positive value, but who cares...

            self.relative_permeability = (
                pp.tobedefined.relative_permeability.rel_perm_quadratic
            )
            self.mobility = pp.tobedefined.mobility.mobility(self.relative_permeability)
            self.mobility_operator = pp.tobedefined.mobility.mobility_operator(
                self.mobility
            )

            # self.root_path = (
            #     "./case_1/slanted_ppu_Kn" + str(Kn) + "/convergence_results/"
            # )
            self.root_path = (
                "./case_1/non-conforming/slanted_ppu_Kn"
                + str(Kn)
                + "/convergence_results/"
            )

            self.output_file_name = self.root_path + "OUTPUT_NEWTON_INFO"
            self.mass_output_file_name = self.root_path + "MASS_OVER_TIME"
            self.flips_file_name = self.root_path + "FLIPS"

    cell_sizes = np.array([0.4, 0.2, 0.1, 0.025])  # last one is the ref value

    # os.system("mkdir -p ./case_1/slanted_ppu_Kn" + str(Kn) + "/convergence_results")
    os.system(
        "mkdir -p ./case_1/non-conforming/slanted_ppu_Kn"
        + str(Kn)
        + "/convergence_results"
    )

    # np.savetxt(
    #     "./case_1/slanted_ppu_Kn" + str(Kn) + "/convergence_results/cell_sizes",
    #     cell_sizes,
    # )
    np.savetxt(
        "./case_1/non-conforming/slanted_ppu_Kn"
        + str(Kn)
        + "/convergence_results/cell_sizes",
        cell_sizes,
    )

    for cell_size in cell_sizes:
        print(
            "\n\n\ncell_size = ",
            cell_size,
            "==========================================",
        )

        # folder_name = (
        #     "./case_1/slanted_ppu_Kn"
        #     + str(Kn)
        #     + "/convergence_results/visualization_"
        #     + str(cell_size)
        # )

        folder_name = (
            "./case_1/non-conforming/slanted_ppu_Kn"
            + str(Kn)
            + "/convergence_results/visualization_"
            + str(cell_size)
        )

        time_manager = two_phase_hu.TimeManagerPP(
            schedule=np.array([0, 1e-4]) / t_0,
            dt_init=1e-4 / t_0,
            dt_min_max=np.array([1e-4, 1e-3]) / t_0,
            constant_dt=False,
            recomp_factor=0.5,
            recomp_max=10,
            iter_max=10,
            print_info=True,
            folder_name=folder_name,
        )

        meshing_kwargs = {"constraints": np.array([1, 2])}
        params = {
            "material_constants": material_constants,
            "max_iterations": 20,
            "nl_convergence_tol": 1e-6,
            "nl_divergence_tol": 1e0,
            "time_manager": time_manager,
            "folder_name": folder_name,
            "meshing_kwargs": meshing_kwargs,
        }

        model = FinalModel(params)
        model.cell_size = cell_size
        model.cell_size_ref = cell_sizes[-1]

        pp.run_time_dependent_model(model, params)
