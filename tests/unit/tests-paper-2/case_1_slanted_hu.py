import scipy as sp
import numpy as np
from typing import Callable, Optional, Type, Literal, Sequence, Union
import porepy as pp
import pygem
import copy
import os
import pdb
import porepy.models.two_phase_hu as two_phase_hu

"""

"""


class SolutionStrategyTest1(two_phase_hu.SolutionStrategyPressureMass):
    def prepare_simulation(self) -> None:
        """ """
        self.clean_working_directory()

        self.set_geometry(mdg_ref=False)
        self.deform_grid()

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
        os.system("rm " + self.flips_file_name)
        os.system("rm " + self.beta_file_name)
        # os.system("rm -r " + self.root_path)
        # os.system("mkdir " + self.root_path)

    @staticmethod
    def compute_normals_tangents(g_new):
        """ """
        # compute fracture normals and tangents:
        frac_faces = g_new.tags["fracture_faces"]
        normals = g_new.face_normals
        normals = normals[[0, 1]]  # no z componnet
        frac_normals = np.zeros([2, normals[0][frac_faces].shape[0]])

        frac_normals[0] = normals[0][frac_faces]
        frac_normals[1] = normals[1][frac_faces]

        # I use only one normal and tangential bcs of straight fault
        normal_ref = frac_normals[:, 0]  # random one, it doens't mattter
        normal_ref = normal_ref / np.sqrt(np.dot(normal_ref, normal_ref))
        tangent_ref = np.array([1, -normal_ref[0] / normal_ref[1]])
        tangent_ref = tangent_ref / np.sqrt(np.dot(tangent_ref, tangent_ref))
        return normal_ref, tangent_ref

    def deform_grid(self, mdg_ref=False):
        """
        basically copied from old code. It very hardcoded
        """

        gb = copy.deepcopy(self.mdg)
        gb_new = copy.deepcopy(self.mdg)

        # g_old = gb.subdomains(dim=2)[0]
        # g_new = gb_new.subdomains(dim=2)[0]

        g_old = self.mdg.subdomains(dim=2)[0]
        g_new = g_old.copy()

        # normal_ref, tangent_ref = self.compute_normals_tangents(g_new)

        # mesh = g_new.nodes[[0, 1]].T

        # # ORIGINAL control points: ----------------------------------
        # # boundary control points:
        # boundary_ctrl_pts = g_new.nodes.T[g_new.get_boundary_nodes()][:, [0, 1]]

        # boundary_ctrl_pts_displ_tbl_ind = np.where(
        #     boundary_ctrl_pts[:, 0] < (self.xmax - 1e-10)
        # )[
        #     0
        # ]  # top, bottom, left # np.where bcs i dont like boolean indices # pay attention, these indeces refer to the subset boundary_ctrl_pts

        # boundary_ctrl_pts_displ_tr_ind = np.where(
        #     np.logical_and(
        #         boundary_ctrl_pts[:, 0] > (self.xmax - 1e-10),
        #         boundary_ctrl_pts[:, 1] > (self.ymax - 1e-10),
        #     )
        # )[
        #     0
        # ]  # top-right point
        # boundary_ctrl_pts_displ_br_ind = np.where(
        #     np.logical_and(
        #         boundary_ctrl_pts[:, 0] > (self.xmax - 1e-10),
        #         boundary_ctrl_pts[:, 1] < (self.ymin + 1e-10),
        #     )
        # )[
        #     0
        # ]  # bottom-right point

        # boundary_ctrl_pts_displ = boundary_ctrl_pts[
        #     np.hstack(
        #         (
        #             boundary_ctrl_pts_displ_tbl_ind,
        #             boundary_ctrl_pts_displ_tr_ind,
        #             boundary_ctrl_pts_displ_br_ind,
        #         )
        #     )
        # ]
        # boundary_ctrl_pts_displ = boundary_ctrl_pts_displ[
        #     np.r_[1, 3, np.arange(4, len(boundary_ctrl_pts_displ))]
        # ]  # remove fault extremes

        # boundary_ctrl_pts_sliding_ind = np.where(
        #     np.logical_and(
        #         boundary_ctrl_pts[:, 0] >= (self.xmax - 1e-10),
        #         np.abs(boundary_ctrl_pts[:, 1] - 0.5) < (0.5 - 1e-10),
        #     )
        # )[0]
        # boundary_ctrl_pts_sliding = boundary_ctrl_pts[boundary_ctrl_pts_sliding_ind]

        # # fracture/fault control points:
        # # find nodes on each side of the fracture:
        # fn = g_new.face_nodes.indices
        # faces_nodes = np.array([fn[::2], fn[1::2]]).T

        # x_intersection = self.x_intersection
        # y_intersection = self.y_intersection

        # negative_pts_fault_ind = faces_nodes[g_new.frac_pairs[0]].flatten()  # indeces
        # negative_pts_fault_ind = np.unique(
        #     negative_pts_fault_ind
        # )  # ? there are repeated indeces
        # negative_pts_fault_ind = negative_pts_fault_ind[negative_pts_fault_ind != 0]
        # negative_pts_fault_ind = negative_pts_fault_ind[negative_pts_fault_ind != 3]

        # positive_pts_fault_ind = faces_nodes[g_new.frac_pairs[1]].flatten()  # indeces
        # positive_pts_fault_ind = np.unique(positive_pts_fault_ind)  # ? vedi spra
        # positive_pts_fault_ind = positive_pts_fault_ind[positive_pts_fault_ind != 1]
        # positive_pts_fault_ind = positive_pts_fault_ind[positive_pts_fault_ind != 2]

        # intersection_upper_ind = np.where(
        #     np.logical_and(
        #         np.isclose(mesh[:, 0], x_intersection),
        #         np.isclose(mesh[:, 1], y_intersection),
        #     )
        # )[0]

        # # positive_pts_fault_ind_no_hor = positive_pts_fault_ind[
        # #     positive_pts_fault_ind != intersection_upper_ind[0]
        # # ]

        # # positive_pts_fault_ind_no_hor = positive_pts_fault_ind[
        # #     positive_pts_fault_ind != intersection_upper_ind[1]
        # # ]

        # positive_pts_fault_ind_no_hor = positive_pts_fault_ind

        # frac_ctrl_pts_positive = mesh[positive_pts_fault_ind_no_hor]
        # frac_ctrl_pts_negative = mesh[negative_pts_fault_ind]

        # # horizons control points:
        # horizon_face_1 = g_new.tags["auxiliary_line_1_faces"]

        # horizon_1_nodes = g_new.face_nodes * horizon_face_1
        # horizon_1_ctrl_pts = mesh[horizon_1_nodes]

        # eps = 1e-8
        # horizon_1_ctrl_pts_right_ind = np.where(
        #     np.logical_and(
        #         horizon_1_ctrl_pts[:, 0] > x_intersection + eps,
        #         horizon_1_ctrl_pts[:, 0] < self.xmax - eps,
        #     )
        # )[0]

        # horizon_1_ctrl_pts_right = horizon_1_ctrl_pts[horizon_1_ctrl_pts_right_ind]
        # horizon_1_ctrl_pts_right = np.vstack(
        #     (
        #         horizon_1_ctrl_pts_right,
        #         np.array([x_intersection, y_intersection]),
        #     )
        # )  # intersections added manually

        # horizons_ctrl_pts_right = horizon_1_ctrl_pts_right

        # # create sets of control points:
        # group_displ = np.array(
        #     [
        #         *boundary_ctrl_pts_displ,
        #         *horizons_ctrl_pts_right,
        #     ]
        # )  # tips are included in the boundary
        # group_displ_fault = np.array(
        #     [*frac_ctrl_pts_negative]
        # )  # matter of notation? isnt frac_ctrl_pts_negative already a np.array?
        # group_sliding = np.array([*boundary_ctrl_pts_sliding])
        # group_sliding_fault = np.array([*frac_ctrl_pts_positive])

        # original_ctrl_pts = np.array(
        #     [*group_displ, *group_displ_fault, *group_sliding, *group_sliding_fault]
        # )

        # S_displ = np.arange(0, group_displ.shape[0])  # ctrl pts id is arbitrary
        # S_displ_fault = np.arange(
        #     group_displ.shape[0], group_displ.shape[0] + group_displ_fault.shape[0]
        # )
        # S_sliding = np.arange(
        #     group_displ.shape[0] + group_displ_fault.shape[0],
        #     group_displ.shape[0] + group_displ_fault.shape[0] + group_sliding.shape[0],
        # )
        # S_sliding_fault = np.arange(
        #     group_displ.shape[0] + group_displ_fault.shape[0] + group_sliding.shape[0],
        #     group_displ.shape[0]
        #     + group_displ_fault.shape[0]
        #     + group_sliding.shape[0]
        #     + group_sliding_fault.shape[0],
        # )

        # # DEFORMED control points: -----------------------------------------
        # displacement_x = self.displacement_max * tangent_ref[0]
        # displacement_y = self.displacement_max * tangent_ref[1]

        # # horizons control points:
        # horizon_1_ctrl_pts_right[:, 0] += displacement_x + (0 - displacement_x) / (
        #     1 - x_intersection
        # ) * (horizon_1_ctrl_pts_right[:, 0] - x_intersection)

        # horizons_ctrl_pts_right = horizon_1_ctrl_pts_right

        # horizons_ctrl_pts_right[:, 1] += displacement_y

        # middle = np.array(
        #     [0.5, 0.5]
        # )  # don't use tips bcs they may be missing, hardcoded

        # group_displ_deformed = np.array(
        #     [
        #         *boundary_ctrl_pts_displ,
        #         *horizons_ctrl_pts_right,
        #     ]
        # )
        # deformed_ctrl_pts = np.array(
        #     [
        #         *group_displ_deformed,
        #         *group_displ_fault,
        #         *group_sliding,
        #         *group_sliding_fault,
        #     ]
        # )

        # # Sets of nodes (NOT control points) in or not in fault, required for evaluation influence matrix:
        # S_nodes_fault = np.where(g_new.tags["fracture_nodes"] == True)[0]
        # S_nodes_fault = S_nodes_fault[S_nodes_fault != 0]
        # S_nodes_fault = S_nodes_fault[S_nodes_fault != 1]
        # S_nodes_fault = S_nodes_fault[S_nodes_fault != 2]
        # S_nodes_fault = S_nodes_fault[S_nodes_fault != 3]
        # S_nodes_not_fault = np.setdiff1d(np.arange(g_new.num_nodes), S_nodes_fault)

        # # generate rbf class:
        # rbf = pygem.RBF(
        #     original_ctrl_pts=original_ctrl_pts,
        #     deformed_ctrl_pts=deformed_ctrl_pts,
        #     S_displ=S_displ,
        #     S_displ_fault=S_displ_fault,
        #     S_sliding=S_sliding,
        #     S_sliding_fault=S_sliding_fault,
        #     normals=[np.array([-1, 0]), normal_ref],
        #     tangents={"non_fault": [np.array([0, 1])], "fault": [tangent_ref]},
        #     x_ref=middle,
        #     S_nodes_not_fault=S_nodes_not_fault,
        #     S_nodes_fault=S_nodes_fault,
        #     func="polyharmonic_spline",
        #     radius=0.2,
        #     influence="linear_1",
        # )

        # # sides of nodes and ctrl pts:
        # side_nodes = rbf.side_function(
        #     mesh,
        #     normals=normal_ref,
        #     x_refs=middle,
        #     positive_pts_fault_ind=positive_pts_fault_ind,
        # )
        # positive_ctrl_pts_fault_ind = np.arange(
        #     group_displ.shape[0] + group_displ_fault.shape[0] + group_sliding.shape[0],
        #     group_displ.shape[0]
        #     + group_displ_fault.shape[0]
        #     + group_sliding.shape[0]
        #     + group_sliding_fault.shape[0],
        # )
        # side_ctrl_pts = rbf.side_function(
        #     original_ctrl_pts,
        #     normals=normal_ref,
        #     x_refs=middle,
        #     positive_pts_fault_ind=positive_ctrl_pts_fault_ind,
        # )

        # # apply deformation:
        # move, _ = rbf(mesh, side_nodes, side_ctrl_pts)

        # # update the porepy mesh:
        # move = move.T
        # g_new.nodes[0] += move[0]
        # g_new.nodes[1] += move[1]

        # g_new.compute_geometry()

        # pp.plot_grid(g_new, alpha=0)
        # pdb.set_trace()

        # update the whole grid bucket: -----------------------------------------------

        self.mdg.replace_subdomains_and_interfaces({g_old: g_new})

        # gb.compute_geometry()

        # # rebuild mortar grid: already done in replace_grids, uncomment if you want to take a look...
        # for intf, data in gb.interfaces(return_data=True):
        #     # queste mappe sono state aggiornate automaticamente e non sono più l'identità come prima
        #     A = intf.primary_to_mortar_int().todense()
        #     B = intf.primary_to_mortar_avg().todense()

        #     # queste invece sono rimaste identità dato che dal mortar alla faglia non mi cambia nulla
        #     C = intf.secondary_to_mortar_int().todense()
        #     D = intf.secondary_to_mortar_avg().todense()

        pp.plot_grid(self.mdg, alpha=0)
        pp.plot_grid(gb, alpha=0)

        # self.mdg = gb

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
            saturation_values[np.where(sd.cell_centers[1] >= self.ymax / 2)] = 1.0
            # saturation_values = 1.0 - 1 * sd.cell_centers[1] / self.ymax

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
                    [self.x_intersection + 0.1, self.xmax],
                    [
                        self.y_intersection,
                        self.y_intersection,
                    ],
                ]
            )
        )

        frac_constr_2 = pp.LineFracture(
            np.array(
                [
                    [self.xmin, self.xmean - 0.1],
                    [self.ymean, self.ymean],
                ]
            )
        )  # -0.1 just to not touch the fracture # Removed
        self._fractures: list = [frac1, frac_constr_1, frac_constr_2]

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": self.cell_size / self.L_0,
            "cell_size_fracture": self.cell_size / self.L_0,
        }
        return self.params.get("meshing_arguments", default_meshing_args)


class PartialFinalModel(
    two_phase_hu.PrimaryVariables,
    two_phase_hu.Equations,
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

    Kn = 10.0
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
            self.displacement_max = -0.0

            self.relative_permeability = (
                pp.tobedefined.relative_permeability.rel_perm_quadratic
            )
            self.mobility = pp.tobedefined.mobility.mobility(self.relative_permeability)
            self.mobility_operator = pp.tobedefined.mobility.mobility_operator(
                self.mobility
            )

            self.number_upwind_dirs = 3
            self.sign_total_flux_internal_prev = None
            self.sign_omega_0_prev = None
            self.sign_omega_1_prev = None

            # self.root_path = "./case_1/slanted_hu_Kn" + str(Kn) + "/"
            self.root_path = "./case_1/slanted_hu_Kn" + str(Kn) + "/non-conforming/"

            self.output_file_name = self.root_path + "OUTPUT_NEWTON_INFO"
            self.mass_output_file_name = self.root_path + "MASS_OVER_TIME"
            self.flips_file_name = self.root_path + "FLIPS"
            self.beta_file_name = self.root_path + "BETA/BETA"

    cell_size = 0.1

    # os.system("mkdir -p ./case_1/slanted_hu_Kn" + str(Kn))
    os.system("mkdir -p ./case_1/slanted_hu_Kn" + str(Kn) + "/non-conforming")

    # os.system("mkdir -p ./case_1/slanted_hu_Kn" + str(Kn) + "/BETA")
    os.system("mkdir -p ./case_1/slanted_hu_Kn" + str(Kn) + "/non-conforming/BETA")

    # folder_name = "./case_1/slanted_hu_Kn" + str(Kn) + "/visualization"
    folder_name = "./case_1/slanted_hu_Kn" + str(Kn) + "/non-conforming/visualization"

    time_manager = two_phase_hu.TimeManagerPP(
        schedule=np.array([0, 10]) / t_0,
        dt_init=1e-1 / t_0,
        dt_min_max=np.array([1e-5, 1e-1]) / t_0,
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

    pp.run_time_dependent_model(model, params)
