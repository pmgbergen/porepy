import porepy as pp
import pytest
import numpy as np


class TestContactMechanicsBiot:
    """The following tests check the correct setup of the ContactMechanicsBiot model."""

    # ----------> Setting up parameters corresponding to the test of default parameters
    model_params = []
    before_runs = [True, False]
    for before_run in before_runs:
        model_param = {"use_ad": True, "before_run": before_run}
        model_params.append(model_param)

    @pytest.mark.parametrize("model_params", model_params)
    def test_default_contact_mechanics_biot(self, model_params):
        """Test default parameters before and after running the model."""

        # ----------> Create model
        model = pp.ContactMechanicsBiot(model_params)
        if model.params["before_run"]:
            model.prepare_simulation()
        else:
            pp.run_time_dependent_model(model, model_params)

        # ----------> Retrieve subdomain and data
        sd = model.mdg.subdomains()[0]
        data = model.mdg.subdomain_data(sd)

        # ----------> Check attributes
        assert model.time_step == 1.0
        assert model.end_time == 1.0
        if model.params["before_run"]:
            assert model.time == 0.0
        else:
            assert model.time == 1.0
        assert model.scalar_variable == "p"
        assert model.mortar_scalar_variable == "mortar_p"
        assert model.scalar_coupling_term == "robin_p"
        assert model.scalar_parameter_key == "flow"
        assert model.scalar_scale == 1.0
        assert model.length_scale == 1.0
        assert model.subtract_fracture_pressure

        # ----------> Check mechanics parameters
        mech_kw = model.mechanics_parameter_key
        # Boundary condition type
        bc_mech = data[pp.PARAMETERS][mech_kw]["bc"]
        assert np.all(bc_mech.is_dir)
        assert not np.all(bc_mech.is_neu)
        assert not np.all(bc_mech.is_rob)
        # Boundary condition values
        bc_mech_vals = data[pp.PARAMETERS][mech_kw]["bc_values"]
        np.testing.assert_equal(
            bc_mech_vals, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        # Stiffness tensor
        stiff = data[pp.PARAMETERS][mech_kw]["fourth_order_tensor"]
        np.testing.assert_equal(stiff.mu, np.array([1.0]))
        np.testing.assert_equal(stiff.lmbda, np.array([1.0]))
        np.testing.assert_equal(
            stiff.values, pp.FourthOrderTensor(np.array([1.0]), np.array([1.0])).values
        )
        # Source term
        source_mech = data[pp.PARAMETERS][mech_kw]["source"]
        np.testing.assert_equal(source_mech, np.array([0.0, 0.0]))
        # Biot alpha
        biot_alpha_mech = data[pp.PARAMETERS][mech_kw]["biot_alpha"]
        np.testing.assert_equal(biot_alpha_mech, np.array([1.0]))
        # Reference pressure
        p_reference = data[pp.PARAMETERS][mech_kw]["p_reference"]
        np.testing.assert_equal(p_reference, np.array([0.0]))
        # Time step
        time_step_mech = data[pp.PARAMETERS][mech_kw]["time_step"]
        np.testing.assert_equal(time_step_mech, np.array([1.0]))

        # ----------> Check flow parameters
        flow_kw = model.scalar_parameter_key
        # Boundary condition type
        bc_flow = data[pp.PARAMETERS][flow_kw]["bc"]
        assert np.all(bc_flow.is_dir)
        assert not np.all(bc_flow.is_neu)
        assert not np.all(bc_flow.is_rob)
        # Boundary condition values
        bc_flow_vals = data[pp.PARAMETERS][flow_kw]["bc_values"]
        np.testing.assert_equal(bc_flow_vals, np.array([0.0, 0.0, 0.0, 0.0]))
        # Mass weights
        mass_weight = data[pp.PARAMETERS][flow_kw]["mass_weight"]
        np.testing.assert_equal(mass_weight, np.array([1.0]))
        # Biot-alpha
        biot_alpha_flow = data[pp.PARAMETERS][flow_kw]["biot_alpha"]
        np.testing.assert_equal(biot_alpha_flow, np.array([1.0]))
        # Source
        source = data[pp.PARAMETERS][flow_kw]["source"]
        np.testing.assert_equal(source, np.array([0.0]))
        # Second order tensor
        perm = data[pp.PARAMETERS][flow_kw]["second_order_tensor"].values
        np.testing.assert_equal(perm, pp.SecondOrderTensor(np.array([1.0])).values)
        # Vector source
        vector_source = data[pp.PARAMETERS]["flow"]["vector_source"]
        np.testing.assert_equal(vector_source, np.array([0.0, 0.0]))
        # Ambient dimension
        ambient_dim = data[pp.PARAMETERS]["flow"]["ambient_dimension"]
        assert ambient_dim == 2

    # ----------> Setting up parameters corresponding to the test of non-default parameters

    # -----> Mixed-dimensional grid
    mdg, box = pp.md_grids_2d.single_vertical(mesh_args=np.array([2, 1]), simplex=False)
    sd_prim = mdg.subdomains(dim=2)[0]
    data_prim = mdg.subdomain_data(sd_prim)
    sd_sec = mdg.subdomains(dim=1)[0]
    data_sec = mdg.subdomain_data(sd_sec)
    intf = mdg.interfaces(dim=1)[0]
    data_inf = mdg.interface_data(intf)

    # -----> Scalar bc types
    all_bf = sd_prim.get_boundary_faces()
    tol = 1e-10
    east = sd_prim.face_centers[0] > box["xmax"] - tol
    west = sd_prim.face_centers[0] < box["xmin"] + tol
    north = sd_prim.face_centers[1] > box["ymax"] - tol
    south = sd_prim.face_centers[1] < box["ymin"] + tol
    east_bf = np.isin(all_bf, np.where(east)).nonzero()
    west_bf = np.isin(all_bf, np.where(west)).nonzero()
    south_bf = np.isin(all_bf, np.where(south)).nonzero()
    north_bf = np.isin(all_bf, np.where(north)).nonzero()

    # Case 0: All Neumann, except south and north which are Dirichlet
    bf_flow_prim_0 = np.array(all_bf.size * ["neu"])
    bf_flow_prim_0[south_bf] = "dir"
    bf_flow_prim_0[north_bf] = "dir"

    # Case 1: All Neumann, except east and west which are Dirichelet
    bf_flow_prim_1 = np.array(all_bf.size * ["neu"])
    bf_flow_prim_1[east_bf] = "dir"
    bf_flow_prim_1[west_bf] = "dir"

    # Case 2: All Neumann, except east, west, and north which are Dirichlet
    bf_flow_prim_2 = np.array(all_bf.size * ["neu"])
    bf_flow_prim_2[east_bf] = "dir"
    bf_flow_prim_2[west_bf] = "dir"
    bf_flow_prim_2[north_bf] = "dir"

    bc_flow_type_primary = [
        pp.BoundaryCondition(sd_prim, all_bf, bf_flow_prim_0),
        pp.BoundaryCondition(sd_prim, all_bf, bf_flow_prim_1),
        pp.BoundaryCondition(sd_prim, all_bf, bf_flow_prim_2),
    ]
    bc_flow_type_secondary = [
        pp.BoundaryCondition(sd_sec, np.array([0, 1]), ["dir", "dir"]),
        pp.BoundaryCondition(sd_sec, np.array([0, 1]), ["neu", "neu"]),
        pp.BoundaryCondition(sd_sec, np.array([0, 1]), ["neu", "dir"]),
    ]

    # -----> Scalar bc values
    bc_flow_values_prim_0 = np.zeros(sd_prim.num_faces)
    bc_flow_values_prim_0[east] = -0.045
    bc_flow_values_prim_0[west] = -1.2
    bc_flow_values_prim_0[south] = -4.21
    bc_flow_values_prim_0[north] = -1.44

    bc_flow_values_prim_1 = np.zeros(sd_prim.num_faces)
    bc_flow_values_prim_1[east] = 0.5
    bc_flow_values_prim_1[west] = -0.5
    bc_flow_values_prim_1[south] = 0.005
    bc_flow_values_prim_1[north] = -0.005

    bc_flow_values_prim_2 = np.zeros(sd_prim.num_faces)
    bc_flow_values_prim_2[east] = 0.1
    bc_flow_values_prim_2[west] = 0.1
    bc_flow_values_prim_2[south] = 0.1
    bc_flow_values_prim_2[north] = 0.0

    bc_flow_values_sec_0 = np.array([-4.21, -1.44])
    bc_flow_values_sec_1 = np.array([0.005, -0.005])
    bc_flow_values_sec_2 = np.array([0.1, 0.0])

    bc_flow_values_primary = [
        bc_flow_values_prim_0,
        bc_flow_values_prim_1,
        bc_flow_values_prim_2,
    ]

    bc_flow_values_secondary = [
        bc_flow_values_sec_0,
        bc_flow_values_sec_1,
        bc_flow_values_sec_2,
    ]

    # -----> Scalar sources
    source_flow_primary = [-2.45 * np.ones(2), np.zeros(2), 3.51 * np.ones(2)]
    source_flow_secondary = [np.array([-3.56]), np.array([0]), np.array([6.31])]

    # -----> Populating list of model parameters
    model_params = []
    before_runs = [True, False]
    for before_run in before_runs:
        for case in range(3):
            model_param = {
                "use_ad": True,
                "case": case,
                "before_run": before_run,
                "mdg": mdg,
                "box": box,
                "bc_flow_type_primary": bc_flow_type_primary[case],
                "bc_flow_type_secondary": bc_flow_type_secondary[case],
                "bc_flow_values_primary": bc_flow_values_primary[case],
                "bc_flow_values_secondary": bc_flow_values_secondary[case],
                "source_flow_primary": source_flow_primary[case],
                "source_flow_secondary": source_flow_secondary[case],
            }
            model_params.append(model_param)

    @pytest.mark.parametrize("model_params", model_params)
    def test_non_default_parameters_contact_mechanics_biot(self, model_params):
        """Test model with non-default scalar sources"""

        # Construct the class
        class NonDefaultContactMechanicsBiot(pp.ContactMechanicsBiot):
            """Model with non-default parameters."""

            def __init__(self, params):
                super().__init__(params)

            def create_grid(self) -> None:
                """Modify default grid.

                Create a Cartesian grid in a unit square with two cells in the horizontal
                direction and one cell in the vertical direction. A single vertical fracture
                is included in the middle of the domain.

                """
                self.mdg = self.params["mdg"]
                self.box = self.params["box"]
                pp.contact_conditions.set_projections(self.mdg)

            def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
                """Modify default scalar boundary condition type"""
                if sd.dim == 2:
                    return self.params["bc_flow_type_primary"]
                else:
                    return self.params["bc_flow_type_secondary"]

            def _bc_values_scalar(self, sd: pp.Grid) -> np.ndarray:
                """Modify default boundary condition values"""
                if sd.dim == 2:
                    return self.params["bc_flow_values_primary"]
                else:
                    return self.params["bc_flow_values_secondary"]

            def _source_scalar(self, sd: pp.Grid) -> np.ndarray:
                """Modifiy default scalar source"""
                if sd.dim == 2:
                    return self.params["source_flow_primary"]
                else:
                    return self.params["source_flow_secondary"]

        model = NonDefaultContactMechanicsBiot(model_params)
        if model.params["before_run"]:
            model.prepare_simulation()
        else:
            pp.run_time_dependent_model(model, model.params)

        # Retrieve subdomain and data
        sd_prim = model.mdg.subdomains(dim=2)[0]
        data_prim = model.mdg.subdomain_data(sd_prim)
        sd_sec = model.mdg.subdomains(dim=1)[0]
        data_sec = model.mdg.subdomain_data(sd_sec)
        intf = model.mdg.interfaces(dim=1)[0]
        data_intf = model.mdg.interface_data(intf)
        flow_kw = model.scalar_parameter_key
        mech_kw = model.mechanics_parameter_key

        # -----> Scalar bc type
        bc_flow_prim = data_prim[pp.PARAMETERS][flow_kw]["bc"]
        _, east, west, north, south, *_ = model._domain_boundary_sides(sd_prim)
        bc_flow_sec = data_sec[pp.PARAMETERS][flow_kw]["bc"]
        if model.params["case"] == 0:  # east: neu, west: neu, south: dir, north: dir
            assert not np.all(bc_flow_prim.is_dir[east])
            assert np.all(bc_flow_prim.is_neu[east])
            assert not np.all(bc_flow_prim.is_rob[east])
            assert not np.all(bc_flow_prim.is_dir[west])
            assert np.all(bc_flow_prim.is_neu[west])
            assert not np.all(bc_flow_prim.is_rob[west])
            assert np.all(bc_flow_prim.is_dir[south])
            assert not np.all(bc_flow_prim.is_neu[south])
            assert not np.all(bc_flow_prim.is_rob[south])
            assert np.all(bc_flow_prim.is_dir[north])
            assert not np.all(bc_flow_prim.is_neu[north])
            assert not np.all(bc_flow_prim.is_rob[north])
            assert np.all(bc_flow_sec.is_dir)
            assert not np.all(bc_flow_sec.is_neu)
            assert not np.all(bc_flow_sec.is_rob)
        elif model.params["case"] == 1:  # east: dir, west: dir, south: neu, north: neu
            assert np.all(bc_flow_prim.is_dir[east])
            assert not np.all(bc_flow_prim.is_neu[east])
            assert not np.all(bc_flow_prim.is_rob[east])
            assert np.all(bc_flow_prim.is_dir[west])
            assert not np.all(bc_flow_prim.is_neu[west])
            assert not np.all(bc_flow_prim.is_rob[west])
            assert not np.all(bc_flow_prim.is_dir[south])
            assert np.all(bc_flow_prim.is_neu[south])
            assert not np.all(bc_flow_prim.is_rob[south])
            assert not np.all(bc_flow_prim.is_dir[north])
            assert np.all(bc_flow_prim.is_neu[north])
            assert not np.all(bc_flow_prim.is_rob[north])
            assert not np.all(bc_flow_sec.is_dir)
            assert np.all(bc_flow_sec.is_neu)
            assert not np.all(bc_flow_sec.is_rob)
        elif model.params["case"] == 2:  # east: dir, west: dir, south: neu, north: dir
            assert np.all(bc_flow_prim.is_dir[east])
            assert not np.all(bc_flow_prim.is_neu[east])
            assert not np.all(bc_flow_prim.is_rob[east])
            assert np.all(bc_flow_prim.is_dir[west])
            assert not np.all(bc_flow_prim.is_neu[west])
            assert not np.all(bc_flow_prim.is_rob[west])
            assert not np.all(bc_flow_prim.is_dir[south])
            assert np.all(bc_flow_prim.is_neu[south])
            assert not np.all(bc_flow_prim.is_rob[south])
            assert np.all(bc_flow_prim.is_dir[north])
            assert not np.all(bc_flow_prim.is_neu[north])
            assert not np.all(bc_flow_prim.is_rob[north])
            np.testing.assert_equal(bc_flow_sec.is_dir, np.array([False, True]))
            np.testing.assert_equal(bc_flow_sec.is_neu, np.array([True, False]))
            assert not np.all(bc_flow_sec.is_rob)
        else:
            raise NotImplementedError("Test case not implemented.")

        # -----> Scalar bc values
        bc_flow_val_prim = data_prim[pp.PARAMETERS][flow_kw]["bc_values"]
        bc_flow_val_sec = data_sec[pp.PARAMETERS][flow_kw]["bc_values"]
        if model_params["case"] == 0:
            np.testing.assert_equal(
                bc_flow_val_prim,
                np.array([-1.2, 0.0, -0.045, -4.21, -4.21, -1.44, -1.44, 0.0]),
            )
            np.testing.assert_equal(bc_flow_val_sec, np.array([-4.21, -1.44]))
        elif model_params["case"] == 1:
            np.testing.assert_equal(
                bc_flow_val_prim,
                np.array([-0.5, 0.0, 0.5, 0.005, 0.005, -0.005, -0.005, 0.0]),
            )
            np.testing.assert_equal(bc_flow_val_sec, np.array([0.005, -0.005]))
        elif model_params["case"] == 2:
            np.testing.assert_equal(
                bc_flow_val_prim, np.array([0.1, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
            )
            np.testing.assert_equal(bc_flow_val_sec, np.array([0.1, 0.0]))
        else:
            raise NotImplementedError("Test case not implemented.")

        # -----> Scalar source
        source_prim_known = data_prim[pp.PARAMETERS][flow_kw]["source"]
        source_sec_known = data_sec[pp.PARAMETERS][flow_kw]["source"]
        if model.params["case"] == 0:
            np.testing.assert_equal(
                source_prim_known, -2.45 * np.ones(sd_prim.num_cells)
            )
            np.testing.assert_equal(source_sec_known, -3.56 * np.ones(sd_sec.num_cells))
        elif model.params["case"] == 1:
            np.testing.assert_equal(source_prim_known, 0 * np.ones(sd_prim.num_cells))
            np.testing.assert_equal(source_sec_known, 0 * np.ones(sd_sec.num_cells))
        elif model.params["case"] == 2:
            np.testing.assert_equal(
                source_prim_known, 3.51 * np.ones(sd_prim.num_cells)
            )
            np.testing.assert_equal(source_sec_known, 6.31 * np.ones(sd_sec.num_cells))
        else:
            raise NotImplementedError("Test case not implemented.")

    # def test_non_default_contact_mechanics_biot(self):
    #     """Test no-default parameters before and after running the model."""
    #
    #     # Construct model
    #     class NonDefaultModel(pp.ContactMechanicsBiot):
    #         """Model containing non-default parameters"""
    #
    #         def __init__(self, params: dict):
    #             """Constructor for the NonDefaultModel class"""
    #             super().__init__(params)
    #
    #         def create_grid(self) -> None:
    #             """Modify default grid.
    #
    #             Create a Cartesian grid in a unit square with two cells in the horizontal
    #             direction and one cell in the vertical direction. A single vertical fracture
    #             is included in the middle of the domain.
    #
    #             """
    #
    #             self.mdg, self.box = pp.md_grids_2d.single_vertical(
    #                 mesh_args=np.array([2, 1]), simplex=False
    #             )
    #             pp.contact_conditions.set_projections(self.mdg)
    #
    #         # ----------> Override flow parameters
    #         def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
    #             """Modify default boundary conditions type"""
    #             if sd.dim == 2:
    #                 all_bc, east, west, *_ = self._domain_boundary_sides(sd)
    #                 east_bc = np.isin(all_bc, np.where(east)).nonzero()
    #                 west_bc = np.isin(all_bc, np.where(west)).nonzero()
    #                 bc_type = np.asarray(all_bc.size * ["dir"])
    #                 bc_type[east_bc] = "neu"
    #                 bc_type[west_bc] = "neu"
    #                 bc = pp.BoundaryCondition(sd, faces=all_bc, cond=bc_type)
    #             else:
    #                 all_bf, *_ = self._domain_boundary_sides(sd)
    #                 bc = pp.BoundaryCondition(sd, all_bf, "dir")
    #             return bc
    #
    #         def _bc_values_scalar(self, sd: pp.Grid) -> np.ndarray:
    #             """Modify default boundary conditions values"""
    #             all_bf, east, west, south, north, *_ = self._domain_boundary_sides(sd)
    #             if sd.dim == 2:
    #                 bc_values = np.zeros(sd.num_faces)
    #                 bc_values[east] = -0.01
    #                 bc_values[west] = 0.01
    #                 bc_values[south] = 1.0
    #                 bc_values[north] = 1.0
    #             else:
    #                 bc_values = np.ones(sd.num_faces)
    #             return bc_values
    #
    #         def _source_scalar(self, sd: pp.Grid) -> np.ndarray:
    #             """Modifiy default scalar source"""
    #             if sd.dim == 2:
    #                 return 5.3 * np.ones(sd.num_cells)
    #             else:
    #                 return 3.5 * np.ones(sd.num_cells)
    #
    #         def _storativity(self, sd: pp.Grid) -> np.ndarray:
    #             """Modify default storativity"""
    #             return 0.4 * np.ones(sd.num_cells)
    #
    #         def _aperture(self, sd: pp.Grid) -> np.ndarray:
    #             """Modify default aperture"""
    #             aperture = np.ones(sd.num_cells)
    #             if sd.dim < self.nd:
    #                 aperture *= 0.6
    #             return aperture
    #
    #         def _biot_alpha(self, sd: pp.Grid) -> Union[float, np.ndarray]:
    #             """Modify default Biot coefficient"""
    #             if sd.dim == 2:
    #                 biot_alpha = 0.15 * np.ones(sd.num_cells)
    #             else:
    #                 biot_alpha = 0.85 * np.ones(sd.num_cells)
    #             return biot_alpha
    #
    #         def _permeability(self, sd: pp.Grid) -> np.ndarray:
    #             """Modify default permeability"""
    #             if sd.dim == 2:
    #                 perm = 2.7 * np.ones(sd.num_cells)
    #             else:
    #                 perm = 0.06 * np.ones(sd.num_cells)
    #             return perm
    #
    #         def _viscosity(self, sd: pp.Grid) -> np.ndarray:
    #             """Modify default viscosity"""
    #             if sd.dim == 2:
    #                 viscosity = 0.63 * np.ones(sd.num_cells)
    #             else:
    #                 viscosity = 0.11 * np.ones(sd.num_cells)
    #             return viscosity
    #
    #     # Instantiate model
    #     model_params = {"use_ad": True}
    #     model = NonDefaultModel(model_params)
    #     model.prepare_simulation()
    #
    #     # Retrieve subdomain and data
    #     sd_primary = model.mdg.subdomains(dim=2)[0]
    #     data_primary = model.mdg.subdomain_data(sd_primary)
    #     sd_secondary = model.mdg.subdomains(dim=1)[0]
    #     data_secondary = model.mdg.subdomain_data(sd_secondary)
    #     intf = model.mdg.interfaces(dim=1)[0]
    #     data_intf = model.mdg.interface_data(intf)
    #
    #     # ----------> Check flow parameters
    #     flow_kw = model.scalar_parameter_key
    #
    #     # -----> Boundary condition type
    #     bc_flow_primary = data_primary[pp.PARAMETERS][flow_kw]["bc"]
    #     _, east, west, south, north, *_ = model._domain_boundary_sides(sd_primary)
    #     bc_flow_secondary = data_secondary[pp.PARAMETERS][flow_kw]["bc"]
    #     # Check bc type of primary grid
    #     assert not np.all(bc_flow_primary.is_dir[east])
    #     assert np.all(bc_flow_primary.is_neu[east])
    #     assert not np.all(bc_flow_primary.is_rob[east])
    #     assert not np.all(bc_flow_primary.is_dir[west])
    #     assert np.all(bc_flow_primary.is_neu[west])
    #     assert not np.all(bc_flow_primary.is_rob[west])
    #     assert np.all(bc_flow_primary.is_dir[south])
    #     assert not np.all(bc_flow_primary.is_neu[south])
    #     assert not np.all(bc_flow_primary.is_rob[south])
    #     assert np.all(bc_flow_primary.is_dir[north])
    #     assert not np.all(bc_flow_primary.is_neu[north])
    #     assert not np.all(bc_flow_primary.is_rob[north])
    #     # Check bc type of secondary grid
    #     assert np.all(bc_flow_secondary.is_dir)
    #     assert not np.all(bc_flow_secondary.is_neu)
    #     assert not np.all(bc_flow_secondary.is_rob)
    #
    #     # -----> Boundary condition values
    #     bc_vals_primary = data_primary[pp.PARAMETERS][flow_kw]["bc_values"]
    #     _, east, west, south, north, *_ = model._domain_boundary_sides(sd_primary)
    #     bc_vals_secondary = data_secondary[pp.PARAMETERS][flow_kw]["bc_values"]
    #     # Check bc values of primary grid
    #     np.testing.assert_equal(bc_vals_primary[east], np.array([-0.01]))
    #     np.testing.assert_equal(bc_vals_primary[west], np.array([0.01]))
    #     np.testing.assert_equal(bc_vals_primary[south], np.array([1.0, 1.0]))
    #     np.testing.assert_equal(bc_vals_primary[north], np.array([1.0, 1.0]))
    #     # Check bc values of secondary grid
    #     np.testing.assert_equal(bc_vals_secondary, np.array([1.0, 1.0]))
    #
    #     # -----> Mass weights
    #     mass_weight_primary = data_primary[pp.PARAMETERS][flow_kw]["mass_weight"]
    #     mass_weight_secondary = data_secondary[pp.PARAMETERS][flow_kw]["mass_weight"]
    #     mwp = 0.4 * 1.0 * 1.0
    #     mws = 0.4 * np.power(0.6, 2 - 1) * 1.0
    #     np.testing.assert_equal(mass_weight_primary, np.array([mwp, mwp]))
    #     np.testing.assert_equal(mass_weight_secondary, np.array([mws]))
    #
    #     # -----> Biot-alpha
    #     biot_alpha_flow_primary = data_primary[pp.PARAMETERS][flow_kw]["biot_alpha"]
    #     biot_alpha_flow_secondary = data_secondary[pp.PARAMETERS][flow_kw]["biot_alpha"]
    #     np.testing.assert_equal(biot_alpha_flow_primary, np.array([0.15, 0.15]))
    #     np.testing.assert_equal(biot_alpha_flow_secondary, np.array([0.85]))
    #
    #     # -----> Source
    #     source_primary = data_primary[pp.PARAMETERS][flow_kw]["source"]
    #     source_secondary = data_secondary[pp.PARAMETERS][flow_kw]["source"]
    #     np.testing.assert_equal(source_primary, 5.3 * np.ones(sd_primary.num_cells))
    #     np.testing.assert_equal(source_secondary, 3.5 * np.ones(sd_secondary.num_cells))
    #
    #     # -----> Diffusivity
    #     sot_prim_known = data_primary[pp.PARAMETERS][flow_kw]["second_order_tensor"].values
    #     kp = (2.7 / 0.63) * 1.0 * np.ones(sd_primary.num_cells)
    #     sot_primary = pp.SecondOrderTensor(kp).values
    #     np.testing.assert_equal(sot_prim_known, sot_primary)
    #     sot_sec_known = data_secondary[pp.PARAMETERS][flow_kw]["second_order_tensor"].values
    #     ks = (0.06 / 0.11) * np.power(0.6, 2 - 1) * np.ones(sd_secondary.num_cells)
    #     sot_secondary = pp.SecondOrderTensor(ks).values
    #     np.testing.assert_equal(sot_sec_known, sot_secondary)
    #
    #     # -----> Normal diffusivity
    #     normal_diffu_known = data_intf[pp.PARAMETERS][flow_kw]["normal_diffusivity"]
    #     normal_diffu = 0
