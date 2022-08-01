"""
This file contains tests that were constructed to check the correct assingment of parameters
in standard models in PorePy.

For each model, two main methods are constructed: one that checks default parameters and
another that checks non-default parameters. For the latter, three different sets of
non-default parameters for a mixed-dimensional Cartesian grid containing a single vertical
fracture have been written.

Moreover, for each model, the parameters are checked at two stages, namely before running
the model and after running the model.

Note that only parameters belonging to the specific class are tested. This means that
inherited parameters are not tested as this would be too cumbersome (and redundant).
However, this naturally requires writing tests for all present models.

Todo:
    Create tests for the rest of the models.
"""

import porepy as pp
import pytest
import numpy as np

from typing import Union


class TestContactMechanicsBiot:
    """The following tests check the correct setup of the ContactMechanicsBiot class."""

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
    data_intf = mdg.interface_data(intf)

    # -----> Time parameters
    time_steps = [2.0, 5.0, 100.0]
    end_times = [2.0, 10.0, 300.0]

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

    # Case 1: All Neumann, except east and west which are Dirichlet
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

    # -----> Storativity
    storativity_primary = [np.zeros(2), 0.6 * np.ones(2), 0.005 * np.ones(2)]
    storativity_secondary = [np.array([0.0]), np.array([0.55]), np.array([1.30])]

    # -----> Aperture
    aperture_primary = [np.ones(2), np.ones(2), np.ones(2)]
    aperture_secondary = [np.array([0.05]), np.array([0.001]), np.array([0.9])]

    # -----> Biot alpha
    biot_alpha_primary = [0.9 * np.ones(2), 0.5 * np.ones(2), 0.0001 * np.ones(2)]
    biot_alpha_secondary = [np.array([0.15]), np.array([0.55]), np.array([0.006])]

    # -----> Permeability
    permeability_primary = [1e-1 * np.ones(2), 1e-5 * np.ones(2), 1e-8 * np.ones(2)]
    permeability_secondary = [np.array([1e-8]), np.array([1e-5]), np.array([1e-1])]

    # -----> Viscosity
    viscosity_primary = [4970 * np.ones(2), 0.76 * np.ones(2), 0.00001 * np.ones(2)]
    viscosity_secondary = [np.array([0.01]), np.array([5.3]), np.array([1450])]

    # -----> Vector source
    vector_source_primary_0 = np.zeros((mdg.dim_max(), sd_prim.num_cells))
    vector_source_primary_0[-1] = -pp.GRAVITY_ACCELERATION * 1014
    vector_source_primary_0 = np.ravel(vector_source_primary_0, "F")

    vector_source_primary_1 = np.zeros((mdg.dim_max(), sd_prim.num_cells))
    vector_source_primary_1[-1] = -1.0
    vector_source_primary_1 = np.ravel(vector_source_primary_1, "F")

    vector_source_primary_2 = np.zeros((mdg.dim_max(), sd_prim.num_cells))
    vector_source_primary_2[-1] = -0.0005
    vector_source_primary_2 = np.ravel(vector_source_primary_2, "F")

    vector_source_primary = [
        vector_source_primary_0,
        vector_source_primary_1,
        vector_source_primary_2,
    ]

    vector_source_secondary_0 = np.zeros((mdg.dim_max(), sd_sec.num_cells))
    vector_source_secondary_0[-1] = -pp.GRAVITY_ACCELERATION * 1014
    vector_source_secondary_0 = np.ravel(vector_source_secondary_0, "F")

    vector_source_secondary_1 = np.zeros((mdg.dim_max(), sd_sec.num_cells))
    vector_source_secondary_1[-1] = -1.0
    vector_source_secondary_1 = np.ravel(vector_source_secondary_1, "F")

    vector_source_secondary_2 = np.zeros((mdg.dim_max(), sd_sec.num_cells))
    vector_source_secondary_2[-1] = -0.0005
    vector_source_secondary_2 = np.ravel(vector_source_secondary_2, "F")

    vector_source_secondary = [
        vector_source_secondary_0,
        vector_source_secondary_1,
        vector_source_secondary_2,
    ]

    vector_source_interface_0 = np.zeros((mdg.dim_max(), intf.num_cells))
    vector_source_interface_0[-1] = -pp.GRAVITY_ACCELERATION * 1014
    vector_source_interface_0 = np.ravel(vector_source_interface_0, "F")

    vector_source_interface_1 = np.zeros((mdg.dim_max(), intf.num_cells))
    vector_source_interface_1[-1] = -1.0
    vector_source_interface_1 = np.ravel(vector_source_interface_1, "F")

    vector_source_interface_2 = np.zeros((mdg.dim_max(), intf.num_cells))
    vector_source_interface_2[-1] = -0.0005
    vector_source_interface_2 = np.ravel(vector_source_interface_2, "F")

    vector_source_interface = [
        vector_source_interface_0,
        vector_source_interface_1,
        vector_source_interface_2,
    ]

    # -----> Mechanics bc type
    bc_init_prim = ["neu"] * sd_prim.num_faces
    bc_type_prim = np.array([bc_init_prim, bc_init_prim])

    # Case 0: All Neumann, except east, west, and north which are Dirichlet
    bc_type_prim_0 = bc_type_prim.copy()
    bc_type_prim_0[:, east] = "dir"
    bc_type_prim_0[:, west] = "dir"
    bc_type_prim_0[:, north] = "dir"

    # Case 1: All Neumann, except east and west, and south which are Dirichlet
    bc_type_prim_1 = bc_type_prim.copy()
    bc_type_prim_1[:, east] = "dir"
    bc_type_prim_1[:, west] = "dir"
    bc_type_prim_1[:, south] = "dir"

    # Case 2: All Neumann, except east and west which are rollers, and south and north
    # Dirichlet
    bc_type_prim_2 = bc_type_prim.copy()
    bc_type_prim_2[0, east] = "dir"
    bc_type_prim_2[0, west] = "dir"
    bc_type_prim_2[:, north] = "dir"
    bc_type_prim_2[:, south] = "dir"

    bc_mech_type_primary = [bc_type_prim_0, bc_type_prim_1, bc_type_prim_2]

    # -----> Mechanics bc values
    zeros = np.zeros(sd_prim.num_faces)

    # Case 0
    bc_val_prim_0 = np.array([zeros, zeros])
    bc_val_prim_0[0, east] = 0.01
    bc_val_prim_0[0, west] = -0.01
    bc_val_prim_0[1, north] = 0.01
    bc_val_prim_0 = np.ravel(bc_val_prim_0, "F")

    # Case 1
    bc_val_prim_1 = np.array([zeros, zeros])
    bc_val_prim_1[0, east] = -0.01
    bc_val_prim_1[0, west] = 0.01
    bc_val_prim_1[1, south] = -0.01
    bc_val_prim_1 = np.ravel(bc_val_prim_1, "F")

    # Case 2
    bc_val_prim_2 = np.array([zeros, zeros])
    bc_val_prim_2[0, east] = 0.01
    bc_val_prim_2[0, west] = -0.01
    bc_val_prim_2[1, south] = 0.1
    bc_val_prim_2[1, north] = -0.1
    bc_val_prim_2 = np.ravel(bc_val_prim_2, "F")

    bc_mech_values_primary = [bc_val_prim_0, bc_val_prim_1, bc_val_prim_2]

    # -----> Reference scalar
    reference_scalar = [np.ones(2), 2.0 * np.ones(2), 100.0 * np.ones(2)]

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
                "time_step": time_steps[case],
                "end_time": end_times[case],
                "bc_flow_type_primary": bc_flow_type_primary[case],
                "bc_flow_type_secondary": bc_flow_type_secondary[case],
                "bc_flow_values_primary": bc_flow_values_primary[case],
                "bc_flow_values_secondary": bc_flow_values_secondary[case],
                "source_flow_primary": source_flow_primary[case],
                "source_flow_secondary": source_flow_secondary[case],
                "storativity_primary": storativity_primary[case],
                "storativity_secondary": storativity_secondary[case],
                "aperture_primary": aperture_primary[case],
                "aperture_secondary": aperture_secondary[case],
                "biot_alpha_primary": biot_alpha_primary[case],
                "biot_alpha_secondary": biot_alpha_secondary[case],
                "permeability_primary": permeability_primary[case],
                "permeability_secondary": permeability_secondary[case],
                "viscosity_primary": viscosity_primary[case],
                "viscosity_secondary": viscosity_secondary[case],
                "vector_source_primary": vector_source_primary[case],
                "vector_source_secondary": vector_source_secondary[case],
                "vector_source_interface": vector_source_interface[case],
                "reference_scalar": reference_scalar[case],
                "bc_mech_type_primary": bc_mech_type_primary[case],
                "bc_mech_values_primary": bc_mech_values_primary[case],
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

            def _storativity(self, sd: pp.Grid) -> np.ndarray:
                """Modifity default storativity"""
                if sd.dim == 2:
                    return self.params["storativity_primary"]
                else:
                    return self.params["storativity_secondary"]

            def _aperture(self, sd: pp.Grid) -> np.ndarray:
                """Modify default aperture"""
                if sd.dim == 2:
                    return self.params["aperture_primary"]
                else:
                    return self.params["aperture_secondary"]

            def _biot_alpha(self, sd: pp.Grid) -> Union[float, np.array]:
                """Modify default biot alpha coefficient"""
                if sd.dim == 2:
                    return self.params["biot_alpha_primary"]
                else:
                    return self.params["biot_alpha_secondary"]

            def _permeability(self, sd: pp.Grid) -> np.ndarray:
                """Modify default permeability"""
                if sd.dim == 2:
                    return self.params["permeability_primary"]
                else:
                    return self.params["permeability_secondary"]

            def _viscosity(self, sd: pp.Grid) -> np.ndarray:
                """Modify default viscosity"""
                if sd.dim == 2:
                    return self.params["viscosity_primary"]
                else:
                    return self.params["viscosity_secondary"]

            def _vector_source(self, g: Union[pp.Grid, pp.MortarGrid]) -> np.ndarray:
                """Modify default vector source"""
                if isinstance(g, pp.Grid):
                    if g.dim == 2:
                        return self.params["vector_source_primary"]
                    else:
                        return self.params["vector_source_secondary"]
                else:
                    return self.params["vector_source_interface"]

            def _reference_scalar(self, sd: pp.Grid) -> np.ndarray:
                """Modify default reference scalar"""
                super()._reference_scalar(sd)
                if sd.dim == 2:
                    return self.params["reference_scalar"]

            def _bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
                """Modifiy default boundary condition types for the mechanics subproblem."""
                super()._bc_type_mechanics(sd)
                data = self.mdg.subdomain_data(sd)
                bc = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc"]

                if sd.dim == 2:
                    is_dir = self.params["bc_mech_type_primary"] == "dir"
                    is_neu = self.params["bc_mech_type_primary"] == "neu"
                    is_rob = self.params["bc_mech_type_primary"] == "rob"
                    bc.is_dir = is_dir
                    bc.is_neu = is_neu
                    bc.is_rob = is_rob

                return bc

            def _bc_values_mechanics(self, sd: pp.Grid) -> np.ndarray:
                """Modify default boundary condition values for the mechanics subproblem"""
                super()._bc_values_mechanics(sd)
                if sd.dim == 2:
                    return self.params["bc_mech_values_primary"]

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

        # -----> Check time attributes
        if model.params["case"] == 0:
            if model.params["before_run"]:
                assert model.time == 0.0
            else:
                assert model.time == 2.0
            assert model.time_step == 2.0
            assert model.end_time == 2.0
        elif model.params["case"] == 1:
            if model.params["before_run"]:
                assert model.time == 0.0
            else:
                assert model.time == 10.0
            assert model.time_step == 5.0
            assert model.end_time == 10.0
        elif model.params["case"] == 2:
            if model.params["before_run"]:
                assert model.time == 0.0
            else:
                assert model.time == 300.0
            assert model.time_step == 100.0
            assert model.end_time == 300.0
        else:
            raise NotImplementedError("Test case not implemented")

        # ----------> Check parameters for the scalar subproblem

        # -----> Time step
        if model.params["case"] == 0:
            assert data_prim[pp.PARAMETERS][flow_kw]["time_step"] == 2.0
            assert data_sec[pp.PARAMETERS][flow_kw]["time_step"] == 2.0
        elif model.params["case"] == 1:
            assert data_prim[pp.PARAMETERS][flow_kw]["time_step"] == 5.0
            assert data_sec[pp.PARAMETERS][flow_kw]["time_step"] == 5.0
        elif model.params["case"] == 2:
            assert data_prim[pp.PARAMETERS][flow_kw]["time_step"] == 100.0
            assert data_sec[pp.PARAMETERS][flow_kw]["time_step"] == 100.0
        else:
            raise NotImplementedError("Test case not implemented.")

        # -----> Ambient dimension
        assert data_prim[pp.PARAMETERS][flow_kw]["ambient_dimension"] == 2.0
        assert data_sec[pp.PARAMETERS][flow_kw]["ambient_dimension"] == 2.0

        # -----> Boundary condition type
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

        # -----> Boundary condition values
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

        # -----> Mass weights
        mass_weight_prim = data_prim[pp.PARAMETERS][flow_kw]["mass_weight"]
        mass_weight_sec = data_sec[pp.PARAMETERS][flow_kw]["mass_weight"]
        if model.params["case"] == 0:
            mw_prim = 0.0 * 1.0 * 1.0
            mw_sec = 0.0 * np.power(0.05, 2 - 1) * 1.0
            np.testing.assert_equal(mass_weight_prim, np.array([mw_prim, mw_prim]))
            np.testing.assert_equal(mass_weight_sec, np.array([mw_sec]))
        elif model.params["case"] == 1:
            mw_prim = 0.6 * 1.0 * 1.0
            mw_sec = 0.55 * np.power(0.001, 2 - 1) * 1.0
            np.testing.assert_equal(mass_weight_prim, np.array([mw_prim, mw_prim]))
            np.testing.assert_equal(mass_weight_sec, np.array([mw_sec]))
        elif model.params["case"] == 2:
            mw_prim = 0.005 * 1.0 * 1.0
            mw_sec = 1.30 * np.power(0.9, 2 - 1) * 1.0
            np.testing.assert_equal(mass_weight_prim, np.array([mw_prim, mw_prim]))
            np.testing.assert_equal(mass_weight_sec, np.array([mw_sec]))
        else:
            raise NotImplementedError("Test case not implemented.")

        # -----> Biot-alpha
        biot_alpha_flow_prim = data_prim[pp.PARAMETERS][flow_kw]["biot_alpha"]
        biot_alpha_flow_sec = data_sec[pp.PARAMETERS][flow_kw]["biot_alpha"]
        if model.params["case"] == 0:
            np.testing.assert_equal(biot_alpha_flow_prim, np.array([0.9, 0.9]))
            np.testing.assert_equal(biot_alpha_flow_sec, np.array([0.15]))
        elif model.params["case"] == 1:
            np.testing.assert_equal(biot_alpha_flow_prim, np.array([0.5, 0.5]))
            np.testing.assert_equal(biot_alpha_flow_sec, np.array([0.55]))
        elif model.params["case"] == 2:
            np.testing.assert_equal(biot_alpha_flow_prim, np.array([0.0001, 0.0001]))
            np.testing.assert_equal(biot_alpha_flow_sec, np.array([0.006]))
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

        # -----> Vector source
        vector_source_prim = data_prim[pp.PARAMETERS][flow_kw]["vector_source"]
        vector_source_sec = data_sec[pp.PARAMETERS][flow_kw]["vector_source"]
        vector_source_intf = data_intf[pp.PARAMETERS][flow_kw]["vector_source"]
        if model.params["case"] == 0:
            val = -pp.GRAVITY_ACCELERATION * 1014
            np.testing.assert_equal(vector_source_prim, np.array([0, val, 0.0, val]))
            np.testing.assert_equal(vector_source_sec, np.array([0.0, val]))
            np.testing.assert_equal(vector_source_intf, np.array([0.0, val, 0.0, val]))
        elif model.params["case"] == 1:
            val = -1.0
            np.testing.assert_equal(vector_source_prim, np.array([0, val, 0.0, val]))
            np.testing.assert_equal(vector_source_sec, np.array([0.0, val]))
            np.testing.assert_equal(vector_source_intf, np.array([0.0, val, 0.0, val]))
        elif model.params["case"] == 2:
            val = -0.0005
            np.testing.assert_equal(vector_source_prim, np.array([0, val, 0.0, val]))
            np.testing.assert_equal(vector_source_sec, np.array([0.0, val]))
            np.testing.assert_equal(vector_source_intf, np.array([0.0, val, 0.0, val]))
        else:
            raise NotImplementedError("Test case not implemented.")

        # -----> Diffusivity
        diffusivity_prim = data_prim[pp.PARAMETERS][flow_kw][
            "second_order_tensor"
        ].values
        diffusivity_sec = data_sec[pp.PARAMETERS][flow_kw]["second_order_tensor"].values
        if model.params["case"] == 0:
            kappa_prim = (1e-1 * np.ones(2)) / (4970 * np.ones(2)) * 1.0
            np.testing.assert_equal(
                diffusivity_prim, pp.SecondOrderTensor(kappa_prim * 1.0).values
            )
            kappa_sec = np.array([1e-8]) / (np.array([0.01])) * 1.0
            np.testing.assert_equal(
                diffusivity_sec,
                pp.SecondOrderTensor(kappa_sec * np.power(0.05, 2 - 1)).values,
            )
        elif model.params["case"] == 1:
            kappa_prim = (1e-5 * np.ones(2)) / (0.76 * np.ones(2)) * 1.0
            np.testing.assert_equal(
                diffusivity_prim, pp.SecondOrderTensor(kappa_prim * 1.0).values
            )
            kappa_sec = np.array([1e-5]) / (np.array([5.3])) * 1.0
            np.testing.assert_equal(
                diffusivity_sec,
                pp.SecondOrderTensor(kappa_sec * np.power(0.001, 2 - 1)).values,
            )
        elif model.params["case"] == 2:
            kappa_prim = (1e-8 * np.ones(2)) / (0.00001 * np.ones(2)) * 1.0
            np.testing.assert_equal(
                diffusivity_prim, pp.SecondOrderTensor(kappa_prim * 1.0).values
            )
            kappa_sec = np.array([1e-1]) / (np.array([1450])) * 1.0
            np.testing.assert_equal(
                diffusivity_sec,
                pp.SecondOrderTensor(kappa_sec * np.power(0.9, 2 - 1)).values,
            )
        else:
            raise NotImplementedError("Test case not implemented")

        # -----> Normal diffusivity
        normal_diffu = data_intf[pp.PARAMETERS][flow_kw]["normal_diffusivity"]
        if model.params["case"] == 0:
            kappa_sec = 1e-8 / 0.01 * 1.0
            val = np.array([(kappa_sec * 2) / 0.05, (kappa_sec * 2) / 0.05])
            np.testing.assert_equal(normal_diffu, val)
        elif model.params["case"] == 1:
            kappa_sec = 1e-5 / 5.3 * 1.0
            val = np.array([(kappa_sec * 2) / 0.001, (kappa_sec * 2) / 0.001])
            np.testing.assert_equal(normal_diffu, val)
        elif model.params["case"] == 2:
            kappa_sec = 1e-1 / 1450 * 1.0
            val = np.array([(kappa_sec * 2) / 0.9, (kappa_sec * 2) / 0.9])
            np.testing.assert_equal(normal_diffu, val)
        else:
            raise NotImplementedError("Test case not implemented")

        # ----------> Check parameters for the mechanics subproblem

        # -----> Boundary condition type
        bc_mech_prim = data_prim[pp.PARAMETERS][model.mechanics_parameter_key]["bc"]

        if model.params["case"] == 0:  # east: dir, west: dir, north: dir, south: neu
            assert np.all(bc_mech_prim.is_dir[:, east])
            assert not np.all(bc_mech_prim.is_neu[:, east])
            assert not np.all(bc_mech_prim.is_rob[:, east])
            assert np.all(bc_mech_prim.is_dir[:, west])
            assert not np.all(bc_mech_prim.is_neu[:, west])
            assert not np.all(bc_mech_prim.is_rob[:, west])
            assert not np.all(bc_mech_prim.is_dir[:, south])
            assert np.all(bc_mech_prim.is_neu[:, south])
            assert not np.all(bc_mech_prim.is_rob[:, south])
            assert np.all(bc_mech_prim.is_dir[:, north])
            assert not np.all(bc_mech_prim.is_neu[:, north])
            assert not np.all(bc_mech_prim.is_rob[:, north])
        elif model.params["case"] == 1:  # east: dir, west: dir, north: neu, south: dir
            assert np.all(bc_mech_prim.is_dir[:, east])
            assert not np.all(bc_mech_prim.is_neu[:, east])
            assert not np.all(bc_mech_prim.is_rob[:, east])
            assert np.all(bc_mech_prim.is_dir[:, west])
            assert not np.all(bc_mech_prim.is_neu[:, west])
            assert not np.all(bc_mech_prim.is_rob[:, west])
            assert np.all(bc_mech_prim.is_dir[:, south])
            assert not np.all(bc_mech_prim.is_neu[:, south])
            assert not np.all(bc_mech_prim.is_rob[:, south])
            assert not np.all(bc_mech_prim.is_dir[:, north])
            assert np.all(bc_mech_prim.is_neu[:, north])
            assert not np.all(bc_mech_prim.is_rob[:, north])
        elif (
            model.params["case"] == 2
        ):  # east: roller, west: roller, north: dir, south: dir
            assert np.all(bc_mech_prim.is_dir[0, east])
            assert not np.all(bc_mech_prim.is_dir[1, east])
            assert not np.all(bc_mech_prim.is_neu[0, east])
            assert np.all(bc_mech_prim.is_neu[1, east])
            assert not np.all(bc_mech_prim.is_rob[:, east])
            assert np.all(bc_mech_prim.is_dir[0, west])
            assert not np.all(bc_mech_prim.is_dir[1, west])
            assert not np.all(bc_mech_prim.is_neu[0, west])
            assert np.all(bc_mech_prim.is_neu[1, west])
            assert not np.all(bc_mech_prim.is_rob[:, west])
            assert np.all(bc_mech_prim.is_dir[:, south])
            assert not np.all(bc_mech_prim.is_neu[:, south])
            assert not np.all(bc_mech_prim.is_rob[:, south])
            assert np.all(bc_mech_prim.is_dir[:, north])
            assert not np.all(bc_mech_prim.is_neu[:, north])
            assert not np.all(bc_mech_prim.is_rob[:, north])
        else:
            raise NotImplementedError("Test case not implemented.")

        # -----> Boundary condition values
        bc_mech_vals = data_prim[pp.PARAMETERS][model.mechanics_parameter_key][
            "bc_values"
        ]
        bc_mech_vals_x = bc_mech_vals[::2]
        bc_mech_vals_y = bc_mech_vals[1::2]
        if model.params["case"] == 0:
            np.testing.assert_equal(bc_mech_vals_x[east], 0.01)
            np.testing.assert_equal(bc_mech_vals_y[east], 0)
            np.testing.assert_equal(bc_mech_vals_x[west], -0.01)
            np.testing.assert_equal(bc_mech_vals_y[west], 0)
            np.testing.assert_equal(bc_mech_vals_x[south], 0)
            np.testing.assert_equal(bc_mech_vals_y[south], 0)
            np.testing.assert_equal(bc_mech_vals_x[north], 0)
            np.testing.assert_equal(bc_mech_vals_y[north], 0.01)
        elif model.params["case"] == 1:
            np.testing.assert_equal(bc_mech_vals_x[east], -0.01)
            np.testing.assert_equal(bc_mech_vals_y[east], 0)
            np.testing.assert_equal(bc_mech_vals_x[west], 0.01)
            np.testing.assert_equal(bc_mech_vals_y[west], 0)
            np.testing.assert_equal(bc_mech_vals_x[south], 0)
            np.testing.assert_equal(bc_mech_vals_y[south], -0.01)
            np.testing.assert_equal(bc_mech_vals_x[north], 0)
            np.testing.assert_equal(bc_mech_vals_y[north], 0)
        elif model.params["case"] == 2:
            np.testing.assert_equal(bc_mech_vals_x[east], 0.01)
            np.testing.assert_equal(bc_mech_vals_y[east], 0)
            np.testing.assert_equal(bc_mech_vals_x[west], -0.01)
            np.testing.assert_equal(bc_mech_vals_y[west], 0)
            np.testing.assert_equal(bc_mech_vals_x[south], 0)
            np.testing.assert_equal(bc_mech_vals_y[south], 0.1)
            np.testing.assert_equal(bc_mech_vals_x[north], 0)
            np.testing.assert_equal(bc_mech_vals_y[north], -0.1)
        else:
            raise NotImplementedError("Test case not implemented.")

        # -----> Time step
        if model.params["case"] == 0:
            assert data_prim[pp.PARAMETERS][mech_kw]["time_step"] == 2.0
            assert data_sec[pp.PARAMETERS][mech_kw]["time_step"] == 2.0
        elif model.params["case"] == 1:
            assert data_prim[pp.PARAMETERS][mech_kw]["time_step"] == 5.0
            assert data_sec[pp.PARAMETERS][mech_kw]["time_step"] == 5.0
        elif model.params["case"] == 2:
            assert data_prim[pp.PARAMETERS][mech_kw]["time_step"] == 100.0
            assert data_sec[pp.PARAMETERS][mech_kw]["time_step"] == 100.0
        else:
            raise NotImplementedError("Test case not implemented.")

        # -----> Biot-alpha
        biot_alpha_mech_prim = data_prim[pp.PARAMETERS][mech_kw]["biot_alpha"]
        if model.params["case"] == 0:
            np.testing.assert_equal(biot_alpha_mech_prim, np.array([0.9, 0.9]))
        elif model.params["case"] == 1:
            np.testing.assert_equal(biot_alpha_mech_prim, np.array([0.5, 0.5]))
        elif model.params["case"] == 2:
            np.testing.assert_equal(biot_alpha_mech_prim, np.array([0.0001, 0.0001]))
        else:
            raise NotImplementedError("Test case not implemented.")

        # -----> Reference scalar
        reference_scalar = data_prim[pp.PARAMETERS][mech_kw]["p_reference"]
        if model.params["case"] == 0:
            np.testing.assert_equal(reference_scalar, 1.0 * np.ones(2))
        elif model.params["case"] == 1:
            np.testing.assert_equal(reference_scalar, 2.0 * np.ones(2))
        elif model.params["case"] == 2:
            np.testing.assert_equal(reference_scalar, 100.0 * np.ones(2))
        else:
            raise NotImplementedError("Test case not implemented.")

        # -----> Mass weight
        mass_weight_sec = data_sec[pp.PARAMETERS][mech_kw]["mass_weight"]
        np.testing.assert_equal(mass_weight_sec, np.array([1.0]))
