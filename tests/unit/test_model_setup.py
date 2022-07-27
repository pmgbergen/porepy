import porepy as pp
import numpy as np
import pytest


class TestDefaultContactMechanicsBiot:
    """The following tests are written to check the default parameters of the model."""

    def test_default_contact_mechanics_biot_before_run(self):
        """Test the default parameters before running the model"""
        model_params = {"use_ad": True}
        model = pp.ContactMechanicsBiot(model_params)
        model.prepare_simulation()
        sd = model.mdg.subdomains()[0]
        data = model.mdg.subdomain_data(sd)

        # Check attributes
        assert model.time == 0.0
        assert model.time_step == 1.0
        assert model.end_time == 1.0
        assert model.scalar_variable == "p"
        assert model.mortar_scalar_variable == "mortar_p"
        assert model.scalar_coupling_term == "robin_p"
        assert model.scalar_parameter_key == "flow"
        assert model.scalar_scale == 1.0
        assert model.length_scale == 1.0
        assert model.subtract_fracture_pressure

        # ----------> Check mechanics parameters
        # Boundary condition type
        bc_mech = data[pp.PARAMETERS][model.mechanics_parameter_key]["bc"]
        assert np.all(bc_mech.is_dir)
        assert not np.all(bc_mech.is_neu)
        assert not np.all(bc_mech.is_rob)
        # Boundary condition values
        bc_mech_vals = data[pp.PARAMETERS][model.mechanics_parameter_key]["bc_values"]
        np.testing.assert_equal(bc_mech_vals, np.zeros(sd.dim * sd.num_faces))
        # Stifness tensor
        stiff = data[pp.PARAMETERS][model.mechanics_parameter_key][
            "fourth_order_tensor"
        ]
        np.testing.assert_equal(stiff.mu, np.array([1.0]))
        np.testing.assert_equal(stiff.lmbda, np.array([1.0]))
        np.testing.assert_equal(
            stiff.values,
            np.array(
                [
                    [[3.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [1.0]],
                    [[0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
                    [[0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0]],
                    [[0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
                    [[1.0], [0.0], [0.0], [0.0], [3.0], [0.0], [0.0], [0.0], [1.0]],
                    [[0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [1.0], [0.0]],
                    [[0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0]],
                    [[0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [1.0], [0.0]],
                    [[1.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [3.0]],
                ]
            ),
        )

        # # Check flow bc type
        # bc_flow = data[pp.PARAMETERS][model.scalar_parameter_key]["bc"]
        # assert np.all(bc_flow.is_dir)
        # assert not np.all(bc_mech.is_neu)
        # assert not np.all(bc_mech.is_rob)
        # # Check flow bc values
        # bc_flow_vals = data[pp.PARAMETERS][model.scalar_parameter_key]["bc_values"]
        # np.testing.assert_equal(bc_flow_vals, np.zeros(sd.num_faces))
        #
        # # Check stifness tesnor
        # pass
