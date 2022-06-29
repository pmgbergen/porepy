"""Tests of IncompressibleFlow model.

The tests focus on model setup/preparation. While some of this may seem trivial,
an important point is that the tests check that no additional data has been
initialized. This is to reveal errenous inheritance between model classes. This
may be even more useful for other models with more complicated inheritance. TODO:
extend to other models?
"""
import numpy as np
import pytest

import porepy as pp


def compare_keywords(l1, l2):
    l1 = list(l1).sort()
    l2 = list(l2).sort()
    assert l1 == l2


def compare_dicts(d1, d2):
    """Check that the dictionaries have the same set of keys and that their
    items agree if type is np.ndarray
    """
    l1, l2 = d1.keys(), d2.keys()
    compare_keywords(l1, l2)
    for k1, k2 in zip(l1, l2):
        it1 = d1[k1]
        if isinstance(it1, np.ndarray):
            assert np.all(np.isclose(it1, d2[k2]))


class FlowModel(pp.IncompressibleFlow):
    def create_grid(self):
        if self.params["n_fracs"] == 1:
            self.mdg, self.box = pp.grid_buckets_2d.single_horizontal()
        elif self.params["n_fracs"] == 2:
            self.mdg, self.box = pp.grid_buckets_2d.two_intersecting()
        pp.contact_conditions.set_projections(self.mdg)


def test_incompressible_flow_model_no_modification():
    """Test that the raw incompressible flow model with no modifications can be run with
    no error messages. Failure of this test would signify rather fundamental problems
    in the model.
    """
    model = pp.IncompressibleFlow({"use_ad": True})
    pp.run_stationary_model(model, {})


@pytest.mark.parametrize(
    "n_fracs",
    [
        1,  # A single horizontal fracture
        2,  # Two intersecting fractures
    ],
)
def test_prepare_simulation(model, n_fracs):
    """Test that the correct entities are initialized in data dictionaries."""
    params = {"n_fracs": n_fracs}
    model = FlowModel(params)
    model.prepare_simulation()

    assert hasattr(model, "params")
    assert model._use_ad
    assert not model._is_nonlinear_problem()
    assert model._nonlinear_iteration == 0
    assert model.convergence_status is False

    num_dofs = 0
    # Loop over dictionaries and assert that the correct sets of variables,
    # parameters, initial values are present
    for g, d in model.mdg:
        var_list = d[pp.PRIMARY_VARIABLES].keys()
        compare_keywords(var_list, ["p"])

        known_initial = {"p": np.zeros(g.num_cells)}
        state = d[pp.STATE].copy().pop(pp.ITERATE)
        compare_dicts(state, known_initial)
        compare_dicts(d[pp.STATE][pp.ITERATE], known_initial)

        param_list = [
            "bc",
            "bc_values",
            "vector_source",
            "source",
            "second_order_tensor",
            "ambient_dimension",
        ]
        if g.dim > 0:
            param_list += [
                "active_cells",
                "active_faces",
            ]
        compare_keywords(d[pp.PARAMETERS]["flow"].keys(), param_list)

        num_dofs += g.num_cells

    for e, d in model.mdg.edges():
        var_list = list(d[pp.PRIMARY_VARIABLES].keys())
        compare_keywords(var_list, ["mortar_p"])

        mg = d["mortar_grid"]
        known_initial = {"mortar_p": np.zeros(mg.num_cells)}
        state = d[pp.STATE].copy().pop(pp.ITERATE)
        compare_dicts(state, known_initial)
        compare_dicts(d[pp.STATE][pp.ITERATE], known_initial)

        param_list = ["vector_source", "normal_diffusivity", "ambient_dimension"]
        compare_keywords(d[pp.PARAMETERS]["flow"].keys(), param_list)

        num_dofs += mg.num_cells

    assert model.dof_manager.num_dofs() == num_dofs
    equations = ["subdomain_flow", "interface_flow"]
    compare_keywords(model._eq_manager.equations.keys(), equations)


@pytest.mark.parametrize(
    "n_fracs",
    [
        1,  # A single horizontal fracture
        2,  # Two intersecting fractures
    ],
)
def test_dimension_reduction(model, n_fracs):
    """Test that expected aperture and specific volumes are returned."""
    params = {"n_fracs": n_fracs}
    model = FlowModel(params)
    model.prepare_simulation()
    for g, d in model.mdg:
        aperture = np.ones(g.num_cells)
        if g.dim < model.mdg.dim_max():
            aperture *= 0.1
        assert np.all(np.isclose(model._aperture(g), aperture))
        specific_volume = np.power(aperture, model.mdg.dim_max() - g.dim)
        assert np.all(np.isclose(model._specific_volume(g), specific_volume))


@pytest.fixture
def model():
    # Method to deliver a model to all tests
    return FlowModel()
