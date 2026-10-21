"""Unit tests for the ModelStorage class."""

from __future__ import annotations

import numpy as np
import pytest

from porepy.applications.initialization.model_state_storage import (
    LoadedStateMismatch,
    ModelStateStorage,
)
from tests.applications.initialization.test_initialization import make_model


@pytest.mark.parametrize(
    "corruption",
    [
        "none",
        "missing_array",
        "wrong_shape",
        "extra_array",
        "non_finite",
        "unreadable",
    ],
)
def test_state_storage(tmp_path, corruption):
    """A saved state is loaded into the model exactly as it was saved.

    Inconsistent states are rejected, and leave the model unchanged.

    """
    model = make_model(tmp_path / "simulation")
    model.prepare_simulation()
    system = model.equation_system
    # A state in which the current, previous and reference values all differ.
    rng = np.random.default_rng(0)
    system.set_variable_values(rng.random(system.num_dofs()), reference=True)
    system.set_variable_values(rng.random(system.num_dofs()), time_step_index=0)
    system.set_variable_values(rng.random(system.num_dofs()), iterate_index=0)
    storage = ModelStateStorage(tmp_path / "state.npz")
    storage.save_state(model)
    saved = storage.capture_state(model)

    path = storage.path
    arrays = dict(np.load(path))
    key = next(k for k in arrays if "::iterate_solutions::" in k and "sd" in k)
    if corruption == "missing_array":
        del arrays[key]
    elif corruption == "wrong_shape":
        arrays[key] = arrays[key][:-1]
    elif corruption == "extra_array":
        arrays["extra"] = np.zeros(1)
    elif corruption == "non_finite":
        arrays[key][0] = np.nan
    if corruption != "none":
        np.savez(path, **arrays)
    if corruption == "unreadable":
        path.write_bytes(b"not an archive")

    # Change the state of the model, to see that it is loaded or not changed.
    for kwargs in [{"reference": True}, {"time_step_index": 0}, {"iterate_index": 0}]:
        system.set_variable_values(rng.random(system.num_dofs()), **kwargs)
    changed = storage.capture_state(model)
    if corruption == "none":
        storage.load_state(model)
        # Loaded successfully, so the original state should be restored.
        expected = saved
    else:
        with pytest.raises(LoadedStateMismatch):
            storage.load_state(model)
        # Could not load the state, so it should be unmodified from "changed".
        expected = changed
    result = storage.capture_state(model)
    assert result.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(result[name], expected[name], err_msg=name)
