"""Saving and loading of the state of a model as numpy arrays.

The state of a model is everything that is stored in the data dictionaries of its
subdomains, interfaces and boundary grids as iterate values, time step values and
reference values: the values of the variables, the boundary values, the time dependent
arrays and the reference quantities (e.g. the reference stress).

Note:
    The functionality partly overlaps with saving and restarting through
    :class:`~porepy.viz.exporter.Exporter` and the data saving mixin of the model
    (``reference_state.npz``, ``load_data_from_pvd``). This module is not a replacement
    for it, since it serves a different purpose:

    - A restart resumes a time history: it reads the last exported vtu files, converts
      them from SI units, and sets the time, the time step size and the step counter of
      the model. Here, the state is a single snapshot, and the time manager is not
      touched.
    - A restart needs the exporter and a pvd history, and the values pass through vtu
      files in SI units. Here, the arrays are stored in the units of the model.

    The storage is meant to be used in the initialization sub-package only. The file
    layout is not a stable format. Talk to the maintainers if you intend to use it in
    your own code.

"""

import logging
from pathlib import Path

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)

__all__ = [
    "LoadedStateMismatch",
    "ModelStateStorage",
]


class LoadedStateMismatch(ValueError):
    """The saved state does not match the model: it is unreadable, or the layout or the
    shapes of its arrays differ from those of the model."""


class ModelStateStorage:
    """Saves the state of a model, and loads it back unchanged.

    The file is a numpy ``.npz`` archive with one array per value stored in the data
    dictionaries of the grids of the md-grid. The name of an array is
    ``<grid>::<location>::<name>[::<index>]``, where

    - ``<grid>`` is ``sd<i>``, ``intf<i>`` or ``bg<i>`` for the ``i``-th subdomain,
      interface or boundary grid of the md-grid,
    - ``<location>`` is the key of the data dictionary, i.e. ``pp.ITERATE_SOLUTIONS``,
      ``pp.TIME_STEP_SOLUTIONS`` or ``pp.REFERENCE_SOLUTIONS``,
    - ``<name>`` is the name of the quantity, e.g. a variable or a boundary value,
    - ``<index>`` is the iterate or time step index. Reference values have none.

    The consistency of a saved state with a model is checked by the names and shapes of
    the arrays only, which must be those of the data dictionaries of the model.

    When loading, the stored values of the model are replaced by the saved ones.
    Nothing else is changed. In particular, the iterates and time steps are not copied
    to each other, and the reference values are not derived from the current values.
    Derived quantities of the model (e.g. fluxes) are updated, since they are computed
    from the state and are not saved.

    Parameters:
        path: File to save to and load from, usually in the simulation folder.

    """

    _locations = (pp.ITERATE_SOLUTIONS, pp.TIME_STEP_SOLUTIONS, pp.REFERENCE_SOLUTIONS)
    """Keys of the data dictionaries that make the state."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def discard(self) -> None:
        """Delete the saved state."""
        self.path.unlink(missing_ok=True)

    def _entries(
        self, model: pp.PorePyModel
    ) -> dict[str, tuple[dict, str, str, int | None]]:
        """Locate the stored values of the model.

        Returns:
            The key in the archive mapped to the data dictionary, the location, the
            name of the quantity and the index (None for reference values).

        """
        mdg = model.mdg
        groups = {
            "sd": mdg.subdomains(return_data=True),
            "intf": mdg.interfaces(return_data=True),
            "bg": mdg.boundaries(return_data=True),
        }
        entries: dict[str, tuple[dict, str, str, int | None]] = {}
        for group, items in groups.items():
            for i, (_, data) in enumerate(items):
                for location in self._locations:
                    for name, stored in data.get(location, {}).items():
                        prefix = f"{group}{i}::{location}::{name}"
                        if location == pp.REFERENCE_SOLUTIONS:
                            entries[prefix] = (data, location, name, None)
                        else:
                            for index in stored:
                                entries[f"{prefix}::{index}"] = (
                                    data,
                                    location,
                                    name,
                                    index,
                                )
        return entries

    @staticmethod
    def _get(entry: tuple[dict, str, str, int | None]) -> np.ndarray:
        data, location, name, index = entry
        stored = data[location][name]
        return np.array(stored if index is None else stored[index])

    def capture_state(self, model: pp.PorePyModel) -> dict[str, np.ndarray]:
        """Collect the arrays defining the current state of the model.

        Returns:
            The arrays, with the names described in the class documentation.

        """
        return {key: self._get(e) for key, e in self._entries(model).items()}

    def save_state(self, model: pp.PorePyModel) -> None:
        """Save the state of the model."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **self.capture_state(model))  # type: ignore[arg-type]
        logger.info(f"Saved state to {self.path}.")

    def load_state(self, model: pp.PorePyModel) -> None:
        """Load the saved state into the model.

        Raises:
            LoadedStateMismatch: If the file is unreadable, or the names or shapes of
                the arrays are inconsistent with the model, or the values are not
                finite. The model is not changed in that case.

        """
        try:
            with np.load(self.path) as archive:
                arrays = {key: archive[key] for key in archive.files}
        except (OSError, ValueError) as e:
            raise LoadedStateMismatch(f"Cannot read {self.path}: {e}") from e
        self._check_consistency(model, arrays)
        self.apply_state(model, arrays)

    def _check_consistency(
        self, model: pp.PorePyModel, arrays: dict[str, np.ndarray]
    ) -> None:
        def fail(message: str) -> None:
            raise LoadedStateMismatch(f"Saved state is inconsistent: {message}")

        entries = self._entries(model)
        if set(arrays) != set(entries):
            fail(f"arrays differ by {sorted(set(arrays) ^ set(entries))}")
        for key, entry in entries.items():
            if arrays[key].shape != self._get(entry).shape:
                fail(f"shape of {key}")
        floats = [a for a in arrays.values() if a.dtype.kind == "f"]
        if not all(np.all(np.isfinite(a)) for a in floats):
            fail("values are not finite")

    def apply_state(self, model: pp.PorePyModel, arrays: dict[str, np.ndarray]) -> None:
        """Replace the stored values of the model by the arrays.

        Parameters:
            model: A prepared model. The arrays must be consistent with it, see
                :meth:`load_state`.
            arrays: Arrays as returned by :meth:`capture_state`.

        """
        for key, (data, location, name, index) in self._entries(model).items():
            if index is None:
                data[location][name] = arrays[key].copy()
            else:
                data[location][name][index] = arrays[key].copy()
        # Fluxes, upwind and other discretizations depend on the state.
        model.update_derived_quantities()
