"""Saving and loading of the state of a model as numpy arrays. The state of a model is
everything that is stored in the data dictionaries of its grids.

The functionality partly overlaps with saving and restarting through
:class:`~porepy.viz.exporter.Exporter` and the DataSavingMixin.

We cannot reuse DataSavingMixin, because it is deeply coupled with the model lifecycle
hooks (before_* and after_*). The initialization procedure uses model beyond the
originally-intended lifecycle, which makes depending on DataSavingMixin error-prone.

We cannot use the Exporter, because it does not support boundary grids and the reference
stress (since it is defined not on the cell centers).

These arguments suggest that it is easier to repeat yourself and write a tailored
exporter/importer for initialization than to extend the existing Exporter. This tailored
class is not intended for ParaView exporting.

The storage is meant to be used in the initialization sub-package only. The file layout
is not a stable format. Talk to the maintainers if you intend to use it in your own
code.

"""

import logging
import re
from pathlib import Path
from typing import NoReturn, Sequence

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)

__all__ = [
    "LoadedStateMismatch",
    "ModelStateStorage",
]


class LoadedStateMismatch(ValueError):
    """The saved state does not match the model.

    The file is unreadable, or the layout or the shapes of its arrays differ from those
    of the model.

    """


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

    When loading, the stored values of the model are replaced by the saved ones, and
    derived quantities of the model (e.g. fluxes) are updated, since they are computed
    from the state and are not saved.

    Parameters:
        path: File to save to and load from, usually in the simulation folder.

    """

    _locations = (pp.ITERATE_SOLUTIONS, pp.TIME_STEP_SOLUTIONS, pp.REFERENCE_SOLUTIONS)
    """Keys of the data dictionaries that make the state."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        """File to save to and load from."""

    def exists(self) -> bool:
        """Check whether the file of the saved state exists.

        Returns:
            True if the file exists, False otherwise.

        """
        return self.path.is_file()

    def discard(self) -> None:
        """Delete the saved state."""
        self.path.unlink(missing_ok=True)

    @staticmethod
    def _groups(model: pp.PorePyModel) -> dict[str, Sequence[tuple[pp.GridLike, dict]]]:
        """Grids of the md-grid with their data dictionaries, by type of the grid.

        Parameters:
            model: A prepared model.

        Returns:
            The subdomains (``sd``), interfaces (``intf``) and boundary grids
            (``bg``) with their data dictionaries.

        """
        mdg = model.mdg
        return {
            "sd": mdg.subdomains(return_data=True),
            "intf": mdg.interfaces(return_data=True),
            "bg": mdg.boundaries(return_data=True),
        }

    def _entries(
        self, model: pp.PorePyModel
    ) -> dict[str, tuple[dict, str, str, int | None]]:
        """Locate the stored values of the model.

        Parameters:
            model: A prepared model.

        Returns:
            The key in the archive mapped to the data dictionary, the location, the
            name of the quantity and the index (None for reference values).

        """
        entries: dict[str, tuple[dict, str, str, int | None]] = {}
        for group, items in self._groups(model).items():
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
        """Copy the stored value of an entry.

        Parameters:
            entry: The data dictionary, the location, the name of the quantity and the
                index (None for reference values), see :meth:`_entries`.

        Returns:
            The copy of the stored value.

        """
        data, location, name, index = entry
        stored = data[location][name]
        return np.array(stored if index is None else stored[index])

    def capture_state(self, model: pp.PorePyModel) -> dict[str, np.ndarray]:
        """Collect the arrays defining the current state of the model.

        Parameters:
            model: A prepared model.

        Returns:
            The arrays, with the names described in the class documentation.

        """
        return {key: self._get(e) for key, e in self._entries(model).items()}

    def save_state(self, model: pp.PorePyModel) -> None:
        """Save the state of the model.

        Parameters:
            model: A prepared model.

        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **self.capture_state(model))  # type: ignore[arg-type]
        logger.info(f"Saved state to {self.path}.")

    def load_state(self, model: pp.PorePyModel) -> None:
        """Load the saved state into the model.

        Parameters:
            model: A prepared model.

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
        """Check that the arrays are consistent with the model.

        Parameters:
            model: A prepared model.
            arrays: Arrays as returned by :meth:`capture_state`.

        Raises:
            LoadedStateMismatch: If the names or shapes of the arrays are inconsistent
                with the model, or the values are not finite.

        """

        def fail(message: str) -> NoReturn:
            """Raise the error of an inconsistent state."""
            raise LoadedStateMismatch(f"Saved state is inconsistent: {message}")

        entries = self._entries(model)
        # Reference values may be absent in the model, they are created by
        # apply_state. Everything else must match.
        reference = f"::{pp.REFERENCE_SOLUTIONS}::"
        saved_keys = {key for key in arrays if reference not in key}
        model_keys = {key for key in entries if reference not in key}
        if saved_keys != model_keys:
            fail(f"arrays differ by {sorted(saved_keys ^ model_keys)}")
        for key, entry in entries.items():
            if key in arrays and arrays[key].shape != self._get(entry).shape:
                fail(f"shape of {key}")
        floats = [a for a in arrays.values() if a.dtype.kind == "f"]
        if not all(np.all(np.isfinite(a)) for a in floats):
            fail("values are not finite")

    def apply_state(self, model: pp.PorePyModel, arrays: dict[str, np.ndarray]) -> None:
        """Replace the stored values of the model by the arrays.

        The values that do not exist in the model yet are created.

        Parameters:
            model: A prepared model. The arrays must be consistent with it, see
                :meth:`load_state`.
            arrays: Arrays as returned by :meth:`capture_state`.

        """
        groups = self._groups(model)
        for key, array in arrays.items():
            grid, location, name, *index = key.split("::")
            # This splits letters from numbers, e.g., "sd0" becomes "sd" and "0".
            group, position = re.fullmatch(r"([a-z]+)(\d+)", grid).groups()  # type: ignore[union-attr]
            data = groups[group][int(position)][1]
            stored = data.setdefault(location, {})
            if location == pp.REFERENCE_SOLUTIONS:
                stored[name] = array.copy()
            else:
                stored.setdefault(name, {})[int(index[0])] = array.copy()
        # Fluxes, upwind and other discretizations depend on the state.
        model.update_derived_quantities()
