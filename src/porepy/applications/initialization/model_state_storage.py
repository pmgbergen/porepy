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
    def _get_mdg_data(
        model: pp.PorePyModel,
    ) -> dict[str, Sequence[tuple[pp.GridLike, dict]]]:
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

    @staticmethod
    def _construct_key(
        domain_type: str,
        domain_idx: int,
        location: str,
        array_name: str,
        array_index: int | None,
    ) -> str:
        """Construct the key of an array in the saved state.

        This is the inverse of :meth:`_key_to_tuple`.

        Parameters:
            domain_type: ``"sd"``, ``"intf"`` or ``"bg"``, for subdomains, interfaces
                and boundary grids respectively.
            domain_idx: Position of the grid among the grids of its type in the
                md-grid.
            location: Key of the data dictionary, one of :attr:`_locations`.
            array_name: Name of the quantity, e.g. a variable or a boundary value.
            array_index: Iterate or time step index. None for reference values.

        Returns:
            The key in format ``<grid>::<location>::<name>[::<index>]``.

        """
        key = f"{domain_type}{domain_idx}::{location}::{array_name}"
        if array_index is not None:
            key = f"{key}::{array_index}"
        return key

    _key_pattern = re.compile(
        r"^(?P<domain_type>sd|intf|bg)(?P<domain_idx>\d+)"
        r"::(?P<location>[^:]+)"
        r"::(?P<array_name>.+?)"
        r"(?:::(?P<array_index>\d+))?$"
    )
    """Inverse of :meth:`_construct_key`."""

    @classmethod
    def _key_to_tuple(cls, key: str) -> tuple[str, int, str, str, int | None]:
        """Split a key into the arguments of :meth:`_construct_key`.

        Parameters:
            key: Key in format ``<grid>::<location>::<name>[::<index>]``.

        Returns:
            The domain type, domain index, location, array name and array index (None
            for reference values).

        Raises:
            LoadedStateMismatch: If the key does not have the expected format.

        """
        match = cls._key_pattern.match(key)
        if match is None:
            raise LoadedStateMismatch(f"Unexpected key in the saved state: {key}")
        index = match["array_index"]
        return (
            match["domain_type"],
            int(match["domain_idx"]),
            match["location"],
            match["array_name"],
            None if index is None else int(index),
        )

    def capture_state(self, model: pp.PorePyModel) -> dict[str, np.ndarray]:
        """Collect the arrays defining the current state of the model.

        Parameters:
            model: A prepared model.

        Returns:
            The arrays, with the keys in format
            ``<grid>::<location>::<name>[::<index>]``, see class docstring. The arrays
            are the ones stored in the model, not copies of them.

        """
        state = {}
        for domain_type, domains_and_data in self._get_mdg_data(model).items():
            # We use domain indices and not their IDs, because IDs can change if we
            # repeatedly create models in a single runscript.
            for domain_idx, (domain, domain_data) in enumerate(domains_and_data):
                for location in self._locations:
                    if location not in domain_data:
                        logger.debug(
                            f"{location} is not present in the data of {domain}, "
                            "skipping it."
                        )
                    for array_name, arrays in domain_data.get(location, {}).items():
                        # In reference solutions, the structure is:
                        # data[pp.REFERENCE_SOLUTIONS][array_name] (there is no index).
                        if location == pp.REFERENCE_SOLUTIONS:
                            assert isinstance(arrays, np.ndarray)
                            key = self._construct_key(
                                domain_type,
                                domain_idx,
                                location,
                                array_name,
                                array_index=None,
                            )
                            state[key] = arrays
                            continue

                        # In time steps and iterates, the structire is:
                        # data[pp.TIME_STEP_SOLUTIONS][array_name][array_index].
                        for array_index, array in arrays.items():
                            assert isinstance(array, np.ndarray)
                            key = self._construct_key(
                                domain_type,
                                domain_idx,
                                location,
                                array_name,
                                array_index,
                            )
                            state[key] = array

        return state

    def save_state(self, model: pp.PorePyModel) -> None:
        """Save the state of the model. See the class docstring for more information.

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
        self, model: pp.PorePyModel, suggested_state: dict[str, np.ndarray]
    ) -> None:
        """Check that the arrays are consistent with the model.

        Parameters:
            model: A prepared model.
            suggested_state: Arrays as returned by :meth:`capture_state`, to be checked
                against the current state of the model.

        Raises:
            LoadedStateMismatch: If the names or shapes of the arrays are inconsistent
                with the model, or the values are not finite.

        """

        def fail(message: str) -> NoReturn:
            """Raise the error of an inconsistent state."""
            raise LoadedStateMismatch(f"Saved state is inconsistent: {message}")

        current_state = self.capture_state(model)

        # Reference values may be absent in the model, they are created by
        # apply_state. Everything else must match.
        reference = f"::{pp.REFERENCE_SOLUTIONS}::"
        current_keys = {key for key in current_state if reference not in key}
        suggested_keys = {key for key in suggested_state if reference not in key}
        if current_keys != suggested_keys:
            fail(f"States differ by {sorted(current_keys ^ suggested_keys)}")

        for key, array in current_state.items():
            if key in suggested_state and suggested_state[key].shape != array.shape:
                fail(f"shape of {key}")

        if not all(np.all(np.isfinite(a)) for a in suggested_state.values()):
            fail("values are not finite")

    def apply_state(self, model: pp.PorePyModel, arrays: dict[str, np.ndarray]) -> None:
        """Replace the stored values of the model by the arrays.

        Copies of the arrays are inserted into the data dictionaries of the model,
        replacing the previous arrays (they are not overwritten in place, so arrays
        returned earlier by :meth:`capture_state` stay unchanged). The values that do
        not exist in the model yet are created. The derived quantities of the model are
        updated afterwards.

        Parameters:
            model: A prepared model. The arrays must be consistent with it, see
                :meth:`load_state`.
            arrays: Arrays as returned by :meth:`capture_state`.

        """
        model_data = self._get_mdg_data(model)

        for key, array in arrays.items():
            domain_type, domain_idx, location, array_name, array_index = (
                self._key_to_tuple(key)
            )
            domains_and_data = model_data[domain_type]
            _, data = domains_and_data[domain_idx]
            inner_dict = data.setdefault(location, {})

            if location == pp.REFERENCE_SOLUTIONS:
                # In reference solutions, the structure is:
                # data[pp.REFERENCE_SOLUTIONS][array_name] (there is no index).
                inner_dict[array_name] = array.copy()
            else:
                # In time steps and iterates, the structire is:
                # data[pp.TIME_STEP_SOLUTIONS][array_name][array_index].
                inner_dict.setdefault(array_name, {})[array_index] = array.copy()

        # Fluxes, upwind and other discretizations depend on the state.
        model.update_derived_quantities()
