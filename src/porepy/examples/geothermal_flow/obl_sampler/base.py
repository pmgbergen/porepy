"""Base class for OBL (Driesner table) samplers.

The Driesner compositional-flow model consumes a sampler through a small, fixed surface:

    sampler.sample_at(points)                       # points = (N, 3) raw (z, coord2, p)
    sampler.sampled_could.point_data["Field"]       # interpolated value        (N,)
    sampler.sampled_could.point_data["grad_Field"]  # interpolated gradient     (N, 3)
    sampler.conversion_factors                       # per-axis input scaling  (get/set)
    sampler.translation_factors                      # per-axis input shift    (get/set)
    sampler.constant_extended_fields                 # fields held flat outside the box (get/set)
    sampler.mutex_state                              # freeze the cloud        (get/set)
    sampler.search_space.bounds                       # (xmin,xmax,ymin,ymax,zmin,zmax)

:class:`OBLSampler` implements everything that is common -- the pyvista-backed constructor (reads the
VTK dataset once and extracts the bounds / value-field names), all of the knobs above, the memoized
``sample_at`` entry point, and the point conversion/translation -- and delegates the actual
interpolation to a subclass hook :meth:`_sample`. The production sampler is
:class:`~obl_sampler.vtk_sampler.VTKSampler`, which auto-selects a grid backend (rectilinear tensor /
hex-AMR / generic probe) and reconstructs a gradient CONSISTENT with the value it returns -- no stored
``grad_`` field is used.
"""
from __future__ import annotations

import numpy as np
import pyvista


class OBLSampler:
    """Common constructor + interface for OBL table samplers; subclasses implement :meth:`_sample`."""

    def __init__(self, file_name, extended_q: bool = True):
        self._file_name = file_name
        self.taylor_extended_q = bool(extended_q)
        self._load_grid(file_name)

    # --- shared pyvista-backed constructor -----------------------------------------------------
    def _load_grid(self, file_name) -> None:
        """Read the VTK dataset once and cache the grid, its bounds and its value-field names. A value
        field is any point-data array that is not itself a ``grad_`` companion -- gradients are
        reconstructed on the fly (analytically by the tensor/hex backends, by finite differences of
        the probe by the generic backend), so a stored ``grad_`` field is neither required nor used.
        Grid-type agnostic: works for any pyvista dataset."""
        grid = pyvista.read(file_name)
        if (not bool(grid.point_data)) and bool(grid.cell_data):
            grid = grid.cell_data_to_point_data()
        self._search_space = grid
        self._boundary_surface = None                 # lazily extracted (heavy on large AMR meshes)
        self._bounds = tuple(float(b) for b in grid.bounds)
        self._field_names = [n for n in grid.point_data.keys() if not n.startswith("grad_")]

    # --- knobs ---------------------------------------------------------------------------------
    @property
    def mutex_state(self):
        return getattr(self, "_mutex_state", False)

    @mutex_state.setter
    def mutex_state(self, v):
        self._mutex_state = v

    @property
    def conversion_factors(self):
        return getattr(self, "_conversion_factors", (1.0, 1.0, 1.0))

    @conversion_factors.setter
    def conversion_factors(self, v):
        self._conversion_factors = tuple(v)

    @property
    def translation_factors(self):
        return getattr(self, "_translation_factors", (0.0, 0.0, 0.0))

    @translation_factors.setter
    def translation_factors(self, v):
        self._translation_factors = tuple(v)

    @property
    def constant_extended_fields(self):
        return getattr(self, "_constant_extended_fields", [])

    @constant_extended_fields.setter
    def constant_extended_fields(self, v):
        self._constant_extended_fields = v

    @property
    def sampled_could(self):
        return getattr(self, "_sampled_could", None)

    @property
    def search_space(self):
        return self._search_space

    @property
    def bounds(self):
        """(xmin,xmax, ymin,ymax, zmin,zmax) of the parametric box -- available even when the grid
        itself was not loaded (a cached backend keeps the bounds but drops ``search_space``)."""
        return self._bounds

    @property
    def boundary_surface(self):
        if self._search_space is None:                # grid not loaded (backend restored from cache)
            return None
        if self._boundary_surface is None:            # extract on first use only
            self._boundary_surface = self._search_space.extract_surface(
                pass_pointid=False, pass_cellid=False, nonlinear_subdivision=0,
                algorithm="dataset_surface")
        return self._boundary_surface

    @property
    def file_name(self):
        return self._file_name

    @property
    def field_names(self):
        return list(self._field_names)

    # --- public sampling: shared scaffolding, subclass does the interpolation -------------------
    def sample_at(self, points) -> None:
        """Convert the raw points (per-axis scale then shift), then delegate to :meth:`_sample`.
        Memoized on the raw points: within one property update the liquid/gas EOS and the secondary
        functions probe the SAME state repeatedly, so the cloud is reused until the state changes."""
        if self.mutex_state and self.sampled_could is not None:
            return
        # memo also keys on the interpolation knobs -- the cloud snapshots conversion/translation/
        # taylor/const at sample time, so mutating any of them must invalidate a same-points reuse.
        knobs = (tuple(self.conversion_factors), tuple(self.translation_factors),
                 bool(self.taylor_extended_q), tuple(self.constant_extended_fields))
        last = getattr(self, "_last_points", None)
        if (self.sampled_could is not None and last is not None
                and last.shape == points.shape and np.array_equal(last, points)
                and getattr(self, "_last_knobs", None) == knobs):
            return
        x = np.asarray(points, float).copy()
        for i, s in enumerate(self.conversion_factors):
            x[:, i] *= s
        for i, t in enumerate(self.translation_factors):
            x[:, i] += t
        self._sampled_could = self._sample(x)
        self._last_points = np.asarray(points, float).copy()
        self._last_knobs = knobs

    def _sample(self, x):
        """Interpolate every field's value and gradient at the CONVERTED points ``x`` and return an
        object exposing ``.point_data[field]`` / ``.point_data['grad_'+field]``. Backend-specific."""
        raise NotImplementedError

    # --- shared helpers ------------------------------------------------------------------------
    def _clamp_inside(self, x: np.ndarray) -> np.ndarray:
        """Per-axis clip of the converted points to just inside the parametric box (also maps any
        out-of-range point to the nearest boundary point, so downstream interpolation stays valid)."""
        b = self._bounds
        eps = 1e-12 * max(b[1] - b[0], b[3] - b[2], b[5] - b[4], 1.0)
        xc = x.copy()
        for i in range(3):
            xc[:, i] = np.clip(xc[:, i], b[2 * i] + eps, b[2 * i + 1] - eps)
        return xc

    def _outside_mask(self, x: np.ndarray) -> np.ndarray:
        """Boolean mask of converted points lying outside the parametric box (need Taylor extension)."""
        b = self._bounds
        out = np.zeros(len(x), bool)
        for i in range(3):
            out |= (x[:, i] < b[2 * i]) | (x[:, i] > b[2 * i + 1])
        return out

    def release_memory(self) -> None:
        try:
            self._search_space.clean()
        except Exception:
            pass
