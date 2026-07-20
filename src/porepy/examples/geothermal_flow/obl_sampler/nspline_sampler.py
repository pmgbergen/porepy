"""Tensor-spline OBL sampler (scipy ``RegularGridInterpolator``).

The value AND its gradient are evaluated from the SAME interpolant -- the gradient via analytic
``nu``-derivatives -- so the sampled gradient IS the derivative of the sampled value (a Jacobian
consistent with the residual, unlike the separately-stored ``grad_`` fields the VTK probe reads).

Default ``method="slinear"`` (multilinear). Two reasons it is the default rather than a smooth cubic:

* MONOTONE: a multilinear interpolant is bounded by the surrounding nodal values, so it cannot
  overshoot. The Driesner property fields have kinks at phase boundaries; a global/tensor CUBIC
  spline rings around them and returns UNPHYSICAL values (negative density, saturation outside
  [0, 1]) that diverge the Newton solve. ``method="cubic"``/``"quintic"`` are accepted (also support
  ``nu``) but must NOT be used on these tables.
* It matches the 1D reference, which reaches HU-mw < HU with exactly this recipe -- bilinear values
  plus a Jacobian consistent with them (there via finite differences; here via exact ``nu``).

The gradient is analytic (no finite differencing), so there is no boundary/``eps`` problem when a
request sits on an axis boundary -- e.g. pure water z = 0: ``nu`` returns the first cell's one-sided
slope. Loading/evaluating needs only numpy + scipy; no coefficient solve and no sidecar cache.
"""
from __future__ import annotations

import types

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .base import OBLSampler


class _LazyPointData:
    """dict-like view over the per-field interpolators at a fixed point set. A field's value is
    computed on first ``["Field"]`` access (one evaluation) and its gradient on first
    ``["grad_Field"]`` access (three ``nu``-derivative evaluations), each cached -- so a caller pays
    only for the fields it reads. In-range value reads skip the gradient (only out-of-range Taylor
    extension needs it)."""

    def __init__(self, rgis, xc, x, ext, conv, taylor, const_ext):
        self._r, self._xc, self._x, self._ext = rgis, xc, x, ext
        self._conv, self._taylor, self._const = conv, taylor, const_ext
        self._cache = {}

    def __contains__(self, key):
        return (key[len("grad_"):] if key.startswith("grad_") else key) in self._r

    def keys(self):
        return list(self._r) + ["grad_" + f for f in self._r]

    def _grad(self, base):
        r = self._r[base]
        return np.column_stack([r(self._xc, nu=(1, 0, 0)), r(self._xc, nu=(0, 1, 0)),
                                r(self._xc, nu=(0, 0, 1))])

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        if key.startswith("grad_"):
            res = self._grad(key[len("grad_"):]) * self._conv
        else:
            val = np.asarray(self._r[key](self._xc), float)
            if self._taylor and self._ext.any():
                g = np.zeros((len(val), 3)) if key in self._const else self._grad(key)
                val = val.copy()
                val[self._ext] += np.sum(g[self._ext] * (self._x[self._ext] - self._xc[self._ext]),
                                         axis=1)
            res = val
        self._cache[key] = res
        return res


class NSplineSampler(OBLSampler):
    """OBL sampler backed by a tensor ``RegularGridInterpolator`` (default multilinear; value and
    consistent analytic gradient from the same interpolant). See the module docstring for why
    ``slinear`` is the default and why ``cubic`` must not be used on the Driesner tables."""

    def __init__(self, file_name, extended_q: bool = True, method: str = "slinear"):
        super().__init__(file_name, extended_q)            # common pyvista read (grid-type agnostic)
        self._method = method
        self._axes, self._dims = self._require_rectilinear()
        pdata = self._search_space.point_data
        self._rgis = {
            f: RegularGridInterpolator(
                tuple(self._axes),
                np.asarray(pdata[f], float).reshape(self._dims, order="F"),   # -> [iz, ih, ip]
                method=method, bounds_error=False, fill_value=None)
            for f in self._field_names}

    def _require_rectilinear(self):
        """A tensor interpolant needs a rectilinear grid: values on ``axis_z x axis_h x axis_p`` with
        1D axis coordinates (non-uniform spacing is fine). Validate and return (axes, dims), or raise
        with a clear pointer to :class:`VTKSampler` for non-tensor grids."""
        grid = self._search_space
        why = None
        try:
            axes = [np.asarray(grid.x, float), np.asarray(grid.y, float), np.asarray(grid.z, float)]
            dims = tuple(int(n) for n in grid.dimensions)
            n_tensor = axes[0].size * axes[1].size * axes[2].size
            if not (all(a.ndim == 1 for a in axes) and n_tensor == grid.n_points
                    and dims == tuple(a.size for a in axes)):
                why = "point layout is not a tensor product of 1D axes"
        except Exception as exc:                            # StructuredGrid/UnstructuredGrid/PolyData
            why = f"{type(grid).__name__} exposes no rectilinear axes ({exc})"
        if why is not None:
            raise TypeError(
                f"NSplineSampler requires a rectilinear (.vtr / RectilinearGrid) table: {why}. "
                f"File: {self._file_name}. Non-uniform axis spacing is supported; curvilinear "
                f"(.vts), unstructured (.vtu) and polydata grids are not -- use VTKSampler for those.")
        return axes, dims

    def _sample(self, x):
        xc = self._clamp_inside(x)
        ext = self._outside_mask(x)
        conv = np.asarray(self.conversion_factors, float)
        pd = _LazyPointData(self._rgis, xc, x, ext, conv,
                            self.taylor_extended_q, self.constant_extended_fields)
        return types.SimpleNamespace(point_data=pd)
