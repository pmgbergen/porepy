"""Tensor cubic B-spline OBL sampler.

One interpolating tensor-product cubic B-spline is fitted per value field; the value and its gradient
are evaluated from that SAME C2 spline (``NdBSpline`` with ``nu``-derivatives), so the sampled
gradient IS the analytic derivative of the sampled value. That removes the value/gradient
inconsistency of the stored ``grad_`` fields and gives the Newton Jacobian a derivative consistent
with the residual it differentiates.

The splines are built once from the grid the base constructor already read and cached to a sidecar
``<file>.nspline.npz`` (shared knots + one coefficient array per field), so later constructions skip
the fit. Loading/evaluating needs only numpy + scipy (no VTK at run time).
"""
from __future__ import annotations

import os
import types

import numpy as np
from scipy.interpolate import NdBSpline, make_interp_spline

from .base import OBLSampler


def _tensor_bspline(axes, values, k=3):
    """Interpolating tensor-product B-spline of degree ``k`` on a rectilinear grid: solve the 1D
    interpolation along each axis in turn (moving that axis to the front so the coefficient axis
    lands back in place). Returns (knot_vectors, coefficient_array)."""
    c = values
    knots = []
    for axis in range(values.ndim):
        spl = make_interp_spline(axes[axis], np.moveaxis(c, axis, 0), k=k, axis=0)
        knots.append(np.asarray(spl.t, float))
        c = np.moveaxis(spl.c, 0, axis)
    return knots, c


class _LazyPointData:
    """dict-like view over the per-field splines at a fixed point set. A field's value is computed on
    first ``["Field"]`` access and its gradient on first ``["grad_Field"]`` access (each cached), so a
    caller pays only for the fields it reads -- one spline evaluation for a value, three for a
    gradient. In-range value reads skip the gradient entirely (only out-of-range Taylor needs it)."""

    def __init__(self, splines, xc, x, ext, conv, taylor, const_ext):
        self._s, self._xc, self._x, self._ext = splines, xc, x, ext
        self._conv, self._taylor, self._const = conv, taylor, const_ext
        self._cache = {}

    def __contains__(self, key):
        return (key[len("grad_"):] if key.startswith("grad_") else key) in self._s

    def keys(self):
        return list(self._s) + ["grad_" + f for f in self._s]

    def _grad(self, base):
        spl = self._s[base]
        return np.column_stack([spl(self._xc, nu=(1, 0, 0)), spl(self._xc, nu=(0, 1, 0)),
                                spl(self._xc, nu=(0, 0, 1))])

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        if key.startswith("grad_"):
            res = self._grad(key[len("grad_"):]) * self._conv
        else:
            val = np.asarray(self._s[key](self._xc), float)
            if self._taylor and self._ext.any():
                g = np.zeros((len(val), 3)) if key in self._const else self._grad(key)
                val = val.copy()
                val[self._ext] += np.sum(g[self._ext] * (self._x[self._ext] - self._xc[self._ext]),
                                         axis=1)
            res = val
        self._cache[key] = res
        return res


class NSplineSampler(OBLSampler):
    """OBL sampler backed by a persisted tensor cubic B-spline (value + consistent gradient)."""

    def __init__(self, file_name, extended_q: bool = True, cache: bool = True, k: int = 3):
        super().__init__(file_name, extended_q)            # common pyvista read (grid-type agnostic)
        self._k = int(k)
        self._axes, self._dims = self._require_rectilinear()
        self._build_or_load_splines(cache)

    def _require_rectilinear(self):
        """A tensor-product spline needs a rectilinear grid: values on ``axis_z x axis_h x axis_p``
        with 1D axis coordinates (non-uniform spacing is fine). Validate and return (axes, dims),
        or raise with a clear pointer to :class:`VTKSampler` for non-tensor grids."""
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

    def _cache_path(self) -> str:
        return f"{self._file_name}.nspline.npz"

    def _build_or_load_splines(self, cache: bool) -> None:
        path = self._cache_path()
        if cache and os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(self._file_name):
            d = np.load(path, allow_pickle=False)
            if int(d["k"]) == self._k and set(map(str, d["fields"])) >= set(self._field_names):
                knots = (d["kz"], d["kh"], d["kp"])
                self._splines = {f: NdBSpline(knots, d["coef__" + f], k=self._k)
                                 for f in self._field_names}
                return
        pdata = self._search_space.point_data
        knots = None
        coefs = {}
        for f in self._field_names:
            vals = np.asarray(pdata[f], float).reshape(self._dims, order="F")   # -> [iz, ih, ip]
            knots, coef = _tensor_bspline(self._axes, vals, self._k)
            coefs[f] = coef
        self._splines = {f: NdBSpline(tuple(knots), coefs[f], k=self._k) for f in self._field_names}
        if cache:
            out = {"kz": knots[0], "kh": knots[1], "kp": knots[2],
                   "k": np.array(self._k), "fields": np.array(self._field_names)}
            out.update({"coef__" + f: c for f, c in coefs.items()})
            try:
                np.savez_compressed(path, **out)
            except Exception:
                pass

    def _sample(self, x):
        xc = self._clamp_inside(x)
        ext = self._outside_mask(x)
        conv = np.asarray(self.conversion_factors, float)
        pd = _LazyPointData(self._splines, xc, x, ext, conv,
                            self.taylor_extended_q, self.constant_extended_fields)
        return types.SimpleNamespace(point_data=pd)
