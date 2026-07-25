"""Unified OBL table sampler: ONE class, three grid backends, auto-selected by the VTK dataset.

    * ``.vtr`` RectilinearGrid  -> ``tensor`` backend : O(1) per-axis multilinear, analytic gradient.
    * ``.vtu`` all-hexahedron   -> ``hex``    backend : per-cell trilinear on a hex octree-AMR mesh,
                                                       O(1) point location, analytic gradient.
    * anything else (mixed/curved/poly) -> ``probe`` backend : generic pyvista probe of the value; its
                                                       gradient is reconstructed by finite differences.

Every backend reconstructs a gradient that is CONSISTENT with the value it returns -- the analytic
derivative of the interpolant for tensor/hex, a central difference-quotient of the probe for the
generic backend -- so no stored ``grad_`` companion field is needed or read. The tensor and hex
backends compute the value AND all three gradient components from a SINGLE 8-corner gather per field,
evaluated LAZILY per field, so a caller pays only for the fields it reads. Backend is auto-detected
from the grid; force one with ``backend=`` ('tensor' | 'hex' | 'probe').

Multilinear (not cubic) on purpose: the Driesner property fields are kinked at phase boundaries and a
tensor cubic rings around them into unphysical values (negative density, saturation outside [0, 1]).
"""
from __future__ import annotations

import os

import numpy as np
import pyvista

from .base import OBLSampler


# --------------------------------------------------------------------------------------- #
#  1-D bracketing kernel (uniform fast path / searchsorted for graded axes)
# --------------------------------------------------------------------------------------- #
def _is_uniform(x):
    """True if the 1-D axis is (float-)uniformly spaced; a <3-node stub counts as uniform."""
    if len(x) < 3:
        return True
    d = np.diff(x)
    return bool(np.max(np.abs(d - d[0])) <= 1e-6 * abs(x[-1] - x[0]))


def _bracket(v, coords, uniform):
    """Lower index ``i``, upper index ``i+1`` and in-cell fraction ``t`` for a monotone axis at points
    ``v`` (already inside the range -- callers clamp first). Uniform: O(1) arithmetic; graded:
    searchsorted. Returns intp arrays ``i``, ``i1`` and float ``t``."""
    n = len(coords)
    if uniform and n > 1:
        dv = (coords[-1] - coords[0]) / (n - 1)
        f = ((v - coords[0]) / dv).clip(0.0, n - 1 - 1e-9)
        i = f.astype(np.intp)
        return i, i + 1, f - i
    i = np.clip(np.searchsorted(coords, v, side="right") - 1, 0, n - 2)
    i1 = i + 1
    t = ((v - coords[i]) / (coords[i1] - coords[i])).clip(0.0, 1.0)
    return i, i1, t


# --------------------------------------------------------------------------------------- #
#  Lazy value/gradient view (shared by the analytic backends: tensor + hex)
# --------------------------------------------------------------------------------------- #
class _LazyCloud:
    """dict-like ``point_data`` over an analytic backend. ``["Field"]`` triggers one 8-corner gather
    that yields BOTH the value and its (table-coord) gradient; the pair is cached, so a later
    ``["grad_Field"]`` is free. Values are held at the clamped points; out-of-range points are
    first-order (Taylor) extended from the boundary using that same gradient. Gradients are returned
    scaled by the conversion factors (``d/d(raw) = conv * d/d(table)``)."""

    def __init__(self, fields, vg, x, xc, ext, conv, taylor, const):
        self._fields = set(fields)
        self._vg = vg                                   # field -> (value(N,), grad_table(N,3))
        self._x, self._xc, self._ext = x, xc, ext
        self._conv = np.asarray(conv, float)
        self._taylor, self._const = taylor, set(const or [])
        self._cache = {}

    def __contains__(self, key):
        return (key[5:] if key.startswith("grad_") else key) in self._fields

    def keys(self):
        return list(self._fields) + ["grad_" + f for f in self._fields]

    def _pair(self, base):
        if base not in self._cache:
            self._cache[base] = self._vg(base)
        return self._cache[base]

    def __getitem__(self, key):
        if key.startswith("grad_"):
            return self._pair(key[5:])[1] * self._conv
        val, g = self._pair(key)
        if self._taylor and self._ext.any():
            gg = np.zeros_like(g) if key in self._const else g
            val = val.copy()
            val[self._ext] += np.sum(gg[self._ext] * (self._x[self._ext] - self._xc[self._ext]), axis=1)
        return val


def _trilinear_vg(C, wz, w2, wp, dz, d2, dp):
    """Value and analytic gradient of a trilinear cell from its 8 corner values ``C`` (N,2,2,2) with
    per-axis weights ``w*`` (N,2) and cell widths ``d*`` (N,). Axes (i,j,k)=(z, coord2, p); the
    gradient columns are (d/dz, d/dcoord2, d/dp). Value + all three derivatives share the one gather."""
    w = wz[:, :, None, None] * w2[:, None, :, None] * wp[:, None, None, :]      # (N,2,2,2)
    val = (C * w).sum((1, 2, 3))
    gz = ((C[:, 1] - C[:, 0]) * (w2[:, :, None] * wp[:, None, :])).sum((1, 2)) / dz
    g2 = ((C[:, :, 1] - C[:, :, 0]) * (wz[:, :, None] * wp[:, None, :])).sum((1, 2)) / d2
    gp = ((C[:, :, :, 1] - C[:, :, :, 0]) * (wz[:, :, None] * w2[:, None, :])).sum((1, 2)) / dp
    return val, np.stack([gz, g2, gp], 1)


# --------------------------------------------------------------------------------------- #
#  Backend: rectilinear tensor (.vtr)
# --------------------------------------------------------------------------------------- #
class _TensorBackend:
    """O(1) multilinear on a RectilinearGrid: bracket each of the three axes, gather the 8 corners of
    the containing box, blend. Non-uniform axes handled per-axis (uniform arithmetic / graded
    searchsorted). Value fields = every point-data array except the ``grad_`` companions."""

    kind = "tensor"

    def __init__(self, owner):
        grid = owner._search_space
        self.az = np.asarray(grid.x, float)             # axis0 = z_NaCl
        self.a2 = np.asarray(grid.y, float)             # axis1 = coord2 (h or T)
        self.ap = np.asarray(grid.z, float)             # axis2 = p
        self.dims = tuple(int(n) for n in grid.dimensions)
        self._validate(grid)
        self.uz, self.u2, self.up = _is_uniform(self.az), _is_uniform(self.a2), _is_uniform(self.ap)
        self.fields = [n for n in grid.point_data.keys() if not n.startswith("grad_")]
        pd = grid.point_data
        self.F = {f: np.asarray(pd[f], float).reshape(self.dims, order="F") for f in self.fields}
        self.owner = owner

    def _validate(self, grid):
        if not (self.az.ndim == self.a2.ndim == self.ap.ndim == 1
                and self.dims == (self.az.size, self.a2.size, self.ap.size)
                and self.az.size * self.a2.size * self.ap.size == grid.n_points):
            raise TypeError("tensor backend needs a rectilinear (.vtr) tensor-product grid")

    def sample(self, x):
        xc = self.owner._clamp_inside(x)
        ext = self.owner._outside_mask(x)
        iz, iz1, tz = _bracket(xc[:, 0], self.az, self.uz)
        i2, i21, t2 = _bracket(xc[:, 1], self.a2, self.u2)
        ip, ip1, tp = _bracket(xc[:, 2], self.ap, self.up)
        wz = np.stack([1 - tz, tz], 1); w2 = np.stack([1 - t2, t2], 1); wp = np.stack([1 - tp, tp], 1)
        dz = self.az[iz1] - self.az[iz]; d2 = self.a2[i21] - self.a2[i2]; dp = self.ap[ip1] - self.ap[ip]
        IZ = np.stack([iz, iz1], 1); I2 = np.stack([i2, i21], 1); IP = np.stack([ip, ip1], 1)

        def vg(field):
            F = self.F[field]
            C = F[IZ[:, :, None, None], I2[:, None, :, None], IP[:, None, None, :]]   # (N,2,2,2)
            return _trilinear_vg(C, wz, w2, wp, dz, d2, dp)

        return _LazyView(self, vg, x, xc, ext)


# --------------------------------------------------------------------------------------- #
#  Backend: hexahedral octree-AMR (.vtu, all VTK_HEXAHEDRON)
# --------------------------------------------------------------------------------------- #
class _HexBackend:
    """Per-cell trilinear on an all-hexahedron AMR mesh. Point location is O(1) and exact: the union
    of all cell boundaries is a background grid, and a precomputed fine-cell -> hex-cell map turns a
    query into three searchsorted + one gather (valid because AMR hexes tile the bounding box). The
    heavy build (cellmap + canonical corner values) is cached next to the ``.vtu``."""

    kind = "hex"
    CACHE_TAG = "oblhexv1"

    def __init__(self, owner):
        self.owner = owner
        cache = owner.file_name + ".oblhex.npz"
        key = f"{self.CACHE_TAG}|{owner.field_names_all}|{os.path.getmtime(owner.file_name):.0f}"
        if not (os.path.exists(cache) and self._load(cache, key)):
            self._build(owner._search_space, owner.field_names_all, key, cache)

    def _load(self, cache, key):
        try:
            z = np.load(cache, allow_pickle=False)
            if str(z["key"]) != key:
                return False
            for s in ("ux", "uy", "uz", "cmin", "cmax", "V", "cellmap"):
                setattr(self, s, z[s])
            self.fields = [str(n) for n in z["names"]]
            self.dims = (self.ux.size - 1, self.uy.size - 1, self.uz.size - 1)
            return True
        except Exception:
            return False

    def _build(self, grid, fields, key, cache):
        pts = np.asarray(grid.points, float)
        conn = grid.cells.reshape(-1, 9)
        if not np.all(conn[:, 0] == 8):
            raise TypeError("hex backend needs an all-hexahedron (.vtu) mesh")
        conn = conn[:, 1:]
        M = conn.shape[0]
        self.fields = list(fields)
        node_xyz = pts[conn]                                        # (M,8,3)
        self.cmin = node_xyz.min(1); self.cmax = node_xyz.max(1)    # (M,3)
        if not np.all(self.cmax - self.cmin > 0):
            raise ValueError("degenerate hex cell")
        mid = 0.5 * (self.cmin + self.cmax)
        gt = (node_xyz > mid[:, None, :]).astype(np.intp)
        corner = gt[:, :, 0] * 4 + gt[:, :, 1] * 2 + gt[:, :, 2]    # canonical slot i*4+j*2+k
        vals = np.stack([np.asarray(grid.point_data[n], float)[conn] for n in self.fields])
        self.V = np.empty((len(self.fields), M, 8), float)          # (field, cell, corner)
        for f in range(len(self.fields)):
            np.put_along_axis(self.V[f], corner, vals[f], axis=1)
        self.ux = np.unique(pts[:, 0]); self.uy = np.unique(pts[:, 1]); self.uz = np.unique(pts[:, 2])
        self.dims = (self.ux.size - 1, self.uy.size - 1, self.uz.size - 1)
        i0 = np.searchsorted(self.ux, self.cmin[:, 0]); i1 = np.searchsorted(self.ux, self.cmax[:, 0])
        j0 = np.searchsorted(self.uy, self.cmin[:, 1]); j1 = np.searchsorted(self.uy, self.cmax[:, 1])
        k0 = np.searchsorted(self.uz, self.cmin[:, 2]); k1 = np.searchsorted(self.uz, self.cmax[:, 2])
        self.cellmap = np.full(self.dims, -1, np.int32)
        for m in range(M):
            self.cellmap[i0[m]:i1[m], j0[m]:j1[m], k0[m]:k1[m]] = m
        if (self.cellmap < 0).any():
            raise ValueError("AMR mesh does not tile its bounding box (gaps in the fine grid)")
        try:
            np.savez(cache, key=np.array(key), ux=self.ux, uy=self.uy, uz=self.uz, cmin=self.cmin,
                     cmax=self.cmax, V=self.V, cellmap=self.cellmap, names=np.array(self.fields))
        except Exception:
            pass

    def sample(self, x):
        xc = self.owner._clamp_inside(x)
        ext = self.owner._outside_mask(x)
        ix = np.clip(np.searchsorted(self.ux, xc[:, 0], side="right") - 1, 0, self.dims[0] - 1)
        iy = np.clip(np.searchsorted(self.uy, xc[:, 1], side="right") - 1, 0, self.dims[1] - 1)
        iz = np.clip(np.searchsorted(self.uz, xc[:, 2], side="right") - 1, 0, self.dims[2] - 1)
        cid = self.cellmap[ix, iy, iz]
        cmn = self.cmin[cid]; d = self.cmax[cid] - cmn
        t = np.clip((xc - cmn) / d, 0.0, 1.0)
        wz = np.stack([1 - t[:, 0], t[:, 0]], 1); w2 = np.stack([1 - t[:, 1], t[:, 1]], 1)
        wp = np.stack([1 - t[:, 2], t[:, 2]], 1)
        idx = {nm: i for i, nm in enumerate(self.fields)}

        def vg(field):
            C = self.V[idx[field], cid, :].reshape(-1, 2, 2, 2)     # (N,2,2,2)
            return _trilinear_vg(C, wz, w2, wp, d[:, 0], d[:, 1], d[:, 2])

        return _LazyView(self, vg, x, xc, ext)


# --------------------------------------------------------------------------------------- #
#  Backend: generic pyvista probe (any dataset) -- historical VTKSampler behaviour
# --------------------------------------------------------------------------------------- #
class _ProbeBackend:
    """Generic pyvista probe for any dataset (mixed / curvilinear / poly cells the tensor and hex
    backends cannot handle). The value is the probe of the field; its gradient is reconstructed ON THE
    FLY by central finite differences of the SAME probe -- a consistent difference-quotient of the
    value, using no stored ``grad_`` field. The gradient costs six extra probes (shared across all
    fields, computed only if a gradient is read), so this backend is the slow fallback."""

    kind = "probe"

    def __init__(self, owner):
        self.owner = owner
        self.fields = [n for n in owner._search_space.point_data.keys() if not n.startswith("grad_")]

    def _probe(self, pts):
        cloud = pyvista.PolyData(np.ascontiguousarray(pts)).sample(self.owner._search_space)
        return {f: np.asarray(cloud.point_data[f], float) for f in self.fields}

    def sample(self, x):
        o = self.owner
        xc = o._clamp_inside(x); ext = o._outside_mask(x)
        base = self._probe(xc)
        b = o._bounds
        eps = 1e-4 * np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
        grads = {}

        def _fill():                                   # 6 probes, all fields at once, on first grad read
            g = {f: np.zeros((len(xc), 3)) for f in self.fields}
            for a in range(3):
                xp = xc.copy(); xp[:, a] += eps[a]; xp = o._clamp_inside(xp)
                xm = xc.copy(); xm[:, a] -= eps[a]; xm = o._clamp_inside(xm)
                vp, vm = self._probe(xp), self._probe(xm)
                h = xp[:, a] - xm[:, a]                 # actual clamped step (avoids /0 at the edge)
                h = np.where(h == 0.0, 1.0, h)
                for f in self.fields:
                    g[f][:, a] = (vp[f] - vm[f]) / h
            grads.update(g)

        def vg(field):
            if not grads:
                _fill()
            return base[field], grads[field]

        return _LazyView(self, vg, x, xc, ext)


def _LazyView(backend, vg, x, xc, ext):
    """Wrap a backend's per-field ``vg`` in the OBL ``.point_data`` surface (snapshotting the owner's
    conversion / taylor / constant-extension knobs at sample time)."""
    import types
    o = backend.owner
    return types.SimpleNamespace(point_data=_LazyCloud(
        backend.fields, vg, x, xc, ext, o.conversion_factors, o.taylor_extended_q,
        o.constant_extended_fields))


# --------------------------------------------------------------------------------------- #
#  The unified sampler
# --------------------------------------------------------------------------------------- #
class VTKSampler(OBLSampler):
    """Unified OBL sampler; picks a backend from the grid (override with ``backend=``). See module
    docstring. Exposes the same ``sample_at`` / ``sampled_could.point_data`` surface as before."""

    _BACKENDS = {"tensor": _TensorBackend, "hex": _HexBackend, "probe": _ProbeBackend}

    def __init__(self, file_name, extended_q: bool = True, backend: str = "auto"):
        super().__init__(file_name, extended_q)
        # all non-grad_ point-data names (the analytic backends compute their own gradients, so they
        # do NOT require pre-stored grad_ companions the way the base's field detection assumes).
        self.field_names_all = [n for n in self._search_space.point_data.keys()
                                if not n.startswith("grad_")]
        self._backend = self._make_backend(backend)
        self._field_names = self._backend.fields

    def _make_backend(self, backend):
        if backend != "auto":
            if backend not in self._BACKENDS:
                raise ValueError(f"backend must be 'auto' or one of {list(self._BACKENDS)}")
            return self._BACKENDS[backend](self)
        grid = self._search_space
        tname = type(grid).__name__
        if tname == "RectilinearGrid":
            return _TensorBackend(self)
        if tname == "UnstructuredGrid" and grid.n_cells and np.all(grid.celltypes == 12):
            return _HexBackend(self)
        return _ProbeBackend(self)                          # generic fallback

    @property
    def backend_kind(self):
        return self._backend.kind

    def _sample(self, x):
        return self._backend.sample(x)
