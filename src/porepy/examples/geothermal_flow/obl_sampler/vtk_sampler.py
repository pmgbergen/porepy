"""VTK-probe OBL sampler: pyvista trilinear interpolation of the value AND of the separately-stored
``grad_`` field. Fast (one probe returns every field), but the sampled gradient is the interpolation
of a distinct stored field, NOT the derivative of the sampled value -- so the two are inconsistent by
O(cell size) (~1-3% on the Driesner ``rho``/``mu`` tables). Use :class:`NSplineSampler` when the
Newton Jacobian needs a gradient consistent with the value it differentiates.
"""
from __future__ import annotations

import numpy as np
import pyvista

from .base import OBLSampler


class VTKSampler(OBLSampler):
    """OBL sampler backed by ``pyvista``'s probe of the pre-tabulated value and gradient fields."""

    def _sample(self, x):
        xc = self._clamp_inside(x)                         # map out-of-range points onto the box
        cloud = pyvista.PolyData(xc).sample(self._search_space)
        # Deep-copy and drop the pipeline: the probe output retains references to the resample filter
        # and its inputs; keeping it directly lets memory grow across the many probes of a long run.
        cloud = cloud.copy(deep=True)
        ext = self._outside_mask(x)
        if self.taylor_extended_q and ext.any():
            self._taylor_extend(cloud, x, xc, ext)
        self._scale_gradients(cloud)
        return cloud

    def _taylor_extend(self, cloud, x, xc, ext) -> None:
        """First-order (Taylor) extension of out-of-range points from their boundary projection
        ``xc``, using the STORED boundary gradient (before the conversion scaling below)."""
        pd = cloud.point_data
        for name in list(pd.keys()):
            if not name.startswith("grad_"):
                continue
            base = name[len("grad_"):]
            if base not in pd:
                continue
            g = np.zeros_like(pd[name]) if base in self.constant_extended_fields else pd[name]
            fv = pd[base]
            fv[ext] = fv[ext] + np.sum(g[ext] * (x[ext] - xc[ext]), axis=1)

    def _scale_gradients(self, cloud) -> None:
        """``d/d(raw coord) = conversion * d/d(table coord)`` -- scale each gradient component by the
        corresponding conversion factor, matching the historical VTKSampler behaviour."""
        conv = self.conversion_factors
        for name in list(cloud.point_data.keys()):
            if name.startswith("grad_"):
                g = cloud.point_data[name]
                for i, s in enumerate(conv):
                    g[:, i] *= s
