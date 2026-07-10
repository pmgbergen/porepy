"""OBL (Driesner table) samplers: a common pyvista-backed base with two interchangeable backends.

    from porepy.examples.geothermal_flow.obl_sampler import VTKSampler, NSplineSampler

* :class:`VTKSampler`     -- pyvista probe of the stored value and ``grad_`` fields (fast; value and
  gradient are inconsistent by O(cell size)).
* :class:`NSplineSampler` -- one tensor cubic B-spline per field; value and gradient come from the
  same C2 spline, so the Jacobian is consistent with the residual it differentiates.

Both share :class:`OBLSampler` (constructor + interface); only :meth:`OBLSampler._sample` differs.
"""
from .base import OBLSampler
from .nspline_sampler import NSplineSampler
from .vtk_sampler import VTKSampler

__all__ = ["OBLSampler", "VTKSampler", "NSplineSampler"]
