"""OBL (Driesner table) samplers.

    from porepy.examples.geothermal_flow.obl_sampler import VTKSampler

* :class:`VTKSampler` -- unified sampler: one class, three grid backends auto-selected from the VTK
  dataset. ``.vtr`` rectilinear -> O(1) multilinear tensor; ``.vtu`` all-hexahedron -> per-cell
  trilinear on an octree-AMR mesh; anything else -> generic pyvista probe. The tensor and hex backends
  return value + analytic gradient from the same interpolant (Jacobian consistent with the residual);
  the probe backend reads the separately-stored ``grad_`` field (inconsistent).

:class:`NSplineSampler` is retained for backward compatibility (its consistent-tensor behaviour is now
the VTKSampler ``tensor`` backend); it will be removed once the solvers are switched over.
"""
from .base import OBLSampler
from .nspline_sampler import NSplineSampler
from .vtk_sampler import VTKSampler

__all__ = ["OBLSampler", "VTKSampler", "NSplineSampler"]
