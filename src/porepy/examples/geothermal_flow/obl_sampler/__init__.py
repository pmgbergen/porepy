"""OBL (Driesner table) samplers.

    from porepy.examples.geothermal_flow.obl_sampler import VTKSampler

* :class:`VTKSampler` -- unified sampler: one class, three grid backends auto-selected from the VTK
  dataset. ``.vtr`` rectilinear -> O(1) multilinear tensor; ``.vtu`` all-hexahedron -> per-cell
  trilinear on an octree-AMR mesh; anything else -> generic pyvista probe. The tensor and hex backends
  return value + analytic gradient from the same interpolant (Jacobian consistent with the residual);
  the probe backend reads the separately-stored ``grad_`` field (inconsistent).

Every solver -- weis_1d_solver and all porepy_*d_solver -- uses this one sampler, so their table
derivative constructions are identical (multilinear value + analytic gradient of that interpolant).
"""
from .base import OBLSampler
from .vtk_sampler import VTKSampler

__all__ = ["OBLSampler", "VTKSampler"]
