"""Shared VTU export for the Driesner brine geothermal solvers (subsection_4_1 / subsection_4_3).

Adds the full THREE-phase Driesner state -- vapor / liquid / halite saturations, densities and
specific enthalpies, plus the mixture density and temperature in Celsius -- on top of whatever
the base model already exports, so ``porepy_1d_solver`` and ``porepy_3d_solver`` write an
identical field set.

All fields are read from the OBL ``xph`` table (``self.obl_sampler``, indexed by
``(z_NaCl, h, p)``) sampled at the model's current state on every subdomain (matrix + fractures
+ intersections).  The model itself solves a two-phase (liquid + vapor) system; the halite
fields (``s_h``, ``rho_h``, ``h_h``) come straight from the table, so a cell that oversaturates
into solid halite still reports it here even though halite is not an independent phase in the
solved equations.

Exported fields (units):
    T_C   temperature [degC]           s_v/s_l/s_h  vapor/liquid/halite saturation [-]
    rho   mixture density [kg/m^3]     rho_v/rho_l/rho_h  phase densities [kg/m^3]
    h_v/h_l/h_h  phase specific enthalpies [same scaling as the model enthalpy variable, 1e-3*H]
"""
from __future__ import annotations

import numpy as np

import porepy as pp


class DriesnerPhaseExport(pp.PorePyModel):
    """Mixin adding the 3-phase Driesner fields to ``data_to_export``.

    Mix in FIRST (before the geometry/BC/IC/flow model) so this ``data_to_export`` wins and its
    ``super()`` call still chains to the base exporter (the default primary-variable output).
    """

    # exported name -> (xph table field, scale, offset);  value = table_field * scale + offset
    _EXPORT_FIELDS: tuple[tuple[str, str, float, float], ...] = (
        ("T_C",   "Temperature", 1.0, -273.15),   # K -> degC
        ("s_v",   "S_v",   1.0, 0.0),
        ("s_l",   "S_l",   1.0, 0.0),
        ("s_h",   "S_h",   1.0, 0.0),
        ("rho_v", "Rho_v", 1.0, 0.0),
        ("rho_l", "Rho_l", 1.0, 0.0),
        ("rho_h", "Rho_h", 1.0, 0.0),
        ("rho",   "Rho",   1.0, 0.0),
        ("h_v",   "H_v", 1.0e-3, 0.0),             # match the model's enthalpy scaling (1e-3 * H)
        ("h_l",   "H_l", 1.0e-3, 0.0),
        ("h_h",   "H_h", 1.0e-3, 0.0),
    )

    def data_to_export(self):
        """Base export (primary variables) + the 3-phase Driesner fields sampled per subdomain."""
        data = super().data_to_export()  # type: ignore[misc]
        sampler = getattr(self, "obl_sampler", None)
        if sampler is None:                        # sampler not attached -> nothing extra to add
            return data
        ev = self.equation_system.evaluate
        z_NaCl = next(c for c in self.fluid.components
                      if c != self.fluid.reference_component)
        for sd in self.mdg.subdomains():
            p = np.asarray(ev(self.pressure([sd])), dtype=float)
            h = np.asarray(ev(self.enthalpy([sd])), dtype=float)
            z = np.asarray(ev(z_NaCl.fraction([sd])), dtype=float)
            sampler.sample_at(np.array((z, h, p)).T)     # (N, 3) = (z_NaCl, h, p)
            pdata = sampler.sampled_could.point_data
            for name, field, scale, offset in self._EXPORT_FIELDS:
                data.append((sd, name, np.asarray(pdata[field], dtype=float) * scale + offset))
        return data
