import numpy as np

import porepy as pp
from ...vtk_sampler import VTKSampler


class ICBrineSystem2D(pp.PorePyModel):

    vtk_sampler_ptz: VTKSampler

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Define an initial pressure distribution that varies linearly from
           the inlet to the outlet of the domain.
        """
        # return np.ones(sd.num_cells) * self._p_INIT
        if sd.dim == 0 and "production_well" in sd.tags:
            return np.ones(sd.num_cells) * self._p_PRODUCTION[0]
        return np.ones(sd.num_cells) * self._p_INIT

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        # evaluation from PTZ specs
        p = self.ic_values_pressure(sd)
        t = self.ic_values_temperature(sd)
        z_NaCl = self._z_INIT["NaCl"] * np.ones_like(p)
        assert len(p) == len(t) == len(z_NaCl)
        par_points = np.array((z_NaCl, t, p)).T
        self.vtk_sampler_ptz.sample_at(par_points)
        h_init = self.vtk_sampler_ptz.sampled_could.point_data['H']
        return h_init

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        # return np.ones(sd.num_cells) * self._T_INIT
        if sd.dim == 0 and "injection_well" in sd.tags:
            return np.ones(sd.num_cells)*self._T_INJECTION[0]
        return np.ones(sd.num_cells) * self._T_INIT

    def ic_values_overall_fraction(
        self, component: pp.Component,
        sd: pp.Grid
    ) -> np.ndarray:
        """Initial composition: default to initial z."""
        z = self._z_INIT.get(component.name, 0.0)
        return np.full(sd.num_cells, z)