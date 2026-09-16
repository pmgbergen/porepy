
import porepy as pp

from porepy.models.compositional_flow import (
    CompositionalFlowTemplate
)

from .constitutive_description.mixture_constitutive_description import (
    FluidMixture,
    SecondaryEquations,
    ComponentSystem
)


class VTKSamplerMixin:
    @property
    def vtk_sampler(self):
        return self._vtk_sampler

    @vtk_sampler.setter
    def vtk_sampler(self, vtk_sampler):
        self._vtk_sampler = vtk_sampler

    @property
    def vtk_sampler_ptz(self):
        return self._vtk_sampler_ptz

    @vtk_sampler_ptz.setter
    def vtk_sampler_ptz(self, vtk_sampler):
        self._vtk_sampler_ptz = vtk_sampler


class LiquidSecondaryEquation(SecondaryEquations):
    component_system = ComponentSystem.WATER
    phase_mode = pp.compositional.PhysicalState.liquid


class SinglePhaseFlowModelConfigurationLiquid(
    FluidMixture,
    LiquidSecondaryEquation,
    CompositionalFlowTemplate,
    VTKSamplerMixin
):
    def relative_permeability(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return phase.saturation(domains)
