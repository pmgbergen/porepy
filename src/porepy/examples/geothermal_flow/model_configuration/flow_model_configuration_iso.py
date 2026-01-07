
import porepy as pp

from porepy.models.isothermal_compositional_flow import (
    IsothermalCompositionalFlowTemplate
)

from .constitutive_description.mixture_constitutive_description_iso import (
    FluidMixture,
    SecondaryEquations,
    ComponentSystem,
    PhaseMode,
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


class TwoPhaseSecondaryEquation(SecondaryEquations):
    component_system = ComponentSystem.WATER
    phase_mode = PhaseMode.TWO_PHASE


class SinglePhaseFlowModelConfigurationLiquid(
    FluidMixture,
    LiquidSecondaryEquation,
    IsothermalCompositionalFlowTemplate,
    VTKSamplerMixin
):
    def relative_permeability(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return phase.saturation(domains)


class TwoPhaseFlowModelConfiguration(
    FluidMixture,
    TwoPhaseSecondaryEquation,
    IsothermalCompositionalFlowTemplate,
    VTKSamplerMixin
):
    def relative_permeability_old(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return phase.saturation(domains)
    
    def relative_permeability(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        
        # residual saturation of the liquid phase (non-wetting phase). Weis et al. (2014)
        r_l = 0.3

        max = pp.ad.Function(pp.ad.maximum, "maximum_function")
        s = phase.saturation(domains)

        if phase == self.fluid.reference_phase:
            kr_l = (s - pp.ad.Scalar(r_l)) / (pp.ad.Scalar(1.0) - pp.ad.Scalar(r_l))
            return max(kr_l, pp.ad.Scalar(0.0))
        return s / (pp.ad.Scalar(1.0) - pp.ad.Scalar(r_l))