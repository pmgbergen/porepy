import porepy as pp
from porepy.models.compositional_flow import (
    CFModelMixin,
    PrimaryEquationsCF,
)

from .constitutive_description.mixture_constitutive_description import (
    FluidMixture,
    SecondaryEquations,
)
from .geometry_description.geometry_market import SimpleGeometry as ModelGeometry


class ModelEquations(
    PrimaryEquationsCF,
    SecondaryEquations,
):
    """Collecting primary flow and transport equations, and secondary equations
    which provide substitutions for independent saturations and partial fractions.
    """

    def set_equations(self):
        """Call to the equation. Parent classes don't use super(). User must provide
        proper order resultion.
        """
        # Flow and transport in MD setting
        PrimaryEquationsCF.set_equations(self)
        # local elimination of dangling secondary variables
        SecondaryEquations.set_equations(self)


class SinglePhaseFlowModelConfiguration(
    ModelGeometry,
    FluidMixture,
    ModelEquations,
    CFModelMixin,
):
    def relative_permeability(
        self,
        saturation: pp.ad.Operator
    ) -> pp.ad.Operator:
        return saturation

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


class TwoPhaseFlowModelConfiguration(
    ModelGeometry,
    FluidMixture,
    ModelEquations,
    CFModelMixin,
):  
    def relative_permeability(
        self, 
        saturation: pp.ad.Operator
    ) -> pp.ad.Operator:

        Rl = 0.3  # residual saturation of the liquid phase. (non-wetting phase)
        if saturation.name == "reference_phase_saturation_by_unity":
            krl = (saturation - pp.ad.Scalar(Rl)) / (pp.ad.Scalar(1.0) - pp.ad.Scalar(Rl))
            krl_positive = pp.ad.Scalar(0.5) * (krl + (krl**2)**0.5)
            return krl_positive
        return saturation / (pp.ad.Scalar(1.0) - pp.ad.Scalar(Rl))
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