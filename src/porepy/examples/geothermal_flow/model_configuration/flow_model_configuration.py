
import porepy as pp
import numpy as np

from porepy.models.compositional_flow import (
    CompositionalFlowTemplate
    # CompositionalFractionalFlowTemplate
)

from .constitutive_description.mixture_constitutive_description import (
    FluidMixture,
    SecondaryEquations,
    ComponentSystem,
    PhaseMode,
)

from .geometry_description.geometry_market import SimpleGeometry as ModelGeometry


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


class VaporSecondaryEquation(SecondaryEquations):
    component_system = ComponentSystem.WATER
    phase_mode = pp.compositional.PhysicalState.gas


class TwoPhaseSecondaryEquations(SecondaryEquations):
    component_system = ComponentSystem.WATER
    phase_mode = PhaseMode.TWO_PHASE


class ThreePhaseSecondaryEquations(SecondaryEquations):
    component_system = ComponentSystem.WATER_SALT
    phase_mode = PhaseMode.THREE_PHASE


class SinglePhaseFlowModelConfigurationVapor(
    ModelGeometry,
    FluidMixture,
    VaporSecondaryEquation,
    CompositionalFlowTemplate,
    VTKSamplerMixin
):
    def relative_permeability(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return phase.saturation(domains)


class SinglePhaseFlowModelConfigurationLiquid(
    ModelGeometry,
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


class TwoPhaseFlowModelConfiguration(
    ModelGeometry,
    FluidMixture,
    TwoPhaseSecondaryEquations,
    CompositionalFlowTemplate,
    VTKSamplerMixin
):

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


def saturations(self) -> dict[str, pp.ad.Operator]:
    """
    Evaluate physically meaningful phase saturations using the VTKSampler,
    based on current values of pressure, enthalpy, and NaCl mass fraction z.

    Parameters:
        pressure: AD operator for pressure.
        enthalpy: AD operator for enthalpy.
        z: Optional AD operator for salt mass fraction (only for multi-component case).

    Returns:
        A dictionary with AD-wrapped saturation operators:
            - "S_l": liquid saturation
            - "S_v": vapor saturation
            - "S_h": halite saturation
    """
    # === Evaluate AD operators ===
    p = self.pressure(self.domains)
    h = self.enthalpy(self.domains)
    z_vals = self.z(self.domains)

    # === Build parameter points ===
    par_points = np.array((z_vals, h, p)).T  # Shape: (ncells, 3)

    # === Sample saturations from the VTK file ===
    self.vtk_sampler.sample_at(par_points)

    s_l = self.vtk_sampler.sampled_could.point_data["S_l"]
    s_v = self.vtk_sampler.sampled_could.point_data["S_v"]
    s_h = self.vtk_sampler.sampled_could.point_data["S_h"]

    # === Clamp halite saturation to [0, 1] ===
    s_h = np.clip(s_h, 0.0, 1.0)

    # === Return AD-wrapped results ===
    return {
        "S_l": pp.ad.wrap_as_ad_array(s_l),
        "S_v": pp.ad.wrap_as_ad_array(s_v),
        "S_h": pp.ad.wrap_as_ad_array(s_h),
    }


class ThermalConductivityMixinWithClampedHalite(pp.PorePyModel):
    """
    Mixin to compute thermal conductivity with halite saturation clamped to [0, 1].

    This avoids divergence or non-physical behavior caused by Newton updates that
    temporarily drive saturation outside [0, 1].
    """
    def fluid_thermal_conductivity(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """
        Computes effective thermal conductivity as:
            κ_eff = sum_j S_j * κ_j

        where S_j is clamped to [0, 1] for the halite phase only.
        """
        max_fn = pp.ad.Function(pp.ad.maximum, name="maximum_function")

        def min_fn(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
            return -max_fn(-a, -b)

        ops = []
        if self.fluid.num_phases > 1:
            ref_sat = self.fluid.reference_phase.saturation(domains)
            halite_phase = [p for p in self.fluid.phases if p.name == "halite"]
            s_h = halite_phase[0].saturation(domains)

            # Clamp only once
            s_ref_clamped = min_fn(pp.ad.Scalar(1.0), max_fn(ref_sat, pp.ad.Scalar(0.0)))
            s_h_clamped = min_fn(pp.ad.Scalar(0.5), max_fn(s_h, pp.ad.Scalar(0.0)))

            for phase in self.fluid.phases:
                if phase.name.lower() == "halite":
                    saturation = s_h_clamped
                elif phase == self.fluid.reference_phase:
                    saturation = s_ref_clamped
                else:
                    inferred = pp.ad.Scalar(1.0) - (s_ref_clamped + s_h_clamped)
                    saturation = min_fn(pp.ad.Scalar(1.0), max_fn(inferred, pp.ad.Scalar(0.0)))

                kappa = phase.thermal_conductivity(domains)
                ops.append(saturation * kappa)

            op = pp.ad.sum_operator_list(ops, name="fluid_thermal_conductivity")
        else:
            op = self.fluid.reference_phase.thermal_conductivity(domains)
            op.set_name("fluid_thermal_conductivity")

        return op

    def fluid_thermal_conductivity_v1(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """
        Computes effective thermal conductivity as:
            κ_eff = sum_j S_j * κ_j

        where S_j is clamped to [0, 1] for the halite phase only.
        """
        max_fn = pp.ad.Function(pp.ad.maximum, name="maximum_function")

        def min_fn(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
            return -max_fn(-a, -b)

        ops = []
        if self.fluid.num_phases > 1:
            ref_sat = self.fluid.reference_phase.saturation(domains)
            halite_phase = [p for p in self.fluid.phases if p.name == "halite"]
            s_h = halite_phase[0].saturation(domains)
            for phase in self.fluid.phases:
                if phase.name.lower() == "halite":
                    # Clamp S_h to [0, 1]
                    saturation = min_fn(pp.ad.Scalar(0.5), max_fn(s_h, pp.ad.Scalar(0.0)))
                elif phase == self.fluid.reference_phase:
                    saturation = min_fn(pp.ad.Scalar(1.0), max_fn(ref_sat, pp.ad.Scalar(0.0)))
                else:
                    s_ref = min_fn(pp.ad.Scalar(1.0), max_fn(ref_sat, pp.ad.Scalar(0.0)))
                    s_hal = min_fn(pp.ad.Scalar(0.5), max_fn(s_h, pp.ad.Scalar(0.0)))
                    inferred = 1.0 - (s_ref + s_hal)
                    saturation = min_fn(pp.ad.Scalar(1.0), max_fn(inferred, pp.ad.Scalar(0.0)))

                kappa = phase.thermal_conductivity(domains)
                ops.append(saturation * kappa)

            op = pp.ad.sum_operator_list(ops, name="fluid_thermal_conductivity")
        else:
            op = self.fluid.reference_phase.thermal_conductivity(domains)
            op.set_name("fluid_thermal_conductivity")

        return op

    def thermal_conductivity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal conductivity [m^2].

        The thermal conductivity is computed as the porosity-weighted average of the
        fluid and solid thermal conductivities. In this implementation, both are
        considered constants, however, if the porosity changes with time, the weighting
        factor will also change.

        Parameters:
            subdomains: List of subdomains where the thermal conductivity is defined.

        Returns:
            Cell-wise conducivity operator.

        """
        phi = self.porosity(subdomains)
        # Since thermal conductivity is used as a discretization parameter, it has to be
        # evaluated before the discretization matrices are computed.
        try:
            self.equation_system.evaluate(phi)
        except KeyError:
            # We assume this means that the porosity includes a discretization matrix
            # for displacement_divergence which has not yet been computed.
            phi = self.reference_porosity(subdomains)
        if isinstance(phi, pp.ad.Scalar):
            size = sum(sd.num_cells for sd in subdomains)
            phi = phi * pp.wrap_as_dense_ad_array(1, size)
        conductivity = phi * self.fluid_thermal_conductivity(subdomains) + (
            pp.ad.Scalar(1.0) - phi
        ) * self.solid_thermal_conductivity(subdomains)

        return self.isotropic_second_order_tensor(subdomains, conductivity)


class PorosityWithHaliteMixin(pp.PorePyModel):
    """
    Porosity model that reduces effective porosity based on halite saturation.

    Assumes that the presence of halite reduces the pore volume available to fluid phases.
    """

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # Base porosity from solid
        phi_0 = pp.ad.Scalar(self.solid.porosity, name="porosity")

        # Retrieve halite phase (must be present in self.fluid.phases)
        halite_phase = [p for p in self.fluid.phases if p.name == "halite"]
        if len(halite_phase) != 1:
            raise ValueError("Exactly one halite phase required for porosity correction.")

        s_h_raw = halite_phase[0].saturation(subdomains)

        # Clamp s_h to [0, 0.3]   
        maximum_fn = pp.ad.Function(pp.ad.maximum, "max_fn")

        def minimum_fn(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
            return -maximum_fn(-a, -b)

        s_h_clamped = minimum_fn(
            pp.ad.Scalar(0.5),
            maximum_fn(s_h_raw, pp.ad.Scalar(0.0))
        )

        # Effective porosity: phi = phi_0 * (1 - s_halite)
        return phi_0 * (1.0 - s_h_clamped)


class PermeabilityWithHaliteMixin(pp.PorePyModel):

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        size = sum(sd.num_cells for sd in subdomains)

        base_perm = pp.wrap_as_dense_ad_array(
            self.solid.permeability, size, 
            name="permeability"
        )

        halite_phase = [p for p in self.fluid.phases if p.name == "halite"]
        if len(halite_phase) != 1:
            raise ValueError("Exactly one halite phase required for permeability correction.")

        s_h_raw = halite_phase[0].saturation(subdomains)

        # Clamp s_h to [0, 0.5]   
        maximum_fn = pp.ad.Function(pp.ad.maximum, "max_fn")

        def minimum_fn(a: pp.ad.Operator, b: pp.ad.Operator) -> pp.ad.Operator:
            return -maximum_fn(-a, -b)
 
        s_h_clamped = minimum_fn(
            pp.ad.Scalar(0.5),
            maximum_fn(s_h_raw, pp.ad.Scalar(0.0))
        )

        # Example reduction: perm_eff = perm_0 * (1 - s_halite)^2
        reduction = (1.0 - s_h_clamped) ** 2
        corrected_perm = base_perm*reduction

        return self.isotropic_second_order_tensor(subdomains, corrected_perm)
    

class ThreePhaseFlowModelConfiguration(
    PorosityWithHaliteMixin,
    PermeabilityWithHaliteMixin,
    ThermalConductivityMixinWithClampedHalite,
    ModelGeometry,
    FluidMixture,
    ThreePhaseSecondaryEquations,
    CompositionalFlowTemplate,
    VTKSamplerMixin
):

    def relative_permeability(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        
        epsilon = pp.ad.Scalar(0.0)
        halite_phase = [p for p in self.fluid.phases if p.name == "halite"]

        if len(halite_phase) != 1:
            raise ValueError("Expected exactly one halite phase.")
        
        max = pp.ad.Function(pp.ad.maximum, "maximum_function")

        # name = phase.name
        s = phase.saturation(domains)

        # Total mobile pore volume
        mobile_pore_volume = pp.ad.Scalar(1.0)  # (1-s_halite)

        # Define residual saturations
        r_l = mobile_pore_volume * pp.ad.Scalar(0.3)
        r_v = pp.ad.Scalar(0.0)

        # Choose appropriate residual saturation
        if phase.name == "halite":
            return pp.ad.Scalar(0.0) * s
        
        if phase == self.fluid.reference_phase:
            s_eff = (s - r_l) / (1.0 - r_l - r_v)
            return max(s_eff, epsilon)
        else:
            return (s - r_v) / (1.0 - r_l - r_v)
