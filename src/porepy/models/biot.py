import porepy as pp
import numpy as np

from porepy.models.poromechanics import Poromechanics, ConstitutiveLawsPoromechanics


# class ConstitutiveLawsBiot(
#     pp.constitutive_laws.SpecificStorage,
#     pp.constitutive_laws.BiotPoromechanicsPorosity,
#     pp.constitutive_laws.ConstantFluidDensity,
#     ConstitutiveLawsPoromechanics
# ):
#     ...

class ConstitutiveLawsBiot(
    # Combined effects
    pp.constitutive_laws.DisplacementJumpAperture,
    pp.constitutive_laws.BiotCoefficient,
    pp.constitutive_laws.SpecificStorage,
    pp.constitutive_laws.PressureStress,
    pp.constitutive_laws.BiotPoromechanicsPorosity,
    # Fluid mass balance subproblem
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.FluidMobility,
    pp.constitutive_laws.ConstantPermeability,
    pp.constitutive_laws.ConstantFluidDensity,
    pp.constitutive_laws.ConstantViscosity,
    # Mechanical subproblem
    pp.constitutive_laws.LinearElasticSolid,
    pp.constitutive_laws.FracturedSolid,
    pp.constitutive_laws.FrictionBound,
):
    """Class for the coupling of mass and momentum balance to obtain Biot equations.

    """

    def stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Stress operator.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """
        # Method from constitutive library's LinearElasticRock.
        return self.mechanical_stress(subdomains) + self.pressure_stress(subdomains)


class BiotPoromechanics(  # type: ignore[misc]
    ConstitutiveLawsBiot,
    Poromechanics,
):
    ...


#%% Runner
params = {}
setup = BiotPoromechanics(params)
pp.run_time_dependent_model(setup, params)
