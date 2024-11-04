"""Module testing the creation of fluid mixtures and their mixins."""

from __future__ import annotations

import numpy as np
import pytest
from numpy import ndarray

import porepy as pp
import porepy.compositional as composit


@pytest.fixture(scope="module")
def dummyeos():
    """Dummy Eos"""

    class DummyEos(composit.AbstractEoS):

        def compute_phase_properties(
            self, 
            phase_type: int, 
            *thermodynamic_input: ndarray
        ) -> composit.PhaseProperties:
            pass

    return DummyEos


class MockModel(
    composit.FluidMixtureMixin,
    composit.CompositionalVariables,
    pp.BoundaryConditionMixin,
):
    """For typing support when using methods of the mixins."""

    pass


def get_mock_model(
    components,
    phase_configuration,
    eliminate_reference,
    equilibrium_type,
):

    # simple set up, 1 cell domain, with 1 boundary grid
    g = pp.CartGrid(np.array([1, 1]))
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains(g)
    mdg.compute_geometry()
    mdg.set_boundary_grid_projections()
    eqsys = pp.ad.EquationSystem(mdg)

    # mimicing pressure and temperatur variable
    def pressure(grid):
        return pp.ad.Variable("pressure", {"cells": 1}, grid[0])

    def temperature(grid):
        return pp.ad.Variable("pressure", {"cells": 1}, grid[0])

    class Model(
        composit.FluidMixtureMixin,
        composit.CompositionalVariables,
        pp.BoundaryConditionMixin,
    ):

        def get_components(self) -> list[composit.Component]:
            return components

        def get_phase_configuration(
            self, components: np.Sequence[composit.Component]
        ) -> np.Sequence[tuple[composit.AbstractEoS, int, str]]:
            return phase_configuration

        # for simplicity of testing
        def dependencies_of_phase_properties(self, phase: composit.Phase):
            return [self.pressure, self.temperature]

    mixin = Model()

    mixin.mdg = mdg
    mixin.equation_system = eqsys
    mixin.params = {
        "eliminate_reference_component": eliminate_reference,
        "eliminate_reference_phase": eliminate_reference,
        "equilibrium_type": equilibrium_type,
    }
    mixin.pressure = pressure
    mixin.temperature = temperature

    return mixin


@pytest.mark.parametrize(
    "phaseconfig",
    [
        [],  # shoud lead to error if no phase
        [("L", composit.PhysicalState.liquid)],
        [
            ("L", composit.PhysicalState.liquid),
            ("L", composit.PhysicalState.liquid),
        ],  # error since same name
        [
            ("G", composit.PhysicalState.gas),
            ("L", composit.PhysicalState.liquid),
        ],  # should re-order s.t. L is ref phase and G at end
        [
            ("L", composit.PhysicalState.gas),
            ("G", composit.PhysicalState.gas),
        ],  # error since two gas phases
        [
            ("L1", composit.PhysicalState.liquid),
            ("L2", composit.PhysicalState.liquid),
            ("G", composit.PhysicalState.gas),
        ],
    ],
)
@pytest.mark.parametrize(
    "species",
    [
        [],  # error if no component
        ["H2O"],
        ["H2O", "H2O"],  # error since same name
        ["H2O", "CO2"],
    ],
)
def test_mixture_contexts(
    species: list[str], phaseconfig: list[list[tuple[str, int]]], dummyeos
):
    """Tests the phase and component context of a fluid mixture, and the assumptions
    on wich the framework is built. They must not be violated by any future development.
    """

    nphase = len(phaseconfig)
    ncomp = len(species)

    species_kwargs = {
        "molar_mass": 1.0,
        "p_crit": 1.0,
        "T_crit": 1.0,
        "V_crit": 1.0,
        "omega": 1.0,
    }

    # h2o = composit.Component.from_species(composit.load_species(["H2O"])[0])
    # Creating dummy components. Physical properties have no relevance for this test

    # 1 separate component for the dummy eos, just to instantiate it.
    h2o = composit.Component(name="H2O", CASr_number="1", **species_kwargs)

    # components: list[composit.Component] = [
    #     composit.Component.from_species(s) for s in composit.load_species(species)
    # ]
    components: list[composit.Component] = [
        composit.Component(name=s, CASr_number=f"{i}", **species_kwargs)
        for i, s in enumerate(species)
    ]

    if ncomp == 0:
        with pytest.raises(composit.CompositionalModellingError):
            _ = dummyeos(components)
        eos: composit.AbstractEoS = dummyeos([h2o])
    else:
        eos: composit.AbstractEoS = dummyeos(components)
    phases: list[composit.Phase] = []
    has_gas = False
    has_more_gas = False
    for conf in phaseconfig:
        name, t = conf
        if t == composit.PhysicalState.gas:
            if has_gas:
                has_more_gas = True
            has_gas = True
        phases.append(composit.Phase(eos, t, name))
        phases[-1].components = [h2o] + components  # to avoid errors

    phasenames = [phase.name for phase in phases]
    compnames = [comp.name for comp in components]

    # cannot create mixtures with no components or phases
    if ncomp == 0 or nphase == 0:
        # need to catch this extra because of parametrization
        if (len(set(phasenames)) < nphase or len(set(compnames)) < ncomp) and (
            ncomp > 0 or nphase > 0
        ):
            with pytest.raises(ValueError):
                composit.FluidMixture(components, phases)
        else:
            with pytest.raises(composit.CompositionalModellingError):
                composit.FluidMixture(components, phases)
    # cannot create mixtures with duplicate names
    elif len(set(phasenames)) < nphase or len(set(compnames)) < ncomp:
        with pytest.raises(ValueError):
            composit.FluidMixture(components, phases)
    # more than 1 gas phase not allowed
    elif has_more_gas:
        with pytest.raises(composit.CompositionalModellingError):
            composit.FluidMixture(components, phases)
    # else the creation should not raise an error, and we check phase ordering
    else:
        mix = composit.FluidMixture(components, phases)

        ordered_phases = [p for p in mix.phases]
        ordered_comps = [c for c in mix.components]
        if has_gas:
            assert ordered_phases[-1].state == composit.PhysicalState.gas

        # asert that the first phase and components are always the reference
        assert ordered_phases[0] == mix.reference_phase
        assert ordered_comps[0] == mix.reference_component

        # check that we cannot create mixtures if a phase has no components
        phases[0].components = []
        with pytest.raises(composit.CompositionalModellingError):
            composit.FluidMixture(components, phases)


@pytest.mark.parametrize(
    "eliminate_reference", [True, False]  # for component and phase, test if eliminated
)
@pytest.mark.parametrize("equilibrium_type", [None, "unified-p-T", "p-T"])
def test_mixture_member_assignment(
    eliminate_reference: bool, equilibrium_type: None | str, dummyeos
):
    """Testint that all requried members of phases, components, compounds and
    fluid mixtures are assigned by the compositional mixins. Tested with and without
    independent reference component/phase fractions."""

    species_kwargs = {
        "molar_mass": 1.0,
        "p_crit": 1.0,
        "T_crit": 1.0,
        "V_crit": 1.0,
        "omega": 1.0,
    }
    # Creating dummy components. Physical properties have no relevance for this test

    comp1 = composit.Compound.from_species(
        composit.ChemicalSpecies(name="H2O", CASr_number="1", **species_kwargs)
    )
    comp1.active_tracers = [
        composit.ChemicalSpecies(name="NaCl", CASr_number="2", **species_kwargs)
    ]
    comp2 = composit.Component.from_species(
        composit.ChemicalSpecies(name="CO2", CASr_number="3", **species_kwargs)
    )
    eos = dummyeos([comp1, comp2])

    mixin: MockModel = get_mock_model(
        [comp1, comp2],
        [(eos, 0, "L"), (eos, 1, "G")],
        eliminate_reference,
        equilibrium_type,
    )

    sds = mixin.mdg.subdomains()
    bgs = mixin.mdg.boundaries()
    ncomp = 2
    nphase = 2

    mixin.create_mixture()
    mixin.create_variables()
    # NOTE if None, there should be no phase fractions, ergo no mixture enthalpy.
    # but we implemented a dummy above
    mixin.assign_thermodynamic_properties_to_mixture()

    # The names should reflect the created variables
    # ref partial fraction eliminated in case of no equilibrium
    if equilibrium_type is None:
        assert len(mixin.fraction_in_phase_variables) == (ncomp - 1) * nphase
    # ref partialf raction also eliminated in the case of non-unified equilibriu,
    elif "unified" not in equilibrium_type:
        assert len(mixin.fraction_in_phase_variables) == (ncomp - 1) * nphase
    # Extended fractions are always independent, if unified equilibrium
    elif "unified" in equilibrium_type:
        assert len(mixin.fraction_in_phase_variables) == ncomp * nphase
    # there are always solute fraction variables in compounds
    assert len(mixin.tracer_fraction_variables) == 1
    # test that the eliminated variables were not created
    if eliminate_reference:
        assert len(mixin.overall_fraction_variables) == ncomp - 1
        assert len(mixin.saturation_variables) == nphase - 1
        if equilibrium_type is not None:
            assert len(mixin.phase_fraction_variables) == nphase - 1
        else:
            assert len(mixin.phase_fraction_variables) == 0
    else:
        assert len(mixin.overall_fraction_variables) == ncomp
        assert len(mixin.saturation_variables) == nphase
        if equilibrium_type is not None:
            assert len(mixin.phase_fraction_variables) == nphase
        else:
            assert len(mixin.phase_fraction_variables) == 0

    # Mixture is set up. First check phases and everything related to them
    for phase in mixin.fluid_mixture.phases:
        assert hasattr(phase, "saturation")
        assert hasattr(phase, "density")
        assert hasattr(phase, "specific_volume")
        assert hasattr(phase, "specific_enthalpy")
        assert hasattr(phase, "viscosity")
        assert hasattr(phase, "conductivity")
        assert hasattr(phase, "partial_fraction_of")

        # NOTE fraction and extended fractions are assigned in any case
        # They cause errors when evaluated though
        assert hasattr(phase, "fraction")
        assert hasattr(phase, "extended_fraction_of")

        # fractions should be defined in equilibrium setting
        if mixin._has_equilibrium:
            if mixin.has_independent_fraction(phase):
                assert isinstance(phase.fraction(sds), pp.ad.Variable)
                assert isinstance(phase.fraction(bgs), pp.ad.TimeDependentDenseArray)
            else:
                assert isinstance(phase.fraction(sds), pp.ad.Operator)
                assert isinstance(phase.fraction(bgs), pp.ad.Operator)
        # If no equilibrium setting, calling phase fractions should yield an error
        else:
            with pytest.raises(composit.CompositionalModellingError):
                phase.fraction(sds)
            for comp in phase:
                with pytest.raises(composit.CompositionalModellingError):
                    phase.extended_fraction_of[comp](sds)

        # extended fraction only in unified equilibrium setting
        if mixin._has_unified_equilibrium:
            for comp in phase:
                assert isinstance(phase.extended_fraction_of[comp](sds), pp.ad.Variable)
                assert isinstance(
                    phase.extended_fraction_of[comp](bgs), pp.ad.TimeDependentDenseArray
                )
                # partial fractions are not variables in unified equilibrium
                assert not isinstance(
                    phase.partial_fraction_of[comp](sds), pp.ad.Variable
                )
                assert not isinstance(
                    phase.partial_fraction_of[comp](bgs), pp.ad.TimeDependentDenseArray
                )
                assert isinstance(phase.partial_fraction_of[comp](sds), pp.ad.Operator)
                assert isinstance(phase.partial_fraction_of[comp](bgs), pp.ad.Operator)
        else:
            # Extended fraction without unified equilibrium make no sense
            for comp in phase:
                with pytest.raises(composit.CompositionalModellingError):
                    phase.extended_fraction_of[comp](sds)
            # Both cases, non-unified equilibrium and no equilibrium have independent
            # partial fractions
            for comp in phase:
                # reference componet partial fraction is always eliminated
                if mixin.has_independent_partial_fraction(comp, phase):
                    assert isinstance(
                        phase.partial_fraction_of[comp](sds), pp.ad.Variable
                    )
                    assert isinstance(
                        phase.partial_fraction_of[comp](bgs),
                        pp.ad.TimeDependentDenseArray,
                    )
                else:
                    assert comp == phase.reference_component
                    assert not isinstance(
                        phase.partial_fraction_of[comp](sds), pp.ad.Variable
                    )
                    assert isinstance(
                        phase.partial_fraction_of[comp](sds), pp.ad.Operator
                    )
                    assert not isinstance(
                        phase.partial_fraction_of[comp](bgs),
                        pp.ad.TimeDependentDenseArray,
                    )
                    assert isinstance(
                        phase.partial_fraction_of[comp](bgs), pp.ad.Operator
                    )

        # Check that the reference phase saturation is a dependent operator
        if not mixin.has_independent_saturation(phase):
            assert phase == mixin.fluid_mixture.reference_phase
            # reference phase fraction and saturations are dependent operators
            # if eliminated
            assert not isinstance(phase.saturation(sds), pp.ad.Variable)
            assert isinstance(phase.saturation(sds), pp.ad.Operator)
            assert not isinstance(phase.saturation(bgs), pp.ad.TimeDependentDenseArray)
            assert isinstance(phase.saturation(bgs), pp.ad.Operator)
            # same holds for reference phase fraction in the equilibrium setting
            if mixin._has_equilibrium:
                assert not isinstance(phase.fraction(sds), pp.ad.Variable)
                assert isinstance(phase.fraction(sds), pp.ad.Operator)
                assert not isinstance(
                    phase.fraction(bgs), pp.ad.TimeDependentDenseArray
                )
                assert isinstance(phase.fraction(bgs), pp.ad.Operator)
        else:
            # otherwise it must be a variable
            assert isinstance(phase.saturation(sds), pp.ad.Variable)
            if mixin._has_equilibrium:
                assert isinstance(phase.fraction(sds), pp.ad.Variable)

            # fraction and saturation on boundaries are time-dependent dense arrays
            assert isinstance(phase.saturation(bgs), pp.ad.TimeDependentDenseArray)
            if mixin._has_equilibrium:
                assert isinstance(phase.saturation(bgs), pp.ad.TimeDependentDenseArray)

        # Now check the thermodynamic properties.
        # On domains it must be some kind of Operator (use most basic for check)
        # On boundaries we expect a time-dependent dense array
        assert isinstance(phase.density(sds), pp.ad.Operator)
        assert isinstance(phase.specific_enthalpy(sds), pp.ad.Operator)
        assert isinstance(phase.specific_volume(sds), pp.ad.Operator)
        assert isinstance(phase.viscosity(sds), pp.ad.Operator)
        assert isinstance(phase.conductivity(sds), pp.ad.Operator)

        assert isinstance(phase.density(bgs), pp.ad.TimeDependentDenseArray)
        assert isinstance(phase.specific_enthalpy(bgs), pp.ad.TimeDependentDenseArray)
        # NOTE Volume is taken as the reciprocal of density, hence a general operator
        assert isinstance(phase.specific_volume(bgs), pp.ad.Operator)
        assert isinstance(phase.viscosity(bgs), pp.ad.TimeDependentDenseArray)
        assert isinstance(phase.conductivity(bgs), pp.ad.TimeDependentDenseArray)

        # Fugacities are always created as well
        for comp in phase:
            assert comp in phase.fugacity_coefficient_of
            assert isinstance(phase.fugacity_coefficient_of[comp](sds), pp.ad.Operator)
            assert isinstance(
                phase.fugacity_coefficient_of[comp](bgs), pp.ad.TimeDependentDenseArray
            )

    # Check components and their overall fractions
    for comp in mixin.fluid_mixture.components:
        assert hasattr(comp, "fraction")

        # Overall fraction of reference component is eliminated (if requested)
        if not mixin.has_independent_fraction(comp):
            assert comp == mixin.fluid_mixture.reference_component
            assert not isinstance(comp.fraction(sds), pp.ad.Variable)
            assert isinstance(comp.fraction(sds), pp.ad.Operator)
            assert not isinstance(comp.fraction(bgs), pp.ad.TimeDependentDenseArray)
            assert isinstance(comp.fraction(bgs), pp.ad.Operator)
        else:
            assert isinstance(comp.fraction(sds), pp.ad.Variable)
            assert isinstance(comp.fraction(bgs), pp.ad.TimeDependentDenseArray)

        # IF it is a compound, check relative fractions of pseudo components
        if isinstance(comp, composit.Compound):
            assert hasattr(comp, "tracer_fraction_of")
            for pc in comp.active_tracers:
                assert pc in comp.tracer_fraction_of
                # solute fractions are aalways variables
                assert isinstance(comp.tracer_fraction_of[pc](sds), pp.ad.Variable)
                assert isinstance(
                    comp.tracer_fraction_of[pc](bgs), pp.ad.TimeDependentDenseArray
                )

    # Finally, check the assignment of the overall properties of the mixture
    # Density, volume and enthalpy
    # The mixture must implement thermodynamic laws, they must not be variables
    assert isinstance(mixin.fluid_mixture.density(sds), pp.ad.Operator)
    assert not isinstance(mixin.fluid_mixture.density(sds), pp.ad.Variable)
    assert isinstance(mixin.fluid_mixture.specific_volume(sds), pp.ad.Operator)
    assert not isinstance(mixin.fluid_mixture.specific_volume(sds), pp.ad.Variable)
    if mixin._has_equilibrium:
        assert isinstance(mixin.fluid_mixture.specific_enthalpy(sds), pp.ad.Operator)
        assert not isinstance(
            mixin.fluid_mixture.specific_enthalpy(sds), pp.ad.Variable
        )
    else:
        # the basic defintion of enthalpy relies on phase fractions, which are not
        # available without equilibrium
        with pytest.raises(composit.CompositionalModellingError):
            mixin.fluid_mixture.specific_enthalpy(sds)


# Parametrization to test for any combination
@pytest.mark.parametrize(
    "equilibrium_type", [None, "unified-p-T", "p-T"]  # for None, no y or extended
)
@pytest.mark.parametrize("phase_names", [["L"], ["L", "G"]])
@pytest.mark.parametrize("species", [["H2O"], ["H2O", "CO2"]])
def test_singular_mixtures(species, phase_names, equilibrium_type, dummyeos):
    """Testing the behavior when only 1 component, or 1 phase or both.
    In this case, the number of created variables follows certain rules."""

    species_kwargs = {
        "molar_mass": 1.0,
        "p_crit": 1.0,
        "T_crit": 1.0,
        "V_crit": 1.0,
        "omega": 1.0,
    }

    # Creating dummy components. Physical properties have no relevance for this test
    components: list[composit.Component] = [
        composit.Component(name=s, CASr_number=f"{i}", **species_kwargs)
        for i, s in enumerate(species)
    ]

    # components = [
    #     composit.Component.from_species(s) for s in composit.load_species(species)
    # ]
    eos = dummyeos(components)
    phases = [(eos, 0, name) for name in phase_names]

    mixin: MockModel = get_mock_model(components, phases, True, equilibrium_type)

    sds = mixin.mdg.subdomains()

    mixin.create_mixture()
    mixin.create_variables()
    # NOTE if None, there should be no phase fractions, ergo no mixture enthalpy.
    # but we implemented a dummy above
    mixin.assign_thermodynamic_properties_to_mixture()

    nphase = mixin.fluid_mixture.num_phases
    ncomp = mixin.fluid_mixture.num_components

    if nphase == 1 and ncomp > 1:
        # In this singular case, the partial fractions are not independent
        # Neither are the phase fraction and saturation
        phase = list(mixin.fluid_mixture.phases)[0]
        assert phase == mixin.fluid_mixture.reference_phase
        assert not isinstance(phase.saturation(sds), pp.ad.Variable)
        if mixin._has_equilibrium:
            assert not isinstance(phase.fraction(sds), pp.ad.Variable)

        for comp in mixin.fluid_mixture.components:
            assert comp in phase
            assert comp in phase.extended_fraction_of
            assert comp in phase.partial_fraction_of
            # In the case of 1 phase, partial fractions are equal overall fractions
            assert comp.fraction == phase.partial_fraction_of[comp]
            # Extended fractions are always variable
            if mixin._has_unified_equilibrium:
                assert isinstance(phase.extended_fraction_of[comp](sds), pp.ad.Variable)

            # checking overall fraction
            if mixin.has_independent_fraction(comp):
                assert comp != mixin.fluid_mixture.reference_component
                assert isinstance(comp.fraction(sds), pp.ad.Variable)
            else:
                assert comp == mixin.fluid_mixture.reference_component
                assert not isinstance(comp.fraction(sds), pp.ad.Variable)
    elif nphase > 1 and ncomp == 1:
        # In this case, there is no overall fraction
        # Partial fractions are also not independent, since only 1 component per phase
        # Extended fractions remain independent
        comp = list(mixin.fluid_mixture.components)[0]
        assert comp == mixin.fluid_mixture.reference_component
        assert not isinstance(comp.fraction(sds), pp.ad.Variable)

        for phase in mixin.fluid_mixture.phases:
            if mixin.has_independent_saturation(phase):
                assert isinstance(phase.saturation(sds), pp.ad.Variable)
                if mixin._has_equilibrium:
                    assert isinstance(phase.fraction(sds), pp.ad.Variable)
                else:
                    with pytest.raises(composit.CompositionalModellingError):
                        phase.fraction(sds)
            else:
                assert phase == mixin.fluid_mixture.reference_phase
                assert not isinstance(phase.saturation(sds), pp.ad.Variable)
                if mixin._has_equilibrium:
                    assert not isinstance(phase.fraction(sds), pp.ad.Variable)
                else:
                    with pytest.raises(composit.CompositionalModellingError):
                        phase.fraction(sds)

            # no phase without the single component
            assert comp in phase.components
            assert comp in phase.partial_fraction_of
            assert comp in phase.extended_fraction_of
            assert comp == phase.reference_component
            assert not isinstance(phase.partial_fraction_of[comp](sds), pp.ad.Variable)
            # Extended fractions remain variables, in case phase vanishes
            if mixin._has_unified_equilibrium:
                assert isinstance(phase.extended_fraction_of[comp](sds), pp.ad.Variable)
    elif nphase == 1 and ncomp == 1:
        # In this singular case, no fractional variable is independent
        comp = list(mixin.fluid_mixture.components)[0]
        phase = list(mixin.fluid_mixture.phases)[0]

        assert comp == mixin.fluid_mixture.reference_component
        assert phase == mixin.fluid_mixture.reference_phase
        assert comp == phase.reference_component

        assert not isinstance(comp.fraction(sds), pp.ad.Variable)
        assert not isinstance(phase.fraction(sds), pp.ad.Variable)
        assert not isinstance(phase.saturation(sds), pp.ad.Variable)
        assert not isinstance(phase.partial_fraction_of[comp](sds), pp.ad.Variable)
        assert not isinstance(phase.extended_fraction_of[comp](sds), pp.ad.Variable)
