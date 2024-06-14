"""Module testing the creation of fluid mixtures and their mixins."""
from __future__ import annotations

from numpy import ndarray
import pytest

import numpy as np
import porepy as pp
import porepy.compositional as composit


@pytest.fixture(scope='module')
def dummyeos():
    """Dummy Eos"""

    class DummyEos(composit.AbstractEoS):

        def compute_phase_state(
            self, phase_type: int, *thermodynamic_input: ndarray
        ) -> composit.PhaseProperties:
            pass

    return DummyEos


class MockModel(composit.FluidMixtureMixin, composit.CompositionalVariables, pp.BoundaryConditionMixin):
    """For typing support when using methods of the mixins."""
    pass


def get_mock_model(
    components,
    phase_configuration,
    eliminate_reference,
    equilibrium_type,
):

    # simple set up, 1 cell domain, with 1 boundary grid
    g = pp.CartGrid(np.array([1,1]))
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains(g)
    mdg.compute_geometry()
    mdg.set_boundary_grid_projections()
    eqsys = pp.ad.EquationSystem(mdg)

    # mimicing pressure and temperatur variable
    def pressure(grid):
        return pp.ad.Variable('pressure', {'cells': 1}, grid[0])
    def temperature(grid):
        return pp.ad.Variable('pressure', {'cells': 1}, grid[0])
    
    
    class Model(composit.FluidMixtureMixin, composit.CompositionalVariables, pp.BoundaryConditionMixin):
        
        def get_components(self) -> list[composit.Component]:
            return components
        
        def get_phase_configuration(
            self, components: np.Sequence[composit.Component]
        ) -> np.Sequence[tuple[composit.AbstractEoS, int, str]]:
            return phase_configuration

        # for simplicity of testing
        def dependencies_of_phase_properties(
                self, phase: composit.Phase
        ):
            return [self.pressure, self.temperature]
        
        # Some dummy enthalpy, for the test case without equilibrium type
        def fluid_specific_enthalpy(
                self, domains: np.Sequence[pp.Grid] | np.Sequence[pp.BoundaryGrid]
        ) -> pp.ad.Operator:
            return pp.ad.Operator('dumme fluid enthalpy', domains)
    
    mixin = Model()

    mixin.mdg = mdg
    mixin.equation_system = eqsys
    mixin.eliminate_reference_component = eliminate_reference
    mixin.eliminate_reference_phase = eliminate_reference
    mixin.equilibrium_type = equilibrium_type
    mixin.pressure = pressure
    mixin.temperature = temperature

    return mixin


@pytest.mark.parametrize(
    'phaseconfig',
    [
        [],  # shoud lead to error if no phase
        [('L', composit.PhysicalState.liquid)],
        [('L', composit.PhysicalState.liquid), ('L', composit.PhysicalState.liquid)],  # error since same name
        [('G', composit.PhysicalState.gas), ('L', composit.PhysicalState.liquid)],  # should re-order s.t. L is ref phase and G at end
        [('L', composit.PhysicalState.gas), ('G', composit.PhysicalState.gas)],  # error since two gas phases
        [('L1', composit.PhysicalState.liquid), ('L2', composit.PhysicalState.liquid), ('G', composit.PhysicalState.gas)],
    ],
)
@pytest.mark.parametrize(
    'species',
    [
        [],  # error if no component
        ['H2O'],
        ['H2O', 'H2O'],  # error since same name
        ['H2O', 'CO2'],
    ],
)
def test_mixture_contexts(species: list[str], phaseconfig: list[list[tuple[str, int]]], dummyeos):
    """Tests the phase and component context of a fluid mixture, and the assumptions
    on wich the framework is built. They must not be violated by any future development.
    """

    nphase = len(phaseconfig)
    ncomp = len(species)

    # 1 component to make an eos
    h2o = composit.Component.from_species(composit.load_species(['H2O'])[0])

    components: list[composit.Component] = [
        composit.Component.from_species(s) for s in composit.load_species(species)
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
        if (
            (len(set(phasenames)) < nphase or len(set(compnames)) < ncomp)
            and
            (ncomp > 0 or nphase > 0)
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
    'eliminate_reference', [True, False]  # for component and phase, test if eliminated
)
@pytest.mark.parametrize(
    'equilibrium_type', [None, 'p-T']  # for None, no y or extended
)
def test_mixture_member_assignment(
    eliminate_reference: bool, equilibrium_type: None | str, dummyeos
):
    """Testint that all requried members of phases, components, compounds and
    fluid mixtures are assigned by the compositional mixins. Tested with and without
    independent reference component/phase fractions."""

    species = composit.load_species(['H2O', 'CO2', 'NaCl'])
    comp1 = composit.Compound.from_species(species[0])
    comp1.pseudo_components = [species[2]]
    comp2 = composit.Component.from_species(species[1])
    eos = dummyeos([comp1, comp2])

    mixin: MockModel = get_mock_model(
        [comp1, comp2],
        [(eos, 0, "L"), (eos, 1, "G")],
        eliminate_reference,
        equilibrium_type
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

    # THe names should reflect the created variables
    # There are always partial or extended variables
    assert len(mixin.relative_fraction_variables) == ncomp * nphase
    # there are always solute fraction variables in compounds
    assert len(mixin.solute_fraction_variables) == 1
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
        assert hasattr(phase, 'saturation')
        assert hasattr(phase, 'density')
        assert hasattr(phase, 'specific_volume')
        assert hasattr(phase, 'specific_enthalpy')
        assert hasattr(phase, 'viscosity')
        assert hasattr(phase, 'conductivity')
        assert hasattr(phase, 'partial_fraction_of')

        if equilibrium_type is not None:
            assert hasattr(phase, 'fraction')
            assert hasattr(phase, 'extended_fraction_of')

        if phase == mixin.fluid_mixture.reference_phase and eliminate_reference:
            # reference phase fraction and saturations are dependent operators
            # if eliminated
            assert not isinstance(phase.saturation(sds), pp.ad.Variable)
            assert isinstance(phase.saturation(sds), pp.ad.Operator)
            assert not isinstance(phase.saturation(bgs), pp.ad.TimeDependentDenseArray)
            assert isinstance(phase.saturation(bgs), pp.ad.Operator)
            if equilibrium_type is not None:
                assert not isinstance(phase.fraction(sds), pp.ad.Variable)
                assert isinstance(phase.fraction(sds), pp.ad.Operator)
                assert not isinstance(phase.fraction(bgs), pp.ad.TimeDependentDenseArray)
                assert isinstance(phase.fraction(bgs), pp.ad.Operator)
        else:
            # otherwise it must be a variable
            assert isinstance(phase.saturation(sds), pp.ad.Variable)
            if equilibrium_type is not None:
                assert isinstance(phase.fraction(sds), pp.ad.Variable)

            # fraction and saturation on boundaries are time-dependent dense arrays
            assert isinstance(phase.saturation(bgs), pp.ad.TimeDependentDenseArray)
            if equilibrium_type is not None:
                assert isinstance(phase.saturation(bgs), pp.ad.TimeDependentDenseArray)

        # Now check the thermodynamic properties.
        # On domains it must be some kind of Operator (use most basic for check)
        # ON boundaries we expect a time-dependent dense array
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

        # if equilibrium type is defined, check extended fractions and fugacities
        if equilibrium_type is not None:
            for comp in phase:
                assert comp in phase.extended_fraction_of
                assert comp in phase.fugacity_coefficient_of
                # Extended fractions are all variables
                assert isinstance(phase.extended_fraction_of[comp](sds), pp.ad.Variable)
                assert isinstance(phase.fugacity_coefficient_of[comp](sds), pp.ad.Operator)

                assert isinstance(phase.extended_fraction_of[comp](bgs), pp.ad.TimeDependentDenseArray)
                assert isinstance(phase.fugacity_coefficient_of[comp](bgs), pp.ad.TimeDependentDenseArray)
        # check the definition of partial fractions
        # in equilibrium setting, they must be dependent operators
        for comp in phase:
            assert comp in phase.partial_fraction_of
            
            if equilibrium_type is not None:
                assert not isinstance(phase.partial_fraction_of[comp](sds), pp.ad.Variable)
                assert isinstance(phase.partial_fraction_of[comp](sds), pp.ad.Operator)
                assert not isinstance(phase.partial_fraction_of[comp](bgs), pp.ad.TimeDependentDenseArray)
                assert isinstance(phase.partial_fraction_of[comp](bgs), pp.ad.Operator)
            else:
                assert isinstance(phase.partial_fraction_of[comp](sds), pp.ad.Variable)
                assert isinstance(phase.partial_fraction_of[comp](bgs), pp.ad.TimeDependentDenseArray)

    # Check components and their overall fractions
    for comp in mixin.fluid_mixture.components:
        assert hasattr(comp, 'fraction')

        if comp == mixin.fluid_mixture.reference_component and eliminate_reference:
            assert not isinstance(comp.fraction(sds), pp.ad.Variable)
            assert isinstance(comp.fraction(sds), pp.ad.Operator)
            assert not isinstance(comp.fraction(bgs), pp.ad.TimeDependentDenseArray)
            assert isinstance(comp.fraction(bgs), pp.ad.Operator)
        else:
            assert isinstance(comp.fraction(sds), pp.ad.Variable)
            assert isinstance(comp.fraction(bgs), pp.ad.TimeDependentDenseArray)

        # IF it is a compound, check relative fractions of pseudo components
        if isinstance(comp, composit.Compound):
            assert hasattr(comp, 'solute_fraction_of')
            for pc in comp.pseudo_components:
                assert pc in comp.solute_fraction_of
                # solute fractions are aalways variables
                assert isinstance(comp.solute_fraction_of[pc](sds), pp.ad.Variable)
                assert isinstance(comp.solute_fraction_of[pc](bgs), pp.ad.Operator)


    # TODO should we include tests for singular mixtures? 1 component and/ or 1 phase
    # Checks would include that some fractions should not be variables, but 1


# Parametrization to test for any combination
@pytest.mark.parametrize(
    'equilibrium_type', [None, 'p-T']  # for None, no y or extended
)
@pytest.mark.parametrize(
    'phase_names', [['L'], ['L', 'G']]
)
@pytest.mark.parametrize(
    'species', [['H2O'], ['H2O', 'CO2']]
)
def test_singular_mixtures(species, phase_names, equilibrium_type, dummyeos):
    """Testing the behavior when only 1 component, or 1 phase or both.
    
    In this case, the number of created variables follows certain rules."""


    components = [composit.Component.from_species(s) for s in composit.load_species(species)]
    eos = dummyeos(components)
    phases = [(eos, 0, name) for name in phase_names]

    mixin: MockModel = get_mock_model(
        components, phases, True, equilibrium_type
    )
    
    sds = mixin.mdg.subdomains()
    bgs = mixin.mdg.boundaries()

    mixin.create_mixture()
    mixin.create_variables()
    # NOTE if None, there should be no phase fractions, ergo no mixture enthalpy.
    # but we implemented a dummy above
    mixin.assign_thermodynamic_properties_to_mixture()

    nphase = mixin.fluid_mixture.num_phases
    ncomp = mixin.fluid_mixture.num_components

    if nphase == 1 and ncomp > 1:
        # In this case, the phase fraction and saturation variable are never
        # variables, but scalars with value 1.
        # Partial fractions of components in phases are equal the overall fractions
        # Extended fractions should not be created since with 1 phase no equilibrium
        # problem
        ...
    elif nphase > 1 and ncomp == 1:
        ...
    elif nphase == 1 and ncomp == 1:
        ...
