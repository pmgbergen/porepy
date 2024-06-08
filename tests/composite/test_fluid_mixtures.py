"""Module testing the creation of fluid mixtures and their mixins."""
from __future__ import annotations

from numpy import ndarray
import pytest

import porepy.composite as ppc


@pytest.fixture
def dummyeos():
    """Dummy Eos"""

    class DummyEos(ppc.AbstractEoS):

        def compute_phase_state(
            self, phase_type: int, *thermodynamic_input: ndarray
        ) -> ppc.PhaseState:
            pass

    return DummyEos

@pytest.mark.parametrize(
    'phaseconfig',
    [
        [],  # shoud lead to error if no phase
        [('L', 0)],
        [('L', 0), ('L', 0)],  # error since same name
        [('G', 1), ('L', 0)],  # should re-order s.t. L is ref phase and G at end
        [('L', 1), ('G', 1)],  # error since two gas phases
        [('L1', 0), ('L2', 0), ('G', 1)],
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
    h2o = ppc.Component.from_species(ppc.load_species(['H2O'])[0])

    components: list[ppc.Component] = [
        ppc.Component.from_species(s) for s in ppc.load_species(species)
    ]

    if ncomp == 0:
        with pytest.raises(ppc.CompositionalModellingError):
            _ = dummyeos(components)
        eos: ppc.AbstractEoS = dummyeos([h2o])
    else:
        eos: ppc.AbstractEoS = dummyeos(components)
    phases: list[ppc.Phase] = []
    has_gas = False
    has_more_gas = False
    for conf in phaseconfig:
        name, t = conf
        if t == 1:
            if has_gas:
                has_more_gas = True
            has_gas = True
        phases.append(ppc.Phase(eos, t, name))
        phases[-1].components = [h2o]  # to avoid errors

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
                ppc.FluidMixture(components, phases)
        else:
            with pytest.raises(ppc.CompositionalModellingError):
                ppc.FluidMixture(components, phases)
    # cannot create mixtures with duplicate names
    elif len(set(phasenames)) < nphase or len(set(compnames)) < ncomp:
        with pytest.raises(ValueError):
            ppc.FluidMixture(components, phases)
    # more than 1 gas phase not allowed
    elif has_more_gas:
        with pytest.raises(ppc.CompositionalModellingError):
            ppc.FluidMixture(components, phases)
    # else the creation should not raise an error, and we check phase ordering
    else:
        mix = ppc.FluidMixture(components, phases)

        ordered_phases = [p for p in mix.phases]
        ordered_comps = [c for c in mix.components]
        if has_gas:
            assert ordered_phases[-1].type == 1

        # asert that the first phase and components are always the reference
        assert ordered_phases[0] == mix.reference_phase
        assert ordered_comps[0] == mix.reference_component

        # check that we cannot create mixtures if a phase has no components
        phases[0].components = []
        with pytest.raises(ppc.CompositionalModellingError):
            ppc.FluidMixture(components, phases)
    



@pytest.mark.parametrize(
    'mixin', [ppc.FluidMixtureMixin]
)
def test_mixture_member_assignment(mixin):
    """Testint that all requried members of phases, components, compounds and
    fluid mixtures are assigned by the fluid mixture mixin.
    
    Parametrizable using child classes of FluidMixtureMixin"""