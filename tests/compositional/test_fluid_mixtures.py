"""Module testing the creation of fluid mixtures and their mixins."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

import porepy as pp
import porepy.compositional as pc
from porepy.models.compositional_flow import SolutionStrategyCF


class MockModel(
    pc.FluidMixin,
    pc.CompositionalVariables,
    pp.BoundaryConditionMixin,
    SolutionStrategyCF,
):
    """For typing support when using methods of the mixins."""

    pass


def get_mock_model(
    components,
    phase_configuration,
    eliminate_reference,
    equilibrium_specification,
):
    # simple set up, 1 cell domain, with 1 boundary grid
    g = pp.CartGrid(np.array([1, 1]))
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains(g)
    mdg.compute_geometry()
    mdg.set_boundary_grid_projections()
    equation_system = pp.ad.EquationSystem(mdg)

    # mimicing pressure and temperatur variable
    def pressure(grid):
        return pp.ad.Variable("pressure", {"cells": 1}, grid[0])

    def temperature(grid):
        return pp.ad.Variable("pressure", {"cells": 1}, grid[0])

    class Model(
        pc.FluidMixin,
        pc.CompositionalVariables,
        pp.BoundaryConditionMixin,
        SolutionStrategyCF,
    ):
        pressure: Callable
        temperature: Callable

        def get_components(self) -> list[pc.Component]:
            return components

        def get_phase_configuration(
            self, components: np.Sequence[pc.Component]
        ) -> np.Sequence[tuple[pc.EquationOfState, int, str]]:
            return phase_configuration

        # for simplicity of testing
        def dependencies_of_phase_properties(self, phase: pc.Phase):
            return [self.pressure, self.temperature]

    model = Model()

    model.mdg = mdg
    model.equation_system = equation_system
    model.params = {
        "eliminate_reference_component": eliminate_reference,
        "eliminate_reference_phase": eliminate_reference,
        "equilibrium_specification": equilibrium_specification,
    }
    model.pressure = pressure
    model.temperature = temperature

    return model


@pytest.mark.parametrize(
    "phaseconfig",
    [
        [],  # shoud lead to error if no phase
        [("L", pc.PhysicalState.liquid)],
        [
            ("L", pc.PhysicalState.liquid),
            ("L", pc.PhysicalState.liquid),
        ],  # error since same name
        [
            ("G", pc.PhysicalState.gas),
            ("L", pc.PhysicalState.liquid),
        ],  # should re-order s.t. L is ref phase and G at end
        [
            ("L", pc.PhysicalState.gas),
            ("G", pc.PhysicalState.gas),
        ],  # error since two gas phases
        [
            ("L1", pc.PhysicalState.liquid),
            ("L2", pc.PhysicalState.liquid),
            ("G", pc.PhysicalState.gas),
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
    species: list[str], phaseconfig: list[list[tuple[str, pc.PhysicalState]]]
):
    """Tests the phase and component context of a fluid mixture, and the assumptions
    on wich the framework is built. They must not be violated by any future development.
    """

    nphase = len(phaseconfig)
    ncomp = len(species)
    # Creating dummy components and eos. Physical properties have no relevance here
    # 1 separate component for the dummy eos, just to instantiate it.
    h2o = pc.Component(name="H2O")
    components: list[pc.Component] = [pc.Component(name=s) for s in species]

    if ncomp == 0:
        with pytest.raises(pc.CompositionalModellingError):
            _ = pc.EquationOfState(components)
        eos = pc.EquationOfState([h2o])
    else:
        eos = pc.EquationOfState(components)

    phases: list[pc.Phase] = []
    has_gas = False
    has_more_gas = False
    for conf in phaseconfig:
        name, t = conf
        if t == pc.PhysicalState.gas:
            if has_gas:
                has_more_gas = True
            has_gas = True
        phases.append(pc.Phase(t, name, eos=eos))
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
                pc.Fluid(components, phases)
        else:
            with pytest.raises(pc.CompositionalModellingError):
                pc.Fluid(components, phases)
    # cannot create mixtures with duplicate names
    elif len(set(phasenames)) < nphase or len(set(compnames)) < ncomp:
        with pytest.raises(ValueError):
            pc.Fluid(components, phases)
    # more than 1 gas phase not allowed
    elif has_more_gas:
        with pytest.raises(pc.CompositionalModellingError):
            pc.Fluid(components, phases)
    # else the creation should not raise an error, and we check phase ordering
    else:
        mix = pc.Fluid(components, phases)

        ordered_phases = [p for p in mix.phases]
        ordered_comps = [c for c in mix.components]
        if has_gas:
            assert ordered_phases[-1].state == pc.PhysicalState.gas

        # asert that the first phase and components are always the reference
        assert ordered_phases[0] == mix.reference_phase
        assert ordered_comps[0] == mix.reference_component

        # check that we cannot create mixtures if a phase has no components
        phases[0].components = []
        with pytest.raises(pc.CompositionalModellingError):
            pc.Fluid(components, phases)


@pytest.mark.parametrize(
    "eliminate_reference",
    [True, False],  # for component and phase, test if eliminated
)
@pytest.mark.parametrize(
    "equilibrium_specification",
    [
        (pc.FlashSpec.none,),
        (pc.FlashSpec.pT, "persistent-variables"),
        (pc.FlashSpec.pT,),
    ],
)
def test_mixture_member_assignment(
    eliminate_reference: bool,
    equilibrium_specification: tuple[pc.FlashSpec | str, ...],
):
    """Testint that all requried members of phases, components, compounds and
    fluid mixtures are assigned by the compositional mixins. Tested with and without
    independent reference component/phase fractions."""

    # Creating dummy components. Physical properties have no relevance for this test
    comp1 = pc.Compound(name="H2O", molar_mass=1.0)
    comp1.active_tracers = [pp.FluidComponent(name="NaCl")]
    comp2 = pp.FluidComponent(name="CO2")
    # dummy EoS for completeness
    eos = pc.EquationOfState([comp1, comp2])
    phases = [
        (pc.PhysicalState.liquid, "L", eos),
        (pc.PhysicalState.gas, "G", eos),
    ]

    model: MockModel = get_mock_model(
        [comp1, comp2],
        phases,
        eliminate_reference,
        equilibrium_specification,
    )

    subdomains = model.mdg.subdomains()
    boundary_grids = model.mdg.boundaries()
    ncomp = 2
    nphase = 2

    model.create_fluid()
    model.create_variables()
    # NOTE if None, there should be no phase fractions, ergo no mixture enthalpy.
    # but we implemented a dummy above
    model.assign_thermodynamic_properties_to_phases()

    # The names should reflect the created variables
    # ref partial fraction eliminated in case of no equilibrium
    if not pc.has_equilibrium_specified(model):
        assert len(model.fraction_in_phase_variables) == (ncomp - 1) * nphase
    # ref partial fraction also eliminated in the case of non-unified equilibrium
    elif not pc.is_persistent_variable_form(model):
        assert len(model.fraction_in_phase_variables) == (ncomp - 1) * nphase
    # Extended fractions are always independent, if unified equilibrium
    elif pc.is_persistent_variable_form(model):
        assert len(model.fraction_in_phase_variables) == ncomp * nphase
    # there are always solute fraction variables in compounds
    assert len(model.tracer_fraction_variables) == 1
    # test that the eliminated variables were not created
    if eliminate_reference:
        assert len(model.overall_fraction_variables) == ncomp - 1
        assert len(model.saturation_variables) == nphase - 1
        if pc.has_equilibrium_specified(model):
            assert len(model.phase_fraction_variables) == nphase - 1
        else:
            assert len(model.phase_fraction_variables) == 0
    else:
        assert len(model.overall_fraction_variables) == ncomp
        assert len(model.saturation_variables) == nphase
        if pc.has_equilibrium_specified(model):
            assert len(model.phase_fraction_variables) == nphase
        else:
            assert len(model.phase_fraction_variables) == 0

    # Mixture is set up. First check phases and everything related to them
    for phase in model.fluid.phases:
        assert hasattr(phase, "saturation")
        assert hasattr(phase, "density")
        assert hasattr(phase, "specific_volume")
        assert hasattr(phase, "specific_enthalpy")
        assert hasattr(phase, "viscosity")
        assert hasattr(phase, "thermal_conductivity")
        assert hasattr(phase, "partial_fraction_of")

        # NOTE fraction and extended fractions are assigned in any case
        # They cause errors when evaluated though
        assert hasattr(phase, "fraction")
        assert hasattr(phase, "extended_fraction_of")

        # fractions should be defined in equilibrium setting
        if pc.has_equilibrium_specified(model):
            if model.has_independent_fraction(phase):
                assert isinstance(phase.fraction(subdomains), pp.ad.Variable)
                assert isinstance(
                    phase.fraction(boundary_grids), pp.ad.TimeDependentDenseArray
                )
            else:
                assert isinstance(phase.fraction(subdomains), pp.ad.Operator)
                assert isinstance(phase.fraction(boundary_grids), pp.ad.Operator)
        # If no equilibrium setting, calling phase fractions should yield an error
        else:
            with pytest.raises(pc.CompositionalModellingError):
                phase.fraction(subdomains)
            for comp in phase:
                with pytest.raises(pc.CompositionalModellingError):
                    phase.extended_fraction_of[comp](subdomains)

        # extended fraction only in unified equilibrium setting
        if pc.is_persistent_variable_form(model):
            for comp in phase:
                assert isinstance(
                    phase.extended_fraction_of[comp](subdomains), pp.ad.Variable
                )
                assert isinstance(
                    phase.extended_fraction_of[comp](boundary_grids),
                    pp.ad.TimeDependentDenseArray,
                )
                # partial fractions are not variables in unified equilibrium
                assert not isinstance(
                    phase.partial_fraction_of[comp](subdomains), pp.ad.Variable
                )
                assert not isinstance(
                    phase.partial_fraction_of[comp](boundary_grids),
                    pp.ad.TimeDependentDenseArray,
                )
                assert isinstance(
                    phase.partial_fraction_of[comp](subdomains), pp.ad.Operator
                )
                assert isinstance(
                    phase.partial_fraction_of[comp](boundary_grids), pp.ad.Operator
                )
        else:
            # Extended fraction without unified equilibrium make no sense
            for comp in phase:
                with pytest.raises(pc.CompositionalModellingError):
                    phase.extended_fraction_of[comp](subdomains)
            # Both cases, non-unified equilibrium and no equilibrium have independent
            # partial fractions
            for comp in phase:
                # reference componet partial fraction is always eliminated
                if model.has_independent_partial_fraction(comp, phase):
                    assert isinstance(
                        phase.partial_fraction_of[comp](subdomains), pp.ad.Variable
                    )
                    assert isinstance(
                        phase.partial_fraction_of[comp](boundary_grids),
                        pp.ad.TimeDependentDenseArray,
                    )
                else:
                    assert comp == phase.reference_component
                    assert not isinstance(
                        phase.partial_fraction_of[comp](subdomains), pp.ad.Variable
                    )
                    assert isinstance(
                        phase.partial_fraction_of[comp](subdomains), pp.ad.Operator
                    )
                    assert not isinstance(
                        phase.partial_fraction_of[comp](boundary_grids),
                        pp.ad.TimeDependentDenseArray,
                    )
                    assert isinstance(
                        phase.partial_fraction_of[comp](boundary_grids), pp.ad.Operator
                    )

        # Check that the reference phase saturation is a dependent operator
        if not model.has_independent_saturation(phase):
            assert phase == model.fluid.reference_phase
            # reference phase fraction and saturations are dependent operators
            # if eliminated
            assert not isinstance(phase.saturation(subdomains), pp.ad.Variable)
            assert isinstance(phase.saturation(subdomains), pp.ad.Operator)
            assert not isinstance(
                phase.saturation(boundary_grids), pp.ad.TimeDependentDenseArray
            )
            assert isinstance(phase.saturation(boundary_grids), pp.ad.Operator)
            # same holds for reference phase fraction in the equilibrium setting
            if pc.has_equilibrium_specified(model):
                assert not isinstance(phase.fraction(subdomains), pp.ad.Variable)
                assert isinstance(phase.fraction(subdomains), pp.ad.Operator)
                assert not isinstance(
                    phase.fraction(boundary_grids), pp.ad.TimeDependentDenseArray
                )
                assert isinstance(phase.fraction(boundary_grids), pp.ad.Operator)
        else:
            # otherwise it must be a variable
            assert isinstance(phase.saturation(subdomains), pp.ad.Variable)
            if pc.has_equilibrium_specified(model):
                assert isinstance(phase.fraction(subdomains), pp.ad.Variable)

            # fraction and saturation on boundaries are time-dependent dense arrays
            assert isinstance(
                phase.saturation(boundary_grids), pp.ad.TimeDependentDenseArray
            )
            if pc.has_equilibrium_specified(model):
                assert isinstance(
                    phase.saturation(boundary_grids), pp.ad.TimeDependentDenseArray
                )

        # Now check the thermodynamic properties.
        # On domains it must be some kind of Operator (use most basic for check)
        # On boundaries we expect a time-dependent dense array
        assert isinstance(phase.density(subdomains), pp.ad.Operator)
        assert isinstance(phase.specific_enthalpy(subdomains), pp.ad.Operator)
        assert isinstance(phase.specific_internal_energy(subdomains), pp.ad.Operator)
        assert isinstance(phase.specific_volume(subdomains), pp.ad.Operator)
        assert isinstance(phase.viscosity(subdomains), pp.ad.Operator)
        assert isinstance(phase.thermal_conductivity(subdomains), pp.ad.Operator)

        assert isinstance(phase.density(boundary_grids), pp.ad.TimeDependentDenseArray)
        assert isinstance(
            phase.specific_enthalpy(boundary_grids), pp.ad.TimeDependentDenseArray
        )
        # NOTE By default, specific internal energy is a dependent quantity, so we
        # do not test it on the boundary.
        # NOTE Volume is taken as the reciprocal of density, hence a general operator
        assert isinstance(phase.specific_volume(boundary_grids), pp.ad.Operator)
        assert isinstance(
            phase.viscosity(boundary_grids), pp.ad.TimeDependentDenseArray
        )
        assert isinstance(
            phase.thermal_conductivity(boundary_grids), pp.ad.TimeDependentDenseArray
        )

        # Fugacities are always created as well
        for comp in phase:
            assert comp in phase.fugacity_coefficient_of
            assert isinstance(
                phase.fugacity_coefficient_of[comp](subdomains), pp.ad.Operator
            )
            assert isinstance(
                phase.fugacity_coefficient_of[comp](boundary_grids),
                pp.ad.TimeDependentDenseArray,
            )

    # Check components and their overall fractions
    for comp in model.fluid.components:
        assert hasattr(comp, "fraction")

        # Overall fraction of reference component is eliminated (if requested)
        if not model.has_independent_fraction(comp):
            assert comp == model.fluid.reference_component
            assert not isinstance(comp.fraction(subdomains), pp.ad.Variable)
            assert isinstance(comp.fraction(subdomains), pp.ad.Operator)
            assert not isinstance(
                comp.fraction(boundary_grids), pp.ad.TimeDependentDenseArray
            )
            assert isinstance(comp.fraction(boundary_grids), pp.ad.Operator)
        else:
            assert isinstance(comp.fraction(subdomains), pp.ad.Variable)
            assert isinstance(
                comp.fraction(boundary_grids), pp.ad.TimeDependentDenseArray
            )

        # IF it is a compound, check relative fractions of pseudo components
        if isinstance(comp, pc.Compound):
            assert hasattr(comp, "tracer_fraction_of")
            for tracer in comp.active_tracers:
                assert tracer in comp.tracer_fraction_of
                # solute fractions are aalways variables
                assert isinstance(
                    comp.tracer_fraction_of[tracer](subdomains), pp.ad.Variable
                )
                assert isinstance(
                    comp.tracer_fraction_of[tracer](boundary_grids),
                    pp.ad.TimeDependentDenseArray,
                )

    # Finally, check the assignment of the overall properties of the mixture
    # Density, volume and enthalpy
    # The mixture must implement thermodynamic laws, they must not be variables
    assert isinstance(model.fluid.density(subdomains), pp.ad.Operator)
    assert not isinstance(model.fluid.density(subdomains), pp.ad.Variable)
    if not pc.has_equilibrium_specified(model):
        # Volume defined using fractions, which are not available without equilibrium.
        with pytest.raises(pc.CompositionalModellingError):
            model.fluid.specific_volume(subdomains)
    else:
        assert isinstance(model.fluid.specific_volume(subdomains), pp.ad.Operator)
        assert not isinstance(model.fluid.specific_volume(subdomains), pp.ad.Variable)
    if pc.has_equilibrium_specified(model):
        assert isinstance(model.fluid.specific_enthalpy(subdomains), pp.ad.Operator)
        assert not isinstance(model.fluid.specific_enthalpy(subdomains), pp.ad.Variable)
    else:
        # the basic defintion of enthalpy relies on phase fractions, which are not
        # available without equilibrium
        with pytest.raises(pc.CompositionalModellingError):
            model.fluid.specific_enthalpy(subdomains)


# Parametrization to test for any combination
@pytest.mark.parametrize(
    "equilibrium_specification",
    [
        (pc.FlashSpec.none,),
        (pc.FlashSpec.pT, "persistent-variables"),
        (pc.FlashSpec.pT,),
    ],
)
@pytest.mark.parametrize("phase_names", [["L"], ["L", "G"]])
@pytest.mark.parametrize("species", [["H2O"], ["H2O", "CO2"]])
def test_singular_mixtures(species, phase_names, equilibrium_specification):
    """Testing the behavior when only 1 component, or 1 phase or both.
    In this case, the number of created variables follows certain rules."""

    # Creating dummy components and EoS. Physical properties have no relevance here
    components: list[pp.FluidComponent] = [pp.FluidComponent(name=s) for s in species]
    hash(components[0])

    eos = pc.EquationOfState(components)
    phases = [(pc.PhysicalState.liquid, name, eos) for name in phase_names]

    model: MockModel = get_mock_model(
        components, phases, True, equilibrium_specification
    )

    subdomains = model.mdg.subdomains()

    model.create_fluid()
    model.create_variables()
    # NOTE if None, there should be no phase fractions, ergo no mixture enthalpy.
    # but we implemented a dummy above
    model.assign_thermodynamic_properties_to_phases()

    nphase = model.fluid.num_phases
    ncomp = model.fluid.num_components

    if nphase == 1 and ncomp > 1:
        # In this singular case, the partial fractions are not independent
        # Neither are the phase fraction and saturation
        phase = list(model.fluid.phases)[0]
        assert phase == model.fluid.reference_phase
        assert not isinstance(phase.saturation(subdomains), pp.ad.Variable)
        if pc.has_equilibrium_specified(model):
            assert not isinstance(phase.fraction(subdomains), pp.ad.Variable)

        for comp in model.fluid.components:
            assert comp in phase
            assert comp in phase.extended_fraction_of
            assert comp in phase.partial_fraction_of
            # In the case of 1 phase, partial fractions are equal overall fractions
            assert comp.fraction == phase.partial_fraction_of[comp]
            # Extended fractions are always variable
            if pc.is_persistent_variable_form(model):
                assert isinstance(
                    phase.extended_fraction_of[comp](subdomains), pp.ad.Variable
                )

            # checking overall fraction
            if model.has_independent_fraction(comp):
                assert comp != model.fluid.reference_component
                assert isinstance(comp.fraction(subdomains), pp.ad.Variable)
            else:
                assert comp == model.fluid.reference_component
                assert not isinstance(comp.fraction(subdomains), pp.ad.Variable)
    elif nphase > 1 and ncomp == 1:
        # In this case, there is no overall fraction
        # Partial fractions are also not independent, since only 1 component per phase
        # Extended fractions remain independent
        comp = list(model.fluid.components)[0]
        assert comp == model.fluid.reference_component
        assert not isinstance(comp.fraction(subdomains), pp.ad.Variable)

        for phase in model.fluid.phases:
            if model.has_independent_saturation(phase):
                assert isinstance(phase.saturation(subdomains), pp.ad.Variable)
                if pc.has_equilibrium_specified(model):
                    assert isinstance(phase.fraction(subdomains), pp.ad.Variable)
                else:
                    with pytest.raises(pc.CompositionalModellingError):
                        phase.fraction(subdomains)
            else:
                assert phase == model.fluid.reference_phase
                assert not isinstance(phase.saturation(subdomains), pp.ad.Variable)
                if pc.has_equilibrium_specified(model):
                    assert not isinstance(phase.fraction(subdomains), pp.ad.Variable)
                else:
                    with pytest.raises(pc.CompositionalModellingError):
                        phase.fraction(subdomains)

            # no phase without the single component
            assert comp in phase.components
            assert comp in phase.partial_fraction_of
            assert comp in phase.extended_fraction_of
            assert comp == phase.reference_component
            assert not isinstance(
                phase.partial_fraction_of[comp](subdomains), pp.ad.Variable
            )
            # Extended fractions remain variables, in case phase vanishes
            if pc.is_persistent_variable_form(model):
                assert isinstance(
                    phase.extended_fraction_of[comp](subdomains), pp.ad.Variable
                )
    elif nphase == 1 and ncomp == 1:
        # In this singular case, no fractional variable is independent
        comp = list(model.fluid.components)[0]
        phase = list(model.fluid.phases)[0]

        assert comp == model.fluid.reference_component
        assert phase == model.fluid.reference_phase
        assert comp == phase.reference_component

        assert not isinstance(comp.fraction(subdomains), pp.ad.Variable)
        assert not isinstance(phase.fraction(subdomains), pp.ad.Variable)
        assert not isinstance(phase.saturation(subdomains), pp.ad.Variable)
        assert not isinstance(
            phase.partial_fraction_of[comp](subdomains), pp.ad.Variable
        )
        assert not isinstance(
            phase.extended_fraction_of[comp](subdomains), pp.ad.Variable
        )
