"""Tests of the component mass balance where a well is present.

The component balance reaches the rock matrix through interfaces of codimension two,
the same path the mass and energy balances take, and was expected to need no change
for the coupling between a well and the rock it passes through. These tests verify
that expectation rather than assume it.

"""

from __future__ import annotations

import numpy as np

import porepy as pp
from porepy.applications.md_grids.model_geometries import TwoWells3d
from porepy.applications.test_utils import well_models
from porepy.compositional.compositional_mixins import CompositionalVariables
from porepy.models.compositional_flow import (
    BoundaryConditionsMulticomponent,
    ComponentMassBalanceEquations,
    InitialConditionsFractions,
)


class TracerWellModel(
    TwoWells3d,
    well_models.TracerInTheWell,
    well_models.BoundaryConditionsOneRateOnePressureWell,
    well_models.WellPermeability,
    well_models.TracerWellFluid,
    CompositionalVariables,
    ComponentMassBalanceEquations,
    BoundaryConditionsMulticomponent,
    InitialConditionsFractions,
    pp.SinglePhaseFlow,
):
    """Two wells carrying a tracer into a rock matrix that starts free of it."""


def _tracer_well_model() -> pp.PorePyModel:
    """A charging and a producing well, both open to the rock, run to completion."""
    model = TracerWellModel(
        {
            "material_constants": {
                "solid": pp.SolidConstants(
                    well_radius=0.02,
                    residual_aperture=1.0,
                    permeability=1e4,
                    normal_permeability=1e4,
                )
            },
            "grid_type": "simplex",
            "meshing_arguments": {"cell_size": 0.25, "cell_size_min": 0.05},
            "times_to_export": [],
            "well_pressure": -1e-1,
            "well_completion": {
                0: {"open_intervals": [(0.0, 10.0)]},
                1: {"open_intervals": [(0.0, 10.0)]},
            },
        }
    )
    pp.ModelRunner(model).run()
    return model


def _well_matrix_interfaces(model: pp.PorePyModel) -> list[pp.MortarGrid]:
    """The interfaces coupling a well to the rock matrix it passes through."""
    return [
        intf
        for intf in model.mdg.interfaces(codim=2)
        if model.mdg.interface_to_subdomain_pair(intf)[1].dim == 1
    ]


def test_the_well_component_flux_is_advected_from_the_side_the_fluid_comes_from():
    """A component crosses a well contact on the fluid that carries it.

    The component flux must equal the advected fraction of the upstream side times the
    mass flux, and the upstream side is the one the sign of the mass flux names. The
    expectation selects that side from the mass flux alone rather than from the
    discretisation's own upwind matrices, so the test can fail. A failure means the
    composition of the wrong side is being carried, which for a charging well would
    have it deliver the rock's composition to the rock.
    """
    model = _tracer_well_model()
    tracer = [c for c in model.fluid.components if c.name == "tracer"][0]

    directions = []
    for intf in _well_matrix_interfaces(model):
        rock, well = model.mdg.interface_to_subdomain_pair(intf)
        mass_flux = model.equation_system.evaluate(model.well_flux([intf]))
        component_flux = model.equation_system.evaluate(
            model.well_component_flux(tracer, [intf])
        )

        # A positive flux runs from the primary, higher-dimensional side to the
        # secondary one, so it carries the rock's composition into the well.
        weight_of = model.advection_weight_component_mass_balance
        from_rock = intf.primary_to_mortar_avg() @ model.equation_system.evaluate(
            weight_of(tracer, [rock])
        )
        from_well = intf.secondary_to_mortar_avg() @ model.equation_system.evaluate(
            weight_of(tracer, [well])
        )
        upstream = np.where(mass_flux > 0, from_rock, from_well)

        np.testing.assert_allclose(
            component_flux, upstream * mass_flux, rtol=1e-10, atol=1e-14
        )
        directions.append(np.sign(mass_flux[np.abs(mass_flux) > 1e-12]))

    # Unless both directions occur, only one branch of the expectation was tested.
    signs = np.concatenate(directions)
    assert np.any(signs > 0) and np.any(signs < 0)


def test_a_tracer_reaches_the_rock_only_through_the_well_contacts():
    """Tracer found in the rock must have crossed a well-matrix contact.

    The rock and the fractures start free of tracer and no tracer enters through their
    boundaries, so the coupling under test is its only way in. A failure means either
    that the component balance carries nothing across the new interfaces, or that
    tracer appears in the rock without being transported there.
    """
    model = _tracer_well_model()
    tracer = [c for c in model.fluid.components if c.name == "tracer"][0]

    rock = model.mdg.subdomains(dim=3)
    fraction = model.equation_system.evaluate(model.overall_fraction(tracer)(rock))
    assert np.max(fraction) > 1e-10, "no tracer reached the rock"
    # Nothing may be created or destroyed on the way: a fraction is bounded by the
    # two states being mixed.
    assert np.min(fraction) >= -1e-12 and np.max(fraction) <= 1 + 1e-12


class ThermalTracerWellModel(
    TwoWells3d,
    well_models.TracerInTheWell,
    well_models.BoundaryConditionsOneRateOnePressureWell,
    well_models.WellPermeability,
    well_models.TracerWellFluid,
    CompositionalVariables,
    ComponentMassBalanceEquations,
    BoundaryConditionsMulticomponent,
    InitialConditionsFractions,
    pp.MassAndEnergyBalance,
):
    """The tracer model of :class:`TracerWellModel`, carrying an energy balance too."""


def _thermal_tracer_well_model() -> pp.PorePyModel:
    """Mass, energy and components in one model, with wells open to the rock."""
    dt = 1e-4
    model = ThermalTracerWellModel(
        {
            "material_constants": {
                "solid": pp.SolidConstants(
                    specific_heat_capacity=1e0,
                    well_radius=0.02,
                    residual_aperture=1.0,
                    thermal_conductivity=1e-6,
                    permeability=1e4,
                    normal_permeability=1e4,
                )
            },
            "time_manager": pp.TimeManager(
                schedule=[0, dt], dt_init=dt, constant_dt=True
            ),
            "grid_type": "simplex",
            "meshing_arguments": {"cell_size": 0.25, "cell_size_min": 0.05},
            "times_to_export": [],
            "well_pressure": -1e-1,
            "well_completion": {
                0: {"open_intervals": [(0.0, 10.0)]},
                1: {"open_intervals": [(0.0, 10.0)]},
            },
        }
    )
    pp.ModelRunner(model).run()
    return model


def test_the_three_balances_advect_from_the_same_side_of_a_well_contact():
    """Mass, energy and composition cross a well contact together, or not at all.

    The three balances are assembled by three separate pieces of code that share only
    the mass flux, and each is tested on its own in a model where the others are
    absent. That leaves the case this test covers: with all three present, they must
    agree about which side of a contact the fluid comes from. A failure means one
    balance carries the state of one side while another carries the state of the
    other, which no test of a single balance can detect.
    """
    model = _thermal_tracer_well_model()
    tracer = [c for c in model.fluid.components if c.name == "tracer"][0]

    directions = []
    for intf in _well_matrix_interfaces(model):
        rock, well = model.mdg.interface_to_subdomain_pair(intf)
        mass_flux = model.equation_system.evaluate(model.well_flux([intf]))

        def upstream(weight) -> np.ndarray:
            """The weight of whichever side the mass flux comes from."""
            from_rock = intf.primary_to_mortar_avg() @ model.equation_system.evaluate(
                weight([rock])
            )
            from_well = intf.secondary_to_mortar_avg() @ (
                model.equation_system.evaluate(weight([well]))
            )
            return np.where(mass_flux > 0, from_rock, from_well)

        np.testing.assert_allclose(
            model.equation_system.evaluate(model.well_enthalpy_flux([intf])),
            upstream(model.advection_weight_energy_balance) * mass_flux,
            rtol=1e-10,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            model.equation_system.evaluate(model.well_component_flux(tracer, [intf])),
            upstream(
                lambda domains: model.advection_weight_component_mass_balance(
                    tracer, domains
                )
            )
            * mass_flux,
            rtol=1e-10,
            atol=1e-14,
        )
        directions.append(np.sign(mass_flux[np.abs(mass_flux) > 1e-12]))

    # Unless both directions occur, only one branch of the expectation was tested.
    signs = np.concatenate(directions)
    assert np.any(signs > 0) and np.any(signs < 0)
