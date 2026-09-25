"""Tests for chemical reactions."""

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.compositional.compositional_mixins import (
    ReactionRatesKineticFromExperiment,
)
from porepy.models.compositional_flow import _fischer_burmeister


def test_fischer_burmeister_has_finite_generalized_derivative_at_origin() -> None:
    """The complementarity function should be usable at a zero mineral state."""
    var_0 = pp.ad.AdArray(
        np.array([0.0, 3.0]),
        sps.csr_matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
    )
    var_1 = pp.ad.AdArray(
        np.array([0.0, 4.0]),
        sps.csr_matrix([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )

    result = _fischer_burmeister(var_0, var_1)

    assert isinstance(result, pp.ad.AdArray)
    assert np.allclose(result.val, [0.0, -2.0])
    assert np.allclose(
        result.jac.toarray(),
        [[-1.0, 0.0, -1.0, 0.0], [0.0, -0.4, 0.0, -0.2]],
    )


def test_user_provided_constant_reaction_rates_are_wrapped_as_ad_scalars() -> None:
    """Each user-provided constant rate should retain its own value."""
    values = [0.0, 1.25, 3.5]
    reactions = [
        pp.Reaction(name=f"reaction_{i}", constant_reaction_rate=value)
        for i, value in enumerate(values)
    ]

    chemical_system = object.__new__(pp.ChemicalSystem)
    chemical_system.set_kinetic_reaction_rates(reactions)

    equation_system = pp.ad.EquationSystem(pp.MixedDimensionalGrid())
    for reaction, expected in zip(reactions, values):
        assert reaction.constant_reaction_rate == expected
        rate = reaction.reaction_rate([])
        assert isinstance(rate, pp.ad.Scalar)
        assert rate.name == "user_defined_reaction_rate"
        assert np.isclose(equation_system.evaluate(rate), expected)


def test_experimental_reaction_rates_bind_parameters_per_reaction() -> None:
    """Reaction-rate closures should not reuse the final reaction's parameters."""

    class Component:
        def __init__(self, name: str, molar_volume: float = 1.0) -> None:
            self.name = name
            self.molar_volume = molar_volume

        def mineral_saturation(self, domains) -> pp.ad.Operator:
            return pp.ad.Scalar(1.0)

    lithium = Component("Li+")
    anion_x = Component("X-")
    anion_y = Component("Y-")
    mineral_x = Component("LiX")
    mineral_y = Component("LiY")
    components = [lithium, anion_x, anion_y, mineral_x, mineral_y]

    def unit_activity(domains) -> pp.ad.Operator:
        return pp.ad.Scalar(1.0)

    liquid = SimpleNamespace(
        state=pp.compositional.PhysicalState.liquid,
        components=[lithium, anion_x, anion_y],
        activity_of={
            lithium: unit_activity,
            anion_x: unit_activity,
            anion_y: unit_activity,
        },
    )
    solid = SimpleNamespace(
        state=pp.compositional.PhysicalState.solid,
        components=[mineral_x, mineral_y],
        activity_of={mineral_x: unit_activity, mineral_y: unit_activity},
    )

    class Model(ReactionRatesKineticFromExperiment):
        def rate_constant(self, reaction: pp.Reaction) -> float:
            return {"reaction_x": 0.1, "reaction_y": 0.2}[reaction.name]

        def equilibrium_lithium_concentration(
            self, reaction: pp.Reaction
        ) -> float:
            return 1.0

        def ic_minerals_bulk_concentration_wrap(self, component, domains):
            return np.ones(1)

        def total_porosity(self, domains) -> pp.ad.Operator:
            return pp.ad.Scalar(1.0)

        def porosity(self, domains) -> pp.ad.Operator:
            return pp.ad.Scalar(1.0)

        def molar_bulk_concentration(self, component, domains) -> pp.ad.Operator:
            return pp.ad.Scalar(0.0)

    model = Model()
    model.species_names = [component.name for component in components]
    model.reaction_formulas = ["LiX = Li+ + X-", "LiY = Li+ + Y-"]
    model.fluid = SimpleNamespace(
        components=components,
        solid_components=[mineral_x, mineral_y],
        phases=[liquid, solid],
        stoichiometric_matrix=np.array(
            [
                [1.0, 1.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, -1.0],
            ]
        ),
    )
    reactions = [
        pp.Reaction(formula=formula, name=name)
        for formula, name in zip(
            model.reaction_formulas, ["reaction_x", "reaction_y"]
        )
    ]

    model.set_kinetic_reaction_rates(reactions)

    equation_system = pp.ad.EquationSystem(pp.MixedDimensionalGrid())
    rates = [
        equation_system.evaluate(reaction.reaction_rate([]))
        for reaction in reactions
    ]
    assert np.allclose(np.hstack(rates), [0.1, 0.2])
