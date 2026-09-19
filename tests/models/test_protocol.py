"""Tests of the runtime behaviour of the PorePyModel protocol.

The protocol is a ``typing.Protocol`` for static type checkers only. At runtime it is
replaced by a placeholder which returns an empty tuple from ``__mro_entries__``, so that
``class X(pp.PorePyModel)`` is compiled as ``class X:``.

Testing covers:
    The protocol is absent from the MRO of the composed models.
    The protocol can be listed in any base position without provoking a TypeError.
    Override mixins keep their intended precedence over the model they are mixed into.

See the Warning in :mod:`porepy.models.protocol` for why this matters: a placeholder
which *is* a class becomes a linearisation barrier, silently demoting override mixins
behind the classes they were written to override.

"""

from __future__ import annotations

import types

import pytest

import porepy as pp
from porepy.models import constitutive_laws, momentum_balance
from porepy.models.contact_mechanics import ContactMechanics
from porepy.models.derived_models.biot import BiotPoromechanics
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.models.mass_and_energy_balance import MassAndEnergyBalance
from porepy.models.momentum_balance import MomentumBalance
from porepy.models.poromechanics import Poromechanics
from porepy.models.thermoporomechanics import Thermoporomechanics

# The composed models shipped with PorePy. Every mixin they are built from inherits the
# protocol, so if the protocol ever re-enters the MRO, it does so here.
composed_models = [
    SinglePhaseFlow,
    MassAndEnergyBalance,
    MomentumBalance,
    ContactMechanics,
    Poromechanics,
    Thermoporomechanics,
    BiotPoromechanics,
]


@pytest.mark.parametrize("model_class", composed_models)
def test_protocol_absent_from_mro(model_class: type) -> None:
    """The protocol must not contribute anything to the runtime MRO."""
    names = [cls.__name__ for cls in model_class.__mro__]
    assert "PorePyModel" not in names
    assert "_PorePyModelPlaceholder" not in names


@pytest.mark.parametrize("model_class", composed_models)
def test_protocol_can_be_listed_before_a_model(model_class: type) -> None:
    """Listing the protocol first is the natural way for a user to pick up type hints.

    With a placeholder class as base this raised ``TypeError: Cannot create a consistent
    method resolution order``, since the placeholder must follow its own subclasses.

    Note that ``types.new_class`` is used rather than the three-argument ``type``: only
    the former performs ``__mro_entries__`` resolution.

    """
    combined = types.new_class("CombinedModel", (pp.PorePyModel, model_class))
    assert combined.__mro__[1] is model_class


def test_protocol_does_not_reorder_mixins() -> None:
    """An override mixin must keep precedence over the model it is mixed into.

    ``VariablesThreeFieldMomentumBalance`` inheriting ``VariableMixin`` used to drag the
    protocol into the middle of ``TpsaMomentumBalanceMixin``'s linearisation. Everything
    behind the protocol there - including
    ``ThreeFieldLinearElasticMechanicalStress`` - was then demoted behind the whole of
    ``MomentumBalance``, so the default ``stress()`` silently won.

    """
    combined = type(
        "TpsaMomentumBalance",
        (momentum_balance.TpsaMomentumBalanceMixin, MomentumBalance),
        {},
    )
    mro = combined.__mro__
    # The stress law which overrides the default must precede it in the MRO.
    override = constitutive_laws.ThreeFieldLinearElasticMechanicalStress
    default = constitutive_laws.LinearElasticMechanicalStress
    assert mro.index(override) < mro.index(default)

    # The whole TPSA bundle, not just the stress law, must precede the model it
    # overrides.
    assert mro.index(momentum_balance.TpsaMomentumBalanceMixin) < mro.index(
        MomentumBalance
    )
    for mixin in momentum_balance.TpsaMomentumBalanceMixin.__bases__:
        assert mro.index(mixin) < mro.index(default)


def test_protocol_is_not_usable_with_isinstance() -> None:
    """``isinstance`` against the protocol fails loudly rather than lying.

    The protocol is not ``runtime_checkable``, so such a check could never be
    meaningful. Annotate with the protocol instead.

    """
    with pytest.raises(TypeError):
        isinstance(object(), pp.PorePyModel)
