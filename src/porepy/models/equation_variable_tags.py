from __future__ import annotations

from dataclasses import dataclass

import porepy as pp


@dataclass(frozen=True)
class DomainTag:
    pass


@dataclass(frozen=True)
class AllSubdomains(DomainTag):
    pass


@dataclass(frozen=True)
class EquationTag:
    name: str
    defined_on: DomainTag


@dataclass(frozen=True)
class VariableTag:
    name: str
    defined_on: DomainTag


class DefaultEquationTags:
    mass_balance = EquationTag(
        name=pp.SinglePhaseFlow.primary_equation_name(), defined_on=AllSubdomains()
    )


class DefaultVariableTags:
    pressure = VariableTag(name="pressure", defined_on=AllSubdomains())


"""
Why do we need an equation identifier? Can't we just take the equation from the equation
system?

- What if outside model?
- ordering of dofs?
- compare them, find intersection, etc.
- give me indices of this equation (equation system does this, but is it convenient?)
- give me indices of these equations if we only use 

Why do we need a variable identifier? Is MdVariable bad?
- outside model
- compare them

"""

"""
Operations we want:
- give me residual for these tags
- give me jacobian for these tags
- give me solution for these tags
- give me indexer for these tags

- switch physics based on something
- set new t and dt ...

"""


def main():
    model = pp.SinglePhaseFlow()
    model.prepare_simulation()
    model.before_time_step()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()

    eq_tags = [DefaultEquationTags.mass_balance]
    var_tags = [DefaultVariableTags.pressure]

    # assemble residual
    for eq_tag in eq_tags:
        equation = model.equation_system.equations.get(eq_tag.name, None)
        if equation is None:
            raise ValueError(eq_tag.name)
        model._ad_parser

    linear_system = model.assemble_linear_system()


if __name__ == "__main__":
    main()
