from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Final, Optional, TypeVar

import numpy as np
from scipy.sparse import csr_matrix

import porepy as pp


@dataclass(frozen=True)
class DomainFilter:
    @abstractmethod
    def filter(self, domain: pp.GridLike) -> bool:
        pass


@dataclass(frozen=True)
class Everywhere(DomainFilter):
    def filter(self, domain: pp.GridLike) -> bool:
        return True


@dataclass(frozen=True)
class EquationTag:
    name: str
    defined_on: DomainFilter = Everywhere()


@dataclass(frozen=True)
class VariableTag:
    name: str
    defined_on: DomainFilter = Everywhere()


class DefaultEquationTags:
    # Mass balance
    mass_balance = EquationTag(name=pp.SinglePhaseFlow.primary_equation_name())
    interface_darcy_flux = EquationTag(name="interface_darcy_flux_equation")
    well_flux = EquationTag(name="well_flux_equation")
    # Energy balance
    energy_balance = EquationTag(
        name=pp.energy_balance.TotalEnergyBalanceEquations.primary_equation_name()
    )
    interface_fourier_flux = EquationTag(name="interface_fourier_flux_equation")
    interface_enthalpy_flux = EquationTag(name="interface_enthalpy_flux_equation")
    well_enthalpy_flux = EquationTag(name="well_enthalpy_flux_equation")
    # Momentum balance MPSA
    momentum_balance = EquationTag(name="momentum_balance_equation")
    # Momentum balance TPSA
    angular_momentum_balance = EquationTag(name="angular_momentum_balance_equation")
    solid_mass = EquationTag(name="solid_mass_equation")
    poromechanics_solid_mass = EquationTag(name="Solid_mass_equation_poromechanics")
    # Contact mechanics
    normal_fracture_deformation = EquationTag(
        name="normal_fracture_deformation_equation"
    )
    tangential_fracture_deformation = EquationTag(
        name="tangential_fracture_deformation_equation"
    )
    interface_force_balance = EquationTag(name="interface_force_balance_equation")
    # Fracture damage (?)
    dilation_damage = EquationTag(name="dilation_damage_equation")
    friction_damage = EquationTag(name="friction_damage_equation")


class DefaultVariableTags:
    # Mass balance
    pressure = VariableTag(name="pressure")
    interface_darcy_flux = VariableTag(name="interface_darcy_flux")
    well_flux = VariableTag(name="well_flux")
    # Energy balance
    temperature = VariableTag(name="temperature")
    enthalpy = VariableTag(name="enthalpy")
    interface_fourier_flux = VariableTag(name="interface_fourier_flux")
    interface_enthalpy_flux = VariableTag(name="interface_enthalpy_flux")
    well_enthalpy_flux = VariableTag(name="well_enthalpy_flux")
    # Momentum balance MPSA and TPSA
    displacement = VariableTag(name="u")
    # Momentum balance TPSA
    rotation_stress = VariableTag(name="rotation_stress")
    total_pressure = VariableTag(name="total_pressure")
    # Contact mechanics
    interface_displacement = VariableTag(name="u_interface")
    contact_traction = VariableTag(name="contact_traction")
    # Fracture damage (?)
    dilation_damage_history = VariableTag(name="dilation_damage_history")
    friction_damage_history = VariableTag(name="friction_damage_history")


T = TypeVar("T")


def apply_cumulative_dof_offsets(
    dict_of_dofs: dict[T, np.ndarray],
) -> dict[T, np.ndarray]:
    result: dict[T, np.ndarray] = {}
    offset = 0
    for key, dofs in dict_of_dofs.items():
        result[key] = dofs + offset
        offset += len(dofs)
    return result


class Indexer:
    def __init__(
        self,
        equations_dofs: dict[EquationTag, np.ndarray],
        variables_dofs: dict[VariableTag, np.ndarray],
        equations_in_porepy_arrangement: Optional[dict[EquationTag, bool]] = None,
        variables_in_porepy_arrangement: Optional[dict[VariableTag, bool]] = None,
    ) -> None:
        self.equations_dofs: Final[dict[EquationTag, np.ndarray]] = equations_dofs
        self.equations_dofs_cumulative: Final[dict[EquationTag, np.ndarray]] = (
            apply_cumulative_dof_offsets(equations_dofs)
        )

        self.variables_dofs: Final[dict[VariableTag, np.ndarray]] = variables_dofs
        self.variables_dofs_cumulative: Final[dict[VariableTag, np.ndarray]] = (
            apply_cumulative_dof_offsets(variables_dofs)
        )

        if equations_in_porepy_arrangement is None:
            equations_in_porepy_arrangement = {tag: False for tag in equations_dofs}
        if variables_in_porepy_arrangement is None:
            variables_in_porepy_arrangement = {tag: False for tag in variables_dofs}
        self.equations_in_porepy_arrangement: Final[dict[EquationTag, bool]] = (
            equations_in_porepy_arrangement
        )
        self.variables_in_porepy_arrangement: Final[dict[VariableTag, bool]] = (
            variables_in_porepy_arrangement
        )

    def equation_tags(self) -> list[EquationTag]:
        return list(self.equations_dofs.keys())

    def variable_tags(self) -> list[VariableTag]:
        return list(self.variables_dofs.keys())


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
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)


class MyModel(SquareDomainOrthogonalFractures, pp.Thermoporomechanics):
    pass


@dataclass(frozen=True)
class DofsInfo:
    cells: int = 0
    faces: int = 0
    nodes: int = 0

    def __post_init__(self) -> None:
        assert self.cells >= 0
        assert self.faces >= 0
        assert self.nodes >= 0
        assert (self.cells + self.faces + self.nodes) > 0


def construct_rearrange(
    model: pp.PorePyModel,
    operator_domains: pp.GridLikeSequence,
    tag: EquationTag | VariableTag,
    dofs_info: DofsInfo,
):
    # TODO YZ docstring, comments
    requested_domains = tag.defined_on.domains(model)
    for domain in requested_domains:
        assert domain in operator_domains

    if len(operator_domains) == 0:
        print(f"No domains in equation tag: {tag}")
        return np.zeros(0), True

    offset = 0
    original_order: dict[pp.GridLike, np.ndarray] = {}
    for domain in operator_domains:
        if isinstance(domain, pp.Grid):
            dofs_per_grid = (
                domain.num_cells * dofs_info.cells
                + domain.num_faces * dofs_info.faces
                + domain.num_nodes * dofs_info.nodes
            )
        elif isinstance(domain, pp.MortarGrid):
            # Mortar grid has no faces.
            dofs_per_grid = (
                domain.num_cells * dofs_info.cells + domain.num_nodes * dofs_info.nodes
            )
        else:
            raise ValueError(type(domain))
        original_order[domain] = np.arange(offset, offset + dofs_per_grid)
        offset += dofs_per_grid

    dof_ranges = [original_order[d] for d in requested_domains]
    assert len(dof_ranges) > 0

    is_porepy_arrangement = requested_domains == operator_domains
    return np.concatenate(dof_ranges), is_porepy_arrangement


def equal_items_are_grouped(items: list):
    seen = set()
    previous = object()

    for item in items:
        if item != previous:
            if item in seen:
                return False
            seen.add(item)
            previous = item

    return True


def assemble_variable_indexer_from_equation_system(
    equation_system: pp.ad.EquationSystem,
):
    vars = equation_system.variables


def construct_variable_rearrange(model: pp.PorePyModel, var_tags: list[VariableTag]):
    available_variables = model.equation_system.variables

    assert equal_items_are_grouped([var.name for var in available_variables])

    previous_name: str | None = None
    offset = 0
    original_vars_order: dict[tuple[str, pp.GridLike], np.ndarray] = {}
    for var in available_variables:
        if var.name != previous_name:
            previous_name = var.name
            offset = 0
        domain = var.domain
        if isinstance(domain, pp.Grid):
            dofs_per_grid = (
                domain.num_cells * var._cells
                + domain.num_faces * var._faces
                + domain.num_nodes * var._nodes
            )
        elif isinstance(domain, pp.MortarGrid):
            # Interfaces have no faces.
            dofs_per_grid = (
                domain.num_cells * var._cells + domain.num_nodes * var._nodes
            )
        else:
            raise ValueError(type(domain))

        original_vars_order[var.name, domain] = np.arange(
            offset, offset + dofs_per_grid
        )
        offset += dofs_per_grid

    variable_dofs: dict[VariableTag, np.ndarray] = {}
    requested_vars: list[pp.ad.Variable] = []
    variable_name_domains: dict[str, pp.GridLikeSequence] = {}
    for var_tag in var_tags:
        domains = var_tag.defined_on.domains(model)
        if len(domains) == 0:
            print(f"No domains for variable tag {var_tag}")
            variable_dofs[var_tag] = np.zeros(0)
            continue

        variable_name_domains[var_tag.name] = domains
        dofs_list: list[np.ndarray] = []
        for d in domains:
            try:
                requested_vars.append(
                    next(
                        v
                        for v in available_variables
                        if v.name == var_tag.name and v.domain == d
                    )
                )
            except StopIteration:
                raise ValueError(f"Not found variable tag '{var_tag.name}' on {d}.")
            dofs_list.append(original_vars_order[var_tag.name, d])
        vals = np.concatenate(dofs_list)
        variable_dofs[var_tag] = vals

    is_porepy_arrangement = available_variables == requested_vars
    return variable_dofs, is_porepy_arrangement


def grid_info_equation(model: pp.PorePyModel, eq_tag: EquationTag):
    assert eq_tag.name in model.equation_system.equation_image_size_info
    dofs_info = model.equation_system.equation_image_size_info[eq_tag.name]
    return DofsInfo(
        cells=dofs_info.get("cells", 0),
        faces=dofs_info.get("faces", 0),
        nodes=dofs_info.get("nodes", 0),
    )


def assemble_residual(
    model: pp.PorePyModel, indexer: Indexer, eq_tags: list[EquationTag]
):
    residuals = []
    for eq_tag in eq_tags:
        equation = model.equation_system.equations.get(eq_tag.name, None)
        if equation is None:
            raise ValueError(eq_tag.name)

        res = model.equation_system._ad_parser.evaluate(
            op=equation,
            equation_system=model.equation_system,
            derivative=False,
            state=None,
        )
        assert isinstance(res, np.ndarray)

        assert eq_tag in indexer.equations_in_porepy_arrangement
        if not indexer.equations_in_porepy_arrangement[eq_tag]:
            permutation = indexer.equations_dofs[eq_tag]
            res = res[permutation]

        residuals.append(res)
    return residuals


def assemble_solution(
    model: pp.PorePyModel, indexer: Indexer, var_tags: list[VariableTag]
):
    solution = []
    for var_tag in var_tags:
        requested_domains = var_tag.defined_on.domains(model)

        md_var = model.equation_system.md_variable(
            name=var_tag.name, domains=requested_domains
        )

        sol = model.equation_system._ad_parser.evaluate(
            op=md_var,
            equation_system=model.equation_system,
            derivative=False,
            state=None,
        )
        assert isinstance(sol, np.ndarray)

        assert var_tag in indexer.variables_in_porepy_arrangement
        if not indexer.variables_in_porepy_arrangement[var_tag]:
            permutation = indexer.variables_dofs[var_tag]
            sol = sol[permutation]

        solution.append(sol)
    return solution


def assemble_residual_jacobian(
    model: pp.PorePyModel,
    indexer: Indexer,
    eq_tags: list[EquationTag],
    var_tags: list[VariableTag],
):
    if (
        all(indexer.variables_in_porepy_arrangement.values())
        and var_tags == indexer.variable_tags()
    ):
        variable_permutation = None
    else:
        variable_permutation = [
            indexer.variables_dofs_cumulative[tag] for tag in var_tags
        ]
        assert len(variable_permutation) > 0
        variable_permutation = np.concatenate(variable_permutation)

    residuals = []
    jacobian_rows = []
    for eq_tag in eq_tags:
        equation = model.equation_system.equations.get(eq_tag.name, None)
        if equation is None:
            raise ValueError(eq_tag.name)

        res_jac = model.equation_system._ad_parser.evaluate(
            op=equation,
            equation_system=model.equation_system,
            derivative=True,
            state=None,
        )
        res = res_jac.val
        jac = res_jac.jac

        assert isinstance(jac, csr_matrix)

        assert eq_tag in indexer.equations_in_porepy_arrangement
        if not indexer.equations_in_porepy_arrangement[eq_tag]:
            eq_permutation = indexer.equations_dofs[eq_tag]
            res = res[eq_permutation]
            jac = jac[eq_permutation]
        if variable_permutation is not None:
            jac = jac[:, variable_permutation]

        res *= -1  # to match what EquationSystem does
        residuals.append(res)
        jacobian_rows.append(jac)

    assert len(residuals) > 0

    return residuals, jacobian_rows


def assemble_indexer(
    model: pp.PorePyModel, eq_tags: list[EquationTag], var_tags: list[VariableTag]
):
    variables_dofs, var_is_porepy_arrangement = construct_variable_rearrange(
        model=model, var_tags=var_tags
    )
    assert variables_dofs is not None
    variables_in_porepy_arrangement = {
        tag: var_is_porepy_arrangement for tag in var_tags
    }

    equations_dofs: list[np.ndarray] = []
    equations_in_porepy_arrangement: dict[EquationTag, bool] = {}
    for eq_tag in eq_tags:
        equation = model.equation_system.equations.get(eq_tag.name, None)
        if equation is None:
            raise ValueError(eq_tag.name)
        assert equation.domains is not None
        eq_dofs, is_porepy_arrangmement = construct_rearrange(
            model=model,
            operator_domains=equation.domains,
            dofs_info=grid_info_equation(model=model, eq_tag=eq_tag),
            tag=eq_tag,
        )
        assert eq_dofs is not None
        equations_dofs.append(eq_dofs)
        equations_in_porepy_arrangement[eq_tag] = is_porepy_arrangmement

    return Indexer(
        equations_dofs={tag: dofs for tag, dofs in zip(eq_tags, equations_dofs)},
        variables_dofs=variables_dofs,
        equations_in_porepy_arrangement=equations_in_porepy_arrangement,
        variables_in_porepy_arrangement=variables_in_porepy_arrangement,
    )


def main():
    model = MyModel(params={"fracture_indices": [0, 1]})
    model.prepare_simulation()
    model.before_time_step()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()

    indexer = assemble_indexer(
        model=model,
        eq_tags=model.equation_system.equation_tags,
        var_tags=model.equation_system.variable_tags,
    )

    eq_tags = [
        pp.DefaultEquationTags.energy_balance,
        pp.DefaultEquationTags.mass_balance,
    ]
    var_tags = [
        pp.DefaultVariableTags.temperature,
        pp.DefaultVariableTags.pressure,
    ]

    # assemble residual
    res = assemble_residual(model=model, indexer=indexer, eq_tags=eq_tags)

    # assemble solution
    sol = assemble_solution(model, indexer=indexer, var_tags=var_tags)

    res1, jac = assemble_residual_jacobian(
        model, indexer=indexer, eq_tags=eq_tags, var_tags=var_tags
    )

    pass


if __name__ == "__main__":
    main()
