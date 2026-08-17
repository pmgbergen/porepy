"""This file is testing the functionality of `pp.BoundaryConditionMixin`.

It also contains tests verifying that `bc_values_*` methods return arrays with their
documented SI units.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils.models import MassBalance as MassBalance_
from porepy.examples.tracer_flow import TracerFlowModel
from porepy.models.momentum_balance import MomentumBalance


class CustomBoundaryCondition(pp.PorePyModel):
    """We define a custom dummy boundary condition.

    Neumann values are explicitly set, they are time dependent.
    Dirichlet values are equal to density on a boundary grid.

    """

    custom_bc_neumann_key = "custom_bc_neumann"

    def update_all_boundary_conditions(self) -> None:
        super().update_all_boundary_conditions()

        self.update_boundary_condition(
            name=self.custom_bc_neumann_key, function=self.bc_values_neumann
        )

    def bc_values_neumann(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Returns values on the whole boundary. We implicitly rely on the filter that
        sets zeros at the cells related to Dirichlet condition.

        Note: the values are time dependent.

        """
        t = self.time_data.time
        return np.arange(bg.num_cells) * bg.parent.dim * t

    def bc_type_dummy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """The north boundary is Dirichlet, the remainder is Neumann."""
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd=sd, faces=domain_sides.north, cond="dir")

    def create_dummy_ad_boundary_condition(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        op = lambda bgs: self.create_boundary_operator(
            name=self.custom_bc_neumann_key, domains=bgs
        )
        return self._combine_boundary_operators(
            subdomains=subdomains,
            dirichlet_operator=self.fluid.density,
            neumann_operator=op,
            robin_operator=op,
            bc_type=self.bc_type_dummy,
            name="boundary_condition_dummy",
            dim=1,
        )


class MassBalance(CustomBoundaryCondition, MassBalance_):
    pass


@pytest.mark.parametrize("t_end", [2, 3])
def test_boundary_condition_mixin(t_end: int):
    """We create a custom boundary condition operator and test that:
    1) The values are set correctly.
    2) Dirichlet values do not intersect Neumann values due to the filters.
    3) Previous timestep values are set correctly for the time dependent Neumann.

    """
    model = MassBalance(
        {
            "times_to_export": [],  # Suppress output for tests
        }
    )
    time_stepper = pp.time_stepper.TimeStepper(
        scheduler=pp.time_stepper.assemble_default_time_scheduler(
            schedule=[0, t_end],
            dt_init=1,
            constant_dt=True,
        )
    )
    pp.ModelRunner(model, time_stepper=time_stepper).run()

    subdomains = model.mdg.subdomains()

    for sd in subdomains:
        bc_type = model.bc_type_dummy(sd)
        bc_operator = model.create_dummy_ad_boundary_condition([sd])
        bc_val = model.equation_system.evaluate(bc_operator)

        # Testing the Dirichlet values. They should be equal to the fluid density.
        expected_val = model.fluid.reference_component.density
        assert np.allclose(bc_val[bc_type.is_dir], expected_val)
        assert not np.allclose(bc_val[bc_type.is_neu], expected_val)

        # Testing the Neumann values.
        bg = model.mdg.subdomain_to_boundary_grid(sd)
        assert bg is not None
        expected_val = np.arange(bg.num_cells) * bg.parent.dim * t_end
        # Projecting the expected value to the subdomain.
        expected_val = bg.projection().T @ expected_val
        assert np.allclose(bc_val[bc_type.is_neu], expected_val[bc_type.is_neu])

        # Testing previous timestep.
        bc_val_prev_ts = model.equation_system.evaluate(bc_operator.previous_timestep())
        expected_val = np.arange(bg.num_cells) * bg.parent.dim * (t_end - 1)
        # Projecting the expected value to the subdomain.
        expected_val = bg.projection().T @ expected_val
        assert np.allclose(bc_val_prev_ts[bc_type.is_neu], expected_val[bc_type.is_neu])


"""Here follows mixins related to testing of Robin limit cases, and eventually the test
itself. """


class BCValuesDirichletIndices(pp.PorePyModel):
    """Boundary values for primary variables on Dirichlet boundaries.

    Used for:
    * Momentum balance
    * Mass and energy balance.

    """

    def rob_inds(self, sd) -> np.ndarray:
        """Indices for the non-Dirichlet boundaries for test.

        The Robin limit case test tests Robin approximating either Dirichlet or Neumann.
        All test models have Dirichlet on dir_inds (Dirichlet index) boundaries, and
        Robin approximating Dirichlet or Neumann on the remaining ones. This method
        returns the indices of the north and south boundaries, which are the Dirichlet
        indices.

        """
        domain_sides = self.domain_boundary_sides(sd)
        return domain_sides.north + domain_sides.south

    def dir_inds(self, sd) -> np.ndarray:
        """Indices for the Dirichlet boundaries for test.

        The Robin limit case test tests Robin approximating either Dirichlet or Neumann.
        All test models have Dirichlet on dir_inds (Dirichlet index) boundaries, and
        Robin approximating Dirichlet or Neumann on the remaining ones. This method
        returns the indices of the west and east boundaries, which are the Robin
        indices.

        """
        domain_sides = self.domain_boundary_sides(sd)
        return domain_sides.west + domain_sides.east

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns displacement values in the x-direction of the Dirichlet
        boundaries."""
        values = np.zeros((self.nd, bg.num_cells))
        values[0, self.dir_inds(bg)] = 42
        return values.ravel("F")

    def _bc_values_scalar(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns scalar values on the Dirichlet boundaries."""
        values = np.zeros(bg.num_cells)
        values[self.dir_inds(bg)] = 42
        return values

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns pressure boundary values."""
        return self._bc_values_scalar(bg=bg)

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns temperature boundary values."""
        return self._bc_values_scalar(bg=bg)


class BCRobin(pp.PorePyModel):
    """Set Dirichlet and Robin for momentum balance and mass and energy balance.

    Sets Dirichlet on dir_inds-boundaries, and Robin on the remaining ones. The value of
    the Robin weight is determined from the parameter "alpha" in the params dictionary.
    This class also sets Robin boundary values.

    This class is common for all the test classes that enters into testing Robin limit
    cases.

    """

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Sets Robin and Dirichlet conditions.

        Sets Dirichlet boundary condition type on the Dirichlet index-boundaries and
        Robin on all others.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.all_bf, "rob")
        bc.is_rob[:, self.dir_inds(sd)] = False
        bc.is_dir[:, self.dir_inds(sd)] = True

        alpha = self.params["alpha"]

        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        bc.robin_weight = (
            np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), order="F") * alpha
        )
        return bc

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Helper function for setting boundary conditions on scalar fields.

        Sets Dirichlet boundary condition type on the Dirichlet index-boundaries and
        Robin on all others.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.all_bf, "rob")
        bc.is_rob[self.dir_inds(sd)] = False
        bc.is_dir[self.dir_inds(sd)] = True

        bc.robin_weight = np.ones(sd.num_faces) * self.params["alpha"]
        return bc

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type_scalar(sd=sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type_scalar(sd=sd)


class BCNeumannReference(pp.PorePyModel):
    """Set Dirichlet and Neumann for momentum balance and mass and energy balance."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Assigns Neumann and Dirichlet boundaries for the Neumann case."""
        bc = pp.BoundaryConditionVectorial(sd, self.dir_inds(sd), "dir")
        return bc

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Helper function for setting boundary conditions on scalar fields.

        The function sets Dirichlet on the Dirichlet index boundaries, and Neumann on
        all others.

        """

        bc = pp.BoundaryCondition(sd, self.dir_inds(sd), "dir")
        return bc

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type_scalar(sd=sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type_scalar(sd=sd)


class BCValuesFlux(pp.PorePyModel):
    def bc_values_fourier_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns Fourier flux values on Robin index boundaries."""
        return self._bc_values_scalar_flux(bg)

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns Darcy flux values on Robin index boundaries."""
        return self._bc_values_scalar_flux(bg)

    def _bc_values_scalar_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        values = np.zeros((bg.num_cells))
        val = 24
        if self.params["alpha"] > 0:  # Robin-Dirichlet
            # The flux value here will be the value of the Robin condition and not seen
            # in the Dirichlet reference case. We need to multiply with the cell volume
            # and the alpha value to account for Robin being interpreted as an
            # integrated flux (volume) and being compared to alpha * u, since the Robin
            # condition is on the form sigma * n + alpha * u = G and the first term is
            # negligible for large alpha.
            volumes = bg.cell_volumes[self.rob_inds(bg)]
            val *= volumes * self.params["alpha"]
        values[self.rob_inds(bg)] = val
        return values.ravel("F")

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns stress values on the non-Dirichlet boundaries."""
        values = np.zeros((self.nd, bg.num_cells))
        val = 24
        if self.params["alpha"] > 0:  # Robin-Dirichlet
            # The flux value here will be the value of the Robin condition and not seen
            # in the Dirichlet reference case. We need to multiply with the cell volume
            # and the alpha value to account for Robin being interpreted as an
            # integrated flux (volume) and being compared to alpha * u, since the Robin
            # condition is on the form sigma * n + alpha * u = G and the first term is
            # negligible for large alpha.
            volumes = bg.cell_volumes[self.rob_inds(bg)]
            val *= volumes * self.params["alpha"]
        values[0, self.rob_inds(bg)] = val
        return values.ravel("F")


class BCDirichletReference(pp.PorePyModel):
    """Set all Dirichlet boundaries for momentum balance and mass and energy balance."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Assigns Dirichlet boundaries on all domain boundary sides."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.all_bf, "dir")
        return bc

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Helper function for setting boundary conditions on scalar fields.

        The function sets Dirichlet on all boundaries.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.all_bf, "dir")
        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns displacement values in the x-direction of the Robin index
        boundaries."""
        values = super().bc_values_displacement(bg=bg)
        values = values.reshape((self.nd, bg.num_cells), order="F")
        inds = self.rob_inds(bg)
        values[0, inds] = 24
        return values.ravel("F")

    def _bc_values_scalar(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Set the values for scalar fields.

        Parameters:
            bg: Boundary grid.

        Returns:
            np.ndarray: Boundary values.

        """
        # Call super to get values for Dirichlet boundaries.
        values = super()._bc_values_scalar(bg)
        values[self.rob_inds(bg)] = 24
        return values


class CommonMassEnergyBalance(
    SquareDomainOrthogonalFractures,
    BCValuesDirichletIndices,
    BCValuesFlux,
    pp.MassAndEnergyBalance,
):
    """Base mass and energy balance model.

    The model in this class is common for the reference class for mass and energy
    balance and for the "test" class for mass and energy balance. The "test" class is
    the class which represents a problem model with Robin boundaries.

    """


class MassAndEnergyBalanceRobin(BCRobin, CommonMassEnergyBalance):
    """Mass and energy balance with Robin and Dirichlet conditions.

    The methods dir_inds and rob_inds determine which boundaries are Dirichlet and which
    are Robin.

    """


class CommonMomentumBalance(
    SquareDomainOrthogonalFractures,
    BCValuesDirichletIndices,
    BCValuesFlux,
    MomentumBalance,
):
    """Base momentum balance model.

    The model in this class is common for the reference class for momentum balance and
    for the "test" class for momentum balance. The "test" class is the class which
    represents a problem model with Robin boundaries.

    """


class MomentumBalanceRobin(BCRobin, CommonMomentumBalance):
    """Momentum balance with Robin and Dirichlet conditions.

    The methods dir_inds and rob_inds determine which boundaries are Dirichlet and which
    are Robin.

    """


def run_model(model_class: type[pp.PorePyModel], alpha: float) -> dict[str, np.ndarray]:
    params = {
        "fracture_indices": [],
        "meshing_arguments": {"cell_size": 0.5},
        "times_to_export": [],  # Suppress output for tests
    }

    params["alpha"] = alpha
    model = model_class(params)
    pp.ModelRunner(model).run()
    sd = model.mdg.subdomains(dim=2)[0]

    if isinstance(model, MomentumBalance):
        displacement = model.equation_system.evaluate(model.displacement([sd]))
        return {"displacement": displacement}
    elif isinstance(model, pp.MassAndEnergyBalance):
        pressure = model.equation_system.evaluate(model.pressure([sd]))
        temperature = model.equation_system.evaluate(model.temperature([sd]))
        return {"temperature": temperature, "pressure": pressure}


# Parameterize the test function with the necessary balance types and conditions
@pytest.mark.parametrize(
    "model_class, reference_model_class, alpha",
    [
        (MomentumBalanceRobin, CommonMomentumBalance, 0),
        (MassAndEnergyBalanceRobin, CommonMassEnergyBalance, 0),
        (MassAndEnergyBalanceRobin, CommonMassEnergyBalance, 1e8),
        (MomentumBalanceRobin, CommonMomentumBalance, 1e8),
    ],
)
def test_robin_limit_case(
    model_class: type[pp.PorePyModel],
    reference_model_class: type[pp.PorePyModel],
    alpha: float,
):
    r"""Test Robin limit cases.

    The Robin conditions are implemented on the form: sigma * n + alpha * u = G. That
    means that setting Robin conditions with alpha = 0 should correspond to setting
    Neumann conditions. For large alpha (alpha -> \infty), the Robin conditions should
    correspond to Dirichlet conditions.

    We test this for momentum balance and mass and energy balance.

    Common for all models is that the have Dirichlet conditions on the boundaries
    returned by the method dir_inds.

    The model classes with documentation are further up in this document.

    """
    if alpha > 0:
        reference_bc_class = BCDirichletReference
    elif alpha == 0:
        reference_bc_class = BCNeumannReference

    class LocalReferenceModel(reference_bc_class, reference_model_class):
        """Reference class with the correct reference boundary types."""

    rob_results = run_model(model_class, alpha)
    reference_results = run_model(LocalReferenceModel, alpha)

    assert all(
        np.allclose(rob_results[key], reference_results[key], atol=1e-7)
        for key in rob_results.keys()
    )


"""Tests that ``bc_values_*`` methods return arrays with their documented SI units, by
exploiting :class:`porepy.Units` non-dimensionalization: a correctly unit-tagged BC
produces a solution that, recovered to SI, is invariant under any choice of internal
unit system.
"""


# Helpers for unit-invariance tests
@dataclass(frozen=True)
class _BCUnitInvarianceSpec:
    """Specification for a single BC unit-invariance test."""

    probe_model_class: type[pp.PorePyModel]
    probe_label: str
    observable_accessor: Callable[[pp.PorePyModel, pp.Grid], np.ndarray]
    observable_si_unit: str
    declared_bc_unit: str
    extraction_subdomain_dim: int = 2


#: Unit scalings probing each axis of the BC's SI dimensions.
_DEFAULT_UNIT_SCALINGS: list[pp.Units] = [
    pp.Units(m=10.0),
    pp.Units(kg=0.01),
    pp.Units(m=5.0, kg=12.0),
    pp.Units(m=3, kg=7, K=0.23),
]


def _unit_scaling_label(units: pp.Units) -> str:
    """Readable pytest parametrize-id for a :class:`pp.Units` instance."""

    return f"m={units.m:g}_kg={units.kg:g}_K={units.K:g}"


def _bc_spec_label(spec: _BCUnitInvarianceSpec) -> str:
    """Readable pytest parametrize-id for a BC unit-invariance spec."""

    return spec.probe_label


def _run_and_recover_in_si(spec: _BCUnitInvarianceSpec, units: pp.Units) -> np.ndarray:
    """Run the probe model under ``units``; return observable in SI."""

    # Convert cell_size to internal units so the resolved mesh is invariant.
    cell_size_si = 0.5
    cell_size_internal = units.convert_units(cell_size_si, "m")

    params: dict[str, Any] = {
        "units": units,
        "fracture_indices": [],
        "meshing_arguments": {"cell_size": cell_size_internal},
        "times_to_export": [],
        "grid_type": "cartesian",
    }
    solver_params = {
        "nl_max_iterations": 100,
        "nl_convergence_res_atol": 1e-6,
        "nl_convergence_inc_atol": 1e-6,
    }

    model = spec.probe_model_class(params)
    pp.ModelRunner(model, solver_params).run()

    sd = model.mdg.subdomains(dim=spec.extraction_subdomain_dim)[0]
    var_internal = spec.observable_accessor(model, sd)

    var_si: np.ndarray = units.convert_units(
        var_internal,
        spec.observable_si_unit,
        to_si=True,
    )
    return var_si


def _assert_bc_unit_invariance(
    spec: _BCUnitInvarianceSpec,
    units: pp.Units,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> None:
    """Recovered SI observable must be invariant under unit rescaling."""

    baseline = _run_and_recover_in_si(spec, pp.Units())

    # Guard against trivially-passing tests: zero, NaN,
    # or degenerate solution.
    assert np.all(np.isfinite(baseline)), (
        f"Baseline for '{spec.probe_label}' contains non-finite values."
    )
    assert np.max(np.abs(baseline)) > 0.0, (
        f"Baseline for '{spec.probe_label}' is identically zero."
    )

    scaled = _run_and_recover_in_si(spec, units)

    np.testing.assert_allclose(
        scaled,
        baseline,
        rtol=rtol,
        atol=atol,
        err_msg=(
            f"BC unit-invariance violated for '{spec.probe_label}' under "
            f"scaling {_unit_scaling_label(units)}. Declared BC unit: "
            f"'{spec.declared_bc_unit}'."
        ),
    )


# ---------------------------------------------------------------
# BC value: Darcy flux
# ---------------------------------------------------------------


# Empirically verified unit for bc_values_darcy_flux in 2D: integrated
# Darcy flux is K*grad(p)*face_area, SI unit m^nd * Pa where nd is the
# ambient dimension. In 2D, this gives m^2*Pa.
_DARCY_FLUX_UNIT: str = "m^2*Pa"
_DARCY_FLUX_VALUE_SI: float = 1.0e-3


class _DarcyFluxBCProbe(SquareDomainOrthogonalFractures, MassBalance_):
    """Single-phase flow: Dirichlet p=0 on west, Neumann Darcy flux on east."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.west, "dir")

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        vals = np.zeros(bg.num_cells)
        sides = self.domain_boundary_sides(bg)
        vals[sides.east] = self.units.convert_units(
            _DARCY_FLUX_VALUE_SI,
            _DARCY_FLUX_UNIT,
        )
        return vals

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros(bg.num_cells)


# Test specification for bc_values_darcy_flux: probe model, primary variable
# accessor (pressure on the matrix subdomain), and the declared BC unit.
_DARCY_FLUX_SPEC = _BCUnitInvarianceSpec(
    probe_model_class=_DarcyFluxBCProbe,
    probe_label="darcy_flux",
    observable_accessor=lambda model, sd: model.equation_system.evaluate(
        model.pressure([sd])
    ),
    observable_si_unit="Pa",
    declared_bc_unit=_DARCY_FLUX_UNIT,
)


# -----------------------------------------------------------------------------
# Bc value: Fluid flux
# -----------------------------------------------------------------------------


# Empirically verified unit for bc_values_fluid_flux in 2D: integrated fluid mass
# flux is rho/mu*K*grad(p)*face_area, and SI unit is kg*m^(nd-3)*s^-1, where nd
# is the ambient dimension. In 2D, this gives kg*m^-1*s^-1.
_FLUID_FLUX_UNIT: str = "kg*m^-1*s^-1"
_FLUID_FLUX_VALUE_SI: float = 1.0e-3


class _FluidFluxBCProbe(SquareDomainOrthogonalFractures, MassBalance_):
    """Single-phase flow: Dirichlet p=0 on west, Neumann fluid flux on east."""

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.west, "dir")

    def bc_values_fluid_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        vals = np.zeros(bg.num_cells)
        sides = self.domain_boundary_sides(bg)
        vals[sides.east] = self.units.convert_units(
            _FLUID_FLUX_VALUE_SI,
            _FLUID_FLUX_UNIT,
        )
        return vals

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros(bg.num_cells)


# Test specification for bc_values_fluid_flux: probe model, primary variable
# accessor (pressure on the matrix subdomain), and the declared BC unit.
_FLUID_FLUX_SPEC = _BCUnitInvarianceSpec(
    probe_model_class=_FluidFluxBCProbe,
    probe_label="fluid_flux",
    observable_accessor=lambda model, sd: model.equation_system.evaluate(
        model.pressure([sd])
    ),
    observable_si_unit="Pa",
    declared_bc_unit=_FLUID_FLUX_UNIT,
)


# -----------------------------------------------------------------------------
# Bc value: stress
# -----------------------------------------------------------------------------


# Empirically verified unit for bc_values_stress in 2D: traction integrated over
# the boundary face, and SI unit is Pa*m^(nd-1), where nd is the ambient dimension.
# In 2D, this gives Pa*m.
_STRESS_UNIT: str = "Pa*m"
_STRESS_VALUE_SI: float = 1.0e-3


class _StressBCProbe(CommonMomentumBalance):
    """Linear elasticity: Dirichlet displacement on west and south,
    Neumann stress on east.
    """

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        sides = self.domain_boundary_sides(sd)
        dir_sides = sides.west + sides.south
        return pp.BoundaryConditionVectorial(sd, dir_sides, "dir")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros((self.nd, bg.num_cells)).ravel("F")

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        vals = np.zeros((self.nd, bg.num_cells))
        sides = self.domain_boundary_sides(bg)
        vals[0, sides.east] = self.units.convert_units(
            _STRESS_VALUE_SI,
            _STRESS_UNIT,
        )
        return vals.ravel("F")


# Test specification for bc_values_stress: probe model, primary variable
# accessor (displacement on the matrix subdomain), and the declared BC unit.
_STRESS_SPEC = _BCUnitInvarianceSpec(
    probe_model_class=_StressBCProbe,
    probe_label="stress",
    observable_accessor=lambda model, sd: model.equation_system.evaluate(
        model.displacement([sd])
    ),
    observable_si_unit="m",
    declared_bc_unit=_STRESS_UNIT,
)


# -----------------------------------------------------------------------------
# Bc value: Fourier flux
# -----------------------------------------------------------------------------


# Empirically verified unit for bc_values_fourier_flux in 2D: integrated conducted
# heat flux lambda * grad(T)* face_area, SI unit W * m^(nd-3) where nd is the
# ambient dimension. In 2D this gives W * m^-1.
_FOURIER_FLUX_UNIT: str = "W*m^-1"
_FOURIER_FLUX_VALUE_SI: float = 1.0e-3


class _FourierFluxBCProbe(SquareDomainOrthogonalFractures, pp.MassAndEnergyBalance):
    """Mass + energy transport: Dirichlet p=0 and T=0 on west, Neumann zero flow
    everywhere; Neumann Fourier flux on east. Pressure is pinned to isolate Fourier
    conduction as the only driver of the temperature field.
    """

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.west, "dir")

    def bc_values_fourier_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        vals = np.zeros(bg.num_cells)
        sides = self.domain_boundary_sides(bg)
        vals[sides.east] = self.units.convert_units(
            _FOURIER_FLUX_VALUE_SI,
            _FOURIER_FLUX_UNIT,
        )
        return vals

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros(bg.num_cells)

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.all_bf, "dir")

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros(bg.num_cells)


# Test specification for bc_values_fourier_flux: probe model, primary variable
# accessor (temperature on the matrix subdomain), and the declared BC unit.
_FOURIER_FLUX_SPEC = _BCUnitInvarianceSpec(
    probe_model_class=_FourierFluxBCProbe,
    probe_label="fourier_flux",
    observable_accessor=lambda model, sd: model.equation_system.evaluate(
        model.temperature([sd])
    ),
    observable_si_unit="K",
    declared_bc_unit=_FOURIER_FLUX_UNIT,
)


# -----------------------------------------------------------------------------
# Bc value: Enthalpy flux
# -----------------------------------------------------------------------------


# Empirically verified unit for bc_values_enthalpy_flux in 2D: integrated enthalpy
# flux rho * enthalpy * volumetric_flux * face_area, SI unit W * m^(nd-3) where nd
# is the ambient dimension. In 2D this gives W * m^-1.
_ENTHALPY_FLUX_UNIT: str = "W*m^-1"
_ENTHALPY_FLUX_VALUE_SI: float = 1.0e-3


class _EnthalpyFluxBCProbe(SquareDomainOrthogonalFractures, pp.MassAndEnergyBalance):
    """Mass + energy transport: Dirichlet T=0 on west, Neumann enthalpy flux on east.
    Pressure pinned to zero everywhere so the Darcy flow that advects enthalpy is
    driven only by the enthalpy BC's coupling, not by independent pressure forcing.
    """

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.west, "dir")

    def bc_values_enthalpy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        vals = np.zeros(bg.num_cells)
        sides = self.domain_boundary_sides(bg)
        vals[sides.east] = self.units.convert_units(
            _ENTHALPY_FLUX_VALUE_SI,
            _ENTHALPY_FLUX_UNIT,
        )
        return vals

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros(bg.num_cells)

    def bc_values_enthalpy(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros(bg.num_cells)

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.all_bf, "dir")

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros(bg.num_cells)


# Test specification for bc_values_enthalpy_flux: probe model, primary variable
# accessor (temperature on the matrix subdomain), and the declared BC unit.
_ENTHALPY_FLUX_SPEC = _BCUnitInvarianceSpec(
    probe_model_class=_EnthalpyFluxBCProbe,
    probe_label="enthalpy_flux",
    observable_accessor=lambda model, sd: model.equation_system.evaluate(
        model.temperature([sd])
    ),
    observable_si_unit="K",
    declared_bc_unit=_ENTHALPY_FLUX_UNIT,
)


# -----------------------------------------------------------------------------
# Bc value: Component flux
# -----------------------------------------------------------------------------


# Empirically verified unit for bc_values_component_flux in 2D: integrated component
# mass flux, SI unit kg * m^(nd-3) * s^-1 where nd is the ambient dimension.
# In 2D this gives kg * m^-1 * s^-1.
_COMPONENT_FLUX_UNIT: str = "kg * m^-1 * s^-1"
_TRACER_COMPONENT_FLUX_VALUE_SI: float = 1.0e-8
_WATER_COMPONENT_FLUX_VALUE_SI: float = 9.0e-8


class _ComponentFluxBCProbe(TracerFlowModel):
    """Component transport: Neumann component mass flux on east.

    East boundary receives prescribed inward component mass fluxes for both
    the tracer and reference/water components. West is a pressure Dirichlet
    outlet, while north and south are no-flow boundaries.
    """

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.all_bf, "neu")
        bc.is_neu[sides.west] = False
        bc.is_dir[sides.west] = True
        return bc

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        vals = np.zeros(bg.num_cells)
        sides = self.domain_boundary_sides(bg)
        vals[sides.west] = self.units.convert_units(2.0, "Pa")
        return vals

    def bc_values_overall_fraction(
        self,
        component: pp.Component,
        bg: pp.BoundaryGrid,
    ) -> np.ndarray:
        return np.zeros(bg.num_cells)

    def bc_values_component_flux(
        self,
        component: pp.Component,
        bg: pp.BoundaryGrid,
    ) -> np.ndarray:
        vals = np.zeros(bg.num_cells)
        sides = self.domain_boundary_sides(bg)

        if component.name == "tracer":
            value_si = _TRACER_COMPONENT_FLUX_VALUE_SI
        else:
            value_si = _WATER_COMPONENT_FLUX_VALUE_SI

        vals[sides.east] = -self.units.convert_units(
            value_si,
            _COMPONENT_FLUX_UNIT,
        )
        return vals


# Test specification for bc_values_component_flux: probe model, derived variable
# accessor (component_flux on the matrix subdomain), and the declared BC unit.
_COMPONENT_FLUX_SPEC = _BCUnitInvarianceSpec(
    probe_model_class=_ComponentFluxBCProbe,
    probe_label="component_flux",
    observable_accessor=lambda model, sd: model.equation_system.evaluate(
        model.component_flux(model.fluid.components[1], [sd])
    ),
    observable_si_unit="kg * m^-1 * s^-1",
    declared_bc_unit=_COMPONENT_FLUX_UNIT,
)


_BC_UNIT_INVARIANCE_SPECS: list[_BCUnitInvarianceSpec] = [
    _DARCY_FLUX_SPEC,
    _FLUID_FLUX_SPEC,
    _STRESS_SPEC,
    _FOURIER_FLUX_SPEC,
    _ENTHALPY_FLUX_SPEC,
    _COMPONENT_FLUX_SPEC,
]


@pytest.mark.skipped(reason="slow")
@pytest.mark.parametrize(
    "spec",
    _BC_UNIT_INVARIANCE_SPECS,
    ids=_bc_spec_label,
)
@pytest.mark.parametrize(
    "units",
    _DEFAULT_UNIT_SCALINGS,
    ids=_unit_scaling_label,
)
def test_bc_values_unit_invariance(
    spec: _BCUnitInvarianceSpec,
    units: pp.Units,
) -> None:
    """Test that verifies the recovered SI observable is invariant
    under unit rescaling.
    """

    _assert_bc_unit_invariance(
        spec,
        units,
        atol=1e-10,
    )
