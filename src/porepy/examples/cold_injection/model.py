"""Example for 2-phase, 2-component flow using equilibrium calculations.

Simulates the injection of CO2 into an initially water-saturated 2D domain, using a
mD model with points as wells.

Note:
    It uses the numba-compiled version of the Peng-Robinson EoS and a numba-compiled
    unified flash.

    Import and compilation take some time.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, cast

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

import porepy as pp
import porepy.compositional.flash as pf
import porepy.compositional.peng_robinson as pr
import porepy.models.compositional_flow as cf
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.compositional.compiled_eos import ScalarFunction

from .config import ModelConfig

logger = logging.getLogger(__name__)

# NOTE this is a bunch of mixins, we omitt defining all the mixed-in functions.

# mypy: disable-error-code="attr-defined"


class FluidPoreInteraction(ModelConfig):
    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]

    def pore_volume(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        cell_volumes = self.wrap_grid_attribute(subdomains, "cell_volumes", dim=1)
        aperture = self.aperture(subdomains)
        porosity = self.porosity(subdomains)

        return cell_volumes * aperture * porosity

    def pore_volume_jump(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        pore_volume = self.pore_volume(subdomains)
        op = pore_volume / pore_volume.previous_timestep()
        op.set_name("pore_volume_jump")
        return op

    @pp.ad.cached_method
    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        a = super().aperture(subdomains)  # type:ignore[safe-super]

        jump_factor = pp.ad.Function(self.a_jump, "a_jump")(self.ad_time)

        sds_w_jump = [sd for sd in subdomains if 0 < sd.dim < self.nd]
        sds_wo_jump = [sd for sd in subdomains if sd not in sds_w_jump]
        projection = pp.ad.SubdomainProjections(subdomains)

        v0 = pp.wrap_as_dense_ad_array(1, size=sum([g.num_cells for g in sds_wo_jump]))
        v1 = pp.wrap_as_dense_ad_array(1, size=sum([g.num_cells for g in sds_w_jump]))

        a *= projection.cell_prolongation(
            sds_wo_jump
        ) @ v0 + projection.cell_prolongation(sds_w_jump) @ (v1 * jump_factor)
        a.set_name("jumping_aperture")
        return a

    def a_jump(self, t: float) -> float:
        """Returns a scaling factor for aperture depending on time."""

        f = 1.0
        for ti, fi in self._APERTURE_FACTOR_AFTER_TIME:
            if t >= ti:
                f = fi

        return f


class FluidEoS(pr.CompiledPengRobinson):
    """Constant viscosity and thermal conductivity for all phases.

    Viscosity is set to 1e-3, thermal conductivity to 1.

    """

    def get_mu_function(self) -> ScalarFunction:
        def mu_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            return 1e-3

        return mu_c

    def get_kappa_function(self) -> ScalarFunction:
        def kappa_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            return 1.0

        return kappa_c


class FluidMixture(ModelConfig):
    """2-component, 2-phase fluid with H2O and CO2, and a liquid and gas phase."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def get_components(self) -> Sequence[pp.FluidComponent]:
        return pp.compositional.load_fluid_constants(self._COMPONENT_NAMES, "chemicals")

    def get_phase_configuration(
        self, components: Sequence[pp.FluidComponent]
    ) -> Sequence[
        tuple[pp.compositional.PhysicalState, str, pp.compositional.EquationOfState]
    ]:
        eos = FluidEoS(
            components,
            self._IDEAL_COMPONENTS,
            pr.get_bip_matrix(components),
        )
        return [
            (pp.compositional.PhysicalState.liquid, "L", eos),
            (pp.compositional.PhysicalState.gas, "G", eos),
        ]

    def dependencies_of_phase_properties(
        self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        d = [self.pressure]
        if isinstance(self, pp.energy_balance.VariablesEnergyBalance):
            d += [self.temperature]
        return d + [  # type:ignore[return-value]
            phase.extended_fraction_of[comp] for comp in phase
        ]


class SolutionStrategy(ModelConfig):
    """Strategy implementing choice of flash based on dimensions and well-tags.

    Performs the pT flash on domains tagged as injection wells in every iteration.
    Otherwise it performs the base specification as specified.
    If isochoric preconditioning is activated, performs isochoric calculations
    before entering the nonlinear loop.

    """

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    porosity: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def nonlinear_flash_preconditioning(self):
        do_default_flash = self.do_flash_preconditioning()

        for grid in self.mdg.subdomains():
            # if "injection_well" in sd.tags and do_default_flash:
            if "injection_well" in grid.tags and (
                self._FLASH_PT_INJECTION or do_default_flash
            ):
                equ_spec = pf.IsobaricSpecifications(
                    p=self.equation_system.evaluate(self.pressure([grid])),
                    T=self.equation_system.evaluate(self.temperature([grid])),
                )

                self.local_equilibrium(
                    grid,
                    specification=equ_spec,
                )

            elif do_default_flash:
                self.local_equilibrium(grid)

    def update_interface_fluxes(self) -> None:
        interfaces = self.mdg.interfaces(codim=1)

        idfe = self.interface_darcy_flux_equation(interfaces)
        idf = self.interface_darcy_flux(interfaces)

        for _ in range(5):
            A, b = self.equation_system.assemble(
                evaluate_jacobian=True, equations=[idfe], variables=[idf]
            )
            norm = np.linalg.norm(b)
            if norm < 1e-5:
                break
            delta_idf = sps.linalg.spsolve(A, b)
            self.equation_system.set_variable_values(
                delta_idf,
                [idf],
                iterate_index=0,
                additive=True,
            )
            self.rediscretize_fluxes()
            self.update_flux_values()
            self.rediscretize()

        self.update_discretization_parameters()
        self.rediscretize_fluxes()
        self.update_flux_values()
        self.rediscretize()

        if not isinstance(self, pp.energy_balance.TotalEnergyBalanceEquations):
            return

        # NOTE: Enthalpy flux equation is linear in the respective unknown.
        # So adding the negative residual of the equation will solve the equation
        # exactly.
        intf_enthalpy = self.equation_system.evaluate(
            self.interface_enthalpy_flux(interfaces)
            - self.interface_enthalpy_flux_equation(interfaces)
        )
        self.equation_system.set_variable_values(
            cast(np.ndarray, intf_enthalpy),
            [self.interface_enthalpy_flux(interfaces)],
            iterate_index=0,
        )

        iffe = self.interface_fourier_flux_equation(interfaces)
        iff = self.interface_fourier_flux(interfaces)

        for _ in range(5):
            A, b = self.equation_system.assemble(
                evaluate_jacobian=True, equations=[iffe], variables=[iff]
            )
            norm = np.linalg.norm(b)
            if norm < 1e-1:
                break
            delta_iff = sps.linalg.spsolve(A, b)
            self.equation_system.set_variable_values(
                delta_iff,
                [iff],
                iterate_index=0,
                additive=True,
            )
            self.rediscretize_fluxes()
            self.update_flux_values()
            self.rediscretize()

        self.update_discretization_parameters()
        self.rediscretize_fluxes()
        self.update_flux_values()
        self.rediscretize()

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()  # type:ignore[misc]

        if self._ISOCHORIC_NPC_SPEC == pf.FlashSpec.none:
            return

        self.isochoric_npc_done = False

        atol = self.params["flash_params"]["solver_params"]["atol_res"]

        for grid in self.mdg.subdomains():
            if 0 < grid.dim < self.nd and isinstance(self, FluidPoreInteraction):
                v_jump_factor = self.equation_system.evaluate(
                    self.pore_volume_jump([grid])
                )
                if np.max(v_jump_factor) > 1 + 1e-5:
                    if self.has_fluid_volume_variable:
                        v = self.equation_system.evaluate(
                            self.specific_fluid_volume([grid])
                        )
                    else:
                        v = self.equation_system.evaluate(
                            self.fluid.specific_volume([grid])
                        )

                    # New volume after jump.
                    v *= v_jump_factor

                    assert np.all(v > 0), "Bad specific volume."
                    if self._ISOCHORIC_NPC_SPEC == pf.FlashSpec.vT:
                        equ_spec = pf.IsochoricSpecifications(
                            v=v,
                            T=self.equation_system.evaluate(self.temperature([grid])),
                        )
                    elif self._ISOCHORIC_NPC_SPEC == pf.FlashSpec.vu:
                        if self.has_fluid_internal_energy_variable:
                            u = self.equation_system.evaluate(
                                self.specific_fluid_internal_energy([grid])
                            )
                        else:
                            u = self.equation_system.evaluate(
                                self.fluid.specific_internal_energy([grid])
                            )
                        equ_spec = pf.IsochoricSpecifications(
                            v=v,
                            u=u,
                        )
                    logger.info(
                        f"Performing isochoric preconditioning on grid {grid.id}."
                    )
                    self.params["flash_params"]["solver_params"]["atol_res"] = min(
                        1e-8, atol
                    )
                    # Perform full, isochoric flash, including initial guess.
                    res: tuple[pf.FlashResults, NDArray[np.bool_]] = (
                        self.local_equilibrium(
                            grid,
                            specification=equ_spec,
                            initial_guess_from_current_state=False,
                            update_secondary_quantities=True,
                        )
                    )
                    if self.has_fluid_volume_variable:
                        self.equation_system.set_variable_values(
                            res[0].v,
                            [self.specific_fluid_volume([grid])],  # type:ignore[arg-type]
                            iterate_index=0,
                        )
                    self.isochoric_npc_done = True
                    self.params["flash_params"]["solver_params"]["atol_res"] = atol

        if self.isochoric_npc_done:
            self.update_derived_quantities()
            self.update_interface_fluxes()
            # self.iterate_pT()
            # self.update_interface_fluxes()

    def get_internal_energy(self, sd: pp.Grid, prev_time: bool) -> np.ndarray:
        subdomains = [sd]

        op: pp.ad.Operator = self.volume_integral(
            (
                self.fluid.density(subdomains)
                * self.fluid.specific_enthalpy(subdomains)
                - self.pressure(subdomains)
            )
            * self.porosity(subdomains),
            subdomains,
            dim=1,
        )

        if prev_time:
            op = op.previous_timestep()
        return cast(np.ndarray, self.equation_system.evaluate(op))

    def solve_linear_system(self) -> np.ndarray:
        try:
            sol = super().solve_linear_system()  # type:ignore[safe-super]
        except Exception as e:
            logger.warning(f"Error occurred while solving linear system: {e}")
            sol = np.full_like(self.linear_system[1], np.nan)

        return sol

    def iterate_pT(self) -> None:
        """Additional iterations over p and T using mass and energy balance to update
        elliptic part."""

        eqs = []
        vars = []
        subdomains = self.mdg.subdomains()
        codim_1_interfaces = self.mdg.interfaces(codim=1)
        codim_2_interfaces = self.mdg.interfaces(codim=2)

        if isinstance(self, pp.fluid_mass_balance.VariablesSinglePhaseFlow):
            vars += [
                self.pressure(subdomains),
                self.interface_darcy_flux(codim_1_interfaces),
                self.well_flux(codim_2_interfaces),
            ]
            eqs += [
                pp.fluid_mass_balance.FluidMassBalanceEquations.primary_equation_name(),
                "interface_darcy_flux_equation",
                "well_flux_equation",
            ]
            if "production_pressure_constraint" in self.equation_system.equations:
                eqs += ["production_pressure_constraint"]
        if isinstance(self, pp.energy_balance.VariablesEnergyBalance):
            vars += [
                self.temperature(subdomains),
                self.interface_enthalpy_flux(codim_1_interfaces),
                self.interface_fourier_flux(codim_1_interfaces),
                self.well_enthalpy_flux(codim_2_interfaces),
            ]
            eqs += [
                pp.energy_balance.TotalEnergyBalanceEquations.primary_equation_name(),
                "interface_fourier_flux_equation",
                "interface_enthalpy_flux_equation",
                "well_enthalpy_flux_equation",
            ]
            if "injection_temperature_constraint" in self.equation_system.equations:
                eqs += ["injection_temperature_constraint"]

        # if (
        #     self.has_fluid_volume_variable
        #     and "local_fluid_volume_constraint" in self.equation_system.equations
        # ):
        #     vars += [self.specific_fluid_volume(subdomains)]
        #     eqs += ["local_fluid_volume_constraint"]
        # if (
        #     self.has_fluid_enthalpy_variable
        #     and "local_fluid_enthalpy_constraint" in self.equation_system.equations
        # ):
        #     vars += [self.specific_fluid_enthalpy(subdomains)]
        #     eqs += ["local_fluid_enthalpy_constraint"]

        chop = 0.4
        projection = self.equation_system.projection_to(
            self.equation_system._parse_variable_type(vars)  # type:ignore[arg-type]
        )
        pT = projection.transpose()

        for i in range(50):
            A, b = self.equation_system.assemble(
                evaluate_jacobian=True,
                equations=eqs,  # variables=vars
            )

            norm = np.linalg.norm(b)
            if norm < 1e-10:
                break
            elif norm < 1e-2:
                chop = 1

            A = self.params["linear_right_preconditioner"]([A])[0]
            A = A * pT

            dx = sps.linalg.spsolve(A, b) * chop
            dx_full = self._scale_back_state(pT @ dx, is_increment=True)
            self.equation_system.set_variable_values(
                dx_full,
                iterate_index=0,
                additive=True,
            )
            self.update_derived_quantities()


class AdjustedPointWellModel(ModelConfig):
    """Adjustment of a 2D model which has wells modelled as point grids.

    Two types of point grids are expected: ``'injection_well'`` and
    ``'production_well'``.

    In the injection well, mass is expected to enter the system (mass per time)
    at a given temperature (fixed value).

    At the production well, a given pressure value is required.

    In injection wells, the energy balance is replaced by a simple constraint
    ``T - T_injection = 0``.
    In production wells, the fluid mass balance (pressure equation) is replaced by
    ``p - p_production = 0``.

    In injection wells, a given inflow per fluid component is expected, which enter the
    system as a source term in the respective point grid.

    In production wells, all DOFs except pressure, and all equations are removed.
    The outflow of mass and energy can be computed with respective well fluxes.
    An exact composition of the fluid at the production well can be obtained from the
    values in the matrix grid, using the cell which was used to construct the mortar
    grid for the production well.

    In this sense, production wells and their respective grids are only used to mimic
    an internal, free-flow boundary.

    Important:
        This is a mixin modifying equations and variables. It must be mixed in
        above all other variable and equation mixins.

    Note:
        In injection wells, only the injected mass is defined, not the injected energy.
        This is due to the energy balance equation being replaced by an temperature
        constraint. This can cause trouble if there is temporarily some backflow in
        the production wells due to pressure drop around the wells. TODO

    """

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def _filter_wells(
        self,
        subdomains: Sequence[pp.Grid],
        well_type: Literal["production", "injection"],
    ) -> tuple[list[pp.Grid], list[pp.Grid]]:
        """Helper method to return the partitioning of subdomains into wells of defined
        ``well_type`` and other grids.

        Parameters:
            subdomains: A list of subdomains.
            well_type: Well type to filter out (injector or producer).

        Returns:
            A 2-tuple containing

            1. All 0D grids tagged as wells of type ``well_type``.
            2. All other grids found in ``subdomains``.

        """
        tag = f"{well_type}_well"
        wells = [sd for sd in subdomains if sd.dim == 0 and tag in sd.tags]
        other_sds = [sd for sd in subdomains if sd not in wells]
        return wells, other_sds

    # Adjusting PDEs
    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Introduced the usual fluid mass balance equations but only on grids which
        are not production wells.

        Important:
            This is a hack which removes production wells from the subdomains, having
            also an impact on the code in the outerscope.

        """
        prod_wells, no_prod_wells = self._filter_wells(subdomains, "production")
        sds_ = [sd for sd in subdomains]
        eq: pp.ad.Operator = super().mass_balance_equation(sds_)  # type:ignore[misc]
        name = eq.name
        eq.set_name(f"{name}_raw")
        projection = pp.ad.SubdomainProjections(sds_)
        eq_slice = projection.cell_restriction(no_prod_wells) @ eq
        eq_slice.set_name(name)

        for pw in prod_wells:
            subdomains.remove(pw)

        return eq_slice

        # name = eq.name
        # volume_stabilization = self.fluid.density(
        #     no_production_wells
        # ) * pp.ad.sum_operator_list(
        #     [
        #         phase.fraction(no_production_wells)
        #         / phase.density(no_production_wells)
        #         for phase in self.fluid.phases
        #     ],
        #     "fluid_specific_volume",
        # ) - self.porosity(no_production_wells)

        # volume_stabilization = self.volume_integral(
        #     volume_stabilization, no_production_wells, dim=1
        # )
        # volume_stabilization = pp.ad.time_derivatives.dt(
        #     volume_stabilization, self.ad_time_step
        # )
        # eq = eq + volume_stabilization
        # eq.set_name(name)
        # return eq

    def energy_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Introduced the usual fluid mass balance equations but only on grids which
        are not production wells."""
        inj_wells, no_inj_wells = self._filter_wells(subdomains, "injection")
        sds_ = [sd for sd in subdomains]
        eq: pp.ad.Operator = super().energy_balance_equation(sds_)  # type:ignore[misc]
        name = eq.name
        eq.set_name(f"{name}_raw")
        projection = pp.ad.SubdomainProjections(sds_)
        eq_slice = projection.cell_restriction(no_inj_wells) @ eq
        eq_slice.set_name(name)
        for iw in inj_wells:
            subdomains.remove(iw)
        return eq_slice

    # Introducing pressure and temperature constraint at production and injection.
    def set_equations(self):
        """Introduces pressure and temperature constraints on production and injection
        wells respectively."""
        super().set_equations()

        subdomains = self.mdg.subdomains()
        injection_wells, _ = self._filter_wells(subdomains, "injection")
        production_wells, _ = self._filter_wells(subdomains, "production")

        p_constraint = self.pressure_constraint_at_production_wells(production_wells)
        self.equation_system.set_equation(p_constraint, production_wells, {"cells": 1})
        if isinstance(self, pp.energy_balance.TotalEnergyBalanceEquations):
            T_constraint = self.temperature_constraint_at_injection_wells(
                injection_wells
            )
            self.equation_system.set_equation(
                T_constraint, injection_wells, {"cells": 1}
            )

    def pressure_constraint_at_production_wells(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Returns an constraint of form :math:`p - p_p=0` which replaces the
        pressure equation in production wells.

        Parameters:
            subdomains: A list of grids (tagged as production wells).

        Returns:
            The left-hand side of above equation.

        """
        p_production = pp.wrap_as_dense_ad_array(
            np.hstack(
                [
                    np.ones(sd.num_cells)
                    * self._p_PRODUCTION[sd.tags["production_well"]]
                    for sd in subdomains
                ]
            ),
            name="production_pressure",
        )

        pressure_constraint_production = self.pressure(subdomains) - p_production
        pressure_constraint_production.set_name("production_pressure_constraint")
        return pressure_constraint_production

    def temperature_constraint_at_injection_wells(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Analogous to :meth:`pressure_constraint_at_production_wells`, but for
        temperature at production wells."""
        T_injection = pp.wrap_as_dense_ad_array(
            np.hstack(
                [
                    np.ones(sd.num_cells) * self._T_INJECTION[sd.tags["injection_well"]]
                    for sd in subdomains
                ]
            ),
            name="injection_temperature",
        )

        temperature_constraint_injection = self.temperature(subdomains) - T_injection
        temperature_constraint_injection.set_name("injection_temperature_constraint")
        return temperature_constraint_injection

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Augments the source term in the pressure equation to account for the mass
        injected through injection wells."""
        source: pp.ad.Operator = super().fluid_source(subdomains)  # type:ignore[misc]

        injection_wells, _ = self._filter_wells(subdomains, "injection")
        # production_wells, _ = self._filter_wells(subdomains, "production")

        projection = pp.ad.SubdomainProjections(subdomains)

        injected_mass: pp.ad.Operator = pp.ad.sum_operator_list(
            [
                self.volume_integral(
                    self.injected_component_mass(comp, injection_wells),
                    injection_wells,
                    1,
                )
                for comp in self.fluid.components
            ],
            "total_injected_fluid_mass",
        )

        # source += projection.cell_restriction(subdomains) @ (
        #     projection.cell_prolongation(injection_wells) @ injected_mass
        # )
        # Adding total injected mass in injection wells.
        source += projection.cell_prolongation(injection_wells) @ injected_mass

        # Removing mass flowing out of the production wells.
        # source -= projection.cell_prolongation(production_wells) @ (
        #     projection.cell_restriction(production_wells) @ source
        # )

        return source

    def component_source(
        self, component: pp.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Adjusted source term for a component's mass balance equation to account
        for the injected mass in the injection wells, and removing all mass in the
        production wells."""
        source: pp.ad.Operator = super().component_source(component, subdomains)  # type:ignore[misc]

        injection_wells, _ = self._filter_wells(subdomains, "injection")
        production_wells, _ = self._filter_wells(subdomains, "production")

        projection = pp.ad.SubdomainProjections(subdomains)

        injected_mass = self.volume_integral(
            self.injected_component_mass(component, injection_wells),
            injection_wells,
            1,
        )

        # source += subdomain_projections.cell_restriction(subdomains) @ (
        #     subdomain_projections.cell_prolongation(injection_wells) @ injected_mass
        # )
        # Adding mass in injection wells
        source += projection.cell_prolongation(injection_wells) @ injected_mass

        # Removing source term in production well, mimicing outflow of mass.
        source -= projection.cell_prolongation(production_wells) @ (
            projection.cell_restriction(production_wells) @ source
        )

        return source

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Adjusted energy source term removing all energy in the production wells."""
        source = super().energy_source(subdomains)  # type:ignore[misc]

        projection = pp.ad.SubdomainProjections(subdomains)
        production_wells, _ = self._filter_wells(subdomains, "production")

        # Removing energy in production well.
        source -= projection.cell_prolongation(production_wells) @ (
            projection.cell_restriction(production_wells) @ source
        )
        return source

    def injected_component_mass(
        self, component: pp.Component, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Returns the injected mass of a fluid component in [kg m^-3 s^-1] (or moles).

        This is used as a source term on balance equations in injection wells. Note that
        the volume integral is not performed here, but in the respective method
        assembling the source term for a balance equation.

        Parameters:
            component: A fluid component.
            subdomains: A list of grids (grids tagged as ``'injection_wells'``)

        Returns:
            The source term wrapped as a dens AD array.
        """
        injected_mass: list[np.ndarray] = []
        for sd in subdomains:
            assert "injection_well" in sd.tags, (
                f"Grid {sd.id} not tagged as injection well."
            )
            injected_mass.append(
                np.ones(sd.num_cells)
                * self._INJECTED_MASS[component.name][sd.tags["injection_well"]]
            )

        if injected_mass:
            source = np.hstack(injected_mass)
        else:
            source = np.zeros((0,))

        return pp.ad.DenseArray(source, f"injected_mass_density_{component.name}")


class InitialConditions(ModelConfig):
    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        # f = lambda x: self._p_IN + x /10 * (self._p_OUT - self._p_IN)
        # vals = np.array(list(map(f, sd.cell_centers[0])))
        # return vals
        p = np.ones(sd.num_cells)
        if sd.dim == 0 and "production_well" in sd.tags:
            return p * self._p_PRODUCTION[sd.tags["production_well"]]
        else:
            return p * self._p_INIT

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        T = np.ones(sd.num_cells)
        if sd.dim == 0 and "injection_well" in sd.tags:
            return T * self._T_INJECTION[sd.tags["injection_well"]]
        else:
            return T * self._T_INIT

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        vals = np.ones(sd.num_cells)
        if self.fluid.num_components == 1:
            return vals
        else:
            return vals * self._z_INIT[component.name]


class BoundaryConditions(ModelConfig):
    """No flow BC, with the exception of a stripe on the bottom boundary where
    temperature Dirichlet-BC are given."""

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if sd.dim == self.nd and self._HEATED_BOUNDARY_ON:
            heated = self._heated_boundary_faces(sd)
            return pp.BoundaryCondition(sd, heated, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if sd.dim == self.nd and self._PRESSURE_BOUNDARY_ON:
            d = self._dirichlet_faces_pressure(sd)
            return pp.BoundaryCondition(sd, d, "dir")
        else:
            return pp.BoundaryCondition(sd)

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_type_equilibrium(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if sd.dim < self.nd:
            return pp.BoundaryCondition(sd)
        else:
            if cf.is_fractional_flow(self):
                return self.bc_type_fourier_flux(sd)
            else:
                return self.bc_type_darcy_flux(sd)

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Sets pressure on the dirichlet boundary (top faces) and on the heated
        boundary on the bottom faces.

        The top values are set using ``_p_BC``.
        The bottom values are approximated using domain height and an estimate that
        pressure increases by 1e6 Pa per hundred meters depth.
        This is added to ``_p_BC`` and set at the bottom boundary.

        """
        vals = np.zeros(boundary_grid.num_cells)
        sd = boundary_grid.parent

        if sd.dim == self.nd:
            sides = self.domain_boundary_sides(sd)
            heated_faces = self._heated_boundary_faces(sd)[sides.all_bf]
            vals[heated_faces] = self._p_BC + self._DOMAIN_DIMENSIONS[1] / 100.0 * 1e6
            dir_faces = self._dirichlet_faces_pressure(sd)[sides.all_bf]
            vals[dir_faces] = self._p_BC

        return vals

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Sets the boundary values for temperature on the heated bottom faces.

        Sets also T values on the pressure-dirichlet faces equal to ``_T_INIT``. This
        is require for the BC flash, which is required for the advection weights.

        """
        vals = np.zeros(boundary_grid.num_cells)
        sd = boundary_grid.parent

        if sd.dim == self.nd:
            sides = self.domain_boundary_sides(sd)
            heated_faces = self._heated_boundary_faces(sd)[sides.all_bf]
            vals[heated_faces] = self._T_BC
            dir_faces = self._dirichlet_faces_pressure(sd)[sides.all_bf]
            vals[dir_faces] = self._T_INIT

        return vals

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        """Sets BC for fractions on the heated and the dirichlet boundary in order for
        the boundary flash to work."""
        vals = np.zeros(boundary_grid.num_cells)
        sd = boundary_grid.parent

        if sd.dim == self.nd:
            sides = self.domain_boundary_sides(sd)
            heated_faces = self._heated_boundary_faces(sd)[sides.all_bf]
            dir_faces = self._dirichlet_faces_pressure(sd)[sides.all_bf]
            if self.fluid.num_components == 1:
                vals[heated_faces] = 1.0
                vals[dir_faces] = 1.0
            else:
                vals[heated_faces] = self._z_INIT[component.name]
                vals[dir_faces] = self._z_INIT[component.name]

        return vals


class Permeability(ModelConfig):
    """Custom permeability with a higher absolute permability around the wells and a
    constant permeability of 1 in the wells.

    Introduces also alternating conductive and blocking fractures, if any.

    """

    total_mass_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return pp.constitutive_laws.DimensionDependentPermeability.permeability(
            self,  # type:ignore[arg-type]
            subdomains,
        )

    def matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Matrix permeability with a higher permeability with factor 1e3 around the
        wells in the matrix."""

        assert len(subdomains) <= 1, "Expecting at most 1 grid as matrix."
        if len(subdomains) == 1:
            assert subdomains[0].dim == self.nd, "Expecting matrix grid."

        K_vals: list[np.ndarray] = [np.zeros((0,))]
        K_base = self.solid.permeability
        K_w = self._PERM_AROUND_WELLS

        for sd in subdomains:
            k = np.ones(sd.num_cells) * K_base
            l, r = self._central_stripe(sd)
            k[sd.cell_centers[0] < l] = K_w
            k[sd.cell_centers[0] > r] = K_w
            self.exporter.add_constant_data([(sd, "absolute_permeability", k)])
            K_vals.append(k)

        K_: pp.ad.Operator = pp.wrap_as_dense_ad_array(
            np.concatenate(K_vals), name="base_matrix_permeability"
        )

        if cf.is_fractional_flow(self):
            K_ *= self.total_mass_mobility(subdomains)

        K = self.isotropic_second_order_tensor(subdomains, K_)
        K.set_name("matrix_permeability")
        return K

    def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Set with model parameter 'fracture_permeability'."""
        N = sum([sd.num_cells for sd in subdomains])
        K_val = self.solid.permeability
        K_: pp.ad.Operator

        # Declare K_ first in case fracture set is empty.
        K_ = pp.wrap_as_dense_ad_array(
            float(K_val), size=N, name="base_fracture_permeability"
        )

        K_vals: list[np.ndarray] = [np.zeros((0,))]

        K_low = self._BLOCKING_FRACTURE_PERM
        K_high = self._CONDUCTIVE_FRACTURE_PERM

        is_impermable = False

        for sd in subdomains:
            k = np.ones(sd.num_cells)
            if is_impermable:
                k *= K_low
            else:
                k *= K_high
            is_impermable = not is_impermable
            self.exporter.add_constant_data([(sd, "absolute_permeability", k)])
            K_vals.append(k)

        K_ = pp.wrap_as_dense_ad_array(
            np.concatenate(K_vals), name="base_fracture_permeability"
        )

        if cf.is_fractional_flow(self):
            K_ *= self.total_mass_mobility(subdomains)

        K = self.isotropic_second_order_tensor(subdomains, K_)
        K.set_name("fracture_permeability")
        return K

    def intersection_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Base permeability of wells is 1."""
        K_vals: list[np.ndarray] = [np.zeros((0,))]

        for sd in subdomains:
            k = np.ones(sd.num_cells)
            self.exporter.add_constant_data([(sd, "absolute_permeability", k)])
            K_vals.append(k)

        K_: pp.ad.Operator = pp.wrap_as_dense_ad_array(
            np.concatenate(K_vals), name="base_well_permeability"
        )

        if cf.is_fractional_flow(self):
            K_ *= self.total_mass_mobility(subdomains)

        K = self.isotropic_second_order_tensor(subdomains, K_)
        K.set_name("well_permeability")
        return K


class DataCollectionMixin(pp.PorePyModel):
    """Collects data required for running the plot script."""

    has_fluid_volume_variable: bool

    def __init__(self, params: dict | None = None):
        super().__init__(params)  # type:ignore[safe-super]
        self._flash_iter_per_grid: dict[pp.Grid, list[np.ndarray]] = {}

    def data_to_export(self):
        data: list = super().data_to_export()

        for sd in self.mdg.subdomains():
            if sd in self._flash_iter_per_grid:
                n = cast(np.ndarray, sum(self._flash_iter_per_grid[sd]))
            else:
                n = np.zeros(sd.num_cells, dtype=int)

            data.append((sd, "flash iterations", n))
            data.append(
                (sd, "aperture", self.equation_system.evaluate(self.aperture([sd])))
            )
            if not isinstance(
                self, pp.energy_balance.VariablesEnergyBalance
            ) and hasattr(self, "temperature"):
                data.append(
                    (
                        sd,
                        "temperature",
                        self.equation_system.evaluate(self.temperature([sd])),
                    )
                )
            if not self.has_fluid_volume_variable:
                data.append(
                    (
                        sd,
                        "specific_fluid_volume",
                        self.equation_system.evaluate(self.fluid.specific_volume([sd])),
                    )
                )

        return data

    def before_nonlinear_loop(self) -> None:
        self._flash_iter_per_grid.clear()
        return super().before_nonlinear_loop()  # type:ignore[misc]

    def local_equilibrium(
        self,
        grid: pp.Grid,
        /,
        *,
        specification: Optional[cfle.StateSpecDict] = None,
        initial_guess_from_current_state: bool = True,
        update_secondary_quantities: bool = True,
        state: Optional[np.ndarray] = None,
        cell_mask: Optional[np.ndarray] = None,
    ) -> tuple[pf.FlashResults, NDArray[np.bool_]]:

        results: tuple[pf.FlashResults, NDArray[np.bool_]] = super().local_equilibrium(  # type:ignore[misc]
            grid,
            specification=specification,
            initial_guess_from_current_state=initial_guess_from_current_state,
            update_secondary_quantities=update_secondary_quantities,
            state=state,
            cell_mask=cell_mask,
        )

        if grid not in self._flash_iter_per_grid:
            self._flash_iter_per_grid[grid] = []
        self._flash_iter_per_grid[grid].append(results[0].num_iter)

        return results

    def plot_from_vec(
        self, vec: np.ndarray, name: str, is_update: bool, suffix: str = ""
    ) -> None:
        i = self.nonlinear_solver_statistics.num_iterations + 1
        xn = self.equation_system.get_variable_values(time_step_index=0)
        self.equation_system.set_variable_values(vec, time_step_index=0)
        plot_name = f"{name}_iter_{i}"
        title = f"{name}, iteration {i}"
        if suffix:
            plot_name += f"_{suffix}"
            title += f" ({suffix})"
        if is_update:
            plot_name = "delta_" + plot_name
            title = "Update " + title
        plot_name += ".png"

        pp.save_img(
            name=Path(f"{self.params['folder_name']}/{plot_name}"),
            grid=self.mdg,
            cell_value=name,
            fracturewidth_1d=10,
            pointsize=20,
            fig_size=(10, 5),
            plot_2d=True,
            linewidth=0.1,
            title=title,
        )

        self.equation_system.set_variable_values(xn, time_step_index=0)


class ColdInjectionMixins(
    DataCollectionMixin,
    Permeability,
    AdjustedPointWellModel,
    FluidMixture,
    InitialConditions,
    BoundaryConditions,
    SolutionStrategy,
):
    """Collection of used mixins in this example."""


class BuoyancyModel(pp.PorePyModel):
    def initial_condition(self):
        super().initial_condition()
        self.set_buoyancy_discretization_parameters()

    def update_flux_values(self):
        super().update_flux_values()
        self.update_buoyancy_driven_fluxes()

    def set_nonlinear_discretizations(self):
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        # g_constant = pp.GRAVITY_ACCELERATION
        # val = self.units.convert_units(g_constant, "m*s^-2")
        # size = np.sum([g.num_cells for g in subdomains]).astype(int)
        # gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
        # gravity_field.set_name("gravity_field")
        # return gravity_field
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2")
        gravity_field = pp.ad.Scalar(val)
        gravity_field.set_name("gravity_field")
        return gravity_field


class NoFluxRediscretization:
    def add_nonlinear_darcy_flux_discretization(self) -> None:
        """If the fractional flow formulation is used, the nonlinear Darcy flux
        discretization is added by default for all subdomains to the update routine."""
        return

    def add_nonlinear_fourier_flux_discretization(self) -> None:
        """Compositional flow models relay on re-discretization of the
        Fourier flux, since the thermal conductivity is presumably a nonlinear fluid
        property.

        The discretization is added by default for all subdomains to the update routine.

        """
        return


class QuadraticRelPerm(pp.PorePyModel):
    """ "Contains the quadratic relative permeability law."""

    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Quadratic relative permeability model."""
        return phase.saturation(domains) ** pp.ad.Scalar(2)
