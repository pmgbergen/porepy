"""Module containing some constant heuristic fluid property implementations and
the mixin :class:`FluidMobility`, which is required in all flow & transport problems.

Most of the laws implemented here are meant for 1-phase, 1-component mixtures, using
some fluid component stored in the fluid's reference component and have an analytical
expression which can be handled by PorePy's AD framework.

Note:
    In order to override default implementations of fluid properties in
    :class:`~porepy.compositional.compositional_mixins.FluidMixin`, the classes hererin
    must be mixed into the model *before* the fluid mixin.

    .. code::python

        class MyModel(
            # ...
            ConstitutiveLawFromFluidLibrary,
            # ...
            FluidMixin,
            # ...
        )

    E.g., Python must find :meth:`FluidDensityFromPressure.density_of_phase` before
    it finds the default
    :meth:`porepy.compositional.compositional_mixins.FluidMixin.density_of_phase`.

"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Callable, List, Literal, Sequence, Union, cast

import numpy as np
import scipy.sparse as sps

import porepy as pp

__all__ = [
    "FluidDensityFromPressure",
    "FluidDensityFromTemperature",
    "FluidDensityFromPressureAndTemperature",
    "FluidMobility",
    "FluidBuoyancy",
    "ConstantViscosity",
    "ConstantFluidThermalConductivity",
    "FluidEnthalpyFromTemperature",
]

Scalar = pp.ad.Scalar

if TYPE_CHECKING:
    ExtendedDomainFunctionType = pp.ExtendedDomainFunctionType


def _single_point_upwind_matrices(
    sd: pp.Grid,
    flux_array: np.ndarray,
    bc: pp.BoundaryCondition,
    num_components: int,
) -> tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix]:
    """Single-point upstream-weighting matrices for one signed flux array.

    Faithful extraction of :meth:`porepy.numerics.fv.upwind.Upwind.discretize` so it can
    be reused for *two* directions in one discretization. Returns
    ``(upwind, bound_transport_dir, bound_transport_neu)``.
    """
    if sd.dim == 0:
        return (
            sps.csr_matrix((0, num_components)),
            sps.csr_matrix((0, 0)),
            sps.csr_matrix((0, 0)),
        )
    sign_flux = np.sign(flux_array)
    pos_flux = sign_flux >= 0
    neg_flux = np.logical_not(pos_flux)

    cf_dense = sd.cell_faces_as_dense()
    upstream_cell_ind = np.zeros(sd.num_faces, dtype=int)
    upstream_cell_ind[pos_flux] = cf_dense[0, pos_flux]
    upstream_cell_ind[neg_flux] = cf_dense[1, neg_flux]

    row = np.arange(sd.num_faces)
    values = np.ones(sd.num_faces, dtype=int)

    neumann_ind = np.where(bc.is_neu)[0]
    inflow_ind = np.where(
        np.logical_and(
            bc.is_dir,
            np.logical_or(
                np.logical_and(pos_flux, cf_dense[0] < 0),
                np.logical_and(neg_flux, cf_dense[1] < 0),
            ),
        )
    )[0]
    delete_ind = np.sort(np.r_[neumann_ind, inflow_ind])
    row = np.delete(row, delete_ind)
    values = np.delete(values, delete_ind)
    col = np.delete(upstream_cell_ind, delete_ind)

    upstream_mat = sps.coo_matrix(
        (values, (row, col)), shape=(sd.num_faces, sd.num_cells)
    ).tocsr()
    upwind = sps.kron(upstream_mat, sps.eye(num_components)).tocsr()

    sgn_div = np.asarray(sd.divergence(dim=1).sum(axis=0)).squeeze()
    bc_discr_neu = sps.coo_matrix(
        (sgn_div[neumann_ind], (neumann_ind, neumann_ind)),
        shape=(sd.num_faces, sd.num_faces),
    ).tocsr()
    bc_discr_dir = sps.coo_matrix(
        (np.ones(inflow_ind.size), (inflow_ind, inflow_ind)),
        shape=(sd.num_faces, sd.num_faces),
    ).tocsr()
    rhs_neu = sps.kron(bc_discr_neu, sps.eye(num_components)).tocsr()
    rhs_dir = sps.kron(bc_discr_dir, sps.eye(num_components)).tocsr()
    return upwind, rhs_dir, rhs_neu


class HUpwind(pp.Upwind):
    """Hybrid-upwinding discretization: two opposite upwind matrices from one direction.

    Hybrid upwinding upwinds the two phases of a pair along a *single* inter-phase
    gravity direction ``w_flux`` (heavier phase one way, lighter phase the other). Rather
    than maintaining two :class:`~porepy.numerics.fv.upwind.Upwind` objects with
    sign-flipped flux arrays, this discretization stores that *one* direction and, in
    :meth:`discretize`, builds the two opposite single-point upwind matrices (plus their
    boundary matrices):

    - ``upwind_gamma`` / ``bound_transport_{dir,neu}_gamma`` -- upstream by ``+w_flux``;
    - ``upwind_delta`` / ``bound_transport_{dir,neu}_delta`` -- upstream by ``-w_flux``.

    The matrix keys are exposed as AD methods by :func:`wrap_discretization` (see
    :class:`HUpwindAd`).
    """

    def __init__(self, keyword: str = "hybrid_upwind") -> None:
        super().__init__(keyword)
        # gamma reuses the base Upwind keys; delta gets its own.
        self.upwind_matrix_key = "transport_gamma"
        self.bound_transport_dir_matrix_key = "rhs_dir_gamma"
        self.bound_transport_neu_matrix_key = "rhs_neu_gamma"
        self.upwind_gamma_matrix_key = "transport_gamma"
        self.bound_transport_dir_gamma_matrix_key = "rhs_dir_gamma"
        self.bound_transport_neu_gamma_matrix_key = "rhs_neu_gamma"
        self.upwind_delta_matrix_key = "transport_delta"
        self.bound_transport_dir_delta_matrix_key = "rhs_dir_delta"
        self.bound_transport_neu_delta_matrix_key = "rhs_neu_delta"
        self._flux_array_key = "hybrid_gravity_flux"

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        num_components: int = parameter_dictionary.get("num_components", 1)

        if "bc" in parameter_dictionary:
            bc = parameter_dictionary["bc"]
        else:
            bc = pp.BoundaryCondition(sd, sd.get_boundary_faces(), "dir")

        direction = np.asarray(parameter_dictionary[self._flux_array_key])
        # gamma: upstream by +direction; delta: by -direction (the single stored gravity
        # flux flipped, so both matrices come from one direction).
        up_g, dir_g, neu_g = _single_point_upwind_matrices(
            sd, +direction, bc, num_components
        )
        up_d, dir_d, neu_d = _single_point_upwind_matrices(
            sd, -direction, bc, num_components
        )
        matrix_dictionary["transport_gamma"] = up_g
        matrix_dictionary["rhs_dir_gamma"] = dir_g
        matrix_dictionary["rhs_neu_gamma"] = neu_g
        matrix_dictionary["transport_delta"] = up_d
        matrix_dictionary["rhs_dir_delta"] = dir_d
        matrix_dictionary["rhs_neu_delta"] = neu_d


class HUpwindAd(pp.ad.Discretization):
    """AD wrapper for :class:`HUpwind`.

    Exposes the per-phase upwind / boundary-transport matrices of the one-direction
    hybrid discretization as :class:`~porepy.numerics.ad.operators.MergedOperator`
    factories: ``upwind_gamma()`` / ``upwind_delta()`` and the boundary variants.
    """

    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._discretization = HUpwind(keyword)
        self._name = "HUpwind"
        self.keyword = keyword

        self.upwind_gamma: Callable[[], pp.ad.MergedOperator]
        self.upwind_delta: Callable[[], pp.ad.MergedOperator]
        self.bound_transport_dir_gamma: Callable[[], pp.ad.MergedOperator]
        self.bound_transport_neu_gamma: Callable[[], pp.ad.MergedOperator]
        self.bound_transport_dir_delta: Callable[[], pp.ad.MergedOperator]
        self.bound_transport_neu_delta: Callable[[], pp.ad.MergedOperator]
        pp.ad.wrap_discretization(self, self._discretization, subdomains=subdomains)


class HUpwindCoupling(pp.UpwindCoupling):
    """Interface (mortar) counterpart of :class:`HUpwind`.

    From the single inter-phase gravity flux on the interface, builds the two opposite
    sets of mortar upwind matrices -- ``upwind_{primary,secondary}_gamma`` (from
    ``+direction``) and ``upwind_{primary,secondary}_delta`` (from ``-direction``), plus
    the per-direction signed ``flux`` -- in one :meth:`discretize`. The geometric
    ``trace`` / ``inv_trace`` / ``mortar_discr`` matrices are built once and shared.
    """

    def __init__(self, keyword: str) -> None:
        super().__init__(keyword)
        # Geometric / shared matrices keep the base keys.
        self.trace_primary_matrix_key = "trace"
        self.inv_trace_primary_matrix_key = "inv_trace"
        self.mortar_discr_matrix_key = "mortar_discr"
        # gamma reuses the base direction-dependent keys; delta gets its own.
        self.upwind_primary_matrix_key = "upwind_primary_gamma"
        self.upwind_secondary_matrix_key = "upwind_secondary_gamma"
        self.flux_matrix_key = "flux_gamma"
        self.upwind_primary_gamma_matrix_key = "upwind_primary_gamma"
        self.upwind_secondary_gamma_matrix_key = "upwind_secondary_gamma"
        self.flux_gamma_matrix_key = "flux_gamma"
        self.upwind_primary_delta_matrix_key = "upwind_primary_delta"
        self.upwind_secondary_delta_matrix_key = "upwind_secondary_delta"
        self.flux_delta_matrix_key = "flux_delta"
        self._flux_array_key = "hybrid_gravity_flux"

    def discretize(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: dict,
        data_secondary: dict,
        data_intf: dict,
    ) -> None:
        if sd_primary.dim - sd_secondary.dim not in [1, 2]:
            raise ValueError(
                "Implementation is only valid for grids one dimension apart."
            )
        matrix_dictionary = data_intf[pp.DISCRETIZATION_MATRICES][self.keyword]
        lam_flux = np.sign(
            data_intf[pp.PARAMETERS][self.keyword][self._flux_array_key]
        )

        inv_trace_h = np.abs(sd_primary.divergence(dim=1))
        matrix_dictionary["inv_trace"] = inv_trace_h
        matrix_dictionary["trace"] = inv_trace_h.T
        matrix_dictionary["mortar_discr"] = sps.eye(intf.num_cells)

        # gamma rides +direction, delta rides -direction (one stored gravity flux).
        for suffix, sign in (("gamma", 1.0), ("delta", -1.0)):
            lf = sign * lam_flux
            flag = (lf > 0).astype(float)
            matrix_dictionary[f"upwind_primary_{suffix}"] = sps.diags(flag)
            matrix_dictionary[f"upwind_secondary_{suffix}"] = sps.diags(1.0 - flag)
            matrix_dictionary[f"flux_{suffix}"] = sps.diags(lf)


class HUpwindCouplingAd(pp.ad.Discretization):
    """AD wrapper for :class:`HUpwindCoupling`."""

    def __init__(self, keyword: str, interfaces: list[pp.MortarGrid]) -> None:
        self.interfaces = interfaces
        self._discretization = HUpwindCoupling(keyword)
        self._name = "HUpwind coupling"
        self.keyword = keyword

        self.upwind_primary_gamma: Callable[[], pp.ad.MergedOperator]
        self.upwind_secondary_gamma: Callable[[], pp.ad.MergedOperator]
        self.upwind_primary_delta: Callable[[], pp.ad.MergedOperator]
        self.upwind_secondary_delta: Callable[[], pp.ad.MergedOperator]
        self.flux_gamma: Callable[[], pp.ad.MergedOperator]
        self.flux_delta: Callable[[], pp.ad.MergedOperator]
        self.trace: Callable[[], pp.ad.MergedOperator]
        self.inv_trace: Callable[[], pp.ad.MergedOperator]
        self.mortar_discr: Callable[[], pp.ad.MergedOperator]
        pp.ad.wrap_discretization(self, self._discretization, interfaces=interfaces)


class FluidDensityFromPressure(pp.PorePyModel):
    """Fluid density as a function of pressure for a single-phase, single-component
    fluid."""

    def fluid_compressibility(self, subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
        """Constant compressibility [Pa^-1] taken from the reference component of the
        fluid.

        Parameters:
            subdomains: List of subdomain grids. Not used in this implementation, but
                included for compatibility with other implementations.

        Returns:
            The fluid constant wrapped as an AD Scalar.

        """
        return Scalar(
            self.fluid.reference_component.compressibility, "fluid_compressibility"
        )

    def density_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a density exponential law for the fluid's phase.

        .. math::
            \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right) \\right]

        The reference density and the compressibility are taken from the material
        constants of the reference component, while the reference pressure is accessible
        by mixin; a typical implementation will provide this in a variable class.

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

        """

        def rho(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            rho_ref = Scalar(
                self.fluid.reference_component.density, "reference_fluid_density"
            )
            rho_ = rho_ref * self.pressure_exponential(cast(list[pp.Grid], domains))
            rho_.set_name("fluid_density")
            return rho_

        return rho

    def pressure_exponential(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Exponential term in the fluid density as a function of pressure.

        Extracted as a separate method to allow for easier combination with temperature
        dependent fluid density.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Exponential term in the fluid density as a function of pressure.

        """
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")

        # Reference variables are defined in a variables class which is assumed to be
        # available by mixin.
        dp = self.perturbation_from_reference("pressure", subdomains)

        # Wrap compressibility from fluid class as matrix (left multiplication with dp).
        c = self.fluid_compressibility(subdomains)
        return exp(c * dp)


class FluidDensityFromTemperature(pp.PorePyModel):
    """Fluid density as a function of temperature for a single-phase, single-component
    fluid."""

    def fluid_thermal_expansion(self, subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
        """Constant thermal expansion [K^-1] taken from the reference component of the
        fluid.

        Parameters:
            subdomains: List of subdomains. Not used, but included for consistency with
                other implementations.

        Returns:
            The constant wrapped in as an AD scalar.

        """
        val = self.fluid.reference_component.thermal_expansion
        return Scalar(val, "fluid_thermal_expansion")

    def density_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Analogous to :meth:`FluidDensityFromPressure.density_of_phase`, but using
        temperature and the thermal expansion of the reference component.

        .. math::
            \\rho = \\rho_0 \\exp \\left[ - c_T \\left(T - T_0\\right) \\right]

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

        """

        def rho(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            rho_ref = Scalar(
                self.fluid.reference_component.density, "reference_fluid_density"
            )
            rho_ = rho_ref * self.temperature_exponential(cast(list[pp.Grid], domains))
            rho_.set_name("fluid_density")
            return rho_

        return rho

    def temperature_exponential(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Exponential term in the fluid density as a function of temperature.

        Extracted as a separate method to allow for easier combination with temperature
        dependent fluid density.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Exponential term in the fluid density as a function of pressure.

        """
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")

        # Reference variables are defined in a variables class which is assumed to be
        # available by mixin.
        dtemp = self.perturbation_from_reference("temperature", subdomains)
        c = self.fluid_thermal_expansion(subdomains)
        return exp(Scalar(-1) * c * dtemp)


class FluidDensityFromPressureAndTemperature(
    FluidDensityFromPressure, FluidDensityFromTemperature
):
    """Fluid density which is a function of pressure and temperature, for a single-phase
    single-component fluid."""

    def density_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Returns a combination of the laws in the parent class methods:

        .. math::
            \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right)
            - c_T\\left(T - T_0\\right) \\right]

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

        """

        def rho(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            rho_ref = Scalar(
                self.fluid.reference_component.density, "reference_fluid_density"
            )

            rho_ = (
                rho_ref
                * self.pressure_exponential(cast(list[pp.Grid], domains))
                * self.temperature_exponential(cast(list[pp.Grid], domains))
            )
            rho_.set_name("fluid_density_from_pressure_and_temperature")
            return rho_

        return rho


class FluidMobility(pp.PorePyModel):
    """Class for fluid mobility and its discretization in flow & transport equations."""

    relative_permeability: Callable[
        [pp.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """Provided by some mixin dealing with the porous medium (work in progress).

    Only relevant in the multi-phase case.

    """

    mobility_keyword: str
    """Keyword for the discretization of the mobility. Normally provided by a mixin of
    instance :class:`~porepy.models.SolutionStrategy`.

    """

    def mobility_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.UpwindAd:
        r"""Discretization of the fluid mobility.

        This includes any non-linear, scalar expression :math:`a` in front of the
        advective flux :math:`q`.

        .. math::

            -\nabla \cdot \left(a q\right).

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Discretization of the fluid mobility.

        """
        return pp.ad.UpwindAd(self.mobility_keyword, subdomains)

    def interface_mobility_discretization(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.UpwindCouplingAd:
        """Discretization of the interface mobility.

        As for :meth:`mobility_discretization`, this involves any advection weight.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Discretization for the interface mobility.

        """
        return pp.ad.UpwindCouplingAd(self.mobility_keyword, interfaces)

    def total_mass_mobility(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        r"""Total mass mobility of the fluid mixture is given by

        .. math::

                \sum_j \frac{\rho_j k_r(s_j)}{\mu_j}.

        Used as a non-linear part of the diffusive tensor in the (total) mass balance
        equation.

        Note:
            In the single-phase, single-component case, this is reduced to
            :math:`\frac{\rho}{\mu}`.

        Parameters:
            domains: A list of subdomains or boundary grids.

        Returns:
            Above expression in operator form.

        """
        name = "total_mass_mobility"
        mobility = pp.ad.sum_operator_list(
            [
                phase.density(domains) * self.phase_mobility(phase, domains)
                for phase in self.fluid.phases
            ],
            name,
        )
        return mobility

    def phase_mobility(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Returns the mobility of a phase :math:`j`

        .. math::

            \frac{k_r(s_j)}{\mu_j}.

        Notes:
            For the single-phase case it returns simply :math:`\frac{1}{\mu}`.

        Important:
            Contrary to all other mobility methods implemented here, this one does not
            contain any mass term, it is a volumetric term. This is the term commonly
            denoted 'mobility in the literature.

        Parameters:
            phase: A phase in the fluid mixture.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            Above expression in operator form.

        """
        # Distinguish between single-phase case and multi-phase case: Usage of rel-perm
        # makes this class compatible with single-phase models, without requiring some
        # rel-perm mixin.
        if self.fluid.num_phases > 1:
            mobility = self.relative_permeability(phase, domains) / phase.viscosity(
                domains
            )
        else:
            assert phase == self.fluid.reference_phase
            mobility = phase.viscosity(domains) ** pp.ad.Scalar(-1.0)
        mobility.set_name(f"phase_mobility_{phase.name}")
        return mobility

    def component_mass_mobility(
        self, component: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Non-linear term in the advective flux in a component mass balance equation.

        It is obtained by summing :meth:`phase_mobility` weighed with
        :attr:`~porepy.compositional.base.Phase.partial_fraction_of` the component,
        and the phase :attr:`~porepy.compositional.base.Phase.density`,
        if the component is present in the phase.

        .. math::

                \sum_j x_{n, ij} \rho_j \frac{k_r(s_j)}{\mu_j},

        Note:
            In the single-phase, single-component case, this is reduced to
            :math:`\frac{\rho}{\mu}`.

        Parameters:
            component: A component in the fluid mixture.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            Above expression in operator form.

        """
        if self.fluid.num_phases > 1 or self.fluid.num_components > 1:
            # NOTE: This method is kept as general as possible when typing the
            # signature. But the default fluid of the PorePyModel consists of
            # FluidComponent, not Component. Adding type:ignore for this reason.
            mobility = pp.ad.sum_operator_list(
                [
                    phase.partial_fraction_of[component](domains)
                    * phase.density(domains)
                    * self.phase_mobility(phase, domains)
                    for phase in self.fluid.phases
                    if component in phase  # type:ignore[operator]
                ],
            )
        # This branch is for compatibility with single-phase or single component
        # models, which do not have the complete notion of fractions.
        else:
            assert component == self.fluid.reference_component
            mobility = self.fluid.reference_phase.density(
                domains
            ) * self.phase_mobility(self.fluid.reference_phase, domains)

        mobility.set_name(f"component_mass_mobility_{component.name}")
        return mobility

    def fractional_component_mass_mobility(
        self, component: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Returns the :meth:`component_mass_mobility` divided by the
        :meth:`total_mass_mobility` for a component :math:`\eta`.

        To be used in component mass balance equations in a fractional flow model, where
        the total mobility is part of the non-linear diffusive tensor in the Darcy flux.

        .. math::

            - \nabla \cdot \left(f_{\eta} D(x) \nabla p\right),

        where the tensor :math:`D(x)` contains the total mobility.

        Parameters:
            component: A component in the fluid mixture.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            The term :math:`f_{\eta}` in above expession in operator form.

        """
        frac_mob = self.component_mass_mobility(
            component, domains
        ) / self.total_mass_mobility(domains)
        frac_mob.set_name(f"fractional_component_mass_mobility_{component.name}")
        return frac_mob

    def fractional_phase_mass_mobility(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        r"""Returns the product of the ``phase`` density and :meth:`phase_mobility`
        divided by the :meth:`total_mass_mobility`.

        To be used in balance equations in a fractional flow model, where the total
        mobility is part of the non-linear diffusive tensor in the Darcy flux.

        I.e. for a phase :math:`\gamma`

        .. math::

            - \nabla \cdot \left(f_{\gamma} D(x) \nabla p\right),

        assuming the tensor :math:`D(x)` contains the total mobility.

        Parameters:
            phase: A phase in the fluid mixture.
            domains: A sequence of subdomains or boundary grids.

        Returns:
            The term :math:`f_{\gamma}` in above expession in operator form.

        """
        frac_mob = (
            phase.density(domains)
            * self.phase_mobility(phase, domains)
            / self.total_mass_mobility(domains)
        )
        frac_mob.set_name(f"fractional_phase_mass_mobility_{phase.name}")
        return frac_mob


class FluidBuoyancy(pp.PorePyModel):
    """
    Buoyancy terms and discretizations for multiphase, multicomponent flow.

    This class is based on the fixed-dimensional hybrid upwinding scheme presented in
    Bosma et al. (2022), "Smooth implicit hybrid upwinding for compositional multiphase
    flow in porous media" (CMA, 388, 114288). Here, we implement an alternate version of
    the scheme using a total mass formulation.

    The main difference in this implementation is the consistent treatment of the
    gravity term, following Starnoni et al. (2019),
    "Consistent MPFA Discretization for Flow in the Presence of Gravity"
    (WRR, 55(12), 10105–10118).

    This implementation is novel in three main aspects: non-isothermal compositional
    multiphase flow, mixed-dimensional formulation,
    and the fractional flow form of the equations.

    Mass, component mass, and energy conservation are tested in `test_buoyancy_flow.py`,
    and a benchmark against an analytical buoyancy solution is provided in
    `test_buoyancy_flow_benchmark.py`.
    """

    component_mass_mobility: Callable[
        [pp.Component, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """See :class:`FluidMobility`."""

    fractional_phase_mass_mobility: Callable[
        [pp.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    """See :class:`FluidMobility`."""

    phase_mobility: Callable[[pp.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`FluidMobility`."""

    mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """See :class:`FluidMobility`."""

    bc_type_fluid_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """See :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`.
    """

    darcy_flux_discretization: Callable[
        [list[pp.Grid]], pp.ad.MpfaAd
    ]  # because it contains the div(w(rho)) term
    """See :class:`~porepy.models.constitutive_laws.DarcysLaw`."""

    normal_permeability: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """See :class:`~porepy.models.constitutive_laws.ConstantPermeability`."""

    _nonlinear_discretizations: list[pp.ad.MergedOperator]
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    def buoyancy_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Key for subdomain buoyancy between phases gamma and delta.

        Parameters:
            gamma: The first phase.
            delta: The second phase.

        Returns:
            A unique key for the buoyancy term between the two phases. Can be used on
                subdomains.

        See also:
            :meth:`~porepy.models.fluid_property_library.FluidBuoyancy.buoyancy_intf_key`

        """
        return "buoyancy_" + gamma.name + "_" + delta.name

    def buoyant_flux_array_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Key for stored buoyant flux array on subdomains.

        Parameters:
            gamma: The first phase.
            delta: The second phase.

        Returns:
            A unique key for the array of buoyancy flux between the two phases on
                subdomains.

        See also:
            :meth:`~porepy.models.fluid_property_library.FluidBuoyancy.buoyant_intf_flux_array_key`

        """
        return "buoyant_flux_" + gamma.name + "_" + delta.name

    def buoyancy_intf_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Key for interface buoyancy between phases gamma and delta.

        Parameters:
            gamma: The first phase.
            delta: The second phase.

        Returns:
            A unique key for the buoyancy term between the two phases. Can be used on
                subdomains.

        See also:
            :meth:`~porepy.models.fluid_property_library.FluidBuoyancy.buoyancy_key`

        """
        return "buoyancy_intf_" + gamma.name + "_" + delta.name

    def buoyant_intf_flux_array_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Key for stored buoyant flux array on interfaces.

        Parameters:
            gamma: The first phase.
            delta: The second phase.

        Returns:
            A unique key for the array of buoyancy flux between the two phases on
                interfaces.

        See also:
            :meth:`~porepy.models.fluid_property_library.FluidBuoyancy.buoyant_flux_array_key`

        """
        return "buoyant_intf_flux_" + gamma.name + "_" + delta.name

    def buoyancy_discretization(
        self, gamma: pp.Phase, delta: pp.Phase, subdomains: list[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Return upwind discretization for subdomain buoyancy term gamma↔delta.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            subdomains: The subdomains to consider for the discretization.

        Returns:
            An Upwind discretization for the buoyancy term between the two phases.

        """
        discr = pp.ad.UpwindAd(self.buoyancy_key(gamma, delta), subdomains)
        assert isinstance(discr._discretization, pp.Upwind)
        discr._discretization.flux_array_key = self.buoyant_flux_array_key(gamma, delta)
        return discr

    def hybrid_upwind_key(self, gamma: pp.Phase, delta: pp.Phase) -> str:
        """Discretization/parameter keyword for the pair's hybrid upwind."""
        return "hybrid_upwind_" + gamma.name + "_" + delta.name

    def hybrid_upwind_discretization(
        self, gamma: pp.Phase, delta: pp.Phase, subdomains: list[pp.Grid]
    ) -> HUpwindAd:
        """Return the one-direction hybrid-upwind discretization for the pair.

        A single :class:`HUpwind` whose stored ``"hybrid_gravity_flux"`` direction (the
        gamma->delta inter-phase gravity flux) yields both ``upwind_gamma`` (``+``) and
        ``upwind_delta`` (``-``), replacing the previous pair of sign-flipped
        :class:`~porepy.numerics.ad.discretizations.UpwindAd` objects.

        Parameters:
            gamma: The first (reference) phase.
            delta: The second phase.
            subdomains: The subdomains to consider.

        Returns:
            The hybrid-upwind AD discretization.

        """
        return HUpwindAd(self.hybrid_upwind_key(gamma, delta), subdomains)

    def hybrid_interface_upwind_discretization(
        self, gamma: pp.Phase, delta: pp.Phase, interfaces: list[pp.MortarGrid]
    ) -> HUpwindCouplingAd:
        """Return the one-direction hybrid interface (mortar) upwind for the pair.

        A single :class:`HUpwindCoupling` whose stored interface gravity direction yields
        both phases' opposite mortar upwind matrices, replacing the pair of sign-flipped
        :class:`~porepy.numerics.ad.discretizations.UpwindCouplingAd` objects.
        """
        return HUpwindCouplingAd(self.hybrid_upwind_key(gamma, delta), interfaces)

    def interface_buoyancy_discretization(
        self, gamma: pp.Phase, delta: pp.Phase, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.UpwindCouplingAd:
        """Return upwind discretization for interface buoyancy term gamma-delta.

        Parameters:
            gamma: The first phase.
            delta: The second phase.
            interfaces: The interfaces to consider for the discretization.

        Returns:
            An Upwind discretization for the buoyancy term between the two phases.

        """
        discr = pp.ad.UpwindCouplingAd(self.buoyancy_intf_key(gamma, delta), interfaces)
        assert isinstance(discr._discretization, pp.UpwindCoupling)
        discr._discretization.flux_array_key = self.buoyant_intf_flux_array_key(
            gamma, delta
        )
        return discr

    def fractionally_weighted_density(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Compute the fractional-flow-weighted density.

        The method computes the sum_j f_j rho_j where f_j is the fractional flow
        function for phase j.

        Parameters:
            domains: The domains to consider for the density computation.

        Returns:
            An operator representing the fractional-flow-weighted density.

        """
        overall_rho = pp.ad.sum_operator_list(
            [
                self.fractional_phase_mass_mobility(phase, domains)
                * phase.density(domains)
                for phase in self.fluid.phases
            ]
        )
        overall_rho.set_name("fractionally_weighted_density")
        return overall_rho

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Return gravity magnitude.

        Parameters:
            subdomains: The subdomains to consider for the gravity field computation.
                Not used, but included for consistency.

        Returns:
            An operator representing the gravity field.

        """
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2")
        gravity_field = pp.ad.Scalar(val)
        gravity_field.set_name("gravity_field")
        return gravity_field

    def gravity_force(
        self,
        subdomains: Union[list[pp.Grid], list[pp.MortarGrid]],
        material: Literal["fluid", "solid", "bulk"],
    ) -> pp.ad.Operator:
        """Return gravity force term (fluid only if buoyancy enabled).

        Depending on the material type, the gravity force term is computed differently.
        If `material` is "fluid" and gravitational effects are not explicitly disabled
        (self.params["enable_buoyancy_effects"] is set to False), the method calculates
        the product of the fractionally weighted density with the gravity field, as a
        vector pointing in the negative third direction. If material is "solid" or
        "bulk", or gravitational effects are disabled, the method passes the calculation
        to a equally named super class.

        Parameters:
            subdomains: The domains to consider for the gravity force computation.
            material: The material for which to compute the gravity force.

        Raises:
            TypeError: If subdomains are instances of pp.MortarGrid.

        Returns:
            An Ad operator representing the gravitational force.

        """
        if material == "fluid" and self.params.get("enable_buoyancy_effects", True):
            # Narrow to list[pp.Grid] for calls needing subdomain grids, which is the
            # intended usage of this method. The listing of list[pp.MortarGrid] in the
            # method annotation is used for compatibility with equally named methods in
            # other parts of the code.
            if not all(isinstance(g, pp.Grid) for g in subdomains):
                raise TypeError(
                    "gravity_force expects only subdomain grids for "
                    "buoyancy computation."
                )
            subdomains_list = cast(list[pp.Grid], list(subdomains))

            fractionally_weighted_rho = self.fractionally_weighted_density(
                subdomains_list
            )
            e_n = self.e_i(subdomains, i=self.nd - 1, dim=self.nd)
            overall_gravity_flux = (
                pp.ad.Scalar(-1)
                * e_n
                @ (fractionally_weighted_rho * self.gravity_field(subdomains_list))
            )
            overall_gravity_flux.set_name("overall gravity flux")
            return overall_gravity_flux
        else:
            return super().gravity_force(subdomains, material)  # type:ignore

    def density_driven_flux(
        self, subdomains: pp.SubdomainsOrBoundaries, density_metric: pp.ad.Operator
    ) -> pp.ad.Operator:
        """Compute flux induced by density_metric * g along gravity direction.

        The density metric will be defined elsewhere, it can for instance be a measure
        of density difference between two phases.

        The gravitational flux is discretized by the vector source scheme of the Darcy
        flux discretization.

        Parameters:
            subdomains: The subdomains to consider for the density-driven flux
                computation.
            density_metric: The density metric to use in the flux computation.

        Raises:
            TypeError: If subdomains are not instances of pp.Grid.

        Returns:
            An Ad operator representing the density-driven flux.

        """
        if not all(isinstance(g, pp.Grid) for g in subdomains):
            raise TypeError("density_driven_flux expects only subdomain grids.")
        subdomains_list = cast(list[pp.Grid], list(subdomains))

        e_n = self.e_i(subdomains_list, i=self.nd - 1, dim=self.nd)
        gravity_flux = (
            pp.ad.Scalar(-1)
            * e_n
            @ (density_metric * self.gravity_field(subdomains_list))
        )

        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains_list
        )

        w_flux = discr.vector_source() @ gravity_flux
        w_flux.set_name("density_driven_flux_" + density_metric.name)
        return w_flux

    def interface_density_driven_flux(
        self, interfaces: list[pp.MortarGrid], density_metric: pp.ad.Operator
    ) -> pp.ad.Operator:
        """Compute interface flux induced by density_metric * g along gravity
        direction.

        The density metric is computed elsewhere and can for instance be a measure of
        density difference between two phases.

        Parameters:
            interfaces: Interfaces where the density metric is applied.
            density_metric: The density metric to use in the flux computation.

        Returns:
            An Ad operator representing the interface density-driven flux.

        """
        normals = self.outwards_internal_boundary_normals(interfaces, unitary=True)
        subdomain_neighbors = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(
            self.mdg, subdomain_neighbors, interfaces, dim=self.nd
        )

        e_n = self.e_i(subdomain_neighbors, i=self.nd - 1, dim=self.nd)
        gravity_flux = (
            pp.ad.Scalar(-1)
            * e_n
            @ (density_metric * self.gravity_field(subdomain_neighbors))
        )

        intf_vector_source = projection.secondary_to_mortar_avg() @ gravity_flux

        normals_times_source = normals * intf_vector_source
        nd_to_scalar_sum = pp.ad.sum_projection_list(
            [e.T for e in self.basis(interfaces, dim=self.nd)]
        )
        w_flux = self.volume_integral(
            self.normal_permeability(interfaces)
            * (nd_to_scalar_sum @ normals_times_source),
            interfaces,
            1,
        )
        w_flux.set_name("interface_density_driven_flux_" + density_metric.name)
        return w_flux

    #: Valid buoyancy upwinding schemes.
    #: ``"phase_potential"`` selects phase-potential upwinding (PPU): the full phase
    #: flux (viscous and gravitational) is upwinded by that phase's own potential.
    #: ``"hybrid"`` selects hybrid upwinding (HU): the viscous part is upwinded by the
    #: total Darcy flux and the buoyancy part by the inter-phase gravity flux.
    _valid_buoyancy_upwinding_schemes: tuple[str, str, str] = (
        "phase_potential",
        "hybrid",
        "PPU_Discriminant",
    )

    def buoyancy_upwinding_scheme(self) -> str:
        """Return the selected buoyancy upwinding scheme.

        Controlled by the model parameter ``"buoyancy_upwinding"``. Defaults to
        ``"phase_potential"`` (PPU).

        Raises:
            ValueError: If the parameter holds an unsupported scheme name.

        Returns:
            Either ``"phase_potential"`` or ``"hybrid"``.

        """
        scheme = self.params.get("buoyancy_upwinding", "phase_potential")
        if scheme not in self._valid_buoyancy_upwinding_schemes:
            raise ValueError(
                f"Unknown buoyancy_upwinding scheme {scheme!r}. "
                f"Choose one of {self._valid_buoyancy_upwinding_schemes}."
            )
        return scheme

    def is_phase_potential_upwinding(self) -> bool:
        """Whether the phase-potential (PPU) buoyancy scheme is active.

        Returns:
            True for PPU, False for hybrid upwinding (HU).

        """
        return self.buoyancy_upwinding_scheme() == "phase_potential"

    def is_PPU_Discriminant(self) -> bool:
        """Whether the discriminant-gated PPU scheme (``"PPU_Discriminant"``) is active.

        Returns:
            True for the product/discriminant PPU, False otherwise.

        """
        return self.buoyancy_upwinding_scheme() == "PPU_Discriminant"

    def potential_driven_flux(
        self, subdomains: pp.SubdomainsOrBoundaries, phase: pp.Phase
    ) -> pp.ad.Operator:

        rho_mixture = self.fractionally_weighted_density(subdomains)
        rho_phase = phase.density(subdomains)
        # Proper PPU direction: sign of the phase-potential flux
        # -K (grad p - rho_phase g). Built from the total (mixture) potential flux as
        # darcy_flux + density_driven_flux(rho_phase - rho_mixture). No explicit total
        # mobility factor: darcy_flux and density_driven_flux share the same Darcy
        # tensor, so lambda_T is handled consistently (inside the tensor when
        # mass-mobility-weighted, absent otherwise) and cannot change the upwind sign.
        m_star = self.darcy_flux(subdomains) + self.density_driven_flux(
            subdomains, rho_phase - rho_mixture
        )
        m_star.set_name("potential_driven_flux_" + phase.name)
        return m_star

    # ------------------------------------------------------------------ #
    #  PPU_Discriminant: product-gated phase-potential upwinding          #
    # ------------------------------------------------------------------ #

    def _PPU_reference_potential(self) -> float:
        r"""Characteristic ``|\Psi_\gamma|`` scale (interior faces) used to size the
        smooth sign/gate widths. Refreshed in :meth:`update_buoyancy_driven_fluxes`;
        falls back to ``1.0`` before the first update."""
        ref = getattr(self, "_PPU_psi_ref", 0.0)
        return ref if ref > 0.0 else 1.0

    def _PPU_update_reference_potential(self) -> None:
        r"""Refresh ``_PPU_psi_ref`` = max interior-face ``|\Psi_\gamma|`` over phases."""
        subdomains = self.mdg.subdomains()
        if len(subdomains) == 0:
            return
        offsets = np.cumsum([0] + [sd.num_faces for sd in subdomains])
        ref = 0.0
        for ph in self.fluid.phases:
            vals = self.equation_system.evaluate(
                self.potential_driven_flux(subdomains, ph)
            )
            for k, sd in enumerate(subdomains):
                interior = sd.get_internal_faces()
                if interior.size == 0:
                    continue
                loc = np.abs(vals[offsets[k] : offsets[k] + sd.num_faces][interior])
                ref = max(ref, float(np.max(loc)))
        if ref > 0.0:
            self._PPU_psi_ref = ref

    def _PPU_face_operators(
        self, subdomains: list[pp.Grid]
    ) -> tuple[pp.ad.SparseArray, pp.ad.SparseArray, pp.ad.SparseArray, pp.ad.Operator]:
        r"""Per-face operators for the discriminant scheme (block-diagonal).

        Returns ``(face_avg, face_jump, boundary_cell_to_face, boundary_orientation)``:

        - ``face_jump @ w = w_+ - w_-`` on interior faces (``= cell_faces @ w``), boundary
          rows zeroed;
        - ``face_avg  @ w = 0.5 (w_+ + w_-)`` on interior faces, boundary rows zeroed;
        - ``boundary_cell_to_face @ w`` = the adjacent cell value on boundary faces, zero
          on interior;
        - ``boundary_orientation`` = ``+/-1`` on boundary faces (sign of ``cell_faces``,
          i.e. whether the global normal points out of the adjacent cell), ``0`` on
          interior; a constant dense array used to orient the boundary inflow test.
        """
        avg_blocks: List[sps.spmatrix] = []
        jump_blocks: List[sps.spmatrix] = []
        bcf_blocks: List[sps.spmatrix] = []
        orient: List[np.ndarray] = []
        for sd in subdomains:
            cf = sd.cell_faces.tocsr().astype(float)
            interior = np.zeros(sd.num_faces)
            interior[sd.get_internal_faces()] = 1.0
            boundary = 1.0 - interior
            row_int = sps.diags(interior, format="csr")
            row_bnd = sps.diags(boundary, format="csr")
            cf_abs = cf.copy()
            cf_abs.data = np.abs(cf_abs.data)
            jump_blocks.append(row_int @ cf)
            avg_blocks.append(0.5 * (row_int @ cf_abs))
            bcf_blocks.append(row_bnd @ cf_abs)
            orient.append(boundary * np.asarray(cf @ np.ones(sd.num_cells)).ravel())
        if len(subdomains) == 0:
            empty = sps.csr_matrix((0, 0))
            avg_blocks = [empty]
            jump_blocks = [empty]
            bcf_blocks = [empty]
            orient = [np.zeros(0)]
        mk = pp.matrix_operations.csr_matrix_from_sparse_blocks
        face_avg = pp.ad.SparseArray(mk(avg_blocks), name="PPU_face_avg")
        face_jump = pp.ad.SparseArray(mk(jump_blocks), name="PPU_face_jump")
        boundary_cell_to_face = pp.ad.SparseArray(
            mk(bcf_blocks), name="PPU_boundary_cell_to_face"
        )
        boundary_orientation = pp.wrap_as_dense_ad_array(
            np.concatenate(orient), name="PPU_boundary_orientation"
        )
        return face_avg, face_jump, boundary_cell_to_face, boundary_orientation

    def _PPU_smoothing_widths(self) -> tuple[float, float]:
        r"""``(eps_sign, eps_gate)`` in absolute units, scaled by the characteristic
        ``|\Psi|`` so the parameters ``"PPU_sign_eps"`` / ``"PPU_gate_eps"`` are
        dimensionless fractions. ``eps_gate`` is in potential-squared units (the gate
        acts on the product ``Psi_0 Psi_1``)."""
        ref = self._PPU_reference_potential()
        sign_frac = max(float(self.params.get("PPU_sign_eps", 0.05)), 1e-12)
        gate_frac = max(float(self.params.get("PPU_gate_eps", 0.3)), 1e-12)
        return sign_frac * ref, (gate_frac * ref) ** 2

    def _PPU_smooth_sign(self, x: pp.ad.Operator, eps_sign: float) -> pp.ad.Operator:
        """Smooth sign ``tanh(x / eps_sign)`` (in ``[-1, 1]``)."""
        return pp.ad.Function(pp.ad.tanh, "PPU_tanh")(
            x * pp.ad.Scalar(1.0 / eps_sign)
        )

    def _PPU_effective_signs(
        self,
        potentials: list[pp.ad.Operator],
        eps_sign: float,
        eps_gate: float,
    ) -> list[pp.ad.Operator]:
        r"""Per-phase effective upwind sign
        ``sigma_eff_gamma = (1-H) sigma_m + H sigma_gamma``.

        ``H = H_eps(- Psi_0 Psi_1)`` is the counter-current fraction (the discriminant
        ``Psi_0 Psi_1 = ΔΦ_m^2 - ΔΦ_b^2``); ``sigma_m`` is the bulk-drive sign and
        ``sigma_gamma`` the per-phase sign. Two-phase only. The list is aligned with
        ``potentials``.
        """
        assert len(potentials) == 2, (
            "PPU_Discriminant is implemented for two phases (vapour/liquid)."
        )
        psi_0, psi_1 = potentials
        sigma_0 = self._PPU_smooth_sign(psi_0, eps_sign)
        sigma_1 = self._PPU_smooth_sign(psi_1, eps_sign)
        sigma_m = self._PPU_smooth_sign(
            pp.ad.Scalar(0.5) * (psi_0 + psi_1), eps_sign
        )
        # Counter-current gate H in [0, 1]: 1 where the product is negative.
        neg_product = pp.ad.Scalar(-1.0) * (psi_0 * psi_1)
        gate = pp.ad.Function(
            lambda v: pp.ad.functions.heaviside_smooth(v, eps_gate), "PPU_gate"
        )(neg_product)
        one = pp.ad.Scalar(1.0)
        sigma_eff_0 = (one - gate) * sigma_m + gate * sigma_0
        sigma_eff_1 = (one - gate) * sigma_m + gate * sigma_1
        return [sigma_eff_0, sigma_eff_1]

    def PPU_discriminant_flux(
        self,
        subdomains: list[pp.Grid],
        advected_weight: Callable[[pp.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator],
        bc_type: Callable[[pp.Grid], pp.BoundaryCondition],
        name: str,
    ) -> pp.ad.Operator:
        r"""Discriminant-gated PPU advective flux for an advected weight.

        Builds, for two phases, the true per-phase PPU flux

        .. math::

            F = \sum_\gamma \Psi_\gamma\,\big(\text{avg}(w_\gamma)
                + \tfrac12\,\sigma^{\text{eff}}_\gamma\,\text{jump}(w_\gamma)\big),

        with each phase riding its own potential ``Psi_gamma = potential_driven_flux``,
        and the effective upwind sign produced by the product gate
        (:meth:`_PPU_effective_signs`). On boundary faces the interior gated weight is
        replaced by a smooth inflow/outflow blend between the EoS-reconstructed Dirichlet
        weight (inflow) and the adjacent cell value (outflow). Neumann faces of the
        benchmark carry ``Psi ~ 0`` (no-flux laterals) so they contribute nothing; a
        prescribed bulk Neumann/interface flux is not split per phase and is omitted here
        (zero for the all-Dirichlet, fracture-free benchmark).

        Parameters:
            subdomains: List of subdomains.
            advected_weight: Callable ``(phase, domains) -> w_gamma`` (cell-wise; must
                evaluate on boundary grids too).
            bc_type: Boundary-condition type callable for the advective flux.
            name: Unique name prefix for the per-phase boundary operators.

        Returns:
            Operator representing the discriminant-gated PPU flux.

        """
        phases = list(self.fluid.phases)
        potentials = [self.potential_driven_flux(subdomains, ph) for ph in phases]
        eps_sign, eps_gate = self._PPU_smoothing_widths()
        sigma_eff = self._PPU_effective_signs(potentials, eps_sign, eps_gate)
        face_avg, face_jump, bnd_cell_to_face, bnd_orient = self._PPU_face_operators(
            subdomains
        )
        half = pp.ad.Scalar(0.5)
        one = pp.ad.Scalar(1.0)

        def zero_boundary(
            boundary_grids: Sequence[pp.BoundaryGrid],
        ) -> pp.ad.Operator:
            n = sum(bg.num_cells for bg in boundary_grids)
            return pp.wrap_as_dense_ad_array(np.zeros(n), name=name + "_zero_bc")

        flux_terms: List[pp.ad.Operator] = []
        for k, ph in enumerate(phases):
            psi = potentials[k]
            w = advected_weight(ph, subdomains)
            # Interior gated upwind weight (boundary rows are zero here).
            w_interior = (face_avg @ w) + half * sigma_eff[k] * (face_jump @ w)
            # Per-phase Dirichlet boundary weight (EoS on the boundary grid).
            bc_w = self._combine_boundary_operators(
                subdomains=subdomains,
                dirichlet_operator=(
                    lambda boundary_grids, p=ph: advected_weight(p, boundary_grids)
                ),
                neumann_operator=zero_boundary,
                robin_operator=None,
                bc_type=bc_type,
                name=f"{name}_w_{ph.name}",
            )
            # Boundary inflow fraction: inflow when Psi points into the cell, i.e. the
            # outward-oriented potential (psi * orientation) is negative. Zero on interior
            # faces (orientation = 0 there), but the boundary weights below are also zero
            # on interior, so the interior contribution is unaffected.
            inflow = half * (
                one - self._PPU_smooth_sign(psi * bnd_orient, eps_sign)
            )
            w_boundary = inflow * bc_w + (one - inflow) * (bnd_cell_to_face @ w)
            flux_terms.append(psi * (w_interior + w_boundary))
        flux = pp.ad.sum_operator_list(flux_terms)
        flux.set_name("PPU_discriminant_flux")
        return flux

    @pp.ad.cached_method
    def __entity_buoyancy_flux(
        self,
        advected_gamma_quantity: pp.ad.Operator,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: pp.SubdomainsOrBoundaries,
    ) -> List[pp.ad.Operator]:
        """Helper function to compute the buoyancy flux on subdomains induced by gamma
        and delta, advecting a quantity associated to phase gamma.

        Parameters:
            advected_gamma_quantity: The quantity advected in phase gamma.
            gamma: The phase gamma.
            delta: The phase delta.
            domains: The domains over which the flux is computed.

        Raises:
            ValueError: If the domains are not subdomains.

        Returns:
            A list of Ad operators representing the buoyancy flux is phase gamma.

        """
        from porepy.models.compositional_flow import is_mass_mobility_weighted_permeability

        b_fluxes: List[pp.ad.Operator] = []
        rho_gamma = gamma.density(domains)
        rho_delta = delta.density(domains)
        density_metric = rho_gamma - rho_delta

        w_flux_gamma_delta = self.density_driven_flux(domains, density_metric)
        f_gamma = self.fractional_phase_mass_mobility(gamma, domains)
        f_delta = self.fractional_phase_mass_mobility(delta, domains)

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        # One hybrid-upwind discretization for the pair: the two phases upwind along the
        # single inter-phase gravity direction, gamma one way (upwind_gamma) and delta the
        # other (upwind_delta), instead of two separate Upwind objects.
        discr = self.hybrid_upwind_discretization(gamma, delta, domains)

        f_gamma_upwind: pp.ad.Operator = discr.upwind_gamma() @ (
            advected_gamma_quantity * f_gamma
        )  # well-defined fractional flow on facets.
        f_delta_upwind: pp.ad.Operator = (
            discr.upwind_delta() @ f_delta
        )  # well-defined fractional flow on facets.

        if  is_mass_mobility_weighted_permeability(self):
            b_flux_gamma_delta = (f_gamma_upwind * f_delta_upwind) * w_flux_gamma_delta
        else:
            l_gamma = gamma.density(domains) * self.phase_mobility(gamma, domains)
            l_delta = delta.density(domains) * self.phase_mobility(delta, domains)
            l_gamma_upwind: pp.ad.Operator = discr.upwind_gamma() @ (l_gamma)  # well-defined lambda on facets.
            l_delta_upwind: pp.ad.Operator = discr.upwind_delta() @ (l_delta)  # well-defined lambda on facets.
            lambda_upwind = l_gamma_upwind + l_delta_upwind
            b_flux_gamma_delta = (f_gamma_upwind * f_delta_upwind) * lambda_upwind * w_flux_gamma_delta

        b_fluxes.append(b_flux_gamma_delta)

        interfaces = self.subdomains_to_interfaces(domains, [1])

        if len(interfaces) != 0:
            # Get interface flux contribution.
            intf_density_metric = rho_gamma - rho_delta
            intf_w_flux_gamma_delta = self.interface_density_driven_flux(
                interfaces, intf_density_metric
            )

            # One hybrid interface discretization: gamma/delta from one gravity direction.
            intf_discr = self.hybrid_interface_upwind_discretization(
                gamma, delta, interfaces
            )

            # Projections and trace operators
            mortar_projection = pp.ad.MortarProjections(
                self.mdg, domains, interfaces, dim=1
            )
            trace = pp.ad.Trace(domains)

            # Modified coupling that properly handles dimensional transition
            primary_trace = trace.trace
            mortar_avg = mortar_projection.primary_to_mortar_avg()
            secondary_to_mortar = mortar_projection.secondary_to_mortar_avg()

            # Project quantities to interface with proper upwinding for both
            # primary and secondary sides
            gamma_interface = (
                intf_discr.upwind_primary_gamma()
                @ mortar_avg
                @ primary_trace
                @ (advected_gamma_quantity * f_gamma)
                + intf_discr.upwind_secondary_gamma()
                @ secondary_to_mortar
                @ (advected_gamma_quantity * f_gamma)
            )
            delta_interface = (
                intf_discr.upwind_primary_delta() @ mortar_avg @ primary_trace @ f_delta
                + intf_discr.upwind_secondary_delta() @ secondary_to_mortar @ f_delta
            )

            # Compute interface contribution and project back to primary grid
            if is_mass_mobility_weighted_permeability(self):
                interface_coupling_intf = (
                    gamma_interface * delta_interface
                ) * intf_w_flux_gamma_delta
            else:
                l_gamma = gamma.density(domains) * self.phase_mobility(gamma, domains)
                l_delta = delta.density(domains) * self.phase_mobility(delta, domains)
                l_gamma_interface = (
                    intf_discr.upwind_primary_gamma() @ mortar_avg @ primary_trace @ l_gamma
                    + intf_discr.upwind_secondary_gamma() @ secondary_to_mortar @ l_gamma
                )
                l_delta_interface = (
                    intf_discr.upwind_primary_delta() @ mortar_avg @ primary_trace @ l_delta
                    + intf_discr.upwind_secondary_delta() @ secondary_to_mortar @ l_delta
                )
                lambda_interface_upwind = l_gamma_interface + l_delta_interface
                interface_coupling_intf = (
                    gamma_interface * delta_interface
                ) * lambda_interface_upwind * intf_w_flux_gamma_delta
            # The gamma/delta boundary-transport matrices are equal; use the gamma one.
            b_intf_flux_gamma_delta = (
                discr.bound_transport_neu_gamma()
                @ mortar_projection.mortar_to_primary_int()
                @ interface_coupling_intf
            )
            b_fluxes.append(b_intf_flux_gamma_delta)
        return b_fluxes

    @pp.ad.cached_method
    def __entity_buoyancy_jump(
        self,
        advected_gamma_quantity: pp.ad.Operator,
        gamma: pp.Phase,
        delta: pp.Phase,
        domains: pp.SubdomainsOrBoundaries,
    ) -> List[pp.ad.Operator]:
        """Helper function to compute the buoyancy flux jump on interfaces induced by
        gamma and delta, advecting a quantity associated to phase gamma.

        Parameters:
            advected_gamma_quantity: The quantity advected in phase gamma.
            gamma: The phase gamma.
            delta: The phase delta.
            domains: The domains over which the flux is computed.

        Raises:
            ValueError: If the domains are not subdomains.

        Returns:
            A list of Ad operators representing the buoyancy flux is phase gamma.

        """

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)

        b_flux_jumps: List[pp.ad.Operator] = []
        size = sum(g.num_cells for g in domains)
        zero = pp.wrap_as_dense_ad_array(
            np.zeros(size), name="component_buoyancy_jump_zero"
        )
        b_flux_jumps.append(zero)

        rho_gamma = gamma.density(domains)
        rho_delta = delta.density(domains)

        f_gamma = self.fractional_phase_mass_mobility(gamma, domains)
        f_delta = self.fractional_phase_mass_mobility(delta, domains)
        interfaces = self.subdomains_to_interfaces(domains, [1])
        if len(interfaces) != 0:
            intf_density_metric = rho_gamma - rho_delta
            intf_w_flux_gamma_delta = self.interface_density_driven_flux(
                interfaces, intf_density_metric
            )

            # One hybrid interface discretization: gamma/delta from one gravity direction.
            intf_discr = self.hybrid_interface_upwind_discretization(
                gamma, delta, interfaces
            )

            # Projections and trace operators
            mortar_projection = pp.ad.MortarProjections(
                self.mdg, domains, interfaces, dim=1
            )
            trace = pp.ad.Trace(domains)

            # Modified coupling that properly handles dimensional transition
            primary_trace = trace.trace
            mortar_avg = mortar_projection.primary_to_mortar_avg()
            secondary_to_mortar = mortar_projection.secondary_to_mortar_avg()

            # Project quantities to interface with proper upwinding for both
            # primary and secondary sides
            gamma_interface = (
                intf_discr.upwind_primary_gamma()
                @ mortar_avg
                @ primary_trace
                @ (advected_gamma_quantity * f_gamma)
                + intf_discr.upwind_secondary_gamma()
                @ secondary_to_mortar
                @ (advected_gamma_quantity * f_gamma)
            )
            delta_interface = (
                intf_discr.upwind_primary_delta() @ mortar_avg @ primary_trace @ f_delta
                + intf_discr.upwind_secondary_delta() @ secondary_to_mortar @ f_delta
            )

            # Compute interface contribution and project back to secondary grid
            from porepy.models.compositional_flow import is_mass_mobility_weighted_permeability
            if is_mass_mobility_weighted_permeability(self):
                interface_coupling_intf = (
                    gamma_interface * delta_interface
                ) * intf_w_flux_gamma_delta
            else:
                l_gamma = gamma.density(domains) * self.phase_mobility(gamma, domains)
                l_delta = delta.density(domains) * self.phase_mobility(delta, domains)
                l_gamma_interface = (
                    intf_discr.upwind_primary_gamma() @ mortar_avg @ primary_trace @ l_gamma
                    + intf_discr.upwind_secondary_gamma() @ secondary_to_mortar @ l_gamma
                )
                l_delta_interface = (
                    intf_discr.upwind_primary_delta() @ mortar_avg @ primary_trace @ l_delta
                    + intf_discr.upwind_secondary_delta() @ secondary_to_mortar @ l_delta
                )
                lambda_interface_upwind = l_gamma_interface + l_delta_interface
                interface_coupling_intf = (
                    gamma_interface * delta_interface
                ) * lambda_interface_upwind * intf_w_flux_gamma_delta
            b_flux_jump_gamma_delta = (
                mortar_projection.mortar_to_secondary_int() @ interface_coupling_intf
            )
            b_flux_jumps.append(b_flux_jump_gamma_delta)
        return b_flux_jumps

    def phase_pairs_for(self, phase: pp.Phase) -> list[tuple[pp.Phase, pp.Phase]]:
        """Get all phase pairs involving a specific phase.

        The pair is ordered so that the given phase is first.

        Parameters:
            phase: The phase for which to get the pairs.

        Returns:
            A list of ordered phase pairs (gamma, delta).

        """
        combination_by_pairs = [
            pair for pair in list(combinations(self.fluid.phases, 2)) if phase in pair
        ]
        selected_pairs = []
        for pair in combination_by_pairs:
            idx = pair.index(phase)
            if idx == 0:
                phase_gamma, phase_delta = pair
            elif idx == 1:
                phase_delta, phase_gamma = pair
            else:
                continue
            selected_pairs.append((phase_gamma, phase_delta))
        return selected_pairs

    def component_buoyancy(
        self, component_xi: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Get the buoyancy flux for a given component.

        Parameters:
            component_xi: The component for which to get the buoyancy flux.
            domains: The domains to consider for the buoyancy flux calculation.

        Returns:
            Ad operator representing the buoyancy flux for the component.

        """
        b_fluxes: List[pp.ad.Operator] = []
        b_fluxes.append(self.density_driven_flux(domains, pp.ad.Scalar(0.0)))
        for phase in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase):
                gamma, delta = pairs
                chi_xi_gamma = gamma.partial_fraction_of[component_xi](domains)
                b_fluxes += self.__entity_buoyancy_flux(
                    chi_xi_gamma, gamma, delta, domains
                )
        b_flux = pp.ad.sum_operator_list(b_fluxes)
        b_flux.set_name("component_buoyancy_" + component_xi.name)
        return b_flux

    def enthalpy_buoyancy(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Get the buoyancy flux for specific enthalpy.

        Parameters:
            domains: The domains to consider for the buoyancy flux calculation.

        Returns:
            Ad operator representing the buoyancy flux for the specific enthalpy.

        """
        b_fluxes: List[pp.ad.Operator] = []
        b_fluxes.append(self.density_driven_flux(domains, pp.ad.Scalar(0.0)))
        for phase in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase):
                gamma, delta = pairs
                h_gamma = gamma.specific_enthalpy(domains)
                b_fluxes += self.__entity_buoyancy_flux(h_gamma, gamma, delta, domains)
        b_flux = pp.ad.sum_operator_list(b_fluxes)
        b_flux.set_name("enthalpy_buoyancy")
        return b_flux

    def component_buoyancy_jump(
        self, component_xi: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Get the interface jump term for component buoyancy.

        Parameters:
            component_xi: The component for which to get the interface jump term.
            domains: The domains to consider for the interface jump term calculation.

        Returns:
            Ad operator representing the interface jump term for the component.

        """
        b_fluxes: List[pp.ad.Operator] = []

        size = sum(g.num_cells for g in domains)
        zero = pp.wrap_as_dense_ad_array(
            np.zeros(size), name="component_buoyancy_jump_zero"
        )
        b_fluxes.append(zero)
        for phase in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase):
                gamma, delta = pairs
                chi_xi_gamma = gamma.partial_fraction_of[component_xi](domains)
                b_fluxes += self.__entity_buoyancy_jump(
                    chi_xi_gamma, gamma, delta, domains
                )
        b_flux = pp.ad.sum_operator_list(b_fluxes)
        b_flux.set_name("component_buoyancy_jump_" + component_xi.name)
        return b_flux

    def enthalpy_buoyancy_jump(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Get the interface jump term for enthalpy buoyancy.

        Parameters:
            domains: The domains to consider for the interface jump term calculation.

        Returns:
            Ad operator representing the interface jump term for the enthalpy.

        """
        b_fluxes: List[pp.ad.Operator] = []
        size = sum(g.num_cells for g in domains)
        zero = pp.wrap_as_dense_ad_array(
            np.zeros(size), name="enthalpy_buoyancy_jump_zero"
        )
        b_fluxes.append(zero)
        for phase in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase):
                gamma, delta = pairs
                h_gamma = gamma.specific_enthalpy(domains)
                b_fluxes += self.__entity_buoyancy_jump(h_gamma, gamma, delta, domains)
        b_flux = pp.ad.sum_operator_list(b_fluxes)
        b_flux.set_name("enthalpy_buoyancy_jump")
        return b_flux

    def set_buoyancy_discretization_parameters(self):
        """Initialize parameter containers and zero flux arrays for buoyancy."""
        for phase_gamma in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase_gamma):
                gamma, delta = pairs
                for sd, data in self.mdg.subdomains(return_data=True):
                    pp.initialize_data(data, self.buoyancy_key(gamma, delta))
                    pp.initialize_data(data, self.buoyancy_key(delta, gamma))
                    null_vals = np.zeros(sd.num_faces)
                    # Set the SAME advective-flux boundary classification used by the
                    # mobility discretization (bc_type_fluid_flux). Without this the
                    # Upwind discretization defaults to all-Dirichlet, so its upwind()
                    # and bound_transport_*() would not partition the boundary faces
                    # consistently with the advective flux (cf. PPU boundary bug).
                    bc = self.bc_type_fluid_flux(sd)
                    data[pp.PARAMETERS][self.buoyancy_key(gamma, delta)].update(
                        {
                            self.buoyant_flux_array_key(gamma, delta): +null_vals,
                            "bc": bc,
                        }
                    )
                    data[pp.PARAMETERS][self.buoyancy_key(delta, gamma)].update(
                        {
                            self.buoyant_flux_array_key(delta, gamma): -null_vals,
                            "bc": bc,
                        }
                    )
                    # Hybrid upwind: a single stored gravity direction per pair (its two
                    # opposite per-phase matrices are built in HUpwind.discretize).
                    pp.initialize_data(data, self.hybrid_upwind_key(gamma, delta))
                    data[pp.PARAMETERS][self.hybrid_upwind_key(gamma, delta)].update(
                        {"hybrid_gravity_flux": +null_vals, "bc": bc}
                    )
                for intf, data in self.mdg.interfaces(return_data=True):
                    null_vals = np.zeros(intf.num_cells)
                    pp.initialize_data(data, self.buoyancy_intf_key(gamma, delta))
                    pp.initialize_data(data, self.buoyancy_intf_key(delta, gamma))
                    data[pp.PARAMETERS][self.buoyancy_intf_key(gamma, delta)].update(
                        {self.buoyant_intf_flux_array_key(gamma, delta): +null_vals}
                    )
                    data[pp.PARAMETERS][self.buoyancy_intf_key(delta, gamma)].update(
                        {self.buoyant_intf_flux_array_key(delta, gamma): -null_vals}
                    )
                    # Hybrid interface upwind: one stored gravity direction per pair.
                    pp.initialize_data(data, self.hybrid_upwind_key(gamma, delta))
                    data[pp.PARAMETERS][self.hybrid_upwind_key(gamma, delta)].update(
                        {"hybrid_gravity_flux": +null_vals}
                    )

    def _register_hybrid_nonlinear_discretization(
        self, op: pp.ad.MergedOperator
    ) -> None:
        """Register a hybrid upwind operator for per-iteration re-discretization.

        Appends directly to the nonlinear-discretization list, bypassing
        :meth:`add_nonlinear_discretization`'s exact-type guardrail. That guardrail admits
        only the base ``Upwind`` / ``UpwindCoupling`` types; ``HUpwind`` /
        ``HUpwindCoupling`` are functionally valid subclasses, so this keeps the change
        contained to this module instead of relaxing the core check.
        """
        if op not in self._nonlinear_discretizations:
            self._nonlinear_discretizations.append(op)

    def set_nonlinear_buoyancy_discretization(self):
        """Register nonlinear upwind discretizations for buoyancy terms."""
        for phase_gamma in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase_gamma):
                gamma, delta = pairs
                # Hybrid upwind: one discretization per pair, two matrices
                # (re-discretized each iteration as the gravity direction is refreshed).
                hybrid_discr = self.hybrid_upwind_discretization(
                    gamma, delta, self.mdg.subdomains()
                )
                self._register_hybrid_nonlinear_discretization(
                    hybrid_discr.upwind_gamma()
                )
                self._register_hybrid_nonlinear_discretization(
                    hybrid_discr.upwind_delta()
                )
                # One hybrid interface (mortar) discretization per pair, four matrices.
                intf_discr = self.hybrid_interface_upwind_discretization(
                    gamma, delta, self.mdg.interfaces(codim=1)
                )
                self._register_hybrid_nonlinear_discretization(
                    intf_discr.upwind_primary_gamma()
                )
                self._register_hybrid_nonlinear_discretization(
                    intf_discr.upwind_secondary_gamma()
                )
                self._register_hybrid_nonlinear_discretization(
                    intf_discr.upwind_primary_delta()
                )
                self._register_hybrid_nonlinear_discretization(
                    intf_discr.upwind_secondary_delta()
                )

    def update_buoyancy_driven_fluxes(self):
        """Update stored buoyancy flux arrays (subdomains and interfaces).

        The stored arrays define the upwinding direction of the buoyancy
        discretizations. In the phase-potential scheme (PPU) they hold each phase's
        phase-potential flux; in the hybrid scheme (HU) they hold the inter-phase
        gravity flux (opposite sign for the two phases).
        """
        # For the discriminant PPU scheme, refresh the characteristic |Psi| scale used to
        # size the smooth sign/gate widths (over interior faces, so the strong boundary
        # fluxes do not inflate it). Constant per Newton solve, no Jacobian pollution.
        if self.is_PPU_Discriminant():
            self._PPU_update_reference_potential()

        ppu_Q = self.is_phase_potential_upwinding()
        for phase_gamma in self.fluid.phases:
            for pairs in self.phase_pairs_for(phase_gamma):
                gamma, delta = pairs

                # Compute the values for all subdomains jointly, then distribute in a
                # for-loop. This is faster evaluation inside a loop over subdomains.
                subdomains = self.mdg.subdomains()

                rho_gamma_full = gamma.density(subdomains)
                rho_delta_full = delta.density(subdomains)
                # The hybrid upwind always rides the single inter-phase gravity direction.
                gravity_vals = self.equation_system.evaluate(
                    self.density_driven_flux(
                        subdomains, rho_gamma_full - rho_delta_full
                    )
                )
                if ppu_Q:
                    subdomain_gamma_vals = self.equation_system.evaluate(
                        self.potential_driven_flux(
                            subdomains, gamma
                        )
                    )
                    subdomain_delta_vals = self.equation_system.evaluate(
                        self.potential_driven_flux(
                            subdomains, delta
                        )
                    )
                else:
                    subdomain_vals = gravity_vals

                # Offsets for the indices of individual subdomains.
                subdomain_offsets = np.cumsum([0] + [sd.num_faces for sd in subdomains])

                for id, (sd, data) in enumerate(self.mdg.subdomains(return_data=True)):
                    sd_offset = subdomain_offsets[id]

                    # Single gravity direction for the hybrid upwind (gamma uses +, delta
                    # uses - internally in HUpwind.discretize).
                    grav_loc = gravity_vals[sd_offset : sd_offset + sd.num_faces]
                    data[pp.PARAMETERS][self.hybrid_upwind_key(gamma, delta)].update(
                        {"hybrid_gravity_flux": +grav_loc}
                    )

                    if ppu_Q:
                        vals_loc = subdomain_gamma_vals[sd_offset: sd_offset + sd.num_faces]
                        data[pp.PARAMETERS][self.buoyancy_key(gamma, delta)].update(
                            {self.buoyant_flux_array_key(gamma, delta): +vals_loc}
                        )
                        vals_loc = subdomain_delta_vals[sd_offset: sd_offset + sd.num_faces]
                        data[pp.PARAMETERS][self.buoyancy_key(delta, gamma)].update(
                            {self.buoyant_flux_array_key(delta, gamma): +vals_loc}
                        )
                    else:
                        vals_loc = subdomain_vals[sd_offset: sd_offset + sd.num_faces]
                        data[pp.PARAMETERS][self.buoyancy_key(gamma, delta)].update(
                            {self.buoyant_flux_array_key(gamma, delta): +vals_loc}
                        )
                        data[pp.PARAMETERS][self.buoyancy_key(delta, gamma)].update(
                            {self.buoyant_flux_array_key(delta, gamma): -vals_loc}
                        )

                # Same procedure for interfaces.
                interfaces = self.subdomains_to_interfaces(subdomains, [1])

                if len(interfaces) < 1:
                    # Shortcut for fracture-less domains.
                    continue

                interface_values = self.equation_system.evaluate(
                    self.interface_density_driven_flux(
                        interfaces, rho_gamma_full - rho_delta_full
                    )
                )
                interface_offsets = np.cumsum(
                    [0] + [intf.num_cells for intf in interfaces]
                )

                for id, (intf, data) in enumerate(
                    self.mdg.interfaces(return_data=True, codim=1)
                ):
                    intf_offset = interface_offsets[id]
                    vals_loc = interface_values[
                        intf_offset : intf_offset + intf.num_cells
                    ]

                    data[pp.PARAMETERS][self.buoyancy_intf_key(gamma, delta)].update(
                        {self.buoyant_intf_flux_array_key(gamma, delta): +vals_loc}
                    )
                    data[pp.PARAMETERS][self.buoyancy_intf_key(delta, gamma)].update(
                        {self.buoyant_intf_flux_array_key(delta, gamma): -vals_loc}
                    )
                    # Single interface gravity direction for the hybrid coupling.
                    data[pp.PARAMETERS][self.hybrid_upwind_key(gamma, delta)].update(
                        {"hybrid_gravity_flux": +vals_loc}
                    )


class ConstantViscosity(pp.PorePyModel):
    """Constant viscosity for a single-phase fluid."""

    def viscosity_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a constant viscosity for the fluid's phase.

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing representing the constant phase viscosity on some
            domains.

        """

        def mu(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            return Scalar(self.fluid.reference_component.viscosity, "viscosity")

        return mu


class ConstantFluidThermalConductivity(pp.PorePyModel):
    """Ïmplementation of a constant thermal conductivity for a single-phase fluid."""

    def thermal_conductivity_of_phase(
        self, phase: pp.Phase
    ) -> ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a constant thermal conductivity for the fluid's phase.

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing the constant phase conductivity on some domains.

        """

        def kappa(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            return Scalar(
                self.fluid.reference_component.thermal_conductivity,
                "fluid_thermal_conductivity",
            )

        return kappa

    def normal_thermal_conductivity(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Constant normal thermal conductivity of the fluid given by the fluid
        constants stored in the fluid's reference component.

        Using the fluid value corresponds to assuming a fluid-filled fracture.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing normal thermal conductivity on the interfaces.

        """
        # NOTE this is not really a fluid-related const. law, it is more related to
        # mixed-dimensional problems.
        val = self.fluid.reference_component.normal_thermal_conductivity
        return Scalar(val, "normal_thermal_conductivity")


class FluidEnthalpyFromTemperature(pp.PorePyModel):
    """Implementation of a linearized fluid enthalpy :math:`c(T - T_{ref})` for a
    single-phase, single-component fluid.

    It uses the specific heat capacity of the fluid's reference component as :math:`c`,
    which is constant.

    """

    def fluid_specific_heat_capacity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """
        Parameters:
            subdomains: List of subdomains. Not used, but included for consistency with
                other implementations.

        Returns:
            Operator representing the fluid specific heat capacity  [J/kg/K]. The value
            is picked from the constants of the reference component.

        """
        return Scalar(
            self.fluid.reference_component.specific_heat_capacity,
            "fluid_specific_heat_capacity",
        )

    def specific_enthalpy_of_phase(self, phase: pp.Phase) -> ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a linear specific enthalpy for the fluid's phase.

        .. math::

            h = c \\left(T - T_0\\right)

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

        """

        def h(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            c = self.fluid_specific_heat_capacity(cast(list[pp.Grid], domains))
            enthalpy = c * self.perturbation_from_reference(
                "temperature", cast(list[pp.Grid], domains)
            )
            enthalpy.set_name("fluid_enthalpy")
            return enthalpy

        return h
