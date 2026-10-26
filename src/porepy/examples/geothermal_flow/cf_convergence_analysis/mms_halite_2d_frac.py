"""Exact solution and source terms of the manufactured-solution test for a 2D fractured porous medium
with halite precipitation dependent aperture.

The matrix fields are prescribed on four smooth regions. In the fracture the
halite saturation is prescribed, and the aperture, pressure, enthalpy and
composition follow from it explicitly. The sources are the residuals of the
exact solution, derived symbolically.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
import sympy as sp

from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_halite_closure import (
    SYM,
)
from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_halite_closure import (
    h as _h,
)
from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_halite_closure import (
    p as _p,
)
from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_halite_closure import (
    s as _s,
)
from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_halite_closure import (
    z as _z,
)
from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_halite_constants import (
    SYMBOLS,
    HaliteMMSConstants,
    MMS_HALITE,
)


x, y, t = sp.symbols("x y t", real=True)

_C = SYMBOLS
p0, h0, z0, T0 = _C["p0"], _C["h0"], _C["z0"], _C["T0"]
P, H1, Z1, xi = _C["P"], _C["H1"], _C["Z1"], _C["xi"]
S0, S1 = _C["S0"], _C["S1"]
a0, theta = _C["a0"], _C["theta"]
K0, K_k, phi0 = _C["K0"], _C["K_k"], _C["phi0"]
cp = _C["cp"]
x_frac, y_lo, y_hi = _C["x_frac"], _C["y_tip_lo"], _C["y_tip_hi"]

COORD = (x, y, t)
"""tuple: Space-time coordinates of matrix callables."""

FRACTURE_COORD = (y, t)
"""tuple: Space-time coordinates of fracture and interface callable."""


def grad(f: sp.Expr) -> sp.Matrix:
    """Gradient of a scalar field on the matrix.

    Parameters:
        f: Scalar expression in ``(x, y, t)``.

    Returns:
        The two-component gradient.

    """
    return sp.Matrix([sp.diff(f, x), sp.diff(f, y)])


def div(F: sp.Matrix) -> sp.Expr:
    """Divergence of a vector field with two components.

    Parameters:
        F: Two-component expression.

    Returns:
        The divergence.

    """
    return sp.diff(F[0], x) + sp.diff(F[1], y)


def omega() -> sp.Expr:
    """Bubble function with double roots at both fracture tips, normalised to a unit maximum
    on the fracture

    Returns:
        The bubble as a function of ``y``.

    """
    y_mid = (y_lo + y_hi) / 2
    normalisation = 1 / ((y_mid - y_lo) ** 2 * (y_mid - y_hi) ** 2)
    return normalisation * (y - y_lo) ** 2 * (y - y_hi) ** 2


def closure_in_primary(name: str, p_e: sp.Expr, h_e: sp.Expr, z_e: sp.Expr) -> sp.Expr:
    """Evaluate a closure quantity at prescribed ``(p, h, z)`` fields (matrix).

    Parameters:
        name: Attribute name on :class:`SymbolicClosure`.
        p_e: Pressure field.
        h_e: Enthalpy field.
        z_e: Overall NaCl mass fraction field.

    Returns:
        The closure quantity at those fields.

    """
    expr = SYM.in_phz(getattr(SYM, name))
    return expr.subs({_p: p_e, _h: h_e, _z: z_e}, simultaneous=True)


def closure_in_saturation(
    expr: sp.Expr, p_e: sp.Expr, h_e: sp.Expr, s_e: sp.Expr
) -> sp.Expr:
    """Evaluate a closure expression at prescribed ``(p, h, s)`` fields (fracture).

    Parameters:
        expr: Closure expression, possibly in the saturation symbol.
        p_e: Pressure field.
        h_e: Enthalpy field.
        s_e: Halite saturation field.

    Returns:
        The expression at those fields.

    """
    return expr.subs({_p: p_e, _h: h_e, _s: s_e}, simultaneous=True)


@lru_cache(maxsize=None)
def _cached_operation_count(expression) -> int:
    """Operation count of a hashable expression, cached.

    Parameters:
        expression: Sympy expression.

    Returns:
        The number of operations.

    """
    return sp.count_ops(expression)


def _operation_count(expression) -> int:
    """Operation count, cached where the expression is hashable.

    Parameters:
        expression: Sympy expression or matrix.

    Returns:
        The number of operations.

    """
    try:
        return _cached_operation_count(expression)
    except TypeError:
        return sp.count_ops(expression)


def _summarise(instance) -> str:
    """One-line summary of a solution container by expression size.

    Parameters:
        instance: Dataclass instance whose fields are sympy expressions.

    Returns:
        The field names with their operation counts.

    """
    parts = []
    for field in fields(instance):
        value = getattr(instance, field.name)
        if isinstance(value, (sp.Basic, sp.MatrixBase)):
            parts.append(f"{field.name}=<{_operation_count(value)} ops>")
        else:
            parts.append(f"{field.name}={value!r}")
    return f"{type(instance).__name__}({', '.join(parts)})"


@dataclass(frozen=True)
class Region:
    """One matrix region with its own smooth distance function ``delta``."""

    name: str
    delta: sp.Expr
    has_bubble: bool
    interface_normal_x: int | None


def _regions() -> tuple[Region, ...]:
    """Build the four matrix regions.

    Returns:
        The regions, in dispatch order.

    """

    radial_lo = sp.sqrt((x - x_frac) ** 2 + (y - y_lo) ** 2)
    radial_hi = sp.sqrt((x - x_frac) ** 2 + (y - y_hi) ** 2)

    return (
        Region("below", radial_lo, False, None),
        Region("left", x_frac - x, True, +1),
        Region("right", x - x_frac, True, -1),
        Region("above", radial_hi, False, None),
    )


REGIONS: tuple[Region, ...] = _regions()
"""tuple[Region, ...]: The four matrix regions."""


@dataclass(frozen=True, repr=False)
class MatrixSolution:
    """Exact matrix fields, fluxes and sources on one region."""

    region: Region
    pressure: sp.Expr
    enthalpy: sp.Expr
    composition: sp.Expr
    halite_saturation: sp.Expr
    temperature: sp.Expr
    porosity: sp.Expr
    permeability: sp.Expr
    darcy_flux: sp.Matrix
    mass_flux: sp.Matrix
    component_flux_NaCl: sp.Matrix
    component_flux_H2O: sp.Matrix
    advective_energy_flux: sp.Matrix
    conductive_energy_flux: sp.Matrix
    energy_flux: sp.Matrix
    source_mass: sp.Expr
    source_NaCl: sp.Expr
    source_H2O: sp.Expr
    source_energy: sp.Expr

    two_phase_margin: sp.Expr

    __repr__ = _summarise


def build_matrix(region: Region) -> MatrixSolution:
    """Assemble the exact matrix solution and its sources on one region.

    Parameters:
        region: Region to build on.

    Returns:
        The exact fields, fluxes and sources on the region.

    """
    base = region.delta ** (xi + 1)
    shape_fun = base + omega() * region.delta if region.has_bubble else base

    pressure = p0 + P * t * shape_fun
    enthalpy = h0 + H1 * t * shape_fun
    # The composition carries no bubble term: it enters no interface law of its
    # own, and the component interface flux is already non-zero through u_k.
    composition = z0 + Z1 * t * base

    def closure(name: str) -> sp.Expr:
        return closure_in_primary(name, pressure, enthalpy, composition)

    saturation = closure("s_hal")
    temperature = closure("temperature")
    porosity = closure("matrix_porosity")
    permeability = closure("matrix_permeability")
    density = closure("rho")
    mobility = closure("lambda_total")
    f_nacl = closure("f_NaCl")
    f_h2o = closure("f_H2O")
    conductivity = SYM.in_phz(SYM.thermal_conductivity(SYM.matrix_porosity)).subs(
        {_p: pressure, _h: enthalpy, _z: composition}, simultaneous=True
    )

    # Mobility-free Darcy flux. With fractional_flow disabled this is
    # what the model discretises with MPFA, and the mobility is upwinded
    # separately into the advective fluxes below.
    darcy_flux = -permeability * grad(pressure)

    mass_flux = mobility * darcy_flux
    component_flux_nacl = f_nacl * mass_flux
    component_flux_h2o = f_h2o * mass_flux
    advective_energy_flux = mobility * enthalpy * darcy_flux
    conductive_energy_flux = -conductivity * grad(temperature)
    energy_flux = advective_energy_flux + conductive_energy_flux

    # Accumulation terms. The matrix specific volume is unity, so it does not
    # appear inside the time derivative.
    acc_mass = porosity * density
    acc_nacl = porosity * density * composition
    acc_h2o = porosity * density * (1 - composition)
    acc_energy = SYM.in_phz(SYM.energy_accumulation(SYM.matrix_porosity)).subs(
        {_p: pressure, _h: enthalpy, _z: composition}, simultaneous=True
    )

    # The matrix has no interface sum: it has no higher-dimensional neighbour,
    # and the interface flux enters instead as the internal Neumann datum.
    source_mass = sp.diff(acc_mass, t) + div(mass_flux)
    source_nacl = sp.diff(acc_nacl, t) + div(component_flux_nacl)
    source_h2o = sp.diff(acc_h2o, t) + div(component_flux_h2o)
    source_energy = sp.diff(acc_energy, t) + div(energy_flux)

    chi_star = closure("chi_star")

    return MatrixSolution(
        region=region,
        pressure=pressure,
        enthalpy=enthalpy,
        composition=composition,
        halite_saturation=saturation,
        temperature=temperature,
        porosity=porosity,
        permeability=permeability,
        darcy_flux=darcy_flux,
        mass_flux=mass_flux,
        component_flux_NaCl=component_flux_nacl,
        component_flux_H2O=component_flux_h2o,
        advective_energy_flux=advective_energy_flux,
        conductive_energy_flux=conductive_energy_flux,
        energy_flux=energy_flux,
        source_mass=source_mass,
        source_NaCl=source_nacl,
        source_H2O=source_h2o,
        source_energy=source_energy,
        two_phase_margin=composition - chi_star,
    )


MATRIX: dict[str, MatrixSolution] = {r.name: build_matrix(r) for r in REGIONS}
"""dict[str, MatrixSolution]: The matrix solution, keyed by region name."""


@dataclass(frozen=True)
class TraceValues:
    """Constant matrix-side trace quantities on the interfaces, used to derive ``p_l`` and ``h_l``."""

    halite_saturation: sp.Expr
    porosity: sp.Expr
    permeability: sp.Expr
    conductivity: sp.Expr
    mobility: sp.Expr
    f_NaCl: sp.Expr
    f_H2O: sp.Expr
    enthalpy_weight: sp.Expr
    temperature: sp.Expr


def build_traces() -> TraceValues:
    """Evaluate the closure at the constant trace state ``(p0, h0, z0)``.

    Returns:
        The trace quantities.

    """

    def closure(name: str) -> sp.Expr:
        return closure_in_primary(name, p0, h0, z0)

    mobility = closure("lambda_total")
    conductivity = SYM.in_phz(SYM.thermal_conductivity(SYM.matrix_porosity)).subs(
        {_p: p0, _h: h0, _z: z0}, simultaneous=True
    )
    return TraceValues(
        halite_saturation=closure("s_hal"),
        porosity=closure("matrix_porosity"),
        permeability=closure("matrix_permeability"),
        conductivity=conductivity,
        mobility=mobility,
        f_NaCl=closure("f_NaCl"),
        f_H2O=closure("f_H2O"),
        enthalpy_weight=mobility * h0,
        temperature=closure("temperature"),
    )


TRACE: TraceValues = build_traces()
"""TraceValues: Constant matrix-side traces on both interfaces."""


@dataclass(frozen=True, repr=False)
class FractureSolution:
    """Exact fracture fields, fluxes and sources."""

    halite_saturation: sp.Expr
    aperture: sp.Expr
    specific_volume: sp.Expr

    permeability: sp.Expr
    pressure: sp.Expr
    enthalpy: sp.Expr
    composition: sp.Expr
    temperature: sp.Expr
    conductivity: sp.Expr
    darcy_flux: sp.Expr
    mass_flux: sp.Expr
    component_flux_NaCl: sp.Expr
    component_flux_H2O: sp.Expr
    advective_energy_flux: sp.Expr
    conductive_energy_flux: sp.Expr
    energy_flux: sp.Expr
    source_mass: sp.Expr
    source_NaCl: sp.Expr
    source_H2O: sp.Expr
    source_energy: sp.Expr

    __repr__ = _summarise


@dataclass(frozen=True, repr=False)
class InterfaceSolution:
    """Exact fluxes on one matrix-fracture interface."""

    darcy_flux: sp.Expr
    mass_flux: sp.Expr
    component_flux_NaCl: sp.Expr
    component_flux_H2O: sp.Expr
    advective_energy_flux: sp.Expr
    conductive_energy_flux: sp.Expr
    energy_flux: sp.Expr
    specific_volume: sp.Expr

    __repr__ = _summarise


def build_fracture_and_interface() -> tuple[FractureSolution, InterfaceSolution]:
    """Build the exact fracture solution and the interface fluxes.

    Returns:
        The fracture solution and the interface fluxes.

    """
    w = omega()

    # Step 2: the halite saturation is prescribed, so the aperture is known
    # before any interface law is written. Its time dependence is what makes
    # this test new: the accumulation term acquires a d(a_l)/dt contribution and
    # the tangential divergence a d(a_l)/dy one.
    saturation = S0 + S1 * t * w

    # Everything below depends only on saturation alone, so p and h are inert here.
    aperture = closure_in_saturation(SYM.aperture, p0, h0, saturation)
    permeability = closure_in_saturation(SYM.fracture_permeability, p0, h0, saturation)

    # Step 3, pressure. Equating the matrix-side mobility-free Darcy flux with
    # the interface law, whose coefficient is the constant
    # normal permeability K_k and not the cubic law. The trace pressure is p0.
    pressure = p0 - aperture * TRACE.permeability * P * t * w / (2 * K_k)

    # The fracture porosity is unity, so the effective conductivity reduces to
    # the fluid conductivity -- exactly what the framework takes as the normal
    # conductivity on the interface.
    conductivity = closure_in_saturation(
        SYM.thermal_conductivity(porosity=sp.Integer(1)), p0, h0, saturation
    )

    # Step 3, enthalpy. The advective parts of the two energy fluxes cancel
    # because the interface weight is upwinded from the matrix, leaving the
    # conductive balance. The factor 1/cp appears on both sides and cancels, so
    # temperature never enters as an independent field.
    enthalpy = h0 - aperture * TRACE.conductivity * H1 * t * w / (2 * conductivity)

    # Step 4: the forward map closes the chain at the now-known (p_l, h_l).
    composition = closure_in_saturation(SYM.z_NaCl, pressure, enthalpy, saturation)

    def closure(expr: sp.Expr) -> sp.Expr:
        return closure_in_saturation(expr, pressure, enthalpy, saturation)

    density = closure(SYM.rho)
    mobility = closure(SYM.lambda_total)
    f_nacl = closure(SYM.f_NaCl)
    f_h2o = closure(SYM.f_H2O)
    temperature = closure(SYM.temperature)

    # Tangential fluxes. In one dimension the divergence reduces to d/dy(.).
    darcy_flux = -permeability * sp.diff(pressure, y)
    mass_flux = mobility * darcy_flux
    component_flux_nacl = f_nacl * mass_flux
    component_flux_h2o = f_h2o * mass_flux
    advective_energy_flux = mobility * enthalpy * darcy_flux
    conductive_energy_flux = -conductivity * sp.diff(temperature, y)
    energy_flux = advective_energy_flux + conductive_energy_flux

    # Interface fluxes. The matrix trace pressure is p0 and the trace temperature is
    # T(p0, h0), both constants.
    normal_gradient = 2 / aperture
    u_k = K_k * normal_gradient * (p0 - pressure)
    q_k = conductivity * normal_gradient * (TRACE.temperature - temperature)

    # The advective weights are upwinded from the matrix, which is legitimate
    # for all time because omega >= 0 and t > 0 keep u_k > 0, so the upwind
    # branch never switches.
    w_k = TRACE.enthalpy_weight * u_k

    interface = InterfaceSolution(
        darcy_flux=u_k,
        mass_flux=TRACE.mobility * u_k,
        component_flux_NaCl=TRACE.f_NaCl * TRACE.mobility * u_k,
        component_flux_H2O=TRACE.f_H2O * TRACE.mobility * u_k,
        advective_energy_flux=w_k,
        conductive_energy_flux=q_k,
        energy_flux=w_k + q_k,
        # A codimension-one interface inherits its specific volume from the
        # higher-dimensional neighbour, whose aperture is one.
        specific_volume=sp.Integer(1),
    )

    # Accumulation, with the fracture porosity equal to one.
    acc_mass = density
    acc_nacl = density * composition
    acc_h2o = density * (1 - composition)
    acc_energy = closure(SYM.energy_accumulation(sp.Integer(1)))

    nu = aperture
    n_interfaces = 2  # Gamma_j and Gamma_k, carrying identical fluxes.

    def source(acc: sp.Expr, flux: sp.Expr, interface_flux: sp.Expr) -> sp.Expr:
        """Fracture residual divided by the specific volume.

        Parameters:
            acc: Accumulation term.
            flux: Tangential flux.
            interface_flux: Flux from each interface into the fracture.

        Returns:
            The source term.

        """

        return (
            sp.diff(nu * acc, t)
            + sp.diff(nu * flux, y)
            - n_interfaces * interface.specific_volume * interface_flux
        ) / nu

    fracture = FractureSolution(
        halite_saturation=saturation,
        aperture=aperture,
        specific_volume=nu,
        permeability=permeability,
        pressure=pressure,
        enthalpy=enthalpy,
        composition=composition,
        temperature=temperature,
        conductivity=conductivity,
        darcy_flux=darcy_flux,
        mass_flux=mass_flux,
        component_flux_NaCl=component_flux_nacl,
        component_flux_H2O=component_flux_h2o,
        advective_energy_flux=advective_energy_flux,
        conductive_energy_flux=conductive_energy_flux,
        energy_flux=energy_flux,
        source_mass=source(acc_mass, mass_flux, interface.mass_flux),
        source_NaCl=source(
            acc_nacl, component_flux_nacl, interface.component_flux_NaCl
        ),
        source_H2O=source(acc_h2o, component_flux_h2o, interface.component_flux_H2O),
        source_energy=source(acc_energy, energy_flux, interface.energy_flux),
    )
    return fracture, interface


FRACTURE, INTERFACE = build_fracture_and_interface()
"""FractureSolution, InterfaceSolution: The exact lower-dimensional solution."""


def _trace_at_interface(
    expr: sp.Expr, region: Region, constants: HaliteMMSConstants
) -> sp.Expr:
    """Trace of a matrix expression on the adjacent interface, by substitution at ``delta = 0``.

    Parameters:
        expr: Matrix expression in ``(x, y, t)``.
        region: Region touching an interface.
        constants: Constants substituted before the coordinate.

    Returns:
        The trace, a function of ``(y, t)``.

    Raises:
        ValueError: If the region does not touch an interface.

    """
    if region.interface_normal_x is None:
        raise ValueError(f"Region {region.name} does not touch an interface.")
    return expr.xreplace(constants.as_sympy_dict(SYMBOLS)).subs(x, constants.x_frac)


def delta_scaling_report(
    constants: HaliteMMSConstants = MMS_HALITE,
    time_value: float = 1.0,
    angles: Sequence[float] = (0.25, 0.5, 0.75),
) -> dict[str, dict[str, float]]:
    """Estimate the leading power of ``delta`` in each source near the fracture.

    Parameters:
        constants: Constants to substitute.
        time_value: Time level of the evaluation.
        angles: Approach directions in the radial regions, as fractions of ``pi``.

    Returns:
        The exponent per region and source, non-negative meaning bounded, and the
        largest magnitude under ``"max_abs"``.

    """
    subs = constants.as_sympy_dict(SYMBOLS)
    deltas = np.geomspace(1e-2, 1e-8, 13)
    report: dict[str, dict[str, float]] = {}

    for region in REGIONS:
        matrix = MATRIX[region.name]
        entry: dict[str, float] = {}
        sources = {
            "mass": matrix.source_mass,
            "NaCl": matrix.source_NaCl,
            "energy": matrix.source_energy,
        }
        for name, expr in sources.items():
            func = sp.lambdify(COORD, expr.xreplace(subs), "numpy")
            magnitudes: list[float] = []
            for d in deltas:
                samples: list[float] = []
                if region.interface_normal_x is not None:
                    # The middle band: delta is exactly the distance in x, and
                    # the approach direction is fixed.
                    x_v = float(constants.x_frac) + (
                        -d if region.interface_normal_x > 0 else d
                    )
                    y_v = 0.5 * (constants.y_tip_lo + constants.y_tip_hi)
                    samples.append(abs(float(func(x_v, y_v, time_value))))
                else:
                    # A radial region: approach the tip from several directions.
                    tip = (
                        constants.y_tip_lo
                        if region.name == "below"
                        else constants.y_tip_hi
                    )
                    sign = -1.0 if region.name == "below" else 1.0
                    for a in angles:
                        angle = float(a) * np.pi
                        x_v = float(constants.x_frac) + d * np.cos(angle)
                        y_v = float(tip) + sign * d * abs(np.sin(angle))
                        samples.append(abs(float(func(x_v, y_v, time_value))))
                magnitudes.append(max(samples))

            values = np.array(magnitudes)
            entry[f"max_abs_{name}"] = float(values.max())
            # Slope of log|f| against log(delta) over the last decade, where the
            # asymptotic behaviour has set in.
            finite = values > 0
            if finite.sum() >= 2:
                slope = np.polyfit(
                    np.log(deltas[finite][-5:]), np.log(values[finite][-5:]), 1
                )[0]
            else:
                slope = np.inf
            entry[f"exponent_{name}"] = float(slope)
        report[region.name] = entry
    return report


def _region_masks(
    xx: np.ndarray, yy: np.ndarray, constants: HaliteMMSConstants
) -> dict[str, np.ndarray]:
    """Boolean masks assigning points to the matrix regions.

    Parameters:
        xx: Abscissae.
        yy: Ordinates.
        constants: Geometry constants.

    Returns:
        A mask per region name.

    """
    below = yy < constants.y_tip_lo
    above = yy > constants.y_tip_hi
    middle = ~below & ~above
    left = middle & (xx < constants.x_frac)
    right = middle & (xx >= constants.x_frac)
    return {"below": below, "above": above, "left": left, "right": right}


def fracture_term_balance(
    constants: HaliteMMSConstants = MMS_HALITE,
    resolution: int = 201,
    time_levels: Sequence[float] = (0.25, 0.5, 1.0),
) -> dict[str, dict[str, float]]:
    """Compare the accumulation, tangential and interface terms of each fracture balance.

    Parameters:
        constants: Constants to substitute.
        resolution: Number of samples along the fracture.
        time_levels: Time levels to check.

    Returns:
        The largest magnitude per balance and term, and the ratio of the tangential
        to the interface term.

    """
    subs = constants.as_sympy_dict(SYMBOLS)
    w = omega()
    saturation = FRACTURE.halite_saturation
    nu = FRACTURE.aperture

    def closure(expr: sp.Expr) -> sp.Expr:
        return closure_in_saturation(
            expr, FRACTURE.pressure, FRACTURE.enthalpy, saturation
        )

    density = closure(SYM.rho)
    balances = {
        "mass": (
            density,
            FRACTURE.mass_flux,
            INTERFACE.mass_flux,
        ),
        "NaCl": (
            density * FRACTURE.composition,
            FRACTURE.component_flux_NaCl,
            INTERFACE.component_flux_NaCl,
        ),
        "energy": (
            closure(SYM.energy_accumulation(sp.Integer(1))),
            FRACTURE.energy_flux,
            INTERFACE.energy_flux,
        ),
    }

    span = np.linspace(constants.y_tip_lo, constants.y_tip_hi, resolution)
    report: dict[str, dict[str, float]] = {}
    for name, (acc, flux, interface_flux) in balances.items():
        terms = {
            "accumulation": sp.diff(nu * acc, t) / nu,
            "tangential": sp.diff(nu * flux, y) / nu,
            "interface": 2 * INTERFACE.specific_volume * interface_flux / nu,
        }
        entry: dict[str, float] = {}
        for term_name, expr in terms.items():
            func = sp.lambdify(FRACTURE_COORD, expr.xreplace(subs), "numpy")
            entry[term_name] = max(
                float(
                    np.abs(
                        np.broadcast_to(
                            np.asarray(func(span, tv), dtype=float), span.shape
                        )
                    ).max()
                )
                for tv in time_levels
            )
        entry["tangential_over_interface"] = (
            entry["tangential"] / entry["interface"]
            if entry["interface"] > 0.0
            else np.inf
        )
        report[name] = entry
    return report


def print_symbolic_sources(
    max_chars: int = 1200, constants: HaliteMMSConstants | None = None
) -> None:
    """Print the manufactured source expressions.

    Parameters:
        max_chars: Truncation length per expression.
        constants: Constants substituted before printing; none if None.

    """
    subs = constants.as_sympy_dict(SYMBOLS) if constants is not None else None

    def show(label: str, expr: sp.Expr) -> None:
        shown = expr.subs(subs) if subs is not None else expr
        text = str(shown)
        suffix = "" if len(text) <= max_chars else f" ... [{len(text)} chars]"
        print(f"\n--- {label}  ({sp.count_ops(shown)} ops) ---")
        print(text[:max_chars] + suffix)

    print("=" * 78)
    print("MATRIX SOURCES, d = 2   (source = d_t A + div F)")
    print("=" * 78)
    for region in REGIONS:
        matrix = MATRIX[region.name]
        print(f"\n### region {region.name!r}, delta = {region.delta}")
        show(f"q_mass[{region.name}]", matrix.source_mass)
        show(f"q_NaCl[{region.name}]", matrix.source_NaCl)
        show(f"q_energy[{region.name}]", matrix.source_energy)

    print("\n" + "=" * 78)
    print("FRACTURE SOURCES, d = 1   (source = [d_t(nu A) + d_y(nu F) - sum_k G_k]/nu)")
    print("=" * 78)
    show("s_hal_l", FRACTURE.halite_saturation)
    show("a_l", FRACTURE.aperture)
    show("p_l", FRACTURE.pressure)
    show("h_l", FRACTURE.enthalpy)
    show("z_l", FRACTURE.composition)
    show("q_mass_l", FRACTURE.source_mass)
    show("q_NaCl_l", FRACTURE.source_NaCl)
    show("q_energy_l", FRACTURE.source_energy)

    print("\n" + "=" * 78)
    print("INTERFACE FLUXES")
    print("=" * 78)
    show("u_k", INTERFACE.darcy_flux)
    show("F_NaCl_k", INTERFACE.component_flux_NaCl)
    show("w_k", INTERFACE.advective_energy_flux)
    show("q_k", INTERFACE.conductive_energy_flux)


class MatrixCallables:
    """Region-dispatching numeric evaluation of the matrix fields and sources."""

    _VECTOR_FIELDS = (
        "darcy_flux",
        "mass_flux",
        "component_flux_NaCl",
        "component_flux_H2O",
        "advective_energy_flux",
        "conductive_energy_flux",
        "energy_flux",
    )

    def __init__(
        self,
        constants: HaliteMMSConstants = MMS_HALITE,
        quantities: Sequence[str] = (
            "pressure",
            "enthalpy",
            "composition",
            "halite_saturation",
            "temperature",
            "porosity",
            "permeability",
            "darcy_flux",
            "component_flux_NaCl",
            "energy_flux",
            "source_mass",
            "source_NaCl",
            "source_energy",
            "two_phase_margin",
        ),
    ) -> None:
        self.constants = constants
        subs = constants.as_sympy_dict(SYMBOLS)
        self._funcs: dict[str, dict[str, Callable]] = {}
        for name in quantities:
            per_region: dict[str, Callable] = {}
            for region in REGIONS:
                expr = getattr(MATRIX[region.name], name)
                if name in self._VECTOR_FIELDS:
                    expr = sp.Matrix([expr[0], expr[1]])
                per_region[region.name] = sp.lambdify(
                    COORD, expr.xreplace(subs), "numpy"
                )
            self._funcs[name] = per_region

    @property
    def names(self) -> tuple[str, ...]:
        """The quantities this instance can evaluate."""
        return tuple(self._funcs)

    def __call__(
        self, name: str, x_v: np.ndarray, y_v: np.ndarray, time_value: float
    ) -> np.ndarray:
        """Evaluate a matrix quantity at the given points.

        Parameters:
            name: One of :attr:`names`.
            x_v: Abscissae of the points.
            y_v: Ordinates of the points.
            time_value: Time level.

        Returns:
            The values, of shape ``(n,)`` for scalars and ``(2, n)`` for vectors.

        Raises:
            KeyError: If the quantity was not requested at construction.

        """
        if name not in self._funcs:
            raise KeyError(
                f"{name!r} was not built; available: {', '.join(self.names)}"
            )
        x_v = np.atleast_1d(np.asarray(x_v, dtype=float))
        y_v = np.atleast_1d(np.asarray(y_v, dtype=float))
        vector = name in self._VECTOR_FIELDS
        out = np.zeros((2, x_v.size)) if vector else np.zeros(x_v.size)
        for region_name, mask in _region_masks(x_v, y_v, self.constants).items():
            if not mask.any():
                continue
            values = self._funcs[name][region_name](x_v[mask], y_v[mask], time_value)
            if vector:
                values = np.asarray(values, dtype=float).reshape(2, -1)
                out[0, mask] = np.broadcast_to(values[0], (int(mask.sum()),))
                out[1, mask] = np.broadcast_to(values[1], (int(mask.sum()),))
            else:
                out[mask] = np.broadcast_to(
                    np.asarray(values, dtype=float), (int(mask.sum()),)
                )
        return out


def make_fracture_callables(
    constants: HaliteMMSConstants = MMS_HALITE,
) -> dict[str, Callable[[np.ndarray, float], np.ndarray]]:
    """Lambdify the fracture and interface quantities.

    Parameters:
        constants: Constants to substitute.

    Returns:
        A callable ``f(y, t)`` per quantity name.

    """
    subs = constants.as_sympy_dict(SYMBOLS)
    expressions: dict[str, sp.Expr] = {
        "pressure": FRACTURE.pressure,
        "enthalpy": FRACTURE.enthalpy,
        "composition": FRACTURE.composition,
        "halite_saturation": FRACTURE.halite_saturation,
        "aperture": FRACTURE.aperture,
        "temperature": FRACTURE.temperature,
        "darcy_flux": FRACTURE.darcy_flux,
        "component_flux_NaCl": FRACTURE.component_flux_NaCl,
        "energy_flux": FRACTURE.energy_flux,
        "source_mass": FRACTURE.source_mass,
        "source_NaCl": FRACTURE.source_NaCl,
        "source_energy": FRACTURE.source_energy,
        "interface_darcy_flux": INTERFACE.darcy_flux,
        "interface_component_flux_NaCl": INTERFACE.component_flux_NaCl,
        "interface_energy_flux": INTERFACE.energy_flux,
        "interface_advective_energy_flux": INTERFACE.advective_energy_flux,
        "interface_conductive_energy_flux": INTERFACE.conductive_energy_flux,
    }
    callables: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {}
    for name, expr in expressions.items():
        raw = sp.lambdify(FRACTURE_COORD, expr.xreplace(subs), "numpy")

        def wrapped(
            y_v: np.ndarray, time_value: float, _raw: Callable = raw
        ) -> np.ndarray:
            y_v = np.atleast_1d(np.asarray(y_v, dtype=float))
            return np.broadcast_to(
                np.asarray(_raw(y_v, time_value), dtype=float), y_v.shape
            ).astype(float, copy=True)

        callables[name] = wrapped
    return callables


if __name__ == "__main__":  # pragma: no cover
    print_symbolic_sources(constants=MMS_HALITE)
