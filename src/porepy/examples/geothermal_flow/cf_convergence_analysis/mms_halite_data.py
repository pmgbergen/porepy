"""Error reporting for the manufactured-solution convergence test: one error column per subdomain and interface quantity."""

from dataclasses import dataclass, field

import numpy as np

from porepy.applications.convergence_analysis import ConvergenceAnalysis


R1_FLUX_BAND = (0.1, 10.0)
"""tuple[float, float]: Admissible ratio of computed to exact interface flux."""


def check_interface_is_exercised(name, reference, approximate, time):
    """Check that an interface flux is non-zero and within :data:`R1_FLUX_BAND` of the exact flux.

    Parameters:
        name: Quantity name, for the failure message.
        reference: Exact flux density on the mortar cells.
        approximate: Computed flux density on the same cells.
        time: Current time, for the failure message.

    Raises:
        AssertionError: If the exact flux vanishes or the ratio falls outside the band.

    """
    exact_scale = float(np.max(np.abs(reference)))
    numerical_scale = float(np.max(np.abs(approximate)))
    if exact_scale <= 0.0:
        raise AssertionError(
            f"R1 failed at t={time}: the exact interface {name} vanishes "
            "identically, so the interface law cannot be verified at all."
        )
    ratio = numerical_scale / exact_scale
    low, high = R1_FLUX_BAND
    if not low <= ratio <= high:
        raise AssertionError(
            f"R1 failed at t={time}: numerical interface {name} is "
            f"{numerical_scale:.4e} against an exact {exact_scale:.4e}, a ratio "
            f"of {ratio:.3e} outside [{low}, {high}]. The interface equation is "
            "not being exercised meaningfully."
        )


@dataclass
class HaliteMMSSaveData:
    """Per-level error columns of the manufactured-solution test."""

    time: float
    cell_diameter: float
    dt: float
    error_matrix_pressure: float
    error_matrix_enthalpy: float
    error_matrix_z_NaCl: float
    error_fracture_pressure: float
    error_fracture_enthalpy: float
    error_fracture_z_NaCl: float
    error_interface_darcy_flux: float
    error_interface_component_flux_NaCl: float
    error_interface_energy_flux: float
    error_interface_advective_energy_flux: float
    error_interface_conductive_energy_flux: float
    error_matrix_halite_saturation: float
    error_fracture_halite_saturation: float
    error_fracture_aperture: float
    interface_sides: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)


def relative_error(grid, exact, approximate, parameter_weight=None, is_relative=True):
    """Discrete :math:`l_2` error, computed with ``ConvergenceAnalysis.lp_norm``.

    Parameters:
        grid: Subdomain or mortar grid of the values.
        exact: Exact values.
        approximate: Computed values.
        parameter_weight: Optional weight on the measure, e.g. the specific volume.
        is_relative: Divide by the norm of the exact values; False gives the absolute error.

    Returns:
        The error.

    Raises:
        ValueError: If any input is not finite.

    """
    for values in (exact, approximate):
        if not np.all(np.isfinite(values)):
            raise ValueError("Non-finite input to the MMS error norm")

    measure = np.asarray(grid.cell_volumes, dtype=float).copy()
    if parameter_weight is not None:
        measure *= np.asarray(parameter_weight, dtype=float)

    difference = np.asarray(exact, dtype=float) - np.asarray(approximate, dtype=float)
    numerator = ConvergenceAnalysis.lp_norm(
        difference, integration_weights=measure, p=2
    )
    if not is_relative:
        return float(numerator)

    denominator = ConvergenceAnalysis.lp_norm(
        np.asarray(exact, dtype=float), integration_weights=measure, p=2
    )

    if np.isclose(denominator, 0.0):
        raise ZeroDivisionError(
            f"Zero exact norm on a dimension-{grid.dim} grid: this quantity "
            "cannot be measured relatively. Report it as an absolute error."
        )
    return float(numerator / denominator)


class HaliteMMSDataSaving:
    """Report cell fields and flux densities on each mortar side independently."""

    def collect_data(self):
        time = self.time_manager.time
        if time <= 0:
            return None
        errors = {}
        stored = {}
        evaluate = self.equation_system.evaluate
        for sd in self.mdg.subdomains():
            prefix = "matrix" if sd.dim == self.nd else "fracture"
            # The measure is the cell volume, weighted by the specific volume on
            # the fracture: nu_1 = a_l. The matrix has nu_2 = 1, so no weight.
            parameter_weight = None
            if sd.dim < self.nd:
                parameter_weight = self.exact_values(sd, "aperture", time)
            fields = {
                "pressure": self.pressure([sd]),
                "enthalpy": self.enthalpy([sd]),
                "z_NaCl": self.fluid.components[1].fraction([sd]),
                "halite_saturation": self.fluid.phases[1].saturation([sd]),
            }

            if sd.dim == 1:
                fields["aperture"] = self.aperture([sd])

            for name, operator in fields.items():
                key = "composition" if name == "z_NaCl" else name
                is_relative = name != "aperture"
                weight = None if name == "aperture" else parameter_weight
                reference = self.exact_values(sd, key, time)
                approximate = evaluate(operator)
                errors[f"error_{prefix}_{name}"] = relative_error(
                    sd, reference, approximate, weight, is_relative=is_relative
                )
                stored[f"{prefix}_{name}"] = {
                    "exact": reference,
                    "approximate": approximate,
                }

        sides = {}
        quantities = (
            "darcy_flux",
            "component_flux_NaCl",
            "advective_energy_flux",
            "conductive_energy_flux",
            "energy_flux",
        )
        pooled = {
            name: {"reference": [], "approximate": [], "measures": []}
            for name in quantities
        }
        interfaces = self.mdg.interfaces()

        for index, interface in enumerate(interfaces):
            operators = {
                "darcy_flux": self.interface_darcy_flux([interface]),
                "component_flux_NaCl": self.interface_component_flux(
                    self.fluid.components[1], [interface]
                ),
                "advective_energy_flux": self.interface_enthalpy_flux([interface]),
                "conductive_energy_flux": self.interface_fourier_flux([interface]),
                "energy_flux": self.interface_enthalpy_flux([interface])
                + self.interface_fourier_flux([interface]),
            }
            # The mortar flux is stored volume-integrated; divide by the cell
            # volumes to obtain the flux density the exact solution provides.
            measures = interface.cell_volumes
            offset = 0
            side_slices = []
            for side, grid in interface.side_grids.items():
                side_slices.append(
                    (str(side), grid, slice(offset, offset + grid.num_cells))
                )
                offset += grid.num_cells

            for name, operator in operators.items():
                reference = self.fracture_exact[f"interface_{name}"](
                    interface.cell_centers[1], time
                )
                approximate = evaluate(operator) / measures
                check_interface_is_exercised(name, reference, approximate, time)

                pooled[name]["reference"].append(reference)
                pooled[name]["approximate"].append(approximate)
                pooled[name]["measures"].append(measures)

                for side, side_grid, selection in side_slices:
                    # Qualify the side by interface index when there is more than
                    # one, so a second fracture cannot overwrite the first.
                    label = side if len(interfaces) == 1 else f"{side}_{index}"
                    sides.setdefault(label, {})[name] = relative_error(
                        side_grid, reference[selection], approximate[selection]
                    )

        for name, parts in pooled.items():
            # One error per quantity, over the mortar cells of every interface.
            reference = np.concatenate(parts["reference"])
            approximate = np.concatenate(parts["approximate"])
            measures = np.concatenate(parts["measures"])
            errors[f"error_interface_{name}"] = float(
                ConvergenceAnalysis.lp_norm(
                    reference - approximate, integration_weights=measures, p=2
                )
                / ConvergenceAnalysis.lp_norm(
                    reference, integration_weights=measures, p=2
                )
            )
            stored[f"interface_{name}"] = {
                "exact": reference,
                "approximate": approximate,
            }

        return HaliteMMSSaveData(
            time=time,
            cell_diameter=self.mdg.diameter(),
            dt=self.time_manager.dt,
            interface_sides=sides,
            values=stored,
            **errors,
        )
