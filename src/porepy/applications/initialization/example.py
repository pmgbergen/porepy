"""Example of the initialization of a 3d thermo-poromechanics model of a fractured
reservoir to a steady state, followed by a simulation from that state.

The model has a horizontal fracture, diagonal fractures and a well. Lithostatic stress,
hydrostatic pressure and a thermal gradient are applied on the boundaries. The
initialization finds the state in which the model is in equilibrium with them, see
:mod:`porepy.applications.initialization.initialization` for what it does and what it
assumes about the model. The result is saved in ``<folder_name>/steady_state.npz``, and
is reused if the script is run again with the same model.

The script needs the package ``pp_solvers`` for the linear solver. Run it with::

    python -m porepy.applications.initialization.example

"""

import logging
from typing import Callable

import numpy as np
import pp_solvers

import porepy as pp
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMechanicsNeumann,
    HydrostaticBoundaryPressureValues,
    LithostaticBoundaryStressValues,
    ThermalGradientBoundaryTemperatureValues,
)
from porepy.applications.initial_conditions.model_initial_conditions import (
    InitialConditionHydrostaticPressureValues,
    InitialConditionThermalGradientTemperatureValues,
)
from porepy.applications.initialization.initialization import default_rate_tolerances, default_residual_tolerances, initialization_pipeline
from porepy.applications.md_grids.model_geometries import SubsurfaceCuboidDomain

logger = logging.getLogger(__name__)


class FracturedReservoirGeometry(SubsurfaceCuboidDomain):
    """Geometry mixin defining a horizontal fracture crossed by several diagonal
    fractures, and a single vertical well penetrating the horizontal fracture.

    All parameters are read from ``self.params["fracture_params"]``, see
    :meth:`fracture_params` for the available keys and their defaults.

    """

    def fracture_params(self) -> dict:
        """Return fracture/well geometry parameters with defaults.

        The available parameters are:
            - num_diagonal_fractures: Number of diagonal fractures (default 4). This
              is the "controlled parameter" for the number of diagonal fractures.
            - diagonal_fracture_dip_angle: Dip angle [rad] of the diagonal fractures
              relative to the horizontal fracture (default pi/3, i.e. 60 degrees).
              This is the "adjustable parameter" for the diagonal fracture angle.
            - horizontal_fracture_extent: Side length of the (square) horizontal
              fracture, as a fraction of min(domain x-size, domain y-size) (default
              0.8).
            - horizontal_fracture_depth_fraction: Depth of the horizontal fracture,
              as a fraction of the domain z-size (default 0.5, i.e. mid-depth).
            - diagonal_fracture_half_height: Vertical half-extent of each diagonal
              fracture, as a fraction of the domain z-size (default 0.3). With the
              default mid-depth horizontal fracture, the diagonal fractures span
              0.2 to 0.8 of the domain z-size.
            - diagonal_fracture_half_width: Half-width of each diagonal fracture
              along the (horizontal) strike direction, as a fraction of the domain
              y-size (default 0.3).
            - well_penetration_fraction: How far past the horizontal fracture depth
              the well extends, as a fraction of the domain z-size (default 0.15),
              ensuring a robust (non-degenerate) well-fracture intersection.

        Returns:
            A dictionary with fracture and well geometry parameters.

        """
        default_params = {
            "num_diagonal_fractures": 4,
            "diagonal_fracture_dip_angle": np.pi / 3,
            "horizontal_fracture_extent": 0.8,
            "horizontal_fracture_depth_fraction": 0.5,
            "diagonal_fracture_half_height": 0.3,
            "diagonal_fracture_half_width": 0.3,
            "well_penetration_fraction": 0.15,
        }
        user_params = self.params.get("fracture_params", {})
        default_params.update(user_params)
        return default_params

    def set_fractures(self) -> None:
        """Set one horizontal fracture and several intersecting diagonal fractures."""
        params = self.fracture_params()
        dx, dy, dz = self.domain_sizes()
        cx, cy = 0.5 * dx, 0.5 * dy
        z_frac = -params["horizontal_fracture_depth_fraction"] * dz

        # Horizontal fracture: a square centered at (cx, cy, z_frac).
        half_size = 0.5 * params["horizontal_fracture_extent"] * min(dx, dy)
        horizontal_pts = np.array(
            [
                [cx - half_size, cx + half_size, cx + half_size, cx - half_size],
                [cy - half_size, cy - half_size, cy + half_size, cy + half_size],
                [z_frac, z_frac, z_frac, z_frac],
            ]
        )
        fractures: list[pp.PlaneFracture] = [pp.PlaneFracture(horizontal_pts, index=0)]

        # Diagonal fractures: rectangles invariant in y, tilted by
        # diagonal_fracture_dip_angle in the x-z plane, each centered on the
        # horizontal fracture's plane so that it is guaranteed to intersect it.
        num_diagonal = params["num_diagonal_fractures"]
        theta = params["diagonal_fracture_dip_angle"]
        half_width = params["diagonal_fracture_half_width"] * dy
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Evenly spaced x-centers across the horizontal fracture's x-extent, using a
        # half-step offset so that, for the default even fracture count, no fracture
        # is centered exactly below the well (which sits at the domain center).
        x_min_frac, x_max_frac = cx - half_size, cx + half_size
        fractions = (np.arange(num_diagonal) + 0.5) / max(num_diagonal, 1)
        x_centers = x_min_frac + fractions * (x_max_frac - x_min_frac)

        # Half-length in the dip direction chosen such that the vertical (z) extent
        # of each fracture is +- diagonal_fracture_half_height * dz around z_frac.
        half_length = params["diagonal_fracture_half_height"] * dz / max(sin_t, 0.05)
        for i, x_i in enumerate(x_centers):
            # Local rectangle corners in (u, v), u along the dip direction, v along
            # the (horizontal) strike direction.
            u = np.array([-half_length, half_length, half_length, -half_length])
            v = np.array([-half_width, -half_width, half_width, half_width])
            diagonal_pts = np.array(
                [
                    x_i + u * cos_t,
                    cy + v,
                    z_frac + u * sin_t,
                ]
            )
            fractures.append(pp.PlaneFracture(diagonal_pts, index=i + 1))

        self._fractures = fractures

    def set_wells(self) -> None:
        """Set a single vertical well penetrating the horizontal fracture.

        The well carries no injection/production protocol; it is purely a
        geometric/hydraulic feature of the domain.
        """
        params = self.fracture_params()
        dx, dy, dz = self.domain_sizes()
        cx, cy = 0.5 * dx, 0.5 * dy
        z_frac = -params["horizontal_fracture_depth_fraction"] * dz
        z_bottom = z_frac - params["well_penetration_fraction"] * dz

        well = pp.Well(
            np.array([[cx, cx], [cy, cy], [0.0, z_bottom]]),
            tags={"well_name": "observation_well"},
        )
        self._wells = [well]

    def well_meshing_arguments(self) -> dict:
        *_, dz = self.domain_sizes()
        return self.params.get("well_meshing_arguments", {"cell_size": 0.025 * dz})

    def grid_type(self) -> str:
        """3D fracture and well meshing requires a simplex grid."""
        return "simplex"


class WellZeroNeumannTemperatureBC:
    """Overrides the well's temperature boundary conditions to be homogeneous Neumann
    (zero diffusive and advective flux), instead of the Dirichlet conditions the well
    otherwise inherits (at its top face) from the rest of the domain boundary.

    """

    domain_boundary_sides: Callable[[pp.Grid], pp.domain.DomainSides]
    """Function returning the domain boundary sides of a given grid."""
    is_well_grid: Callable[[pp.Grid], bool]
    """Function checking whether a subdomain grid is a well."""

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if self.is_well_grid(sd):
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "neu")
        return super().bc_type_fourier_flux(sd)  # type: ignore[misc]

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if self.is_well_grid(sd):
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "neu")
        return super().bc_type_enthalpy_flux(sd)  # type: ignore[misc]

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if self.is_well_grid(sd):
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "neu")
        return super().bc_type_fluid_flux(sd)  # type: ignore[misc]

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if self.is_well_grid(sd):
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "neu")
        return super().bc_type_darcy_flux(sd)  # type: ignore[misc]


class ThmFracturedReservoir(  # type: ignore[misc]
    pp.constitutive_laws.GravityForce,
    pp.constitutive_laws.CubicLawPermeability,
    WellZeroNeumannTemperatureBC,
    HydrostaticBoundaryPressureValues,
    ThermalGradientBoundaryTemperatureValues,
    BoundaryConditionsMechanicsNeumann,
    LithostaticBoundaryStressValues,
    InitialConditionHydrostaticPressureValues,
    InitialConditionThermalGradientTemperatureValues,
    FracturedReservoirGeometry,
    pp.models.solution_strategy.ContactIndicators,
    pp.Thermoporomechanics,
):
    """3D THM model with a horizontal fracture crossed by several diagonal
    fractures, and a single, disabled vertical well penetrating the horizontal
    fracture.

    Lithostatic mechanical and hydrostatic/thermal-gradient fluid/heat boundary
    conditions are applied on all external boundaries, with gravity enabled.
    """


def set_model_params() -> dict:
    solid_values = pp.solid_values.granite.copy()
    solid_values.update(
        {
            "normal_permeability": 1.0e-14,  # [m^2]
            "residual_aperture": 1e-4,  # [m]
            "well_radius": 0.1,  # [m]
            # YZ: increase diffusion to speed up equilibrating????
            # "thermal_conductivity": 1e3,
        }
    )
    fluid_values = pp.fluid_values.water.copy()
    fluid_values.update(
        {
            # YZ: increase diffusion to speed up equilibrating????
            # "thermal_conductivity": 1e3,
        }
    )

    # Domain size (x, y, z).
    domain_sizes = np.array([1000.0, 1000.0, 1000.0])  # [m]

    return {
        "time_manager": pp.TimeManager(
            pp.time_stepper.Schedule(
                intervals=[
                    pp.time_stepper.TimeInterval.create(
                        t_start=0,
                        dt_start=pp.DAY,
                        constraints=[pp.time_stepper.TargetNonlinearIterations()],
                        name="injection",
                    ),
                ],
                t_end=31 * pp.DAY,
            )
        ),
        "lithostatic_stress_multipliers": np.array([0.8, 1.2, 1.0]),
        "material_constants": {
            "solid": pp.SolidConstants(**solid_values),  # type: ignore[arg-type]
            "fluid": pp.FluidComponent(**fluid_values),  # type: ignore[arg-type]
            "numerical": pp.NumericalConstants(characteristic_displacement=1e-2),
        },
        "datum_pressure": 1e6,
        "reference_variable_values": pp.ReferenceVariableValues(
            temperature=300.0, pressure=1e6
        ),  # type: ignore[arg-type]
        "units": pp.Units(m=1.0, kg=1.0e9, K=1.0),
        "grid_type": "simplex",
        "meshing_arguments": {
            "cell_size": domain_sizes[2] / 5.0,
            "cell_size_fracture": domain_sizes[2] / 5.0,
        },
        "well_meshing_arguments": {
            "cell_size": domain_sizes[2] / 5.0,
        },
        "fracture_params": {
            # Controlled parameter: number of diagonal fractures.
            "num_diagonal_fractures": 1,  # <-------------------------------------------
            # Adjustable parameter: dip angle [rad] of diagonal fractures relative to
            # the horizontal fracture.
            "diagonal_fracture_dip_angle": np.pi / 10,
        },
        "domain_sizes": domain_sizes,
        "adaptive_indicator_scaling": 1,
        "folder_name": "thm_fractured_reservoir",
        "initialize_operator_reference_from_initial_values": True,
    }


def set_solver_params() -> dict:
    return {
        "nl_max_iterations": 25,
        "nl_convergence_inc_atol": 1e-7,
        "nl_convergence_res_atol": 1e-7,
        "nl_divergence_inc_atol": 1e12,
        "nl_divergence_res_atol": 1e12,
        "nonlinear_solver": pp.solvers.ConstraintLineSearchNonlinearSolver,
        "global_line_search": 1,
        "local_line_search": 0,
    }


def make_solver_params() -> dict:
    return {
        "nl_max_iterations": 25,
        "nl_convergence_inc_atol": 1e-7,
        "nl_convergence_res_atol": 1e-7,
        "nl_divergence_inc_atol": 1e12,
        "nl_divergence_res_atol": 1e12,
        "nonlinear_solver": pp.solvers.ConstraintLineSearchNonlinearSolver,
        "global_line_search": 1,
        "local_line_search": 0,
    }


def make_nonlinear_solver() -> pp.solvers.NewtonSolver:
    return pp.solvers.NewtonSolver(
        params=make_solver_params(),
        linear_solver=pp_solvers.IterativeLinearSolver(),
    )


def run_main_simulation(model: pp.PorePyModel):
    runner = pp.ModelRunner(
        model=model,
        params={"prepare_simulation": False},
        nonlinear_solver=make_nonlinear_solver(),
    )
    status = runner.run()
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("porepy.numerics.solvers.newton_solver").setLevel(logging.WARNING)
    logging.getLogger("porepy.numerics.solvers.linear_solvers.linear_solver").setLevel(
        logging.WARNING
    )
    logging.getLogger("pp_solvers.porepy_integration").setLevel(logging.WARNING)
    model = ThmFracturedReservoir(set_model_params())

    model = initialization_pipeline(
        model,
        rate_tolerances=default_rate_tolerances(model),
        residual_tolerances=default_residual_tolerances(),
    )

    run_main_simulation(model)

    pass
