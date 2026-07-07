# Model definition.
from typing import Callable
import porepy as pp
import numpy as np
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.applications.material_values.fluid_values import (
    extended_water_values_for_testing as water,
)
from porepy.applications.material_values.solid_values import (
    extended_granite_values_for_testing as granite,
)
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMechanicsNeumann,
    LithostaticBoundaryStressValues,
    HydrostaticBoundaryPressureValues,
)
from porepy.applications.initial_conditions.model_initial_conditions import (
    InitialConditionHydrostaticPressureValues,
)
# from porepy.models.contact_mechanics import RadialReturnTangentialContactMechanicsEquation

# Set up logging. Uncomment for debugging purposes. Note that this will produce a lot of
# output.
# import logging
# logging.basicConfig(level=logging.INFO)


# --- Geometry modification ---
class DiagonalLineFractureGeometry:
    """Simple 2D square domain with a single diagonal fracture."""
    units: pp.Units

    def set_domain(self) -> None:
        """Define 2D square domain of size 100 m x 100 m."""
        size = self.units.convert_units(100.0, "m")
        self._domain = nd_cube_domain(2, size)

    def set_fractures(self) -> None:
        """Define a single diagonal fracture from (40, 40) to (60, 60) in the domain."""
        frac_1_points = self.units.convert_units(
            np.array([[40.0, 60.0], [40.0, 60.0]]), "m"
        )
        frac_1 = pp.LineFracture(frac_1_points)
        self._fractures = [frac_1]

    def grid_type(self) -> str:
        # Use simplex grid (non-Cartesian due to fracture).
        return "simplex"

    def meshing_arguments(self) -> dict:
        """Set mesh cell size.

        Returns:
            Dictionary with meshing arguments.
        """
        cell_size = self.units.convert_units(10, "m")
        return {"cell_size": cell_size}


# --- Lithostatic initial and boundary conditions ---

# TODO: Integrate in the core module and extend to cover 2D and 3D?
# TODO: Consider plane-strain and plane-stress assumptions for 2D problems.


class BoundaryConditionsMechanicsNeumann2D(BoundaryConditionsMechanicsNeumann):
    """Two-dimensional variant of BoundaryConditionsMechanicsNeumann.

    Boundary conditions for the mechanics with Neumann conditions on almost all
    boundaries.

    The only exception is that internal boundaries are converted to Dirichlet and three
    points are partly fixed to avoid rigid body motions.

    We pick the following points:
            1) min x and mean y coordinate, (fixed in y direction)
            2) max x and mean y coordinate, (fixed in y direction)
            3) mean x and min y coordinate. (fixed in x direction)

    """

    domain: pp.domain.Domain
    """Model domain."""
    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]
    """Function returning the domain boundary sides of a given grid."""
    nd: int
    """Number of spatial dimensions."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for mechanics.

        Neumann boundary conditions are defined on all boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.all_bf, "neu")
        if sd.dim < self.nd:
            # No displacement is implemented on grids of co-dimension >= 1.
            return bc
        bc.internal_to_dirichlet(sd)
        faces_to_fix = self.faces_to_fix(sd)
        # Fix y displacements on face 1 & 3, x displacements on face 2.
        dir = [
            np.array([False, True]),  # Fix y on face 1 (west)
            np.array([True, False]),  # Fix x on face 2 (south)
            np.array([False, True]),  # Fix y on face 3 (east)
        ]
        for i, face in enumerate(faces_to_fix):
            bc.is_dir[:, face] = dir[i]
            bc.is_neu[:, face] = ~dir[i]  # Negate for Neumann
        return bc

    def faces_to_fix(self, sd: pp.Grid) -> list[np.int64]:
        """Return list of faces to fix to avoid rigid body motions.

        See class documentation for more details on the choice of points.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            List of face indices to fix, ordered as in the class documentation.

        """
        domain_sides = self.domain_boundary_sides(sd)
        box = self.domain.bounding_box

        # Mean x and y coordinates.
        x_mean = 0.5 * (box["xmax"] + box["xmin"])
        y_mean = 0.5 * (box["ymax"] + box["ymin"])

        # Point 1: center on west boundary.
        point_1 = np.array([box["xmin"], y_mean])
        pts = sd.face_centers[:2, domain_sides.west]
        ind_1 = domain_sides.west.nonzero()[0][
            np.argmin(pp.distances.point_pointset(point_1, pts))
        ]

        # Point 2: center of the south boundary.
        point_2 = np.array([x_mean, box["ymin"]])
        pts = sd.face_centers[:2, domain_sides.south]
        ind_2 = domain_sides.south.nonzero()[0][
            np.argmin(pp.distances.point_pointset(point_2, pts))
        ]

        # Point 3: center of the east boundary.
        point_3 = np.array([box["xmax"], y_mean])
        pts = sd.face_centers[:2, domain_sides.east]
        ind_3 = domain_sides.east.nonzero()[0][
            np.argmin(pp.distances.point_pointset(point_3, pts))
        ]

        indices = [ind_1, ind_2, ind_3]

        return indices


class LithostaticBoundaryStressValues2D(LithostaticBoundaryStressValues):
    """Two-dimensional variant of LithostaticBoundaryStressValues.

    Boundary conditions for the mechanics with lithostatic stress on all boundaries.

    Neumann boundary conditions are defined on all boundaries. Zero stress is assumed at
    time zero. This corresponds to an initial stress-free state, and presumably to zero
    initial displacement as well. For positive times, lithostatic stress is applied
    according to the depth of the boundary faces. The principal stresses are assumed to
    align with the coordinate axes, and the relative magnitudes can be adjusted through
    the parameter "lithostatic_stress_multipliers", which should be an array of three
    values. The default is an array of ones, corresponding to equal stresses in all
    directions.

    """

    params: dict
    """Model parameters."""
    depth: Callable[[np.ndarray], np.ndarray]
    """Function to compute depth of points."""
    equation_system: pp.EquationSystem
    """Equation system associated with the model."""
    gravity_force: Callable[[pp.GridLikeSequence], pp.ad.Operator]
    """Function to compute gravity force."""
    domain_boundary_sides: Callable[[pp.GridLike], pp.domain.DomainSides]
    """Function returning the domain boundary sides of a given grid."""
    time_manager: pp.TimeManager
    """Time manager associated with the model."""

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Stress values.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """
        # Assume zero initial stress state.
        if (
            self.time_manager.time
            < self.time_manager.time_init + 0.5 * self.time_manager.dt_min_max[0]
        ):
            # Identify dimension of the problem from the lithostatic stress multipliers.
            dim = len(self.lithostatic_stress_multipliers)
            # Initialize array for stress values.
            values = np.zeros((dim, boundary_grid.num_cells))
            return values.ravel("F")
        else:
            return self.active_bc_values_stress(boundary_grid)

    def active_bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Stress values.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """
        # Identify dimension of the problem from the lithostatic stress multipliers.
        dim = len(self.lithostatic_stress_multipliers)

        # Initialize array for stress values.
        values = np.zeros((dim, boundary_grid.num_cells))

        gravity = self.gravity_force_magnitude("bulk")
        # Multiply with lithostatic stress multipliers, which can be used to set
        # different stresses in different directions.
        gradient = self.lithostatic_stress_multipliers * gravity
        # Allow for explicit offset (typically lithostatic stress multiplier times
        # effective overburden stress at surface of the domain).
        offset = self.lithostatic_stress_offset
        # Get domain sides and depth at boundary cell centers.
        domain_sides = self.domain_boundary_sides(boundary_grid)
        depth = self.depth(boundary_grid.cell_centers)

        # The sign of the stress depends on the side of the domain according to the
        # direction of the outer normal vector on the boundary. Loop over directions.
        for i, sides in enumerate([["west", "east"], ["south", "north"]]):
            # Apply stress on both sides of the domain in direction i.
            for side, sign in zip(sides, [1, -1]):
                # Get indices of faces on the given side.
                ind = getattr(domain_sides, side)
                if np.any(ind):
                    # Set ith component of stress on these faces.
                    values[i, ind] = (
                        (offset[i] + gradient[i] * depth[ind])
                        * sign
                        * boundary_grid.cell_volumes[ind]
                    )

        return values.ravel("F")


class MatplotlibExportingPoromechanics:
    """Auxiliary class to enable visualization in this notebook."""
    displacement_variable: str
    mdg: pp.MixedDimensionalGrid
    pressure_variable: str
    time_manager: pp.TimeManager

    def plot_model(self) -> None:
        """Plot pressure and displacement distribution of the model."""
        pp.plot_grid(
            self.mdg,
            cell_value=self.pressure_variable,
            vector_value=self.displacement_variable,
            figsize=(10, 8),
            linewidth=0.25,
            title=(
                """Pressure and displacement distribution at """
                f"""time {self.time_manager.time // 3600} hrs"""
            ),
            plot_2d=True,
            color_map_limits=[0, 0.00012],
        )

    def after_time_step_convergence(self) -> None:
        """Export results after each time step convergence."""
        self.plot_model()
        super().after_time_step_convergence()  # type:ignore[safe-super]


class ModelWithInitialization(
    # Our fractured domain
    DiagonalLineFractureGeometry,
    # Initial conditions for flow
    InitialConditionHydrostaticPressureValues,
    # Boundary conditions for flow
    HydrostaticBoundaryPressureValues,
    # Boundary conditions for mechanics
    BoundaryConditionsMechanicsNeumann2D,
    LithostaticBoundaryStressValues2D,
    MatplotlibExportingPoromechanics,
    # Base model
    pp.constitutive_laws.CharacteristicDisplacementFromTraction,
    pp.constitutive_laws.GravityForce,
    pp.constitutive_laws.CubicLawPermeability,
    pp.Poromechanics,
): ...


def set_model_parameters() -> dict:
    # --- Material properties (fetch realistic values) ---
    fluid_constants = pp.FluidComponent(**water)
    solid_constants = pp.SolidConstants(**granite)
    numerical_constants = pp.NumericalConstants(
        **{"characteristic_contact_traction": 10 * 3000 * 9.81}
    )
    material_constants = {
        "fluid": fluid_constants,
        "solid": solid_constants,
        "numerical": numerical_constants,
    }

    # --- Time stepping (single short time step) ---
    time_manager = pp.TimeManager(
        schedule=[0, 5 * 1000 * pp.YEAR],
        dt_init=1000 * pp.YEAR,
        constant_dt=True,
        iter_max=10,
        print_info=True,
    )

    # --- Units for scaling ---
    units = pp.Units(kg=1e10)  # Avoid numerical issues by scaling units.

    # --- Summarize model parameters ---

    # Apply 80% horizontal stress and 100% vertical stress, corresponding to a normal
    # faulting regime. Define offsets corresponding to a domain that is 1000 m below
    # the surface.
    depth = 1000  # Depth in meters
    rho_fluid = 1000  # Fluid density in kg/m^3
    rho_bulk = 2500  # Bulk density in kg/m^3
    hydrostatic_pressure_offset = depth * rho_fluid * 9.81
    lithostatic_stress_multipliers = np.array([0.8, 1])
    lithostatic_stress_offset = depth * rho_bulk * 9.81 * lithostatic_stress_multipliers

    model_params = {
        "material_constants": material_constants,
        "time_manager": time_manager,
        "units": units,
        "hydrostatic_pressure_datum": hydrostatic_pressure_offset,
        "lithostatic_stress_offset": lithostatic_stress_offset,
        "lithostatic_stress_multipliers": lithostatic_stress_multipliers,
    }
    return model_params

def set_solver_parameters(model) -> dict:

    solver_params = {
        "nl_convergence_criteria": {
            "inc_rel": pp.IncrementBasedRelativeCriterion(
                tol=1e-6, metric=pp.VariableBasedEuclideanMetric(model)
            ),
            "res_rel": pp.ResidualBasedRelativeCriterion(
                tol=1e-6, metric=pp.EquationBasedEuclideanMetric(model)
            ),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.MaxIterationsCriterion(max_iterations=25),
            "inc_nan": pp.IncrementBasedNanCriterion(),
            "res_nan": pp.ResidualBasedNanCriterion(),
        },
    }
    return solver_params