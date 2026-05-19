import copy
from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np

import porepy as pp
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMechanicsDirNorthSouth,
)
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils.models import add_mixin
from porepy.compositional.materials import FractureDamageSolidConstants
from porepy.models import fracture_damage as damage
from porepy.numerics.nonlinear.line_search import ConstraintLineSearchNonlinearSolver


class TimeDependentDamageBCs:
    """Model mixin for time-dependent boundary conditions for fracture damage models.

    Defines time-dependent displacement values on all faces satisfying x[1] > 0.5. The
    time dependence is defined by the parameter "north_displacements" passed on model
    initialization.
    """

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary values for the mechanics problem as a numpy array.

        Values for the north boundaries are retrieved from the parameter dictionary
        passed on model initialization. The values are time dependent and are retrieved
        from the parameter dictionary using the key "north_displacements" and indexed in
        the second dimension by the current time index.

        Parameters:
            bg: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                domain, for each face in the subdomain.

        """
        values = np.zeros((self.nd, bg.num_cells))
        if bg.dim < self.nd - 1:
            # No displacement is implemented on grids of co-dimension >= 2.
            return values.ravel("F")

        # Wrap as array for convert_units. Thus, the passed values can be scalar or
        # list. Then tile for correct broadcasting below.
        u_north = self.params["north_displacements"][:, self.time_manager.time_index]
        u_n = np.tile(u_north, (bg.num_cells, 1)).T
        north_sides = bg.cell_centers[1] > 0.5
        values[:, north_sides] = self.units.convert_units(u_n, "m")[:, north_sides]
        return values.ravel("F")


DATA_SAVING_METHOD_NAMES = [
    "normalized_traction_for_damage",
    "damage_length",
    "dilation_damage_state",
    "dilation_damage_evolution_coefficient",
    "dilation_damage_history",
    "friction_damage_state",
    "friction_damage_evolution_coefficient",
    "friction_damage_history",
]


def make_damagesavedata_class(method_names: list[str]) -> type:
    """Create a dataclass type with fields for exact/approx values and errors."""
    annotations: dict[str, type] = {}
    namespace: dict[str, object] = {"__annotations__": annotations}

    for name in method_names:
        annotations[f"exact_{name}"] = np.ndarray
        annotations[f"approx_{name}"] = np.ndarray

    cls = type("DamageSaveData", (object,), namespace)
    return dataclass(cls)


DamageSaveData = make_damagesavedata_class(DATA_SAVING_METHOD_NAMES)


class DamageDataSaving(pp.PorePyModel):
    """Model mixin responsible for saving data for verification purposes."""

    damage_length: Callable[[list[pp.Grid], int], tuple[pp.ad.Operator, pp.ad.Operator]]
    """Damage length operator."""
    dilation_damage_state: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Dilation damage operator."""
    dilation_damage_history: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Dilation damage history."""
    friction_damage_state: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Friction damage operator."""
    friction_damage_history: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Friction damage history."""
    solid: FractureDamageSolidConstants

    def initialize_data_saving(self) -> None:
        """Set material parameters.

        Add exact solution object to the simulation model after materials have been set.

        """
        super().initialize_data_saving()  # type: ignore[safe-super]
        self.exact_sol: ExactSolution = self.params["exact_solution"](self)

    def collect_data(self) -> Any:
        """Collect the data from the verification setup.

        Returns:
            DamageSaveData object containing the results of the verification for the
            current time.

        """
        # Retrieve information from setup.
        sds = self.mdg.subdomains(dim=self.nd - 1)
        sd = sds[0]
        n: int = self.time_manager.time_index
        names = DATA_SAVING_METHOD_NAMES
        vals = {}
        for name in names:
            if name == "damage_length":
                # Treat damage length as a special case because of signature
                exact_val = cast(np.ndarray, self.exact_sol.damage_length(sd, n, n))
                # Since we have already updated the solution, time_step_index=1 gives
                # the most recent increment.
                length, _ = self.damage_length(sds, 1)
                approx_val = cast(np.ndarray, length.value(self.equation_system))
            else:
                if hasattr(self, name):
                    # Collect data.
                    exact_val = cast(np.ndarray, getattr(self.exact_sol, name)(sd, n))
                    approx_val = cast(
                        np.ndarray, getattr(self, name)(sds).value(self.equation_system)
                    )
                else:
                    # By setting different values, we ensure that the error is large if
                    # the lack of the method masks some other error.
                    exact_val = np.ones(sd.num_cells)
                    approx_val = np.zeros_like(exact_val)

            vals["exact_" + name] = exact_val
            vals["approx_" + name] = approx_val
        collected_data = DamageSaveData(**vals)
        return collected_data


class FractureDamageMomentumBalance(  # type: ignore[misc]
    pp.models.solution_strategy.ContactIndicators,
    DamageDataSaving,
    pp.constitutive_laws.FractureDamageEvolutionCoefficients,
    TimeDependentDamageBCs,
    pp.MomentumBalance,
):
    """Fracture damage momentum balance model.

    This model combines fracture damage mechanics with momentum balance and force
    balance across interfaces. Variables are matrix and interface displacements, contact
    traction, and damage. The model is isotropic, i.e., the damage is independent of the
    loading direction.

    Also contains specifics defining a test case in terms of the boundary conditions.

    """


class DilationDamageMixin(
    pp.constitutive_laws.DilationDamage,
    damage.DilationDamageEquation,
    damage.DilationDamageVariable,
):
    """Fracture damage model with dilation damage.

    To be used as a mixin for the momentum balance model and isotropic or anisotropic
    damage models when dilation damage is activated. Can be used on its own or together
    with friction damage.
    """

    pass


class FrictionDamageMixin(
    pp.constitutive_laws.FrictionDamage,
    damage.FrictionDamageEquation,
    damage.FrictionDamageVariable,
):
    """Fracture damage model with friction damage.

    To be used as a mixin for the momentum balance model and isotropic or anisotropic
    damage models when friction damage is activated. Can be used on its own or together
    with dilation damage.
    """

    pass


# Collect the damage types in a dictionary for easy access when building models with
# different regimes.
damage_types = {
    "dilation": DilationDamageMixin,
    "friction": FrictionDamageMixin,
}


class ExactSolution:
    """Exact solution for the damage model.

    The driving force of the problem is assumed to be a Dirichlet boundary condition
    with transient values defined in the class parameters.

    """

    params: dict

    model: FractureDamageMomentumBalance

    damage_length: Callable[[pp.Grid, int, int], np.ndarray]

    def __init__(self, model) -> None:
        """Constructor of the class."""
        self.model = model

    def boundary_displacement(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of boundary displacements for the given time step. Shape is
            (nd - 1, num_cells), where nd is the number of dimensions of the model.
        """
        num_cells = sd.num_cells
        if self.model.nd == 3:
            inds: np.ndarray = np.array([0, 2])
        else:
            inds = np.array([0])
        displacements = cast(np.ndarray, self.model.params["north_displacements"])
        u = displacements[inds, n]
        # u shape: (nd-1,)
        # Tile to (nd-1, num_cells)
        u_tiled = np.tile(u[:, np.newaxis], (1, num_cells))
        return u_tiled

    def normal_traction(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the normal traction is defined.
            n: Time step index.

        Returns:
            Array of normal tractions for the given time step. Default is zero if not
            defined in the model parameters.

        """
        return np.zeros(sd.num_cells)

    def displacement_jump(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of displacement jumps for the given time step.
        """
        # Always return (nd-1, num_cells)
        return self.boundary_displacement(sd, n)

    def displacement_increment(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of displacement increments for the given time step.
        """
        disp_n = self.boundary_displacement(sd, n)
        disp_prev = self.boundary_displacement(sd, n - 1)
        # Both should be (nd-1, num_cells)
        return disp_n - disp_prev

    def friction_damage_state(self, sd: pp.Grid, n: int):
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of friction damage for the given time step.

        """
        h = self.friction_damage_history(sd, n)
        d0 = self.model.solid.residual_friction_damage
        return d0 + (1 - d0) * np.exp(-h)

    def friction_damage_history(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the friction damage history at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of friction damage history for the given time step.

        """
        return self.convolution(sd, n, self.friction_damage_evolution_coefficient)

    def dilation_damage_state(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of dilation damage for the given time step.

        """
        h = self.dilation_damage_history(sd, n)
        d0 = self.model.solid.residual_dilation_damage
        return d0 + (1 - d0) * np.exp(-h)

    def dilation_damage_history(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the dilation damage history at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of dilation damage history for the given time step.

        """
        return self.convolution(sd, n, self.dilation_damage_evolution_coefficient)

    def convolution(self, sd: pp.Grid, n: int, coefficient_function) -> np.ndarray:
        """Return the convolution of the displacement increment with the damage kernel.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.
            coefficient_function: Function to compute the coefficient for the
                convolution.

        Returns:
            Array of convolution values for the given time step.

        """
        var = np.zeros(sd.num_cells)
        # This method can be implemented in subclasses if needed.
        for i in range(1, n + 1):
            # Compute the contribution to the damage from the current time step.
            var_i = self.damage_length(sd, n, i) * coefficient_function(sd, i)
            var += var_i
        return var

    def normalized_traction_for_damage(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Convenience funtion for common parts of the damage functions.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of damage for the given time step."""
        t = self.normal_traction(sd, n)
        transitional_strength = 0.2 * self.model.solid.uniaxial_compressive_strength
        return -t / (transitional_strength)

    def dilation_damage_evolution_coefficient(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the dilation damage coefficient at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of dilation damage for the given time step.

        """
        K_ad = np.log(
            -self.model.solid.uniaxial_compressive_strength
            / np.clip(self.normal_traction(sd, n), None, -1e-15)
        )
        roughness = self.model.solid.characteristic_fracture_roughness

        return self.normalized_traction_for_damage(sd, n) * K_ad / roughness

    def friction_damage_evolution_coefficient(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the friction damage coefficient at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of friction damage for the given time step.

        """
        roughness = self.model.solid.characteristic_fracture_roughness

        return self.normalized_traction_for_damage(sd, n) * 3 / roughness


class ExactSolutionIsotropic(ExactSolution):
    def damage_length(self, sd: pp.Grid, n: int, i: int):
        """Damage length contribution from step i to exact length for time step n.

        Parameters:
            sd: Subdomain where the damage length is to be evaluated.
            n: Current time step index.
            i: Index of the time step we are collecting data from.

        Returns:
            Array of damage for the given time step.

        """
        return np.linalg.norm(self.displacement_increment(sd, i), axis=0)


class ExactSolutionAnisotropic(ExactSolution):
    def damage_length(self, sd: pp.Grid, n: int, i: int):
        """Damage length contribution from step i to exact length for time step n.

        Parameters:
            sd: Subdomain where the damage length is to be evaluated.
            n: Current time step index.
            i: Index of the time step we are collecting data from.

        Returns:
            Array of damage for the given time step combination.

        """
        # Compute normalized m for nonzero values.
        m = self.displacement_jump(sd, n)
        norm = np.linalg.norm(m, axis=0)
        nonzero = norm > 0

        m[:, nonzero] /= norm[nonzero]

        def oriented_length(j):
            u_j = self.displacement_jump(sd, j)
            return np.clip(np.einsum("ij,ij->j", u_j, m), 0, None)

        return np.abs(oriented_length(i) - oriented_length(i - 1))


# Collect parameters etc. This defines the test case as used in test_fracture_damage.py
# (dim x number of time steps) displacement values on the north boundary
num_time_steps = 5
north_displacements_3d = np.zeros((3, num_time_steps))
# Steps/increments in the tangential direction:
# 1. d_0  (north_displacements[:, 1] - north_displacements[:, 0])
# 2. 0
# 3. -d_0
# 4. -d_0
# 5. new direction
north_displacements_3d[0] = np.array([0.0, -2.0, -2.0, 2.0, 1.0])
north_displacements_3d[2] = np.array([0.0, 1.0, 1.0, -1.0, 1.0])
north_displacements_3d *= 1.0e-4

solid_params = pp.solid_values.extended_granite_values_for_testing.copy()
solid_params.update(
    {
        "friction_coefficient": 0.01,  # Low friction => slip \approx bc displacement
        "uniaxial_compressive_strength": 1e8,
        "characteristic_fracture_roughness": 1e-4,  # Same order as bc displacements.
        "residual_friction_damage": 0.3,
        "residual_dilation_damage": 0.6,
        "dilation_angle": 0.01,  # [rad] # Low but nonzero dilation angle to get some
        # dilation damage without incurring too much normal opening and stress.
        "maximum_elastic_fracture_opening": 0.0,  # [m] Simplify by assuming no elastic
        # opening.
    }
)
# Increase shear modulus to suppress shear displacements relative to normal ones.
solid_params["shear_modulus"] = 1e3 * cast(float, solid_params["shear_modulus"])

model_params = {
    # We need two cells in the y direction to get a fracture. In the x direction, we
    # also need two cells to avoid nasty singular matrix from MPSA discretization and
    # certain boundary conditions combinations. We can get away with one cell in the z
    # direction in 3d.
    "meshing_arguments": {"cell_size_x": 0.50, "cell_size_y": 0.50, "cell_size_z": 1.0},
    "fracture_indices": [1],  # Fracture 1 has constant y coordinate.
    # Set the schedule using arange to save data from all time steps.
    "time_manager": pp.TimeManager(np.arange(0, num_time_steps), 1, True),
    "north_displacements": north_displacements_3d,
    "interface_displacement_parameter_values": north_displacements_3d,
    # "times_to_export": [],  # Suppress export of data for testing.
    "material_constants": {
        "solid": FractureDamageSolidConstants(**solid_params),  # type: ignore[arg-type]
        "numerical": pp.NumericalConstants(characteristic_displacement=1e-2),
    },
    "adaptive_indicator_scaling": True,  # Needed for nonlinear convergence.
}


# If executed as main, run simulation.
if __name__ == "__main__":
    # Run a selected fracture damage example.

    # This executable block provides a lightweight demonstration of running a fracture
    # damage model. The model is the momentum balance model of fracture damage with
    # time-dependent displacement boundary conditions set by the key
    # "north_displacements" in the parameter dictionary.

    # The parameter `regimes` controls which mechanisms will be activated. Three regimes
    # are available: "dilation", "friction", or both, set to "dilation" below.

    dim = 2  # 2D case
    time_steps = 5
    # Choose damage regimes: "dilation", "friction", or both. Set to "dilation" for the
    # executable example.
    regimes = ["dilation"]

    model_params.update(
        {
            "time_manager": pp.TimeManager(np.arange(0, time_steps), 1, True),
        }
    )

    # Build the model class by adding the requested damage mechanisms as mixins to the
    # momentum balance model and the geometry mixin for the target dimension.
    class _Model(
        SquareDomainOrthogonalFractures,
        # Can be replaced by AnisotropicFractureDamageLength if desired:
        damage.IsotropicFractureDamageLength,
        FractureDamageMomentumBalance,
    ):
        pass

    for regime in regimes:
        model_class = add_mixin(
            damage_types[regime],  # type: ignore[type-abstract]
            _Model,
        )

    # Parameter setup for the momentum balance model.
    model_params["exact_solution"] = ExactSolutionIsotropic
    # Only pass active dimensions.
    model_params["north_displacements"] = model_params["north_displacements"][:dim]

    model = model_class(model_params)  # type: ignore[abstract]
    # In some cases, the momentum balance model cannot converge with the default solver
    # settings. A relaxed nonlinear solver setting is used for the executable example.
    solver_params = {
        "nl_convergence_inc_atol": 1e-6,
        "nl_convergence_res_atol": 1e-6,
        "nl_max_iterations": 35,
        "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
        "local_line_search": True,
        "constraint_violation_tolerance": 1e-5,
    }
    pp.ModelRunner(model, solver_params).run()
