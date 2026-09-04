import copy
from dataclasses import dataclass
from typing import Any, Callable, Sequence, cast

import numpy as np

import porepy as pp
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMechanicsDirNorthSouth,
)
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils import models
from porepy.applications.test_utils.models import add_mixin
from porepy.compositional.materials import FractureDamageSolidConstants
from porepy.models import fracture_damage as damage


class TimeDependentDamageBCs:
    """Model mixin for time-dependent boundary conditions for fracture damage models.

    Defines time-dependent displacement values on all faces satisfying x[1] > 0.5. The
    time dependence is defined by the parameter "north_displacements" passed on model
    initialization.
    """

    nd: int
    params: dict
    time_manager: pp.TimeManager
    units: pp.Units

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
    "damage_length",
    "damage_evolution_coefficient",
    "damage_history",
    "dilation_damage_state",
    "friction_damage_state",
]
"""Quantities recorded from the model at every time step."""

EXACT_SOLUTION_METHOD_NAMES = ["damage_length"]
"""Of those, the ones :class:`ExactSolution` supplies a reference value for."""


def make_damagesavedata_class(
    method_names: list[str], exact_names: list[str] | None = None
) -> type:
    """Create a dataclass type with fields for approximate and exact values.

    Parameters:
        method_names: Names recorded from the model, each gaining an ``approx_`` field.
        exact_names: Subset of those that also gain an ``exact_`` field. Defaults to all
            of them, for callers that supply a reference for every quantity.

    Returns:
        The dataclass type.

    """
    if exact_names is None:
        exact_names = list(method_names)
    annotations: dict[str, type] = {}
    namespace: dict[str, object] = {"__annotations__": annotations}

    for name in method_names:
        if name in exact_names:
            annotations[f"exact_{name}"] = np.ndarray
        annotations[f"approx_{name}"] = np.ndarray

    cls = type("DamageSaveData", (object,), namespace)
    return dataclass(cls)


DamageSaveData = make_damagesavedata_class(
    DATA_SAVING_METHOD_NAMES, EXACT_SOLUTION_METHOD_NAMES
)


class DamageDataSaving(pp.PorePyModel):
    """Model mixin responsible for saving data for verification purposes."""

    damage_length: Callable[[list[pp.Grid], int], tuple[pp.ad.Operator, pp.ad.Operator]]
    """Damage length operator."""
    damage_history: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Damage history."""
    dilation_damage_state: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Dilation damage operator."""
    friction_damage_state: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Friction damage operator."""
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
        vals: dict[str, np.ndarray] = {}
        for name in DATA_SAVING_METHOD_NAMES:
            if name == "damage_length":
                # Treated separately because of its signature. Since the solution has
                # already been updated, time_step_index=1 gives the most recent
                # increment.
                length, _ = self.damage_length(sds, 1)
                vals["approx_" + name] = cast(
                    np.ndarray, length.value(self.equation_system)
                )
                vals["exact_" + name] = cast(
                    np.ndarray, self.exact_sol.damage_length(sd, n, n)
                )
            else:
                vals["approx_" + name] = cast(
                    np.ndarray, getattr(self, name)(sds).value(self.equation_system)
                )
        return DamageSaveData(**vals)


class FractureDamageMomentumBalance(  # type: ignore[misc]
    pp.models.solution_strategy.ContactIndicators,
    DamageDataSaving,
    pp.constitutive_laws.FractureDamage,
    pp.constitutive_laws.AsperityStressPartition,
    pp.constitutive_laws.DilationRotatedFriction,
    pp.constitutive_laws.FractureDamageEvolutionCoefficients,
    TimeDependentDamageBCs,
    pp.MomentumBalance,
):
    """Fracture damage momentum balance model.

    This model combines fracture damage mechanics with momentum balance and force
    balance across interfaces. Variables are matrix and interface displacements, contact
    traction, and damage.

    The class carries no damage length of its own: mix in either
    :class:`~porepy.models.fracture_damage.IsotropicFractureDamageLength` or
    :class:`~porepy.models.fracture_damage.AnisotropicFractureDamageLength` to choose
    whether the damage depends on the loading direction, as
    :func:`create_displacement_controlled_setup` does via its ``isotropic`` argument.

    Also contains specifics defining a test case in terms of the boundary conditions.

    """


class FractureDamageHistoryMixin(
    damage.FractureDamageEquation,
    damage.FractureDamageVariable,
):
    """The damage history variable and its convolution equation.

    Separate from the constitutive laws in
    :class:`~porepy.constitutive_laws.FractureDamage` so that the history, which is
    common to both channels, is supplied exactly once.
    """

    pass


class ExactSolution:
    """Reference values for the damage verification setup.

    The driving force of the problem is assumed to be a Dirichlet boundary condition
    with transient values defined in the class parameters, and the displacement jump is
    taken to follow it. That is what makes a closed form available at all, and it is why
    the reference covers the damage length rather than the damage itself: the length
    follows from the prescribed displacements, whereas the history and the damage states
    would additionally require the contact traction and the whole constitutive law.

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
        # Wear energy scales, chosen so that the boundary displacements below accumulate
        # softening exponents of order one, i.e. visible but not saturated damage. Since
        # the history is the frictional work, the scales that achieve this are smaller
        # than the traction alone would suggest by roughly the friction coefficient,
        # which this fixture keeps deliberately low. The ratio of the two sets which
        # channel degrades faster.
        # IS: Could be changed later to more physically motivated values.
        "friction_wear_energy_scale": 6.666666666666667,
        "dilation_wear_energy_scale": 4.088726586591035,
        "residual_friction_damage": 0.3,
        "residual_dilation_damage": 0.6,
        "dilation_angle": 0.01,  # [rad] # Low but nonzero dilation angle to get some
        # dilation damage without incurring too much normal opening and stress.
        "maximum_elastic_fracture_opening": 0.0,  # [m] Simplify by assuming no elastic
        # opening.
        "fracture_gap": 0.0,  # [m] Mated aperture.
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
    "material_constants": {
        "solid": FractureDamageSolidConstants(**solid_params),  # type: ignore[arg-type]
        "numerical": pp.NumericalConstants(characteristic_displacement=1e-2),
    },
    "adaptive_indicator_scaling": True,  # Needed for nonlinear convergence.
}


def run_example(damages: Sequence[str] = ("dilation",)) -> list[pp.PorePyModel]:
    """Run a selected fracture damage example and return the model.

    This thin wrapper is motivated by the contract for testing of examples, namely that
    the example should return a list of models. For details on the setup, see
    ``create_displacement_controlled_setup``.

    Parameters:
        damages: A sequence of strings specifying which damage mechanisms to activate.
            Options are "dilation", "friction", or both. Defaults to ("dilation",).

    Returns:
        A list containing the model(s) used in the simulation. Length of the list equals
        the number of damages specified.
    """
    model_class, params, solver_params = create_displacement_controlled_setup(
        isotropic=True,
        dim=2,
        damages=damages,
    )
    model = model_class(params)
    pp.ModelRunner(
        model,
        nonlinear_solver=pp.solvers.ConstraintLineSearchNonlinearSolver(solver_params),
    ).run()
    return [model]


def create_displacement_controlled_setup(
    isotropic: bool,
    dim: int,
    damages: Sequence[str],
) -> tuple[type[Any], dict[str, Any], dict[str, Any]]:
    """Create a displacement-controlled fracture damage setup.

    This builder assembles the model class and returns mutable model and solver
    parameter dictionaries. Callers may update the returned dictionaries before
    instantiating the model and running the simulation.

    Parameters:
        isotropic: If True, use isotropic damage length; otherwise anisotropic.
        dim: Spatial dimension of the bulk domain (2 or 3).
        damages: Damage channels to activate (subset of {"dilation", "friction"}).
            Channels left out have their residual damage state pinned at one, which
            holds them intact.

    Returns:
        Tuple ``(model_class, model_params, solver_params)`` with caller-owned mutable
        objects ready for post-processing before execution.
    """
    params = copy.deepcopy(model_params)
    model_class: type[Any] = FractureDamageMomentumBalance

    if isotropic:
        params["exact_solution"] = ExactSolutionIsotropic
        model_class = add_mixin(damage.IsotropicFractureDamageLength, model_class)
    else:
        params["exact_solution"] = ExactSolutionAnisotropic
        model_class = add_mixin(damage.AnisotropicFractureDamageLength, model_class)

    model_class = add_mixin(FractureDamageHistoryMixin, model_class)

    geom = (
        SquareDomainOrthogonalFractures if dim == 2 else CubeDomainOrthogonalFractures
    )
    model_class = add_mixin(geom, model_class)

    displacements = north_displacements_3d.copy()
    displacements = displacements[:dim]
    # Keep the fracture closed for the first steps and open it in the last one.
    displacements[1] = -2e-5
    displacements[1, 4] = 2e-3

    params.update(
        {
            "time_manager": pp.TimeManager(np.arange(0.0, 5.0), 1.0, True),
            "north_displacements": displacements,
        }
    )
    solid = solid_params.copy()
    for name in ("dilation", "friction"):
        if name not in damages:
            solid[f"residual_{name}_damage"] = 1.0
    params["material_constants"] = {
        "solid": FractureDamageSolidConstants(**solid),  # type: ignore[arg-type]
    }

    solver_params = {
        "nl_max_iterations": 30,
        "nl_convergence_res_atol": 1e-8,
        "nl_convergence_inc_atol": 1e-8,
        "local_line_search": True,
    }

    return model_class, params, solver_params


# If executed as main, run simulation.
if __name__ == "__main__":
    run_example()
