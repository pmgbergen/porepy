from dataclasses import dataclass
from typing import Callable, cast
from abc import ABC, abstractmethod

import numpy as np

import porepy as pp
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMechanicsDirNorthSouth,
)
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.applications.test_utils.models import ContactMechanicsTester
from porepy.compositional.materials import FractureDamageSolidConstants
from porepy.models import fracture_damage as damage


class TimeDependentDamageBCs(BoundaryConditionsMechanicsDirNorthSouth):
    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary values for the mechanics problem as a numpy array.

        Values for the north boundary are retrieved from the parameter dictionary passed
        on model initialization. The values are time dependent and are retrieved from
        the parameter dictionary using the key "north_displacements" and indexed in the
        second dimension by the current time index.

        Parameters:
            bg: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                domain, for each face in the subdomain.

        """
        sides = self.domain_boundary_sides(bg)
        values = np.zeros((self.nd, bg.num_cells))
        if bg.dim < self.nd - 1:
            # No displacement is implemented on grids of co-dimension >= 2.
            return values.ravel("F")

        # Wrap as array for convert_units. Thus, the passed values can be scalar or
        # list. Then tile for correct broadcasting below.
        u_north = self.params["north_displacements"][:, self.time_manager.time_index]
        u_n = np.tile(u_north, (bg.num_cells, 1)).T
        values[:, sides.north] = self.units.convert_units(u_n, "m")[:, sides.north]
        return values.ravel("F")


class MixedNorthMechanicsBCs(pp.PorePyModel):
    """Boundary conditions for the mechanics problem with mixed north boundary."""

    def characteristic_contact_traction(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Characteristic traction [Pa].

        Parameters:
            subdomains: List of subdomains where the characteristic traction is defined.

        Returns:
            Scalar operator representing the characteristic traction.

        """
        return pp.ad.Scalar(1)

    def characteristic_displacement(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Characteristic displacement [m].

        Parameters:
            subdomains: List of subdomains where the characteristic displacement is
                defined.

        Returns:
            Scalar operator representing the characteristic displacement.

        """
        return pp.ad.Scalar(1)

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Return the type of the mechanics boundary condition."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd, domain_sides.north + domain_sides.south, "dir"
        )
        if sd.dim < self.nd:
            return bc  # No displacement bcs needed.
        bc.internal_to_dirichlet(sd)
        bc.is_neu[1, domain_sides.north] = True
        bc.is_dir[1, domain_sides.north] = False
        return bc

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary values for the stress problem as a numpy array.

        The stress boundary conditions are set to zero.

        Parameters:
            bg: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                domain, for each face in the subdomain.

        """
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros((self.nd, bg.num_cells))
        if bg.dim < self.nd - 1:
            # No displacement is implemented on grids of co-dimension > 1.
            return values.ravel("F")

        # Wrap as array for convert_units. Thus, the passed values can be scalar or
        # list. Then tile for correct broadcasting below.
        sigma_north = self.params["north_stress"][self.time_manager.time_index]
        values[1, domain_sides.north] = (
            self.units.convert_units(sigma_north, "Pa")
            * bg.cell_volumes[domain_sides.north]
        )
        return values.ravel("F")


class FractureDamageCoefficientsWhite:
    """Fracture damage coefficients for the White paper.

    This class is used to test the fracture damage model with the simpler damage
    coefficients used in the White paper. It overrides the methods as defined in the
    FractureDamageCoefficients class, which uses the more complex coefficients based on
    Gao et al. (2024).
    """

    solid: FractureDamageSolidConstants
    """SolidConstants with dilation damage parameters."""

    def friction_damage_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Damage coefficient for friction damage [-].

        Parameters:
            subdomains: List of subdomains where the damage coefficient is defined.
                Should be of co-dimension one, i.e. fractures.

        Returns:
            Operator for the friction damage coefficient.
        """
        coefficient = pp.ad.Scalar(self.solid.friction_damage_decay)
        coefficient.set_name("friction_damage_coefficient_white")
        return coefficient

    def dilation_damage_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Damage coefficient for dilation damage [-].

        Parameters:
            subdomains: List of subdomains where the damage coefficient is defined.
                Should be of co-dimension one, i.e. fractures.

        Returns:
            Operator for the dilation damage coefficient.
        """
        coefficient = pp.ad.Scalar(self.solid.dilation_damage_decay)
        coefficient.set_name("dilation_damage_coefficient_white")
        return coefficient


DATA_SAVING_METHOD_NAMES = [
    "friction_damage",
    "friction_damage_coefficient",
    "dilation_damage",
    "dilation_damage_coefficient",
    "damage_length",
    "_common_factor_damage_coefficients",
]


def make_damagesavedata_class(method_names):
    """Create a dataclass type with fields for exact/approx values and errors."""
    annotations: dict[str, type] = {}
    namespace: dict[str, object] = {"__annotations__": annotations}

    for name in method_names:
        annotations[f"exact_{name}"] = np.ndarray
        annotations[f"approx_{name}"] = np.ndarray
        annotations[f"{name}_error"] = float

    cls = type("DamageSaveData", (object,), namespace)
    return dataclass(cls)


DamageSaveData = make_damagesavedata_class(DATA_SAVING_METHOD_NAMES)


class DamageDataSaving(pp.PorePyModel):
    """Model mixin responsible for saving data for verification purposes."""

    damage_length: Callable[[list[pp.Grid], int], pp.ad.Operator]
    """Damage length operator."""
    dilation_damage: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Dilation damage variable."""
    friction_damage: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Friction damage variable."""
    solid: FractureDamageSolidConstants

    def initialize_data_saving(self) -> None:
        """Set material parameters.

        Add exact solution object to the simulation model after materials have been set.

        """
        super().initialize_data_saving()  # type: ignore[safe-super]
        self.exact_sol: ExactSolution = self.params["exact_solution"](self)

    def collect_data(self) -> DamageSaveData:
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
                length, _ = self.damage_length(sds, time_step_index=1)
                approx_val = length.value(self.equation_system)
            else:
                # Collect data.
                exact_val = cast(np.ndarray, getattr(self.exact_sol, name)(sd, n))
                if hasattr(self, name):
                    approx_val = cast(
                        np.ndarray, getattr(self, name)(sds).value(self.equation_system)
                    )
                else:
                    approx_val = np.zeros_like(exact_val)

            error = ConvergenceAnalysis.lp_error(
                grid=sd,
                true_array=exact_val,
                approx_array=approx_val,
                is_scalar=True,
                is_cc=True,
            )
            vals["exact_" + name] = exact_val
            vals["approx_" + name] = approx_val
            vals[name + "_error"] = error
        collected_data = DamageSaveData(**vals)
        return collected_data


class FractureDamageMomentumBalance(  # type: ignore[misc]
    pp.models.solution_strategy.ContactIndicators,
    DamageDataSaving,
    pp.constitutive_laws.FractureDamageCoefficients,
    MixedNorthMechanicsBCs,
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


class DilationDamageMomentumBalance(
    pp.constitutive_laws.DilationDamage,
    damage.DilationDamageEquation,
    damage.DilationDamageVariable,
):
    pass


class FrictionDamageMomentumBalance(
    pp.constitutive_laws.FrictionDamage,
    damage.FrictionDamageEquation,
    damage.FrictionDamageVariable,
):
    pass


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
            Array of normal tractions for the given time step.

        """
        return self.model.params["north_stress"][n]

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

    def friction_damage(self, sd: pp.Grid, n: int):
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of friction damage for the given time step.

        """
        h = self.convolution(sd, n, self.friction_damage_coefficient)
        d0 = self.model.solid.initial_friction_damage
        return d0 + (1 - d0) * np.exp(-h)

    def dilation_damage(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of dilation damage for the given time step.

        """
        h = self.convolution(sd, n, self.dilation_damage_coefficient)
        d0 = self.model.solid.initial_dilation_damage
        return d0 + (1 - d0) * np.exp(-h)

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

    def _common_factor_damage_coefficients(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Convenience funtion for common parts of the damage functions.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of damage for the given time step."""
        t = self.normal_traction(sd, n)
        transitional_strength = 0.2 * self.model.solid.uniaxial_compressive_strength
        roughness = self.model.solid.characteristic_fracture_roughness
        return -t / (transitional_strength * roughness)

    def dilation_damage_coefficient(self, sd: pp.Grid, n: int) -> np.ndarray:
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
        return self._common_factor_damage_coefficients(sd, n) * K_ad

    def friction_damage_coefficient(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the friction damage coefficient at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of friction damage for the given time step.

        """
        return self._common_factor_damage_coefficients(sd, n) * 3


class ExactSolutionWhite:
    model: FractureDamageMomentumBalance

    def friction_damage_coefficient(self, sd: pp.Grid, n: int) -> np.ndarray:
        return np.full(sd.num_cells, self.model.solid.friction_damage_decay)

    def dilation_damage_coefficient(self, sd: pp.Grid, n: int) -> np.ndarray:
        return np.full(sd.num_cells, self.model.solid.dilation_damage_decay)


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
num_time_steps = 7
north_displacements_3d = np.zeros((3, num_time_steps))
# Steps/increments in the tangential direction:
# 1. d_0  (north_displacements[:, 1] - north_displacements[:, 0])
# 2. 0
# 3. -d_0
# 4. d_0
# 5. -(d_0 - 0.01)  # 0.01
# 6. d_1 (different from d_0)
north_displacements_3d[0] = np.array([0.0, -0.2, 0.2, 1e-5, 0.2, 1e-5, -0.2])
# The 1e-5 avoids m=0, which leads to zero damage in the analytical model. This is not
# reproduced in the numerical model due to the presence of numerical noise.
north_displacements_3d[2] = np.array([0.0, 0.1, 0.1, 0.0, 0.1, 0.0, 0.1])
# Set constant y component smaller than the elastic opening to get compression.
north_displacements_3d[1] = 0.1
north_displacements_3d[1, 2] = 0.3
north_stress = -1e-1 * np.ones(num_time_steps)
solid_params = {
    "friction_coefficient": 0.01,  # Low friction to get slip \approx bc displacement
    "dilation_angle": 0.1,
    "shear_modulus": 1e6,  # Suppress shear displacement in the matrix.
    "uniaxial_compressive_strength": 1e1,
    "characteristic_fracture_roughness": 0.1,  # Low roughness to promote slip
    "fracture_normal_stiffness": 1.0e-3,
    "maximum_elastic_fracture_opening": 0.2,  # Larger than (bc displacement + shear
    # dilation).
}
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
    "north_stress": north_stress,
    "interface_displacement_parameter_values": north_displacements_3d,
    "times_to_export": [],  # Suppress export of data for testing.
}
