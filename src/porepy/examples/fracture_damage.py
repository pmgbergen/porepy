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


class FractureDamageCoefficientsWhite:
    """Fracture damage coefficients for the White paper.

    This class is used to test the fracture damage model with the simpler damage
    coefficients used in the White paper. It overrides the methods as defined in the
    FractureDamageCoefficients class, which uses the more complex coefficients based on
    Gao et al. (2024).
    """

    def friction_damage_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Damage coefficient for friction damage [-].

        Parameters:
            subdomains: List of subdomains where the damage coefficient is defined.
                Should be of co-dimension one, i.e. fractures.

        Returns:
            Operator for the friction damage coefficient.
        """
        u_t = self.tangential_component(subdomains) @ self.plastic_displacement_jump(
            subdomains
        )
        c = self.friction_damage_decay(subdomains)
        coefficient = c * u_t
        coefficient.set_name("friction_damage_coefficient_white")
        return coefficient

    def dilation_damage_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Damage coefficient for dilation damage [-].

        Parameters:
            subdomains: List of subdomains where the damage coefficient is defined. Should
                be of co-dimension one, i.e. fractures.

        Returns:
            Operator for the dilation damage coefficient.
        """
        u_t = self.tangential_component(subdomains) @ self.plastic_displacement_jump(
            subdomains
        )
        c = self.dilation_damage_decay(subdomains)
        coefficient = c * u_t
        coefficient.set_name("dilation_damage_coefficient_white")
        return coefficient


class DamageBase(
    pp.constitutive_laws.FrictionDamage,
    pp.constitutive_laws.DilationDamage,
    pp.constitutive_laws.FractureDamageCoefficients,
    damage.FractureDamageHistoryVariables,
    damage.FractureDamageEquations,
):
    pass


class FractureDamageContactMechanics(  # type: ignore[misc]
    DamageBase,
    pp.contact_mechanics.ContactMechanics,
):
    """Fracture damage model.

    This class needs to be combined with a class defining the damage equation, i.e.,
    (An)IsotropicDamageEquation. This is to allow for different damage equations to be
    used with the same model, specifically during testing. TODO: Consider to choose one
    as default.

    Two methods are overridden in this class:
        - variables_stored_all_time_steps: The interface displacement is not included in
        the model, and thus the method is overridden to exclude it.
        - update_interface_displacement_parameter: The interface displacement parameter
        needs to be stored at all time steps.

    The combination of the two overrides amounts to replacing the variable stored at
    each time step with the parameter value being stored.

    """

    def variables_stored_all_time_steps(self) -> list[pp.ad.Variable]:
        """Return the variables stored at all time steps.

        Override default implementation, since the interface displacement is not
        included in the model.

        Returns:
            List of variables.

        """
        return self.equation_system.get_variables(
            variables=[
                self.contact_traction_variable,
            ]
        )

    def update_interface_displacement_parameter(self) -> None:
        """Update the interface displacement parameter."""

        name = self.interface_displacement_parameter_key
        for intf, data in self.mdg.interfaces(return_data=True):
            if pp.ITERATE_SOLUTIONS in data and name in data[pp.ITERATE_SOLUTIONS]:
                # Use the values at the unknown time step from the previous time step.
                vals = pp.get_solution_values(name=name, data=data, iterate_index=0)
            else:
                # No current value stored. The method was called during the
                # initialization.
                vals = self.interface_displacement_parameter_values(intf).ravel(
                    order="F"
                )

            # Before setting the new, most recent time step, shift the stored values
            # backwards in time.
            pp.shift_solution_values(
                name=name,
                data=data,
                location=pp.TIME_STEP_SOLUTIONS,
                max_index=None,
            )
            # Set the values of current time to most recent previous time.
            pp.set_solution_values(name=name, values=vals, data=data, time_step_index=0)

            # Set the unknown time step values.
            vals = self.interface_displacement_parameter_values(intf).ravel(order="F")
            pp.set_solution_values(name=name, values=vals, data=data, iterate_index=0)


@dataclass
class DamageSaveData:
    """Dataclass for saving data for verification purposes."""

    exact_friction_damage: np.ndarray
    approx_friction_damage: np.ndarray
    friction_damage_error: float
    exact_dilation_damage: np.ndarray
    approx_dilation_damage: np.ndarray
    dilation_damage_error: float


class ExactSolution(ABC):
    """Exact solution for the damage model.

    The driving force of the problem is assumed to be a Dirichlet boundary condition
    with transient values defined in the class parameters.

    """

    params: dict

    model: FractureDamageContactMechanics

    dilation_damage: Callable[[pp.Grid, int], np.ndarray]

    friction_damage: Callable[[pp.Grid, int], np.ndarray]

    def __init__(self, model) -> None:
        """Constructor of the class."""
        self.model = model

    def boundary_displacement(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of boundary displacements for the given time step.

        """
        num_cells = sd.num_cells
        if self.model.nd == 3:
            # Get only 0th and 2nd components of the displacement, as the 1st component
            # is the normal component, which is constant.
            inds: np.ndarray = np.array([0, 2])
        else:
            inds = np.array([0])
        displacements_3d = cast(np.ndarray, self.model.params["north_displacements"])
        u: np.ndarray = displacements_3d[inds, n]
        return np.tile(u, (num_cells, 1))

    def displacement_jump(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of displacement jumps for the given time step.

        """
        # This class assumes the interface to have the topmost half of the subdomain
        # as the k side, with the jump being u_k - u_j = u_top - u_bottom \approx
        # u_boundary - 0 = u_boundary.
        return self.boundary_displacement(sd, n)

    def displacement_increment(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of displacement increments for the given time step.

        """
        return self.boundary_displacement(sd, n) - self.boundary_displacement(sd, n - 1)

    def friction_damage(self, sd: pp.Grid, n: int):
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of friction damage for the given time step.

        """
        h = self._damage(sd, n)
        d0 = self.model.solid.initial_friction_damage
        return d0 + (1 - d0) * np.exp(-self.model.solid.friction_damage_decay * h)

    def dilation_damage(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of dilation damage for the given time step.

        """
        h = self._damage(sd, n)
        d0 = self.model.solid.initial_dilation_damage
        return d0 + (1 - d0) * np.exp(-self.model.solid.dilation_damage_decay * h)

    @abstractmethod
    def _damage(self, sd: pp.Grid, n: int) -> np.ndarray:
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of damage for the given time step.

        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class ExactSolutionIsotropic(ExactSolution):
    def _damage(self, sd: pp.Grid, n: int):
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of damage for the given time step.

        """
        return sum(
            [
                np.linalg.norm(self.displacement_increment(sd, i), axis=1)
                for i in range(1, n + 1)
            ]
        )


class ExactSolutionAnisotropic(ExactSolution):
    def _damage(self, sd: pp.Grid, n: int):
        """Return the exact solution at time step n.

        Parameters:
            sd: Subdomain where the boundary displacement is defined.
            n: Time step index.

        Returns:
            Array of damage for the given time step.

        """
        var = 0
        u = self.displacement_jump(sd, n)
        u_norm = np.linalg.norm(u, axis=1)
        nonzero = u_norm > 0
        # Compute normalized m for nonzero values.
        m = u.copy()
        m[nonzero] /= u_norm[nonzero][:, np.newaxis]
        # Loop through time steps and compute the damage.
        for i in range(1, n + 1):
            # Obtain the displacement increment and displacement jump.
            u_dot = self.displacement_increment(sd, i)
            # Compute inner products with m.
            inner_u = np.einsum("ij,ij->i", u, m)
            inner_dot = np.einsum("ij,ij->i", u_dot, m)
            # Compute the contribution to the damage from the current time step.
            var_i = np.heaviside(inner_u, 0.5) * np.abs(inner_dot)
            var += var_i
        return var


class DamageDataSaving(pp.PorePyModel):
    """Model mixin responsible for saving data for verification purposes."""

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

        # Collect data.
        friction_damage = self.exact_sol.friction_damage(sd, n)
        approx_friction_damage = cast(
            np.ndarray, self.friction_damage(sds).value(self.equation_system)
        )

        # initial_friction_damage = 1 implies no damage. In this case, the exact
        # solution, which is used to normalize the relative error, is zero. This can
        # lead to division by zero. To avoid this, we check if the
        # initial_friction_damage is 1 and set the error to zero. To avoid masking
        # errors, we also check that the approximated friction damage and the exact
        # solution are zero.
        if (
            np.isclose(self.solid.initial_friction_damage, 1.0)
            and np.allclose(approx_friction_damage, 0)
            and np.allclose(friction_damage, 0)
        ):
            error_friction_damage = 0.0
        else:
            error_friction_damage = ConvergenceAnalysis.lp_error(
                grid=sd,
                true_array=friction_damage,
                approx_array=approx_friction_damage,
                is_scalar=True,
                is_cc=True,
            )
        dilation_damage = self.exact_sol.dilation_damage(sd, n)
        approx_dilation_damage = cast(
            np.ndarray, self.dilation_damage(sds).value(self.equation_system)
        )
        # See comment above for the friction damage error.
        if (
            np.isclose(self.solid.initial_dilation_damage, 1.0)
            and np.allclose(approx_dilation_damage, 0)
            and np.allclose(dilation_damage, 0)
        ):
            error_dilation_damage = 0.0
        else:
            error_dilation_damage = ConvergenceAnalysis.lp_error(
                grid=sd,
                true_array=dilation_damage,
                approx_array=approx_dilation_damage,
                is_scalar=True,
                is_cc=True,
            )
        collected_data = DamageSaveData(
            exact_friction_damage=friction_damage,
            approx_friction_damage=approx_friction_damage,
            friction_damage_error=error_friction_damage,
            exact_dilation_damage=dilation_damage,
            approx_dilation_damage=approx_dilation_damage,
            dilation_damage_error=error_dilation_damage,
        )
        return collected_data


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
north_displacements_3d[0] = np.array([0.0, 0.2, 0.2, 0.0, 0.2, 0.01, -0.2])
north_displacements_3d[2] = np.array([0.0, 0.1, 0.1, 0.0, 0.1, 0.01, 0.1])
# Set constant y component smaller than the elastic opening to get compression.
north_displacements_3d[1] = 0.1
solid_params = {
    "friction_damage_decay": 0.5,
    "dilation_damage_decay": 0.5,
    "friction_coefficient": 0.1,  # Low friction to get slip \approx bc displacement
    "dilation_angle": 0.1,
    "shear_modulus": 1e6,  # Suppress shear displacement in the matrix.
    "universal_compressive_strength": 1e6,  # Suppress compressive damage.
    "characteristic_fracture_roughness": 0.1,  # Low roughness to promote slip
    "fracture_normal_stiffness": 1.0e-3,  # Low normal stiffness to promote slip
    "fracture_tangential_stiffness": 1.0e3,  # High tangential stiffness to suppress
    # elastic deformation
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
    "interface_displacement_parameter_values": north_displacements_3d,
    "exact_solution": ExactSolutionAnisotropic,
    "times_to_export": [],  # Suppress export of data for testing.
}


class IsotropicFractureDamage(  # type: ignore[misc]
    damage.IsotropicFractureDamageEquations,
    DamageDataSaving,
    ContactMechanicsTester,
    FractureDamageContactMechanics,
):
    """Isotropic fracture damage model.

    The equations are fracture damage and contact mechanics. Variables are contact
    traction and damage. The model is isotropic, i.e., the damage is independent of the
    loading direction.

    Also contains specifics defining a test case.

    """


class AnisotropicFractureDamage(  # type: ignore[misc]
    damage.AnisotropicFractureDamageEquations,
    DamageDataSaving,
    ContactMechanicsTester,
    FractureDamageContactMechanics,
):
    """Anisotropic fracture damage model.

    The equations are fracture damage and contact mechanics. Variables are contact
    traction and damage. The model is anisotropic, i.e., the damage is dependent on the
    loading direction.

    Also contains specifics defining a test case.
    """


class FractureDamageMomentumBalance(  # type: ignore[misc]
    damage.IsotropicFractureDamageEquations,
    DamageDataSaving,
    DamageBase,
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
