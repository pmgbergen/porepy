"""
This module contains the implementation of a verification model for a poromechanical
system (without fractures) using three-dimensional manufactured solutions.

The implementation considers a compressible fluid, resulting in a coupled non-linear
set of equations. The dependency of the fluid density with the pressure is given by [1]:

.. math::

    \\rho(p) =  \\rho_0 \\exp{c_f \\left(p - p_0\\right)},

where :math:`\\rho` and :math:`p` are the density and pressure, :math:`\\rho_0`` and
:math:``p_0`` are the _reference_ density and pressure, and :math:`c_f` is the
_constant_ fluid compressibility.

We provide two different manufactured solutions, namely `parabolic`, `nordbotten2016`
[2],  which can be specified as a model parameter.

Parabolic manufactured solution:

.. math::

    p(x, y, z, t) = t x (1 - x) y (1- y) z (1 - z),

    u_x(x, y, t) = p(x, y, z, t),

    u_y(x, y, t) = p(x, y, z, t),

    u_z(x, y, t) = p(x, y, z, t).

Manufactured solution based on [2]:

.. math::

    p(x, y, z, t) = t x (1 - x) \\sin{2 \\pi y} \\sin{2 \\pi z},

    u_x(x, y, z, t) = p(x, y, z, t),

    u_y(x, y, z, t) = t \\sin{2 \\pi x} y (1 - y) \\sin{2 \\pi z},

    u_z(x, y, z, t) = t \\sin{2 \\pi y} \\sin{2 \\pi y} \\sin{2 \\pi z}.


References:

    - [1] Garipov, T. T., & Hui, M. H. (2019). Discrete fracture modeling approach for
      simulating coupled thermo-hydro-mechanical effects in fractured reservoirs.
      International Journal of Rock Mechanics and Mining Sciences, 122, 104075.

    - [2] Nordbotten, J. M. (2016). Stable cell-centered finite volume discretization
      for Biot equations. SIAM Journal on Numerical Analysis, 54(2), 942-968.

"""

from __future__ import annotations

from typing import Callable

import numpy as np
import sympy as sym

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from tests.functional.setups.manu_poromech_nofrac_2d import (
    ManuPoroMechModel2d,
    ManuPoroMechSolutionStrategy2d,
)

# PorePy typings
number = pp.number
grid = pp.GridLike


# -----> Exact solution
class ManuPoroMechExactSolution3d:
    """Class containing the exact manufactured solution for the verification model."""

    def __init__(self, model: pp.PorePyModel):
        """Constructor of the class."""

        # Physical parameters
        lame_lmbda = model.solid.lame_lambda  # [Pa] Lamé parameter
        lame_mu = model.solid.shear_modulus  # [Pa] Lamé parameter
        alpha = model.solid.biot_coefficient  # [-] Biot coefficient
        rho_0 = model.fluid.reference_component.density  # [kg / m^3] Reference density
        phi_0 = model.solid.porosity  # [-] Reference porosity
        p_0 = model.reference_variable_values.pressure  # [Pa] Reference pressure
        # [Pa^-1] Fluid compressibility
        c_f = model.fluid.reference_component.compressibility
        k = model.solid.permeability  # [m^2] Permeability
        mu_f = model.fluid.reference_component.viscosity  # [Pa * s] Fluid viscosity
        K_d = lame_lmbda + (2 / 3) * lame_mu  # [Pa] Bulk modulus

        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")
        pi = sym.pi

        # Exact pressure and displacement solutions
        manufactured_sol = model.params.get("manufactured_solution", "parabolic")
        if manufactured_sol == "parabolic":
            p = t * x * (1 - x) * y * (1 - y) * (1 - z)
            u = [p, p, p]
        elif manufactured_sol == "nordbotten_2016":
            p = t * x * (1 - x) * sym.sin(2 * pi * y) * sym.sin(2 * pi * z)
            u = [
                p,
                t * sym.sin(2 * pi * x) * y * (1 - y) * sym.sin(2 * pi * z),
                t * sym.sin(2 * pi * x) * sym.sin(2 * pi * y) * sym.sin(2 * pi * z),
            ]
        else:
            raise NotImplementedError("Manufactured solution is not available.")

        # Exact density
        rho = rho_0 * sym.exp(c_f * (p - p_0))

        # Exact darcy flux
        q = [
            -1 * (k / mu_f) * sym.diff(p, x),
            -1 * (k / mu_f) * sym.diff(p, y),
            -1 * (k / mu_f) * sym.diff(p, z),
        ]

        # Exact mass flux
        mf = [rho * q[0], rho * q[1], rho * q[2]]

        # Exact divergence of the mass flux
        div_mf = sym.diff(mf[0], x) + sym.diff(mf[1], y) + sym.diff(mf[2], z)

        # Exact divergence of the displacement
        div_u = sym.diff(u[0], x) + sym.diff(u[1], y) + sym.diff(u[2], z)

        # Exact poromechanical porosity
        phi = phi_0 + ((alpha - phi_0) * (1 - alpha) / K_d) * (p - p_0) + alpha * div_u

        # Exact flow accumulation
        accum_flow = sym.diff(phi * rho, t)

        # Exact flow source
        source_flow = accum_flow + div_mf

        # Exact gradient of the displacement
        grad_u = [
            [sym.diff(u[0], x), sym.diff(u[0], y), sym.diff(u[0], z)],
            [sym.diff(u[1], x), sym.diff(u[1], y), sym.diff(u[1], z)],
            [sym.diff(u[2], x), sym.diff(u[2], y), sym.diff(u[2], z)],
        ]

        # Exact transpose of the gradient of the displacement
        trans_grad_u = [
            [grad_u[0][0], grad_u[1][0], grad_u[2][0]],
            [grad_u[0][1], grad_u[1][1], grad_u[2][1]],
            [grad_u[0][2], grad_u[1][2], grad_u[2][2]],
        ]

        # Exact strain
        epsilon = [
            [
                0.5 * (grad_u[0][0] + trans_grad_u[0][0]),
                0.5 * (grad_u[0][1] + trans_grad_u[0][1]),
                0.5 * (grad_u[0][2] + trans_grad_u[0][2]),
            ],
            [
                0.5 * (grad_u[1][0] + trans_grad_u[1][0]),
                0.5 * (grad_u[1][1] + trans_grad_u[1][1]),
                0.5 * (grad_u[1][2] + trans_grad_u[1][2]),
            ],
            [
                0.5 * (grad_u[2][0] + trans_grad_u[2][0]),
                0.5 * (grad_u[2][1] + trans_grad_u[2][1]),
                0.5 * (grad_u[2][2] + trans_grad_u[2][2]),
            ],
        ]

        # Exact trace (in the linear algebra sense) of the strain
        tr_epsilon = epsilon[0][0] + epsilon[1][1] + epsilon[2][2]

        # Exact elastic stress
        sigma_elas = [
            [
                lame_lmbda * tr_epsilon + 2 * lame_mu * epsilon[0][0],
                2 * lame_mu * epsilon[0][1],
                2 * lame_mu * epsilon[0][2],
            ],
            [
                2 * lame_mu * epsilon[1][0],
                lame_lmbda * tr_epsilon + 2 * lame_mu * epsilon[1][1],
                2 * lame_mu * epsilon[1][2],
            ],
            [
                2 * lame_mu * epsilon[2][0],
                2 * lame_mu * epsilon[2][1],
                lame_lmbda * tr_epsilon + 2 * lame_mu * epsilon[2][2],
            ],
        ]

        # Exact poroelastic stress
        sigma_total = [
            [sigma_elas[0][0] - alpha * p, sigma_elas[0][1], sigma_elas[0][2]],
            [sigma_elas[1][0], sigma_elas[1][1] - alpha * p, sigma_elas[1][2]],
            [sigma_elas[2][0], sigma_elas[2][1], sigma_elas[2][2] - alpha * p],
        ]

        # Mechanics source term
        # We basically need to compute the divergence of a symmetric tensor
        source_mech = [
            (
                sym.diff(sigma_total[0][0], x)
                + sym.diff(sigma_total[0][1], y)
                + sym.diff(sigma_total[0][2], z)
            ),
            (
                sym.diff(sigma_total[1][0], x)
                + sym.diff(sigma_total[1][1], y)
                + sym.diff(sigma_total[1][2], z)
            ),
            (
                sym.diff(sigma_total[2][0], x)
                + sym.diff(sigma_total[2][1], y)
                + sym.diff(sigma_total[2][2], z)
            ),
        ]

        # Public attributes
        self.p = p  # pressure
        self.u = u  # displacement
        self.sigma_elast = sigma_elas  # elastic stress
        self.sigma_total = sigma_total  # poroelastic (total) stress
        self.q = q  # Darcy flux
        self.mf = mf  # Mass flux, i.e., density * Darcy flux
        self.phi = phi  # poromechanics porosity
        self.rho = rho  # density
        self.source_mech = source_mech  # Source term entering the momentum balance
        self.source_flow = source_flow  # Source term entering the mass balance

    # -----> Primary and secondary variables
    def pressure(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact pressure [Pa] at the cell centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_cells, )`` containing the exact pressures at the
            cell centers at the given ``time``.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers
        cc = sd.cell_centers

        # Lambdify expression
        p_fun: Callable = sym.lambdify((x, y, z, t), self.p, "numpy")

        # Cell-centered pressures
        p_cc: np.ndarray = p_fun(cc[0], cc[1], cc[2], time)

        return p_cc

    def displacement(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact displacement [m] at the cell centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(3 * sd.num_cells, )`` containing the exact displacements
            at the cell centers for the given ``time``.

        Notes:
            The returned displacement is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers
        cc = sd.cell_centers

        # Lambdify expression
        u_fun: list[Callable] = [
            sym.lambdify((x, y, z, t), self.u[0], "numpy"),
            sym.lambdify((x, y, z, t), self.u[1], "numpy"),
            sym.lambdify((x, y, z, t), self.u[2], "numpy"),
        ]

        # Cell-centered displacements
        u_cc: list[np.ndarray] = [
            u_fun[0](cc[0], cc[1], cc[2], time),
            u_fun[1](cc[0], cc[1], cc[2], time),
            u_fun[2](cc[0], cc[1], cc[2], time),
        ]

        # Flatten array
        u_flat: np.ndarray = np.asarray(u_cc).ravel("F")

        return u_flat

    def darcy_flux(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact Darcy flux [m^3 * s^-1] at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_faces, )`` containing the exact Darcy fluxes at
            the face centers for the given ``time``.

        Note:
            The returned fluxes are already scaled with ``sd.face_normals``.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get list of face indices
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression

        q_fun: list[Callable] = [
            sym.lambdify((x, y, z, t), self.q[0], "numpy"),
            sym.lambdify((x, y, z, t), self.q[1], "numpy"),
            sym.lambdify((x, y, z, t), self.q[2], "numpy"),
        ]

        # Face-centered Darcy fluxes
        q_fc: np.ndarray = (
            q_fun[0](fc[0], fc[1], fc[2], time) * fn[0]
            + q_fun[1](fc[0], fc[1], fc[2], time) * fn[1]
            + q_fun[2](fc[0], fc[1], fc[2], time) * fn[2]
        )

        return q_fc

    def poroelastic_force(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact poroelastic force [N] at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(3 * sd.num_faces, )`` containing the exact poroealstic
            force at the face centers for the given ``time``.

        Notes:
            - The returned poroelastic force is given in PorePy's flattened vector
              format.
            - Recall that force = (stress dot_prod unit_normal) * face_area.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers and face normals
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression
        sigma_total_fun: list[list[Callable]] = [
            [
                sym.lambdify((x, y, z, t), self.sigma_total[0][0], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[0][1], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[0][2], "numpy"),
            ],
            [
                sym.lambdify((x, y, z, t), self.sigma_total[1][0], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[1][1], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[1][2], "numpy"),
            ],
            [
                sym.lambdify((x, y, z, t), self.sigma_total[2][0], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[2][1], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[2][2], "numpy"),
            ],
        ]

        # Face-centered poroelastic force
        force_total_fc: list[np.ndarray] = [
            # (sigma_xx * n_x + sigma_xy * n_y + sigma_xz * n_z) * face_area
            sigma_total_fun[0][0](fc[0], fc[1], fc[2], time) * fn[0]
            + sigma_total_fun[0][1](fc[0], fc[1], fc[2], time) * fn[1]
            + sigma_total_fun[0][2](fc[0], fc[1], fc[2], time) * fn[2],
            # (sigma_yx * n_x + sigma_yy * n_y + sigma_yz * n_z) * face_area
            sigma_total_fun[1][0](fc[0], fc[1], fc[2], time) * fn[0]
            + sigma_total_fun[1][1](fc[0], fc[1], fc[2], time) * fn[1]
            + sigma_total_fun[1][2](fc[0], fc[1], fc[2], time) * fn[2],
            # (sigma_zx * n_x + sigma_zy * n_y + sigma_zz * n_z) * face_area
            sigma_total_fun[2][0](fc[0], fc[1], fc[2], time) * fn[0]
            + sigma_total_fun[2][1](fc[0], fc[1], fc[2], time) * fn[1]
            + sigma_total_fun[2][2](fc[0], fc[1], fc[2], time) * fn[2],
        ]

        # Flatten array
        force_total_flat: np.ndarray = np.asarray(force_total_fc).ravel("F")

        return force_total_flat

    # -----> Sources
    def mechanics_source(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Compute exact source term for the momentum balance equation.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Exact right hand side of the momentum balance equation with ``shape=(
            3 * sd.num_cells, )``.

        Notes:
            The returned array is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers and cell volumes
        cc = sd.cell_centers
        vol = sd.cell_volumes

        # Lambdify expression
        source_mech_fun: list[Callable] = [
            sym.lambdify((x, y, z, t), self.source_mech[0], "numpy"),
            sym.lambdify((x, y, z, t), self.source_mech[1], "numpy"),
            sym.lambdify((x, y, z, t), self.source_mech[2], "numpy"),
        ]

        # Evaluate and integrate source
        source_mech: list[np.ndarray] = [
            source_mech_fun[0](cc[0], cc[1], cc[2], time) * vol,
            source_mech_fun[1](cc[0], cc[1], cc[2], time) * vol,
            source_mech_fun[2](cc[0], cc[1], cc[2], time) * vol,
        ]

        # Flatten array
        source_mech_flat: np.ndarray = np.asarray(source_mech).ravel("F")

        # Invert sign according to sign convention.
        return -source_mech_flat

    def flow_source(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Compute exact source term for the fluid mass balance equation.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Exact right hand side of the fluid mass balance equation with ``shape=(
            sd.num_cells, )``.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers and cell volumes
        cc = sd.cell_centers
        vol = sd.cell_volumes

        # Lambdify expression
        source_flow_fun: Callable = sym.lambdify(
            (x, y, z, t), self.source_flow, "numpy"
        )

        # Evaluate and integrate source
        source_flow: np.ndarray = source_flow_fun(cc[0], cc[1], cc[2], time) * vol

        return source_flow


# -----> Geometry
class UnitCubeGrid:
    """Class for setting up the geometry of the unit cube domain."""

    params: dict
    """Simulation model parameters."""

    def set_domain(self) -> None:
        """Set fracture network. Unit square with no fractures."""
        self._domain = nd_cube_domain(3, 1.0)

    def mesh_ingarguments(self) -> dict[str, float]:
        """Set meshing arguments."""
        default_mesh_arguments = {"cell_size": 0.1}
        return self.params.get("meshing_arguments", default_mesh_arguments)


# -----> Solution strategy
class ManuPoroMechSolutionStrategy3d(ManuPoroMechSolutionStrategy2d):
    """Solution strategy for the verification model."""

    exact_sol: ManuPoroMechExactSolution3d
    """Exact solution object."""

    def __init__(self, params: dict):
        """Constructor for the class."""
        super().__init__(params)

        self.exact_sol: ManuPoroMechExactSolution3d
        """Exact solution object."""

    def set_materials(self):
        """Set material parameters."""
        super().set_materials()

        # Instantiate exact solution object after materials have been set
        self.exact_sol = ManuPoroMechExactSolution3d(self)


# -----> Mixer class
class ManuPoroMechModel3d(  # type: ignore[misc]
    UnitCubeGrid,
    ManuPoroMechSolutionStrategy3d,
    ManuPoroMechModel2d,
):
    """
    Mixer class for the two-dimensional non-linear poromechanics verification model.

    """
