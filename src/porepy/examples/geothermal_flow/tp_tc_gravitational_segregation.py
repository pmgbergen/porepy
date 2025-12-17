from __future__ import annotations
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

from typing import Callable, Optional, Sequence, cast, Any
import numpy as np
import logging
import time
import csv
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.sparse.csgraph import reverse_cuthill_mckee
from porepy.fracs.fracture_network_3d import FractureNetwork3d
import porepy as pp
from porepy.models.abstract_equations import LocalElimination
from porepy.models.compositional_flow import (
    CompositionalFractionalFlowTemplate as FlowTemplate,
)
from abc import abstractmethod

# PETSc imports (only if available)
try:
    import petsc4py
    petsc4py.init()
    from petsc4py import PETSc
    PETSC_AVAILABLE = True
except ImportError:
    PETSC_AVAILABLE = False
    logging.warning("*** ITERATIVE SOLVER NOT AVAILABLE ***")
    logging.warning("PETSc not available. All linear systems will use direct solver (MUMPS/UMFPACK).")
    logging.warning("For large systems, consider installing PETSc for iterative solver options.")

# Configure logging to show info messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Ensure specific loggers are enabled for linear solver information
logging.getLogger('porepy.models.solution_strategy').setLevel(logging.INFO)
logging.getLogger('porepy').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# test parameters
expected_order_loss = 4
mesh_2d_Q = True


residual_tolerance = 10.0 ** (-expected_order_loss)

# define constant phase densities
rho_l = 1000.0
rho_g = 500.0
h_l = 1.0
h_g = 2.0
to_Mega = 1.0e-6


# geometry description
class Geometry(pp.PorePyModel):
    @abstractmethod
    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> tuple[np.ndarray]:
        pass

    @staticmethod
    def harvest_sphere_members(xc, rc, x):
        dx = x - xc
        r = np.linalg.norm(dx, axis=1)
        return np.where(r < rc, True, False)


class ModelGeometry(Geometry):
    _sphere_radius: float = 0.0125 * 64
    _sphere_centre: np.ndarray = np.array([2.5, 5.0, 0.0])

    def set_domain(self) -> None:
        x_length = self.units.convert_units(5.0, "m")
        y_length = self.units.convert_units(5.0, "m")
        box: dict[str, pp.number] = {"xmax": x_length, "ymax": y_length}
        self._domain = pp.Domain(box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.0125 * 64, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    # def set_fractures(self) -> None:
    #     points = np.array(
    #         [
    #             [1.0, 2.0],
    #             [4.0, 2.0],
    #             [1.0, 2.0],
    #             [1.0, 4.0],
    #             [4.0, 2.0],
    #             [4.0, 4.0],
    #             [2.0, 1.0],
    #             [2.0, 4.0],
    #             [3.0, 1.0],
    #             [3.0, 4.0],
    #         ]
    #     ).T
    #     fracs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]).T
    #     self._fractures = pp.frac_utils.pts_edges_to_linefractures(points, fracs)

    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        if isinstance(sd, pp.Grid):
            face_centers = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            face_centers = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf

        def find_facets(center: np.ndarray) -> np.ndarray:
            logical = Geometry.harvest_sphere_members(
                center, self._sphere_radius, face_centers[bf_indices]
            )
            return bf_indices[logical]

        return find_facets(self._sphere_centre)


class ModelGeometry3D(Geometry):
    _sphere_radius: float = 0.0625* 4
    _sphere_centre: np.ndarray = np.array([2.5, 2.5, 5.0])

    def set_domain(self) -> None:
        x_length = self.units.convert_units(5.0, "m")
        y_length = self.units.convert_units(5.0, "m")
        z_length = self.units.convert_units(5.0, "m")
        box: dict[str, pp.number] = {
            "xmax": x_length,
            "ymax": y_length,
            "zmax": z_length,
        }
        self._domain = pp.Domain(box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.0625 * 4, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        if isinstance(sd, pp.Grid):
            face_centers = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            face_centers = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf

        def find_facets(center: np.ndarray) -> np.ndarray:
            logical = Geometry.harvest_sphere_members(
                center, self._sphere_radius, face_centers[bf_indices]
            )
            return bf_indices[logical]

        return find_facets(self._sphere_centre)

    # def set_fractures(self) -> None:
    #
    #
    #     kind_1_square_u = np.array([1.0, 1.0, 4.0, 4.0])
    #     kind_1_square_v = np.array([1.0, 4.0, 4.0, 1.0])
    #
    #     kind_2_square_u = np.array([2.0, 2.0, 4.0, 4.0])
    #     kind_2_square_v = np.array([2.0, 4.0, 4.0, 2.0])
    #
    #     # normal along z from z = 2.0
    #     f1 = np.vstack([kind_1_square_u, kind_1_square_v, np.full(4, 2.0)])
    #
    #     # normal along y from y = 1.0
    #     f2 = np.vstack([kind_1_square_u,  np.full(4, 1.0), kind_1_square_v])
    #
    #     # normal along y from y = 4.0
    #     f3 = np.vstack([kind_1_square_u,  np.full(4, 4.0), kind_1_square_v])
    #
    #     # normal along y from y = 3.0
    #     f4 = np.vstack([kind_1_square_u, np.full(4, 3.0), kind_1_square_v])
    #
    #     # normal along x from x = 2.0
    #     f5 = np.vstack([np.full(4, 2.0), kind_2_square_u, kind_2_square_v])
    #
    #     disjoint_set = [f1,f2,f3,f4,f5]
    #     self._fractures = [pp.PlaneFracture(p) for p in disjoint_set]

# constitutive description
def gas_saturation_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = (z_CO2 * rho_l) / (z_CO2 * rho_l + rho_g - z_CO2 * rho_g)
    vals = np.clip(vals, 1.0e-16, 1.0)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = (rho_l * rho_g) / (
        (z_CO2 * (rho_l - rho_g) + rho_g) * (z_CO2 * (rho_l - rho_g) + rho_g)
    )
    return vals, diffs


def temperature_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])

    factor = 0.0
    vals = np.array(h) * factor
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0 * factor
    return vals, diffs


def CO2_liq_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CO2)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def CO2_gas_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = np.ones_like(z_CO2)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


chi_functions_map = {
    "CO2_liq": CO2_liq_func,
    "CO2_gas": CO2_gas_func,
}


class LiquidEOS(pp.compositional.EquationOfState):
    def rho_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = rho_l * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def mu_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = (1.0e-3) * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def h(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = h_l * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def compute_phase_properties(
        self,
        phase_state: pp.compositional.PhysicalState,
        *thermodynamic_input: np.ndarray,
        params: Optional[Sequence[np.ndarray | float]] = None,
    ) -> pp.compositional.PhaseProperties:
        nc = len(thermodynamic_input[0])
        # mass density of phase
        rho, drho = self.rho_func(*thermodynamic_input)  # (n,), (3, n) array
        # specific enthalpy of phase
        h, dh = self.h(*thermodynamic_input)  # (n,), (3, n) array
        # dynamic viscosity of phase
        mu, dmu = self.mu_func(*thermodynamic_input)  # (n,), (3, n) array
        # thermal conductivity of phase
        kappa, dkappa = self.kappa(*thermodynamic_input)  # (n,), (3, n) array

        return pp.compositional.PhaseProperties(
            state=phase_state,
            rho=rho,
            drho=drho,
            h=h,
            dh=dh,
            mu=mu,
            dmu=dmu,
            kappa=kappa,
            dkappa=dkappa,
            phis=np.empty((2, nc)),
            dphis=np.empty((2, 3, nc)),
        )


class GasEOS(LiquidEOS):
    def rho_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = rho_g * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def h(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = h_g * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))


class FluidMixture(pp.PorePyModel):
    def get_components(self) -> Sequence[pp.FluidComponent]:
        return pp.compositional.load_fluid_constants(["H2O", "CO2"], "chemicals")

    def get_phase_configuration(
        self, components: Sequence[pp.Component]
    ) -> Sequence[
        tuple[pp.compositional.EquationOfState, pp.compositional.PhysicalState, str]
    ]:
        eos_L = LiquidEOS(components)
        eos_G = GasEOS(components)
        configuration_L = (pp.compositional.PhysicalState.liquid, "liq", eos_L)
        configuration_G = (pp.compositional.PhysicalState.gas, "gas", eos_G)
        return [configuration_L, configuration_G]

    def dependencies_of_phase_properties(
        self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        z_CO2 = [
            comp.fraction
            for comp in self.fluid.components
            if comp != self.fluid.reference_component
        ]
        return [self.pressure, self.enthalpy] + z_CO2  # type:ignore[return-value]


class SecondaryEquations(LocalElimination):
    dependencies_of_phase_properties: Callable[
        ..., Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]
    ]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]

    def set_equations(self) -> None:
        super().set_equations()
        subdomains = self.mdg.subdomains()

        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        matrix_boundary = cast(
            pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix)
        )
        subdomains_and_matrix = subdomains + [matrix_boundary]

        # liquid phase is dependent
        rphase = self.fluid.reference_phase
        # gas phase is independent
        independent_phases = [p for p in self.fluid.phases if p != rphase]

        for phase in independent_phases:
            self.eliminate_locally(
                phase.saturation,  # callable giving saturation on ``subdomains``
                self.dependencies_of_phase_properties(
                    phase
                ),  # callables giving primary variables on subdoains
                gas_saturation_func,  # numerical function implementing correlation
                subdomains_and_matrix,  # all subdomains on which to eliminate s_gas
            )

        ### Providing constitutive laws for partial fractions based on correlations
        for phase in self.fluid.phases:
            for comp in phase:
                check = self.has_independent_partial_fraction(comp, phase)
                if check:
                    self.eliminate_locally(
                        phase.partial_fraction_of[comp],
                        self.dependencies_of_phase_properties(phase),
                        chi_functions_map[comp.name + "_" + phase.name],
                        subdomains_and_matrix,
                    )

        ### Provide constitutive law for temperature
        self.eliminate_locally(
            self.temperature,
            self.dependencies_of_phase_properties(rphase),  # since same for all.
            temperature_func,
            subdomains_and_matrix,
        )


# model description
class BoundaryConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self.dirichlet_facets(sd), "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self.dirichlet_facets(sd), "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # Match the value used in buoyancy_flow_model.py
        p_top = 10.0e6 * to_Mega
        return np.ones(boundary_grid.num_cells) * p_top

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        h_inlet = np.zeros(boundary_grid.num_cells)
        return h_inlet

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        z_CO2 = np.zeros(boundary_grid.num_cells)
        return z_CO2

    def bc_values_fractional_flow_energy(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return self.bc_values_enthalpy(bg)

class InitialConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_condition(self) -> None:
        super().initial_condition()

        # set the values to be the custom functions
        liq, gas = self.fluid.phases
        for sd in self.mdg.subdomains():
            s_gas_val = self.ic_values_saturation(sd)
            x_CO2_liq_v = np.zeros_like(s_gas_val)
            x_CO2_gas_v = np.ones_like(s_gas_val)

            x_CO2_liq = liq.partial_fraction_of[self.fluid.components[1]]([sd])
            x_CO2_gas = gas.partial_fraction_of[self.fluid.components[1]]([sd])

            s_gas = gas.saturation([sd])
            self.equation_system.set_variable_values(s_gas_val, [s_gas], 0, 0)
            self.equation_system.set_variable_values(x_CO2_liq_v, [x_CO2_liq], 0, 0)
            self.equation_system.set_variable_values(x_CO2_gas_v, [x_CO2_gas], 0, 0)

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_init = 10.0e6 * to_Mega
        return np.ones(sd.num_cells) * p_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        # Compute initial enthalpy as a weighted average of phase enthalpies and densities
        ic_sg = self.ic_values_saturation(sd)
        ic_rho = rho_g * ic_sg + rho_l * (1.0 - ic_sg)
        h = (ic_sg * h_g * rho_g + (1.0 - ic_sg) * h_l * rho_l) / ic_rho
        return h

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        T = 0.0 * self.ic_values_enthalpy(sd)
        return T

    def ic_values_saturation(self, sd: pp.Grid) -> np.ndarray:
        # Match the logic from buoyancy_flow_model.py
        z_v = self.ic_values_overall_fraction(self.fluid.components[1], sd)
        return (z_v * rho_l) / (z_v * rho_l + rho_g - z_v * rho_g)

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        xc = sd.cell_centers.T
        # z = (
        #     np.where((xc[:, 1] >= 1.0) & (xc[:, 1] <= 2.0), 0.5, 0.0)
        #     + np.where((xc[:, 1] >= 3.0) & (xc[:, 1] <= 4.0), 0.5, 0.0)
        #     + np.where((xc[:, 0] >= 1.0) & (xc[:, 0] <= 2.0), 0.5, 0.0)
        #     + np.where((xc[:, 0] >= 3.0) & (xc[:, 0] <= 4.0), 0.5, 0.0)
        # )
        z = np.where((xc[:, 1] >= 0.0) & (xc[:, 1] <= 2.5), 0.33333333333333326, 0.33333333333333326)
        # z = np.where((xc[:, 2] >= 0.0) & (xc[:, 2] <= 2.5), 0.95, 0.05)
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class FlowModel(
    ModelGeometry if mesh_2d_Q else ModelGeometry3D,
    FluidMixture,
    InitialConditions,
    BoundaryConditions,
    SecondaryEquations,
    FlowTemplate,
):
    def __init__(self, params):
        super().__init__(params)
        self.newton_iterations_per_timestep = []
        self.total_newton_iterations = 0
        # Flag to use PETSc with MUMPS solver
        self.use_petsc = params.get("use_petsc", False)

        # Preconditioner selection for PETSc solver
        self.petsc_preconditioner = params.get("petsc_preconditioner", "bjacobi")
        valid_preconditioners = {"bjacobi", "asm", "jacobi", "lump_colsum", "amg_hypre","ilu0", "cpr"}
        if self.petsc_preconditioner not in valid_preconditioners:
            logger.warning(f"Invalid preconditioner '{self.petsc_preconditioner}'. Using 'bjacobi' as default.")
            self.petsc_preconditioner = "bjacobi"

        # Flag to enable Cuthill-McKee permutation for bandwidth reduction
        self.use_cuthill_mckee = params.get("use_cuthill_mckee", True)

        # Check if PETSc is available when requested
        if self.use_petsc and not PETSC_AVAILABLE:
            logger.warning("*** SOLVER CONFIGURATION MISMATCH ***")
            logger.warning("PETSc iterative solver was requested (use_petsc=True) but PETSc is not available.")
            logger.warning("All linear systems will use the default direct solver instead.")
            logger.warning("To use iterative solvers, install PETSc with: pip install petsc petsc4py")
            self.use_petsc = False

    def solve_linear_system_petsc(self, A: sps.spmatrix, b: np.ndarray, preconditioner: str = "asm") -> np.ndarray:
        """
        Solve linear system using PETSc with selectable preconditioners and detailed logging.
        """
        if not PETSC_AVAILABLE:
            raise RuntimeError("PETSc is not available")

        # Validate preconditioner
        valid_preconditioners = {"bjacobi", "asm", "jacobi", "lump_colsum", "amg_hypre", "ilu0", "lu", "cpr"}
        if preconditioner not in valid_preconditioners:
            logger.warning(f"Invalid preconditioner '{preconditioner}'. Using 'lu' as default.")
            preconditioner = "lu"

        logger.info(f"Solving linear system with PETSc {preconditioner.upper()}")

        # 1. Convert to CSR and prepare working vector
        A_csr = A.tocsr()
        b_working = b.copy()

        # Initialize permutation variables to None for safety
        perm = None
        eq_perm = None
        var_perm = None
        field_split = None

        # 1.5. Apply equation permutation
        try:
            A_csr, b_working, eq_perm, var_perm, field_split = self.apply_equation_permutation(A_csr, b_working)
        except Exception as e:
            logger.warning(f"Equation permutation failed: {e}. Continuing with original ordering.")
            if preconditioner == "cpr":
                logger.warning("CPR requires successful equation permutation. Falling back to 'asm'.")
                preconditioner = "asm"

        # 2. Apply Cuthill-McKee permutation
        if self.use_cuthill_mckee and preconditioner not in ["lu", "cpr"]:
            try:
                perm = reverse_cuthill_mckee(A_csr, symmetric_mode=False)
                A_csr = A_csr[perm, :][:, perm]
                b_working = b_working[perm]
            except Exception as e:
                logger.warning(f"Cuthill-McKee permutation failed: {e}. Continuing with original ordering.")
                perm = None

        # 3. Regularize Diagonal
        if preconditioner not in ["lump_colsum", "lu"]:
            diagonal = A_csr.diagonal()
            zero_diag_indices = np.where(np.abs(diagonal) < 1e-14)[0]
            if len(zero_diag_indices) > 0:
                logger.info(f"Regularizing {len(zero_diag_indices)} zero diagonal entries")
                A_lil = A_csr.tolil()
                matrix_norm = np.mean(np.abs(A_csr.data))
                regularization_value = max(1e-12, matrix_norm * 1e-8)
                for idx in zero_diag_indices:
                    A_lil[idx, idx] = regularization_value
                A_csr = A_lil.tocsr()

        # 4. Apply Matrix Scaling
        row_scaling, col_scaling, A_csr, b_scaled = self._apply_matrix_scaling(A_csr, b_working)

        # 5. Create PETSc Matrices/Vectors
        petsc_A = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
        petsc_A.assemblyBegin()
        petsc_A.assemblyEnd()

        petsc_b = PETSc.Vec().createWithArray(b_scaled)
        petsc_x = PETSc.Vec().createWithArray(np.zeros_like(b_scaled))

        # 6. Setup KSP
        ksp = PETSc.KSP().create()
        ksp_prefix = "fluid_buoyancy_"
        ksp.setOptionsPrefix(ksp_prefix)

        # Initialize explicit references for cleanup
        petsc_M = None
        is_p = None
        is_t = None

        # 7. Configure Solver
        if preconditioner == "lu":
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setFactorSolverType("mumps")
            ksp.setOperators(A=petsc_A, P=petsc_A)
        else:
            ksp.setType(PETSc.KSP.Type.FGMRES)
            ksp.setGMRESRestart(50)

            # Setup Operators
            if preconditioner == "lump_colsum":
                col_sums = np.array(np.abs(A_csr).sum(axis=0)).flatten()
                zero_cols = np.where(col_sums < 1e-14)[0]
                if len(zero_cols) > 0:
                    col_sums[zero_cols] = 1e-12
                diag_vals = 1.0 / col_sums

                petsc_M = PETSc.Mat().createAIJ(size=A_csr.shape)
                petsc_M.setUp()
                for i in range(len(diag_vals)):
                    petsc_M.setValue(i, i, diag_vals[i])
                petsc_M.assemblyBegin()
                petsc_M.assemblyEnd()
                ksp.setOperators(A=petsc_A, P=petsc_M)
            else:
                ksp.setOperators(A=petsc_A, P=petsc_A)

            # Setup PC
            pc = ksp.getPC()
            opts = PETSc.Options()

            if preconditioner == "cpr":
                if not field_split:
                    raise RuntimeError("CPR preconditioner requires 'field_split' data.")

                try:
                    n_pressure = field_split.get('pressure',
                                                 field_split.get('pressure_size', list(field_split.values())[0]))
                except (AttributeError, IndexError):
                    raise RuntimeError("Could not parse 'field_split' dictionary.")

                n_total = A_csr.shape[0]
                is_p = PETSc.IS().createStride(n_pressure, first=0, step=1)
                is_t = PETSc.IS().createStride(n_total - n_pressure, first=n_pressure, step=1)

                pc.setType(PETSc.PC.Type.FIELDSPLIT)
                pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
                pc.setFieldSplitIS(('pressure', is_p), ('transport', is_t))

                # --- Block 0: Pressure ---
                # Protection: Use LU for small matrices (<10k rows) to avoid AMG setup failures
                if n_pressure < 10000:
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_ksp_type", "preonly")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_type", "lu")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_factor_shift_type", "nonzero")
                else:
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_ksp_type", "richardson")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_type", "hypre")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_hypre_type", "boomeramg")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_hypre_boomeramg_strong_threshold", "0.7")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_hypre_boomeramg_coarsen_type", "HMIS")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_hypre_boomeramg_interp_type", "ext+i")
                    opts.setValue(f"-{ksp_prefix}fieldsplit_pressure_pc_hypre_boomeramg_agg_nl", "1")

                # --- Block 1: Transport ---
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_ksp_type", "richardson")
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_pc_type", "ilu")
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_pc_factor_levels", "0")
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_pc_factor_shift_type", "nonzero")
                opts.setValue(f"-{ksp_prefix}fieldsplit_transport_pc_factor_shift_amount", "1e-10")

            elif preconditioner == "ilu0":
                pc.setType(PETSc.PC.Type.ILU)
                pc.setFactorLevels(0)
                opts.setValue(f"-{ksp_prefix}pc_factor_shift_type", "nonzero")
                opts.setValue(f"-{ksp_prefix}pc_factor_shift_amount", "1e-12")

            elif preconditioner == "amg_hypre":
                pc.setType(PETSc.PC.Type.HYPRE)
                pc.setHYPREType("boomeramg")
                opts.setValue(f"-{ksp_prefix}pc_hypre_boomeramg_strong_threshold", "0.25")
                opts.setValue(f"-{ksp_prefix}pc_hypre_boomeramg_coarsen_type", "HMIS")
                opts.setValue(f"-{ksp_prefix}pc_hypre_boomeramg_interp_type", "ext+i")

            elif preconditioner == "bjacobi":
                pc.setType(PETSc.PC.Type.BJACOBI)
            elif preconditioner == "asm":
                pc.setType(PETSc.PC.Type.ASM)
                pc.setASMOverlap(1)
            elif preconditioner == "jacobi":
                pc.setType(PETSc.PC.Type.JACOBI)
            elif preconditioner == "lump_colsum":
                pc.setType(PETSc.PC.Type.MAT)

        # 8. Finalize Options
        ksp.setTolerances(rtol=1.0e-7, atol=1.0e-12, max_it=500)
        ksp.setFromOptions()

        # 9. Solve and Log
        solution = None
        try:
            # Step A: Explicitly time the Preconditioner Setup
            t_setup_start = time.time()
            ksp.setUp()
            t_setup_end = time.time()
            setup_dur = t_setup_end - t_setup_start

            # Step B: Time the Solve
            t_solve_start = time.time()
            ksp.solve(petsc_b, petsc_x)
            t_solve_end = time.time()
            solve_dur = t_solve_end - t_solve_start

            # Step C: Retrieve Metrics
            iters = ksp.getIterationNumber()
            resid = ksp.getResidualNorm()

            # Step D: Log Report
            logger.info(
                f"PETSc {preconditioner.upper()} Report | Setup: {setup_dur:.4f}s | Solve: {solve_dur:.4f}s | Iters: {iters} | Residual: {resid:.4e}")

            if ksp.getConvergedReason() < 0:
                logger.warning(f"Solver failed. Reason: {ksp.getConvergedReason()}")
            else:
                # 10. Unscale and Reverse Permutations
                scaled_sol = petsc_x.getArray().copy()
                unscaled_sol = col_scaling * scaled_sol

                if perm is not None:
                    cuthill_reversed_sol = np.zeros_like(unscaled_sol)
                    cuthill_reversed_sol[perm] = unscaled_sol
                    unscaled_sol = cuthill_reversed_sol

                if var_perm is not None:
                    solution = np.zeros_like(unscaled_sol)
                    solution[var_perm] = unscaled_sol
                else:
                    solution = unscaled_sol

        except Exception as e:
            # Fallback for LU
            if preconditioner == "lu" and "mumps" in str(e).lower():
                logger.warning("MUMPS failed. Retrying with PETSc native LU...")
                try:
                    pc.setFactorSolverType("petsc")
                    ksp.setFromOptions()
                    ksp.solve(petsc_b, petsc_x)
                    if ksp.getConvergedReason() >= 0:
                        scaled_sol = petsc_x.getArray().copy()
                        unscaled_sol = col_scaling * scaled_sol

                        if perm is not None:
                            cuthill_reversed_sol = np.zeros_like(unscaled_sol)
                            cuthill_reversed_sol[perm] = unscaled_sol
                            unscaled_sol = cuthill_reversed_sol

                        if var_perm is not None:
                            solution = np.zeros_like(unscaled_sol)
                            solution[var_perm] = unscaled_sol
                        else:
                            solution = unscaled_sol
                except Exception as e2:
                    logger.error(f"Fallback solver failed: {e2}")
            else:
                logger.error(f"Solver execution error: {e}")
        # Cleanup
        petsc_A.destroy()
        petsc_b.destroy()
        petsc_x.destroy()
        ksp.destroy()
        if petsc_M: petsc_M.destroy()
        if is_p: is_p.destroy()
        if is_t: is_t.destroy()

        return solution

    def _apply_matrix_scaling(self, A_csr, b):
        """
        Apply row and column scaling to improve matrix conditioning.

        Parameters:
        -----------
        A_csr : scipy sparse matrix
            Input matrix in CSR format
        b : numpy array
            Right-hand side vector

        Returns:
        --------
        tuple
            (row_scaling, col_scaling, scaled_A_csr, scaled_b) where scaling factors and scaled matrix/vector
        """

        # Compute row and column norms for scaling
        row_norms = np.array(np.sqrt((A_csr.multiply(A_csr)).sum(axis=1))).flatten()
        col_norms = np.array(np.sqrt((A_csr.multiply(A_csr)).sum(axis=0))).flatten()

        # Avoid division by zero
        row_norms = np.where(row_norms < 1e-16, 1.0, row_norms)
        col_norms = np.where(col_norms < 1e-16, 1.0, col_norms)

        # Create scaling factors (inverse of norms for better conditioning)
        row_scaling = 1.0 / np.sqrt(row_norms)
        col_scaling = 1.0 / np.sqrt(col_norms)

        # Apply scaling: D_r * A * D_c where D_r, D_c are diagonal scaling matrices
        A_scaled = sps.diags(row_scaling) @ A_csr @ sps.diags(col_scaling)

        # Scale right-hand side: D_r * b
        b_scaled = row_scaling * b

        logger.debug(f"Matrix scaling applied. Row norm range: [{np.min(row_norms):.2e}, {np.max(row_norms):.2e}], "
                    f"Col norm range: [{np.min(col_norms):.2e}, {np.max(col_norms):.2e}]")

        return row_scaling, col_scaling, A_scaled.tocsr(), b_scaled

    def solve_linear_system(self) -> np.ndarray:
        """
        Solve the linear system using either PETSc GMRES with selectable preconditioner or default solver.

        Preconditioner options (set via petsc_preconditioner parameter):
        - 'bjacobi': Block Jacobi preconditioner (default)
        - 'asm': Additive Schwarz Method
        - 'jacobi': Point Jacobi preconditioner
        - 'lump_colsum': Lumped column sum diagonal preconditioner
        - 'amg_hypre': Algebraic Multigrid with Hypre BoomerAMG

        Returns:
            np.ndarray: Solution vector (the nonlinear increment).
        """
        if self.use_petsc and PETSC_AVAILABLE:
            # Use PETSc solver with selected preconditioner
            A, b = self.linear_system
            solution = self.solve_linear_system_petsc(A, b, preconditioner=self.petsc_preconditioner)
            if solution is None:
                logger.warning(f"PETSc iterative solver with {self.petsc_preconditioner.upper()} preconditioner failed to converge.")
                return super().solve_linear_system()
            return solution
        else:
            # Check if PETSc was requested but not available
            if self.use_petsc and not PETSC_AVAILABLE:
                logger.info("*** SOLVER FALLBACK ***")
                logger.info("PETSc was requested but not available. Using default direct solver.")

            # Use default solver
            solution = super().solve_linear_system()
            if solution is None:
                raise RuntimeError("Linear solver returned None - this should not happen")
            return solution

    def linear_solver(self) -> pp.LinearSolver:
        """Return a custom linear solver that uses our PETSc solution when available."""
        if self.use_petsc and PETSC_AVAILABLE and hasattr(self, '_linear_system_solution'):
            # Return a dummy solver that just returns our precomputed solution
            class CustomPETScSolver(pp.LinearSolver):
                def __init__(self, solution):
                    self.solution = solution

                def __call__(self, A, b):
                    return self.solution

            return CustomPETScSolver(self._linear_system_solution)
        else:
            # Use default linear solver
            return super().linear_solver()

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
        return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

    def fourier_flux_discretization(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.TpfaAd:
        return pp.ad.TpfaAd(self.fourier_keyword, list(subdomains))

    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return phase.saturation(domains)**2

    def assemble_linear_system(self) -> None:
        """Custom assemble linear system that updates Jacobian every 0, 3, 6, 9... Newton iterations.

        This method implements a dedicated solution strategy that:
        - Assembles the full linear system (Jacobian + residual) at iterations 0, 3, 6, 9, etc.
        - Updates only the residual part for other iterations (1, 2, 4, 5, 7, 8, etc.)
        """
        t_0 = time.time()

        # Get current Newton iteration number
        iteration_num = self.nonlinear_solver_statistics.num_iteration

        if iteration_num % 2 == 0 or iteration_num < 10:
            # Update both Jacobian and residual at iterations 0, 3, 6, 9, ...
            logger.info(f"Newton iteration {iteration_num}: Updating Jacobian and residual")
            self.linear_system = self.equation_system.assemble(evaluate_jacobian=True)
        else:
            # Update only residual at iterations 1, 2, 4, 5, 7, 8, ...
            logger.info(f"Newton iteration {iteration_num}: Updating residual only")
            if hasattr(self, 'linear_system') and self.linear_system is not None:
                # Keep the existing Jacobian, update only the residual
                new_residual = self.equation_system.assemble(evaluate_jacobian=False)
                # Update the residual part of the linear system (tuple format: (matrix, rhs))
                self.linear_system = (
                    self.linear_system[0],  # Keep existing Jacobian
                    -new_residual  # Update residual with new evaluation
                )
            else:
                # Fallback: if no previous linear system exists, assemble full system
                logger.warning("No previous linear system found, assembling full system")
                if self._apply_schur_complement_reduction():
                    assert self.schur_complement_primary_variables, (
                        "Primary column block for Schur technique not found."
                    )
                    assert self.schur_complement_primary_equations, (
                        "Primary row block for Schur technique not defined."
                    )
                    self.linear_system = self.equation_system.assemble_schur_complement_system(
                        self.schur_complement_primary_equations,
                        self.schur_complement_primary_variables,
                        inverter=cast(
                            Callable[[sps.spmatrix], sps.spmatrix],
                            self.params.get("schur_complement_inverter", None),
                        ),
                    )
                else:
                    self.linear_system = self.equation_system.assemble()

        t_1 = time.time()
        logger.debug(f"Assembled linear system in {t_1 - t_0:.2e} seconds.")

    def matrix_plot(self, J:sps.spmatrix):
        plt.figure(figsize=(6, 6))
        plt.spy(J, markersize=2)
        plt.title('Sparsity Pattern')
        plt.savefig('sparsity_pattern.png', dpi=300, bbox_inches='tight')
        plt.close()

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        # Track Newton iterations for this timestep
        current_iterations = self.nonlinear_solver_statistics.num_iteration
        self.newton_iterations_per_timestep.append(current_iterations)
        self.total_newton_iterations += current_iterations

        # Print Newton iteration info for current timestep
        current_time = self.time_manager.time
        timestep_number = self.time_manager.time_index
        logger.info(f"Timestep {timestep_number} (t={current_time:.2e}): {current_iterations} Newton iterations")

        phases = list(self.fluid.phases)
        components = list(self.fluid.components)
        # Buoyancy reciprocity check
        flux_buoyancy_c0 = self.component_buoyancy(components[0], self.mdg.subdomains())
        flux_buoyancy_c1 = self.component_buoyancy(components[1], self.mdg.subdomains())
        b_c0 = self.equation_system.evaluate(flux_buoyancy_c0)
        b_c1 = self.equation_system.evaluate(flux_buoyancy_c1)
        assert np.all(np.isclose(b_c0 + b_c1, 0.0))

        # Reference and numerical integrals
        ref_rho = ref_rho_z = ref_energy = 0.0
        num_rho = num_rho_z = num_energy = 0.0
        for sd in self.mdg.subdomains():
            ic_sg_val = self.ic_values_saturation(sd)
            rho_l = phases[0].density([sd])
            rho_g_phase = phases[1].density([sd])
            ic_rho = pp.wrap_as_dense_ad_array(1.0 - ic_sg_val) * rho_l + pp.wrap_as_dense_ad_array(ic_sg_val) * rho_g_phase
            ref_rho += self.norm_vol_int(ic_rho, sd)

            ic_z_val = self.ic_values_overall_fraction(self.fluid.components[1], sd)
            ic_rho_z = ic_rho * pp.wrap_as_dense_ad_array(ic_z_val)
            ref_rho_z += self.norm_vol_int(ic_rho_z, sd)

            ic_p_val = self.ic_values_pressure(sd)
            ic_h_val = self.ic_values_enthalpy(sd)
            ic_energy = ic_rho * pp.wrap_as_dense_ad_array(ic_h_val) - pp.wrap_as_dense_ad_array(ic_p_val)
            ref_energy += self.norm_vol_int(ic_energy, sd)

            cur_rho = self.fluid.density([sd])
            num_rho += self.norm_vol_int(cur_rho, sd)

            cur_rho_z = cur_rho * components[1].fraction([sd])
            num_rho_z += self.norm_vol_int(cur_rho_z, sd)

            cur_energy = cur_rho * self.enthalpy([sd]) - self.pressure([sd])
            num_energy += self.norm_vol_int(cur_energy, sd)

        def order(loss: float) -> float:
            return np.inf if loss <= 0.0 else abs(np.floor(np.log10(loss)))

        expected = getattr(self, "expected_order_loss", expected_order_loss)
        mass_loss = abs(ref_rho - num_rho)
        z_mass_loss = abs(ref_rho_z - num_rho_z)
        energy_loss = abs(ref_energy - num_energy)
        # Diagnostic prints for conservation checks
        print("ref mass integral: ", ref_rho)
        print("num mass integral: ", num_rho)
        print("Order of mass loss: ", order(mass_loss))
        print("ref z-mass integral: ", ref_rho_z)
        print("num z-mass integral: ", num_rho_z)
        print("Order of z-mass loss: ", order(z_mass_loss))
        print("ref energy integral: ", ref_energy)
        print("num energy integral: ", num_energy)
        print("Order of energy loss: ", order(energy_loss))
        # Boolean checks with messages

        mass_conservative_Q = order(mass_loss) >= expected
        print("buoyancy discretization is mass conservative Q: ", mass_conservative_Q)
        z_mass_conservative_Q = order(z_mass_loss) >= expected
        print("buoyancy discretization is z mass conservative Q: ", z_mass_conservative_Q)
        energy_conservative_Q = order(energy_loss) >= expected
        print("buoyancy discretization is energy conservative Q: ", energy_conservative_Q)
        # assert mass_conservative_Q
        # assert z_mass_conservative_Q
        # assert energy_conservative_Q
        print("")
        print("")

    def norm_vol_int(self, op: pp.ad.Operator, sd: pp.Grid) -> float:
        # Global volume normalization (sum over all subdomains) as in buoyancy_flow_model.py
        total_volume = 0.0
        for g in self.mdg.subdomains():
            total_volume += np.sum(
                self.equation_system.evaluate(
                    self.volume_integral(pp.ad.Scalar(1), [g], dim=1)
                )
            )
        return (
            np.sum(
                self.equation_system.evaluate(
                    self.volume_integral(op, [sd], dim=1)
                )
            )
            / total_volume
        )

    def set_equations(self):
        super().set_equations()
        self.set_buoyancy_discretization_parameters()

    def set_nonlinear_discretizations(self) -> None:
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

    def before_nonlinear_iteration(self) -> None:
        self.update_buoyancy_driven_fluxes()
        self.rediscretize()

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2") * to_Mega
        size = np.sum([g.num_cells for g in subdomains]).astype(int)
        gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
        gravity_field.set_name("gravity_field")
        return gravity_field

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: Optional[np.ndarray],
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        if self._is_nonlinear_problem():

            total_volume = 0.0
            for sd in model.mdg.subdomains():
                total_volume += np.sum(
                    self.equation_system.evaluate(self.volume_integral(pp.ad.Scalar(1), [sd], dim=1)))

            # nonlinear_increment based norm
            nonlinear_increment_norm = self.compute_nonlinear_increment_norm(
                nonlinear_increment
            )

            # Residual per subsystem
            residual_norm = np.linalg.norm(residual) * total_volume
            # Check convergence requiring both the increment and residual to be small.
            converged_inc = (
                nl_params["nl_convergence_tol"] is np.inf
                or nonlinear_increment_norm < nl_params["nl_convergence_tol"]
            )
            converged_res = (
                nl_params["nl_convergence_tol_res"] is np.inf
                or residual_norm < nl_params["nl_convergence_tol_res"]
            )
            converged = converged_inc and converged_res
            diverged = False
        else:
            raise ValueError(
                "Gravitational segregation is nonlinear in its simpler form."
            )
        print("residual norm: ", residual_norm)
        return converged, diverged

    def set_geometry(self) -> None:
        """Define geometry and create a mixed-dimensional grid.

        The default values provided in set_domain, set_fractures, grid_type and
        meshing_arguments produce a 2d unit square domain with no fractures and a
        four Cartesian cells.

        """
        # Create the geometry through domain amd fracture set.
        self.set_domain()
        self.set_fractures()
        # Create a fracture network and a mixed-dimensional grid.
        self.create_fracture_network()
        self.create_mdg()

        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fracture + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures.
            self.well_network.mesh(self.mdg)

        apply_distortion_Q = False
        if apply_distortion_Q:
            for grid in self.mdg.subdomains():
                xc = grid.nodes.T
                x1 = 0.25 * np.sin(2.0 * np.pi * xc[:, 0] / 5) * np.sin(2.0 * np.pi * xc[:, 1] / 5)
                x2 = 0.25 * np.sin(2.0 * np.pi * xc[:, 1] / 5) * np.sin(2.0 * np.pi * xc[:, 0] / 5)
                xc[:, 0] += x1
                xc[:, 1] += x2
                grid.compute_geometry()

    def permute_equations_and_variables(self):
        """
        Permute equations and variables in the following order:
        1. Elliptic equations: mass_balance_equation, interface_darcy_flux_equation, well_flux_equation
        2. Transport equations: component_mass_balance_equation_CO2, energy_balance_equation,
           interface_fourier_flux_equation, interface_enthalpy_flux_equation, well_enthalpy_flux_equation
        3. Algebraic equations: elimination_of_s_gas_on_grids_[0], elimination_of_x_CO2_liq_on_grids_[0],
           elimination_of_x_CO2_gas_on_grids_[0], elimination_of_temperature_on_grids_[0]

        Returns:
            tuple: (equation_permutation, variable_permutation) where each is an array of indices
        """

        # Inputs provided
        equation_keys = list(self.equation_system.assembled_equation_indices.keys())
        variables_keys = list(set([v.name for v in self.equation_system.variables]))

        # Initialize the dictionary
        variable_equation_map = {}

        # Helper function to find equation in list
        def find_eq(keyword, eq_list):
            for eq in eq_list:
                if keyword in eq:
                    return eq
            return None

        # 1. Map Global Conservation Laws & Fluxes (Standard Physics Mappings)
        # Pressure <-> Mass Balance
        variable_equation_map['pressure'] = (
            find_eq('mass_balance_equation', equation_keys),
            'pressure'
        )

        # z_CO2 <-> Component Mass Balance
        variable_equation_map['z_CO2'] = (
            find_eq('component_mass_balance_equation_CO2', equation_keys),
            'z_CO2'
        )

        # Enthalpy <-> Energy Balance
        variable_equation_map['enthalpy'] = (
            find_eq('energy_balance_equation', equation_keys),
            'enthalpy'
        )

        # Fluxes (Direct name matching)
        flux_vars = ['interface_darcy_flux', 'interface_fourier_flux', 'interface_enthalpy_flux']
        for var in flux_vars:
            # Matches "interface_darcy_flux" to "interface_darcy_flux_equation"
            variable_equation_map[var] = (find_eq(var, equation_keys), var)

        # 2. Map Local Elimination/Constraint Equations
        # These look for the variable name inside the elimination string
        # e.g., "s_gas" is found inside "elimination_of_s_gas_..."
        elimination_vars = ['s_gas', 'x_CO2_liq', 'x_CO2_gas', 'temperature']

        for var in elimination_vars:
            # Search for the equation string that contains "elimination_of_{var}"
            target_str = f"elimination_of_{var}"
            found_eq = find_eq(target_str, equation_keys)

            if found_eq:
                variable_equation_map[var] = (found_eq, var)

        def find_variable_idxs(name):
            if 'interface' in name:
                md_var = self.equation_system.md_variable(name, self.mdg.interfaces())
            else:
                md_var = self.equation_system.md_variable(name, self.mdg.subdomains())
            var_dof = self.equation_system.dofs_of(md_var.sub_vars)
            return var_dof

        equation_e_indices = []
        variable_e_indices = []

        # order for field split
        elliptic_keys = ['pressure', 'interface_darcy_flux','interface_fourier_flux']
        elliptic_keys.extend(elimination_vars)
        for key in elliptic_keys:
            eq_name, var_name = variable_equation_map[key]
            if eq_name and var_name:
                # Get equation indices
                eq_idxs = self.equation_system.assembled_equation_indices[eq_name]
                equation_e_indices.extend(eq_idxs)

                # Get variable indices
                var_dofs = find_variable_idxs(var_name)
                variable_e_indices.extend(var_dofs)
                assert len(eq_idxs) == len(var_dofs), f"Mismatch in lengths for {key}: {len(eq_idxs)} equations vs {len(var_dofs)} variables"

        equation_t_indices = []
        variable_t_indices = []
        transport_keys = ['enthalpy', 'interface_enthalpy_flux','z_CO2']
        for key in transport_keys:
            eq_name, var_name = variable_equation_map[key]
            if eq_name and var_name:
                # Get equation indices
                eq_idxs = self.equation_system.assembled_equation_indices[eq_name]
                equation_t_indices.extend(eq_idxs)

                # Get variable indices
                var_dofs = find_variable_idxs(var_name)
                variable_t_indices.extend(var_dofs)
                assert len(eq_idxs) == len(var_dofs), f"Mismatch in lengths for {key}: {len(eq_idxs)} equations vs {len(var_dofs)} variables"

        equation_indices = equation_e_indices + equation_t_indices
        variable_indices = variable_e_indices + variable_t_indices
        return np.array(equation_indices), np.array(variable_indices), {'elliptic': len(equation_e_indices), 'transport': len(equation_t_indices)}

    def apply_equation_permutation(self, A: sps.spmatrix, b: np.ndarray) -> tuple[sps.spmatrix, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Apply equation and variable permutation to the linear system.

        Args:
            A: Jacobian matrix
            b: Right-hand side vector

        Returns:
            tuple: (permuted_A, permuted_b, equation_permutation, variable_permutation)
        """
        try:
            eq_perm, var_perm, field_split = self.permute_equations_and_variables()

            # Permute rows (equations) and columns (variables) of the matrix
            A_permuted = A[eq_perm, :][:, var_perm]

            # Permute the right-hand side vector
            b_permuted = b[eq_perm]

            logger.info(f"Applied equation permutation: {len(eq_perm)} equations, {len(var_perm)} variables")

            return A_permuted, b_permuted, eq_perm, var_perm, field_split

        except Exception as e:
            logger.warning(f"Failed to apply equation permutation: {e}. Using original ordering.")
            return A, b, None, None

day = 86400
t_scale = 1.0
tf = 300.0 * day
dt = 1.0 * day
time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    permeability=1.0e-14,
    porosity=0.1,
    thermal_conductivity=2.0 * to_Mega,
    density=2500.0,
    specific_heat_capacity=1000.0 * to_Mega,
)
material_constants = {"solid": solid_constants}
params = {
    "fractional_flow": True,
    "enable_buoyancy_effects": True,
    "material_constants": material_constants,
    "time_manager": time_manager,
    "prepare_simulation": False,
    "apply_schur_complement_reduction": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": residual_tolerance,
    "flag_failure_as_diverged": False,
    "max_iterations": 100,
    "use_petsc": True,  # Set to True to use PETSc with MUMPS solver
    "petsc_preconditioner": "cpr",  # Options: 'bjacobi', 'asm', 'jacobi', 'lump_colsum', 'amg_hypre', 'ilu0', 'cpr'
}

model = FlowModel(params)

model.prepare_simulation()

# Print number of cells and DOFs
total_cells = sum(sd.num_cells for sd in model.mdg.subdomains())
total_dofs = model.equation_system.num_dofs()
print(f"Number of cells: {total_cells}")
print(f"Number of DOFs: {total_dofs}")

def write_newton_iterations_to_csv(model, filename="newton_iterations.csv"):
    """Write Newton iteration data to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(['Timestep', 'Time', 'Newton_Iterations'])

        # Write data for each timestep
        for i, iterations in enumerate(model.newton_iterations_per_timestep):
            timestep_number = i + 1
            # Calculate time value based on time manager
            time_value = model.time_manager.schedule[0] + (timestep_number * model.time_manager.dt_init)
            writer.writerow([timestep_number, f"{time_value:.6e}", iterations])

        # Write summary row
        writer.writerow(['', '', ''])
        writer.writerow(['SUMMARY', '', ''])
        writer.writerow(['Total_Timesteps', len(model.newton_iterations_per_timestep), ''])
        writer.writerow(['Total_Newton_Iterations', model.total_newton_iterations, ''])
        if model.newton_iterations_per_timestep:
            avg_iterations = model.total_newton_iterations / len(model.newton_iterations_per_timestep)
            writer.writerow(['Average_Iterations_Per_Timestep', f"{avg_iterations:.2f}", ''])
            writer.writerow(['Max_Iterations', max(model.newton_iterations_per_timestep), ''])
            writer.writerow(['Min_Iterations', min(model.newton_iterations_per_timestep), ''])

pp.run_time_dependent_model(model, params)

# Write Newton iteration statistics to CSV
write_newton_iterations_to_csv(model, "newton_iterations.csv")
print(f"Newton iteration data written to newton_iterations.csv")
print(f"Total Newton iterations: {model.total_newton_iterations}")
print(f"Total timesteps: {len(model.newton_iterations_per_timestep)}")
if model.newton_iterations_per_timestep:
    avg_iterations = model.total_newton_iterations / len(model.newton_iterations_per_timestep)
print(f"Average iterations per timestep: {avg_iterations:.2f}")
