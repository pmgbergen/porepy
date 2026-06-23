"""Flow model configuration file for the cold injection examples.

Setting of fluid mixtures, initial and boundary conditions, and other model parameters
such as position of wells and injection rates.

"""

from __future__ import annotations

from typing import Callable, TypeAlias

import numpy as np
import scipy.sparse as sps

import porepy as pp
import porepy.compositional.flash as pf
import porepy.models.compositional_flow as cf
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.applications.material_values.solid_values import basalt
from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceInfo,
    ConvergenceStatus,
)

from .solver import CFLESolver

CIModel: TypeAlias = (
    cfle.CFLEModelTemplate | cfle.CFFLEModelTemplate | cfle.IsothermalCFLEModelTemplate
)


class RelaxedCFLEResidualCriterion(pp.ResidualBasedAbsoluteCriterion):
    """Relaxed residual-based convergence criterion applying a separate tolerance
    for isofugacity constraints."""

    def __init__(
        self,
        tol: float,
        metric: Callable[[np.ndarray], ConvergenceInfo],
        tol_isofug: float | None = None,
    ):
        super().__init__(tol, metric)
        self.tol_isofug = tol_isofug if tol_isofug is not None else tol

    def check(self, *args, **kwargs) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Check convergence using :attr:`tol_isofug` for isofugacity constraints, and
        :attr:`tol` for all other equations.

        Parameters:
            args: Positional arguments for the convergence check.
            kwargs: Quantities to check for convergence.
                - value: The value to check for convergence.

        Returns:
            tuple[ConvergenceStatus, ConvergenceInfo]: Convergence status of
                the non-linear iteration and information about the convergence check.

        """
        metric_value = self.metric(kwargs["residual"])
        if isinstance(metric_value, dict):
            status = (
                ConvergenceStatus.CONVERGED
                if all(
                    [
                        v < self.tol_isofug if "isofugacity" in k else v < self.tol
                        for k, v in metric_value.items()
                    ]
                )
                else ConvergenceStatus.NOT_CONVERGED
            )
        else:
            status = (
                ConvergenceStatus.CONVERGED
                if metric_value < self.tol
                else ConvergenceStatus.NOT_CONVERGED
            )

        return status, metric_value


class ModelConfig(pp.PorePyModel):
    """Helper class to bundle the model configuration and inherit from
    ``PorePyModel``.

    Domain configurations have default values, shared by all cases.

    Fluid configurations, IC, BC, and injection and production values must be set.

    """

    ### Domain configurations.

    _DOMAIN_DIMENSIONS: list[float] = [100.0, 20.0, 100.0]
    """Domain dimensions in meters."""

    _INJECTION_POINTS: list[np.ndarray] = [np.array([10.0, 10.0])]
    """Coordinates of injection wells in meters."""

    _PRODUCTION_POINTS: list[np.ndarray] = [np.array([90.0, 10.0])]
    """Coordinates of production wells in meters."""

    _NUM_FRACTURES: int = 0
    """Number of fractures for random fracture geometry."""

    _CONDUCTIVE_FRACTURE_PERM: float = 1e-10
    """Permability of conductive fractures."""

    _BLOCKING_FRACTURE_PERM: float = 1e-16
    """Permability of blocking fractures."""

    _PERM_AROUND_WELLS: float = 1e-12

    _APERTURE_FACTOR_AFTER_TIME: list[tuple[float, pp.number]] = []
    """2-tuples of time-factor pairs, indicating at which time the aperture is
    multiplied with given factor."""

    _HEATED_BOUNDARY_ON: bool = False
    """Switch on Dirichlet-BC for Fourier flux on heated boundary."""

    _PRESSURE_BOUNDARY_ON: bool = False
    """Switch on Dirichlet-BC for the Darcy flux on the top boundary."""

    _ISOCHORIC_NPC_SPEC: pf.FlashSpec = pf.FlashSpec.none
    """Specification of isochoric flash equilibrium for nonlinear preconditioning in
    case of an aperture jump."""

    _FLASH_PT_INJECTION: bool = True
    """If True, the pT flash is always run in the injection well. Otherwise if falls
    back to the default schedule."""

    ### Fluid components.

    _COMPONENT_NAMES: list[str]
    """Names of fluid components used in the model."""

    _IDEAL_COMPONENTS: list[pp.compositional.ideal.IdealFluid]
    """Ideal fluid components used in the model."""

    ### Initial values.

    _p_INIT: float
    """Initial pressure in the whole domain in Pascals."""

    _T_INIT: float
    """Initial temperature in the whole domain in Kelvin."""

    _z_INIT: dict[str, float]
    """Initial overall fractions of fluid components in the whole domain, given as a
    dictionary mapping component names to values."""

    ### Injection and production conditions.

    _p_OUT: float
    """Pressure at production well in Pascals."""

    _T_IN: float
    """Temperature of injected fluid in Kelvin."""

    _z_IN: dict[str, float]
    """Overall fractions of injected fluid, given as a dictionary mapping component
    names to values. Total injection is scaled by these values per component."""

    _TOTAL_INJECTED_MASS: float = 10 * 27430.998956110157 / (60 * 60)
    """Total injected mass in mol/m^3/s.
    
    Note:
        Compute using flash and feed fraction ``_z_IN``, pressure ``_p_OUT`` and
        temperature ``_T_IN``.
    
    """

    ### Boundary conditions.
    _T_BC: float
    """Temperature at the heated boundary in Kelvin (Dirichlet condition for Fourier
    flux). Used if ``_HEATED_BOUNDARY_ON`` is True."""
    _p_BC: float = 10e6
    """Pressure at the Dirichlet boundary for the Darcy flux. Used if
    ``_PRESSURE_BOUNDARY_ON`` is True."""

    @property
    def _INJECTED_MASS(self) -> dict[str, dict[int, float]]:
        """Injected mass per component name and injection well index, calculated from
        total injected mass and injected overall fractions."""
        d: dict[str, dict[int, float]] = {}
        for n in self._COMPONENT_NAMES:
            d[n] = {}
            for i in range(len(self._INJECTION_POINTS)):
                if self.fluid.num_components == 1:
                    d[n][i] = self._TOTAL_INJECTED_MASS
                else:
                    d[n][i] = self._TOTAL_INJECTED_MASS * self._z_IN[n]

        return d

    @property
    def _T_INJECTION(self) -> dict[int, float]:
        """Injection temperature per injection well index."""
        d = {}
        for i in range(len(self._INJECTION_POINTS)):
            d[i] = self._T_IN
        return d

    @property
    def _p_PRODUCTION(self) -> dict[int, float]:
        """Production pressure per production well index."""
        d = {}
        for i in range(len(self._PRODUCTION_POINTS)):
            d[i] = self._p_OUT
        return d

    def _central_stripe(self, sd: pp.Grid) -> tuple[float, float]:
        """Returns the left and right boundary of the central, vertical stripe of the
        matrix, which represents roughly a third of the area.

        The x-axis is used to determin what is a third.

        """

        x_min = float(sd.cell_centers[0].min())
        x_max = float(sd.cell_centers[0].max())

        c = (x_min + x_max) / 2.0
        s = (x_max - x_min) / 6.0

        return c - s, c + s

    def _dirichlet_faces_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Defines the top boundary faces as the faces where pressure is fixed."""
        sides = self.domain_boundary_sides(sd)

        d = np.zeros(sd.num_faces, dtype=bool)
        if self.nd == 2:
            d[sides.north] = True
        elif self.nd == 3:
            d[sides.top] = True

        return d

    def _heated_boundary_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define heated boundary with D-type conditions for Fourier flux.

        In 2D, uses the central stripe of the x-axis.
        In 3D, uses a circle around the center with radius half the domain width.

        """
        sides = self.domain_boundary_sides(sd)

        if self.nd == 2:
            heated = np.zeros(sd.num_faces, dtype=bool)
            heated[sides.south] = True
            left, right = self._central_stripe(sd)
            heated &= sd.face_centers[0] >= left
            heated &= sd.face_centers[0] <= right
        elif self.nd == 3:
            heated = np.zeros(sd.num_faces, dtype=bool)
            heated[sides.bottom] = True

            x = sd.face_centers[0]
            y = sd.face_centers[0]

            L = self.domain.bounding_box["xmax"] - self.domain.bounding_box["xmin"]
            W = self.domain.bounding_box["ymax"] - self.domain.bounding_box["ymin"]
            x0 = self.domain.bounding_box["xmin"] + 0.5 * L
            y0 = self.domain.bounding_box["ymin"] + 0.5 * W
            r = 0.5 * min(W, L)
            circle = (x - x0) ** 2 + (y - y0) ** 2 <= r**2

            heated &= circle

        return heated


CFLEModel: TypeAlias = cfle.CFLEModelTemplate | cfle.CFFLEModelTemplate


def get_default_params(
    *,
    base_permeability: float = 1e-14,
) -> tuple[dict, dict]:
    """Get default parametrization for example.

    Some parametrization is supported by this method, others must be set directly when
    obtaining the dict.

    """
    phase_property_params = {
        "phase_property_params": [1e-4, 1e-2, 0.25, 5.0],
        "gen_arg_params": [1e-4, 1e-2, 0.25, 5.0],
    }

    basalt_ = basalt.copy()
    basalt_["permeability"] = base_permeability
    # basalt_["specific_heat_capacity"] = 0.0
    material_params = {"solid": pp.SolidConstants(**basalt_)}  # type:ignore

    flash_params = {
        "mode": "parallel",
        "solver": "npipm",
        "solver_params": {
            "atol_res": 1e-3,
            "max_iterations": 80,
        },
        "global_iteration_stride": 3,
        "compile": True,
        "compile_args": (pp.compositional.FlashSpec.pT,),
        "initializer": pf.HeuristicVLInitializer,
    }
    flash_params.update(phase_property_params)

    meshing_params = {
        "grid_type": "simplex",
        "meshing_arguments": {
            "cell_size": 5e-1,
            "cell_size_fracture": 5e-1,
        },
    }

    MODEL_PARAMS = {
        "solver_statistics_file_name": "solver_statistics.json",
        "equilibrium_specification": (pp.compositional.FlashSpec.none,),
        "linear_solver": "scipy_sparse",
        "apply_schur_complement_reduction": True,
        "flash_params": flash_params,
        "fractional_flow": False,
        "material_constants": material_params,
        "enable_buoyancy_effects": False,
    }

    MODEL_PARAMS.update(phase_property_params)
    MODEL_PARAMS.update(meshing_params)

    SOLVER_PARAMS = {
        "prepare_simulation": False,
        "nonlinear_solver": CFLESolver,
    }
    SOLVER_PARAMS.update(CFLESolver.default_params())

    return MODEL_PARAMS, SOLVER_PARAMS


def get_default_convergence_criteria(
    model: ModelConfig,
    max_iterations: int,
    atol_res: float,
    atol_inc: float,
    atol_res_isofug: float,
    atol_div: float = 1e8,
) -> dict:
    """Returns the default convergence criteria for the CFLE setup."""

    return {
        "max_iterations": int(max_iterations),
        "nl_convergence_criteria": {
            "inc_abs": pp.IncrementBasedAbsoluteCriterion(
                tol=atol_inc, metric=pp.VariableBasedLebesgueMetric(model)
            ),
            "res_abs": RelaxedCFLEResidualCriterion(
                tol=atol_res,
                metric=pp.EquationBasedLebesgueMetric(model),
                tol_isofug=atol_res_isofug,
            ),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.MaxIterationsCriterion(max_iterations=max_iterations),
            "inc_nan": pp.IncrementBasedNanCriterion(),
            "res_nan": pp.ResidualBasedNanCriterion(),
            "res_div": pp.ResidualBasedAbsoluteDivergenceCriterion(
                tol=atol_div, metric=pp.EquationBasedLebesgueMetric(model)
            ),
        },
    }


def get_rpc(model: CIModel) -> Callable[[list[sps.csr_matrix]], list[sps.csr_matric]]:
    """Construct the linear right-preconditioner for a model."""

    def rpc(mats: list[sps.csr_matrix]) -> list[sps.csr_matrix]:
        ref_vals = {
            # "pressure": 22064000.0,
            "pressure": 10e6,
            "temperature": 647.096,
            "specific_fluid_enthalpy": 524641.0735546586,
            "specific_fluid_volume": 5.59480372671e-05,
            "well_flux": 1e-4,  # 1e-5
            "interface_darcy_flux": 1e-6,  # 1e-5
        }

        ncol = model.equation_system.num_dofs()
        shape = (ncol,)

        s: np.ndarray  # Column scaling vector.

        # # Scaling with reference value
        s = np.ones(ncol)
        for k, v in ref_vals.items():
            s[model.equation_system.dofs_of([k])] = v

        # Scaling with column-inf norm of matrices
        # frac_vars = (
        #     model.overall_fraction_variables
        #     + model.saturation_variables
        #     + model.phase_fraction_variables
        #     + model.fraction_in_phase_variables
        # )
        # c_inf_norms = np.array([sps.linalg.norm(m, np.inf, axis=0) for m in mats])
        # d = np.maximum.reduce(c_inf_norms, axis=0)
        # frac_dofs = model.equation_system.dofs_of(frac_vars)
        # d[frac_dofs] = np.maximum(1.0, d[frac_dofs])
        # for k, v in ref_vals.items():
        #     dofs_k = model.equation_system.dofs_of([k])
        #     d[dofs_k] = np.maximum(v, d[dofs_k])

        # # NOTE careful, CF solver makes nother call to this, without switiching
        # # iterations.
        # # if isinstance(model.current_column_scales, np.ndarray):
        # #     damped_scaling = 0.7  # (0.3, 0.7)
        # #     s = (
        # #         damped_scaling * 1.0 / d
        # #         + (1 - damped_scaling) / model.current_column_scales
        # #     )
        # # else:
        # s = 1.0 / d

        # Nonlinear log-p scaling.
        if model._uses_logp():
            s[model.equation_system.dofs_of([model.pressure_variable])] = (
                model.equation_system.get_variable_values(
                    [model.pressure_variable], iterate_index=0
                )
            )

        assert s.shape == (ncol,), (
            f"Inconsistent shape for column scales: Got {s.shape}, expected {shape}"
        )
        for m in mats:
            assert m.shape[1] == ncol, (
                f"Inconsistent shape of matrix for RPC: Got {m.shape[1]} columns, "
                f"expected {ncol}"
            )
            m.data *= s[m.indices]
        model.current_column_scales = s

        return mats

    return rpc


def set_schur_complement(model: CIModel, use_extensives: bool = False) -> None:
    """Sets primary and secondary variables for the eliminating the local equilibrium
    DOFs."""

    primary_equations = cf.get_primary_equations_cf(model)
    primary_equations += [
        eq for eq in model.equation_system.equations.keys() if "flux" in eq
    ]
    if "production_pressure_constraint" in model.equation_system.equations:
        primary_equations += ["production_pressure_constraint"]
    if "injection_temperature_constraint" in model.equation_system.equations:
        primary_equations += ["injection_temperature_constraint"]

    primary_variables = cf.get_primary_variables_cf(
        model, use_extensives=use_extensives
    )
    primary_variables += list(
        set([v.name for v in model.equation_system.variables if "flux" in v.name])
    )

    model.schur_complement_primary_equations = primary_equations
    model.schur_complement_primary_variables = primary_variables
