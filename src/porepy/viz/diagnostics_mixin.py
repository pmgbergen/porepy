"""This module provides DiagnosticsMixin class for PorePy models."""

import logging
from collections import defaultdict
from itertools import product
from typing import Literal, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.linalg import svds

from porepy import GridLike
from porepy.grids.mortar_grid import MortarGrid
from porepy.numerics.ad.equation_system import EquationSystem
from porepy.numerics.ad.operators import Variable

try:
    import seaborn as sns  # type: ignore[import]
except ImportError:
    _IS_SEABORN_AVAILABLE = False
else:
    _IS_SEABORN_AVAILABLE = True

logger = logging.getLogger(__name__)


class DiagnosticsMixin:
    """This is an auxilary mixin class for the PorePy model that allows to study
    the block structure of the Jacobian matrix.

    Currently supports plotting the condition numbers and/or
    the absolute maximum value of the matrix subblocks.
    For detailed information, see the [tutorial](../../tutorials/diagnostics.ipynb).

    Example:
        Basic usage:

        >>> from porepy.applications.verification_setups.mandel_biot import MandelSetup

        >>> class MandelDiagnostics(DiagnosticsMixin, MandelSetup):
                pass

        >>> setup = MandelDiagnostics(params={})
        >>> pp.run_time_dependent_model(setup, params={})
        >>> setup.show_diagnostics()

    """

    submatrices: dict[tuple[int, int], spmatrix]
    """The submatrices are stored here as they are computed.

    Index represents the pair (row, column) of the block.
    """

    # The mixin expects the model to have these properties defined.
    linear_system: tuple[csr_matrix, np.ndarray]
    equation_system: EquationSystem

    def show_diagnostics(
        self,
        grouping: Optional[Literal["subdomains"]] = None,
        is_plotting_condition_number: bool = False,
        is_plotting_max: bool = True,
    ) -> None:
        """Collects and plots diagnostics from the Jacobian matrix laying in
        `self.linear_system`.

        It is assumed that the full Jacobian matrix is stored in `self.linear_system`,
        and full information about the matrix indices is stored in
        `self.equation_system._equation_image_space_composition`.

        Note:
            It is assumed that variables with the same name on different grids
            are one variable.

        Args:
            grouping (optional): Either "subdomains" or None. If "subdomains",
                gathers the data related to one equation / variables from all subdomains
                or interfaces into one cell.
                If None, treats equations / variables in different subdomains or
                interfaces separately.
                Defaults to None.
            is_plotting_condition_number (optional): Whether to plot
                the condition number. Causion - this might take time if the matrix
                is big. Defaults to False.
            is_plotting_max (optional): Whether to plot the absolute maximum.
                This option is cheap even for big matrices.
                Defaults to True.
        """

        def get_eig(mat, which) -> float:
            return svds(mat, k=1, return_singular_vectors=False, which=which).item()

        def get_condition_number(mat) -> float:
            largest_eig = get_eig(mat, which="LM")
            # Smallest eigenvalue might be zero.
            smallest_eig = get_eig(mat, which="SM")
            if mat.data.size == 0:
                return 0.0
            else:
                return float(largest_eig / (smallest_eig + 1e-15))

        full_matrix: csr_matrix = self.linear_system[0]
        if not _IS_SEABORN_AVAILABLE:
            logger.warning(
                "Plotting the diagnostics image requires seaborn package."
                ' Run "pip install seaborn". Falling back to text mode.'
            )

        if is_plotting_condition_number and full_matrix.shape[0] > 1e5:
            logger.warning(
                "Computing condition number might take significant time for "
                "big matrices. It is recommended to reduce the problem size or to use "
                'the option "is_plotting_max" only.'
            )

        # Determining the block indices and collecting the submatrices.
        equation_indices = self._equations_indices(grouping=grouping)
        variable_indices = self._variable_indices(grouping=grouping)

        if grouping is None and max(len(equation_indices), len(variable_indices)) > 10:
            logger.warning(
                "Treating all the subdomains separately might not be "
                "informative if there are many of them. It is recommended to set"
                "\"grouping='subdomains'\""
            )

        self._collect_submatrices(
            mat=full_matrix,
            equation_indices=equation_indices,
            variable_indices=variable_indices,
        )

        # Computing the required features of each block.
        block_matrix_shape = len(equation_indices), len(variable_indices)
        block_data: dict[tuple[int, int], dict] = defaultdict(dict)
        for i, j in product(range(block_matrix_shape[0]), range(block_matrix_shape[1])):
            submat = self.submatrices[i, j]
            is_empty_block = submat.data[submat.data != 0].size == 0
            block_data[i, j]["is_empty_block"] = is_empty_block
            if not is_empty_block:
                if is_plotting_condition_number:
                    block_data[i, j]["condition_number"] = get_condition_number(submat)
                if is_plotting_max:
                    block_data[i, j]["max"] = abs(submat.data).max()

        # Plotting the figures.
        if is_plotting_condition_number:
            _plot_condition_number(
                block_data,
                variable_names=tuple(variable_indices.keys()),
                equation_names=tuple(equation_indices.keys()),
            )
            plt.show()

        if is_plotting_max:
            _plot_max(
                block_data,
                variable_names=variable_indices.keys(),
                equation_names=equation_indices.keys(),
            )
            plt.show()

    def _equations_indices(
        self, grouping: Optional[Literal["subdomains"]] = None
    ) -> dict[str, np.ndarray]:
        """Collects the indices of equations presented in the Jacobian matrix as rows.

        Returns:
            A dictionary with keys - printed names of the equations, values - indices of
            this equation in the global Jacobian matrix.
        """
        equation_indices = {}
        assert grouping in (None, "subdomains"), f"Unknown grouping: {grouping}"

        assembled_equation_indices = self.equation_system.assembled_equation_indices

        if grouping is None:
            block_indices = 0
            for eq_name, dof_indices in assembled_equation_indices.items():
                equation_image_space = (
                    self.equation_system._equation_image_space_composition[eq_name]
                )
                for grid, grid_dof_indices in equation_image_space.items():
                    grid_dof_indices = (
                        grid_dof_indices + block_indices
                    )  # Making a copy not to modify _equation_image_space_composition
                    assert np.all(np.isin(grid_dof_indices, dof_indices))
                    is_interface = isinstance(grid, MortarGrid)
                    grid_equation_name = _format_ticks(
                        name=eq_name,
                        dim=grid.dim,
                        grid_id=grid.id,
                        is_interface=is_interface,
                    )
                    equation_indices[grid_equation_name] = grid_dof_indices
                block_indices += dof_indices.size

        elif grouping == "subdomains":
            for eq_name, dof_indices in assembled_equation_indices.items():
                equation_indices[eq_name] = dof_indices

        return equation_indices

    def _variable_indices(
        self, grouping: Optional[Literal["subdomains"]] = None
    ) -> dict[str, np.ndarray]:
        """Collects the indices of variables presented in the Jacobian matrix as
        columns.

        Returns:
            A dictionary with keys - printed names of the variables, values - indices of
            this variables in the global Jacobian matrix.
        """
        variable_indices = {}
        assert grouping in (None, "subdomains"), f"Unknown grouping: {grouping}"

        if grouping is None:
            for variable in self.equation_system.variables:
                grid_variable_name = _format_variable_name_with_grid(variable)
                dofs = self.equation_system.dofs_of([variable])
                variable_indices[grid_variable_name] = dofs

        elif grouping == "subdomains":
            previous_variable_name = None
            for variable in self.equation_system.variables:
                if previous_variable_name != variable.name:
                    dofs = self.equation_system.dofs_of(
                        [self.equation_system.md_variable(variable.name)]
                    )
                    variable_indices[variable.name] = dofs
                    previous_variable_name = variable.name

        return variable_indices

    def _collect_submatrices(
        self,
        mat: spmatrix,
        equation_indices: dict[str, np.ndarray],
        variable_indices: dict[str, np.ndarray],
    ) -> None:
        """Slices the Jacobian matrix into block based on provided equations and
        variables.

        Args:
            mat: The full Jacobian matrix.
            equation_indices: The dictionary with keys - written names of equations and
                values - indices of these equations in the Jacobian. (Rows index).
            variable_indices: The dictionary with keys - written names of variables and
                values - indices of these variables in the Jacobian (Columns index).
        """
        submatrices = {}

        for i, ind_row in enumerate(equation_indices.values()):
            for j, ind_col in enumerate(variable_indices.values()):
                submatrix_indices = np.meshgrid(
                    ind_row, ind_col, sparse=True, indexing="ij"
                )
                submatrices[i, j] = mat[tuple(submatrix_indices)]

        self.submatrices = submatrices


def _plot_condition_number(
    block_data: dict[tuple[int, int], dict],
    variable_names: tuple[str, ...],
    equation_names: tuple[str, ...],
) -> None:
    """Utility function to plot the condition numbers of the matrix blocks.

    Plotting the image requires seaborn. If not available, falls back to text printing.

    Args:
        block_data: Dictionary with key - 2D index of the block, value - the data
            dictionary about selected block.
        variable_names: Sequence of variable names to be printed.
        equation_names: Sequence of equation names to be printed.
    """
    block_condition_numbers = np.zeros((len(equation_names), len(variable_names)))
    for (i, j), data in block_data.items():
        if not data["is_empty_block"]:
            block_condition_numbers[i, j] = data["condition_number"]

    if _IS_SEABORN_AVAILABLE:
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        ax = sns.heatmap(
            block_condition_numbers,
            mask=block_condition_numbers == 0,
            square=False,
            norm=LogNorm(vmin=1, vmax=1e3),
            annot=True,
            fmt=".1e",
            xticklabels=variable_names,
            yticklabels=equation_names,
            linewidths=0.01,
            linecolor="grey",
            cbar=False,
            cmap=cmap,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Block condition numbers")
    else:
        print(np.array_str(block_condition_numbers, precision=2))


def _plot_max(block_data, variable_names, equation_names) -> None:
    """Utility function to plot the max numbers of the matrix blocks.

    Plotting the image requires seaborn. If not available, falls back to text printing.

    Args:
        block_data: Dictionary with key - 2D index of the block, value - the data
            dictionary about selected block.
        variable_names: Sequence of variable names to be printed.
        equation_names: Sequence of equation names to be printed.
    """

    block_max = np.zeros((len(equation_names), len(variable_names)))
    empty = block_max.copy().astype(bool)
    for (i, j), data in block_data.items():
        empty[i, j] = data["is_empty_block"]
        if not empty[i, j]:
            block_max[i, j] = data["max"]
    if _IS_SEABORN_AVAILABLE:
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        ax = sns.heatmap(
            block_max,
            mask=empty,
            square=False,
            norm=LogNorm(),
            annot=True,
            fmt=".1e",
            xticklabels=variable_names,
            yticklabels=equation_names,
            linewidths=0.01,
            linecolor="grey",
            cbar=False,
            cmap=cmap,
            robust=True,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Absolute maximum value")
    else:
        print(np.array_str(block_max, precision=2))


def _format_variable_name_with_grid(variable: Variable) -> str:
    """Formats variable name for printing.

    Returns:
        str: Name to be printed.
    """
    grid: GridLike
    try:
        grid = variable.subdomains[0]
        is_interface = False
    except IndexError:
        grid = variable.interfaces[0]
        is_interface = True
    return _format_ticks(
        name=str(variable), dim=grid.dim, grid_id=grid.id, is_interface=is_interface
    )


def _format_ticks(name: str, dim: int, grid_id: int, is_interface: bool) -> str:
    """Formats variable or equation name for printing.

    Adds grid id and says if the grid is an interface.

    Returns:
        str: Name to be printed.
    """

    if is_interface:
        return f"{name} ({dim}D, intf. id={grid_id})"
    return f"{name} ({dim}D, id={grid_id})"
