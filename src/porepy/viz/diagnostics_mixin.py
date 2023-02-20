"""Module for diagnostics of PorePy's models built on seaborn.

"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from itertools import product
from typing import Any, Callable, Literal, Optional

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.linalg import svds
from typing_extensions import TypeAlias

from porepy import GridLike
from porepy.grids.md_grid import MixedDimensionalGrid
from porepy.grids.mortar_grid import MortarGrid
from porepy.numerics.ad.equation_system import EquationSystem
from porepy.numerics.ad.operators import Variable

# Seaborn is a visualization library based on Matplotlib. It allows for building nice
# figures. Seaborn library is not one of the dependencies of PorePy. Thus, it might not
# be present on the user's device. In this case, Python raises ImportError exception.
try:
    # MyPy is not happy with Seaborn since it's not typed. We silence this warning.
    import seaborn as sns  # type: ignore[import]
except ImportError:
    _IS_SEABORN_AVAILABLE: bool = False
else:
    _IS_SEABORN_AVAILABLE = True

logger = logging.getLogger(__name__)


# Type aliases. See the docs of `DiagnosticsMixin.show_diagnostics` for the details.
GridGroupingType: TypeAlias = "list[list[GridLike]]"
"""A type representing the structuring of grouping the diagnostics among the grids.

"""
SubmatrixHandlerType: TypeAlias = Callable[[spmatrix, str, str], float]
"""A type representing the diagnostics handler function to be applied to the submatrix.

"""
DiagnosticsData: TypeAlias = "dict[tuple[int, int], dict[str, Any]]"
"""A type representing the diagnostics data for each submatrix in the Jacobian.

The key represents the pair (row, column) of the block. The value is a dictionary of all
diagnostical values collected for this submatrix -- names and values.

"""


class DiagnosticsMixin:
    """This is a mixin class for a PorePy model that allows running the model
    diagnostics.

    Currently supports dividing the matrix into blocks based on different equations and
    subdomains/interfaces. Can plot condition number of each block and absolute maximum
    value of each block. Supports grouping grids arbitrarily. For detailed information,
    see the `tutorial <https://github.com/pmgbergen/porepy/blob/feat-diagnostics/tutoria
    ls/diagnostics.ipynb>`__.

    Example:
        Basic usage:

        >>> import porepy
        >>> from porepy.applications.verification_setups.mandel_biot import MandelSetup
        >>> from porepy.viz.diagnostics_mixin import DiagnosticsMixin
        >>>
        >>> class MandelDiagnostics(DiagnosticsMixin, MandelSetup):
        >>>     pass
        >>> setup = MandelDiagnostics(params={})
        >>> porepy.run_time_dependent_model(setup, params={})
        >>> setup.run_diagnostics()

    """

    # The mixin expects the model to have these properties defined.
    linear_system: tuple[csr_matrix, np.ndarray]
    equation_system: EquationSystem
    mdg: MixedDimensionalGrid

    def run_diagnostics(
        self,
        grouping: GridGroupingType
        | Literal["dense", "subdomains", "interfaces"]
        | None = None,
        is_plotting_condition_number: bool = False,
        is_plotting_max: bool = True,
        additional_handlers: Optional[dict[str, SubmatrixHandlerType]] = None,
    ) -> DiagnosticsData:
        """Collects and plots diagnostics from the last assembled Jacobian matrix stored
        in :attr:`~self.linear_system`.

        It is assumed that the full Jacobian matrix is stored in `self.linear_system`,
        and full information about the matrix indices is stored in
        `self.equation_system._equation_image_space_composition`.

        Note:
            It is assumed that variables with the same name defined on different grids
            are semantically the same variable.

        Args:
            grouping (optional): Supports gathering the data related to one equation /
                variable from grids of interest. For this, pass a list of grid blocks.
                Each grid block must be a list of grids. Pass `None` to treat all grids
                separately. Pass `'dense'` to group all grids -- useful to look at the
                equation in general, when you are not interested in specific grids. Pass
                `'subdomains'` or `'interfaces'` to investigate only subdomains or
                interfaces respectively.
            is_plotting_condition_number (optional): Whether to plot
                the condition number. Caution - this might take time if the matrix
                is big. Defaults to False.
            is_plotting_max (optional): Whether to plot the absolute maximum.
                This option is computationally cheap even for big matrices.
            additional_handlers (optional): A dictionary of additional functions to be
                applied to the submatrices. The keys represent the name of the handler.
                The values are the functions with the arguments: the matrix, the
                equation name and the variable name. The equation and the variable names
                are passed to allow the user for calculating the handler value only in a
                subset of equations and variables of interest.

        Returns:
            A dictionary with keys corresponding to block index and values of
            diagnostics data collected for this block.

        """

        def get_singular_val(mat: spmatrix, which: Literal["LM", "SM"]) -> float:
            # Helper function to get the largest or the smallest singular values.
            return svds(mat, k=1, return_singular_vectors=False, which=which).item()

        def get_condition_number(
            mat: spmatrix, equation_name: str, variable_name: str
        ) -> float:
            largest_sing = get_singular_val(mat, which="LM")
            # Smallest singular value might be zero.
            smallest_sing = get_singular_val(mat, which="SM")
            if mat.data.size == 0:
                return 0.0
            else:
                if smallest_sing < 1e-9:
                    return float("inf")
                return float(largest_sing / smallest_sing)

        def get_max(mat: spmatrix, equation_name: str, variable_name: str) -> float:
            return abs(mat.data).max()

        if additional_handlers is None:
            additional_handlers = {}

        add_grid_info = True
        if grouping is None:
            # We want all the grids to be treated separately.
            grouping = [
                [grid] for grid in self.mdg.subdomains() + self.mdg.interfaces()
            ]
        elif grouping == "dense":
            # We don't care about grids.
            add_grid_info = False
            grouping = [
                [grid for grid in self.mdg.subdomains() + self.mdg.interfaces()]
            ]
        elif grouping == "subdomains":
            grouping = [[grid] for grid in self.mdg.subdomains()]
        elif grouping == "interfaces":
            grouping = [[grid] for grid in self.mdg.interfaces()]
        if not isinstance(grouping, Iterable):
            raise ValueError(
                f"grouping must be a list of lists of GridLike, got "
                f"{grouping=} instead."
            )
        # The type is checked one line earlier.
        grouping_: GridGroupingType = grouping  # type: ignore

        # Listing all the handlers to be applied to the submatrices.
        active_handlers: dict[str, SubmatrixHandlerType] = additional_handlers.copy()
        if is_plotting_condition_number:
            active_handlers["Block condition number"] = get_condition_number
        if is_plotting_max:
            active_handlers["Absolute maximum value"] = get_max

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
        equation_data = self._equations_data(
            grouping=grouping_, add_grid_info=add_grid_info
        )
        variable_data = self._variable_data(
            grouping=grouping_, add_grid_info=add_grid_info
        )

        submatrices = self._collect_submatrices(
            mat=full_matrix,
            equation_indices=[equ["block_dofs"] for equ in equation_data],
            variable_indices=[var["block_dofs"] for var in variable_data],
        )

        # Computing the required features of each block.
        block_matrix_shape = len(equation_data), len(variable_data)
        block_data: DiagnosticsData = defaultdict(dict)
        for i, j in product(range(block_matrix_shape[0]), range(block_matrix_shape[1])):
            submat = submatrices[i, j]
            is_empty_block = submat.data[submat.data != 0].size == 0
            equation_name = equation_data[i]["equation_name"]
            variable_name = variable_data[j]["variable_name"]

            # Adding information about each block. Might be helpful for external use.
            block_data[i, j]["is_empty_block"] = is_empty_block
            block_data[i, j]["variable_name"] = variable_name
            block_data[i, j]["equation_name"] = equation_name
            block_data[i, j]["equation_printed_name"] = equation_data[i]["printed_name"]
            block_data[i, j]["variable_printed_name"] = variable_data[j]["printed_name"]
            block_data[i, j]["block_dofs_row"] = equation_data[i]["block_dofs"]
            block_data[i, j]["block_dofs_col"] = variable_data[j]["block_dofs"]

            if not is_empty_block:
                for handler_name, handler in active_handlers.items():
                    block_data[i, j][handler_name] = handler(
                        submat, equation_name, variable_name
                    )

        # Plotting the figures.
        if is_plotting_condition_number:
            self.plot_diagnostics(
                block_data,
                key="Block condition number",
                norm=matplotlib.colors.LogNorm(vmin=1, vmax=1e3),
            )
            plt.show()
        if is_plotting_max:
            self.plot_diagnostics(
                block_data,
                key="Absolute maximum value",
            )
            plt.show()

        return block_data

    def plot_diagnostics(
        self, diagnostics_data: DiagnosticsData, key: str, **kwargs
    ) -> None:
        """Plots the heatmap of diagnostics data for the block matrix.

        Plotting the image requires seaborn. If not available, falls back to text
        printing. **kwargs are passed to Seaborn to modify image style.

        Args:
            diagnostics_data: The return value of :meth:`~self.run_diagnostics`.
            key: The key of diagnostics entry to be plotted.
            **kwargs: Passed to Seaborn.

        """
        row_names: list[str] = []
        col_names: list[str] = []

        # Collecting unique row and column names.
        i_prev, j_prev = -1, -1
        for (i, j), data in diagnostics_data.items():
            if i > i_prev:
                row_names.append(data["equation_printed_name"])
                i_prev = i
            if j > j_prev:
                col_names.append(data["variable_printed_name"])
                j_prev = j

        # Collecting values to be plotted.
        block_data = np.zeros((len(row_names), len(col_names)))
        mask = np.zeros((len(row_names), len(col_names)), dtype=bool)
        for (i, j), data in diagnostics_data.items():
            mask[i, j] = (is_empty := data["is_empty_block"])
            if not is_empty:
                # We expect this to be a number, as handlers must return float.
                block_data[i, j] = data[key]  # Type: ignore

        if _IS_SEABORN_AVAILABLE:
            # These are the default plotting argumenst.
            plot_kwargs = (
                dict(
                    mask=mask,
                    square=False,
                    annot=True,
                    norm=matplotlib.colors.LogNorm(),
                    fmt=".1e",
                    xticklabels=col_names,
                    yticklabels=row_names,
                    linewidths=0.01,
                    linecolor="grey",
                    cbar=False,
                    cmap=sns.color_palette("coolwarm", as_cmap=True),
                )
                | kwargs
            )  # Updating them with custom plotting arguments provided by user.
            ax = sns.heatmap(block_data, **plot_kwargs)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title(key)
        else:
            print(np.array_str(block_data, precision=2))

    def _equations_data(
        self, grouping: GridGroupingType, add_grid_info: bool
    ) -> tuple[dict[str, Any], ...]:
        """Collects the indices of equations presented in the Jacobian matrix as rows.

        Returns:
            A tuple with dictionaries -- data of each row in the block matrix. The keys
            are: "block_dofs", "grids", "printed_name", "equation_name".

        """
        equation_indices = []

        assembled_equation_indices = self.equation_system.assembled_equation_indices

        # `_equation_image_space_composition` has dof indices starting from zero for
        # each equation. We need to count the offset to get the global dof indices.
        block_indices = 0

        for eq_name, eq_dof_indices in assembled_equation_indices.items():
            equation_image_space = (
                self.equation_system._equation_image_space_composition[eq_name]
            )
            # Forming a block from required grids.
            for block_of_grids in grouping:
                block_dofs = []
                for grid, grid_dof_indices in equation_image_space.items():

                    if grid not in block_of_grids:
                        continue
                    block_dofs.extend(grid_dof_indices.tolist())

                if len(block_dofs) == 0:
                    # This equation is not present on the grids of interest.
                    continue

                printed_name = _format_ticks(
                    name=eq_name,
                    block_of_grids=block_of_grids,
                    add_grid_info=add_grid_info,
                )
                # Adding the offset to form the global dofs of this block.
                block_dofs_array = np.array(block_dofs).flatten() + block_indices
                # Sanity check - we're still within the dofs of this equation.
                assert np.all(np.isin(block_dofs_array, eq_dof_indices))

                equation_indices.append(
                    {
                        "block_dofs": block_dofs_array,
                        "printed_name": printed_name,
                        "equation_name": eq_name,
                        "grids": block_of_grids,
                    }
                )

            block_indices += eq_dof_indices.size

        return tuple(equation_indices)

    def _variable_data(
        self, grouping: GridGroupingType, add_grid_info: bool
    ) -> tuple[dict[str, Any], ...]:
        """Collects the indices of variables presented in the Jacobian matrix as
        columns.

        Returns:
            A tuple with dictionaries -- data of each column in the block matrix. The
            keys are: "block_dofs", "grids", "printed_name", "variable_name".

        """
        variable_indices = []

        # First, we group variables by names. We assume that variables with the same
        # name are one variable on multiple grids.
        names_to_variables: dict[str, list[Variable]] = defaultdict(list)
        for variable in self.equation_system.variables:
            names_to_variables[variable.name].append(variable)
        names_to_variables = dict(names_to_variables)

        for variable_name, variable_on_grids in names_to_variables.items():
            for block_of_grids in grouping:
                variables_of_interest = []
                # We add the variable to the list only on the grids of this block.
                for variable_on_grid in variable_on_grids:
                    grids: list[GridLike] = (
                        variable_on_grid.subdomains + variable_on_grid.interfaces
                    )
                    assert len(grids) == 1, (
                        "Something changed in how we store variables in the equation"
                        "system."
                    )
                    if grids[0] in block_of_grids:
                        variables_of_interest.append(variable_on_grid)

                dofs = self.equation_system.dofs_of(variables_of_interest)
                if len(dofs) == 0:
                    # This variable is not present on the grids of interest.
                    continue
                printed_name = _format_ticks(
                    name=variable_name,
                    block_of_grids=block_of_grids,
                    add_grid_info=add_grid_info,
                )
                # variable_indices[printed_name] = dofs
                variable_indices.append(
                    {
                        "block_dofs": dofs,
                        "printed_name": printed_name,
                        "variable_name": variable_name,
                        "grids": block_of_grids,
                    }
                )
        return tuple(variable_indices)

    def _collect_submatrices(
        self,
        mat: spmatrix,
        equation_indices: list[np.ndarray],
        variable_indices: list[np.ndarray],
    ) -> dict[tuple[int, int], spmatrix]:
        """Slices the Jacobian matrix into block based on provided equations and
        variables.

        Args:
            mat: The full Jacobian matrix.
            equation_indices: The dictionary with keys - printed names of equations and
                values - indices of these equations in the Jacobian. (Rows index).
            variable_indices: The dictionary with keys - printed names of variables and
                values - indices of these variables in the Jacobian (Columns index).
        Returns:
            The dictionary with submatrices.

        """
        submatrices = {}

        for i, ind_row in enumerate(equation_indices):
            for j, ind_col in enumerate(variable_indices):
                submatrix_indices = np.meshgrid(
                    ind_row, ind_col, sparse=True, indexing="ij"
                )
                submatrices[i, j] = mat[tuple(submatrix_indices)]

        return submatrices


def _format_ticks(
    name: str, block_of_grids: list[GridLike], add_grid_info: bool
) -> str:
    """Formats variable or equation name for printing.

    Returns:
        Name to be printed.

    """
    if add_grid_info:
        if len(block_of_grids) == 1:
            grid = block_of_grids[0]
            if isinstance(grid, MortarGrid):
                grid_name = f"{grid.dim}D, intf. id={grid.id}"
            else:
                grid_name = f"{grid.dim}D, id={grid.id}"
        elif len(block_of_grids) > 1:
            data = []
            for grid in block_of_grids:
                if isinstance(grid, MortarGrid):
                    data.append(f"intf. {grid.id}")
                else:
                    data.append(str(grid.id))
            grid_name = "grids: [" + ",".join(data) + "]"

        name = f"{name} {grid_name}"
    return name
