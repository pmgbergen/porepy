import time
import numpy as np
import pyvista
from functools import cached_property
from typing import Tuple


class VTKSampler:
    """Interpolation of a structured VTK property table at query points.

    Query points are scaled by :attr:`conversion_factors` and shifted by
    :attr:`translation_factors` into table coordinates. With ``extend_q``, points
    outside the table take the values at their clipped positions on its surface.

    """

    def __init__(
        self, file_name: str, extend_q: bool = True, first_order_expansion: bool = False
    ) -> None:
        """Create the sampler and load the table.

        Parameters:
            file_name: Path to the VTK table.
            extend_q: Clip points outside the table to its surface instead of returning zeros.
            first_order_expansion: Add a first-order Taylor correction to the clipped values.

        """
        self.file_name = file_name
        self._extend_q = extend_q
        self._conversion_factors = (1.0, 1.0, 1.0)  # Default values
        self._translation_factors = (0.0, 0.0, 0.0)  # Default values
        self._first_order_expansion = first_order_expansion
        self._sampled_cloud = None
        # self._load_search_space() #TODO: already done in the property by filename

    @property
    def conversion_factors(self) -> Tuple[float, float, float]:
        """Scaling applied to the query coordinates and to the ``grad_*`` fields."""
        return self._conversion_factors

    @conversion_factors.setter
    def conversion_factors(
        self, conversion_factors: Tuple[float, float, float]
    ) -> None:
        # TODO: Add sanity checks for the length of conversion_factors
        self._conversion_factors = conversion_factors

    @property
    def translation_factors(self) -> Tuple[float, float, float]:
        """Offset added to the query coordinates after scaling."""
        return self._translation_factors

    @translation_factors.setter
    def translation_factors(
        self, translation_factors: Tuple[float, float, float]
    ) -> None:
        # TODO: Add sanity checks for the length of conversion_factors
        self._translation_factors = translation_factors

    @property
    def file_name(self) -> str:
        """Path to the VTK table; setting it reloads the table."""
        return self._file_name

    @file_name.setter
    def file_name(self, file_name: str) -> None:
        self._file_name = file_name
        self._refresh()

    @cached_property
    def search_space(self):
        """The table, read from :attr:`file_name`."""
        return pyvista.read(self.file_name)

    @cached_property
    def bc_surface(self):
        """Outer surface of the table."""
        return self.search_space.extract_surface()

    @cached_property
    def bc_surface_bounds(self):  # -> pyvista.BoundsLike:
        """Bounds ``(x_min, x_max, y_min, y_max, z_min, z_max)`` of the table surface."""
        return self.bc_surface.bounds

    @property
    def sampled_cloud(self):
        """Point cloud holding the values interpolated by the last :meth:`sample_at` call."""
        return self._sampled_cloud

    @sampled_cloud.setter
    def sampled_cloud(self, sampled_cloud):
        if self._sampled_cloud is not None:
            self._sampled_cloud.clear_data()
        self._sampled_cloud = (
            sampled_cloud.copy() if sampled_cloud is not None else None
        )

    def _load_search_space(self):
        """Load the table and its surface bounds into the cache, and print the load time."""
        tb = time.time()
        _ = self.search_space  # This will trigger the cached_property to load the data
        _ = (
            self.bc_surface_bounds
        )  # This will also trigger the cached_property for _bc_surface_bounds
        te = time.time()
        print("VTKSampler:: Time for loading interpolation space: ", te - tb)

    def sample_at(self, points: np.ndarray):  # points: z,T,p // z,h,p
        """Interpolate the table at query points; the result is stored in :attr:`sampled_cloud`.

        Parameters:
            points: Array of shape ``(n, 3)`` in the solver's coordinates.

        """
        points = self._apply_conversions(points)
        # points = self._apply_translation_factor(points)

        point_cloud = pyvista.PolyData(points)
        # if point_cloud fall outside the search space then interpolated values will be zero.
        self.sampled_cloud = point_cloud.sample(
            self.search_space
        )  # Interpolation region

        if self._extend_q:
            is_external = self._capture_points_outside_bounds(points)
            if np.any(is_external):
                self._extrapolate(points, is_external)

        self._apply_conversion_factor_on_gradients()

    def _apply_conversions(self, points):
        """Scale and shift query points into table coordinates.

        Parameters:
            points: Array of shape ``(n, 3)``.

        Returns:
            The converted points.

        """
        return points * self._conversion_factors + self._translation_factors

    def _refresh(self) -> None:
        """Clear the cached table and reload it from :attr:`file_name`."""
        for attr in ["search_space", "bc_surface", "bc_surface_bounds"]:
            if attr in self.__dict__:
                del self.__dict__[attr]
        self._load_search_space()

    def _apply_conversion_factor_on_gradients(self):
        """Scale the ``grad_*`` fields of the sampled cloud by the conversion factors."""
        if self._sampled_cloud is None:
            return

        grad_fields = [
            name for name in self._sampled_cloud.point_data if name.startswith("grad_")
        ]
        for name in grad_fields:
            grad = self._sampled_cloud[name]
            grad *= (
                self._conversion_factors
            )  # Element-wise multiply each component (uses broadcasting)

    def _capture_points_outside_bounds(self, points: np.ndarray) -> np.ndarray:
        """Flag points outside the bounds of the table surface.

        Parameters:
            points: Array of shape ``(n, 3)`` in table coordinates.

        Returns:
            Boolean mask, True for points outside the bounds.

        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.bc_surface_bounds
        # Create a mask for points that are within the bounds
        within_bounds = (
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)  # x bound
            & (points[:, 1] >= y_min)
            & (points[:, 1] <= y_max)  # y bound
            & (points[:, 2] >= z_min)
            & (points[:, 2] <= z_max)  # z bound
        )

        # Return points that lie outside the bounds
        return ~within_bounds

    def _map_external_points_to_surface(self, points: np.ndarray) -> None:
        """Clip points, in place, to just inside the table bounds.

        Parameters:
            points: Array of shape ``(n, 3)`` in table coordinates, modified in place.

        """
        # Get the bounds of the search space
        xmin, xmax, ymin, ymax, zmin, zmax = self.search_space.bounds

        # Add a small epsilon to ensure points are not exactly at the boundary
        eps_x = 1.0e-4
        eps_y = 1.0
        eps_z = 1.0

        xmin += 0.0 if xmin == xmax else eps_x
        ymin += eps_y
        zmin += eps_z
        xmax -= 0.0 if xmin == xmax else eps_x
        ymax -= eps_y
        zmax -= eps_z

        # Clip points to the nearest boundary within the bounds
        points[:, 0] = np.clip(points[:, 0], xmin, xmax)  # Clip x-coordinates
        points[:, 1] = np.clip(points[:, 1], ymin, ymax)  # Clip y-coordinates
        points[:, 2] = np.clip(points[:, 2], zmin, zmax)  # Clip z-coordinates

    def _extrapolate(self, points: list[np.ndarray], external_idx: np.ndarray) -> None:
        """Fill the values at points outside the table from their clipped positions on its surface.

        With ``first_order_expansion``, a first-order Taylor correction from the stored
        gradients is added.

        Parameters:
            points: Array of shape ``(n, 3)`` in table coordinates.
            external_idx: Boolean mask of the points outside the table.

        """
        glob_idx = np.nonzero(external_idx)[0]
        # xv = points[external_idx].copy()
        xv = points[external_idx]
        self._map_external_points_to_surface(xv)

        # Compute data for zero order expansion
        epoint_cloud = pyvista.PolyData(xv)
        sampled_cloud = epoint_cloud.sample(self.search_space)

        # Build a list of gradient field names, iterate over these and populate the
        # extrapolated values.
        keys = [k for k in sampled_cloud.point_data if k.startswith("grad_")]
        for grad_field_name in keys:
            field_name = grad_field_name[5:]  # More efficient than lstrip
            f_extrapolated = sampled_cloud[field_name]
            grad_fv = sampled_cloud[grad_field_name]

            # update fields
            # First order extrapolation using Taylor expansion
            if self._first_order_expansion:
                f_extrapolated += np.sum(grad_fv * (points[external_idx] - xv), axis=1)
            self.sampled_cloud[field_name][glob_idx] = f_extrapolated
            self.sampled_cloud[grad_field_name][glob_idx] = grad_fv
