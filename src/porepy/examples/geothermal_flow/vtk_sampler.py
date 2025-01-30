import time
import numpy as np
import pyvista


class VTKSampler:

    def __init__(self, file_name, extended_q=True):
        self.file_name = file_name
        self.taylor_extended_q = extended_q
        self.__build_search_space()

    @property
    def conversion_factors(self):
        if hasattr(self, "_conversion_factors"):
            return self._conversion_factors
        else:
            return (1.0, 1.0, 1.0)  # No conversion

    @conversion_factors.setter
    def conversion_factors(self, conversion_factors):
        self._conversion_factors = conversion_factors

    @property
    def translation_factors(self):
        if hasattr(self, "_translation_factors"):
            return self._translation_factors
        else:
            return (0.0, 0.0, 0.0)  # No translation

    @translation_factors.setter
    def translation_factors(self, translation_factors):
        self._translation_factors = translation_factors

    @property
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, file_name):
        self._file_name = file_name

    @property
    def search_space(self):
        return self._search_space

    @property
    def bc_surface(self):
        return self._bc_surface

    @property
    def sampled_could(self):
        if hasattr(self, "_sampled_could"):
            return self._sampled_could
        else:
            return None

    @sampled_could.setter
    def sampled_could(self, sampled_could):
        if hasattr(self, "_sampled_could"):
            self._sampled_could.clear_data()
        self._sampled_could = sampled_could.copy()

    def sample_at(self, points):  # points: z,T,p // z,h,p
        points = self._apply_conversion_factor(points)
        points = self._apply_translation_factor(points)

        point_cloud = pyvista.PolyData(points)
        # if point_cloud fall outside the search space then interpolated values will be zero.
        self.sampled_could = point_cloud.sample(self._search_space)  # Interpolation region
        check_enclosed_points = point_cloud.select_enclosed_points(
            self.bc_surface, 
            check_surface=False,
            tolerance=1.0e-7,
        )
        external_idx = np.logical_not(
            check_enclosed_points.point_data["SelectedPoints"]
        )
        if self.taylor_extended_q:
            self.__taylor_expansion(points, external_idx)

        self._apply_conversion_factor_on_gradients()

    def _apply_conversion_factor(self, points):
        for i, scale in enumerate(self.conversion_factors):
            points[:, i] *= 1.0 * scale
        return points

    def _apply_translation_factor(self, points):
        for i, translation in enumerate(self.translation_factors):
            points[:, i] += translation
        return points

    def _apply_conversion_factor_on_gradients(self):
        for name, grad in self.sampled_could.point_data.items():
            if name.startswith("grad_"):
                for i, scale in enumerate(self.conversion_factors):
                    grad[:, i] *= 1.0 * scale
        return

    def __build_search_space(self):
        tb = time.time()
        # this is the grid to interpolate on
        self._search_space = pyvista.read(self.file_name)  # grid with field data
        self._bc_surface = self._search_space.extract_surface()
        te = time.time()
        print("VTKSampler:: Time for loading interpolation space: ", te - tb)

    def capture_points_outside_bounds(self, points):
        """
        Function to capture points lying outside the specified bounds.
        Args:
            points (numpy.ndarray): Nx3 array of points (x, y, z)
            bounds (tuple): A 6-tuple representing the bounds 
            (x_min, x_max, y_min, y_max, z_min, z_max)  
        Returns:
            numpy.ndarray: An array of points that lie outside the bounds
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self._search_space.bounds 
        # Create a mask for points that are within the bounds
        within_bounds = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &  # x bound
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &  # y bound
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)    # z bound
        )
        # Invert the mask to get points outside the bounds
        outside_bounds = ~within_bounds
        # Return points that lie outside the bounds
        return points[outside_bounds]

    def __map_external_points_to_surface(self, xv):
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
        xv[:, 0] = np.clip(xv[:, 0], xmin, xmax)  # Clip x-coordinates
        xv[:, 1] = np.clip(xv[:, 1], ymin, ymax)  # Clip y-coordinates
        xv[:, 2] = np.clip(xv[:, 2], zmin, zmax)  # Clip z-coordinates

    def __taylor_expansion(self, points, external_idx):

        xv = points[external_idx].copy()
        self.__map_external_points_to_surface(xv)
        # compute data for zero order expansion
        epoint_cloud = pyvista.PolyData(xv)
        sampled_could = epoint_cloud.sample(self._search_space)
        # for all the fields
        glob_idx = np.nonzero(external_idx)[0]
        # x = points[external_idx]

        for grad_field_name, grad in self.sampled_could.point_data.items():
            if grad_field_name.startswith("grad_"):
                field_name = grad_field_name.lstrip("grad_")
                fv = sampled_could[field_name]
                grad_fv = sampled_could[grad_field_name]

                # taylor expansion all at once
                f_extrapolated = fv  # + np.sum(grad_fv * (x - xv), axis=1)
                # update fields
                self.sampled_could[field_name][glob_idx] = f_extrapolated
                self.sampled_could[grad_field_name][glob_idx] = grad_fv        
        return
