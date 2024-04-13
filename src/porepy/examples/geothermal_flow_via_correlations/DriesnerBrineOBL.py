import time

import numpy as np
import pyvista


class DriesnerBrineOBL:
    # Operator-based linearization (OBL) for Driesner correlations

    def __init__(self, file_name):
        self.file_name = file_name
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
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, file_name):
        self._file_name = file_name

    @property
    def search_space(self):
        return self._search_space

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

    def sample_at(self, points):
        # tb = time.time()
        points = self._apply_conversion_factor(points)
        point_cloud = pyvista.PolyData(points)
        self.sampled_could = point_cloud.sample(self._search_space)
        self._apply_conversion_factor_on_gradients()
        # te = time.time()
        # print("DriesnerBrineOBL:: Sampled n_points: ", len(points))
        # print("DriesnerBrineOBL:: Time for sampling: ", te - tb)

    def _apply_conversion_factor(self, points):
        for i, scale in enumerate(self.conversion_factors):
            points[:, i] *= scale
        return points

    def _apply_conversion_factor_on_gradients(self):
        for name, grad in self.sampled_could.point_data.items():
            if name.startswith("grad_"):
                for i, scale in enumerate(self.conversion_factors):
                    grad[:, i] *= scale
        return

    def __build_search_space(self):
        tb = time.time()
        self._search_space = pyvista.read(self.file_name)
        te = time.time()
        print("DriesnerBrineOBL:: Time for loading interpolation space: ", te - tb)
