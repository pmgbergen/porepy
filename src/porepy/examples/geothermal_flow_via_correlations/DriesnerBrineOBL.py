import pyvista
import numpy as np
import time


class DriesnerBrineOBL:
    # Operator-based linearization (OBL) for Driesner correlations

    def __init__(self, file_name):
        self.file_name = file_name
        self.__build_search_space()

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
        point_cloud = pyvista.PolyData(points)
        tb = time.time()
        self.sampled_could = point_cloud.sample(self._search_space)
        te = time.time()
        print("DriesnerBrineOBL:: Sampled n_points: ", len(points))
        print("DriesnerBrineOBL:: Time for sampling: ", te - tb)

    def __build_search_space(self):
        tb = time.time()
        self._search_space = pyvista.read(self.file_name)
        te = time.time()
        print("DriesnerBrineOBL:: Time for loading search_space: ", te - tb)
