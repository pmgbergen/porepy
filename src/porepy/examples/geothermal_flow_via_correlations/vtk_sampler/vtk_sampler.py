import time

import numpy as np
import pyvista
import sampler.classify_points as cp


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

    def sample_at(self, points):
        # tb = time.time()
        points = self._apply_conversion_factor(points)

        point_cloud = pyvista.PolyData(points)
        self.sampled_could = point_cloud.sample(self._search_space)
        check_enclosed_points = point_cloud.select_enclosed_points(
            self.bc_surface, check_surface=False
        )
        external_idx = np.logical_not(
            check_enclosed_points.point_data["SelectedPoints"]
        )
        if self.taylor_extended_q:
            self.__taylor_expansion(points, external_idx)

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
        self._bc_surface = self._search_space.extract_surface()
        te = time.time()
        print("DriesnerBrineOBL:: Time for loading interpolation space: ", te - tb)

    def __map_external_points_to_surface(self, xv):
        bounds = self.search_space.bounds
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        # ensure that vtk sampling for zero order expansion is performed internally
        eps = 1.0e-10
        xmin += eps
        ymin += eps
        zmin += eps

        xmax -= eps
        ymax -= eps
        zmax -= eps

        # detect regions
        w_q = cp.w_predicate(*xv.T, bounds)
        e_q = cp.e_predicate(*xv.T, bounds)
        s_q = cp.s_predicate(*xv.T, bounds)
        n_q = cp.n_predicate(*xv.T, bounds)
        b_q = cp.b_predicate(*xv.T, bounds)
        t_q = cp.t_predicate(*xv.T, bounds)

        # x range
        sb_q = cp.sb_predicate(*xv.T, bounds)
        nb_q = cp.nb_predicate(*xv.T, bounds)
        st_q = cp.st_predicate(*xv.T, bounds)
        nt_q = cp.nt_predicate(*xv.T, bounds)

        # y range
        wb_q = cp.wb_predicate(*xv.T, bounds)
        eb_q = cp.eb_predicate(*xv.T, bounds)
        wt_q = cp.wt_predicate(*xv.T, bounds)
        et_q = cp.et_predicate(*xv.T, bounds)

        # z range
        ws_q = cp.ws_predicate(*xv.T, bounds)
        es_q = cp.es_predicate(*xv.T, bounds)
        wn_q = cp.wn_predicate(*xv.T, bounds)
        en_q = cp.en_predicate(*xv.T, bounds)

        # bottom vertices
        wsb_q = cp.wsb_predicate(*xv.T, bounds)
        esb_q = cp.esb_predicate(*xv.T, bounds)
        wnb_q = cp.wnb_predicate(*xv.T, bounds)
        enb_q = cp.enb_predicate(*xv.T, bounds)

        # top vertices
        wst_q = cp.wst_predicate(*xv.T, bounds)
        est_q = cp.est_predicate(*xv.T, bounds)
        wnt_q = cp.wnt_predicate(*xv.T, bounds)
        ent_q = cp.ent_predicate(*xv.T, bounds)

        # map points to surface
        xv[w_q, 0] = xmin
        xv[e_q, 0] = xmax
        xv[s_q, 1] = ymin
        xv[n_q, 1] = ymax
        xv[b_q, 2] = zmin
        xv[t_q, 2] = zmax

        # x range
        xv[sb_q, 1] = ymin
        xv[sb_q, 2] = zmin
        xv[nb_q, 1] = ymax
        xv[nb_q, 2] = zmin
        xv[st_q, 1] = ymin
        xv[st_q, 2] = zmax
        xv[nt_q, 1] = ymax
        xv[nt_q, 2] = zmax

        # y range
        xv[wb_q, 0] = xmin
        xv[wb_q, 2] = zmin
        xv[eb_q, 0] = xmax
        xv[eb_q, 2] = zmin
        xv[wt_q, 0] = xmin
        xv[wt_q, 2] = zmax
        xv[et_q, 0] = xmax
        xv[et_q, 2] = zmax

        # z range
        xv[ws_q, 0] = xmin
        xv[ws_q, 1] = ymin
        xv[es_q, 0] = xmax
        xv[es_q, 1] = ymin
        xv[wn_q, 0] = xmin
        xv[wn_q, 1] = ymax
        xv[en_q, 0] = xmax
        xv[en_q, 1] = ymax

        # bottom vertices
        xv[wsb_q, 0] = xmin
        xv[wsb_q, 1] = ymin
        xv[wsb_q, 2] = zmin
        xv[esb_q, 0] = xmax
        xv[esb_q, 1] = ymin
        xv[esb_q, 2] = zmin

        xv[wnb_q, 0] = xmin
        xv[wnb_q, 1] = ymax
        xv[wnb_q, 2] = zmin
        xv[enb_q, 0] = xmax
        xv[enb_q, 1] = ymax
        xv[enb_q, 2] = zmin

        # top vertices
        xv[wst_q, 0] = xmin
        xv[wst_q, 1] = ymin
        xv[wst_q, 2] = zmax
        xv[est_q, 0] = xmax
        xv[est_q, 1] = ymin
        xv[est_q, 2] = zmax

        xv[wnt_q, 0] = xmin
        xv[wnt_q, 1] = ymax
        xv[wnt_q, 2] = zmax
        xv[ent_q, 0] = xmax
        xv[ent_q, 1] = ymax
        xv[ent_q, 2] = zmax

    def __taylor_expansion(self, points, external_idx):

        xv = points[external_idx].copy()
        self.__map_external_points_to_surface(xv)

        # compute data for zero order expansion
        epoint_cloud = pyvista.PolyData(xv)
        sampled_could = epoint_cloud.sample(self._search_space)

        # for all the fields
        glob_idx = np.nonzero(external_idx)[0]
        x = points[external_idx]

        for grad_field_name, grad in self.sampled_could.point_data.items():
            if grad_field_name.startswith("grad_"):
                field_name = grad_field_name.lstrip("grad_")
                fv = sampled_could[field_name]
                grad_fv = sampled_could[grad_field_name]

                # taylor expansion all at once
                f_extrapolated = fv + np.sum(grad_fv * (x - xv), axis=1)

                # update fields
                self.sampled_could[field_name][glob_idx] = f_extrapolated
                self.sampled_could[grad_field_name][glob_idx] = grad_fv

        return
