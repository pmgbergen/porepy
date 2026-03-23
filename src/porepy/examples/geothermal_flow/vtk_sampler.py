import time

import numpy as np
import pyvista  # type:ignore[import-not-found]

from . import classify_points as cp


class VTKSampler:
    def __init__(self, file_name, extended_q=True):
        self.file_name = file_name
        self.taylor_extended_q = extended_q
        self.__build_search_space()

    @property
    def mutex_state(self):
        if hasattr(self, "_mutex_state"):
            return self._mutex_state
        else:
            return False  # Able to modify

    @mutex_state.setter
    def mutex_state(self, mutex_state):
        self._mutex_state = mutex_state

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
    def boundary_surface(self):
        return self._boundary_surface

    @property
    def sampled_could(self):
        if hasattr(self, "_sampled_could"):
            return self._sampled_could
        else:
            return None

    @sampled_could.setter
    def sampled_could(self, sampled_could):
        # Avoid calling dataset-specific methods like .clean() which are not
        # present on all pyvista dataset types (e.g. RectilinearGrid). Always
        # create a deep copy of the incoming dataset and replace any existing
        # one. If an old dataset exists, delete the reference to help free
        # memory.
        if hasattr(self, "_sampled_could"):
            try:
                # Attempt to explicitly free resources if a clear-like
                # method exists (some datasets provide `.clean()`), but don't
                # assume it is present.
                cleanup = getattr(self._sampled_could, "clean", None)
                if callable(cleanup):
                    cleanup()
            except Exception:
                # Ignore any errors — we will replace the reference below.
                pass
            # Replace with a deep copy of the new dataset
            try:
                self._sampled_could = sampled_could.copy(deep=True)
            except Exception:
                # Fallback: assign directly if copy fails
                self._sampled_could = sampled_could
        else:
            try:
                self._sampled_could = sampled_could.copy(deep=True)
            except Exception:
                self._sampled_could = sampled_could
        # Remove caller reference
        try:
            del sampled_could
        except Exception:
            pass

    @property
    def constant_extended_fields(self):
        if hasattr(self, "_constant_extended_fields"):
            return self._constant_extended_fields
        else:
            return []

    @constant_extended_fields.setter
    def constant_extended_fields(self, constant_extended_fields):
        self._constant_extended_fields = constant_extended_fields

    def sample_at(self, points):
        if self.mutex_state and self.sampled_could is not None:
            return
        x_par = points.copy()
        self._apply_conversion_factor(x_par)
        self._apply_translation_factor(x_par)

        # Clamp coordinates that are exactly on the bounds slightly inside the
        # search space using a dedicated helper.
        self._clamp_points_inside_bounds(x_par)

        point_cloud = pyvista.PolyData(x_par)
        # Sample the VTK dataset at the provided point cloud so the returned
        # dataset has one entry per input point.
        self.sampled_could = point_cloud.sample(self._search_space)
        external_idx = self.__points_out_side_parametric_space(x_par)
        if self.taylor_extended_q:
            self.__taylor_expansion(x_par, external_idx)

        self._apply_conversion_factor_on_gradients()

    def _apply_conversion_factor(self, points):
        for i, scale in enumerate(self.conversion_factors):
            points[:, i] *= scale
        return points

    def _apply_translation_factor(self, points):
        for i, translation in enumerate(self.translation_factors):
            points[:, i] += translation
        return points

    def _apply_conversion_factor_on_gradients(self):
        for name, grad in self.sampled_could.point_data.items():
            if name.startswith("grad_"):
                for i, scale in enumerate(self.conversion_factors):
                    grad[:, i] *= scale
        return

    def __release_memory_of(self, point_cloud):
        point_cloud.clean()
        del point_cloud

    def release_memory(self):
        self.__release_memory_of(self._search_space)
        self.__release_memory_of(self._boundary_surface)

    def __build_search_space(self):
        tb = time.time()
        self._search_space = pyvista.read(self.file_name)
        # If data arrays are stored as cell_data instead of point_data, convert
        # them to point_data so that probing (sampling) at arbitrary points
        # returns interpolated scalars/gradients. Some Driesner VTK files
        # contain cell-centered values.
        try:
            if (not bool(self._search_space.point_data)) and bool(self._search_space.cell_data):
                # Convert cell data to point data (creates a new dataset)
                self._search_space = self._search_space.cell_data_to_point_data()
                print("VTKSampler:: Converted cell_data to point_data for sampling.")
        except Exception:
            # If conversion fails, continue — sampling may still work.
            pass
        self._boundary_surface = self._search_space.extract_surface(
            pass_pointid=False, pass_cellid=False, nonlinear_subdivision=0
        )
        te = time.time()
        print("VTKSampler:: Time for loading interpolation space: ", te - tb)

    def __points_out_side_parametric_space(self, xv):
        bounds = self.search_space.bounds
        # facets predicates
        predicate = cp.inside_predicate(*xv.T, bounds)
        return np.logical_not(predicate)

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

        # facets predicates
        w_q = cp.w_predicate(*xv.T, bounds)
        e_q = cp.e_predicate(*xv.T, bounds)
        s_q = cp.s_predicate(*xv.T, bounds)
        n_q = cp.n_predicate(*xv.T, bounds)
        b_q = cp.b_predicate(*xv.T, bounds)
        t_q = cp.t_predicate(*xv.T, bounds)

        # x range: edges parallel to x axis
        sb_q = cp.sb_predicate(*xv.T, bounds)
        nb_q = cp.nb_predicate(*xv.T, bounds)
        st_q = cp.st_predicate(*xv.T, bounds)
        nt_q = cp.nt_predicate(*xv.T, bounds)

        # y range: edges parallel to y axis
        wb_q = cp.wb_predicate(*xv.T, bounds)
        eb_q = cp.eb_predicate(*xv.T, bounds)
        wt_q = cp.wt_predicate(*xv.T, bounds)
        et_q = cp.et_predicate(*xv.T, bounds)

        # z range: edges parallel to z axis
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
        no_external_points_Q = np.all(np.logical_not(external_idx))
        if no_external_points_Q:
            return

        xv = points[external_idx].copy()
        self.__map_external_points_to_surface(xv)

        # compute data for zero order expansion
        epoint_cloud = pyvista.PolyData(xv)
        # Sample the VTK dataset at the small point cloud mapped to the
        # boundary. This returns a dataset with topology matching xv.
        sampled_could = epoint_cloud.sample(self._search_space)

        # for all the fields
        glob_idx = np.nonzero(external_idx)[0]
        x = points[external_idx]

        for grad_field_name, grad in self.sampled_could.point_data.items():
            if grad_field_name.startswith("grad_"):
                field_name = grad_field_name.lstrip("grad_")
                fv = sampled_could[field_name]
                if field_name in self.constant_extended_fields:
                    grad_fv = np.zeros_like(sampled_could[grad_field_name])
                else:
                    grad_fv = sampled_could[grad_field_name]

                # taylor expansion all at once
                f_extrapolated = fv + np.sum(grad_fv * (x - xv), axis=1)

                # update fields
                self.sampled_could[field_name][glob_idx] = f_extrapolated
                self.sampled_could[grad_field_name][glob_idx] = grad_fv

        return

    def _clamp_points_inside_bounds(self, x_par: np.ndarray) -> np.ndarray:
        """Internal helper: nudge coordinates that lie on dataset bounds a
        tiny amount inside the search space so VTK probing does not mark
        them as invalid. Modifies x_par in-place and also returns it.

        The method is conservative: if bounds are not available or an
        unexpected error occurs we leave x_par unchanged.
        """
        try:
            xmin, xmax, ymin, ymax, zmin, zmax = self._search_space.bounds
            span_x = max(abs(xmax - xmin), 1.0)
            span_y = max(abs(ymax - ymin), 1.0)
            span_z = max(abs(zmax - zmin), 1.0)
            eps = 1e-12 * max(span_x, span_y, span_z)

            # If a coordinate is very close to the upper bound, move it
            # slightly inward; similarly for the lower bound.
            x_par[:, 0] = np.where(
                np.isclose(x_par[:, 0], xmax, atol=eps), xmax - eps, x_par[:, 0]
            )
            x_par[:, 0] = np.where(
                np.isclose(x_par[:, 0], xmin, atol=eps), xmin + eps, x_par[:, 0]
            )

            x_par[:, 1] = np.where(
                np.isclose(x_par[:, 1], ymax, atol=eps), ymax - eps, x_par[:, 1]
            )
            x_par[:, 1] = np.where(
                np.isclose(x_par[:, 1], ymin, atol=eps), ymin + eps, x_par[:, 1]
            )

            x_par[:, 2] = np.where(
                np.isclose(x_par[:, 2], zmax, atol=eps), zmax - eps, x_par[:, 2]
            )
            x_par[:, 2] = np.where(
                np.isclose(x_par[:, 2], zmin, atol=eps), zmin + eps, x_par[:, 2]
            )
        except Exception:
            # Be conservative: if bounds are not available for some reason,
            # continue without clamping.
            pass

        return x_par
