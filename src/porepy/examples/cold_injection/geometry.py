"""Contains the geometry for the cold injection examples.

Most notably, contains a 2D geometry with 0D points as injection and production wells.

Fractures are added on top of it.

Individual configurations are possible in ``config.py``.

"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.fracs.wells_3d import _add_interface

from .config import ModelConfig


def _set_random_seed(*args):
    s = 2025
    for a in args:
        s += int(a)
    np.random.seed(s)


class PointWells(ModelConfig):
    """Geometry adding point grids as wells after super-call in set_geometry."""

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmin": 0.0,
                "xmax": self.units.convert_units(self._DOMAIN_DIMENSIONS[0], "m"),
                "ymin": 0.0,
                "ymax": self.units.convert_units(self._DOMAIN_DIMENSIONS[1], "m"),
            }
        )

    def set_geometry(self):
        super().set_geometry()

        for i, injection_point in enumerate(self._INJECTION_POINTS):
            self._add_well(injection_point, i, "injection")

        for i, production_point in enumerate(self._PRODUCTION_POINTS):
            self._add_well(production_point, i, "production")

    def _add_well(
        self,
        point: np.ndarray,
        well_index: int,
        well_type: Literal["injection", "production"],
    ) -> None:
        """Helper method to construct a well in 2D as a PointGrid and add respective
        interface.

        Parameters:
            point: Point in space representing well.
            well_index: Assigned number for well of type ``well_type``.
            well_type: Label to add a tag to the point grid labelng as injector or
            producer.

        """
        matrix = self.mdg.subdomains(dim=self.nd)[0]
        assert isinstance(point, np.ndarray)
        p: np.ndarray
        if point.shape == (2,):
            p = np.zeros(3)
            p[:2] = point
        elif point.shape == (3,):
            p = point
        else:
            raise ValueError(
                f"Point for well {(well_type, well_index)} must be 1D array of length "
                + "2 or 3."
            )

        sd_0d = pp.PointGrid(self.units.convert_units(p, "m"))
        # Tag for processing of equations.
        sd_0d.tags[f"{well_type}_well"] = well_index
        sd_0d.compute_geometry()

        self.mdg.add_subdomains(sd_0d)

        # Motivated by wells_3d.py#L828
        cell_matrix = matrix.closest_cell(sd_0d.cell_centers)
        cell_well = np.array([0], dtype=int)
        cell_cell_map = sps.coo_matrix(
            (np.ones(1, dtype=bool), (cell_well, cell_matrix)),
            shape=(sd_0d.num_cells, matrix.num_cells),
        )

        _add_interface(0, matrix, sd_0d, self.mdg, cell_cell_map)


class RandomFracturesAndPointWells2D(PointWells):
    """2D matrix with point grids as injection and production points, and a random
    distribution of fractures.

    The number of fractures can be defined via a model parameter ``'_num_fractures'``.

    Alternative for the ``WellNetwork3d`` in 2d.

    """

    def set_fractures(self) -> None:
        x_min, y_min = 0.0, 0.0
        x_max = self._DOMAIN_DIMENSIONS[0]
        y_max = self._DOMAIN_DIMENSIONS[1]
        domain_width = x_max - x_min
        domain_height = y_max - y_min
        min_length = domain_height
        max_length = domain_width * 0.8

        num_fracs = int(self.params.get("_num_fractures", 0))
        fractures = []
        _set_random_seed(num_fracs)
        for _ in range(num_fracs):
            # Random center within bounds
            x_center = np.random.uniform(
                x_min + 0.1 * domain_width, x_max - 0.1 * domain_width
            )
            y_center = np.random.uniform(
                y_min + 0.1 * domain_height, y_max - 0.1 * domain_height
            )

            # Random angle and length
            theta = np.random.uniform(-np.pi / 3, np.pi / 3)
            length = np.random.uniform(min_length, max_length)

            dx = 0.5 * length * np.cos(theta)
            dy = 0.5 * length * np.sin(theta)

            x0, x1 = x_center - dx, x_center + dx
            y0, y1 = y_center - dy, y_center + dy

            coords = np.array([[x0, x1], [y0, y1]])
            fractures.append(pp.LineFracture(coords))

        self._fractures = fractures


class HorizontalFractureAndPointWells2D(PointWells):
    """Introduces a single horizontal fracture in the middel of the domain,
    with length equal to half the domain length in x-direction."""

    def set_fractures(self) -> None:
        x_min, y_min = 0.0, 0.0
        x_max = self._DOMAIN_DIMENSIONS[0]
        y_max = self._DOMAIN_DIMENSIONS[1]
        domain_width = x_max - x_min
        domain_height = y_max - y_min

        x_center = x_min + domain_width * 0.5
        y_center = y_min + domain_height * 0.5

        frac_width = domain_width * 0.6

        self._fractures = [
            pp.LineFracture(
                np.array(
                    [
                        [x_center - 0.5 * frac_width, x_center + 0.5 * frac_width],
                        [y_center, y_center],
                    ]
                )
            )
        ]
