import porepy as pp
import numpy as np
from typing import Optional, Union

__all__ = ["Domain"]


class Domain:

    def __init__(self, box: dict[str, pp.number]):

        self.domain: np.ndarray = self.domain_from_box(box)
        self.dim: int = self.dimension_from_box(box)

    @classmethod
    def as_polygon(cls, polygon: np.ndarray):
        setattr(cls, "domain", polygon)
        setattr(cls, "dim", 2)

    def domain_from_box(self, box: dict) -> np.ndarray:

        x0 = box["xmin"]
        x1 = box["xmax"]
        y0 = box["ymin"]
        y1 = box["ymax"]

        domain = np.array([[x0, y0], [x1, y1]])
        print(domain.shape)
        return domain

    def dimension_from_box(self, box: dict[str, pp.number]) -> int:

        dim = 0
        if "xmin" in box.keys() and "xmax" in box.keys():
            dim += 1
        if "ymin" in box.keys() and "ymax" in box.keys():
            dim += 1
        if "zmin" in box.keys() and "zmax" in box.keys():
            dim += 1

        return dim