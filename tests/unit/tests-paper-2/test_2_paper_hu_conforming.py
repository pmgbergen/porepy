import numpy as np
from typing import Callable, Optional, Type, Literal, Sequence, Union
import porepy as pp
import pdb
import porepy.models.two_phase_hu as two_phase_hu
import copy
import pygem

import test_2_paper_hu_non_conforming


class FinalModelConforming(test_2_paper_hu_non_conforming.FinalModel):
    def set_fractures(self) -> None:
        """ """
        frac1 = pp.LineFracture(
            np.array([[self.x_bottom, self.x_top], [self.ymin, self.ymax]])
        )

        self.x_intersection = self.xmean
        self.y_intersection = self.ymean

        frac_constr_1 = pp.LineFracture(
            np.array(
                [
                    [self.x_intersection, self.xmax],
                    [
                        self.y_intersection,
                        self.y_intersection,
                    ],
                ]
            )
        )

        frac_constr_2 = pp.LineFracture(
            np.array([[self.xmin, self.xmean - 0.1], [self.ymean, self.ymean]])
        )  # -0.1 just to not touch the fracture

        self._fractures: list = [frac1, frac_constr_1, frac_constr_2]


if __name__ == "__main__":
    params = test_2_paper_hu_non_conforming.params
    model = FinalModelConforming(params)
    model.displacement_max = 0
    pp.run_time_dependent_model(model, params)
