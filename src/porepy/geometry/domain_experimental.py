import numpy as np

import porepy as pp

domain = pp.Domain(bounding_box={"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})

line_1 = np.array([[0, 0], [0, 1]])
line_2 = np.array([[0, 0.5], [1, 1.5]])
line_3 = np.array([[0.5, 1], [1.5, 1]])
line_4 = np.array([[1, 1], [1, 0]])
line_5 = np.array([[1, 0], [0, 0]])
irregular_pentagon = [line_1, line_2, line_3, line_4, line_5]
domain_from_polytope = pp.Domain(polytope=irregular_pentagon)

#%% Check point cloud from polytope
point_cloud = pp.domain.point_cloud_from_polygon(irregular_pentagon)