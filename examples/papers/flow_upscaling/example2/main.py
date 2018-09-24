import numpy as np
import porepy as pp

from examples.papers.flow_upscaling.import_grid import raw_from_csv
from examples.papers.flow_upscaling.frac_gen import fit, generate

if __name__ == "__main__":
    file_geo = "algeroyna_1to100.csv"

    pts, edges, frac = raw_from_csv(file_geo, {'mesh_size_frac': 10}, {"snap": 1e-3})

    domain = {'xmin': pts[0].min(), 'xmax': pts[0].max(),
              'ymin': pts[1].min(), 'ymax': pts[1].max()}

    pp.plot_fractures(domain, pts ,edges)

    # we assume only 1 family, need to change the next line instead
    family = np.ones(frac.size)

    dist_l, dist_a = fit(pts, edges, frac, family)
    pts_n, edges_n = generate(pts, edges, frac, dist_l, dist_a)

    pp.plot_fractures(domain, pts_n ,edges_n)

