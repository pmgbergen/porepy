import numpy as np

from examples.papers.flow_upscaling.import_grid import raw_from_csv
from examples.papers.flow_upscaling.frac_gen import fit

if __name__ == "__main__":
    file_geo = "../example1/Algeroyna.csv"

    pts, edges = raw_from_csv(file_geo, {'mesh_size_frac': 10}, {"snap": 1e-3})

    # we assume 1 segment 1 fracture, need to change the next line instead
    frac = np.arange(edges.shape[1])

    # we assume only 1 family, need to change the next line instead
    family = np.ones(frac.size)

    print( fit(pts, edges, frac, family) )
