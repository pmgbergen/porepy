import numpy as np
import csv

import porepy as pp

def main():

    file_name = 'network.csv'
    new_file_name = 'new_network.csv'

    tol = 1e-4 * (pp.METER)

    pts, edges = pp.importer.lines_from_csv(file_name, tol=tol)
    domain = pp.cg.bounding_box(pts)
    x = 0.5*(domain['xmin']+domain['xmax'])
    y = 0.5*(domain['ymin']+domain['ymax'])

    xy = np.tile([x, y], (pts.shape[1], 1)).T
    pts -= xy

    with open(new_file_name, 'w') as csvfile:
        fieldnames = ['FID', 'START_X', 'START_Y', 'END_X', 'END_Y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx, edge in enumerate(edges.T):
            writer.writerow({'FID': idx,
                             'START_X': pts[0, edge[0]],
                             'START_Y': pts[1, edge[0]],
                             'END_X':   pts[0, edge[1]],
                             'END_Y':   pts[1, edge[1]]})

if __name__ == "__main__":
    main()
