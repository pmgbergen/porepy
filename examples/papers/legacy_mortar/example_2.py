import numpy as np

from porepy.fracs import importer, mortars, meshing
from porepy.numerics import elliptic

import example_2_data
from solvers import *

case = 1
h = 0.1
list_of_solvers = {"tpfa": solve_tpfa, "p1": solve_p1, "mpfa": solve_mpfa,
                   "rt0": solve_rt0, "vem": solve_vem}

for solver_name, solver_fct in list_of_solvers.items():

    gb, domain = example_2_data.create_gb(h)
    example_2_data.add_data(gb, domain, solver_name, case)
    folder = "example_2_"+solver_name
    solver_fct(gb, folder)
