#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import example_1
import solvers

# ------------------------------------------------------------------------------#

if __name__ == "__main__":

    gb_ref = example_1.reference_solution()
    N = 5

    solver = "mpfa"
    example_1.convergence_test(N, gb_ref, solver, solvers.solve_mpfa)

    solver = "vem"
    example_1.convergence_test(N, gb_ref, solver, solvers.solve_vem)

    solver = "rt0"
    example_1.convergence_test(N, gb_ref, solver, solvers.solve_rt0)

    solver = "p1"
    example_1.convergence_test(N, gb_ref, solver, solvers.solve_p1)

    solver = "tpfa"
    example_1.convergence_test(N, gb_ref, solver, solvers.solve_tpfa)

# ------------------------------------------------------------------------------#
