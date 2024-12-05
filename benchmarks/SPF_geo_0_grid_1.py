import porepy as pp
from porepy.applications.profiling.run_profiling import make_benchmark_model
from argparse import Namespace


class Profiling:

    def __init__(self):
        self.model = make_benchmark_model(
            Namespace(**{"geometry": 0, "grid_refinement": 1, "physics": "flow"})
        )

    def time_prepare_simulation(self):
        self.model.prepare_simulation()

    def time_run_model(self):
        pp.run_time_dependent_model()

    def time_before_nonlinear_loop(self):
        self.model.before_nonlinear_loop()

    def time_before_nonlinear_iteration(self):
        self.model.before_nonlinear_iteration()

    def time_assemble_linear_system(self):
        self.model.assemble_linear_system()

    def time_solve_linear_system(self):
        self.model.solve_linear_system()

    def time_after_nonlinear_iteration(self):
        self.model.after_nonlinear_iteration()

    def time_after_nonlinear_convergence(self):
        self.model.after_nonlinear_convergence()
