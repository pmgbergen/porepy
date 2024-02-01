"""Simulation examples for PorePy.

This module contains a number of examples of how to run a simulation using PorePy. Each
example contains functionality specific to the setup in question.

The simulation setups serve the following purposes:
    - Reuse in testing
    - Use in documentation
    - Use in run scripts and development
    - Promotion of specific test cases (e.g. benchmarks)

In the interest of manageable maintenance, the examples are actively curated and the
acceptance criteria for new examples are relatively strict.
"""

from .flow_benchmark_3d_case_3 import FlowBenchmark3dCase3Model
from .mandel_biot import MandelExactSolution, MandelSolutionStrategy
from .terzaghi_biot import TerzaghiExactSolution, TerzaghiSetup
