from scalene import scalene_profiler

import porepy as pp
from porepy.examples.flow_benchmark_2d_case_1 import (
    FlowBenchmark2dCase1Model,
    solid_constants_blocking_fractures,
    solid_constants_conductive_fractures,
)

params = {
    "solver": "direct",
    "residual_tolerance": 1e-8,
    "max_iterations": 100,
    "reference_pressure": 1e5,
}

scalene_profiler.start()
solid_constants = [
    solid_constants_blocking_fractures,
    solid_constants_conductive_fractures,
]
for solid_constants, grid, discr in zip(
    solid_constants, ["cartesian", "simplex"], ["tpfa", "mpfa"]
):
    # We use default fluid parameters but tailored solid parameters.
    # Note that the cell size needs to match the fracture geometry for the cartesian grid.
    model_params = {
        "material_constants": {"solid": solid_constants},
        "grid_type": grid,
        "meshing_arguments": {"cell_size": 0.125},
    }
    model = FlowBenchmark2dCase1Model(model_params)
    pp.run_time_dependent_model(model)
scalene_profiler.stop()
