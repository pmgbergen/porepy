"""This runscripts runs a selected porepy benchmark automatically with ``viztracer``
enabled and opens the results in a browser.

Example (intended to show basic input and output functionality):
    # If not already installed, get viztracer:
    >>> pip install viztracer
    # Run the script:
    >>> python run_profiler.py --physics poromechanics --geometry 0 --grid_refinement 0
    # This will run a single-phase poromechanics benchmark on the first 2D case with the
    coarsest grid refinement.
    >>> python run_profiler.py --physics flow --geometry 1 --grid_refinement 2
    # This will run a single-phase flow benchmark on the second 2D case with the finest
    grid refinement.
    >>> python run_profiler.py --physics poromechanics --geometry 2 --grid_refinement 2
    # This will run a single-phase poromechanics benchmark on a 3D grid with the finest
    # grid refinement.

Note: Running the 3D model on the finest grid requires ~20 GB ram (!), thus is not
    recommended on a local machine.

"""

import argparse
import pathlib
import subprocess
from typing import Optional, Type

# VizTracer is missing stubs or py.typed marker, hence we ignore type errors.
from viztracer import VizTracer  # type: ignore[import]

import porepy as pp
from porepy.examples.flow_benchmark_2d_case_1 import BoundaryConditions as Case1BC
from porepy.examples.flow_benchmark_2d_case_1 import FlowBenchmark2dCase1Model
from porepy.examples.flow_benchmark_2d_case_1 import Geometry as Case1Geo
from porepy.examples.flow_benchmark_2d_case_1 import Permeability as Case1Permeability
from porepy.examples.flow_benchmark_2d_case_3 import (
    Case3aBoundaryConditions as Case3aBC,
)
from porepy.examples.flow_benchmark_2d_case_3 import FlowBenchmark2dCase3aModel
from porepy.examples.flow_benchmark_2d_case_3 import Geometry as Case3Geo
from porepy.examples.flow_benchmark_2d_case_3 import Permeability as Case3Permeability
from porepy.examples.flow_benchmark_2d_case_4 import BoundaryConditions as Case4BC
from porepy.examples.flow_benchmark_2d_case_4 import FlowBenchmark2dCase4Model
from porepy.examples.flow_benchmark_2d_case_4 import Geometry as Case4Geo
from porepy.examples.flow_benchmark_3d_case_3 import BoundaryConditions as Case3dBC
from porepy.examples.flow_benchmark_3d_case_3 import FlowBenchmark3dCase3Model
from porepy.examples.flow_benchmark_3d_case_3 import Geometry as Case3dGeo
from porepy.examples.flow_benchmark_3d_case_3 import Permeability as Case3dPermeability
from porepy.models.poromechanics import Poromechanics


# Ignore type errors inherent to the ``Poromechanics`` class.
class Case1Poromech2D(  # type: ignore[misc]
    Case1Permeability,
    Case1Geo,
    Case1BC,
    Poromechanics,
):
    pass


class Case3aPoromech2D(  # type: ignore[misc]
    Case3Permeability,
    Case3Geo,
    Case3aBC,
    Poromechanics,
):
    pass


class Case3Poromech3D(  # type: ignore[misc]
    Case3dPermeability,
    Case3dGeo,
    Case3dBC,
    Poromechanics,
):
    pass


class Case4Poromech2D(  # type: ignore[misc]
    Case4Geo,
    Case4BC,
    Poromechanics,
):
    pass


def make_benchmark_model(args: argparse.Namespace):
    """Create a benchmark model based on the provided arguments.

    Parameters:
        args: Command-line arguments containing the following
            attributes:
            - geometry (int): Specifies the geometry type (0, 1, or 2). Geometry 0 and 1
            are 2D grids, geometry 2 is a 2D grid with 64 fractures, and geometry 3 is a
            3D grid.
            - grid_refinement (int): Specifies the grid refinement level.
            - physics (str): Specifies the type of physics ("flow" or "poromechanics").

    Returns:
        model: An instance of the selected benchmark model with the specified
            parameters.

    Raises:
        ValueError: If the geometry or grid_refinement values are invalid, or if the
            combination of geometry and physics is not supported.

    """
    # Set up fixed model parameters.
    model_params = {
        "grid_type": "simplex",
        "time_manager": pp.TimeManager(
            dt_init=1,
            schedule=[0, 2],
            constant_dt=True,
        ),
    }

    # Warn user that the finest grid will likely take significant time.
    if args.grid_refinement >= 2:
        print(f"{args.grid_refinement=} will likely take significant time to run.")

    # Set cell_size/refinement_level model parameter based on choice of geometry and
    # grid refinement.
    if args.geometry in [0, 1, 2]:
        if args.grid_refinement == 0:
            cell_size = 0.1
        elif args.grid_refinement == 1:
            cell_size = 0.01
        elif args.grid_refinement == 2:
            cell_size = 0.005
        else:
            raise ValueError(f"{args.grid_refinement=}")
        model_params["meshing_arguments"] = {"cell_size": cell_size}
    elif args.geometry == 3:
        model_params["refinement_level"] = args.grid_refinement
    else:
        raise ValueError(f"{args.grid_refinement=}")

    # Select a model based on choice of physics and geometry.
    model: Optional[Type] = None
    if args.geometry == 0:
        if args.physics == "flow":
            model = FlowBenchmark2dCase1Model
        elif args.physics == "poromechanics":
            model = Case1Poromech2D
    elif args.geometry == 1:
        if args.physics == "flow":
            model = FlowBenchmark2dCase3aModel
        elif args.physics == "poromechanics":
            model = Case3aPoromech2D
    elif args.geometry == 2:
        if args.physics == "flow":
            model = FlowBenchmark2dCase4Model
        elif args.physics == "poromechanics":
            model = Case4Poromech2D
    elif args.geometry == 3:
        if args.physics == "flow":
            model = FlowBenchmark3dCase3Model
        elif args.physics == "poromechanics":
            model = Case3Poromech3D

    if model is None:
        raise ValueError(f"{args.geometry=}, {args.physics=}")

    return model(model_params)


def run_model_with_tracer(args, model) -> None:
    """Run a model with VizTracer enabled for performance profiling.

    Parameters:
        args: Command-line arguments containing the following attributes:
            - physics (str): The physics of the model.
            - geometry (str): The geometry of the model.
            - grid_refinement (int): The grid refinement level for the model.
            - save_file (str): The file path to save the profiling results. If empty, a
            default name is generated based on chosen physics, geometry, and grid
            refinement.
            - min_duration (int): Minimum duration in microseconds for a function to be
            recorded by VizTracer.
            - keep_output (bool): Whether to keep the output file after viewing it.
        model: The model to be run and profiled.

    Raises:
        ValueError: If ``args.save_file`` does not end in .json.

    Returns:
        None

    """
    if args.save_file == "":
        save_file: str = (
            f"profiling_{args.physics}_{args.geometry}_{args.grid_refinement}.json"
        )
    else:
        if not args.save_file.endswith(".json"):
            raise ValueError(f"{args.save_file=}")
        save_file = args.save_file

    # Run model with viztracer enabled.
    tracer = VizTracer(
        min_duration=args.min_duration,  # μs
        ignore_c_function=True,
        ignore_frozen=True,
    )
    tracer.start()
    model.prepare_simulation()
    print("Num dofs:", model.equation_system.num_dofs())
    # Simulations use a single time step and relaxed Newton tolerance to ensure 1-2
    # Newton iterations. Material parameters are defaults and not realistic, as these
    # bencmarks are focusing on code segments (e.g., AD assembly) independent of
    # parameter realism.
    pp.run_time_dependent_model(
        model,
        {
            "prepare_simulation": False,
            "nl_divergence_tol": 1e8,
            "max_iterations": 25,
            "nl_convergence_tol": 1e-2,
            "nl_convergence_tol_res": 1e-2,
        },
    )
    tracer.stop()

    # Save the results and open them in a browser with vizviewer.
    results_path = pathlib.Path(__file__).parent / save_file
    tracer.save(str(results_path))
    subprocess.run(["vizviewer", "--port", "9002", results_path])
    if not args.keep_output:
        results_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physics",
        type=str,
        default="flow",
        choices=["flow", "poromechanics"],
        help="Physics to run. Choices are single-phase flow or poromechanics.",
    )
    parser.add_argument(
        "--geometry",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help=(
            "0: 1st 2D case, 1: 2nd 2D case, 2: 2D case with 64 fractures, 3: 3D case."
        ),
    )
    parser.add_argument(
        "--grid_refinement",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Level of grid refinement. For the 2D cases, this corresponds to cell"
        + " sizes 0.1, 0.01, and 0.005. For the 3D cases, this corresponds to 30K,"
        + " 140K, 350K cells.",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default="",
        help="File to save the viztracer output to. If not specified, the file will be"
        + " named after the chosen physics, geometry, and grid refinement.",
    )
    parser.add_argument(
        "--keep_output",
        action="store_true",
        default=True,
        help="Keep viztracer output after running.",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1e5,
        help="Profiling will include only the function calls with execution time higher"
        + " than this threshold, μs.",
    )

    args = parser.parse_args()
    model = make_benchmark_model(args)
    run_model_with_tracer(args, model)
