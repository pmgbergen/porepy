# Examples

The `src/porepy/examples/` folder contains a collection of Python modules demonstrating how to set up and run physical models in PorePy. These examples serve both as reusable model setups and as references for extending PorePy through its mixin-based design. Some of examples are used and further illustrated in tutorials. 

## Overview
These examples act as:
- Reference implementations of physical models,
- Verication cases with known solutions
- Templates for building more advanced simulations

They are categorized into two types:
1. **Standalone Runscripts**: Scripts that can be executed directly to run a full simulation.
2. **Model Definitions**: Modules defining model classes (mixins) that are intended to be used with a runner (like a functional test or a profiling script).

### Prerequisites
- **PorePy**: The core library.
- **Gmsh**: Required for examples involving unstructured or 3D grids (e.g., 3D flow benchmarks and geothermal reservoir).
- **Matplotlib**: Required for examples that generate verification plots (Mandel, Terzaghi).
- **ParaView**: Recommended for visualizing `.vtk` or `.pvd` output files.

---

## Standalone Runscripts
These scripts include a `__main__` block and can be executed directly from the terminal.

### 1. Geothermal Reservoir ([`geothermal_reservoir.py`](geothermal_reservoir.py))
- **Physics**: Coupled thermoporomechanics (momentum balance, fluid mass balance, energy balance).
- **Problem**: Simulates a geothermal doublet (injection and production wells) in a 3D fractured reservoir.
- **Key Features**: Well boundary condition protocols, lithostatic/hydrostatic initial and boundary conditions, and the `ConstraintLineSearchNonlinearSolver`.
- **Output**: VTK files in the `geothermal_reservoir/` directory.
- **Execution**: `python -m porepy.examples.geothermal_reservoir`

### 2. Tracer Flow ([`tracer_flow.py`](tracer_flow.py))
- **Physics**: Single-phase, two-component transport (water + tracer).
- **Problem**: Transport of a tracer through a 2D fractured domain with impermeable fractures.
- **Key Features**: Compositional flow framework with `CompositionalVariables` and `ComponentMassBalanceEquations`.
- **Output**: On-screen plots of pressure and tracer distribution after 20 minutes.
- **Execution**: `python -m porepy.examples.tracer_flow`

---

## Model Definitions
These modules define model classes assembled via PorePy's mixin pattern. They cannot be run directly as standalone scripts but are used in PorePy's functional tests and profiling applications.

### Flow Benchmarks
Implementations of the 2D and 3D single-phase flow benchmarks for fractured porous media.

**References**:
- Flemisch, B. et al. (2018). Benchmarks for single-phase flow in fractured porous media. *Advances in Water Resources*, 111, 239–258. [link](https://doi.org/10.1016/j.advwatres.2017.10.036)
- Berre, I. et al. (2021). Verification benchmarks for single-phase flow in three-dimensional fractured porous media. *Advances in Water Resources*, 147, 103759. [link](https://doi.org/10.1016/j.advwatres.2020.103759)

| Module | Case | Description |
| :--- | :--- | :--- |
| [`flow_benchmark_2d_case_1.py`](flow_benchmark_2d_case_1.py) | 2D Case 1 | Two variants are included: Case 1a (conductive fractures) and Case 1b (blocking fractures). Corresponding solid material parameters should be passed to the model using `FractureSolidConstants`. |
| [`flow_benchmark_2d_case_3.py`](flow_benchmark_2d_case_3.py) | 2D Case 3 | Two variants are included: Case 3a (top-to-bottom flow) and Case 3b (left-to-right flow). The variants are defined by boundary conditions and implemented by separate model classes. |
| [`flow_benchmark_2d_case_4.py`](flow_benchmark_2d_case_4.py) | 2D Case 4 | 64 fractures grouped in 13 connected fracture networks. Simplex grids are used for meshing. Reference solid material parameters should be passed to the model using `FractureSolidConstants`. |
| [`flow_benchmark_3d_case_2.py`](flow_benchmark_3d_case_2.py) | 3D Case 2 | Unit cube with 9 fractures and heterogeneous matrix permeability (low-permeability zones). Supports refinement levels 0–2 (~500 to ~32K cells). |
| [`flow_benchmark_3d_case_3.py`](flow_benchmark_3d_case_3.py) | 3D Case 3 | Unit cube with 6 fractures. Supports refinement levels 0–3 (~30K to ~500K cells). |

### Poromechanics Verification
Models used to verify PorePy's poromechanics implementation against exact analytical solutions.

- **Mandel's Consolidation ([`mandel_biot.py`](mandel_biot.py))**
    - **Physics**: Biot poromechanics.
    - **Problem**: A rectangular poroelastic sample under a vertical load, with lateral drainage.
    - **Verification**: Compares numerical results with an infinite series exact solution. Includes convergence analysis infrastructure.

- **Terzaghi's consolidation ([`terzaghi_biot.py`](terzaghi_biot.py))**
    - **Physics**: Biot poromechanics.
    - **Problem**: The one-dimensional consolidation problem represented by a 2D domain, where a vertical load is applied on the top boundary. Drainage is permitted only in the vertical direction, with impermeable bottom.
    - **Verification**: Compares numerical results with the exact solutions for pressure and degree of consolidation. Includes convergence analysis infrastructure.

### Contact Mechanics

- **Fracture Damage ([`fracture_damage.py`](fracture_damage.py))**
    - **Physics**: Contact mechanics with friction and dilation damage.
    - **Problem**: Evolution of damage and slip on a fracture surface under time-dependent loading. 
    - **Key Features**: Provides three model variants: an isotropic fracture damage model, an anisotropic fracture damage model, and a fracture damage momentum balance model. Includes exact solutions for isotropic and anisotropic damage formulations.  
    - **Verification**: Compares numerical results with exact solution. Includes convergence analysis and data storage infrastructure.
---

## Utilities

- **Example Parameters ([`example_params.py`](example_params.py))**
- **Key Features**: A complete reference set of top-level parameters for a PorePy model, including material constants, meshing arguments, convergence criteria, and line search options. Use this as a starting point and copy the relevant parameters for your own problem. 

---

## How to Use Model Definitions

### Adapting for Your Own Simulations
The model definitions follow PorePy's mixin-based design pattern. To adapt an example:

1. Identify the example closest to your problem.
2. Create your own mixin classes to override the aspects you want to change (e.g., boundary conditions, material properties, geometry).
3. Assemble your model class by combining your custom mixins with the existing ones, paying attention to the method resolution order (MRO).

For more guidance, we refer to tutorials that illustrate how models in examples can be adapted:
- [Benchmark simulation tutorial](../../../tutorials/benchmark_simulation.ipynb)
- [Tracer flow tutorial](../../../tutorials/tracer_flow.ipynb)
- [Mandel's problem tutorial](../../../tutorials/mandels_problem.ipynb)

### Running via Functional Tests
Many examples are verified in `tests/functional/`:
```bash
pytest tests/functional/test_mandel.py
pytest tests/functional/test_terzaghi.py
pytest tests/functional/test_benchmark_2d_case_3.py
pytest tests/functional/test_benchmark_3d_case_2.py
pytest tests/functional/test_benchmark_3d_case_3.py
```

### Running via Profiling Script
Use the profiling utility to run benchmarks with `viztracer` for performance analysis:
```bash
python src/porepy/applications/profiling/run_profiling.py --physics flow --geometry 0 --grid_refinement 1
python src/porepy/applications/profiling/run_profiling.py --physics poromechanics --geometry 0 --grid_refinement 0
```

---

## Troubleshooting
- **ModuleNotFoundError**: Ensure PorePy is installed (e.g., `pip install -e .` from the repository root).
- **Gmsh Issues**: Unstructured grid generation requires `gmsh` to be installed and accessible in your system path.
- **Visualization**: If no plots appear, ensure Matplotlib is installed. For VTK output, use [ParaView](https://www.paraview.org/).

## Further Reading
- [Tutorials](https://github.com/pmgbergen/porepy/tree/develop/tutorials) — Step-by-step introductions to PorePy concepts.
- [API Documentation](https://pmgbergen.github.io/porepy/html/docsrc/porepy/porepy.html) — Full reference (under construction).
- [Contributing Guidelines](https://github.com/pmgbergen/porepy/blob/develop/CONTRIBUTING.md) — How to contribute to PorePy.
