# Examples
This folder contains a collection of simulation examples using PorePy to demonstrate how PorePy can be applied to specific physical problems and benchmark cases. 

The examples are designed as reusable model classes with reference configurations. They illustrate how to build simulation models by combining geometry, boundary/initial conditions, constitutive laws, and discretization strategies. The examples contains:
*	benchmark problems with known behavior
*	Practical configurations of models, grids, and parameters
*	Templates that can be extended to new applications

These scripts are particularly helpful for users who would like to need a reference implementation for specific physics or build their own models based on existing setups. For more guidance, we refer to the [benchmark simulation tutorial](../../../tutorials/benchmark_simulation.ipynb), where some of examples are further developed and extended.

# How to run examples
In order to run a simulation, the examples will be combined with parameter dictionaries and a solver setup. 

# Overview
There are brief descriptions for the PorePy examples below: 

1. [Example parameters](./example_params.py) defines reference parameter dictonaries for configurations of PorePy models and related solvers. The parameters include physical properties, grid and meshing, time and simulation control, output and restart, and detailed nonlinear solver settings such as convergence criteria and line search strategies.
2. [2D flow benchmark 1: regular fracture network](./flow_benchmark_2d_case_1.py) implements Case 1 of the 2D flow benchmark for single-phase flow in fractured porous media, as defined in Sections 4.1 of [Flemisch et al. (2018)](https://doi.org/10.1016/j.advwatres.2017.10.036). Two variants of the benchmark are provided, Case 1a (conductive fractures) and Case 1b (blocking fractures). These variants are implemented by assigning different solid material parameters. 
3. [2D flow benchmark 3: complex fracture network](./flow_benchmark_2d_case_3.py) implements Case 3 of the 2D flow benchmark for single-phase flow in fractured porous media, as defined in Sections 4.3 of [Flemisch et al. (2018)](https://doi.org/10.1016/j.advwatres.2017.10.036). Two variants of the benchmark are provided, Case 3a (top-to-bottom flow) and Case 3b (left-to-right flow). These variants are defined by different boundary conditions and implemented by separate model classes in this example. 
4. [2D flow benchmark 4: a realistic case](./flow_benchmark_2d_case_4.py) implements Case 4 of the 2D flow benchmark for single-phase flow in fractured porous media, as defined in Sections 4.4 of [Flemisch et al. (2018)](https://doi.org/10.1016/j.advwatres.2017.10.036). This model involves a large-scale, realistic fracture system with multiple groups of connected fracture networks. Simplex grids are used for meshing. The example also provides a set of predefined solid material parameters, which should be explicitly passed to the model when initializing the simulation. 
5. [Fracture damage model](./fracture_damage.py) implements a set of models for coupled fracture damage and contact mechanics with time-dependent boundary conditions. Three model variants are provided as separate model classes: an isotropic fracture damage model, an anisotropic fracture damage model, and a fracture damage momentum balance model. The example also defines a set of parameters, exact solutions, and data collection functionality for verification. 
6. [Terzaghi's consolidation problem](./terzaghi_biot.py) is implemented within PorePy using the Biot formulation of poromechanics. This model is a verification benchmark for coupled flow-mechanics simulations. The example provides suggested parameters of materials (solid and fluid), model geometries (domain height and the grid size), and applied load. Eaxct solutions of pressure and degree of consolidation are computed with a setting for truncating the infinite series in solutions. The functionality of data collection is included for accuracy verification, convergence studies, and solution storage. Besides solving the model, the example also provides several utilities. 

