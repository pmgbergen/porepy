# Examples
This folder contains a collection of simulation examples using PorePy to demonstrate how PorePy can be applied to specific physical problems and benchmark cases. 

The examples focus on the following aspects: 
*	complete and runnable simulation setups
*	benchmark problems with known behavior
*	Practical configurations of models, grids, and parameters
*	Templates that can be adapted to new applications

These scripts are particularly helpful for users who would like to quickly run a working model, need a reference implementation for specific physics, or build their own models based on existing
setups. For more guidance, we refer to the [benchmark simulation tutorial](../../../tutorials/benchmark_simulation.ipynb), where some of examples are further developed and extended.

# How to run examples
Each example is a self-contained Python script and can be run directly by `python <example_name>.py`. 
Users are also encouraged to explore different physical scenarios by modifying material parameters, boundary conditions, source terms, and grid size. 

# Overview
There are brief descriptions for the PorePy examples below: 

1. [Example parameters](./example_params.py) defines reference parameter dictonaries for configurations of PorePy models and related solvers. The parameters include physical properties grid and meshing, time stepping, output control, and detailed nonlinear solver settings such as convergence criteria and line search strategies.

