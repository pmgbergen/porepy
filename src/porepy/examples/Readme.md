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
2. [2D flow benchmark 1: regular fracture network](./flow_benchmark_2d_case_1.py) implements Case 1 of the 2D flow benchmark for single-phase flow problem in fractured porous media, as defined in Sections 4.1 of [Flemisch et al. 2017](https://doi.org/10.1016/j.advwatres.2017.10.036). Two variants of the benchmark are provided, Case 1a (conductive fractures) and Case 1b (blocking fractures). These variants are implemented by assigning different solid material parameters. 

