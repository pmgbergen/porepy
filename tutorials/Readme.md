# Tutorials
This folder contains tutorials to demonstrate how PorePy can be utilized.
To follow the tutorials, you can run the files as Jupyter notebooks in your own installation of PorePy or using a GitHub codespace.

The tutorials are divided into two sections, where the first section is oriented towards usage of PorePy as it is.
In practice, this means running the existing simulation models and making only minor adjustments, such as modifying boundary conditions, sources, initial conditions, material parameters, etc.
We recommend new users to familiarize themselves with this section.

The second section includes more specific tutorials and is recommended to those who aim to use PorePy in a more advanced way or contribute to it. New users may skip this section.

# Overview
The PorePy tutorials are designed as stand-alone documentation of different components and capabilities.
Below we suggest an order in which to read the tutorials.
It should however be noted that the appropriate order of reading may depend on the reader's background and ambitions.

## Introductory tutorials
1. [Introduction](./introduction.ipynb) describes the overarching conceptual framework and its high-level implementation. It also lists some problems which may be solved using PorePy.
2. [Single phase flow](./single_phase_flow.ipynb) is where we introduce PorePy model classes and show how to run a simulation, including how to do minor adjustments in a model. We also demonstrate how the simulation data is exported to formats suitable for external visualization (e.g ParaView).
3. [Mixed-dimensional meshing](./mixed_dimensional_meshing.ipynb) describes the construction of mixed-dimensional grids. These grids represent a fracture network and the surrounding porous medium.
4. [Boundary conditions](./boundary_conditions.ipynb) demonstrates how to set boundary conditions in PorePy. It visits how to set scalar boundary conditions for the single phase flow problem, and vectorial boundary conditions for the momentum balance problem.
5. [Poromechanics](./poromechanics.ipynb) covers the concept of setting up a multiphysics simulation by reusing single-physics model classes.
6. [Conventions](./conventions.ipynb) defines some conventions used in PorePy. Specifically this is related to signs, boundary conditions etc.

## Advanced tutorials
For the more experienced user, some more specific tutorials are also available. The order of these is not important - we suggest reading them in the order of your interest.

7. [Exporter](./exporter.ipynb) gives a thorough introduction to exporting of simulation data in PorePy, including examples of how to modify the exporter.
8. [Exporting in models](./exporting_models.ipynb) shows how exporting of data can be combined with the model class framework of PorePy.
9. [Diagnostics](./diagnostics.ipynb) provides a description of how to use the diagnostics tool in PorePy. The tool allows for visualizing properties of the discretized system of equations in PorePy.
10. [Benchmark simulation](./benchmark_simulation.ipynb) defines a problem from [this benchmark study](https://doi.org/10.1016/j.advwatres.2017.10.036) and also illustrates some ways of modifying a simulation model.
11. [Mandel's problem](./mandels_problem.ipynb) shows how to set up and run the Mandel's consolidation problem based on the Biot equations of poroelasticity.
12. [Linear Tracer Flow](./tracer_flow.ipynb) describes the setup of a linear single-phase, 2-component model based on tutorial #6, and showcases a simulation of tracer transport through a fractured domain.
13. [Fluid modeling](./fluid_modeling.ipynb) explains how to set up multicomponent, multiphase fluids in a model, and various approaches to modeling fluid properties.

Also, see the [reference parameter dictionaries](../src/porepy/examples/example_params.py) for PorePy models. They can be used as a starting point for a PorePy user to copy the relevant parameters for your own problem.

## Technical tutorials
These are tutorials that cover technical aspects of the code, 

14. [Grids](./grids.ipynb) describes the structure of individual grids and demonstrates construction of different types of grids. It also shows how to access and manipulate grid quantities.
15. [Grid topology](./grid_topology.ipynb) covers how to access various topological properties of grids in PorePy. This includes the relation between cells, faces and nodes, as well as the direction of the face normal vectors.
16. [Flux discretizations](./flux_discretizations.ipynb) shows different discretization methods available for diffusive fluxes. These are used for Darcy's law for fluid fluxes in a mass balance equation.
17. [Stress discretization](./stress_discretization.ipynb) describes the discretization method used for the vector version of tutorial #16, which arises in the linear elastisity equations.
18. [Equations](./equations.ipynb) briefly covers some general basics about automatic differentiation (AD). It mainly covers how AD is extensively used in PorePy equations, which means that it is most useful for the users that want to define their own equations.
