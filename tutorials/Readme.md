# Tutorials
This folder contains tutorials to demonstrate how PorePy can be utilized.
More specifically, they cover both the basic workings of PorePy and how to use it, as well as some more specific use.
To follow the tutorials you can run them as Jupyter Notebooks, which is done either by downloading the individual files or by cloning the repository.

The new or average user need only care about the tutorials 1.-10. that are shown below.
Those tutorials are mainly oriented towards use of PorePy as it is. 
In practice this means only making minor adjustments, such as modifying boundary conditions, initial conditions and similar.

The tutorials' goal is to show how simulations on various different physical problems can be done using PorePy.
In addition to this, the user will learn about how to export files for visualization externally.

Note that the lower three tutorials cover some very specific use of PorePy. 
In other words, they are not needed for most users' everyday PorePy use.
# Overview
The PorePy tutorials are designed as stand-alone documentation of different components and capabilities. 
Below we provide a general suggestion of in which order to read the tutorials.
It should however be noted that the appropriate order of reading may depend on the reader's background and ambitions.

1. [introduction](./introduction.ipynb) describes the overarching conceptual framework and its high-level implementation. It also lists some problems which may be solved using PorePy.
2. [grids](./grids.ipynb) describes the structure of individual grids and demonstrates construction of different types of grids. It also dives into the data structure, and shows how to access and manipulate grid quantities.
3. [grid topology](./grid_topology.ipynb) covers how to access various topological properties of grids in PorePy. This includes the relation between cells, faces and nodes, as well as the direction of the face normal vectors.
4. [meshing_of_fractures](./meshing_of_fractures.ipynb) describes the construction of mixed-dimensional grids. These grids represent a fracture network and the surrounding porous medium.
5. [conventions](./conventions.ipynb) defines some conventions used in PorePy. Specifically this is related to signs, boundary conditions etc.
6. [single phase flow](./single_phase_flow.ipynb) describes how to use the model class `SinglePhaseFlow`. The tutorial exposes several extensions and how to solve a mixed-dimensional problem with only a few lines of code.
7. []() describes how to combine two single physics problems into one coupled problem. TODO: make better description and include reference when the file exists. 
8. [exporter](./exporter.ipynb) documents how to export data from PorePy for external visualization (e.g. ParaView). Several examples of exporting is demonstrated.
9. [simulation exporting](./simulation_exporting.ipynb) TODO: Either add description, or if it's combined with the above tutorial: edit above description.

For the experienced user, some more specific tutorials are also available:

10. [equations](./equations.ipynb) covers briefly some general basics about automatic differentiation (AD). It mainly covers how AD is extensively used in PorePy equations, which means that it is most useful for the users that want to define their own equations.
11. [mandels problem](./mandels_problem.ipynb) shows how to set up and run the Mandel's consolidation problem based on the Biot equations of poroelasticity. 
12. [flux_discretizations](./flux_discretizations.ipynb) shows different discretization methods available for diffusive fluxes. These are used for Darcy's law for fluid fluxes in a mass balance equation. 
13. [stress_discretization](./stress_discretization.ipynb) describes the discretization method used for the vector version of 12. that arises in the stress-strain relationship of Hooke's law.