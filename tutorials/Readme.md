# Tutorials
This folder contains tutorials that show the basic workings of PorePy. To follow the tutorials, you can download the files (either by cloning the repository, or download the individual files), and run them as Jupyter Notebooks.

In addition to the tutorials contained herein, we are working on providing more advanced examples that further illustrate the usage of PorePy. 

# Overview
The PorePy tutorials are designed as stand-alone documentation of different components and capabilities. 
The appropriate order of reading may depend on the reader's background and ambitions.
However, the following may serve as a general suggestion:
    
1. [introduction](./introduction.ipynb) describes the overarching conceptual framework and its high-level implementation and lists some problems which may be solved using PorePy.
2. [grid_structure](./grid_structure.ipynb) describes the structure of individual grids. It demonstrates construction of different types of grids. It also dives into the data structure, and shows how to access and manipulate grid quantities.
3. [meshing_of_fractures](./meshing_of_fractures.ipynb) describes the construction of mixed-dimensional grids representing a fracture network and the surrounding porous medium.
4. [automatic_differentiation](./automatic_differentiation.ipynb) provides an introduction to automatic differentiation (AD) and how to solve a generic equation using the AD framework. The tutorial includes setup of parameters and discretizations.
5. [incompressible_flow_model](./incompressible_flow_model.ipynb) describes how to use a model class `Incompressible Flow`. The tutorial exposes several extensions and how to solve a mixed-dimensional problem with a few lines of code.

More specific tutorials are also available:

6. [flux_discretizations](./flux_discretizations.ipynb) shows different discretization methods available for diffusive fluxes, which are used for Darcy's law for fluid fluxes in a mass balance equation. 
7. [stress_discretization](./stress_discretization.ipynb) describes the discretization method used for the vector version of 6. arising in the stress-strain relationship of Hooke's law.
8. [contact_mechanics_with_AD](./contact_mechanics_with_AD.ipynb) delves deeper into the model classes and their use of the ad framework.
9. [exporter](./exporter.ipynb) documents how to export data from PorePy for external visualization (e.g. ParaView).
10. [equation_definition](./equation_definition.ipynb) details the steps required to set up and solve a mixed-dimensional partial differential equation with emphasis on the PorePy AD framework.
11. [conventions](./conventions.ipynb) defines some conventions used in PorePy related to signs, boundary conditions etc.
