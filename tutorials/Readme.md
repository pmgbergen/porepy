# Tutorials
This folder contains tutorials that show the basic workings of PorePy. To follow the tutorials, you can download the files (either by cloning the repository, or download the individual files), and run them as Jupyter Notebooks.

In addition to the tutorials contained herein, we are working on providing more advanced examples that further illustrate the usage of PorePy. 

# Overview
The PorePy tutorials are designed as stand-alone documentation of different components and capabilities. 
The appropriate order of reading may depend on the reader's background and ambitions.
However, the following may serve as a general suggestion:
    
1. [introduction](./introduction.ipynb) describes the overarching conceptual framework and its high-level implementation and lists some problems which may be solved using PorePy.
2. [grid_structure](./grid_structure.ipynb) describes the structure of individual grids. Shows construction of different types of grids. Also dives into the data structure, and shows how to access and manipulate grid quantities.
3. [meshing_of_fractures](./meshing_of_fractures.ipynb) describes the construction of grid buckets for mixed-dimensional grids.
4. [single_phase_flow](./single_phase_flow.ipynb) shows different discretization methods available for the pressure equation.
5. [ad_framework](./ad_framework.ipynb) describes how to solve a problem using the AD framework. The tutorial includes setup of parameters and discretizations.
6. [incompressible_flow_model](incompressible_flow_model.ipynb) describes how to use a model class `Incompressible Flow`. The tutorial exposes several extensions and how to solve a mixed-dimensional problem with a few lines of code.