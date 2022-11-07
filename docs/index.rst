Welcome to PorePy!
==================

PorePy is a research code that is mainly developed at the Department of Mathematics at the University of Bergen.
The code is aimed at simulation of multiphysics processes in fractured porous media.
Specifically, the code is typically applied to

1. Develop simulation technology for fractured porous media. This includes discretization methods and coupling schemes.
2. Study processes that cannot be properly represented by standard simulation tools.

As a by-product of these two main use cases, a third main research topic is

3. Develop design principles for simulation software for mixed-dimensional multiphysics problems.

What is PorePy good at?
-----------------------

The main strengths of PorePy are:

* Natural representation of variables and processes in fracture networks, and on the fracture-matrix interface.
* Semi-automatic construction of meshes that conform to fracture networks.
* Discretization methods for flow, deformation, transport and frictional fracture deformation.
* Flexibility in combining discretizations and constitutive relations in the matrix and in the fracture network.

Other notable features are:

* Simple functionality for fracture propagation (newly added, use with care).
* Extensive functionality for computational geometry.
* Automatic differentiation. Under active improvement.

What is PorePy not so good at?
------------------------------

- No native support for linear and non-linear solvers. This will be improved in the near future.
- Few standardized setups are available; mostly, the user is responsible for specifying problems and discretizations.
  The exception is the hierarchy of models for mechanics, poromechanics and thermo-poromechanics, all coupled with fracture deformation.

PorePy cannot be expected to function as a black box tool for general multiphysics problems.
While simple flow and transport problems can be solved relatively straightforwardly,
more complex problems will likely require diving into parts of the source code.
This is partly due to lack of resources from the developer side,
but it also reflects that these are difficult problems for which robust and standardized setups cannot be defined.

The PorePy package
==================

PorePy is divided into the following modules

.. toctree::
   :titlesonly:

   docsrc/porepy/numerics/numerics

For developers
==============

Below you will find guidelines and instructions for developing activities in PorePy.
All new developers are required to familiarize themselves with the *How-To-X* sections before
attempting a pull-request.

.. toctree::
   :numbered:
   :titlesonly:
   
   docsrc/howto/howto-docstring

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
