=========
Tutorials
=========

PorePy's tutorials are Jupyter notebooks that introduce the package through
hands-on examples. They cover the major parts of PorePy, from setting up models
and multiphysics simulations to numerical methods, grids, and benchmarks.

Modelling and multiphysics
==========================

Learn how to configure and run PorePy models, couple physical processes, and
work with fluid and transport models:

* `Single-phase flow <https://github.com/pmgbergen/porepy/blob/develop/tutorials/single_phase_flow.ipynb>`_
* `Poromechanics <https://github.com/pmgbergen/porepy/blob/develop/tutorials/poromechanics.ipynb>`_
* `Fluid modeling <https://github.com/pmgbergen/porepy/blob/develop/tutorials/fluid_modeling.ipynb>`_
* `Tracer flow <https://github.com/pmgbergen/porepy/blob/develop/tutorials/tracer_flow.ipynb>`_
* `Solution strategies <https://github.com/pmgbergen/porepy/blob/develop/tutorials/solution_strategies.ipynb>`_

Numerical methods
=================

Explore discretizations, equations, and solver behavior:

* `Flux discretizations <https://github.com/pmgbergen/porepy/blob/develop/tutorials/flux_discretizations.ipynb>`_
* `Stress discretization <https://github.com/pmgbergen/porepy/blob/develop/tutorials/stress_discretization.ipynb>`_
* `Equations and automatic differentiation <https://github.com/pmgbergen/porepy/blob/develop/tutorials/equations.ipynb>`_
* `Convergence criteria <https://github.com/pmgbergen/porepy/blob/develop/tutorials/convergence_criteria.ipynb>`_

Grids and meshing
=================

See how PorePy represents grids and builds meshes for fractured domains:

* `Mixed-dimensional meshing <https://github.com/pmgbergen/porepy/blob/develop/tutorials/mixed_dimensional_meshing.ipynb>`_
* `Grid class <https://github.com/pmgbergen/porepy/blob/develop/tutorials/grid_class.ipynb>`_
* `Grid topology <https://github.com/pmgbergen/porepy/blob/develop/tutorials/grid_topology.ipynb>`_
* `Geometry generation <https://github.com/pmgbergen/porepy/blob/develop/tutorials/geometry_generation.ipynb>`_

Benchmarks and additional topics
================================

The notebooks also demonstrate benchmark problems and tools for analyzing or
exporting simulation results:

* `Benchmark simulation <https://github.com/pmgbergen/porepy/blob/develop/tutorials/benchmark_simulation.ipynb>`_
* `Mandel's problem <https://github.com/pmgbergen/porepy/blob/develop/tutorials/mandels_problem.ipynb>`_
* `Diagnostics <https://github.com/pmgbergen/porepy/blob/develop/tutorials/diagnostics.ipynb>`_
* `Exporter <https://github.com/pmgbergen/porepy/blob/develop/tutorials/exporter.ipynb>`_

The `tutorials directory on GitHub <https://github.com/pmgbergen/porepy/tree/develop/tutorials>`_
contains all notebooks. Its `README <https://github.com/pmgbergen/porepy/blob/develop/tutorials/Readme.md>`_
provides a suggested reading order and further details.

Running the notebooks
=====================

You can run the notebooks in a local PorePy installation, in a GitHub
Codespace, or in a PorePy Docker container. In a local installation or Docker
container, install Jupyter if it is not already available:

.. code-block:: console

   python -m pip install jupyter ipykernel notebook

Then start Jupyter Lab and open a notebook from the ``tutorials/`` directory.

The repository's `GitHub Codespace <https://github.com/codespaces/new?repo=pmgbergen/porepy&ref=develop>`_
uses the project's devcontainer configuration, which installs Jupyter and the
Jupyter notebook extension. After the Codespace starts, open a notebook from
the ``tutorials/`` directory in VS Code.

To use the notebooks directly from GitHub, browse the
`tutorials directory <https://github.com/pmgbergen/porepy/tree/develop/tutorials>`_.
GitHub renders notebooks in the browser; to execute them, open them in a
Codespace or run them in your local or container-based Jupyter environment.
