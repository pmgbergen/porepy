![Pytest](https://github.com/pmgbergen/porepy/actions/workflows/run-pytest.yml/badge.svg)
![Pytest including slow](https://github.com/pmgbergen/porepy/actions/workflows/run-pytest-all.yml/badge.svg)
![Mypy, ruff, isort](https://github.com/pmgbergen/porepy/actions/workflows/run-static-checks.yml/badge.svg)
![Tutorials](https://github.com/pmgbergen/porepy/actions/workflows/check_tutorials.yml/badge.svg)
[![DOI](https://zenodo.org/badge/89228838.svg)](https://zenodo.org/badge/latestdoi/89228838)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


# PorePy: A Simulation Tool for Fractured and Deformable Porous Media written in Python.
PorePy is a simulation tool that targets multiphysics processes in fractured porous media. PorePy comes with:
* Functionality for automatic mesh generation for complex fracture networks in two and three dimensions.
* Numerical methods that allow for simulation of non-linearly coupled multiphysics processes.
* Ready-made simulation setups for coupled processes, including:
    * Thermo-poromechanics coupled with fracture deformation.
    * Multiphase flow and transport.
    
    The code design prioritizes easy adaptation of these setups to allow for rapid prototyping.

The video below showcases a fully coupled flow and heat transport simulation in a fractured porous medium with 52 fractures. 
In the simulation, cold fluid is injected through an injection well in the top right of the domain and produced from a production well on the opposite side.

<p align="center">
    <img src='tutorials/img/showcase_thermohydro.gif' width=540>
</p>

# How do I get started?
The best place to start is the [tutorials]( https://github.com/pmgbergen/porepy/tree/develop/tutorials); we suggest looking at the Readme file for guidance on how to approach the tutorials.
The tutorials show how to use the code for various common cases and explain key PorePy functionality. 
For additional inspiration, the [examples](https://github.com/pmgbergen/porepy/tree/develop/src/porepy/examples) folder contains a curated collection of simulation setups, including flow and poromechanics benchmarks, which can serve as starting points for your own simulations.

The code can be accessed in several ways:
*	The most immediate access is running PorePy in your web browser. If you have a GitHub account, use a GitHub codespace for PorePy following [these instructions]( https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository#creating-a-codespace-for-a-repository). Note that the building time will be a few minutes.
*	If you want to run the code on your own machine, you have two options:
    *	If you have Docker installed, we recommend pulling the PorePy Docker image through ‘docker pull porepy/stable’.
    *	To install PorePy manually, follow the [install instructions]( https://github.com/pmgbergen/porepy/blob/develop/Install.md).

Documentation can be found [here](https://pmgbergen.github.io/porepy/html/docsrc/porepy/porepy.html) (still under construction).

# How can I get involved?
Please see the [guidelines]( https://github.com/pmgbergen/porepy/blob/develop/CONTRIBUTING.md) for contributing.

# Acknowledgements
PorePy is mainly developed by the [Porous Media Group](https://www4.uib.no/en/research/research-groups/porous-media-group) at the University of Bergen, Norway. 
The software is developed under projects funded by the Research Council of Norway, the European Research Council and Equinor.


# Citing
If you use PorePy in your research, we ask you to cite the following publication

Keilegavlen, E., Berge, R., Fumagalli, A., Starnoni, M., Stefansson, I., Varela, J., & Berre, I. PorePy: an open-source software for simulation of multiphysics processes in fractured porous media. Computational Geosciences,  25, 243–265 (2021), [doi:10.1007/s10596-020-10002-5](https://doi.org/10.1007/s10596-020-10002-5)
