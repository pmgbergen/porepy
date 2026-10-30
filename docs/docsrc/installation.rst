============
Installation
============

Choose the installation method that best fits how you want to use PorePy:

* **Docker** is the quickest way to use a prebuilt stable release.
* **GitHub Codespaces** provides a browser-based development environment.
* **Installation from source** gives you direct control over the version and Python environment.

Docker image
============

The recommended option for using a stable release is the PorePy Docker image.
Install `Docker <https://www.docker.com/>`_, then pull and run the image:

.. code-block:: console

   docker pull porepy/stable
   docker run -it porepy/stable

For development, the README also describes the ``porepy/dev:latest`` image.
The ``porepy/extended`` image includes PETSc and linear solvers; its ``pp_solver``
component is experimental and has limited documentation. See the
`repository README <https://github.com/pmgbergen/porepy/blob/develop/Readme.md>`_
for details.

GitHub Codespaces
=================

To try PorePy in a browser, create a
`Codespace for the develop branch <https://github.com/codespaces/new?repo=pmgbergen/porepy&ref=develop>`_.
The repository's devcontainer configures the environment and installs Jupyter,
so you can work with PorePy and its tutorial notebooks without setting up a
local Python environment. Initial environment setup can take a few minutes.

Install from source on Linux
============================

Clone the `PorePy repository <https://github.com/pmgbergen/porepy>`_ and enter
the project directory:

.. code-block:: console

   git clone https://github.com/pmgbergen/porepy.git
   cd porepy

The default branch, ``develop``, contains ongoing development. To use the
stable release branch instead, check out ``main``:

.. code-block:: console

   git checkout main

Install PorePy with its development and testing dependencies:

.. code-block:: console

   python -m pip install ".[development,testing]"

For development on the checked-out source, use an editable install:

.. code-block:: console

   python -m pip install -e ".[development,testing]"

The source installation instructions and optional system dependencies are
described in the repository's
`Install.md <https://github.com/pmgbergen/porepy/blob/develop/Install.md>`_.
For contribution workflow and code checks, see
`Contributing to PorePy <https://github.com/pmgbergen/porepy/blob/develop/CONTRIBUTING.md>`_.

After installation, see the :doc:`tutorials` for example models, numerical
methods, grid construction, and benchmark problems.
