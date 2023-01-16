==========================
How to build documentation
==========================

PorePy's documentation is built using `Sphinx <https://www.sphinx-doc.org/en/master/>`_.
In this section you will find information on requirements and instructions on how to
build and inspect the documentation of your source code, using PorePy's configurations.

The following packages are required:

1. ``sphinx``
2. ``sphinx-rtd-theme``

You can instal above packages using ``pip`` in your working environment. Make sure they
are installed in the same environment as the PorePy version you are currently working with.

After installing the required packages you need to get all files necessary for the build.
They can be found on the `docs branch <https://github.com/pmgbergen/porepy/tree/docs>`_
on GitHub. Checkout this branch and copy the directory ``/docs/`` somewhere so you can access it
after switching back to your branch or environment.

Copy the ``/docs/`` directory into your ``/porepy/`` repository on the same level as ``/porepy/src``,
i.e. ``/porepy/docs/``.
Go to ``/porepy/docs/`` and run the following command:

- Linux: ``make html``
- Windows (Powershell): ``.\make.bat html``

This will run the Sphinx builder and create a local version of PorePy's documentation in ``/porepy/docs/html/``.

To inspect the webpage, open ``porepy/docs/index.html`` in a browser.
Navigate through the documentation and inspect what you want.

Re-building the complete API
============================

PorePy uses ``sphinx-apidoc`` to produce a complete documentation of the package.

It will produce the complete webpage structure using ``sphinx-autodoc`` by importing Porepy as is.
You do not have to (and should not) deal with the details, as PorePy has a complete Sphinx configuration in place.
Your code must be compatible with the latest style guid and PorePy settings.

If you want to re-build the webpage/rst structure from scratch, you can use the target ``complete``:

- Linux: ``make complete html``
- Windows (Powershell): ``.\make.bat complete html``

This option will run ``apidoc`` as configured by PorePy before the builder,
and it will produce the whole structure (stored in ``/porepy/docs/docsrc/porepy/``).

Use ``porepy/docs/index.html`` again as the entrypoint to inspect the docs.

.. note::

    Running the build from scratch is especially important if the structure of the PorePy package is modified.

    I.e. any new module or package, function, class or data which is added after the last run of ``apidoc``
    (including modifications of ``__all__`` in Python files)
    must first be included into the webpage structure using ``apidoc``.

.. warning::

    PorePy's documentation is a work in progress. Not all docstrings have been re-worked such that they can be processed
    by Sphinx. Any compilation will as of now yield numerous errors and warnings.
    
    The builder should not fail though and you will obtain a displayable webpage.

Building the documentation for *LaTex*
======================================

If you want to build the documentation for LaTex, to be subsequently compiled into a pdf,
use the target ``latex`` instead of ``html``

- Linux: ``make [complete] latex``
- Windows (Powershell): ``.\make.bat [complete] latex``

The output is stored in ``/porepy/docs/latex/``.

.. note::

    PorePy does not include a LaTex engine or respective compilation.
    You have to install an engine by yourself and compile the pdf manually using the files
    created in ``/porepy/docs/latex/``.

Building and inspecting local docs
==================================

During development it is often necessary to inspect the docs of a single module or Python code.

It is time-consuming to build the whole PorePy documentation every time you make minor local changes.
To avoid that you can build your own webpage which compiles only the docs you want to inspect.

To do so, perform the following steps:

1. Create an own page ``/porepy/docs/my_page.rst`` as a rst file and write a title inside your file

   .. code::

       =============
       My Page Title
       =============

2. Open ``/porepy/docs/index.rst`` and navigate to the section ``Documentation``.
3. Modify the TOC tree such that it points to ``my_page.rst``, i.e. replace

   ``docsrc/porepy/modules``

   with

   ``my_page``.

   This will replace the PorePy pages with your own.
4. Use an ``autodoc``-directive of your choice to import the code you wish to build.

   If you have created new code, say ``/porepy/src/porepy/my_package/my_module.py``,
   you can add the following lines in ``my_page.rst`` to have your code included:

   .. code::

       .. automodule:: porepy.my_package.my_module
           :members:
           :undoc-members:
           :private-members:
           :special-members:

   This will import every member available in your ``my_module.py`` file and build respective docs.

   Provide the **complete** namespace to your file, i.e. ``porepy.my_package.my_module``. 
   The file name alone will not suffice.

   Run the make command again to build your docs. It might be necessary to run it several times for Sphinx
   to realize the changes.
5. Open again ``porepy/docs/index.html`` and you will find a link to your webpage instead of a link to
   PorePy's documentation on the main page.
6. Inspect your documentation, modify your Python code, run make and repeat until you are done.

There are solutions to update the webpage in realtime without having to re-run the Sphinx builder.
They rely mostly on third-party software and are not included in PorePy.
Feel free to use them on your own responsibility.

.. note::
    When working on docstrings and inspecting your documentation,
    make sure you resolve **all** errors and warnings Sphinx produces!
    
    If your code is to be included into the main branch, an error- and warning-free documentation will be required.
