================
How to docstring
================

.. automodule:: example_docstrings
    :no-members:

Docstring-Basics
================

1. Python docstrings are started and ended using a triple \"\"\", and are placed *below* the
   declaration of an object.

   .. code:: Python
        
        var_1: int = 1
        """This is a docstring for ``var_1``."""

2. Inline literals are rendered using \`\`: \`\`literal\`\` becomes ``literal``.
3. Every member (variable, function, class or method) must have a docstring.
   Though only *public* members are included by current configurations, it is demanded that all members,
   including *private* and *special* members have a proper docstring.
4. From the docstring it should be clear what an object does or is, what it takes as arguments
   and what it returns. Avoid overly scientific or complicated language. 
5. Sphinx provides extensive linking and cross-referencing features.
   See e.g., `Cross-referencing syntax <https://sphinx-experiment.readthedocs.io/en/latest/markup/inline.html#xref-syntax>`_
   and `Python object referencing <https://sphinx-experiment.readthedocs.io/en/latest/domains.html#python-roles>`_.
   The domain ``:py`` can be omitted in this project.
6. Limit the length of every line to 88 characters. PorePy's PR routines include checks using
   ``isort``, ``black``, ``flake8`` and ``mypy``, which check exactly that.
   
   .. hint::
        Most IDEs support a modification of the code editor such that a vertical line is displayed
        after the 88th character column.

7. End every multi-line docstring with a blank line before \"\"\", independent of what you are
   documenting.

Directives
----------

Next to plain text, the content of your documentation can be created using *directives*,
followed by a block of indented text.

Directives are structural blocks inside the text of a documentation.
They provide specific formatting for sections dedicated to e.g., function parameters,
return values or exmaple code.

For example, the *note* directive in plain rst code ::

    .. note:: Implementation
        
        This is a note about the way something is implemented.

will compile into

.. note:: Implementation
    
    This is a note about the way something is implemented.

Directive breaks are created by resuming non-indented text. They are also indirectly
created anytime a new section or directive starts.

Other useful directives are::

    .. code-block:: Python
        :linenos:

        # we demonstrate an example usage of Python code,
        #including the option 'linenos' to display line numbers
        print("Hello World!")
    
    .. deprecated:: VERSIONNUMBER
        This is deprecated, use ``X`` instead.

    .. todo:: missing functionality
        
        * This is a a to-do list
        * for missing functionality.

resulting in

.. code-block:: Python
    :linenos:
    
    # we demonstrate an example usage of Python code,
    #including the option 'linenos' to display line numbers
    print("Hello World!")

.. deprecated:: VERSIONNUMBER
    This is deprecated, use ``X`` instead.

.. todo:: missing functionality
    
    * This is a a to-do list
    * for missing functionality.

Other types of useful
`directives in Sphinx <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html>`_
can be found online. 

.. warning::
    Some directives have all kind of *options*.
    Options demand to be at the top of the indented text and a 
    blank line between them and the directive's content.
    
    Additionally, some directives allow text or input after ``::``, some do not.
    Some have no options at all and you do not need blank lines.
    Some need blank lines before and after the directive and its indented block. Etc....
    
    **Get familiar with each directive before you use it**.

PorePy style
============

As mentioned above, the PorePy style is Google Style with minor changes. These changes involve
the developer team's principles of good code and aesthetics. They also involve the project's 
style guide, especially naming conventions. Some general rules are provided below. 
Examples of acceptable documentation are provided in the subsections after.

.. rubric:: Naming conventions

Certain acronyms are reserved and developers are encouraged to use them consistently in their
docstrings to provide a uniform appearance.
Acronyms are formatted with inline literals using \`\`x\`\`.
These include (among others):




.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Name
     - Type
     - Definition
   * - intf
     - pp.MortarGrid
     - Grid representing an interface
   * - mdg
     - pp.MixedDimensionalGrid
     - Mixed-dimensional grid
   * - p
     - E.g. pp.ad.Variable
     - Abbreviation for pressure
   * - primary, secondary
     - None
     - Pre- or suffix identifying the two subdomains of an interface. Unless the two subdomains have the same dimension, sd_primary.dim > sd_secondary.dim.
   * - sd
     - pp.Grid
     - Grid representing a subdomain
   * - T
     - E.g. pp.ad.Variable
     - Abbreviation for temperature
   * - u
     - E.g. pp.ad.Variable
     - Abbreviation for displacement
   * - a
     - E.g. np.ndarray
     - Aperture
   * - nd
     - int
     - The ambient dimension of a domain, usually corresponding to the highest dimension of any subdomain in a mixed-dimensional grid (3 or 2).
   * - data
     - dict
     - Dictionary containing data related to a subdomain or interface, including parameters and solution vectors. Sometimes pre or suffixed with e.g. “sd” or “secondary”.
   * - AD
     - str
     - Documentation of automatic differentiation.
   * - num_dofs
     - int
     - Number of degrees of freedom
   * - num_points
     - int
     - Number of points
   * - num_cells
     - int
     - Number of cells

.. rubric:: Google style directives

Google style directives are aliases for standard directives, which simplify the usage and syntax.
Besides the obligatory declaration of parameters, raised errors and return values,
the PorePy style encourages the use of other sections to provide notes, example code usage, references, etc.

For a list of all supported Google style directives,
`see Google style docstring sections <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#docstring-sections>`_.

The **compulsory order** of directives inside a docstring is:

    1. Examples, Note, ...
    2. References, See Also
    3. Parameters
    4. Raises
    5. Returns/ Yields

Other rules:

    * When using Google Style directives do not type additional text **between** and **after** the
      directives References/See Also, Parameters, Raises and Returns/Yields.
    * End every docstring with a blank line before \"\"\". This is especially important after
      the *Returns:* directive and its indented block.

.. note::
    The Google Style directives are only valid when used inside docstrings,
    but not so when used in .rst files. Keep this in mind, in case you deal with the .rst
    structure of the webpage.

.. rubric:: Type hints and annotations
    
PorePy demands a complete provision of annotations using
`Python typing <https://docs.python.org/3/library/typing.html>`_.
This implicates that the type of every variable, attribute, argument and return value is known.
Sphinx is able to interpret and create respective intra- and inter-doc linking based on 
the annotations.

Type hints can be given using ``(type):``, with or without brackets, at multiple occasions.
But doing so overrides the configurations of Sphinx,
rendering the type annotations useless and making the documentation inconsistent.

.. important::

    As a consequence, **we abstain from including type hints in docstrings** and keep them light.

    **Do:**

    .. code:: Python

        def my_function(argument: type) -> ReturnType:
            """...

            Parameters:
                argument: Description of argument.

            Returns:
                Description of return value.

            """

    **Do NOT:**

    .. code:: Python

        def my_function:
            """...

            Parameters:
                argument (type): Description of argument.

            Returns:
                ReturnType: Description of return value.

            """

    **Do:**

    .. code:: Python

        variable: str = "A"
        """Description of variable."""

    **Do NOT:**

    .. code:: Python

        variable: str = "A"
        """str: Description of variable."""

.. rubric:: Linking/ Cross-referencing to other objects

Linking to other objects or members using

- `:obj:`,
- `:class:`,
- `:func:`,
- `:meth` and
- `:data:`

to provide useful cross-referencing *in the text* is encouraged.
This will help readers to quickly navigate through the docs.

Documenting variables
---------------------

.. autodata:: example_docstrings.example_var_1

.. autodata:: example_docstrings.example_var_2

.. autodata:: example_docstrings.example_var_3

.. autodata:: example_docstrings.ExampleArrayLike


Documenting functions
---------------------

.. autofunction:: example_docstrings.example_function_1

.. autofunction:: example_docstrings.example_function_2

.. autofunction:: example_docstrings.example_function_3

.. autofunction:: example_docstrings.example_function_4

.. autofunction:: example_docstrings.example_function_5

.. autofunction:: example_docstrings.example_generator

Documenting classes
-------------------

When documenting classes, three aspects have to be considered.

.. rubric:: Documenting Constructor Arguments

In PorePy, the constructor description is to be included in the class-docstring, not in the 
``__init__``-docstring!
**There must be no ``__init__``-docstring at all!**

The *class-docstring* is written directly below the declaration of the class

.. code:: Python

    class ExampleClass:
        """ Class-docstring """

        def __init(self):
        """ __init__ - docstring """

whereas the ``__init__``-docstring is found below the constructor declaration.

Inside class docstrings, you can sue the same directives as for functions.
**The same rules apply.**

.. rubric:: Documentation of attributes

Attributes must be documented using docstrings.
This holds for instance-level attributes set in the constructor,
as well as for class-level attributes.
PorePy's sphinx build is configured such that it will render those docstrings the same way as method
docstrings.

Though only public attributes are included by Sphinx' configuration,
private and name-mangled attributes must be documented as well.

In any case, use type annotations in the Python code and **do not** use type hints in the
docstring. This will avoid a second, redundant display of the type
(similar for argument and return type hints).

This guideline holds for instance-level attributes, as well as for class-level attributes.

Though attributes can be documented in the class docstring using the directive
*Attributes:*, abstain from doing so. Here the type annotation cannot be exploited in all 
cases and additional type hints would be necessary in the docstrings.

The following example shows why we do not use *Attributes:*

.. code:: Python
    
    """
    Attributes:
        class_attribute_1: This is a class-level attribute documented in the
            class docstring.
            The annotation would be exploited in this case.
        attribute_1 (int): This is an instance-level attribute documented in the
            class docstring.
            Here we would have to use additional type hints because the annotation inside 
            the constructor is not exploited by Sphinx.

            If you document any attribute here AND in a custom docstring, Sphinx will raise an
            error.
    """

.. rubric:: Documenting methods


* All rules introduced in section `Documenting functions`_ for regular functions,
  hold for methods, class methods and static methods as well,
  independent of whether they are public, private or special.
* Never document the ``self`` argument of a method, or the ``cls`` argument of a class method.
* By default, only public methods are included in the documentation.
  Nevertheless, private methods and special methods need to be documented
  properly as well. If the developer wants them included in the compiled documentation, he
  must provide respective options to the audodoc directives, later when integrating his
  documentation.

Sphinx groups the documentation of attributes, properties, methods, class methods
and static methods and sorts them according to the configuration.

.. rubric:: Example class documentation

.. autoclass:: example_docstrings.ExampleClass

.. autoclass:: example_docstrings.ChildClass

Module-level documentation
--------------------------

Module level documentation can be achieved by writing a docstring at the very top of a file.
Subpackages i.e., folders containing an ``__init__`` file and multiple modules, have their
docstring at the top of the ``__init__`` file.
These docstrings usually describes the purpose and the contents of the module/package and can
be included in the compiled documentation.

They can be used as an introduction to the content.
In Fact this page's introduction is written in the module-docstring and
imported into the compiled documentation. It is then supplemented with additional content like
the sections on `Docstring-Basics`_ or `Directives`_.

The content and design of module-level docstrings should be discussed with the core developers.