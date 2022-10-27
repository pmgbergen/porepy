================
How-To docstring
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
3. Every public object must have a docstring. It's good practice (not to say demanded) to
   document private objects as well, but they will not be included in the docs by default.
4. From the docstring it should be clear what an object does or is, what it takes as arguments
   and what it returns. Avoid overly scientific or complicated language. 
5. Sphinx provides extensive linking and cross-referencing features.
   See e.g., `Cross-referencing syntax <https://sphinx-experiment.readthedocs.io/en/latest/markup/inline.html#xref-syntax>`_
   and `Python object referencing <https://sphinx-experiment.readthedocs.io/en/latest/domains.html#python-roles>`_.
   The domain ``:py`` can be omitted in this project.
6. End every multi-line docstring with a blank line before \"\"\", independent of what you are
   documenting.

Directives
----------

Next to plain text, the content of your documentation can be created using *directives*,
followed by a block of indented text.

For example, the *note* directive in plain rst code ::

    .. note:: Implementation
        
        This is a note about the way something is implemented.

will compile into

.. note:: Implementation
    
    This is a note about the way something is implemented.

Directive breaks are created by resuming non-indented text. They are also indirectly
created anytime a new section starts.

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
style guide, especially naming conventions. For more information check out the
official PorePy User Manual. Some general rules are provided below. 
Examples of acceptable documentation are provided in the subsections after.

.. rubric:: Naming conventions

Certain acronyms are reserved and developers are encouraged to use them consistently in their
docstrings to provide a uniform appearance.
Acronyms are formatted with inline literals using \`\`x\`\`.
These include:

- ``sd`` : single SubDomain
- ``mdg`` : MixedDimensionalGrid
- ``p``, ``T``, ``u``,... : names of common physical variables
- ``nd`` : ambient dimension of a ``mdg``, corresponding to the highest dimension of subdomains
- ``np`` : number of points
- ``nc`` : number of cells


.. rubric:: Typing and linking
    
PorePy demands a complete provision of annotations using
`Python typing <https://docs.python.org/3/library/typing.html>`_.
This implicates that the type of every variable, attribute, argument and return value is known.
Sphinx is able to interpret and create respective intra- and inter-doc linking based on 
the annotations.

As a consequence, **we abstain from including type hints in docstrings** and keep them light.

This rule does not exclude custom linking and referencing in-text, where a developer feels the
necessity to do so.

The only exemption to this rule is when providing types of *Attributes* in class-docstrings.
See `Documenting classes`_ for more.


Documenting variables
---------------------

.. autodata:: example_docstrings.module_level_var_1

.. autodata:: example_docstrings.module_level_var_2

.. autodata:: example_docstrings.module_level_var_3

Documenting functions
---------------------

.. autofunction:: example_docstrings.module_level_function_1

.. autofunction:: example_docstrings.module_level_function_2

.. autofunction:: example_docstrings.module_level_function_3

.. autofunction:: example_docstrings.example_generator


.. note::
    1. When using Google Style directives do not type text between two directive blocks
    2. Write the directives blocks in the following order
        1. Examples
        2. Parameters
        3. Returns/ Yields
        4. Raises
    3. Do not write any more text after the directive *Raises*. End every docsting with a blank
       line before \"\"\".

Documenting classes
-------------------

When documenting classes, three aspects have to be considered:

.. rubric:: Documenting Constructor Arguments

In PorePy, the constructor description is to be included in the class-docstring, **not** in the 
``__init__``-docstring! There must be no ``__init__``-docstring at all!

The *class-docstring* is the one docstring written directly below the declaration of
the class

.. code:: Python

    class ExampleClass:
        """ Class-docstring """

        def __init(self):
        """ __init__ - docstring """

whereas the ``__init__``-docstring is found below the constructor declaration.

At the very end of the class-docstring, use the Google Style directives
*Examples:*, *Parameters:* and *Raises:*,
in the same order as for functions, to document how to instantiate a class and if errors might
be raised.

.. rubric:: Documentation of attributes

Public attributes must be documented using docstrings below their declaration.
This holds for instance-level attributes set in the constructor,
as well as for class-level attributes.
Sphinx is configured such that it will render those docstrings the same way as method
docstrings.

In any case, use type annotations in the Python code and **do not** use type hints in the
docstring. This will avoid a second, redundant display of the type
(similar for argument and return type hints).

This guideline holds for instance-level attributes, as well as for class-level attributes.

Though attributes can be documented in the class docstring using the directive
*Attributes:*, abstain from doing so. Here the type annotation cannot be exploited in all 
cases and additional type hints would be necessary in the docstrings.

.. code:: Python
    
    """
    Attributes:
        class_attribute_1: This is a class-level attribute documented in the
            class docstring.
            The annotation is exploited in this case.
        attribute_1 (int): This is an instance-level attribute documented in the
            class docstring.
            Here we would have to use additional type hints because the annotation inside 
            the constructor is not exploited by Sphinx.

            If you document any attribute here AND in a custom docstring, Sphinx will raise an
            error.
    """

Attributes are grouped by Sphinx into the two groups, instance- and class-level,
and rendered the same way as methods.

.. rubric:: Documenting methods

Note:
    All rules introduced in section `Documenting functions`_ for regular functions,
    hold for methods, class methods and static methods as well,
    independent of whether they are public, private or special.

Note:
    Never document the ``self`` argument of a method, or the ``cls`` argument of a 
    class method.

By default, only public methods are included in the documentation.
Nevertheless, private methods and special methods need to be documented
properly as well. If the developer wants them included in the compiled documentation, he
must provide respective options to the audodoc directives, later when integrating his
documentation.

Sphinx groups the documentation of attributes, properties, methods, class methods
and static methods and sorts them alphabetically inside each group.

.. autoclass:: example_docstrings.ExampleClass

Module-level documentation
--------------------------

Module level documentation can be achieved by writing a docstring at the very top of a file.
Subpackages i.e., folders containing an ``__init__`` file and multiple modules, have their
docstring at the top of the ``__init__`` file.
These docstrings usually describes the purpose and the contents of the module/package and can
be included in the compiled documentation.

Though we do not include them by default, it can be used as an introduction
to the content. In Fact this page's introduction is written in the module-docstring and
imported into the compiled documentation. It is then supplemented with additional content like
the sections on `Docstring-Basics`_ or `Directives`_.

The decision on whether and how to incorporate the module-level docstrings should be made upon
integration of a module's documentation with the core developers.