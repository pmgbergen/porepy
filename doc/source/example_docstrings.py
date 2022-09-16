"""
Welcome to the *How-To* on creating documentation for your functions, classes and submodules.
Here we explain shortly how the documentation works and give examples for how to write proper
docstrings.

The PorePy documentation is created using *Sphinx*, an engine effectively importing the Python
package and extracting its docstrings. The docstrings are then used to fill a pre-defined
structure and create a HTML-based documentation.

Docstrings are written using a markup language called *reStructuredText* (short *rst*).
A quick overview can be found at
`rst-quickref <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_.
You can also find vast support for rst elsewhere on the internet.

PorePy follows closely the
`Google Python Style <https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google>`_
for its docstrings. The Sphinx-extension *napoleon* ensures a proper interpretation.

While the general layout follows Google's recommendations, PorePy introduces minor adaptions
for readability and functionality reasons. After familiarizing yourself with rst and the
Google style, you can read through the rest of this How-To to get to know the PorePy specifics.

Once you have programmed, tested, documented and *merged* your contribution into the PorePy 
source, contact one of the core developers to ensure a proper integration of your documentation
into the whole structure.

-----

In the following we demonstrate how to document various aspects of your code in
`PorePy style`_.
While we aim for providing complete instructions, we cannot guarantee the coverage of every 
possible case of the vast Python world. If you are unsure about an issue, don't hesitate to
contact one of the core developers.

Next to the compiled documentation you will find fleeting links ``[source]`` to the source code 
``example_docstrings.py``, where you can see the raw docstring in all its rst-glory and the
respective Python code. This way you will be able to directly compare what certain syntax will
in the end look like in the compiled format.

Once the your code is properly documented, it will be included using
`autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ directives
into the global structure of the PorePy documentation.
This will be done in a next step. Before that, familiarize yourself with the various directives
and their options beforehand.

""" 

import porepy as pp 
import numpy as np 
import scipy.sparse as sps 

from typing import Generator, Optional


module_level_var_1: str = "var" 
"""Variables at the module level are documented inline after declaration."""


module_level_var_2: int = 1 
"""Multi-line docstrings are also valid. 

You might want to use multiline docstrings when a more detailed description
of the variable is required.

"""

module_level_var_3: int = 42
"""int: **DO NOT** start the docstring with ``int:``, i.e. with type hints.

This will render the below ``Type: int``, which is unnecessary due to the already present
annotation after the variable name. As you can test above, clicking on *int* will forward you
to the right documentation.

**DO:**

.. code:: Python

    module_level_var_3: int = 1
    \"\"\"This is var 3.\"\"\"

**DO NOT:**

.. code:: Python
    
    module_level_var_3 = 1
    \"\"\"int: This is var 3.\"\"\"

"""

def module_level_function_1(arg1: int, arg2: str, arg3: bool) -> bool: 
    """A short description of the functionality must fit in the first line. 
    
    Detailed description can be given next if necessary. The extended description 
    can of course span several lines.     

    When natural, the description can be break into several paragraphs that start 
    without indentation. 

    The description of arguments is started using the Google Style directive *Parameters:*,
    or equivalently *Args:*.

    .. note:: 

        You can find the ``[source]`` link in the upper right corner of this function's
        documentation and take a look at the Python and rst source code of this docstring, 
        as well as for every docstring that follows.

    The return value can be described using the Google Style directive *Returns:*.

    Parameters: 
        arg1: Description of arg1.
        arg2: Description of arg2. If the description of a parameter spans more than one line,
            the lines must be indented.  

            You can even use multiple paragraphs for describing a parameter. There 
            are no restrictions in this regard.
        arg3 (bool): **DO NOT** include type hints in the *Parameters* directive.
            For this the annotations in the signature are utilized.

    Returns:
        Describe the return value here. Multiple lines can be used and you have
        to indent the lines.
        
        **DO NOT** use type hints here as they are present in the signature.

    """   
    return (arg1 > len(arg2)) and arg3

 
def module_level_function_2(optional_arg: Optional[int]=None, *args, **kwargs) -> None: 
    """Optional arguments must be type-hinted using :class:`typing.Optional`.
    
    As evident, you can link objects in-text. You can also link functions like
    :meth:`module_level_function_1`.

    If ``*args`` or ``**kwargs`` are accepted,
    they must be listed as ``*args`` and ``**kwargs`` in the *Parameters* directive. 

    Args: 
        optional_arg: Description of ``optional_arg``.
            The default value is described at the end.
            Use easily spottable wording like 'Defaults to None.'
        *args: Arbitrary arguments. Describe adequately how they are used. 
        **kwargs: Arbitrary keyword arguments. You can use rst-lists to describe admissible
            keywords:

            - ``kw_arg1``: Some admissible keyword argument
            - ``kw_arg2``: Some other admissible keyword argument

    Raises: 
        ValueError: You can use the the Google Style directive *Raises:* to describe error
            sources you have programmed. Here for example we can raise  value error
            if an unexpected keyword argument was passed.
        TypeError: You can hint multiple errors. Be aware of necessary indentations when
            descriptions span multiple lines.

    .. note::
        We note that this function returns ``None`` as per signature.
        This is the default return value of Python and as such does not need documentation.
        I.e., we can omit the *Returns:* directive. 

    """   
    pass


def module_level_function_3(
    mdg: pp.MixedDimensionalGrid,
    vector: np.ndarray,
    matrix: sps.spmatrix,
    option: str = "A"
    ) -> sps.spmatrix: 
    """We demonstrate here the intra- and inter-doc linking to non-base-type Python objects
    in the signature,
    as well as how to document special requirements for the arguments.

    Args: 
        mdg: A mixed-dimensional grid. Notice the usage of the acronym ``mdg``
        vector (len=nc): This vectors special requirement is a certain length.
            We use the type hint brackets to indicate its supposed length.
            The required length is ``nc``, the number of cells (obviously in ``mdg``).
        matrix (shape=(3,nc)): This argument is a sparse matrix with 3 rows and ``nc``
            columns. 
        option ({'A', 'B', 'C'}): This argument has admissible values, namely it can only be 
            a string ``'A'``, ``'B'`` or ``'C'``.
 
    Returns: 
        Describe the return value here. You can also describe its dependence on the value of
        ``option`` here, or alternatively in the extended description at the beginning of the
        docstring.
 
    Raises: 
        ValueError: If ``option`` is not in ``{'A', 'B', 'C'}``. 

    """ 
    pass
 
 
def example_generator(n: int) -> Generator[int, None, None]: 
    """When documenting a generator, use the *Yields:* directive instead of *Returns:*.
    Their usage is equivalent, indicating only that this is a generator object, not a regular
    function.
 
    Args: 
        n: Some upper integer limit. 
 
    Yields: 
        numbers until ``n`` is reached. 
 
    Examples: 
        The Google Style offers an *Examples:* directives where you can write text but also
        include inline examples of code usage, without the special code directive from Sphinx.
 
        >>> print([i for i in example_generator(4)]) 
        [0, 1, 2, 3] 

    """ 
    for i in range(n): 
        yield i 
 
 
class ExampleClass: 
    """A quick purpose of the class is to be provided in the first line.
    
    After the first line, a broader, multi-line description of the class can follow. It can
    contain multiple paragraphs as well as elements like code snippets.

    At the end follows the *Parameters:* directive and an eventual *Raises:* directive.

    Parameters:
        arg1: first argument
        arg2: second argument    

    """

    class_attribute_1: int = 1
    """This is how to properly document a class-level attribute.
    
    The value is shown because it is static upon class type creation.

    """

    class_attribute_2: str = "A"
    """str: This is a class attribute with a redundant field 'Type:' caused by type hints
    in the docstring. The type should only be given by its annotation.
    **DO NOT** use type hints in docstrings.

    """ 
 
    def __init__(self, arg1: int, arg2: str) -> None:
        # !!! no docstring for __init__ !!!

        self.attribute_1: int = arg1 
        """This is how to properly document a instance-level attribute inside the constructor.
        """

        self.attribute_2: str = arg2
        """str: This is an instance-level attribute with a redundant field 'Type:' caused by
        type hints in the docstring. The type should only be given by its annotation.
        **DO NOT** use type hints in docstrings.

        """
     
    @property 
    def readonly_property(self) -> str: 
        """Document properties only in their getter-methods.""" 
        return 'readonly' 
 
    @property 
    def readwrite_property(self) -> str: 
        """Properties with both a getter and setter should only be documented in their
        getter method. 
 
        If the setter method contains notable behavior, it should be 
        mentioned here. 
        """ 
        return 'readwrite' 
 
    @readwrite_property.setter 
    def readwrite_property(self, value): 
        value 
    
    @staticmethod
    def example_static_method() -> None:
        """This is the docstring of a static method.
        
        Note the absence of ``self``.

        """
    
    @classmethod
    def example_class_method(cls) -> None:
        """This is the docstring of a class method.

        Note that the ``self`` argument is replaced by the ``cls`` argument.

        """
 
    def example_method(self, arg: bool) -> bool: 
        """A short description of the method must fit in the first line.

        A more elaborate description can follow afterwards.
 
        Args: 
            arg: The first argument (even though ``self`` is technically the first one).
 
        Returns: 
            the argument itself. 
 
        """ 
        return True 

    def __special__(self) -> None: 
        """Special methods (starting and ending with two underscores) are by default not
        included. Their documentation needs to be assured by an *option* in the rst files.
 
        """ 
        pass 
 
    def _private(self) -> None: 
        """Private methods are by default not included in the compiled documentation.
        They should be nevertheless documented properly
 
        """ 
        pass 
