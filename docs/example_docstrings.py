"""
Welcome to the *How-To* on creating documentation for your functions, classes and submodules.
Here we explain shortly how the documentation works and give examples for how to write proper
docstrings.

The PorePy documentation is created using *Sphinx*, an engine effectively importing the Python
package and extracting docstrings. The docstrings are then used to fill a pre-defined
structure and create a HTML-based documentation.

Docstrings are written using a markup language called *reStructuredText* (short *rst*).
A quick overview can be found at
`rst-quickref <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_.
You can also find vast support for rst elsewhere on the internet.

Though PorePy closely follows the
`Google Python Style
<https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google>`_,
it introduces minor adaptions for readability and functionality reasons.
After familiarizing yourself with rst and the Google style,
you can read through the rest of this How-To to get to know the PorePy specifics.

Once you have programmed, tested, documented and *merged* your contribution into the PorePy
source, contact one of the core developers to ensure a proper integration of your documentation
into the whole structure.

In the remaining section we demonstrate how to document various aspects of your code in
`PorePy style`_.

While we aim for providing complete instructions, we cannot guarantee the coverage of every
possible case of the vast Python world. If you are unsure about an issue, don't hesitate to
contact one of the core developers.

Next to the compiled documentation you will find fleeting links ``[source]`` to the source code
``example_docstrings.py``, where you can see the raw docstring in all its rst-glory and the
respective Python code. This way you will be able to directly compare what certain syntax will
in the end look like in the compiled format.

Once your code is properly documented, it will be included using
`autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ directives
into the global structure of the PorePy documentation.
This will be done in a next step. Before that, familiarize yourself with the various directives
and their options beforehand.

"""
from __future__ import annotations

from typing import Generator, Literal, Optional, Union
from numpy.typing import ArrayLike, NDArray, DTypeLike

import numpy as np
import scipy.sparse as sps

import porepy as pp


example_var_1: str = "var"
"""Variables at the module level are documented inline after declaration."""


example_var_2: int = 1
"""Multi-line docstrings are also valid.

You might want to use multiline docstrings when a more detailed description
of the variable is required.

In every case, the closing \"\"\" of a multiline docstring must be on a separate line after a
blank line.

**DO:**

.. code:: Python

    var: int = 1
    \"\"\"Introduction line.

    Line 2.

    \"\"\"

**DO NOT:**

.. code:: Python

    var: int = 1
    \"\"\"Introduction line.

    Line 2.\"\"\"

.. code:: Python

    var: int = 1
    \"\"\"Introduction line.

    Line 2.
    \"\"\"

"""


example_var_3: int = 42
"""int: **DO NOT** start the docstring with ``int:``, i.e. with type hints.

This will render the below ``Type: int``, which is unnecessary due to the already present
annotation after the variable name. As you can test above, clicking on *int* will forward you
to the right documentation.

**DO:**

.. code:: Python

    example_var_3: int = 1
    \"\"\"This is var 3.\"\"\"

**DO NOT:**

.. code:: Python

    example_var_3 = 1
    \"\"\"int: This is var 3.\"\"\"

"""


ExampleArrayLike = Union[list, np.ndarray]
"""This is a custom type alias, including lists and numpy's ``ndarray``."""


def example_function_1(arg1: int, arg2: str, arg3: bool) -> bool:
    """A short description of the functionality must fit in the first line.

    Detailed description can be given next if necessary. The extended description
    can of course span several lines.

    When natural, the description can be broken into several paragraphs that start
    without indentation.

    The description of arguments is started using the Google Style directive *Parameters:*.

    Note:
        You can find the ``[source]`` link in the upper right corner of this function's
        signature and inspect the Python and rst source code of this docstring,
        as well as for every docstring that follows.

    The return value must be documented using the Google Style directive *Returns:*.

    Parameters:
        arg1: Description of arg1.
        arg2: Description of arg2. If the description of a parameter spans more than one line,
            the lines must be indented.

            You can even use multiple paragraphs for describing a parameter. There
            are no restrictions in this regard.
        arg3 (bool): **DO NOT** include type hints in the *Parameters* directive.
            Utilize annotations in the signature instead.

    Returns:
        Describe the return value here. Multiple lines can be used and you have
        to indent the lines.

        **DO NOT** use type hints here as they are present in the signature.

    """
    # In addition to docstrings, you should also add comments to the code. These should
    # be sufficiently detailed that it is possible to understand the logic of the code
    # without analyzing the code in detail.
    return (arg1 > len(arg2)) and arg3


def example_function_2(
    optional_arg: Optional[int] = None, *args, **kwargs
) -> None:
    """This function demonstrates linking to objects inside docstrings, optional arguments and
    the *Raises:* directive.
    
    Optional arguments must be type-hinted using :obj:`typing.Optional`.

    As evident, you can link objects in-text. You can also link functions like
    :meth:`example_function_1`. Note the different usage of ``:obj:`` and ``:class:`` for
    objects in third-party packages and objects in PorePy.

    If ``*args`` or ``**kwargs`` are accepted,
    they must be listed as ``*args`` and ``**kwargs`` in the *Parameters* directive.

    Note:
        This function returns ``None`` as per signature.
        It is the default return value of Python and as such does not need documentation.
        I.e., we can omit the *Returns:* directive. However, it is obligatory to explicitly
        specify the None return in the signature.

    Parameters:
        optional_arg (optional): Description of ``optional_arg``.
            The default value is described at the end.
            Use easily spottable wording like 'Defaults to None.' and mark the argument using
            ``(optional)`` after the name.
        *args: Arbitrary arguments. Describe adequately how they are used.
        **kwargs: Arbitrary keyword arguments. You can use rst-lists to describe admissible
            keywords:

            - ``kw_arg1``: Some admissible keyword argument
            - ``kw_arg2``: Some other admissible keyword argument

            Note that Sphinx can only interpret the rst markdown language.

    Raises:
        ValueError: You can use the the Google Style directive *Raises:* to describe error
            sources you have programmed. Here for example we can raise a value error
            if an unexpected keyword argument was passed.
        TypeError: You can hint multiple errors. Be aware of necessary indentations when
            descriptions span multiple lines.

    """
    pass


def example_function_3(
    mdg: pp.MixedDimensionalGrid,
    vector: np.ndarray,
    matrix: sps.spmatrix,
    option: Literal["A", "B", "C"] = "A",
) -> sps.spmatrix:
    """This function demonstrates how to document special requirements for input arguments.

    Parameters:
        mdg: A mixed-dimensional grid. Notice the usage of the acronym ``mdg``.
        vector (len=num_cells): This vector's special requirement is a certain length.
            We use the type hint brackets to indicate its supposed length.
            The required length is ``num_cells``, the number of cells. If unclear to what the
            number refers, explain.
            E.g. where ``num_cells`` denotes the number of cells in ``mdg``.
        matrix (shape=(3, num_cells)): This argument is a sparse matrix with 3 rows and
            ``num_cells`` columns.
        option (optional): This argument has admissible values and a default value.
            it can only be a string ``'A'``, ``'B'`` or ``'C'``. For such admissible values use
            :obj:`typing.Literal` to denote them in the signature.
            Since it has a default value, we mark it with ``(optional)``.

    Returns:
        Describe the return value here. You can also describe its dependence on the value of
        ``option`` here, or alternatively in the extended description at the beginning of the
        docstring.

        Exemplarily we can write here:

        - If ``option`` is ``'A'``, return ...
        - If ``option`` is ``'B'``, return ...
        - ...

    Raises:
        ValueError: If ``option`` is not in ``{'A', 'B', 'C'}``.

    """
    pass


def example_function_4(
    vector: np.ndarray,
    matrix: sps.spmatrix,
) -> tuple[sps.spmatrix, np.ndarray]:
    """This function demonstrates how to document multiple return values in the *Returns:*
    directive.

    Parameters:
        vector: some vector
        matrix: some matrix

    Returns:
        Every returned object needs a description. We can use the same structure as for
        the parameters: indented block per returned object or value.

        spmatrix ``(shape=(3, nd))``:
            Though technically the tuple is a single object, it often is of
            importance what the tuple contains. If the returned object has noteworthy
            characteristic, explain them next to the bullet header, the same way as for
            parameters.
        ndarray:
            Use the definition list structure with indented blocks to
            describe the content individually. This makes of course only sense if the tuple
            is of fixed length and/or contains objects of various types.

    """
    pass


def example_function_5(
    array_like: ArrayLike,
    array: np.ndarray,
    # ndarray_like: NDArray,
    # dtype_like: DTypeLike,
) -> ExampleArrayLike:
    """This function demonstrates type annotations using custom type aliases, including those
    from third-party packages.

    For Sphinx to display type aliases without parsing their complete expression,
    a configuration is needed which maps the type alias to its source.

    When encountering type aliases, which are not already covered by the configuration of
    PorePy's docs, please contact the core developers.

    Parameters:
        array_like: an argument of type :obj:`~numpy.typing.ArrayLike`
        ndarray_like: an argument of type :obj:`~numpy.typing.NDArray`
        dtype_like: an argument of type :obj:`~numpy.typing.DTypeLike`

    Returns:
        a value of a custom array type :data:`ExampleArrayLike`.

    """
    pass


def example_generator(n: int) -> Generator[int, None, None]:
    """When documenting a generator, use the *Yields:* directive instead of *Returns:*.
    Their usage is equivalent, indicating only that this is a generator object, not a regular
    function.

    Examples:
        The Google Style offers an *Examples:* directives where you can write text but also
        include inline examples of code usage, without the special code directive from Sphinx.
        An indented block with the content has to follow the directive. Like all for all
        directives, a blank line after the indented block signalizes the end of the directive.

        >>> print([i for i in example_generator(4)])
        [0, 1, 2, 3]

    Parameters:
        n: Some upper integer limit.

    Yields:
        numbers until ``n`` is reached.

    """
    for i in range(n):
        yield i


class ExampleClass:
    """A quick purpose of the class is to be provided in the first line.

    After the first line, a broader, multi-line description of the class can follow. It can
    contain multiple paragraphs as well as elements like code snippets.

    *Example:* directives are encouraged to demonstrate the usage of a class.

    *Note:* directives can be used the same way as for functions.

    Example:
        Using the *Example:* directive, you can provide simplified python syntax using ``>>>``
        on how to instantiate the class for example:

        >>> import example_docstrings as ed
        >>> A = ed.ExampleClass()
        >>> print(type(A))
        <class 'example_docstrings.ExampleClass'>

    Use the *References:* and *See Also:* directives to provide links to sources or
    inspirations for your implementation.

    For readability reasons, we impose the same rules on the order of directives as for
    functions. Meaning references/ see also, followed by parameters, return/yield and raises
    must be placed at the end with no additional text in between.

    References:
        1. `Basic Google Style 
           <https://www.sphinx-doc.org/en/master/usage/extensions/
           example_google.html#example-google>`_.

    See Also:
        `Same, but in different directive 
        <https://www.sphinx-doc.org/en/master/usage/extensions/
        example_google.html#example-google>`_.

    Parameters:
        arg1: first argument.
        arg2: second argument.

    """

    class_attribute_1: int = 1
    """This is how to properly document a class-level attribute.

    The value 1 is shown in the docs, because it is static upon class type creation and Sphinx
    realizes that.

    """

    class_attribute_2: str = "A"
    """str: This is a class attribute with a redundant field 'Type:' caused by type hints
    in the docstring. The type should only be given by its annotation.
    **DO NOT** use type hints in docstrings.

    """

    _private_class_attribute: float = np.pi
    """Document private class attributes as well.

    Like everything private, they are not included by default in the docs.
    Mention their inclusion into the docs upon integration, if necessary.

    """

    def __init__(self, arg1: int, arg2: str) -> None:
        # !!! no docstring for __init__ !!!

        self.attribute_1: int = arg1
        """This is how to properly document an instance-level attribute in the constructor."""

        self.attribute_2: str = arg2
        """str: This is an instance-level attribute with a redundant field 'Type:' caused by
        type hints in the docstring. The type should only be given by its annotation.
        **DO NOT** use type hints in docstrings (cannot repeat often enough).

        """

        self.attribute_3: dict[str, int] = self._calculate_attribute_3()
        """Some attributes are calculated during construction, possibly in a separate method.

        Document nevertheless what the attribute contains and what it is used for etc.,
        but leave the documentation of how it is calculated to the separate method.

        """

        self._private_attribute: str = "This is a private attribute."
        """This is a private (instance) attribute and is not included by default.

        Document them properly anyway.

        """

    @property
    def readonly_property(self) -> str:
        """Document properties only in their getter-methods."""
        return "readonly"

    @property
    def readwrite_property(self) -> str:
        """Properties with both a getter and setter should only be documented in their
        getter method.

        If the setter method contains notable behavior, it should be
        mentioned here in the getter documentation.

        """
        return "readwrite"

    @readwrite_property.setter
    def readwrite_property(self, value):
        value

    @staticmethod
    def example_static_method() -> None:
        """This is the docstring of a static method.

        Note the absence of ``self``.

        """
        pass

    @classmethod
    def example_class_method(cls) -> None:
        """This is the docstring of a class method.

        Note that the ``self`` argument is replaced by the ``cls`` argument.

        """
        pass

    def example_method(self, arg: bool) -> bool:
        """A short description of the method must fit in the first line.

        A more elaborate description can follow afterwards.

        Parameters:
            arg: The first argument (even though ``self`` is technically the first one).

        Returns:
            True.

        """
        return True

    def _calculate_attribute_3(self) -> dict[str, int]:
        """The private attribute is calculated in this private method.

        Document here what is done, like for any other method, including parameters and return
        values.

        Private methods are not included by default in the compiled documentation.
        If necessary, talk to the core developers upon integration of docs.

        """
        pass

    def __special__(self) -> None:
        """Special methods (starting and ending with two underscores) are by default not
        included.

        Their inclusion needs to be assured by an *option* in the rst files.
        Talk to the core developers upon integration of docs.

        Nevertheless document them properly, including parameters and return values.

        Note:
            The __init__ method *is* a special method, but for which above mentioned special
            rules apply, i.e. no documentation in the __init__ method.

        """
        pass

    def _private(self) -> None:
        """Another private method, by default not included.

        They should be nevertheless documented properly, including parameters, returns values
        etc.

        """
        pass
