""" 
Official PorePy docstring style guide.

This guide is written as a complement to the Google style guide [1, 2] 

[1] https://google.github.io/styleguide/pyguide.html 

[2] https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google 

Example Google style-based PorePy docstrings.
This module demonstrates documentation as specified by the `Google Python Style Guide` but
tailored to PorePy's specific needs.
In addition to the Google style guide rules, we further require the description of
class arguments and functions/methods to follow specific standards.  
The current document aims to be as self-contained as possible.
However, some indications/hints might be given as side comments for now.
In a docstring, sections are created with a header and a colon followed by a block of
indented text. Consider the Example section shown below. 

 
Example: 
    Examples can be given using either the ``Example`` or ``Examples`` 
    sections. Example sections are particularly useful to show how to instantiate 
    specific classes or how to use certain methods.  
 
Section breaks are created by resuming unintended text. Section breaks 
are also implicitly created anytime a new section starts. 

Todo: 
    * Here you can add the TODOs of the module. 
    * You can have as many TODOs as you want. 
""" 

import porepy as pp 
import numpy as np 
import scipy.sparse as sps 

from typing import Optional 


module_level_var1: str = "var" 
"""str: Variables at the module level are documented inline after declaration."""


module_level_var2: int = 216879 
"""int: Multiline docstrings are also valid. 

You might want to use multiline docstrings when a more detailed description
of the variable is required.
In both cases, the type must be specified at the beginning of the docstring followed by a
colon. 
""" 

def module_level_function(param1: int, param2: str) -> bool: 
    """General description of the module function must fit in one line. 
    Detailed description can be given next if necessary. Of course, this description 
    can span several lines.     

    When natural, the description can be break into several paragraphs that start 
    without indentation. 

    Args: 
        param1 (int): Description of param1.  
        param2 (str): Description of param2. If the description of a parameter spans 
            more than one line, the lines must be indented.  

            You can even use multiple paragraphs for describing a parameter. There 
            are no restrictions in this regard. 

    Returns: 
        bool: True if param1 is bigger than the length of param2, False otherwise. 
    """   
    return param1 > len(param2) 

 
def module_level_function2(
    param1: int, 
    param2: Optional[int]=None, 
    *args, 
    **kwargs, 
    ): 
    """General description of the module function must fit in one line. 

    If ``*args`` or ``**kwargs`` are accepted, 
    they should be listed as ``*args`` and ``**kwargs``. 

    Args: 
        param1 (int): Description of ``param1``.  
        param2 (int, default=None): Description of ``param2``. Default value for the 
            parameter is given in parentheses next to the type. 
        *args: Variable length argument list. 
        **kwargs: Arbitrary keyword arguments. 

    Raises: 
        ValueError: If ``param1`` is     
    """   
    if param1 == param2: 
        raise ValueError("param1 cannot be equal to param2.") 


def module_level_function(param1, param2=None, *args, **kwargs): 
    """This is an example of a module level function. 
 
    Function parameters should be documented in the ``Args`` section. The name 
    of each parameter is required. The type and description of each parameter 
    is optional but should be included if not obvious. 
 
    If ``*args`` or ``**kwargs`` are accepted, 
    they should be listed as ``*args`` and ``**kwargs``. 
 
    The format for a parameter is: 
 
        name (type): description 
            The description may span multiple lines. Following 
            lines should be indented. The "(type)" is optional. 
 
            Multiple paragraphs are supported in parameter 
            descriptions. 
 
    Args: 
        param1 (int): The first parameter. 
        param2 (:obj:`str`, optional): The second parameter. Defaults to None. 
            The second line of description should be indented. 
        *args: Variable length argument list. 
        **kwargs: Arbitrary keyword arguments. 
 
    Returns: 
        bool: True if successful, False otherwise. 
 
        The return type is optional and may be specified at the beginning of 
        the ``Returns`` section followed by a colon. 
 
        The ``Returns`` section may span multiple lines and paragraphs. 
        The following lines should be indented to match the first line. 
 
        The ``Returns`` section supports any reStructuredText formatting, 
        including literal blocks: 
 
            { 
                'param1': param1, 
                'param2': param2 
            } 
 
    Raises: 
        AttributeError: The ``Raises`` section is a list of all exceptions 
            that are relevant to the interface. 
        ValueError: If ``param2`` is equal to ``param1``. 
 
    """ 
    if param1 == param2: 
        raise ValueError('param1 may not be equal to param2') 
    return True 
 
 
def example_generator(n): 
    """Generators have a ``Yields`` section instead of a ``Returns`` section. 
 
    Args: 
        n (int): The upper limit of the range to generate, from ``0`` to ``n-1``. 
 
    Yields: 
        int: The next number in the range of 0 to ``0`` to ``n-1``. 
 
    Examples: 
        Examples should be written in doctest format, and should illustrate how 
        to use the function. 
 
        >>> print([i for i in example_generator(4)]) 
        [0, 1, 2, 3] 
 
    """ 
    for i in range(n): 
        yield i 
 
 
class ExampleError(Exception): 
    """Exceptions are documented in the same way as classes. 
 
    The __init__ method may be documented in either the class level 
    docstring, or as a docstring on the __init__ method itself. 
 
    Either form is acceptable, but the two should not be mixed. Choose one 
    convention to document the __init__ method and be consistent with it. 
 
    Note: 
        Do not include the ``self`` parameter in the ``Args`` section. 
 
    Args: 
        msg (str): Human readable string describing the exception. 
        code (:obj:`int`, optional): Error code. 
 
    Attributes: 
        msg (str): Human readable string describing the exception. 
        code (int): Exception error code. 
 
    """ 
 
    def __init__(self, msg, code): 
        self.msg = msg 
        self.code = code 
 
 
class ExampleClass: 
    """The summary line for a class docstring should fit on one line. 
 
    If the class has public attributes, they may be documented here 
    in an ``Attributes`` section and follow the same formatting as a 
    function's ``Args`` section. Alternatively, attributes may be documented 
    inline with the attribute's declaration (see __init__ method below). 
 
    Properties created with the ``@property`` decorator should be documented 
    in the property's getter method. 
 
    Attributes: 
        attr1 (str): Description of `attr1`. 
        attr2 (:obj:`int`, optional): Description of `attr2`. 
 
    """ 
 
    def __init__(self, param1, param2, param3): 
        """Example of docstring on the __init__ method. 
 
        The __init__ method may be documented in either the class level 
        docstring, or as a docstring on the __init__ method itself. 
 
        Either form is acceptable, but the two should not be mixed. Choose one 
        convention to document the __init__ method and be consistent with it. 
 
        Note: 
            Do not include the `self` parameter in the ``Args`` section. 
 
        Args: 
            param1 (str): Description of `param1`. 
            param2 (:obj:`int`, optional): Description of `param2`. Multiple 
                lines are supported. 
            param3 (list(str)): Description of `param3`. 
 
        """ 
        self.attr1 = param1 
        self.attr2 = param2 
        self.attr3 = param3  #: Doc comment *inline* with attribute 
 
        #: list(str): Doc comment *before* attribute, with type specified 
        self.attr4 = ['attr4'] 
 
        self.attr5 = None 
        """str: Docstring *after* attribute, with type specified.""" 
 
    @property 
    def readonly_property(self): 
        """str: Properties should be documented in their getter method.""" 
        return 'readonly_property' 
 
    @property 
    def readwrite_property(self): 
        """list(str): Properties with both a getter and setter 
        should only be documented in their getter method. 
 
        If the setter method contains notable behavior, it should be 
        mentioned here. 
        """ 
        return ['readwrite_property'] 
 
    @readwrite_property.setter 
    def readwrite_property(self, value): 
        value 
 
    def example_method(self, param1, param2): 
        """Class methods are similar to regular functions. 
 
        Note: 
            Do not include the ``self`` parameter in the ``Args`` section. 
 
        Args: 
            param1: The first parameter. 
            param2: The second parameter. 
 
        Returns: 
            True if successful, False otherwise. 
 
        """ 
        return True 

 
    def example_method_with_typings(self,  

        param1: np.ndarray,  

        param2: bool = False 

        ) -> bool :
        """Class methods are similar to regular functions. 
        
        Note: 
            Do not include the `self` parameter in the ``Args`` section. 

        Args: 
            param1 (np.ndarray, (num_cells, )): Description of parameter1. 
            param2 (bool, optional): Description of parameter2. Default is False. 

        Returns: 
            True if successful, False otherwise. 

        """ 
        return True 

    def __special__(self): 
        """By default special members with docstrings are not included. 
 
        Special members are any methods or attributes that start with and 
        end with a double underscore. Any special member with a docstring 
        will be included in the output, if 
        ``napoleon_include_special_with_doc`` is set to True. 
 
        This behavior can be enabled by changing the following setting in 
        Sphinx's conf.py:: 
 
            napoleon_include_special_with_doc = True 
 
        """ 
        pass 
 
    def __special_without_docstring__(self): 
        pass 
 
    def _private(self): 
        """By default private members are not included. 
 
        Private members are any methods or attributes that start with an 
        underscore and are *not* special. By default they are not included 
        in the output. 
 
        This behavior can be changed such that private members *are* included 
        by changing the following setting in Sphinx's conf.py:: 
 
            napoleon_include_private_with_doc = True 
 
        """ 
        pass 
 
    def _private_without_docstring(self): 
        pass 
 
class ExamplePEP526Class: 
    """The summary line for a class docstring should fit on one line. 
 
    If the class has public attributes, they may be documented here 
    in an ``Attributes`` section and follow the same formatting as a 
    function's ``Args`` section. If ``napoleon_attr_annotations`` 
    is True, types can be specified in the class body using ``PEP 526`` 
    annotations. 
 
    Attributes: 
        attr1: Description of `attr1`. 
        attr2: Description of `attr2`. 
 
    """ 
 
    attr1: str 
    attr2: int 