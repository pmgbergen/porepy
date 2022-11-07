=========================
Automatic Differentiation
=========================

.. currentmodule:: porepy.ad

The AD framework (*Algorithmic* or *Automatic* Differentiation) allows an object-oriented
representation of variables and other mathematical terms, which appear in equations
and models. The goal as for any AD framework is to provide a cheap and and automated evaluation
of derivatives, to be utilized in numerical approximation techniques.

All classes and functions can be accessed as shown the following example:

.. code-block:: python
    :caption: Access to AD submodule

    import porepy as pp

    a = pp.ad.Scalar(1.)

PorePy uses the *Forward Accumulation* technique. The numerical value of a nested expression
and its derivative are evaluated recursively and inside out using the chain rule.
The order of evaluations of subterms is given by Python's
`operator precedence <https://docs.python.org/3.6/reference/expressions.html#operator-precedence>`_.

Central to the AD framework are the classes :class:`Ad_array` and :class:`Operator`.
They provide access to numeric values and derivatives of basic arithmetic operations, and a
tree-like data structure for the recursive evaluation respectively.
It also provides a wrapper for discretizations of operators like the mathematical divergence,
enabling the user to program equations in a symbolic way.

.. note::
    The expression *operator* might cause confusion. 
    In the sense of ``porepy.ad``, every object is an operator returning a numerical
    representation of the represented term, based on some information. What information is
    necessary and what is provided is documented in individual classes.

PorePy AD provides objects for

    - `Elemental AD data structures`_
    - `Common mathematical functions`_
    - `AD-representation of constant scalars, vectors and matrices`_
    - `Variables on single and mixed-dimensional domains`_
    - `Operator functions`_
    - `Grid operators`_
    - `Numerical discretization schemes`_
    - `AD Equation Manager`_

Elemental AD data structures
============================

.. autoclass:: Ad_array
    :members:
    :undoc-members:

.. autoclass:: Operator
    :members:
    :undoc-members:

Common mathematical functions
=============================

.. automodule:: porepy.numerics.ad.functions
    :members:
    :undoc-members:

AD-representation of constant scalars, vectors and matrices
===========================================================
.. currentmodule:: porepy.ad

.. autoclass:: Scalar
    :members:
    :undoc-members:

.. autoclass:: Array
    :members:
    :undoc-members:

.. autoclass:: TimeDependentArray
    :members:
    :undoc-members:

.. autoclass:: Matrix
    :members:
    :undoc-members:

Variables on single and mixed-dimensional domains
=================================================

.. autoclass:: Variable
    :members:
    :undoc-members:

.. autoclass:: MergedVariable
    :members:
    :undoc-members:

Operator functions
==================
.. currentmodule:: porepy.numerics.ad.operator_functions

.. autoclass:: AbstractFunction
    :members:

.. autoclass:: AbstractJacobianFunction
    :members:

.. automodule:: porepy.numerics.ad.operator_functions
    :members:

Grid operators
==============

todo

Numerical discretization schemes
================================

todo

AD Equation Manager
===================

todo