# Solver and mixed-dimensional interfaces

In this part the basic classes to be inherited by the solvers and the couplers, and the main mixed-dimensional coupler.

We have:
* abstract interface for numerical solvers [solver.py](solver.py)
* abstract interface for numerical couplers between solvers for the mixed-dimensional [abstract_coupling.py](abstract_coupling.py)
* coupler for the mixed-dimensional approach [coupler.py](coupler.py)
* static condensation or Schur complement proceedure [condensation.py](condensation.py)
