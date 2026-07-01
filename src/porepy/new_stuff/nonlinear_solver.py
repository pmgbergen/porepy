from abc import ABC, abstractmethod
from typing import Optional
from attr import dataclass
import numpy as np
from porepy.new_stuff.dof_manager import EquationVariableTag
from porepy.numerics.nonlinear.nonlinear_solver_status import NonlinearSolverStatus
from scipy.sparse import csr_matrix


class Indexer:
    pass


class PhysicsProvider(ABC):
    @abstractmethod
    def assemble_jacobian_residual(
        self, tags: list[EquationVariableTag]
    ) -> tuple[csr_matrix, np.ndarray]:
        pass

    @abstractmethod
    def assemble_residual(self, tags: list[EquationVariableTag]) -> tuple[np.ndarray]:
        pass

    @abstractmethod
    def get_solution_values(self, tags: list[EquationVariableTag]) -> tuple[np.ndarray]:
        pass

    @abstractmethod
    def set_solution_values(
        self, tags: list[EquationVariableTag], values: np.ndarray, additive: bool
    ) -> None:
        pass

    @abstractmethod
    def get_indexer(self, tags: list[EquationVariableTag]) -> Indexer:
        pass


class NonlinearSolverConvergenceCriterion(ABC):
    @abstractmethod
    def check_convergence(
        self,
        tags: list[EquationVariableTag],
        model: PhysicsProvider,
        iteration: int,
        increment: Optional[np.ndarray] = None,
    ) -> NonlinearSolverStatus:
        pass


class LinearSolver(ABC):
    @abstractmethod
    def solve_linear_system(
        self, J: csr_matrix, rhs: np.ndarray, indexer: Indexer
    ) -> np.ndarray:
        pass


class NonlinearSolverBase(ABC):
    def __init__(
        self,
        tags: list[EquationVariableTag],
        convergence_criterion: NonlinearSolverConvergenceCriterion,
    ) -> None:
        self.tags: list[EquationVariableTag] = tags
        self.convergence_criterion: NonlinearSolverConvergenceCriterion = (
            convergence_criterion
        )

    @abstractmethod
    def solve_nonlinear_system(self, model: PhysicsProvider) -> NonlinearSolverStatus:
        pass


class NewtonNonlinearSolver(NonlinearSolverBase):
    def __init__(
        self,
        tags: list[EquationVariableTag],
        convergence_criterion: NonlinearSolverConvergenceCriterion,
        linear_solver: LinearSolver,
    ) -> None:
        super().__init__(tags=tags, convergence_criterion=convergence_criterion)
        self.linear_solver: LinearSolver = linear_solver

    def solve_nonlinear_system(self, model: PhysicsProvider) -> NonlinearSolverStatus:
        MAX_ITER = 10
        for i in range(MAX_ITER):
            J, rhs = model.assemble_jacobian_residual(tags=self.tags)
            increment = self.linear_solver.solve_linear_system(
                J, rhs, indexer=model.get_indexer(tags=self.tags)
            )

            model.set_solution_values(tags=self.tags, values=increment, additive=True)
            # rediscretize happens inside set_solution_values or called here (?)

            solver_status = self.convergence_criterion.check_convergence(
                increment=increment, model=model, tags=self.tags, iteration=i
            )

            if solver_status.is_converged() or solver_status.is_failed():
                break
        else:
            return ...

        return solver_status


class CoupledMultiphysicsSolver(NonlinearSolverBase):
    def __init__(
        self,
        subsolvers: list[NonlinearSolverBase],
        convergence_criterion: NonlinearSolverConvergenceCriterion,
    ) -> None:
        self.subsolvers: list[NonlinearSolverBase] = subsolvers
        tags = flatten_lists([subsolver.tags for subsolver in subsolvers])
        assert len(tags) == len(set(tags)), "Must be unique (?)"
        super().__init__(tags=tags, convergence_criterion=convergence_criterion)

    def solve_nonlinear_system(self, model: PhysicsProvider) -> NonlinearSolverStatus:
        MAX_ITER = 10
        for i in range(MAX_ITER):
            for subsolver in self.subsolvers:
                subsolver_status = subsolver.solve_nonlinear_system(model)
                if subsolver_status.is_failed():
                    return ...

            solver_status = self.convergence_criterion.check_convergence(
                model=model, tags=self.tags, iteration=i
            )

            if solver_status.is_converged() or solver_status.is_failed():
                break
        else:
            return ...

        return solver_status


def flatten_lists(list_of_lists):
    return [x for sublist in list_of_lists for x in sublist]


### EXAMPLE


@dataclass
class MechanicsDisplacementTag(EquationVariableTag):
    pass


@dataclass
class FlowPressureTag(EquationVariableTag):
    pass


class SpecificConvergenceCriterion(NonlinearSolverConvergenceCriterion):
    pass


class SpecificLinearSolver(LinearSolver):
    pass


fixed_stress = CoupledMultiphysicsSolver(
    subsolvers=[
        NewtonNonlinearSolver(
            tags=[FlowPressureTag()],
            linear_solver=SpecificLinearSolver(),
            convergence_criterion=SpecificConvergenceCriterion(),
        ),
        NewtonNonlinearSolver(
            tags=[MechanicsDisplacementTag()],
            linear_solver=SpecificLinearSolver(),
            convergence_criterion=SpecificConvergenceCriterion(),
        ),
    ],
    convergence_criterion=SpecificConvergenceCriterion(),
)
