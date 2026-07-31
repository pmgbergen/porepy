from abc import ABC, abstractmethod

__all__ = [
    "TimeConstraint",
    "TimeInterval",
    "TimeScheduler",
]


class TimeConstraint(ABC):
    @abstractmethod
    def apply_constraint(self, dt: float) -> float:
        pass


class TimeInterval:
    def __init__(
        self, interval: tuple[float, float], constraints: list[TimeConstraint]
    ) -> None:
        self.interval = interval
        self.constraints = constraints


class TimeScheduler:
    def __init__(self, intervals: list[TimeInterval]) -> None:
        self.intervals = intervals
