from abc import ABC, abstractmethod
from copy import copy
from enum import StrEnum
from typing import Callable, cast
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class NonlinearSolverStatus:
    pass


@dataclass
class NonlinearSolverStatusSuccess(NonlinearSolverStatus):
    pass


@dataclass
class NonlinearSolverStatusFailure(NonlinearSolverStatus):
    msg: str
