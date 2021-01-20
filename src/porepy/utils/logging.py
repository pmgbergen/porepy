""" Logging functionality for PorePy.
"""
import functools
import logging
import time

import numpy as np
import scipy.sparse as sps

__all__ = ["time_logger"]

t_logger = logging.getLogger("Timer")
t_logger.setLevel(logging.INFO)


if not t_logger.hasHandlers():
    time_handler = logging.FileHandler("PorePyTimings.log")
    time_handler.setLevel(logging.INFO)
    time_formatter = logging.Formatter("%(message)s")
    time_handler.setFormatter(time_formatter)
    t_logger.addHandler(time_handler)


# @pp.time_logger
def time_logger(func):
    """A decorator that measures ellapsed time for a function."""

    @functools.wraps(func)
    def log_time(*args, **kwargs):
        t_logger.log(level=logging.INFO, msg=f"Calling {func.__name__}")
        start_time = time.perf_counter()
        value = func(*args, **kwargs)

        end_time = time.perf_counter()
        run_time = end_time - start_time

        t_logger.log(
            level=logging.INFO,
            msg=f"Finished {func.__name__}. Elapsed time: {run_time:.4f} s",
        )

        return value

    return log_time


trace_logger = logging.getLogger("Trace")
trace_logger.setLevel(logging.INFO)


if not trace_logger.hasHandlers():
    trace_handler = logging.FileHandler("PorePyTraces.log")
    trace_handler.setLevel(logging.INFO)
    trace_formatter = logging.Formatter("%(message)s")
    trace_handler.setFormatter(trace_formatter)
    trace_logger.addHandler(trace_handler)


def trace(func):
    @functools.wraps(func)
    def analyze_args(*args, **kwargs):
        msg = f"Calling function {func.__name__}\n"
        for a in args:
            if isinstance(a, np.ndarray):
                s = f"\t numpy array of shape {a.shape} and type {a.dtype}"
            elif isinstance(a, sps.spmatrix):
                s = f"\t sparse matrix of shape {a.shape} with {a.data.size} nonzeros"
