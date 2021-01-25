""" Logging functionality for PorePy.

Logging is controlled by the configuration file porepy.cfg, which should be
placed in the current working directory (where the python script is initiated).
All loging-related information is located in a section in the cfg-file with
heading logging; see sample file below.

By default, logging is switched off. It can be turned on by setting the keyword
'active' to True.

Logging can be time consuming if applied to functions called many times: Estimates
indicate that the writting of the log message for a single function call
takes on the order 1e-5 seconds, but when applied to a function called 1000s of times,
the cost becomes relevant. To log only parts of the code, all functions are classified
as relevant for the following (overlapping) categories

    geometry: Computations related to geometry
    gridding: Everything that has to do with mesh construction
    parameters: Define parameters
    models: Functions related to multi-physics models
    numerics: Discretization-related
    utils: All (minor) utility-related functions


Example logging section of porepy.cfg:

    [logging]
    # Activate logging. Without this, the rest of the section has no effect
    active: True
    # To log all functions in PorePy, there is no need for more information.

    # To only log specific sections, use e.g.
    sections: gridding
    # multiple sections are separated by commas:
    sections: models, discretization

"""
import functools
import logging, logging.handlers
import time
import inspect
import configparser

import porepy as pp
import numpy as np
import scipy.sparse as sps

__all__ = ["time_logger"]


try:
    config = dict(pp.config["logging"])
    raw_sections = config.get("sections", "all")
    active_sections = ["all"] + [s.strip() for s in raw_sections.split(",")]
except KeyError:
    config = {}
    active_sections = ["all"]

t_logger = logging.getLogger("Timer")
t_logger.setLevel(logging.INFO)


if not t_logger.hasHandlers():
    time_handler = logging.FileHandler("PorePyTimings.log")
    time_handler.setLevel(logging.INFO)
    time_formatter = logging.Formatter("%(message)s")
    time_handler.setFormatter(time_formatter)
    t_logger.addHandler(time_handler)


path_length = __file__.split("/").index("porepy")


# @pp.time_logger
def time_logger(func, sections, active=config.get("active", False)):
    """A decorator that measures ellapsed time for a function."""

    @functools.wraps(func)
    def log_time(*args, **kwargs):
        if not active:
            return func(*args, **kwargs)
        elif not any([s in active_sections for s in sections]):
            return func(*args, **kwargs)

        fn = "/".join(inspect.getfile(func).split("/")[path_length + 1 :])

        name = f"{func.__name__} in file {fn}."

        t_logger.log(level=logging.INFO, msg=f"Calling {name}")

        #        log.l += name + "\n"

        start_time = time.perf_counter()
        value = func(*args, **kwargs)

        end_time = time.perf_counter()
        run_time = end_time - start_time

        t_logger.log(
            level=logging.INFO,
            msg=f"Finished {name} Elapsed time: {run_time:.8f} s",
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
    @pp.time_logger
    def analyze_args(*args, **kwargs):
        msg = f"Calling function {func.__name__}\n"
        for a in args:
            if isinstance(a, np.ndarray):
                s = f"\t numpy array of shape {a.shape} and type {a.dtype}"
            elif isinstance(a, sps.spmatrix):
                s = f"\t sparse matrix of shape {a.shape} with {a.data.size} nonzeros"
