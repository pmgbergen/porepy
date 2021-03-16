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

    all: Used to log all methods.
    assembly: Assembly of discretization matrices.
    discretization: Specific to discretization (less general than numerics)
    geometry: Computations related to geometry
    gridding: Everything that has to do with mesh construction
    grids: Manipulations of grids beyond mesh construction
    parameters: Define parameters
    matrix: Related to matrix algebra.
    models: Functions related to multi-physics models
    numerics: Discretization-related
    utils: All (minor) utility-related functions
    visualization: Related to visualization

To see which methods are logged by the different logging keywords, one must access the
implementation of individual methods. The logging keywords are commonly set on the
module level. Note that there are borderline items relative to the categorization, in
particular for the narrower categories.

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
import inspect
import logging
import logging.handlers
import os
import time
from typing import Dict

import porepy as pp

__all__ = ["time_logger"]


# Try to access configuration information, as activated by the import of PorePy
try:
    config: Dict = pp.config["logging"]  # type: ignore
    raw_sections = config.get("sections", "all")
    active_sections = [s.strip().lower() for s in raw_sections.split(",")]
    logger_is_active = config["active"].strip().lower() == "true"
    always_log = "all" in active_sections

except KeyError:
    config = {}
    active_sections = ["all"]
    logger_is_active = False

t_logger = logging.getLogger("Timer")
t_logger.setLevel(logging.INFO)


if not t_logger.hasHandlers():
    # Add handler to write to file.
    # FIXME: Should it be possible to change the name of the log?
    time_handler = logging.FileHandler("PorePyTimings.log")
    time_handler.setLevel(logging.INFO)
    time_formatter = logging.Formatter("%(message)s")
    time_handler.setFormatter(time_formatter)
    t_logger.addHandler(time_handler)

# Find where in the file path the directory 'porepy' is located.
# We will use this below to strip away the common parts of file names.
# The separator (/ or \) depends on operating system.
separator = os.sep
path_length = __file__.split(separator).index("porepy")


# @pp.time_logger
def time_logger(sections):
    """A decorator that measures ellapsed time for a function."""

    # The double nested function is needed to allow decorators with
    # default arguments (it turned out)
    # Inspiration: https://realpython.com/primer-on-python-decorators/
    def inner_func(func):
        @functools.wraps(func)
        def log_time(*args, **kwargs):
            if not logger_is_active:
                # Shortcut if logging is not activated.
                # This is
                return func(*args, **kwargs)
            elif always_log or any([s in active_sections for s in sections]):

                # Get the name of the file, but strip away the part above
                # '/src/porepy'
                fn = separator.join(
                    inspect.getfile(func).split(separator)[path_length + 1 :]
                )

                # String representation of the file
                name = f"{func.__name__} in file {fn}."

                # Log the calling of the file
                t_logger.log(level=logging.INFO, msg=f"Calling {name}")

                start_time = time.perf_counter()
                # Run the function
                value = func(*args, **kwargs)

                # Timing
                end_time = time.perf_counter()
                run_time = end_time - start_time

                # Logging
                t_logger.log(
                    level=logging.INFO,
                    msg=f"Finished {name} Elapsed time: {run_time:.8f} s",
                )

                return value
            else:
                return func(*args, **kwargs)

        return log_time

    return inner_func


# trace_logger = logging.getLogger("Trace")
# trace_logger.setLevel(logging.INFO)


# if not trace_logger.hasHandlers():
#    trace_handler = logging.FileHandler("PorePyTraces.log")
#    trace_handler.setLevel(logging.INFO)
#    trace_formatter = logging.Formatter("%(message)s")
#    trace_handler.setFormatter(trace_formatter)
#    trace_logger.addHandler(trace_handler)


# def trace(func):
#    @functools.wraps(func)
#    def analyze_args(*args, **kwargs):
#        msg = f"Calling function {func.__name__}\n"
#        for a in args:
#            if isinstance(a, np.ndarray):
#                s = f"\t numpy array of shape {a.shape} and type {a.dtype}"
#            elif isinstance(a, sps.spmatrix):
#                s = f"\t sparse matrix of shape {a.shape} with {a.data.size} nonzeros"
