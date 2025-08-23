"""This module contains functionality for logging and tqdm progressbar_classs."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator


class DummyProgressBar:
    """Dummy class to replace :class:~`tqdm.trange` when it is not installed or used.

    All methods of :class:`~tqdm.trange` that may be called in
    :mod:`~porepy.numerics.nonlinearnonlinear_solvers`
    and :mod:`~porepy.models.run_models` are replaced with empty methods.

    """

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def write(cls, *args, **kwargs):
        # _TqdmLoggingHandler.emit calls this method to write messages. However, this
        # functionality does not appear to be needed in the dummy class. Logging works
        # fine with an empty dummy class.
        pass

    def update(self, *args, **kwargs):
        pass

    def set_description_str(self, *args, **kwargs):
        pass

    def set_postfix_str(self, *args, **kwargs):
        pass

    def close(self):
        pass


# ``tqdm`` is not a dependency. Up to the user to install it.
try:
    # Avoid some mypy trouble due to missing library stubs for tqdm.
    from tqdm.autonotebook import tqdm as progressbar_class  # type: ignore
    from tqdm.contrib.logging import (  # type: ignore
        _get_first_found_console_logging_handler,
        _TqdmLoggingHandler,
        logging_redirect_tqdm,
    )
except ImportError:
    progressbar_class = DummyProgressBar


@contextmanager
def logging_redirect_tqdm_with_level(
    loggers: list[logging.Logger] | None = None,
    tqdm_class: type = progressbar_class,
) -> Iterator[None]:
    """Extend capability of ``tqdm.contrib.logging_redirect_tqdm`` s.t. the logging
    handler level gets passed.

    Parameters:
        loggers: List of loggers to redirect. If not provided, the root logger is used.
        tqdm_class: The class to use for the progress bar. Defaults to
            :class:~`tqdm.autonotebook.tqdm` or :class:~`DummyProgressBar` if ``tqdm``
            is not installed.

    Returns:
        An iterator that redirects logging to the tqdm progress bar. The logging level
        of the handlers is set to the level of the original logger or the handler if it
        exists.

    """
    # ``logging_redirect_tqdm`` provides functionality to make ``tqdm`` work together
    # with loggers. It takes a list of ``logging.logger`` as inputs and adds additional
    # handlers that work with ``tqdm``. However, currently (as of 10.06.2025) these
    # handlers do not account for the logging level of the passed logger. This issue is
    # known (see e.g., https://github.com/tqdm/tqdm/issues/1272) but not resolved yet.
    # Development of ``tqdm`` is not very active, so it is unclear if and when this will
    # be fixed. At some point in the future, it might be advisable to switch to a
    # different progressbar_class library with more active development.
    #
    # This function provides a workaround until the issue is fixed.
    #
    # NOTE: The functionality to differentiate between the ``tqdm`` class and the dummy
    # needs to remain, even if the issue is fixed. It serves as a check to see if, e.g.,
    # :meth:`_get_first_found_console_logging_handler` was imported and can be called.

    # If the dummy class is used, we do not need to redirect logging.
    if tqdm_class is DummyProgressBar:
        yield

    else:
        if loggers is None:
            loggers = [logging.root]
        # Save loggers and a corresponding handler for each logger.
        original_handlers_dict: dict[logging.Logger, logging.Handler] = {
            logger: _get_first_found_console_logging_handler(logger.handlers)
            for logger in loggers
        }
        # Let ``logging_redirect_tqdm`` do its work and change the level of the handlers
        # afterwards.
        with logging_redirect_tqdm(loggers, tqdm_class):
            try:
                for logger, orig_handler in original_handlers_dict.items():
                    for handler in logger.handlers:
                        if isinstance(handler, _TqdmLoggingHandler):
                            # If the original logger has a handler, copy its level.
                            if orig_handler is not None:
                                handler.level = orig_handler.level
                            # Otherwise, copy the level of the logger.
                            else:
                                handler.level = logger.level

                yield
            finally:
                pass
