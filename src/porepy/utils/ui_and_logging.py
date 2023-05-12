"""This module contains functionality to make the progress bars work correctly with
logging."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Optional, Type
from tqdm.contrib.logging import logging_redirect_tqdm

from tqdm.autonotebook import tqdm as std_tqdm
from tqdm.contrib.logging import (
    _get_first_found_console_logging_handler,
    _TqdmLoggingHandler,
)


@contextmanager
def logging_redirect_tqdm_with_level(
    loggers: Optional[list[logging.Logger]] = None,
    tqdm_class: Type[std_tqdm] = std_tqdm,
) -> Iterator[None]:
    """Extend capability of ``logging_redirect_tqdm`` s.t. the logging handler level
    gets passed."""
    if loggers is None:
        loggers = [logging.root]
    # Save loggers and a corresponding handler for each
    original_handlers_dict = {
        logger: _get_first_found_console_logging_handler(logger.handlers)
        for logger in loggers
    }
    # Let ``logging_redirect_tqdm`` do its work, and change the level of the
    # handlers afterwards.
    with logging_redirect_tqdm(loggers, tqdm_class):
        try:
            for logger, orig_handler in original_handlers_dict.items():
                for handler in logger.handlers:
                    if isinstance(handler, _TqdmLoggingHandler):
                        handler.level = orig_handler.level
            yield
        finally:
            pass
