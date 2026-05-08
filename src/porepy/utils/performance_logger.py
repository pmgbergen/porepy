# src/porepy/utils/performance_logger.py

from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class PerformanceLogger:
    """
    Lightweight runtime logger for profiling simulations.

    The logger records timing information for coarse-grained phases like:

        - time step
        - nonlinear iteration
        - assembly
        - linear solve
        - export

    Data is stored internally and can later be written to CSV.

    Example
    -------
    >>> logger = PerformanceLogger(
    ...     "performance.csv",
    ...     run_info={"num_cells": 10000},
    ... )

    >>> with logger.timer("assembly", time_step=0, nl_iter=1):
    ...     assemble_system()

    >>> logger.write()
    """

    def __init__(
        self,
        filename: str | Path,
        run_info: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        self.filename = Path(filename)
        self.run_info = run_info or {}
        self.enabled = enabled

        self.rows: list[dict[str, Any]] = []

    @contextmanager
    def timer(self, phase: str, **info: Any):
        """
        Time a code block.

        Parameters
        ----------
        phase:
            Name of the phase being timed.
            Example:
                "assembly"
                "linear_solve"
                "time_step_total"

        **info:
            Additional metadata stored with the timing row.
        """

        if not self.enabled:
            yield
            return

        start = time.perf_counter()

        yield

        end = time.perf_counter()

        row = {
            **self.run_info,
            **info,
            "phase": phase,
            "wall_time_s": end - start,
        }

        self.rows.append(row)

    def log(self, phase: str, **info: Any) -> None:
        """
        Store a non-timing data row.

        Example
        -------
        >>> logger.log(
        ...     "time_step_summary",
        ...     time_step=4,
        ...     dt=1e-3,
        ...     nonlinear_iterations=6,
        ... )
        """

        if not self.enabled:
            return

        row = {
            **self.run_info,
            **info,
            "phase": phase,
        }

        self.rows.append(row)

    def write(self) -> None:
        """
        Write all logged rows to CSV.
        """

        if not self.enabled:
            return

        if not self.rows:
            return

        self.filename.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = sorted(
            {
                key
                for row in self.rows
                for key in row.keys()
            }
        )

        with open(self.filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(self.rows)

    def clear(self) -> None:
        """
        Remove all stored rows.
        """

        self.rows.clear()

    def __len__(self) -> int:
        return len(self.rows)