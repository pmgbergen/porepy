"""Module for exporting/importing numpy arrays to/from txt files."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import numpy as np


@dataclass
class TxtData:
    """Class for the representation of txt data."""

    header: str
    """Name of the data."""

    array: np.ndarray
    """Value of the data."""

    format: str = "%.6e"
    """Format of the array values to be included in the txt file."""


def export_data_to_txt(
    list_of_txt_data: list[TxtData],
    file_name: str = "out.txt",
) -> None:
    """Write data into a txt file.

    Raises:
        ValueError
            If the sizes of the arrays from the `list_of_txt_data` are not equal.

    Parameters:
        list_of_txt_data: List of txt data.
        file_name: ``default="out.txt"`

            Name of the txt file to be exported.

    """
    # Sanity check
    array_sizes = [np.size(data.array) for data in list_of_txt_data]
    if not all(x == array_sizes[0] for x in array_sizes):
        raise ValueError("Expected arrays of equal length.")

    # Number of columns
    cols: int = len(list_of_txt_data)
    rows: int = array_sizes[0]

    # Initialize export table
    data_type: list[tuple[str, Type[float]]] = []
    for idx in range(cols):
        data_type.append((f"var{idx}", float))
    export = np.zeros(rows, dtype=data_type)

    # Loop through the list and collect relevant data
    header = ""
    fmt = ""
    for idx, data in enumerate(list_of_txt_data):
        export[f"var{idx}"] = data.array
        header += data.header + " "
        fmt += data.format + " "
    header.rstrip(" ")  # strip last added empty space
    fmt.rstrip(" ")  # strip last added empty space

    # Write into the file
    np.savetxt(fname=file_name, X=export, header=header, fmt=fmt)  # type: ignore
