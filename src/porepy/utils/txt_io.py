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

    format: str = "%2.2e"
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
    # Sanity check.
    array_sizes = [np.size(data.array) for data in list_of_txt_data]
    if not all(x == array_sizes[0] for x in array_sizes):
        raise ValueError("Expected arrays of equal length.")

    # Number of columns.
    cols: int = len(list_of_txt_data)
    rows: int = array_sizes[0]

    # Initialize export table.
    data_type: list[tuple[str, Type[float]]] = []
    for idx in range(cols):
        data_type.append((f"var{idx}", float))
    export = np.zeros(rows, dtype=data_type)

    # Loop through the list and collect relevant data.
    header = ""
    fmt = ""
    for idx, data in enumerate(list_of_txt_data):
        export[f"var{idx}"] = data.array
        header += data.header + " "
        fmt += data.format + " "
    header.rstrip(" ")  # strip last added empty space.
    fmt.rstrip(" ")  # strip last added empty space.

    # Write into the file
    np.savetxt(fname=file_name, X=export, header=header, fmt=fmt)  # type: ignore


def read_data_from_txt(file_name: str) -> dict[str, np.ndarray]:
    """Read data from a txt file.

    Parameters:
        file_name: Name of the file that should be read. It must end with ``.txt``.

    Returns:
        Dictionary containing the read data.

    """
    # Open file and retrieve the header.
    with open(file_name) as f:
        lines = f.readlines()
    header = lines[0]
    header = header.lstrip("# ")
    header = header.rstrip("\n")

    # Get all variable names and values.
    names = header.split()
    values = np.loadtxt(
        fname=file_name,
        dtype=np.float64,
        skiprows=1,
        unpack=True,
    )

    # Prepare to return
    read_data: dict[str, np.ndarray] = {}
    for name, val in zip(names, values):
        read_data[name] = val

    return read_data
