"""Tests for the importing/exporting of numpy arrays into txt files."""

from __future__ import annotations

import os

import numpy as np
import pytest

from porepy.utils.txt_io import TxtData, export_data_to_txt, read_data_from_txt

"""Check that an error is raised when the arrays don't have the same size."""


def test_sanity_of_list_of_txt_data() -> None:
    data0 = TxtData(header="pink", array=np.array([0, 1, 2]))
    data1 = TxtData(header="floyd", array=np.array([1973]))
    list_of_txt_data = [data0, data1]
    msg = "Expected arrays of equal length."
    with pytest.raises(ValueError) as excinfo:
        export_data_to_txt(list_of_txt_data)
    assert msg in str(excinfo.value)


"""Check that the data is correctly exported."""


def test_export_data() -> None:
    # Create a list of txt data
    pressure = TxtData(
        header="pressure",
        array=np.array([0.0, 3.0, 500.3, 456.7]),
        format="%5.3e",
    )
    flux = TxtData(
        header="flux",
        array=np.array([-2.0, 4.33, 60.46, 1.23e-5]),
        format="%5.3e",
    )
    list_of_txt_data = [pressure, flux]

    # Export the data
    export_data_to_txt(list_of_txt_data)

    # Read back the file
    read_data: dict[str, np.ndarray] = read_data_from_txt("out.txt")

    # Compare
    assert len(read_data.keys()) == 2
    assert "pressure" in read_data.keys()
    assert "flux" in read_data.keys()
    np.testing.assert_allclose(pressure.array, read_data["pressure"])
    np.testing.assert_allclose(flux.array, read_data["flux"])

    # Delete file
    os.remove("out.txt")
