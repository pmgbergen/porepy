"""Tests for the importing/exporting of numpy arrays into txt files."""
from __future__ import annotations

import numpy as np
import pytest
import os
from porepy.utils.txt_io import TxtData, export_data_to_txt


class TestExportData:
    """Collection of tests for `export_data_to_txt()`."""

    def test_sanity_of_list_of_txt_data(self) -> None:
        """Check that an error is raised when the arrays don't have the same size."""
        data0 = TxtData(header="pink", array=np.array([0, 1, 2]))
        data1 = TxtData(header="floyd", array=np.array([1973]))
        list_of_txt_data = [data0, data1]
        msg = "Expected arrays of equal length."
        with pytest.raises(ValueError) as excinfo:
            export_data_to_txt(list_of_txt_data)
        assert msg in str(excinfo.value)

    def test_export_data(self) -> None:
        """Check that the data is correctly exported."""
        # Create a list of txt data
        pressure = TxtData(header="pressure", array=np.array([0.0, 3.0, 500.3, 456.7]))
        flux = TxtData(header="flux", array=np.array([-2.0, 4.33, 60.46, 1.23e-5]))
        list_of_txt_data = [pressure, flux]

        # Export the data
        export_data_to_txt(list_of_txt_data)

        # Read back the file
        read_data: dict[str, np.ndarray] = self.read_txt_file("out.txt")

        # Compare
        assert len(read_data.keys()) == 2
        assert "pressure" in read_data.keys()
        assert "flux" in read_data.keys()
        np.testing.assert_allclose(pressure.array, read_data["pressure"])
        np.testing.assert_allclose(flux.array, read_data["flux"])

        # Delete file
        os.remove("out.txt")

    def read_txt_file(self, file_name: str) -> dict[str, np.ndarray]:
        """Helper function to read the txt file.

        Parameters:
            file_name: File of the name to be read.

        Returns:
            Dictionary containing the read data.

        """
        # Open file and retrieve the header
        with open(file_name) as f:
            lines = f.readlines()
        header = lines[0]
        header = header.lstrip('# ')
        header = header.rstrip("\n")

        # Get all variable names
        names = header.split()
        values = np.loadtxt(fname="out.txt", dtype=np.float64, unpack=True)

        # Prepare to return
        read_data: dict[str, np.ndarray] = {}
        for name, val in zip(names, values):
            read_data[name] = val

        return read_data

