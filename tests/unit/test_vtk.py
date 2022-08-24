""" This module contains tests of the vtk export filter.

The tests verify that the output has the same format (outputs the same string)
as before. Failure of any test thus indicates that something in the export
filter, or in the vtk python bindings has changed. If the change is external to
PorePy, this does not necessarily mean that something is wrong.

"""
import os
import sys
import unittest

import meshio
import numpy as np
from deepdiff import DeepDiff

import porepy as pp


class MeshioExporterTest(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName)
        self.folder = "./test_vtk/"
        self.file_name = "grid"

    def compare_vtu_files(self, test_file: str, reference_file: str) -> bool:
        """Determine whether two vtu files, accessed by their paths, are identical.
        Returns True if both files are identified as the same, False otherwise.
        This is the main auxiliary routine used to compare down below wheter the
        Exporter produces identical outputs as stored reference files."""

        # Trust meshio to read the vtu files
        test_data = meshio.read(test_file)
        reference_data = meshio.read(reference_file)

        # Determine the difference between the two meshio objects.
        # Ignore differences in the data type if values close.
        # To judge whether values are close, only consider certain
        # number of significant digits and base the comparison in
        # exponential form.
        diff = DeepDiff(
            test_data.__dict__,
            reference_data.__dict__,
            significant_digits=8,
            number_format_notation="e",
            ignore_numeric_type_changes=True,
        )

        # If the difference is empty, the meshio objects are identified as identical.
        return diff == {}

    def test_single_subdomain_1d(self):
        sd = pp.CartGrid(3, 1)
        sd.compute_geometry()

        dummy_scalar = np.ones(sd.num_cells) * sd.dim
        dummy_vector = np.ones((3, sd.num_cells)) * sd.dim

        save = pp.Exporter(
            sd,
            self.file_name,
            self.folder,
            export_constants_separately=False,
        )
        save.write_vtu([("dummy_scalar", dummy_scalar), ("dummy_vector", dummy_vector)])

        same_vtu_files: bool = self.compare_vtu_files(
            f"{self.folder}/{self.file_name}_1.vtu",
            "test_vtk_reference/single_subdomain_1d.vtu",
        )
        self.assertTrue(same_vtu_files)

    def test_single_subdomain_2d_simplex_grid(self):
        sd = pp.StructuredTriangleGrid([3] * 2, [1] * 2)
        sd.compute_geometry()

        dummy_scalar = np.ones(sd.num_cells) * sd.dim
        dummy_vector = np.ones((3, sd.num_cells)) * sd.dim

        save = pp.Exporter(
            sd,
            self.file_name,
            self.folder,
            export_constants_separately=False,
        )
        save.write_vtu([("dummy_scalar", dummy_scalar), ("dummy_vector", dummy_vector)])

        same_vtu_files: bool = self.compare_vtu_files(
            f"{self.folder}/{self.file_name}_2.vtu",
            "test_vtk_reference/single_subdomain_2d_simplex_grid.vtu",
        )
        self.assertTrue(same_vtu_files)

    def test_single_subdomain_2d_cart_grid(self):
        sd = pp.CartGrid([4] * 2, [1] * 2)
        sd.compute_geometry()

        dummy_scalar = np.ones(sd.num_cells) * sd.dim
        dummy_vector = np.ones((3, sd.num_cells)) * sd.dim

        save = pp.Exporter(
            sd,
            self.file_name,
            self.folder,
            export_constants_separately=False,
        )
        save.write_vtu([("dummy_scalar", dummy_scalar), ("dummy_vector", dummy_vector)])

        same_vtu_files: bool = self.compare_vtu_files(
            f"{self.folder}/{self.file_name}_2.vtu",
            "test_vtk_reference/single_subdomain_2d_cart_grid.vtu",
        )
        self.assertTrue(same_vtu_files)

    def test_single_subdomain_2d_polytop(self):
        sd = pp.StructuredTriangleGrid([2] * 2, [1] * 2)
        sd.compute_geometry()
        pp.coarsening.generate_coarse_grid(sd, [0, 1, 3, 3, 1, 1, 2, 2])
        sd.compute_geometry()

        dummy_scalar = np.ones(sd.num_cells) * sd.dim
        dummy_vector = np.ones((3, sd.num_cells)) * sd.dim

        save = pp.Exporter(
            sd,
            self.file_name,
            self.folder,
            export_constants_separately=False,
        )
        save.write_vtu([("dummy_scalar", dummy_scalar), ("dummy_vector", dummy_vector)])

        same_vtu_files: bool = self.compare_vtu_files(
            f"{self.folder}/{self.file_name}_2.vtu",
            "test_vtk_reference/single_subdomain_2d_polytop_grid.vtu",
        )
        self.assertTrue(same_vtu_files)

    def test_single_subdomain_3d_simplex_grid(self):
        sd = pp.StructuredTetrahedralGrid([3] * 3, [1] * 3)
        sd.compute_geometry()

        dummy_scalar = np.ones(sd.num_cells) * sd.dim
        dummy_vector = np.ones((3, sd.num_cells)) * sd.dim

        save = pp.Exporter(
            sd,
            self.file_name,
            self.folder,
            export_constants_separately=False,
        )
        save.write_vtu([("dummy_scalar", dummy_scalar), ("dummy_vector", dummy_vector)])

        same_vtu_files: bool = self.compare_vtu_files(
            f"{self.folder}/{self.file_name}_3.vtu",
            "test_vtk_reference/single_subdomain_3d_simplex_grid.vtu",
        )
        self.assertTrue(same_vtu_files)

    def test_single_subdomain_3d_cart_grid(self):
        sd = pp.CartGrid([4] * 3, [1] * 3)
        sd.compute_geometry()

        dummy_scalar = np.ones(sd.num_cells) * sd.dim
        dummy_vector = np.ones((3, sd.num_cells)) * sd.dim

        save = pp.Exporter(
            sd,
            self.file_name,
            self.folder,
            export_constants_separately=False,
        )
        save.write_vtu([("dummy_scalar", dummy_scalar), ("dummy_vector", dummy_vector)])

        same_vtu_files: bool = self.compare_vtu_files(
            f"{self.folder}/{self.file_name}_3.vtu",
            "test_vtk_reference/single_subdomain_3d_cart_grid.vtu",
        )
        self.assertTrue(same_vtu_files)

    def test_single_subdomain_3d_polytop_grid(self):
        sd = pp.CartGrid([3, 2, 3], [1] * 3)
        sd.compute_geometry()
        pp.coarsening.generate_coarse_grid(
            sd, [0, 0, 1, 0, 1, 1, 0, 2, 2, 3, 2, 2, 4, 4, 4, 4, 4, 4]
        )
        sd.compute_geometry()

        dummy_scalar = np.ones(sd.num_cells) * sd.dim
        dummy_vector = np.ones((3, sd.num_cells)) * sd.dim

        save = pp.Exporter(
            sd,
            self.file_name,
            self.folder,
            export_constants_separately=False,
        )
        save.write_vtu([("dummy_scalar", dummy_scalar), ("dummy_vector", dummy_vector)])

        same_vtu_files: bool = self.compare_vtu_files(
            f"{self.folder}/{self.file_name}_3.vtu",
            "test_vtk_reference/single_subdomain_3d_polytop_grid.vtu",
        )
        self.assertTrue(same_vtu_files)

    # NOTE: Suggest removing this test.
    def test_mdg_1(self):
        mdg, _ = pp.md_grids_2d.single_horizontal([4, 4], simplex=False)

        for sd, sd_data in mdg.subdomains(return_data=True):
            pp.set_state(
                sd_data,
                {
                    "dummy_scalar": np.ones(sd.num_cells) * sd.dim,
                    "dummy_vector": np.ones((3, sd.num_cells)) * sd.dim,
                },
            )

        save = pp.Exporter(
            mdg,
            self.file_name,
            self.folder,
            export_constants_separately=False,
        )
        save.write_vtu(["dummy_scalar", "dummy_vector"])

        same_vtu_files: list = []
        for appendix in ["1", "2", "mortar_1"]:
            same_vtu_files.append(
                self.compare_vtu_files(
                    f"{self.folder}/{self.file_name}_{appendix}.vtu",
                    f"test_vtk_reference/mdg_1_grid_{appendix}.vtu",
                )
            )
        self.assertTrue(all(same_vtu_files))

    # TODO Do we need this test if we have the subsequent one?
    def test_mdg_2(self):
        mdg, _ = pp.md_grids_2d.two_intersecting(
            [4, 4], y_endpoints=[0.25, 0.75], simplex=False
        )

        for sd, sd_data in mdg.subdomains(return_data=True):
            pp.set_state(
                sd_data,
                {
                    "dummy_scalar": np.ones(sd.num_cells) * sd.dim,
                    "dummy_vector": np.ones((3, sd.num_cells)) * sd.dim,
                },
            )

        for intf, intf_data in mdg.interfaces(return_data=True):
            pp.set_state(
                intf_data,
                {
                    "dummy_scalar": np.zeros(intf.num_cells),
                    "unique_dummy_scalar": np.zeros(intf.num_cells),
                },
            )

        save = pp.Exporter(
            mdg,
            self.file_name,
            self.folder,
            export_constants_separately=False,
        )
        save.write_vtu(["dummy_scalar", "dummy_vector", "unique_dummy_scalar"])

        same_vtu_files: list = []
        for appendix in ["1", "2", "mortar_1"]:
            same_vtu_files.append(
                self.compare_vtu_files(
                    f"{self.folder}/{self.file_name}_{appendix}.vtu",
                    f"test_vtk_reference/mdg_2_grid_{appendix}.vtu",
                )
            )
        self.assertTrue(all(same_vtu_files))

    def test_mdg_3(self):
        mdg, _ = pp.md_grids_2d.two_intersecting(
            [4, 4], y_endpoints=[0.25, 0.75], simplex=False
        )

        for sd, sd_data in mdg.subdomains(return_data=True):
            pp.set_state(
                sd_data,
                {
                    "dummy_scalar": np.ones(sd.num_cells) * sd.dim,
                    "dummy_vector": np.ones((3, sd.num_cells)) * sd.dim,
                },
            )

        for intf, intf_data in mdg.interfaces(return_data=True):
            pp.set_state(
                intf_data,
                {
                    "dummy_scalar": np.zeros(intf.num_cells),
                    "unique_dummy_scalar": np.zeros(intf.num_cells),
                },
            )

        subdomains_1d = mdg.subdomains(dim=1)
        subdomains_2d = mdg.subdomains(dim=2)
        sd_2d = subdomains_2d[0]
        # interfaces_1d = mdg.interfaces(dim=1) # FIXME not used below

        save = pp.Exporter(
            mdg,
            self.file_name,
            self.folder,
            export_constants_separately=False,
        )
        save.write_vtu(
            [
                (subdomains_1d, "dummy_scalar"),
                "dummy_vector",
                "unique_dummy_scalar",
                (sd_2d, "cc", sd_2d.cell_centers),
            ]
        )

        same_vtu_files: list = []
        for appendix in ["1", "2", "mortar_1"]:
            same_vtu_files.append(
                self.compare_vtu_files(
                    f"{self.folder}/{self.file_name}_{appendix}.vtu",
                    f"test_vtk_reference/mdg_3_grid_{appendix}.vtu",
                )
            )
        self.assertTrue(all(same_vtu_files))

    def test_constant_data(self):
        g = pp.StructuredTriangleGrid([3] * 2, [1] * 2)
        g.compute_geometry()

        dummy_scalar = np.ones(g.num_cells) * g.dim
        dummy_vector = np.ones((3, g.num_cells)) * g.dim

        save = pp.Exporter(
            g,
            self.file_name,
            self.folder,
        )
        save.add_constant_data([(g, "cc", g.cell_centers)])
        save.write_vtu([("dummy_scalar", dummy_scalar), ("dummy_vector", dummy_vector)])

        same_vtu_files: list = []
        for appendix in ["2", "constant_2"]:
            same_vtu_files.append(
                self.compare_vtu_files(
                    f"{self.folder}/{self.file_name}_{appendix}.vtu",
                    f"test_vtk_reference/constant_data_test_grid_{appendix}.vtu",
                )
            )
        self.assertTrue(all(same_vtu_files))

    def test_fractures_2d(self):
        p = np.array([[0, 2, 1, 2, 1], [0, 0, 0, 1, 2]])
        e = np.array([[0, 2, 3], [1, 3, 4]])
        domain = {"xmin": -2, "xmax": 3, "ymin": -2, "ymax": 3}
        network_2d = pp.FractureNetwork2d(p, e, domain)

        dummy_scalar = np.ones(network_2d.num_frac())
        dummy_vector = np.ones((3, network_2d.num_frac()))
        data = {"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector}

        network_2d.to_file(
            self.folder + self.file_name + ".vtu",
            data=data,
        )

        same_vtu_files: bool = self.compare_vtu_files(
            f"{self.folder}/{self.file_name}.vtu",
            "test_vtk_reference/fractures_2d.vtu",
        )
        self.assertTrue(same_vtu_files)

    def test_fractures_3d(self):
        f_1 = pp.PlaneFracture(np.array([[0, 1, 2, 0], [0, 0, 1, 1], [0, 0, 1, 1]]))
        f_2 = pp.PlaneFracture(
            np.array([[0.5, 0.5, 0.5, 0.5], [-1, 2, 2, -1], [-1, -1, 2, 2]])
        )
        domain = {"xmin": -2, "xmax": 3, "ymin": -2, "ymax": 3, "zmin": -3, "zmax": 3}
        network_3d = pp.FractureNetwork3d([f_1, f_2], domain=domain)

        num_frac = len(network_3d._fractures)
        dummy_scalar = [[1] for _ in range(num_frac)]
        dummy_vector = [[np.ones(3)] for _ in range(num_frac)]
        data = {"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector}

        network_3d.to_file(
            self.folder + self.file_name + ".vtu",
            data=data,
        )

        same_vtu_files: bool = self.compare_vtu_files(
            f"{self.folder}/{self.file_name}.vtu",
            "test_vtk_reference/fractures_3d.vtu",
        )
        self.assertTrue(same_vtu_files)


if __name__ == "__main__":
    unittest.main()
