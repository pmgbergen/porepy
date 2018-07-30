import sys
import numpy as np
import unittest

from porepy.grids import structured, simplex
from porepy.fracs import meshing
from porepy.grids import coarsening as co

from porepy.viz.exporter import Exporter

if_vtk = "vtk" in sys.modules
if not if_vtk:
    import warnings

    warnings.warn("No vtk module loaded.")

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_single_grid_1d(self):
        if not if_vtk:
            return

        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        dummy_scalar = np.ones(g.num_cells) * g.dim
        dummy_vector = np.ones((3, g.num_cells)) * g.dim

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(g, file_name, folder, binary=False)
        save.write_vtk({"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector})

        with open(folder + file_name + ".vtu", "r") as content_file:
            content = content_file.read()

        assert content == self._single_grid_1d_grid_vtu()

    # ------------------------------------------------------------------------------#

    def test_single_grid_2d_simplex(self):
        if not if_vtk:
            return

        g = simplex.StructuredTriangleGrid([3] * 2, [1] * 2)
        g.compute_geometry()

        dummy_scalar = np.ones(g.num_cells) * g.dim
        dummy_vector = np.ones((3, g.num_cells)) * g.dim

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(g, file_name, folder, binary=False)
        save.write_vtk({"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector})

        with open(folder + file_name + ".vtu", "r") as content_file:
            content = content_file.read()
        assert content == self._single_grid_2d_simplex_grid_vtu()

    # ------------------------------------------------------------------------------#

    def test_single_grid_2d_cart(self):
        if not if_vtk:
            return

        g = structured.CartGrid([4] * 2, [1] * 2)
        g.compute_geometry()

        dummy_scalar = np.ones(g.num_cells) * g.dim
        dummy_vector = np.ones((3, g.num_cells)) * g.dim

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(g, file_name, folder, binary=False)
        save.write_vtk({"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector})

        with open(folder + file_name + ".vtu", "r") as content_file:
            content = content_file.read()
        assert content == self._single_grid_2d_cart_grid_vtu()

    # ------------------------------------------------------------------------------#

    def test_single_grid_2d_polytop(self):
        if not if_vtk:
            return

        g = structured.CartGrid([3, 2], [1] * 2)
        g.compute_geometry()
        co.generate_coarse_grid(g, [0, 0, 1, 0, 1, 1])
        g.compute_geometry()

        dummy_scalar = np.ones(g.num_cells) * g.dim
        dummy_vector = np.ones((3, g.num_cells)) * g.dim

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(g, file_name, folder, binary=False)
        save.write_vtk({"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector})

        with open(folder + file_name + ".vtu", "r") as content_file:
            content = content_file.read()
        assert content == self._single_grid_2d_polytop_grid_vtu()

    # ------------------------------------------------------------------------------#

    def test_single_grid_3d_simplex(self):
        if not if_vtk:
            return

        g = simplex.StructuredTetrahedralGrid([3] * 3, [1] * 3)
        g.compute_geometry()

        dummy_scalar = np.ones(g.num_cells) * g.dim
        dummy_vector = np.ones((3, g.num_cells)) * g.dim

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(g, file_name, folder, binary=False)
        save.write_vtk({"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector})

        with open(folder + file_name + ".vtu", "r") as content_file:
            content = content_file.read()
        assert content == self._single_grid_3d_simplex_grid_vtu()

    # ------------------------------------------------------------------------------#

    def test_single_grid_3d_cart(self):
        if not if_vtk:
            return

        g = structured.CartGrid([4] * 3, [1] * 3)
        g.compute_geometry()

        dummy_scalar = np.ones(g.num_cells) * g.dim
        dummy_vector = np.ones((3, g.num_cells)) * g.dim

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(g, file_name, folder, binary=False)
        save.write_vtk({"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector})

        with open(folder + file_name + ".vtu", "r") as content_file:
            content = content_file.read()
        assert content == self._single_grid_3d_cart_grid_vtu()

    # ------------------------------------------------------------------------------#

    def test_single_grid_3d_polytop(self):
        if not if_vtk:
            return

        g = structured.CartGrid([3, 2, 3], [1] * 3)
        g.compute_geometry()
        co.generate_coarse_grid(
            g, [0, 0, 1, 0, 1, 1, 0, 2, 2, 3, 2, 2, 4, 4, 4, 4, 4, 4]
        )
        g.compute_geometry()

        dummy_scalar = np.ones(g.num_cells) * g.dim
        dummy_vector = np.ones((3, g.num_cells)) * g.dim

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(g, file_name, folder, binary=False)
        save.write_vtk({"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector})

        with open(folder + file_name + ".vtu", "r") as content_file:
            content = content_file.read()
        assert content == self._single_grid_3d_polytop_grid_vtu()

    # ------------------------------------------------------------------------------#

    def test_gb_1(self):
        if not if_vtk:
            return

        f1 = np.array([[0, 1], [.5, .5]])
        gb = meshing.cart_grid([f1], [4] * 2, **{"physdims": [1, 1]})
        gb.compute_geometry()

        gb.add_node_props(["scalar_dummy", "dummy_vector"])

        for g, d in gb:
            d["dummy_scalar"] = np.ones(g.num_cells) * g.dim
            d["dummy_vector"] = np.ones((3, g.num_cells)) * g.dim

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(gb, file_name, folder, binary=False)
        save.write_vtk(["dummy_scalar", "dummy_vector"])

        with open(folder + file_name + ".pvd", "r") as content_file:
            content = content_file.read()
        assert content == self._gb_1_grid_pvd()

        with open(folder + file_name + "_1.vtu", "r") as content_file:
            content = content_file.read()
        assert content == self._gb_1_grid_1_vtu()

        with open(folder + file_name + "_2.vtu", "r") as content_file:
            content = content_file.read()
        assert content == self._gb_1_grid_2_vtu()

        with open(folder+"grid_mortar_1.vtu", "r") as content_file:
            content = content_file.read()
        assert content == self._gb_1_mortar_grid_vtu()

#------------------------------------------------------------------------------#

    def test_gb_2(self):
        if not if_vtk:
            return

        f1 = np.array([[0, 1], [.5, .5]])
        f2 = np.array([[.5, .5], [.25, .75]])
        gb = meshing.cart_grid([f1, f2], [4] * 2, **{"physdims": [1, 1]})
        gb.compute_geometry()

        gb.add_node_props(["dummy_scalar", "dummy_vector"])

        for g, d in gb:
            d["dummy_scalar"] = np.ones(g.num_cells) * g.dim
            d["dummy_vector"] = np.ones((3, g.num_cells)) * g.dim

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(gb, file_name, folder, binary=False)
        save.write_vtk(["dummy_scalar", "dummy_vector"])

        with open(folder + file_name + ".pvd", "r") as content_file:
            content = content_file.read()
        assert content == self._gb_2_grid_pvd()

        with open(folder + file_name + "_1.vtu", "r") as content_file:
            content = content_file.read()
        assert content == self._gb_2_grid_1_vtu()

        with open(folder + file_name + "_2.vtu", "r") as content_file:
            content = content_file.read()
        print(content)
        assert content == self._gb_2_grid_2_vtu()

        with open(folder+"grid_mortar_1.vtu", "r") as content_file:
            content = content_file.read()
        assert content == self._gb_2_mortar_grid_1_vtu()

#------------------------------------------------------------------------------#

    def _single_grid_1d_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="3">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="1.7320508076" RangeMax="1.7320508076">
          1 1 1 1 1 1
          1 1 1
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="2">
          0 1 2
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0" RangeMax="1">
          0 0 0 0.33333334327 0 0
          0.66666668653 0 0 1 0 0
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="3">
          0 1 1 2 2 3
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="2" RangeMax="6">
          2 4 6
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="3" RangeMax="3">
          3 3 3
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    # ------------------------------------------------------------------------------#

    def _single_grid_2d_simplex_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="16" NumberOfCells="18">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="2" RangeMax="2">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="3.4641016151" RangeMax="3.4641016151">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="2" RangeMax="2">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="17">
          0 1 2 3 4 5
          6 7 8 9 10 11
          12 13 14 15 16 17
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0" RangeMax="1.4142135624">
          0 0 0 0.33333334327 0 0
          0.66666668653 0 0 1 0 0
          0 0.33333334327 0 0.33333334327 0.33333334327 0
          0.66666668653 0.33333334327 0 1 0.33333334327 0
          0 0.66666668653 0 0.33333334327 0.66666668653 0
          0.66666668653 0.66666668653 0 1 0.66666668653 0
          0 1 0 0.33333334327 1 0
          0.66666668653 1 0 1 1 0
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="15">
          0 1 5 0 4 5
          1 2 6 1 5 6
          2 3 7 2 6 7
          4 5 9 4 8 9
          5 6 10 5 9 10
          6 7 11 6 10 11
          8 9 13 8 12 13
          9 10 14 9 13 14
          10 11 15 10 14 15
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="3" RangeMax="54">
          3 6 9 12 15 18
          21 24 27 30 33 36
          39 42 45 48 51 54
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="7" RangeMax="7">
          7 7 7 7 7 7
          7 7 7 7 7 7
          7 7 7 7 7 7
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    # ------------------------------------------------------------------------------#

    def _single_grid_2d_cart_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="25" NumberOfCells="16">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="2" RangeMax="2">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="3.4641016151" RangeMax="3.4641016151">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="2" RangeMax="2">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="15">
          0 1 2 3 4 5
          6 7 8 9 10 11
          12 13 14 15
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0" RangeMax="1.4142135624">
          0 0 0 0.25 0 0
          0.5 0 0 0.75 0 0
          1 0 0 0 0.25 0
          0.25 0.25 0 0.5 0.25 0
          0.75 0.25 0 1 0.25 0
          0 0.5 0 0.25 0.5 0
          0.5 0.5 0 0.75 0.5 0
          1 0.5 0 0 0.75 0
          0.25 0.75 0 0.5 0.75 0
          0.75 0.75 0 1 0.75 0
          0 1 0 0.25 1 0
          0.5 1 0 0.75 1 0
          1 1 0
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="24">
          0 5 6 1 1 6
          7 2 2 7 8 3
          3 8 9 4 5 10
          11 6 6 11 12 7
          7 12 13 8 8 13
          14 9 10 15 16 11
          11 16 17 12 12 17
          18 13 13 18 19 14
          15 20 21 16 16 21
          22 17 17 22 23 18
          18 23 24 19
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="4" RangeMax="64">
          4 8 12 16 20 24
          28 32 36 40 44 48
          52 56 60 64
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="7" RangeMax="7">
          7 7 7 7 7 7
          7 7 7 7 7 7
          7 7 7 7
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    # ------------------------------------------------------------------------------#

    def _single_grid_2d_polytop_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="12" NumberOfCells="2">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="2" RangeMax="2">
          2 2
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="3.4641016151" RangeMax="3.4641016151">
          2 2 2 2 2 2
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="2" RangeMax="2">
          2 2
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="1">
          0 1
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0" RangeMax="1.4142135624">
          0 0 0 0.33333334327 0 0
          0.66666668653 0 0 1 0 0
          0 0.5 0 0.33333334327 0.5 0
          0.66666668653 0.5 0 1 0.5 0
          0 1 0 0.33333334327 1 0
          0.66666668653 1 0 1 1 0
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="11">
          0 4 8 9 5 6
          2 1 2 6 5 9
          10 11 7 3
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="8" RangeMax="16">
          8 16
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="7" RangeMax="7">
          7 7
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    # ------------------------------------------------------------------------------#

    def _single_grid_3d_simplex_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="64" NumberOfCells="162">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="3" RangeMax="3">
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="5.1961524227" RangeMax="5.1961524227">
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="3" RangeMax="3">
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="161">
          0 1 2 3 4 5
          6 7 8 9 10 11
          12 13 14 15 16 17
          18 19 20 21 22 23
          24 25 26 27 28 29
          30 31 32 33 34 35
          36 37 38 39 40 41
          42 43 44 45 46 47
          48 49 50 51 52 53
          54 55 56 57 58 59
          60 61 62 63 64 65
          66 67 68 69 70 71
          72 73 74 75 76 77
          78 79 80 81 82 83
          84 85 86 87 88 89
          90 91 92 93 94 95
          96 97 98 99 100 101
          102 103 104 105 106 107
          108 109 110 111 112 113
          114 115 116 117 118 119
          120 121 122 123 124 125
          126 127 128 129 130 131
          132 133 134 135 136 137
          138 139 140 141 142 143
          144 145 146 147 148 149
          150 151 152 153 154 155
          156 157 158 159 160 161
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0" RangeMax="1.7320508076">
          0 0 0 0.33333334327 0 0
          0.66666668653 0 0 1 0 0
          0 0.33333334327 0 0.33333334327 0.33333334327 0
          0.66666668653 0.33333334327 0 1 0.33333334327 0
          0 0.66666668653 0 0.33333334327 0.66666668653 0
          0.66666668653 0.66666668653 0 1 0.66666668653 0
          0 1 0 0.33333334327 1 0
          0.66666668653 1 0 1 1 0
          0 0 0.33333334327 0.33333334327 0 0.33333334327
          0.66666668653 0 0.33333334327 1 0 0.33333334327
          0 0.33333334327 0.33333334327 0.33333334327 0.33333334327 0.33333334327
          0.66666668653 0.33333334327 0.33333334327 1 0.33333334327 0.33333334327
          0 0.66666668653 0.33333334327 0.33333334327 0.66666668653 0.33333334327
          0.66666668653 0.66666668653 0.33333334327 1 0.66666668653 0.33333334327
          0 1 0.33333334327 0.33333334327 1 0.33333334327
          0.66666668653 1 0.33333334327 1 1 0.33333334327
          0 0 0.66666668653 0.33333334327 0 0.66666668653
          0.66666668653 0 0.66666668653 1 0 0.66666668653
          0 0.33333334327 0.66666668653 0.33333334327 0.33333334327 0.66666668653
          0.66666668653 0.33333334327 0.66666668653 1 0.33333334327 0.66666668653
          0 0.66666668653 0.66666668653 0.33333334327 0.66666668653 0.66666668653
          0.66666668653 0.66666668653 0.66666668653 1 0.66666668653 0.66666668653
          0 1 0.66666668653 0.33333334327 1 0.66666668653
          0.66666668653 1 0.66666668653 1 1 0.66666668653
          0 0 1 0.33333334327 0 1
          0.66666668653 0 1 1 0 1
          0 0.33333334327 1 0.33333334327 0.33333334327 1
          0.66666668653 0.33333334327 1 1 0.33333334327 1
          0 0.66666668653 1 0.33333334327 0.66666668653 1
          0.66666668653 0.66666668653 1 1 0.66666668653 1
          0 1 1 0.33333334327 1 1
          0.66666668653 1 1 1 1 1
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="63">
          0 1 4 16 1 4
          16 20 1 16 17 20
          1 4 5 20 1 5
          17 20 5 17 20 21
          1 2 5 17 2 5
          17 21 2 17 18 21
          2 5 6 21 2 6
          18 21 6 18 21 22
          2 3 6 18 3 6
          18 22 3 18 19 22
          3 6 7 22 3 7
          19 22 7 19 22 23
          4 5 8 20 5 8
          20 24 5 20 21 24
          5 8 9 24 5 9
          21 24 9 21 24 25
          5 6 9 21 6 9
          21 25 6 21 22 25
          6 9 10 25 6 10
          22 25 10 22 25 26
          6 7 10 22 7 10
          22 26 7 22 23 26
          7 10 11 26 7 11
          23 26 11 23 26 27
          8 9 12 24 9 12
          24 28 9 24 25 28
          9 12 13 28 9 13
          25 28 13 25 28 29
          9 10 13 25 10 13
          25 29 10 25 26 29
          10 13 14 29 10 14
          26 29 14 26 29 30
          10 11 14 26 11 14
          26 30 11 26 27 30
          11 14 15 30 11 15
          27 30 15 27 30 31
          16 17 20 32 17 20
          32 36 17 32 33 36
          17 20 21 36 17 21
          33 36 21 33 36 37
          17 18 21 33 18 21
          33 37 18 33 34 37
          18 21 22 37 18 22
          34 37 22 34 37 38
          18 19 22 34 19 22
          34 38 19 34 35 38
          19 22 23 38 19 23
          35 38 23 35 38 39
          20 21 24 36 21 24
          36 40 21 36 37 40
          21 24 25 40 21 25
          37 40 25 37 40 41
          21 22 25 37 22 25
          37 41 22 37 38 41
          22 25 26 41 22 26
          38 41 26 38 41 42
          22 23 26 38 23 26
          38 42 23 38 39 42
          23 26 27 42 23 27
          39 42 27 39 42 43
          24 25 28 40 25 28
          40 44 25 40 41 44
          25 28 29 44 25 29
          41 44 29 41 44 45
          25 26 29 41 26 29
          41 45 26 41 42 45
          26 29 30 45 26 30
          42 45 30 42 45 46
          26 27 30 42 27 30
          42 46 27 42 43 46
          27 30 31 46 27 31
          43 46 31 43 46 47
          32 33 36 48 33 36
          48 52 33 48 49 52
          33 36 37 52 33 37
          49 52 37 49 52 53
          33 34 37 49 34 37
          49 53 34 49 50 53
          34 37 38 53 34 38
          50 53 38 50 53 54
          34 35 38 50 35 38
          50 54 35 50 51 54
          35 38 39 54 35 39
          51 54 39 51 54 55
          36 37 40 52 37 40
          52 56 37 52 53 56
          37 40 41 56 37 41
          53 56 41 53 56 57
          37 38 41 53 38 41
          53 57 38 53 54 57
          38 41 42 57 38 42
          54 57 42 54 57 58
          38 39 42 54 39 42
          54 58 39 54 55 58
          39 42 43 58 39 43
          55 58 43 55 58 59
          40 41 44 56 41 44
          56 60 41 56 57 60
          41 44 45 60 41 45
          57 60 45 57 60 61
          41 42 45 57 42 45
          57 61 42 57 58 61
          42 45 46 61 42 46
          58 61 46 58 61 62
          42 43 46 58 43 46
          58 62 43 58 59 62
          43 46 47 62 43 47
          59 62 47 59 62 63
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="4" RangeMax="648">
          4 8 12 16 20 24
          28 32 36 40 44 48
          52 56 60 64 68 72
          76 80 84 88 92 96
          100 104 108 112 116 120
          124 128 132 136 140 144
          148 152 156 160 164 168
          172 176 180 184 188 192
          196 200 204 208 212 216
          220 224 228 232 236 240
          244 248 252 256 260 264
          268 272 276 280 284 288
          292 296 300 304 308 312
          316 320 324 328 332 336
          340 344 348 352 356 360
          364 368 372 376 380 384
          388 392 396 400 404 408
          412 416 420 424 428 432
          436 440 444 448 452 456
          460 464 468 472 476 480
          484 488 492 496 500 504
          508 512 516 520 524 528
          532 536 540 544 548 552
          556 560 564 568 572 576
          580 584 588 592 596 600
          604 608 612 616 620 624
          628 632 636 640 644 648
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="42" RangeMax="42">
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
        </DataArray>
        <DataArray type="Int64" Name="faces" format="ascii" RangeMin="0" RangeMax="63">
          4 3 0 4 1 3
          0 16 1 3 16 4
          0 3 16 4 1 4
          3 16 4 1 3 20
          4 1 3 1 20 16
          3 4 20 16 4 3
          16 1 17 3 1 20
          16 3 1 20 17 3
          16 20 17 4 3 4
          5 1 3 20 4 1
          3 20 5 1 3 4
          20 5 4 3 17 5
          1 3 20 5 1 3
          1 20 17 3 5 20
          17 4 3 5 20 17
          3 5 21 17 3 20
          5 21 3 20 21 17
          4 3 1 5 2 3
          1 17 2 3 17 5
          1 3 17 5 2 4
          3 17 5 2 3 21
          5 2 3 2 21 17
          3 5 21 17 4 3
          17 2 18 3 2 21
          17 3 2 21 18 3
          17 21 18 4 3 5
          6 2 3 21 5 2
          3 21 6 2 3 5
          21 6 4 3 18 6
          2 3 21 6 2 3
          2 21 18 3 6 21
          18 4 3 6 21 18
          3 6 22 18 3 21
          6 22 3 21 22 18
          4 3 2 6 3 3
          2 18 3 3 18 6
          2 3 18 6 3 4
          3 18 6 3 3 22
          6 3 3 3 22 18
          3 6 22 18 4 3
          18 3 19 3 3 22
          18 3 3 22 19 3
          18 22 19 4 3 6
          7 3 3 22 6 3
          3 22 7 3 3 6
          22 7 4 3 19 7
          3 3 22 7 3 3
          3 22 19 3 7 22
          19 4 3 7 22 19
          3 7 23 19 3 22
          7 23 3 22 23 19
          4 3 4 8 5 3
          4 20 5 3 20 8
          4 3 20 8 5 4
          3 20 8 5 3 24
          8 5 3 5 24 20
          3 8 24 20 4 3
          20 5 21 3 5 24
          20 3 5 24 21 3
          20 24 21 4 3 8
          9 5 3 24 8 5
          3 24 9 5 3 8
          24 9 4 3 21 9
          5 3 24 9 5 3
          5 24 21 3 9 24
          21 4 3 9 24 21
          3 9 25 21 3 24
          9 25 3 24 25 21
          4 3 5 9 6 3
          5 21 6 3 21 9
          5 3 21 9 6 4
          3 21 9 6 3 25
          9 6 3 6 25 21
          3 9 25 21 4 3
          21 6 22 3 6 25
          21 3 6 25 22 3
          21 25 22 4 3 9
          10 6 3 25 9 6
          3 25 10 6 3 9
          25 10 4 3 22 10
          6 3 25 10 6 3
          6 25 22 3 10 25
          22 4 3 10 25 22
          3 10 26 22 3 25
          10 26 3 25 26 22
          4 3 6 10 7 3
          6 22 7 3 22 10
          6 3 22 10 7 4
          3 22 10 7 3 26
          10 7 3 7 26 22
          3 10 26 22 4 3
          22 7 23 3 7 26
          22 3 7 26 23 3
          22 26 23 4 3 10
          11 7 3 26 10 7
          3 26 11 7 3 10
          26 11 4 3 23 11
          7 3 26 11 7 3
          7 26 23 3 11 26
          23 4 3 11 26 23
          3 11 27 23 3 26
          11 27 3 26 27 23
          4 3 8 12 9 3
          8 24 9 3 24 12
          8 3 24 12 9 4
          3 24 12 9 3 28
          12 9 3 9 28 24
          3 12 28 24 4 3
          24 9 25 3 9 28
          24 3 9 28 25 3
          24 28 25 4 3 12
          13 9 3 28 12 9
          3 28 13 9 3 12
          28 13 4 3 25 13
          9 3 28 13 9 3
          9 28 25 3 13 28
          25 4 3 13 28 25
          3 13 29 25 3 28
          13 29 3 28 29 25
          4 3 9 13 10 3
          9 25 10 3 25 13
          9 3 25 13 10 4
          3 25 13 10 3 29
          13 10 3 10 29 25
          3 13 29 25 4 3
          25 10 26 3 10 29
          25 3 10 29 26 3
          25 29 26 4 3 13
          14 10 3 29 13 10
          3 29 14 10 3 13
          29 14 4 3 26 14
          10 3 29 14 10 3
          10 29 26 3 14 29
          26 4 3 14 29 26
          3 14 30 26 3 29
          14 30 3 29 30 26
          4 3 10 14 11 3
          10 26 11 3 26 14
          10 3 26 14 11 4
          3 26 14 11 3 30
          14 11 3 11 30 26
          3 14 30 26 4 3
          26 11 27 3 11 30
          26 3 11 30 27 3
          26 30 27 4 3 14
          15 11 3 30 14 11
          3 30 15 11 3 14
          30 15 4 3 27 15
          11 3 30 15 11 3
          11 30 27 3 15 30
          27 4 3 15 30 27
          3 15 31 27 3 30
          15 31 3 30 31 27
          4 3 16 20 17 3
          16 32 17 3 32 20
          16 3 32 20 17 4
          3 32 20 17 3 36
          20 17 3 17 36 32
          3 20 36 32 4 3
          32 17 33 3 17 36
          32 3 17 36 33 3
          32 36 33 4 3 20
          21 17 3 36 20 17
          3 36 21 17 3 20
          36 21 4 3 33 21
          17 3 36 21 17 3
          17 36 33 3 21 36
          33 4 3 21 36 33
          3 21 37 33 3 36
          21 37 3 36 37 33
          4 3 17 21 18 3
          17 33 18 3 33 21
          17 3 33 21 18 4
          3 33 21 18 3 37
          21 18 3 18 37 33
          3 21 37 33 4 3
          33 18 34 3 18 37
          33 3 18 37 34 3
          33 37 34 4 3 21
          22 18 3 37 21 18
          3 37 22 18 3 21
          37 22 4 3 34 22
          18 3 37 22 18 3
          18 37 34 3 22 37
          34 4 3 22 37 34
          3 22 38 34 3 37
          22 38 3 37 38 34
          4 3 18 22 19 3
          18 34 19 3 34 22
          18 3 34 22 19 4
          3 34 22 19 3 38
          22 19 3 19 38 34
          3 22 38 34 4 3
          34 19 35 3 19 38
          34 3 19 38 35 3
          34 38 35 4 3 22
          23 19 3 38 22 19
          3 38 23 19 3 22
          38 23 4 3 35 23
          19 3 38 23 19 3
          19 38 35 3 23 38
          35 4 3 23 38 35
          3 23 39 35 3 38
          23 39 3 38 39 35
          4 3 20 24 21 3
          20 36 21 3 36 24
          20 3 36 24 21 4
          3 36 24 21 3 40
          24 21 3 21 40 36
          3 24 40 36 4 3
          36 21 37 3 21 40
          36 3 21 40 37 3
          36 40 37 4 3 24
          25 21 3 40 24 21
          3 40 25 21 3 24
          40 25 4 3 37 25
          21 3 40 25 21 3
          21 40 37 3 25 40
          37 4 3 25 40 37
          3 25 41 37 3 40
          25 41 3 40 41 37
          4 3 21 25 22 3
          21 37 22 3 37 25
          21 3 37 25 22 4
          3 37 25 22 3 41
          25 22 3 22 41 37
          3 25 41 37 4 3
          37 22 38 3 22 41
          37 3 22 41 38 3
          37 41 38 4 3 25
          26 22 3 41 25 22
          3 41 26 22 3 25
          41 26 4 3 38 26
          22 3 41 26 22 3
          22 41 38 3 26 41
          38 4 3 26 41 38
          3 26 42 38 3 41
          26 42 3 41 42 38
          4 3 22 26 23 3
          22 38 23 3 38 26
          22 3 38 26 23 4
          3 38 26 23 3 42
          26 23 3 23 42 38
          3 26 42 38 4 3
          38 23 39 3 23 42
          38 3 23 42 39 3
          38 42 39 4 3 26
          27 23 3 42 26 23
          3 42 27 23 3 26
          42 27 4 3 39 27
          23 3 42 27 23 3
          23 42 39 3 27 42
          39 4 3 27 42 39
          3 27 43 39 3 42
          27 43 3 42 43 39
          4 3 24 28 25 3
          24 40 25 3 40 28
          24 3 40 28 25 4
          3 40 28 25 3 44
          28 25 3 25 44 40
          3 28 44 40 4 3
          40 25 41 3 25 44
          40 3 25 44 41 3
          40 44 41 4 3 28
          29 25 3 44 28 25
          3 44 29 25 3 28
          44 29 4 3 41 29
          25 3 44 29 25 3
          25 44 41 3 29 44
          41 4 3 29 44 41
          3 29 45 41 3 44
          29 45 3 44 45 41
          4 3 25 29 26 3
          25 41 26 3 41 29
          25 3 41 29 26 4
          3 41 29 26 3 45
          29 26 3 26 45 41
          3 29 45 41 4 3
          41 26 42 3 26 45
          41 3 26 45 42 3
          41 45 42 4 3 29
          30 26 3 45 29 26
          3 45 30 26 3 29
          45 30 4 3 42 30
          26 3 45 30 26 3
          26 45 42 3 30 45
          42 4 3 30 45 42
          3 30 46 42 3 45
          30 46 3 45 46 42
          4 3 26 30 27 3
          26 42 27 3 42 30
          26 3 42 30 27 4
          3 42 30 27 3 46
          30 27 3 27 46 42
          3 30 46 42 4 3
          42 27 43 3 27 46
          42 3 27 46 43 3
          42 46 43 4 3 30
          31 27 3 46 30 27
          3 46 31 27 3 30
          46 31 4 3 43 31
          27 3 46 31 27 3
          27 46 43 3 31 46
          43 4 3 31 46 43
          3 31 47 43 3 46
          31 47 3 46 47 43
          4 3 32 36 33 3
          32 48 33 3 48 36
          32 3 48 36 33 4
          3 48 36 33 3 52
          36 33 3 33 52 48
          3 36 52 48 4 3
          48 33 49 3 33 52
          48 3 33 52 49 3
          48 52 49 4 3 36
          37 33 3 52 36 33
          3 52 37 33 3 36
          52 37 4 3 49 37
          33 3 52 37 33 3
          33 52 49 3 37 52
          49 4 3 37 52 49
          3 37 53 49 3 52
          37 53 3 52 53 49
          4 3 33 37 34 3
          33 49 34 3 49 37
          33 3 49 37 34 4
          3 49 37 34 3 53
          37 34 3 34 53 49
          3 37 53 49 4 3
          49 34 50 3 34 53
          49 3 34 53 50 3
          49 53 50 4 3 37
          38 34 3 53 37 34
          3 53 38 34 3 37
          53 38 4 3 50 38
          34 3 53 38 34 3
          34 53 50 3 38 53
          50 4 3 38 53 50
          3 38 54 50 3 53
          38 54 3 53 54 50
          4 3 34 38 35 3
          34 50 35 3 50 38
          34 3 50 38 35 4
          3 50 38 35 3 54
          38 35 3 35 54 50
          3 38 54 50 4 3
          50 35 51 3 35 54
          50 3 35 54 51 3
          50 54 51 4 3 38
          39 35 3 54 38 35
          3 54 39 35 3 38
          54 39 4 3 51 39
          35 3 54 39 35 3
          35 54 51 3 39 54
          51 4 3 39 54 51
          3 39 55 51 3 54
          39 55 3 54 55 51
          4 3 36 40 37 3
          36 52 37 3 52 40
          36 3 52 40 37 4
          3 52 40 37 3 56
          40 37 3 37 56 52
          3 40 56 52 4 3
          52 37 53 3 37 56
          52 3 37 56 53 3
          52 56 53 4 3 40
          41 37 3 56 40 37
          3 56 41 37 3 40
          56 41 4 3 53 41
          37 3 56 41 37 3
          37 56 53 3 41 56
          53 4 3 41 56 53
          3 41 57 53 3 56
          41 57 3 56 57 53
          4 3 37 41 38 3
          37 53 38 3 53 41
          37 3 53 41 38 4
          3 53 41 38 3 57
          41 38 3 38 57 53
          3 41 57 53 4 3
          53 38 54 3 38 57
          53 3 38 57 54 3
          53 57 54 4 3 41
          42 38 3 57 41 38
          3 57 42 38 3 41
          57 42 4 3 54 42
          38 3 57 42 38 3
          38 57 54 3 42 57
          54 4 3 42 57 54
          3 42 58 54 3 57
          42 58 3 57 58 54
          4 3 38 42 39 3
          38 54 39 3 54 42
          38 3 54 42 39 4
          3 54 42 39 3 58
          42 39 3 39 58 54
          3 42 58 54 4 3
          54 39 55 3 39 58
          54 3 39 58 55 3
          54 58 55 4 3 42
          43 39 3 58 42 39
          3 58 43 39 3 42
          58 43 4 3 55 43
          39 3 58 43 39 3
          39 58 55 3 43 58
          55 4 3 43 58 55
          3 43 59 55 3 58
          43 59 3 58 59 55
          4 3 40 44 41 3
          40 56 41 3 56 44
          40 3 56 44 41 4
          3 56 44 41 3 60
          44 41 3 41 60 56
          3 44 60 56 4 3
          56 41 57 3 41 60
          56 3 41 60 57 3
          56 60 57 4 3 44
          45 41 3 60 44 41
          3 60 45 41 3 44
          60 45 4 3 57 45
          41 3 60 45 41 3
          41 60 57 3 45 60
          57 4 3 45 60 57
          3 45 61 57 3 60
          45 61 3 60 61 57
          4 3 41 45 42 3
          41 57 42 3 57 45
          41 3 57 45 42 4
          3 57 45 42 3 61
          45 42 3 42 61 57
          3 45 61 57 4 3
          57 42 58 3 42 61
          57 3 42 61 58 3
          57 61 58 4 3 45
          46 42 3 61 45 42
          3 61 46 42 3 45
          61 46 4 3 58 46
          42 3 61 46 42 3
          42 61 58 3 46 61
          58 4 3 46 61 58
          3 46 62 58 3 61
          46 62 3 61 62 58
          4 3 42 46 43 3
          42 58 43 3 58 46
          42 3 58 46 43 4
          3 58 46 43 3 62
          46 43 3 43 62 58
          3 46 62 58 4 3
          58 43 59 3 43 62
          58 3 43 62 59 3
          58 62 59 4 3 46
          47 43 3 62 46 43
          3 62 47 43 3 46
          62 47 4 3 59 47
          43 3 62 47 43 3
          43 62 59 3 47 62
          59 4 3 47 62 59
          3 47 63 59 3 62
          47 63 3 62 63 59
        </DataArray>
        <DataArray type="Int64" Name="faceoffsets" format="ascii" RangeMin="17" RangeMax="2754">
          17 34 51 68 85 102
          119 136 153 170 187 204
          221 238 255 272 289 306
          323 340 357 374 391 408
          425 442 459 476 493 510
          527 544 561 578 595 612
          629 646 663 680 697 714
          731 748 765 782 799 816
          833 850 867 884 901 918
          935 952 969 986 1003 1020
          1037 1054 1071 1088 1105 1122
          1139 1156 1173 1190 1207 1224
          1241 1258 1275 1292 1309 1326
          1343 1360 1377 1394 1411 1428
          1445 1462 1479 1496 1513 1530
          1547 1564 1581 1598 1615 1632
          1649 1666 1683 1700 1717 1734
          1751 1768 1785 1802 1819 1836
          1853 1870 1887 1904 1921 1938
          1955 1972 1989 2006 2023 2040
          2057 2074 2091 2108 2125 2142
          2159 2176 2193 2210 2227 2244
          2261 2278 2295 2312 2329 2346
          2363 2380 2397 2414 2431 2448
          2465 2482 2499 2516 2533 2550
          2567 2584 2601 2618 2635 2652
          2669 2686 2703 2720 2737 2754
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    # ------------------------------------------------------------------------------#

    def _single_grid_3d_cart_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="125" NumberOfCells="64">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="3" RangeMax="3">
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="5.1961524227" RangeMax="5.1961524227">
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="3" RangeMax="3">
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3 3
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="63">
          0 1 2 3 4 5
          6 7 8 9 10 11
          12 13 14 15 16 17
          18 19 20 21 22 23
          24 25 26 27 28 29
          30 31 32 33 34 35
          36 37 38 39 40 41
          42 43 44 45 46 47
          48 49 50 51 52 53
          54 55 56 57 58 59
          60 61 62 63
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0" RangeMax="1.7320508076">
          0 0 0 0.25 0 0
          0.5 0 0 0.75 0 0
          1 0 0 0 0.25 0
          0.25 0.25 0 0.5 0.25 0
          0.75 0.25 0 1 0.25 0
          0 0.5 0 0.25 0.5 0
          0.5 0.5 0 0.75 0.5 0
          1 0.5 0 0 0.75 0
          0.25 0.75 0 0.5 0.75 0
          0.75 0.75 0 1 0.75 0
          0 1 0 0.25 1 0
          0.5 1 0 0.75 1 0
          1 1 0 0 0 0.25
          0.25 0 0.25 0.5 0 0.25
          0.75 0 0.25 1 0 0.25
          0 0.25 0.25 0.25 0.25 0.25
          0.5 0.25 0.25 0.75 0.25 0.25
          1 0.25 0.25 0 0.5 0.25
          0.25 0.5 0.25 0.5 0.5 0.25
          0.75 0.5 0.25 1 0.5 0.25
          0 0.75 0.25 0.25 0.75 0.25
          0.5 0.75 0.25 0.75 0.75 0.25
          1 0.75 0.25 0 1 0.25
          0.25 1 0.25 0.5 1 0.25
          0.75 1 0.25 1 1 0.25
          0 0 0.5 0.25 0 0.5
          0.5 0 0.5 0.75 0 0.5
          1 0 0.5 0 0.25 0.5
          0.25 0.25 0.5 0.5 0.25 0.5
          0.75 0.25 0.5 1 0.25 0.5
          0 0.5 0.5 0.25 0.5 0.5
          0.5 0.5 0.5 0.75 0.5 0.5
          1 0.5 0.5 0 0.75 0.5
          0.25 0.75 0.5 0.5 0.75 0.5
          0.75 0.75 0.5 1 0.75 0.5
          0 1 0.5 0.25 1 0.5
          0.5 1 0.5 0.75 1 0.5
          1 1 0.5 0 0 0.75
          0.25 0 0.75 0.5 0 0.75
          0.75 0 0.75 1 0 0.75
          0 0.25 0.75 0.25 0.25 0.75
          0.5 0.25 0.75 0.75 0.25 0.75
          1 0.25 0.75 0 0.5 0.75
          0.25 0.5 0.75 0.5 0.5 0.75
          0.75 0.5 0.75 1 0.5 0.75
          0 0.75 0.75 0.25 0.75 0.75
          0.5 0.75 0.75 0.75 0.75 0.75
          1 0.75 0.75 0 1 0.75
          0.25 1 0.75 0.5 1 0.75
          0.75 1 0.75 1 1 0.75
          0 0 1 0.25 0 1
          0.5 0 1 0.75 0 1
          1 0 1 0 0.25 1
          0.25 0.25 1 0.5 0.25 1
          0.75 0.25 1 1 0.25 1
          0 0.5 1 0.25 0.5 1
          0.5 0.5 1 0.75 0.5 1
          1 0.5 1 0 0.75 1
          0.25 0.75 1 0.5 0.75 1
          0.75 0.75 1 1 0.75 1
          0 1 1 0.25 1 1
          0.5 1 1 0.75 1 1
          1 1 1
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="124">
          0 1 5 6 25 26
          30 31 1 2 6 7
          26 27 31 32 2 3
          7 8 27 28 32 33
          3 4 8 9 28 29
          33 34 5 6 10 11
          30 31 35 36 6 7
          11 12 31 32 36 37
          7 8 12 13 32 33
          37 38 8 9 13 14
          33 34 38 39 10 11
          15 16 35 36 40 41
          11 12 16 17 36 37
          41 42 12 13 17 18
          37 38 42 43 13 14
          18 19 38 39 43 44
          15 16 20 21 40 41
          45 46 16 17 21 22
          41 42 46 47 17 18
          22 23 42 43 47 48
          18 19 23 24 43 44
          48 49 25 26 30 31
          50 51 55 56 26 27
          31 32 51 52 56 57
          27 28 32 33 52 53
          57 58 28 29 33 34
          53 54 58 59 30 31
          35 36 55 56 60 61
          31 32 36 37 56 57
          61 62 32 33 37 38
          57 58 62 63 33 34
          38 39 58 59 63 64
          35 36 40 41 60 61
          65 66 36 37 41 42
          61 62 66 67 37 38
          42 43 62 63 67 68
          38 39 43 44 63 64
          68 69 40 41 45 46
          65 66 70 71 41 42
          46 47 66 67 71 72
          42 43 47 48 67 68
          72 73 43 44 48 49
          68 69 73 74 50 51
          55 56 75 76 80 81
          51 52 56 57 76 77
          81 82 52 53 57 58
          77 78 82 83 53 54
          58 59 78 79 83 84
          55 56 60 61 80 81
          85 86 56 57 61 62
          81 82 86 87 57 58
          62 63 82 83 87 88
          58 59 63 64 83 84
          88 89 60 61 65 66
          85 86 90 91 61 62
          66 67 86 87 91 92
          62 63 67 68 87 88
          92 93 63 64 68 69
          88 89 93 94 65 66
          70 71 90 91 95 96
          66 67 71 72 91 92
          96 97 67 68 72 73
          92 93 97 98 68 69
          73 74 93 94 98 99
          75 76 80 81 100 101
          105 106 76 77 81 82
          101 102 106 107 77 78
          82 83 102 103 107 108
          78 79 83 84 103 104
          108 109 80 81 85 86
          105 106 110 111 81 82
          86 87 106 107 111 112
          82 83 87 88 107 108
          112 113 83 84 88 89
          108 109 113 114 85 86
          90 91 110 111 115 116
          86 87 91 92 111 112
          116 117 87 88 92 93
          112 113 117 118 88 89
          93 94 113 114 118 119
          90 91 95 96 115 116
          120 121 91 92 96 97
          116 117 121 122 92 93
          97 98 117 118 122 123
          93 94 98 99 118 119
          123 124
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="8" RangeMax="512">
          8 16 24 32 40 48
          56 64 72 80 88 96
          104 112 120 128 136 144
          152 160 168 176 184 192
          200 208 216 224 232 240
          248 256 264 272 280 288
          296 304 312 320 328 336
          344 352 360 368 376 384
          392 400 408 416 424 432
          440 448 456 464 472 480
          488 496 504 512
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="42" RangeMax="42">
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42 42 42
          42 42 42 42
        </DataArray>
        <DataArray type="Int64" Name="faces" format="ascii" RangeMin="0" RangeMax="124">
          6 4 25 30 5 0
          4 26 31 6 1 4
          25 0 1 26 4 30
          5 6 31 4 0 5
          6 1 4 25 30 31
          26 6 4 26 31 6
          1 4 27 32 7 2
          4 26 1 2 27 4
          31 6 7 32 4 1
          6 7 2 4 26 31
          32 27 6 4 27 32
          7 2 4 28 33 8
          3 4 27 2 3 28
          4 32 7 8 33 4
          2 7 8 3 4 27
          32 33 28 6 4 28
          33 8 3 4 29 34
          9 4 4 28 3 4
          29 4 33 8 9 34
          4 3 8 9 4 4
          28 33 34 29 6 4
          30 35 10 5 4 31
          36 11 6 4 30 5
          6 31 4 35 10 11
          36 4 5 10 11 6
          4 30 35 36 31 6
          4 31 36 11 6 4
          32 37 12 7 4 31
          6 7 32 4 36 11
          12 37 4 6 11 12
          7 4 31 36 37 32
          6 4 32 37 12 7
          4 33 38 13 8 4
          32 7 8 33 4 37
          12 13 38 4 7 12
          13 8 4 32 37 38
          33 6 4 33 38 13
          8 4 34 39 14 9
          4 33 8 9 34 4
          38 13 14 39 4 8
          13 14 9 4 33 38
          39 34 6 4 35 40
          15 10 4 36 41 16
          11 4 35 10 11 36
          4 40 15 16 41 4
          10 15 16 11 4 35
          40 41 36 6 4 36
          41 16 11 4 37 42
          17 12 4 36 11 12
          37 4 41 16 17 42
          4 11 16 17 12 4
          36 41 42 37 6 4
          37 42 17 12 4 38
          43 18 13 4 37 12
          13 38 4 42 17 18
          43 4 12 17 18 13
          4 37 42 43 38 6
          4 38 43 18 13 4
          39 44 19 14 4 38
          13 14 39 4 43 18
          19 44 4 13 18 19
          14 4 38 43 44 39
          6 4 40 45 20 15
          4 41 46 21 16 4
          40 15 16 41 4 45
          20 21 46 4 15 20
          21 16 4 40 45 46
          41 6 4 41 46 21
          16 4 42 47 22 17
          4 41 16 17 42 4
          46 21 22 47 4 16
          21 22 17 4 41 46
          47 42 6 4 42 47
          22 17 4 43 48 23
          18 4 42 17 18 43
          4 47 22 23 48 4
          17 22 23 18 4 42
          47 48 43 6 4 43
          48 23 18 4 44 49
          24 19 4 43 18 19
          44 4 48 23 24 49
          4 18 23 24 19 4
          43 48 49 44 6 4
          50 55 30 25 4 51
          56 31 26 4 50 25
          26 51 4 55 30 31
          56 4 25 30 31 26
          4 50 55 56 51 6
          4 51 56 31 26 4
          52 57 32 27 4 51
          26 27 52 4 56 31
          32 57 4 26 31 32
          27 4 51 56 57 52
          6 4 52 57 32 27
          4 53 58 33 28 4
          52 27 28 53 4 57
          32 33 58 4 27 32
          33 28 4 52 57 58
          53 6 4 53 58 33
          28 4 54 59 34 29
          4 53 28 29 54 4
          58 33 34 59 4 28
          33 34 29 4 53 58
          59 54 6 4 55 60
          35 30 4 56 61 36
          31 4 55 30 31 56
          4 60 35 36 61 4
          30 35 36 31 4 55
          60 61 56 6 4 56
          61 36 31 4 57 62
          37 32 4 56 31 32
          57 4 61 36 37 62
          4 31 36 37 32 4
          56 61 62 57 6 4
          57 62 37 32 4 58
          63 38 33 4 57 32
          33 58 4 62 37 38
          63 4 32 37 38 33
          4 57 62 63 58 6
          4 58 63 38 33 4
          59 64 39 34 4 58
          33 34 59 4 63 38
          39 64 4 33 38 39
          34 4 58 63 64 59
          6 4 60 65 40 35
          4 61 66 41 36 4
          60 35 36 61 4 65
          40 41 66 4 35 40
          41 36 4 60 65 66
          61 6 4 61 66 41
          36 4 62 67 42 37
          4 61 36 37 62 4
          66 41 42 67 4 36
          41 42 37 4 61 66
          67 62 6 4 62 67
          42 37 4 63 68 43
          38 4 62 37 38 63
          4 67 42 43 68 4
          37 42 43 38 4 62
          67 68 63 6 4 63
          68 43 38 4 64 69
          44 39 4 63 38 39
          64 4 68 43 44 69
          4 38 43 44 39 4
          63 68 69 64 6 4
          65 70 45 40 4 66
          71 46 41 4 65 40
          41 66 4 70 45 46
          71 4 40 45 46 41
          4 65 70 71 66 6
          4 66 71 46 41 4
          67 72 47 42 4 66
          41 42 67 4 71 46
          47 72 4 41 46 47
          42 4 66 71 72 67
          6 4 67 72 47 42
          4 68 73 48 43 4
          67 42 43 68 4 72
          47 48 73 4 42 47
          48 43 4 67 72 73
          68 6 4 68 73 48
          43 4 69 74 49 44
          4 68 43 44 69 4
          73 48 49 74 4 43
          48 49 44 4 68 73
          74 69 6 4 75 80
          55 50 4 76 81 56
          51 4 75 50 51 76
          4 80 55 56 81 4
          50 55 56 51 4 75
          80 81 76 6 4 76
          81 56 51 4 77 82
          57 52 4 76 51 52
          77 4 81 56 57 82
          4 51 56 57 52 4
          76 81 82 77 6 4
          77 82 57 52 4 78
          83 58 53 4 77 52
          53 78 4 82 57 58
          83 4 52 57 58 53
          4 77 82 83 78 6
          4 78 83 58 53 4
          79 84 59 54 4 78
          53 54 79 4 83 58
          59 84 4 53 58 59
          54 4 78 83 84 79
          6 4 80 85 60 55
          4 81 86 61 56 4
          80 55 56 81 4 85
          60 61 86 4 55 60
          61 56 4 80 85 86
          81 6 4 81 86 61
          56 4 82 87 62 57
          4 81 56 57 82 4
          86 61 62 87 4 56
          61 62 57 4 81 86
          87 82 6 4 82 87
          62 57 4 83 88 63
          58 4 82 57 58 83
          4 87 62 63 88 4
          57 62 63 58 4 82
          87 88 83 6 4 83
          88 63 58 4 84 89
          64 59 4 83 58 59
          84 4 88 63 64 89
          4 58 63 64 59 4
          83 88 89 84 6 4
          85 90 65 60 4 86
          91 66 61 4 85 60
          61 86 4 90 65 66
          91 4 60 65 66 61
          4 85 90 91 86 6
          4 86 91 66 61 4
          87 92 67 62 4 86
          61 62 87 4 91 66
          67 92 4 61 66 67
          62 4 86 91 92 87
          6 4 87 92 67 62
          4 88 93 68 63 4
          87 62 63 88 4 92
          67 68 93 4 62 67
          68 63 4 87 92 93
          88 6 4 88 93 68
          63 4 89 94 69 64
          4 88 63 64 89 4
          93 68 69 94 4 63
          68 69 64 4 88 93
          94 89 6 4 90 95
          70 65 4 91 96 71
          66 4 90 65 66 91
          4 95 70 71 96 4
          65 70 71 66 4 90
          95 96 91 6 4 91
          96 71 66 4 92 97
          72 67 4 91 66 67
          92 4 96 71 72 97
          4 66 71 72 67 4
          91 96 97 92 6 4
          92 97 72 67 4 93
          98 73 68 4 92 67
          68 93 4 97 72 73
          98 4 67 72 73 68
          4 92 97 98 93 6
          4 93 98 73 68 4
          94 99 74 69 4 93
          68 69 94 4 98 73
          74 99 4 68 73 74
          69 4 93 98 99 94
          6 4 100 105 80 75
          4 101 106 81 76 4
          100 75 76 101 4 105
          80 81 106 4 75 80
          81 76 4 100 105 106
          101 6 4 101 106 81
          76 4 102 107 82 77
          4 101 76 77 102 4
          106 81 82 107 4 76
          81 82 77 4 101 106
          107 102 6 4 102 107
          82 77 4 103 108 83
          78 4 102 77 78 103
          4 107 82 83 108 4
          77 82 83 78 4 102
          107 108 103 6 4 103
          108 83 78 4 104 109
          84 79 4 103 78 79
          104 4 108 83 84 109
          4 78 83 84 79 4
          103 108 109 104 6 4
          105 110 85 80 4 106
          111 86 81 4 105 80
          81 106 4 110 85 86
          111 4 80 85 86 81
          4 105 110 111 106 6
          4 106 111 86 81 4
          107 112 87 82 4 106
          81 82 107 4 111 86
          87 112 4 81 86 87
          82 4 106 111 112 107
          6 4 107 112 87 82
          4 108 113 88 83 4
          107 82 83 108 4 112
          87 88 113 4 82 87
          88 83 4 107 112 113
          108 6 4 108 113 88
          83 4 109 114 89 84
          4 108 83 84 109 4
          113 88 89 114 4 83
          88 89 84 4 108 113
          114 109 6 4 110 115
          90 85 4 111 116 91
          86 4 110 85 86 111
          4 115 90 91 116 4
          85 90 91 86 4 110
          115 116 111 6 4 111
          116 91 86 4 112 117
          92 87 4 111 86 87
          112 4 116 91 92 117
          4 86 91 92 87 4
          111 116 117 112 6 4
          112 117 92 87 4 113
          118 93 88 4 112 87
          88 113 4 117 92 93
          118 4 87 92 93 88
          4 112 117 118 113 6
          4 113 118 93 88 4
          114 119 94 89 4 113
          88 89 114 4 118 93
          94 119 4 88 93 94
          89 4 113 118 119 114
          6 4 115 120 95 90
          4 116 121 96 91 4
          115 90 91 116 4 120
          95 96 121 4 90 95
          96 91 4 115 120 121
          116 6 4 116 121 96
          91 4 117 122 97 92
          4 116 91 92 117 4
          121 96 97 122 4 91
          96 97 92 4 116 121
          122 117 6 4 117 122
          97 92 4 118 123 98
          93 4 117 92 93 118
          4 122 97 98 123 4
          92 97 98 93 4 117
          122 123 118 6 4 118
          123 98 93 4 119 124
          99 94 4 118 93 94
          119 4 123 98 99 124
          4 93 98 99 94 4
          118 123 124 119
        </DataArray>
        <DataArray type="Int64" Name="faceoffsets" format="ascii" RangeMin="31" RangeMax="1984">
          31 62 93 124 155 186
          217 248 279 310 341 372
          403 434 465 496 527 558
          589 620 651 682 713 744
          775 806 837 868 899 930
          961 992 1023 1054 1085 1116
          1147 1178 1209 1240 1271 1302
          1333 1364 1395 1426 1457 1488
          1519 1550 1581 1612 1643 1674
          1705 1736 1767 1798 1829 1860
          1891 1922 1953 1984
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    # ------------------------------------------------------------------------------#

    def _single_grid_3d_polytop_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="48" NumberOfCells="5">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="3" RangeMax="3">
          3 3 3 3 3
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="5.1961524227" RangeMax="5.1961524227">
          3 3 3 3 3 3
          3 3 3 3 3 3
          3 3 3
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="3" RangeMax="3">
          3 3 3 3 3
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="4">
          0 1 2 3 4
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0" RangeMax="1.7320508076">
          0 0 0 0.33333334327 0 0
          0.66666668653 0 0 1 0 0
          0 0.5 0 0.33333334327 0.5 0
          0.66666668653 0.5 0 1 0.5 0
          0 1 0 0.33333334327 1 0
          0.66666668653 1 0 1 1 0
          0 0 0.33333334327 0.33333334327 0 0.33333334327
          0.66666668653 0 0.33333334327 1 0 0.33333334327
          0 0.5 0.33333334327 0.33333334327 0.5 0.33333334327
          0.66666668653 0.5 0.33333334327 1 0.5 0.33333334327
          0 1 0.33333334327 0.33333334327 1 0.33333334327
          0.66666668653 1 0.33333334327 1 1 0.33333334327
          0 0 0.66666668653 0.33333334327 0 0.66666668653
          0.66666668653 0 0.66666668653 1 0 0.66666668653
          0 0.5 0.66666668653 0.33333334327 0.5 0.66666668653
          0.66666668653 0.5 0.66666668653 1 0.5 0.66666668653
          0 1 0.66666668653 0.33333334327 1 0.66666668653
          0.66666668653 1 0.66666668653 1 1 0.66666668653
          0 0 1 0.33333334327 0 1
          0.66666668653 0 1 1 0 1
          0 0.5 1 0.33333334327 0.5 1
          0.66666668653 0.5 1 1 0.5 1
          0 1 1 0.33333334327 1 1
          0.66666668653 1 1 1 1 1
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="47">
          0 1 2 4 5 6
          8 9 12 13 14 16
          17 18 20 21 24 25
          28 29 2 3 5 6
          7 9 10 11 14 15
          17 18 19 21 22 23
          13 14 15 17 18 19
          21 22 23 25 26 27
          29 30 31 33 34 35
          16 17 20 21 28 29
          32 33 24 25 26 27
          28 29 30 31 32 33
          34 35 36 37 38 39
          40 41 42 43 44 45
          46 47
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="20" RangeMax="86">
          20 36 54 62 86
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="42" RangeMax="42">
          42 42 42 42 42
        </DataArray>
        <DataArray type="Int64" Name="faces" format="ascii" RangeMin="0" RangeMax="47">
          18 4 12 16 4 0
          4 14 18 6 2 4
          16 20 8 4 4 17
          21 9 5 4 24 28
          16 12 4 25 29 17
          13 4 12 0 1 13
          4 13 1 2 14 4
          17 5 6 18 4 20
          8 9 21 4 24 12
          13 25 4 28 16 17
          29 4 0 4 5 1
          4 1 5 6 2 4
          4 8 9 5 4 13
          17 18 14 4 16 20
          21 17 4 24 28 29
          25 14 4 14 18 6
          2 4 15 19 7 3
          4 17 21 9 5 4
          19 23 11 7 4 14
          2 3 15 4 17 5
          6 18 4 21 9 10
          22 4 22 10 11 23
          4 2 6 7 3 4
          5 9 10 6 4 6
          10 11 7 4 14 18
          19 15 4 17 21 22
          18 4 18 22 23 19
          16 4 25 29 17 13
          4 27 31 19 15 4
          29 33 21 17 4 31
          35 23 19 4 25 13
          14 26 4 26 14 15
          27 4 33 21 22 34
          4 34 22 23 35 4
          13 17 18 14 4 14
          18 19 15 4 17 21
          22 18 4 18 22 23
          19 4 25 29 30 26
          4 26 30 31 27 4
          29 33 34 30 4 30
          34 35 31 6 4 28
          32 20 16 4 29 33
          21 17 4 28 16 17
          29 4 32 20 21 33
          4 16 20 21 17 4
          28 32 33 29 22 4
          36 40 28 24 4 39
          43 31 27 4 40 44
          32 28 4 43 47 35
          31 4 36 24 25 37
          4 37 25 26 38 4
          38 26 27 39 4 44
          32 33 45 4 45 33
          34 46 4 46 34 35
          47 4 24 28 29 25
          4 25 29 30 26 4
          26 30 31 27 4 28
          32 33 29 4 29 33
          34 30 4 30 34 35
          31 4 36 40 41 37
          4 37 41 42 38 4
          38 42 43 39 4 40
          44 45 41 4 41 45
          46 42 4 42 46 47
          43
        </DataArray>
        <DataArray type="Int64" Name="faceoffsets" format="ascii" RangeMin="91" RangeMax="385">
          91 162 243 274 385
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    # ------------------------------------------------------------------------------#

    def _gb_1_grid_pvd(self):
        return """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
<Collection>
\t<DataSet group="" part="" file="grid_1.vtu"/>
\t<DataSet group="" part="" file="grid_2.vtu"/>
\t<DataSet group="" part="" file="grid_mortar_1.vtu"/>
</Collection>
</VTKFile>"""

    # ------------------------------------------------------------------------------#

    def _gb_1_grid_1_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="5" NumberOfCells="4">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="1.7320508076" RangeMax="1.7320508076">
          1 1 1 1 1 1
          1 1 1 1 1 1
        </DataArray>
        <DataArray type="Int32" Name="grid_dim" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="3">
          0 1 2 3
        </DataArray>
        <DataArray type="Int32" Name="grid_node_number" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1
        </DataArray>
        <DataArray type="Int8" Name="is_mortar" format="ascii" RangeMin="0" RangeMax="0">
          0 0 0 0
        </DataArray>
        <DataArray type="Int32" Name="mortar_side" format="ascii" RangeMin="0" RangeMax="0">
          0 0 0 0
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0.5" RangeMax="1.1180339887">
          1 0.5 -5.5511151231e-17 0.75 0.5 -2.7755575616e-17
          0.5 0.5 0 0.25 0.5 2.7755575616e-17
          0 0.5 5.5511151231e-17
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="4">
          0 1 1 2 2 3
          3 4
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="2" RangeMax="8">
          2 4 6 8
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="3" RangeMax="3">
          3 3 3 3
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    # ------------------------------------------------------------------------------#

    def _gb_1_grid_2_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="30" NumberOfCells="16">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="2" RangeMax="2">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="3.4641016151" RangeMax="3.4641016151">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
        </DataArray>
        <DataArray type="Int32" Name="grid_dim" format="ascii" RangeMin="2" RangeMax="2">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="15">
          0 1 2 3 4 5
          6 7 8 9 10 11
          12 13 14 15
        </DataArray>
        <DataArray type="Int32" Name="grid_node_number" format="ascii" RangeMin="0" RangeMax="0">
          0 0 0 0 0 0
          0 0 0 0 0 0
          0 0 0 0
        </DataArray>
        <DataArray type="Int8" Name="is_mortar" format="ascii" RangeMin="0" RangeMax="0">
          0 0 0 0 0 0
          0 0 0 0 0 0
          0 0 0 0
        </DataArray>
        <DataArray type="Int32" Name="mortar_side" format="ascii" RangeMin="0" RangeMax="0">
          0 0 0 0 0 0
          0 0 0 0 0 0
          0 0 0 0
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0" RangeMax="1.4142135624">
          0 0 0 0.25 0 0
          0.5 0 0 0.75 0 0
          1 0 0 0 0.25 0
          0.25 0.25 0 0.5 0.25 0
          0.75 0.25 0 1 0.25 0
          0 0.5 0 0 0.5 0
          0.25 0.5 0 0.25 0.5 0
          0.5 0.5 0 0.5 0.5 0
          0.75 0.5 0 0.75 0.5 0
          1 0.5 0 1 0.5 0
          0 0.75 0 0.25 0.75 0
          0.5 0.75 0 0.75 0.75 0
          1 0.75 0 0 1 0
          0.25 1 0 0.5 1 0
          0.75 1 0 1 1 0
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="29">
          0 5 6 1 1 6
          7 2 2 7 8 3
          3 8 9 4 5 10
          12 6 6 12 14 7
          7 14 16 8 8 16
          18 9 11 20 21 13
          13 21 22 15 15 22
          23 17 17 23 24 19
          20 25 26 21 21 26
          27 22 22 27 28 23
          23 28 29 24
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="4" RangeMax="64">
          4 8 12 16 20 24
          28 32 36 40 44 48
          52 56 60 64
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="7" RangeMax="7">
          7 7 7 7 7 7
          7 7 7 7 7 7
          7 7 7 7
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    # ------------------------------------------------------------------------------#

    def _gb_1_mortar_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="10" NumberOfCells="8">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Int32" Name="grid_dim" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1 1 1
          1 1
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="3">
          0 1 2 3 0 1
          2 3
        </DataArray>
        <DataArray type="Int32" Name="grid_edge_number" format="ascii" RangeMin="0" RangeMax="0">
          0 0 0 0 0 0
          0 0
        </DataArray>
        <DataArray type="Int8" Name="is_mortar" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1 1 1
          1 1
        </DataArray>
        <DataArray type="Int32" Name="mortar_side" format="ascii" RangeMin="1" RangeMax="2">
          1 1 1 1 2 2
          2 2
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0.5" RangeMax="1.1180339887">
          1 0.5 -5.5511151231e-17 0.75 0.5 -2.7755575616e-17
          0.5 0.5 0 0.25 0.5 2.7755575616e-17
          0 0.5 5.5511151231e-17 1 0.5 -5.5511151231e-17
          0.75 0.5 -2.7755575616e-17 0.5 0.5 0
          0.25 0.5 2.7755575616e-17 0 0.5 5.5511151231e-17
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="9">
          0 1 1 2 2 3
          3 4 5 6 6 7
          7 8 8 9
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="2" RangeMax="16">
          2 4 6 8 10 12
          14 16
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="3" RangeMax="3">
          3 3 3 3 3 3
          3 3
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

#------------------------------------------------------------------------------#

    def _gb_1_mortar_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="10" NumberOfCells="8">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Int32" Name="grid_dim" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1 1 1
          1 1
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="3">
          0 1 2 3 0 1
          2 3
        </DataArray>
        <DataArray type="Int32" Name="grid_edge_number" format="ascii" RangeMin="0" RangeMax="0">
          0 0 0 0 0 0
          0 0
        </DataArray>
        <DataArray type="Int8" Name="is_mortar" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1 1 1
          1 1
        </DataArray>
        <DataArray type="Int32" Name="mortar_side" format="ascii" RangeMin="1" RangeMax="2">
          1 1 1 1 2 2
          2 2
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0.5" RangeMax="1.1180339887">
          1 0.5 -5.5511151231e-17 0.75 0.5 -2.7755575616e-17
          0.5 0.5 0 0.25 0.5 2.7755575616e-17
          0 0.5 5.5511151231e-17 1 0.5 -5.5511151231e-17
          0.75 0.5 -2.7755575616e-17 0.5 0.5 0
          0.25 0.5 2.7755575616e-17 0 0.5 5.5511151231e-17
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="9">
          0 1 1 2 2 3
          3 4 5 6 6 7
          7 8 8 9
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="2" RangeMax="16">
          2 4 6 8 10 12
          14 16
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="3" RangeMax="3">
          3 3 3 3 3 3
          3 3
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

#------------------------------------------------------------------------------#

    def _gb_2_grid_pvd(self):
        return """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
<Collection>
\t<DataSet group="" part="" file="grid_1.vtu"/>
\t<DataSet group="" part="" file="grid_2.vtu"/>
\t<DataSet group="" part="" file="grid_mortar_0.vtu"/>
\t<DataSet group="" part="" file="grid_mortar_1.vtu"/>
</Collection>
</VTKFile>"""

    # ------------------------------------------------------------------------------#

    def _gb_2_grid_1_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="10" NumberOfCells="6">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1 1 1
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="1.7320508076" RangeMax="1.7320508076">
          1 1 1 1 1 1
          1 1 1 1 1 1
          1 1 1 1 1 1
        </DataArray>
        <DataArray type="Int32" Name="grid_dim" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1 1 1
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="3">
          0 1 2 3 0 1
        </DataArray>
        <DataArray type="Int32" Name="grid_node_number" format="ascii" RangeMin="1" RangeMax="2">
          1 1 1 1 2 2
        </DataArray>
        <DataArray type="Int8" Name="is_mortar" format="ascii" RangeMin="0" RangeMax="0">
          0 0 0 0 0 0
        </DataArray>
        <DataArray type="Int32" Name="mortar_side" format="ascii" RangeMin="0" RangeMax="0">
          0 0 0 0 0 0
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0.5" RangeMax="1.1180339887">
          1 0.5 -5.5511151231e-17 0.75 0.5 -2.7755575616e-17
          0.5 0.5 0 0.5 0.5 0
          0.25 0.5 2.7755575616e-17 0 0.5 5.5511151231e-17
          0.5 0.75 -2.7755575616e-17 0.5 0.5 0
          0.5 0.5 0 0.5 0.25 2.7755575616e-17
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="9">
          0 1 1 2 3 4
          4 5 6 7 8 9
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="2" RangeMax="12">
          2 4 6 8 10 12
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="3" RangeMax="3">
          3 3 3 3 3 3
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    # ------------------------------------------------------------------------------#

    def _gb_2_grid_2_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="32" NumberOfCells="16">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="2" RangeMax="2">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="3.4641016151" RangeMax="3.4641016151">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
        </DataArray>
        <DataArray type="Int32" Name="grid_dim" format="ascii" RangeMin="2" RangeMax="2">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="15">
          0 1 2 3 4 5
          6 7 8 9 10 11
          12 13 14 15
        </DataArray>
        <DataArray type="Int32" Name="grid_node_number" format="ascii" RangeMin="0" RangeMax="0">
          0 0 0 0 0 0
          0 0 0 0 0 0
          0 0 0 0
        </DataArray>
        <DataArray type="Int8" Name="is_mortar" format="ascii" RangeMin="0" RangeMax="0">
          0 0 0 0 0 0
          0 0 0 0 0 0
          0 0 0 0
        </DataArray>
        <DataArray type="Int32" Name="mortar_side" format="ascii" RangeMin="0" RangeMax="0">
          0 0 0 0 0 0
          0 0 0 0 0 0
          0 0 0 0
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0" RangeMax="1.4142135624">
          0 0 0 0.25 0 0
          0.5 0 0 0.75 0 0
          1 0 0 0 0.25 0
          0.25 0.25 0 0.5 0.25 0
          0.75 0.25 0 1 0.25 0
          0 0.5 0 0 0.5 0
          0.25 0.5 0 0.25 0.5 0
          0.5 0.5 0 0.5 0.5 0
          0.5 0.5 0 0.5 0.5 0
          0.75 0.5 0 0.75 0.5 0
          1 0.5 0 1 0.5 0
          0 0.75 0 0.25 0.75 0
          0.5 0.75 0 0.75 0.75 0
          1 0.75 0 0 1 0
          0.25 1 0 0.5 1 0
          0.75 1 0 1 1 0
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="31">
          0 5 6 1 1 6
          7 2 2 7 8 3
          3 8 9 4 5 10
          12 6 6 12 14 7
          7 15 18 8 8 18
          20 9 11 22 23 13
          13 23 24 16 17 24
          25 19 19 25 26 21
          22 27 28 23 23 28
          29 24 24 29 30 25
          25 30 31 26
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="4" RangeMax="64">
          4 8 12 16 20 24
          28 32 36 40 44 48
          52 56 60 64
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="7" RangeMax="7">
          7 7 7 7 7 7
          7 7 7 7 7 7
          7 7 7 7
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

#------------------------------------------------------------------------------#

    def _gb_2_mortar_grid_1_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="20" NumberOfCells="12">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Int32" Name="grid_dim" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1 1 1
          1 1 1 1 1 1
        </DataArray>
        <DataArray type="Int32" Name="cell_id" format="ascii" RangeMin="0" RangeMax="3">
          0 1 2 3 0 1
          2 3 0 1 0 1
        </DataArray>
        <DataArray type="Int32" Name="grid_edge_number" format="ascii" RangeMin="0" RangeMax="1">
          0 0 0 0 0 0
          0 0 1 1 1 1
        </DataArray>
        <DataArray type="Int8" Name="is_mortar" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1 1 1
          1 1 1 1 1 1
        </DataArray>
        <DataArray type="Int32" Name="mortar_side" format="ascii" RangeMin="1" RangeMax="2">
          1 1 1 1 2 2
          2 2 1 1 2 2
        </DataArray>
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0.5" RangeMax="1.1180339887">
          1 0.5 -5.5511151231e-17 0.75 0.5 -2.7755575616e-17
          0.5 0.5 0 0.5 0.5 0
          0.25 0.5 2.7755575616e-17 0 0.5 5.5511151231e-17
          1 0.5 -5.5511151231e-17 0.75 0.5 -2.7755575616e-17
          0.5 0.5 0 0.5 0.5 0
          0.25 0.5 2.7755575616e-17 0 0.5 5.5511151231e-17
          0.5 0.75 -2.7755575616e-17 0.5 0.5 0
          0.5 0.5 0 0.5 0.25 2.7755575616e-17
          0.5 0.75 -2.7755575616e-17 0.5 0.5 0
          0.5 0.5 0 0.5 0.25 2.7755575616e-17
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="19">
          0 1 1 2 3 4
          4 5 6 7 7 8
          9 10 10 11 12 13
          14 15 16 17 18 19
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="2" RangeMax="24">
          2 4 6 8 10 12
          14 16 18 20 22 24
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="3" RangeMax="3">
          3 3 3 3 3 3
          3 3 3 3 3 3
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

#------------------------------------------------------------------------------#

if __name__ == '__main__':
    BasicsTest().test_gb_2()
    unittest.main()
