import numpy as np
import unittest
import filecmp

from porepy.grids import structured, simplex
from porepy.fracs import meshing
from porepy.grids import coarsening as co

from porepy.viz.exporter import Exporter

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_single_grid_1d(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        np.random.seed(0)
        dummy_scalar = np.random.rand(g.num_cells)
        dummy_vector = np.random.rand(3, g.num_cells)

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(g, file_name, folder, binary=False)
        save.write_vtk({'dummy_scalar': dummy_scalar,
                        'dummy_vector': dummy_vector})

        with open(folder+file_name+".vtu", 'r') as content_file:
            content = content_file.read()
        assert content == self._single_grid_1d_grid_vtu()

#------------------------------------------------------------------------------#

    def test_single_grid_2d_simplex(self):
        g = simplex.StructuredTriangleGrid([3]*2, [1]*2)
        g.compute_geometry()

        np.random.seed(0)
        dummy_scalar = np.random.rand(g.num_cells)
        dummy_vector = np.random.rand(g.num_cells, 3)

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(g, file_name, folder, binary=False)
        save.write_vtk({'dummy_scalar': dummy_scalar,
                        'dummy_vector': dummy_vector})

        with open(folder+file_name+".vtu", 'r') as content_file:
            content = content_file.read()
        assert content == self._single_grid_2d_simplex_grid_vtu()

#------------------------------------------------------------------------------#

    def test_single_grid_2d_cart(self):
        g = structured.CartGrid([4]*2, [1]*2)
        g.compute_geometry()

        np.random.seed(0)
        dummy_scalar = np.random.rand(g.num_cells)
        dummy_vector = np.random.rand(3, g.num_cells)

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(g, file_name, folder, binary=False)
        save.write_vtk({'dummy_scalar': dummy_scalar,
                        'dummy_vector': dummy_vector})

        with open(folder+file_name+".vtu", 'r') as content_file:
            content = content_file.read()
        assert content == self._single_grid_2d_cart_grid_vtu()

#------------------------------------------------------------------------------#

    def test_single_grid_2d_polytop(self):
        g = structured.CartGrid([3, 2], [1]*2)
        g.compute_geometry()
        co.generate_coarse_grid(g, [0, 0, 1, 0, 1, 1])
        g.compute_geometry()

        np.random.seed(0)
        dummy_scalar = np.random.rand(g.num_cells)
        dummy_vector = np.random.rand(3, g.num_cells)

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(g, file_name, folder, binary=False)
        save.write_vtk({'dummy_scalar': dummy_scalar,
                        'dummy_vector': dummy_vector})

        with open(folder+file_name+".vtu", 'r') as content_file:
            content = content_file.read()
        assert content == self._single_grid_2d_polytop_grid_vtu()

#------------------------------------------------------------------------------#

    def test_single_grid_3d_simplex(self):
        g = simplex.StructuredTetrahedralGrid([5]*3, [1]*3)
        g.compute_geometry()

        np.random.seed(0)
        dummy_scalar = np.random.rand(g.num_cells)
        dummy_vector = np.random.rand(3, g.num_cells)

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(g, file_name, folder, binary=False)
        save.write_vtk({'dummy_scalar': dummy_scalar,
                        'dummy_vector': dummy_vector})

        with open(folder+file_name+".vtu", 'r') as content_file:
            content = content_file.read()
#        assert content == self._single_grid_3d_simplex_grid_vtu()

#------------------------------------------------------------------------------#

    def test_single_grid_3d_cart(self):
        pass

#------------------------------------------------------------------------------#

    def test_single_grid_3d_polytop(self):
        pass

#------------------------------------------------------------------------------#

    def test_gb_1(self):
        f1 = np.array([[0, 1], [.5, .5]])
        gb = meshing.cart_grid([f1], [4]*2, **{'physdims': [1, 1]})
        gb.compute_geometry()

        gb.add_node_props(['scalar_dummy', 'dummy_vector'])
        np.random.seed(0)

        for g, d in gb:
            d['dummy_scalar'] = np.random.rand(g.num_cells)
            d['dummy_vector'] = np.random.rand(3, g.num_cells)

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(gb, file_name, folder, binary=False)
        save.write_vtk(['dummy_scalar', 'dummy_vector'])

        with open(folder+file_name+".pvd", 'r') as content_file:
            content = content_file.read()
        assert content == self._gb_1_grid_pvd()

        with open(folder+file_name+"_1.vtu", 'r') as content_file:
            content = content_file.read()
        assert content == self._gb_1_grid_1_vtu()

        with open(folder+file_name+"_2.vtu", 'r') as content_file:
            content = content_file.read()
        assert content == self._gb_1_grid_2_vtu()

#------------------------------------------------------------------------------#

    def test_gb_2(self):
        f1 = np.array([[0, 1], [.5, .5]])
        f2 = np.array([[.5, .5], [.25, .75]])
        gb = meshing.cart_grid([f1, f2], [4]*2, **{'physdims': [1, 1]})
        gb.compute_geometry()

        gb.add_node_props(['dummy_scalar', 'dummy_vector'])
        np.random.seed(0)

        for g, d in gb:
            d['dummy_scalar'] = np.random.rand(g.num_cells)
            d['dummy_vector'] = np.random.rand(3, g.num_cells)

        folder = "./test_vtk/"
        file_name = "grid"
        save = Exporter(gb, file_name, folder, binary=False)
        save.write_vtk(['dummy_scalar', 'dummy_vector'])

        with open(folder+file_name+".pvd", 'r') as content_file:
            content = content_file.read()
        assert content == self._gb_2_grid_pvd()

        with open(folder+file_name+"_1.vtu", 'r') as content_file:
            content = content_file.read()
        assert content == self._gb_2_grid_1_vtu()

        with open(folder+file_name+"_2.vtu", 'r') as content_file:
            content = content_file.read()
        assert content == self._gb_2_grid_2_vtu()

#------------------------------------------------------------------------------#

    def _single_grid_1d_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="3">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="0.54881350393" RangeMax="0.71518936637">
          0.54881350393 0.71518936637 0.60276337607
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="0.79712461318" RangeMax="1.274972532">
          0.544883183 0.43758721126 0.38344151883 0.42365479934 0.89177300078 0.79172503808
          0.64589411307 0.9636627605 0.52889491975
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1
        </DataArray>
        <DataArray type="Float64" Name="cell_id" format="ascii" RangeMin="0" RangeMax="2">
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

#------------------------------------------------------------------------------#

    def _single_grid_2d_simplex_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="16" NumberOfCells="18">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="0.02021839744" RangeMax="0.9636627605">
          0.54881350393 0.71518936637 0.60276337607 0.544883183 0.42365479934 0.64589411307
          0.43758721126 0.89177300078 0.9636627605 0.38344151883 0.79172503808 0.52889491975
          0.56804456109 0.92559663829 0.071036058198 0.087129299702 0.02021839744 0.83261984555
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="0.56967075244" RangeMax="1.5082787475">
          0.77815675095 0.61209572272 0.20887675609 0.87001214825 0.61693399687 0.16130951788
          0.97861834223 0.94374807851 0.65310832547 0.79915856422 0.6818202991 0.25329160254
          0.46147936225 0.35950790057 0.46631077286 0.78052917629 0.4370319538 0.244425592
          0.11827442587 0.69763119593 0.15896958365 0.63992102133 0.060225471629 0.11037514116
          0.14335328741 0.66676671545 0.65632958947 0.94466891705 0.67063786962 0.13818295135
          0.52184832175 0.21038256107 0.19658236168 0.41466193999 0.12892629765 0.36872517066
          0.2645556121 0.31542835092 0.82099322985 0.77423368943 0.36371077094 0.097101275793
          0.45615033222 0.57019677042 0.8379449075 0.56843394887 0.43860151346 0.096098407894
          0.018789800436 0.98837383806 0.97645946501 0.61763549708 0.10204481075 0.46865120165
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="2" RangeMax="2">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2 2 2
        </DataArray>
        <DataArray type="Float64" Name="cell_id" format="ascii" RangeMin="0" RangeMax="17">
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

#------------------------------------------------------------------------------#

    def _single_grid_2d_cart_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="25" NumberOfCells="16">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="0.071036058198" RangeMax="0.9636627605">
          0.54881350393 0.71518936637 0.60276337607 0.544883183 0.42365479934 0.64589411307
          0.43758721126 0.89177300078 0.9636627605 0.38344151883 0.79172503808 0.52889491975
          0.56804456109 0.92559663829 0.071036058198 0.087129299702
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="0.5549567134" RangeMax="1.5196176747">
          0.02021839744 0.45615033222 0.31542835092 0.83261984555 0.56843394887 0.36371077094
          0.77815675095 0.018789800436 0.57019677042 0.87001214825 0.61763549708 0.43860151346
          0.97861834223 0.61209572272 0.98837383806 0.79915856422 0.61693399687 0.10204481075
          0.46147936225 0.94374807851 0.20887675609 0.78052917629 0.6818202991 0.16130951788
          0.11827442587 0.35950790057 0.65310832547 0.63992102133 0.4370319538 0.25329160254
          0.14335328741 0.69763119593 0.46631077286 0.94466891705 0.060225471629 0.244425592
          0.52184832175 0.66676671545 0.15896958365 0.41466193999 0.67063786962 0.11037514116
          0.2645556121 0.21038256107 0.65632958947 0.77423368943 0.12892629765 0.13818295135
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="2" RangeMax="2">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2
        </DataArray>
        <DataArray type="Float64" Name="cell_id" format="ascii" RangeMin="0" RangeMax="15">
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

#------------------------------------------------------------------------------#

    def _single_grid_2d_polytop_grid_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="12" NumberOfCells="2">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="0.54881350393" RangeMax="0.71518936637">
          0.54881350393 0.71518936637
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="0.85690702179" RangeMax="1.2285503544">
          0.60276337607 0.42365479934 0.43758721126 0.544883183 0.64589411307 0.89177300078
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="2" RangeMax="2">
          2 2
        </DataArray>
        <DataArray type="Float64" Name="cell_id" format="ascii" RangeMin="0" RangeMax="1">
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

#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#

    def _gb_1_grid_pvd(self):
        return """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
<Collection>
	<DataSet group="" part="" file="grid_1.vtu"/>
	<DataSet group="" part="" file="grid_2.vtu"/>
</Collection>
</VTKFile>"""

#------------------------------------------------------------------------------#

    def _gb_1_grid_1_vtu(self):
        return \
"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="5" NumberOfCells="4">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="0.097101275793" RangeMax="0.82099322985">
          0.19658236168 0.36872517066 0.82099322985 0.097101275793
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="0.48504216633" RangeMax="1.3176470201">
          0.8379449075 0.97676108819 0.28280696258 0.096098407894 0.60484551975 0.12019656121
          0.97645946501 0.7392635794 0.29614019752 0.46865120165 0.039187792254 0.11872771895
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1
        </DataArray>
        <DataArray type="Float64" Name="cell_id" format="ascii" RangeMin="0" RangeMax="3">
          0 1 2 3
        </DataArray>
        <DataArray type="Float64" Name="grid_node_number" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1
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

#------------------------------------------------------------------------------#

    def _gb_1_grid_2_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="30" NumberOfCells="16">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="0.071036058198" RangeMax="0.9636627605">
          0.54881350393 0.71518936637 0.60276337607 0.544883183 0.42365479934 0.64589411307
          0.43758721126 0.89177300078 0.9636627605 0.38344151883 0.79172503808 0.52889491975
          0.56804456109 0.92559663829 0.071036058198 0.087129299702
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="0.5549567134" RangeMax="1.5196176747">
          0.02021839744 0.45615033222 0.31542835092 0.83261984555 0.56843394887 0.36371077094
          0.77815675095 0.018789800436 0.57019677042 0.87001214825 0.61763549708 0.43860151346
          0.97861834223 0.61209572272 0.98837383806 0.79915856422 0.61693399687 0.10204481075
          0.46147936225 0.94374807851 0.20887675609 0.78052917629 0.6818202991 0.16130951788
          0.11827442587 0.35950790057 0.65310832547 0.63992102133 0.4370319538 0.25329160254
          0.14335328741 0.69763119593 0.46631077286 0.94466891705 0.060225471629 0.244425592
          0.52184832175 0.66676671545 0.15896958365 0.41466193999 0.67063786962 0.11037514116
          0.2645556121 0.21038256107 0.65632958947 0.77423368943 0.12892629765 0.13818295135
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="2" RangeMax="2">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2
        </DataArray>
        <DataArray type="Float64" Name="cell_id" format="ascii" RangeMin="0" RangeMax="15">
          0 1 2 3 4 5
          6 7 8 9 10 11
          12 13 14 15
        </DataArray>
        <DataArray type="Float64" Name="grid_node_number" format="ascii" RangeMin="0" RangeMax="0">
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

#------------------------------------------------------------------------------#

    def _gb_2_grid_pvd(self):
        return """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
<Collection>
	<DataSet group="" part="" file="grid_1.vtu"/>
	<DataSet group="" part="" file="grid_2.vtu"/>
</Collection>
</VTKFile>"""

#------------------------------------------------------------------------------#

    def _gb_2_grid_1_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="10" NumberOfCells="6">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="0.097101275793" RangeMax="0.82099322985">
          0.19658236168 0.36872517066 0.82099322985 0.097101275793 0.31798317939 0.41426299451
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="0.48504216633" RangeMax="1.3176470201">
          0.8379449075 0.97676108819 0.28280696258 0.096098407894 0.60484551975 0.12019656121
          0.97645946501 0.7392635794 0.29614019752 0.46865120165 0.039187792254 0.11872771895
          0.064147496349 0.56660145421 0.52324805347 0.69247211937 0.26538949094 0.093940510758
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="1" RangeMax="1">
          1 1 1 1 1 1
        </DataArray>
        <DataArray type="Float64" Name="cell_id" format="ascii" RangeMin="0" RangeMax="3">
          0 1 2 3 0 1
        </DataArray>
        <DataArray type="Float64" Name="grid_node_number" format="ascii" RangeMin="1" RangeMax="2">
          1 1 1 1 2 2
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

#------------------------------------------------------------------------------#

    def _gb_2_grid_2_vtu(self):
        return """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="32" NumberOfCells="16">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="dummy_scalar" format="ascii" RangeMin="0.071036058198" RangeMax="0.9636627605">
          0.54881350393 0.71518936637 0.60276337607 0.544883183 0.42365479934 0.64589411307
          0.43758721126 0.89177300078 0.9636627605 0.38344151883 0.79172503808 0.52889491975
          0.56804456109 0.92559663829 0.071036058198 0.087129299702
        </DataArray>
        <DataArray type="Float64" Name="dummy_vector" NumberOfComponents="3" format="ascii" RangeMin="0.5549567134" RangeMax="1.5196176747">
          0.02021839744 0.45615033222 0.31542835092 0.83261984555 0.56843394887 0.36371077094
          0.77815675095 0.018789800436 0.57019677042 0.87001214825 0.61763549708 0.43860151346
          0.97861834223 0.61209572272 0.98837383806 0.79915856422 0.61693399687 0.10204481075
          0.46147936225 0.94374807851 0.20887675609 0.78052917629 0.6818202991 0.16130951788
          0.11827442587 0.35950790057 0.65310832547 0.63992102133 0.4370319538 0.25329160254
          0.14335328741 0.69763119593 0.46631077286 0.94466891705 0.060225471629 0.244425592
          0.52184832175 0.66676671545 0.15896958365 0.41466193999 0.67063786962 0.11037514116
          0.2645556121 0.21038256107 0.65632958947 0.77423368943 0.12892629765 0.13818295135
        </DataArray>
        <DataArray type="Float64" Name="grid_dim" format="ascii" RangeMin="2" RangeMax="2">
          2 2 2 2 2 2
          2 2 2 2 2 2
          2 2 2 2
        </DataArray>
        <DataArray type="Float64" Name="cell_id" format="ascii" RangeMin="0" RangeMax="15">
          0 1 2 3 4 5
          6 7 8 9 10 11
          12 13 14 15
        </DataArray>
        <DataArray type="Float64" Name="grid_node_number" format="ascii" RangeMin="0" RangeMax="0">
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


BasicsTest().test_gb_1()
#BasicsTest().test_single_grid_2d_simplex()
