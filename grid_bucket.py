import numpy as np

from graph_tool.all import *
# https://graph-tool.skewed.de/
# conda install -c ostrokach graph-tool=2.19

class Grid_Bucket(object):

#------------------------------------------------------------------------------#

    def __init__(self):
        self.graph = Graph(directed=False)
        self.grids = self.graph.new_vertex_property("object")
        self.face_cells = self.graph.new_edge_property("object")

#------------------------------------------------------------------------------#

    def add_grids(self, grids):
        assert not np.any([ i is j for i in grids for j in self.grids ])
        for g in grids: self.grids[self.graph.add_vertex()] = g

#------------------------------------------------------------------------------#

    def add_relations(self, grids, face_cells):
        [ self.add_relation(g, f) for g,f in zip(grids, face_cells) ]

#------------------------------------------------------------------------------#

    def add_relation(self, grids, face_cells):
        assert np.asarray(grids).size == 2
        assert grids[0].dim == grids[1].dim-1 or grids[1].dim == grids[0].dim-1

        v = [ self.__find(g) for g in grids ]
        assert not self.graph.edge(*v)
        self.face_cells[self.graph.add_edge(*v)] = face_cells

#------------------------------------------------------------------------------#

    def __find(self, grid):
        vertex_id = np.where([ grid is g for g in self.grids ])[0]
        assert vertex_id.size == 1
        return self.graph.vertex(vertex_id)

#------------------------------------------------------------------------------#
