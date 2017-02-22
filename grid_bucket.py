import numpy as np

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from graph_tool.all import *
# https://graph-tool.skewed.de/
# conda install -c ostrokach graph-tool=2.19

class Grid_Bucket(object):

#------------------------------------------------------------------------------#

    def __init__(self):
        self.graph = Graph(directed=False)
        self.grids = self.g_prop()
        self.face_cells = self.e_prop()
        self.size = 0

#------------------------------------------------------------------------------#

    def __iter__(self):
        self.i = 0
        return self

#------------------------------------------------------------------------------#

    def __next__(self):
        if self.i < self.graph.num_vertices():
            v = self.graph.vertex(self.i)
            self.i += 1
            return self.grids[v], v
        else:
            self.i = 0
            raise StopIteration()

#------------------------------------------------------------------------------#

    def v(self): return self.graph.vertices()

#------------------------------------------------------------------------------#

    def e(self): return self.graph.edges()

#------------------------------------------------------------------------------#

    def v_of_e(self, e): return e.source(), e.target()

#------------------------------------------------------------------------------#

    def g_prop(self, prop=None, property_type="object"):
        g_prop = self.graph.new_vertex_property(property_type)
        if prop is not None:
            for v, p in zip(self.v(), prop): g_prop[v] = p
        return g_prop

#------------------------------------------------------------------------------#

    def e_prop(self, prop=None, property_type="object"):
        e_prop = self.graph.new_edge_property(property_type)
        if prop is not None:
            for e, p in zip(self.e(), prop): e_prop[e] = p
        return e_prop

#------------------------------------------------------------------------------#

    def add_grids(self, grids):
        assert not np.any([ i is j for i in grids for j in self.grids ])
        v = np.empty(np.asarray(grids).size, dtype=object)
        self.size += np.asarray(grids).size
        for idx, g in enumerate(grids):
            v[idx] = self.graph.add_vertex()
            self.grids[v[idx]] = g
        return v

#------------------------------------------------------------------------------#

    def add_relations(self, grids, face_cells):
        [ self.add_relation(g, f) for g,f in zip(grids, face_cells) ]

#------------------------------------------------------------------------------#

    def add_relation(self, grids, face_cells):
        assert np.asarray(grids).size == 2
        assert grids[0].dim == grids[1].dim-1 or grids[1].dim == grids[0].dim-1

        v = [ self.find(g) for g in grids ]
        assert not self.graph.edge(*v)
        self.face_cells[self.graph.add_edge(*v)] = face_cells

#------------------------------------------------------------------------------#

    def find(self, grid):
        vertex_id = np.where([ grid is g for g in self.grids ])[0]
        assert vertex_id.size == 1
        return self.graph.vertex(vertex_id)

#------------------------------------------------------------------------------#
