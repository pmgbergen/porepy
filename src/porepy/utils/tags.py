"""
Methods for tag handling. The following primary applications are intended:
    --Grid tags, stored in the grids and data fields of the grid bucket.
    Geometry tags are stored in the grids, typical fields
    are cell, face or node tags, lists of length g.num_cell etc. Other
    information is storad in the data fields. The entries
    of the lists should be boolean or integers. Examples:
        g.tags['fracture_faces'] = [0, 1, 1, 0, 1, 1]
        g.tags['fracture_face_ids'] = [0, 1, 2, 0, 1, 2]

    for a grid with two immersed fractures (neighbour to faces (1 and 4) and
    (2 and 5), respectively). If the wells are located in cells 1 and 3, this
    may be tagged in the data as e.g.
        data['well_cells'] = [0 1 0 -1]

    --Fracture network tags, stored in the fracture network field .tags. One
    list entry for each fracture:
        network.tags['fracture_id'] = [1,2]
    --
"""
import numpy as np


def append_tags(tags, kws, appendices):
    for i, kw in enumerate(kws):
        tags[kw] = np.append(tags[kw], appendices[i])


def standard_face_tags():
    keys = ['fracture_faces', 'boundary_faces',
            'tip_faces', 'domain_boundary_faces']
    return keys


def extract(all_tags, indices, keys=None):
    """
    Extracts only the values of indices (e.g. a face subset) for the given
    keys. Any unspecified keys are left untouched (e.g. all node tags). If
    keys=None, the extraction is performed on all fields.
    """
    if keys is None:
        keys = all_tags.keys()
    new_tags = all_tags.copy()
    for k in keys:
        new_tags[k] = all_tags[k][indices]
    return new_tags


def add_tags(parent, new_tags):
    """
    Add new tags (as a premade dictionary) to the tags of the parent object
    (usually a grid). Values corresponding to keys existing in both
    dictionaries (parent.tags and new_tags) will be decided by those in
    new_tags.
    """
    old_tags = getattr(parent, 'tags', {}).copy()
    nt = dict(old_tags)
    nt.update(new_tags)
    parent.tags = nt

def add_node_tags_from_face_tags(gb, tag_base):
    """
    Set domain boundary tags for all nodes at at least one domain boundary
    face. The tag base should exist for all faces of all grids, and may e.g.
    be domain_boundary.
    """
    for g, _ in gb:
        nodes = g.face_nodes[:, g.tags[tag_base + '_faces']].nonzero()[0]
        t = np.zeros(g.num_nodes, dtype=bool)
        t[np.unique(nodes)] = True
        g.tags[tag_base + '_nodes'] = t