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
        data['well_cells'] = [0 1 0 2]
    with 1 indicating injection and 2 production.

    --Fracture network tags, stored in the fracture network field .tags. One
    list entry for each fracture:
        network.tags['fracture_id'] = [1,2]
    --
"""
import numpy as np


def append_tags(tags, keys, appendices):
    """
    Append tags of certain keys.
    tags:       dictionary with existing entries corresponding to
    keys:       list of keys
    appendices: list of values to be appended, typicall numpy arrays
    """
    for i, key in enumerate(keys):
        tags[key] = np.append(tags[key], appendices[i])


def standard_face_tags():
    """
    Returns the three standard face tag keys.
    """
    return ["fracture_faces", "tip_faces", "domain_boundary_faces"]

def standard_node_tags():
    """
    Returns the standard node tag key.
    """
    return ["fracture_nodes", "tip_nodes", "domain_boundary_nodes"]


def all_tags(parent, ft):
    """
    Return a logical array indicate which of the parent objects are
    tagged with any of the standard object tags.
    """
    return np.logical_or(np.logical_or(parent[ft[0]], parent[ft[1]]), parent[ft[2]])


def all_face_tags(parent):
    """
    Return a logical array indicate which of the parent (grid.tags) faces are
    tagged with any of the standard face tags.
    """
    return all_tags(parent, standard_face_tags())


def all_node_tags(parent):
    """
    Return a logical array indicate which of the parent (grid.nodes) nodes are
    tagged with any of the standard node tags.
    """
    return all_tags(parent, standard_node_tags())


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
    old_tags = getattr(parent, "tags", {}).copy()
    nt = dict(old_tags)
    nt.update(new_tags)
    parent.tags = nt
