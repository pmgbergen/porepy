"""
Methods for tag handling. The following primary applications are intended:
    --Grid tags, stored in the data field of the grid bucket. Typically fields
    are cell, face or node tags, lists of length g.num_cell etc. The entries
    of the lists should be boolean or integers. Examples:
        data['tags']['fracture_faces'] = [0, 1, 1, 0, 1, 1]
        data['tags']['fracture_face_ids'] = [0, 1, 2, 0, 1, 2]
    for a grid with two immersed fractures (neighbour to faces (1 and 4) and
    (2 and 5), respectively).
    --Fracture network tags, stored in the fracture network field .tags. One
    list entry for each fracture:
        network.tags['fracture_id'] = [1,2]
    --
"""
import numpy as np
import warnings
def append_tags(tags, kws, appendices):
        for i, kw in enumerate(kws):
            tags[kw] = np.append(tags[kw], appendices[i])

#------------------------------------------------------------------------------#
# The tags below are the dictionary values, i.e. to be called as f(tag['key'])
def add_face_tag(old_tags, f, new_tags):
    """
    Equivalent to or for logicals. Else: use with care.
    """
    old_tags[f] = np.amax(old_tags[f], old_tags)

def remove_face_tag(self, f, tag):
    self.face_tags[f] = np.bitwise_and(
        self.face_tags[f], np.bitwise_not(tag))

def remove_face_tag_if_tag(self, tag, if_tag):
    f = self.has_face_tag(if_tag)
    self.face_tags[f] = np.bitwise_and(
        self.face_tags[f], np.bitwise_not(tag))

def remove_face_tag_if_not_tag(self, tag, if_tag):
    f = self.has_not_face_tag(if_tag)
    self.face_tags[f] = np.bitwise_and(
        self.face_tags[f], np.bitwise_not(tag))

def has_face_tag(self, tag):
    return np.bitwise_and(self.face_tags, tag).astype(np.bool)

def has_not_face_tag(self, tag):
    return np.bitwise_not(self.has_face_tag(tag))

def has_only_face_tag(self, tag):
    return self.face_tags == tag