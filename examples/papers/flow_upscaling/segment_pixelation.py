#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The purpose of the module is to convert segments (fractures) from lines to
pixels, effectively moving from a vector to raster data format.

The main function of the module is pixelate.

"""
import numpy as np


class Bresenham(object):
    """ https://gist.github.com/flags/1132363
    """
    def __init__(self, start, end):
        self.start = list(start)
        self.end = list(end)
        self.path = []

        self.steep = abs(self.end[1] - self.start[1]) > abs(
            self.end[0] - self.start[0])

        if self.steep:
            self.start = self.swap(self.start[0], self.start[1])
            self.end = self.swap(self.end[0], self.end[1])

        if self.start[0] > self.end[0]:
            _x0 = int(self.start[0])
            _x1 = int(self.end[0])
            self.start[0] = _x1
            self.end[0] = _x0

            _y0 = int(self.start[1])
            _y1 = int(self.end[1])
            self.start[1] = _y1
            self.end[1] = _y0

        dx = self.end[0] - self.start[0]
        dy = abs(self.end[1] - self.start[1])
        error = 0
        derr = dy / float(dx)

        ystep = 0
        y = self.start[1]

        if self.start[1] < self.end[1]:
            ystep = 1
        else:
            ystep = -1

        for x in range(self.start[0], self.end[0] + 1):
            if self.steep:
                self.path.append((y, x))
            else:
                self.path.append((x, y))

            error += derr

            if error >= 0.5:
                y += ystep
                error -= 1.0

    def swap(self, n1, n2):
        return [n2, n1]

def points_to_cartind(p, dx):
    return (np.round((p - 0.5 * dx) / dx)).astype('int')


def pixelate(pts, segments, nx, dx, segment_tags=None):
    # Create initial pixelation
    segments = segments.T
    pts = pts.T

    cart_ind = points_to_cartind(pts, dx)
    num_segments = segments.shape[0]
    if segment_tags is None:
        segment_tags = np.ones(num_segments)
    interpolated_data = np.zeros(nx, dtype='int')
    for iter1 in range(num_segments):
        p0 = segments[iter1, 0]
        p1 = segments[iter1, 1]
        l = Bresenham([cart_ind[p0, 0], cart_ind[p0, 1]],
                      [cart_ind[p1, 0], cart_ind[p1, 1]])
        inds = np.array(l.path)
        interpolated_data[inds[:, 0], inds[:, 1]] = segment_tags[iter1]

    return interpolated_data