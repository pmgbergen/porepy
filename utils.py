# Various utility functions for  gridding

import numpy as np
import matplotlib.pyplot as plt

def plot_fractures(d, p, c, colortag=None):
    """
    Plot fractures as lines in a domain

    d: domain size in the form of a dictionary
    p - points
    c - connection between fractures
    colortag - indicate that fractures should have different colors
    """
    
    # Assign a color to each tag. We define these by RBG-values (simplest option in pyplot).
    # For the moment, some RBG values are hard coded, do something more intelligent if necessary.
    if colortag is None:
        tagmap = np.zeros(c.shape[1], dtype='int')
        col = [(0, 0, 0)];
    else:
        utag, tagmap = np.unique(colortag, return_inverse=True)
        ntag = utag.size
        if ntag <= 3:
            col = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        elif ntag < 6:
            col = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (1, 1, 0), (1, 0, 1), (0, 0, 1)]
        else:
            raise NotImplementedError('Have not thought of more than six colors')
    
    
    plt.figure()
    # Simple for-loop to draw one fracture after another. Not fancy, but it serves its purpose.
    for i in range(c.shape[1]):
        plt.plot([p[0, c[0, i]], p[0, c[1, i]]], [p[1, c[0, i]], p[1, c[1, i]]], 'o-',color=col[tagmap[i]])
    plt.axis([d['xmin'], d['xmax'], d['ymin'], d['ymax']])
    plt.show()
