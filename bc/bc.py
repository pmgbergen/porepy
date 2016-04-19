# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 14:30:22 2016

@author: eke001
"""

import numpy as np

from utils import accumarray

class BoundaryCondition(object):
    """ Class to store information on boundary conditions. 
    
        The BCs are specified by face number, and can have type Dirichlet or 
        Neumann (Robin may be included later).
        The face-based indexing makes this natural for control volume methods.
        For vector equations, it is ismplicitly assumed that the 
        
    """
    
    """ Initialize boundary conditions
        
        Note: By default, all boundaries
        
    """
    def __init__(self, g, faces=None, cond=None):
        
        
        self.Nf = g.Nf
        self.dim = g.dim - 1

        # Find boundary faces
        fi, ci = g.cellFaces.nonzero()
        fcnt = accumarray.accum(fi, np.ones(fi.size))
        bf = np.argwhere(fcnt == 1).ravel()
        
        self.isbnd = np.zeros(self.Nf, dtype=bool)
        self.isDir = np.zeros(self.Nf, dtype=bool)

        self.isbnd[bf] = True
        self.isNeu = self.isbnd

        if faces is not None:
            # Validate arguments
            assert cond is not None 
            if not np.all(np.in1d(faces,bf)):
                raise ValueError('Give boundary condition only on the boundary')
            if faces.size != len(cond):
                raise ValueError('One BC per face')

            
            for l in range(len(cond)):
                s = cond[l]
                if s.lower() == 'neu':
                    pass  # Neumann is already default
                elif s.lower() == 'dir':
                    self.isDir[faces[l]] = True
                    self.isNeu[faces[l]] = False
                else:
                    raise ValueError('Boundary should be Dirichlet or Neumann')
                
            
        