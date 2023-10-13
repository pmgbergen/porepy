"""
Various method for creating grids for relatively simple fracture networks.

The module doubles as a test framework (though not unittest), and will report
on any problems if ran as a main method.

"""
import pytest
from collections import namedtuple

import numpy as np

import porepy as pp
from porepy.fracs import structured
from porepy.fracs.utils import pts_edges_to_linefractures
