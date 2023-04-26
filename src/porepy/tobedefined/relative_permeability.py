
import numpy as np


def rel_perm_brooks_corey(saturation): # ->... :
    """ """
    relative_perm = saturation**2
    return relative_perm


def second_derivative(saturation):
    """
    move it? i need the second derivative in hu 
    """
    return 2*np.ones(saturation.shape) 



