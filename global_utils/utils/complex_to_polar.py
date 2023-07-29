
__all__ = ["complex_to_polar"]

import numpy as np
from numpy import ndarray

def complex_to_polar(array : ndarray) : 
    r = np.abs(array)
    angle = np.angle(array)
    
    result_array = np.zeros(
        shape=(*array.shape, 2), 
        dtype=array[...,0].real.dtype
    )
    result_array[..., 0] = r
    result_array[..., 1] = angle
    
    return result_array