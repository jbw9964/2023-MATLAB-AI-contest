
__all__ = ["complex_to_polar"]

import numpy as np
from numpy import ndarray

def complex_to_polar(array : ndarray) : 
    """Make complex-valued array to polar-coordinated array.

    If shape of input array is `(a, b, ..., c)`, the return shape will be `(a, b, ..., c, 2)`.

    The `(a, b, ..., c, 0)` element will be the radius in complex plane, 
    and `(a, b, ..., c, 1)` element will be radian angle.

    Args:
        array (ndarray): input array

    Returns:
        result_array (ndarray): polar-coordinated array
    ---
    ## Example
    ```python
    >>> A = np.array([0, 1, 2, 3])
    >>> complex_to_polar(A)
    array([[0, 0],
           [1, 0],
           [2, 0],
           [3, 0]])
    ```

    Input shape and output shape
    ```python
    >>> A = np.array([0, 1, 2, 3])
    >>> print(A.shape, complex_to_polar(A).shape)
    (4,) (4, 2)
    ```

    Complex array
    ```python
    >>> A = np.array([1, 1j, -1, -1j])
    >>> complex_to_polar(A)
    array([[ 1.        ,  0.        ],
           [ 1.        ,  1.57079633],
           [ 1.        ,  3.14159265],
           [ 1.        , -1.57079633]])
    ```

    """
    r = np.abs(array)
    angle = np.angle(array)
    
    result_array = np.zeros(shape=(*array.shape, 2), dtype=array[...,0].real.dtype)
    result_array[..., 0] = r
    result_array[..., 1] = angle
    
    return result_array
