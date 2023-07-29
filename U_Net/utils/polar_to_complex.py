
__all__ = ["polar_to_complex"]

from numpy import cos, sin
from numpy import ndarray

def polar_to_complex(array : ndarray) : 
    """Make polar-coordinated array to complex-valued array.

    If input shape of input array is `(a, b, ..., c, 2)`, the return shape will be `(a, b, ..., c)`.

    The `(a, b, ..., c, 0)` element will be the radius in complex plane, 
    and `(a, b, ..., c, 1)` element will be radian angle.

    Args:
        array (ndarray): input array

    Returns:
        complex_array (ndarray): complex-valued array

    ---
    ## Example
    ```python
    >>> A = np.array([
    ...     [1, 0],
    ...     [5, pi]
    ... ])
    >>> polar_to_complex(A)
    array([ 1.+0.000000e+00j, -5.+6.123234e-16j])
    
    ```

    """
    r = array[..., 0]
    theta = array[..., 1]

    real_part = r * cos(theta)
    imag_part = r * sin(theta)
    
    complex_array = real_part + imag_part * 1j

    return complex_array
