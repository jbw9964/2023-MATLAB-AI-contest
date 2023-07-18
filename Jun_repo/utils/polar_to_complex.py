
__all__ = ["polar_to_complex"]

from numpy import cos, sin

def polar_to_complex(array) : 
    r = array[..., 0]
    theta = array[..., 1]

    real_part = r * cos(theta)
    imag_part = r * sin(theta)
    
    complex_array = real_part + imag_part * 1j

    return complex_array