function result_array = polar_to_complex(array)
    r = array(:, :, 1);
    theta = array(:, :, 2);
    real_part = r.*cos(theta);
    imag_part = r.*sin(theta);
    
    result_array = complex(real_part, imag_part);
end