
% Function to convert polar coordinates to complex numbers
function result_array = polar_to_complex(array)
    r = array(:, :, 1);                             % Assign distance
    theta = array(:, :, 2);                         % Assign angle
    real_part = r.*cos(theta);                      % Calculate the real part
    imag_part = r.*sin(theta);                      % Calculate the imaginary part
    
    result_array = complex(real_part, imag_part);   % Assign complex number
end