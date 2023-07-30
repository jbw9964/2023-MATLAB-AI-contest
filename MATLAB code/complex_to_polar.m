
% Function to convert complex numbers to polar coordinates
function result_array = complex_to_polar(array)
    r = abs(array);                                 % Distance
    theta = angle(array);                           % Angle
    result_array = zeros([size(array), 2], ...
        'like', real(array(:,:,1)));                % Array to store polar coordinates
    
    result_array(:, :, 1) = r;                      % Assign distance to array  
    result_array(:, :, 2) = theta;                  % Assign angle to array
end