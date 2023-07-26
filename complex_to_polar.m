function result_array = complex_to_polar(array)
    r = abs(array);
    theta = angle(array);
    result_array = zeros([size(array), 2], 'like', real(array(:,:,1)));
    
    result_array(:, :, 1) = r;
    result_array(:, :, 2) = theta;
end