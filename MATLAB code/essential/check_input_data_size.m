
% Function to check the input size of the input model
function [input_shape, sample_rate] = check_input_data_size(model, model_name, n_fft, win_length)   

    model_input_size = model.Layers(1).InputSize;                           % Check the size of the input layer
    polar_to_input_size = zeros(model_input_size(1), ...
        model_input_size(2), model_input_size(3));                          % Generate same size data to check input size
    
    D_input_shape = polar_to_complex(polar_to_input_size);                  % Convert the generated data for checking the size into a complex number
    input_shape = istft(D_input_shape, "Window", hamming(win_length), ...
        "OverlapLength", win_length * 3 / 4, "FFTLength", n_fft);           % Istft conversion of complex data

    if strcmp(model_name, 'TB3SR6.h5') || strcmp(model_name, 'TB6SR6.h5')   % Set the sample rate according to the input model
        sample_rate = 60000;
    else
        sample_rate = 110000;
    end


end