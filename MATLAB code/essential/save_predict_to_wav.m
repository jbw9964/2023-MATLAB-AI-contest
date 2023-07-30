
% Function that stores the results in a directory for each model
function save_predict_to_wav(complex_data, model_name, data_name, sample_rate)
    full_path = fullfile('./predict_result',model_name);            % Create path to save

    if ~exist(full_path, 'dir')                                     % If there is no path, create a file to save
        mkdir(fullfile('./predict_result', model_name))
    end

    data_name = ['predict_of_', data_name];                         % Create result data name
    filename = fullfile(full_path, data_name);                      % Create each data storage path
    
    real_part = real(complex_data);                                 % Extract real part for storage
    imag_part = imag(complex_data);                                 % Extract the imaginary part for storage
    
    max_value = max([abs(real_part(:)); abs(imag_part(:))]);        % Calculations for Normalization
    real_part_normalized = real_part / max_value;                   % Real part normalization
    imag_part_normalized = imag_part / max_value;                   % Imaginary part normalization
    
    real_int = int16(real_part_normalized * 32767);                 % Converted to a 16-bit signed integer
    imag_int = int16(imag_part_normalized * 32767);                 % Converted to a 16-bit signed integer
    
    combined_data = [real_int, imag_int];                           % Store complex data as single real data
    
    audiowrite(filename, combined_data, sample_rate);               % Save as wav file
end

