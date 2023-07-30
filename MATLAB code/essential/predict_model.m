function predict_model(input_model_path, input_test_data_path, n_fft, win_length)
% input_model_path = Enter the file address where the model is saved
% input_test_data_path = Enter the file address where the test data is saved
% n_fft=960, win_length = 960       

    h5_file_list = load_file(input_model_path);                                                             % Call up model names
    test_data_list = load_file(input_test_data_path);                                                       % Call up test data names
    
    for i = 1:numel(h5_file_list)                                                                           % Test each model
        model_path = fullfile(input_model_path, h5_file_list{i});                                           % Route according to model
        model = load_model(model_path);                                                                     % Enter the model according to the path
        [input_shape, sample_rate] = check_input_data_size(model, h5_file_list{i}, n_fft, win_length);      % Check the input shape of the model
        
        disp([h5_file_list{i}, ' start'])
        for j = 1:numel(test_data_list)                                                                     % Call each test data
                test_path = fullfile(input_test_data_path, test_data_list{j});                              % Routing according to data
                test_data = load_data(test_path, sample_rate);                                              % Receive data path and call data
                test_data_stft = preprocessing(test_data, input_shape, n_fft, win_length);                  % Test data preprocessing
                
                predict_result_cell = cell(max(size(test_data_stft)), 1);                                   % Create empty cells to store predictions
                for k = 1: max(size(test_data_stft))
                    polar_test_data_stft = complex_to_polar(test_data_stft{k});                             % Convert complex number data to polar data
                    polar_predict_result = predict(model, polar_test_data_stft);                            % Put it in the model and check the result
                    predict_result = polar_to_complex(polar_predict_result);                                % Convert polar data to complex data
                    predict_result_cell{k} = istft(predict_result, "Window", hamming(win_length), ...
                        "OverlapLength", win_length * 3 / 4, "FFTLength", n_fft);                           % Inverse short-time Fourier transforming of prediction results
                end

                total_result = cell_to_double(predict_result_cell);                                         % Prediction result saved in double format
                save_predict_to_wav(total_result, h5_file_list{i}, test_data_list{j}, sample_rate);         % Prediction result saved in wav format
                
                disp([test_data_list{j},' is clear'])
        end
        disp([h5_file_list{i}, ' is finished'])
    end
end