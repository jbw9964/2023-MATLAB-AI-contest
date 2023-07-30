
% Function that loads the data by receiving the path and sample rate of the input data
function data = load_data(data_path, sample_rate)

    [test_data, actual_sample_rate] = audioread(data_path);     % Audio calls along the data path

    if actual_sample_rate ~= sample_rate                        % Change the existing sample rate of audio data to the set sample rate
        test_data = resample(test_data, sample_rate, actual_sample_rate);
    end

    if size(test_data, 2) > 1                                   % If the audio data to be loaded is in stereo type, it is converted to mono type.
        test_data = mean(test_data, 2);                         % Take the average and convert it to mono audio
    end
    
    data = test_data;
end