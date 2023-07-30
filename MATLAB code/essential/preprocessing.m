
% Function that converts test data to fit the input shape of the model
function D_merge_data = preprocessing(test_data,input_shape,n_fft,win_length)
    D = max(size(input_shape));                                                         % The model's input shape
    
    Quotient = floor(max(size(test_data)) / D);                                         % Operation to know the difference between test data and input size
    add_number = (Quotient+1) * D - max(size(test_data));                               % The amount of data to be added by the difference
    add_zeros = zeros(add_number, 1);                                                   % Zero data added as much as the difference
    test_data = vertcat(test_data, add_zeros);                                          % Allocate the required size for test data
    
    test_data_cell = cell(Quotient+1,1);                                                % Cell to save test data in cell format
    for i = 1:(Quotient+1)                                                              % Allocate data to each cell through a loop statement
        start_idx = (i - 1) * D + 1;
        end_idx = start_idx + D - 1;
        test_data_cell{i} = stft(test_data(start_idx:end_idx), "Window", hamming(win_length), ...
        'OverlapLength', win_length * 3 / 4, "FFTLength", n_fft);                       % STFT conversion for each allocated cell
    end

    D_merge_data = test_data_cell;
end