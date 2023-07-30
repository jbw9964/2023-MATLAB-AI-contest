
% Function to save cell data as double data
function total_result = cell_to_double(data_save_cell)                          
    total_length = max(size(data_save_cell)) * max(size(data_save_cell{1}));    % Measurement of cell data length to save as double data
    total_result = zeros(total_length, 1);                                      % Double list to store the result

    current_index = 1;
    for i = 1:max(size(data_save_cell))                                         % The data of each cell is stored in double space in order through a loop statement.
        data_length = numel(data_save_cell{i});
        total_result(current_index:current_index + data_length - 1) = double(data_save_cell{i});
        current_index = current_index + data_length;
    end

end