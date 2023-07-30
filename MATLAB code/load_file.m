
% Function that gets the address of a file
function file_names = load_file(data_path)
    dir_info = dir(data_path);                          % Save the path of the input file as an address
    file_names_cell = cell(length(dir_info) - 2, 1);    % Create a cell to receive file names

    for i = 3:length(dir_info)
        file_names_cell{i - 2} = dir_info(i).name;      % Store the names of files in each cell
    end

    file_names = strings(numel(file_names_cell), 1);    % Create a list to store the names of files

    for i = 1:numel(file_names_cell)
        file_names{i} = file_names_cell{i};             % Store the saved names of cells in a list
    end
end