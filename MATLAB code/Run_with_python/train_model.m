
% Args
% ------------------------------------------------------------------------------------- %
% 1. model_save_path    :   path to save trained model.                                 %
% 2. src_path           :   path where train data exists.                               %
% 3. val_src_path       :   path where validation data exists.                          %
%                                                                                       %
% 4. input_pattern      :   data pattern to use input training data.                    %
%                           if input_pattern="*.wav", it will use every .wav files as   %
%                           input data.                                                 %
%                                                                                       %
% 5. output_pattern     :   data pattern to use output training data.                   %
%                           if output_pattern="*.wav", it will use every .wav files as  %
%                           input data.                                                 %
%                                                                                       %
% 6. sample_rate        :   sample rate to resolve data.                                %
% 7. epoch              :   epochs to train data.                                       %
% ------------------------------------------------------------------------------------- %

function train_model(model_save_path, src_path, val_src_path, ...
    input_pattern, output_pattern, sample_rate, epoch)
    
    args = model_save_path + " " + src_path + " " + val_src_path + " " ...
        + input_pattern + " " + output_pattern + " " ...
        + sample_rate + " " + epoch;

    args = "train_model.py " + args;

    pyrunfile(args)
end