
% Args
% ------------------------------------------------------------------------------------- %
% 1. model_path     :   path where model was saved.                                     %
% 2. src_path       :   path where data exists.                                         %
% 3. pred_dir       :   path where converted data will be stored.                       %
%                                                                                       %
% 4. pattern        :   file pattern to use data.                                       %
%                       if pattern="*.wav", it will use every .wav files as data        %
% 5. sample_rate    :   sample rate to resolve data.                                    %
% ------------------------------------------------------------------------------------- %

function convert_pred(model_path, src_path, pred_dir, pattern, sample_rate)
    
    args = model_path + " " + src_path + " " + pred_dir + " " + pattern + " " + sample_rate;
    args = "convert_pred.py " + args;
    pyrunfile(args)
end