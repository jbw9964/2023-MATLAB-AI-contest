
% Args
% ------------------------------------------------------------------------------------- %
% 1. target_dir : path where generated dataset will be stored.                          %
% 2. music_dir  : path where music source were stored.                                  %
% 3. voice_dir  : path where voice source were stored.                                  %
%                                                                                       %
% 4. voice_amp_ratio    :   the ratio to fit the amplitude of voice with music source.  %
%                           if voice_amp_ratio=0, the voice will be large as the music. %
%                                                                                       %
% 5. train_test_split   :   the ratio to split test dataset.                            %
%                           if it's 0, test dataset won't be generated.                 %
% ------------------------------------------------------------------------------------- %

function gen_dataset(target_dir, music_dir, voice_dir, voice_amp_ratio, train_test_split)

    args = target_dir + " " + music_dir + " " + voice_dir + " "...
        + voice_amp_ratio + " " + train_test_split;

    args = "gen_dataset.py " + args;
    pyrunfile(args)
end
