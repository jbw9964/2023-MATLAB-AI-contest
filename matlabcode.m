% Data_path = "./Data/kaggle_sample/kaggle_sample/";
% test_data_path = Data_path + "cv-valid-train/sample-000006.mp3";
% 
% sample_rate = 25000;
% 
% fprintf("------------------------- Original voice track -------------------------\n");
% 
% [voice_data, actual_sample_rate] = audioread(test_data_path);
% 
% if actual_sample_rate ~= sample_rate
%     voice_data = resample(voice_data, sample_rate, actual_sample_rate);
% end
% 
% if size(voice_data, 2) > 1 % 만약 불러오는 audio data가 stero 타입이면 mono로 변환
%     voice_data = mean(voice_data, 2); % 평균을 취해 모노 오디오로 변환
% end
% 
% fprintf("Voice\n");
% fprintf("Length : %d \n",max(size(voice_data)));
% fprintf("Sample rate : %d \n", sample_rate);
% fprintf("L / R : %.2f sec \n", max(size(voice_data)) / sample_rate);
% % audioPlayerGUI(voice_data, sample_rate, 'audio_data');
% % 불러온 Voice data를 확인하는 코드.
% fprintf("------------------------- Original music track -------------------------\n");
% 
% Data_path = "C:\Users\smart\Desktop\2023-MATLAB-AI-contest";
% music_data_path = Data_path + "\Data\music\Jonas Blue - Fast Car ft. Dakota (Official Video).mp3";
% 
% [music_data, actual_sample_rate] = audioread(music_data_path);
% 
% if size(music_data, 2) > 1 % 만약 불러오는 audio data가 stero 타입이면 mono로 변환
%     music_data = mean(music_data, 2); % 평균을 취해 모노 오디오로 변환
% end
% 
% if actual_sample_rate ~= sample_rate
%     music_data = resample(music_data, sample_rate, actual_sample_rate);
% end 
% 
% % 음성 데이터의 현재 duration 계산 (단위: 초)
% current_duration = size(music_data, 1) / sample_rate;
% desired_duration = max(size(voice_data)) / sample_rate;
% desired_num_samples = desired_duration * sample_rate;
% music_data = music_data(1:min(desired_num_samples, size(music_data, 1)), :);
% 
% fprintf("music\n");
% fprintf("Length : %d \n",max(size(music_data)));
% fprintf("Sample rate : %d \n", sample_rate);
% fprintf("L / R : %.2f sec \n", max(size(music_data)) / sample_rate);
% % audioPlayerGUI(music_data, sample_rate, 'music_data');
% 
% % 불러온 Music data를 확인하는 코드.
% fprintf("-------------------------    Merge track    -------------------------\n");
% 
% merge_data = voice_data + music_data;
% % audioPlayerGUI(merge_data, sample_rate, 'merge_data');
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% stft %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% n_fft = 960;
% win_length = 960;
% D_voice_raw = stft(voice_data, "Window", hamming(win_length), 'OverlapLength', win_length*3/4, "FFTLength", n_fft);
% D_music_raw = stft(music_data, "Window", hamming(win_length), 'OverlapLength', win_length*3/4, "FFTLength", n_fft);
% D_merge_raw = stft(merge_data, "Window", hamming(win_length), 'OverlapLength', win_length*3/4, "FFTLength", n_fft);
% 
% Quotient = floor(max(size(D_voice_raw(1,:))) / 64);
% fprintf("Quotient : %d \n",Quotient);
% 
% round_off = round(Quotient * 64);
% 
% D_voice_raw = D_voice_raw(:,1:round_off);
% D_music_raw = D_music_raw(:,1:round_off);
% D_merge_raw = D_merge_raw(:,1:round_off);
% 
% fprintf("-------------------------    Edited track    ------------------------- \n")
% fprintf("Voice \n")
% % audioPlayerGUI(istft(D_voice_raw, "Window", hamming(win_length), "OverlapLength", win_length*3/4, "FFTLength", n_fft), sample_rate, 'D_voice_raw');
% fprintf("Music \n")
% % audioPlayerGUI(istft(D_music_raw, "Window", hamming(win_length), "OverlapLength", win_length*3/4, "FFTLength", n_fft), sample_rate, 'D_music_raw');
% fprintf("Merge \n")
% % audioPlayerGUI(istft(D_merge_raw, "Window", hamming(win_length), "OverlapLength", win_length*3/4, "FFTLength", n_fft), sample_rate, 'D_merge_raw');
% % stft, istft 코드
% 
% disp(size(D_voice_raw));
% disp(size(D_music_raw));
% disp(size(D_merge_raw));
% %raw data의 원래 크기
% 
% polar_voice = complex_to_polar(D_voice_raw);
% polar_music = complex_to_polar(D_music_raw);
% polar_merge = complex_to_polar(D_merge_raw);
% % raw data(complex data) to polar data
% 
% disp(size(polar_voice));
% disp(size(polar_music));
% disp(size(polar_merge));

music_path = "./Data/music_data/";
voice_path = "./Data/voice_data/";
data_path = "./data_path/";
h5_file_path = 'checkpoint.h5';

% h5disp('checkpoint.h5') %h5 파일의 모든 정보 출력
% data = hdf5read('checkpoint.h5', '/');

info = h5info('checkpoint.h5', '/'); %h5 파일 정보
info2 = h5info('checkpoint.h5', '/model_weights'); %h5 파일 속 model_weights 정보호출
info3 = h5info('checkpoint.h5', '/model_weights/batch_normalization/batch_normalization/beta:0'); %h5 파일 속 model_weights 정보호출


% H5 파일로부터 가중치 읽기
weights = h5read(h5_file_path, '/model_weights');
