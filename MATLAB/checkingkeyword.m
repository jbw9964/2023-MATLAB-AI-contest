%%%%%%%%%%% Another Toolbox. just check other things %%%%%%%%%%%%%%
% h5disp('checkpoint.h5') %h5 파일의 모든 정보 출력
% data = hdf5read('checkpoint.h5', '/');

% info = h5info('checkpoint.h5', '/'); %h5 파일 정보
% info2 = h5info('checkpoint.h5', '/model_weights'); %h5 파일 속 model_weights 정보호출
% info3 = h5info('checkpoint.h5', '/model_weights/batch_normalization/batch_normalization/beta:0'); %h5 파일 속 model_weights 정보호출


% H5 파일로부터 가중치 읽기
% weights = h5read(h5_file_path, '/model_weights');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% model = importKerasLayers(h5_file_path); % load model structure

% plot(layers); %check model structure
% weights = layers.Layers(2).Weights; $check weights

