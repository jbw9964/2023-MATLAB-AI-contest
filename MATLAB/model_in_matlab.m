music_path = "./Data/music_data/";
voice_path = "./Data/voice_data/";
data_path = "./data_path/";
h5_file_path = 'checkpoint.h5';

model = importKerasLayers(h5_file_path, 'ImportWeights', true); %load model weights

% analyzeNetwork(layer) % information of model
%placeholderLayers = findPlaceholderLayers(layer) %Keras에서 불러왔지만 못쓰는 키워드가 있는 레이어 확인
replace = maeRegressionLayer('replaced_multlply_OuptLayer');
replace_layer = replaceLayer(model,'multiply_OutputLayer_PLACEHOLDER', replace);
% analyzeNetwork(replace_layer)

net = assembleNetwork(replace_layer);
test1_raw = predict(net, polar_merge);
test1 = polar_to_complex(test1_raw);
audioPlayerGUI(istft(annss, "Window", hamming(win_length), "OverlapLength", win_length*3/4, "FFTLength", n_fft), sample_rate, 'D_voice_raw');
%predict(net, data)하면 예측값이 나와야 함

% layer_to_remove = 'multiply_OutputLayer_PLACEHOLDER';
% remove_layer = removeLayers(layer, layer_to_remove);
% analyzeNetwork(remove_layer)