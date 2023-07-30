
% Function that receives the address of a model
function model = load_model(h5)
    importmodel = importKerasLayers(h5, 'ImportWeights', true);          % Calling the model via the received address

    replaced_outputlayer = maeRegressionLayer('replaced_OutputLayer');   % last layer settings
    replaced_model = replaceLayer(importmodel, ...
        'multiply_OutputLayer_PLACEHOLDER', replaced_outputlayer);       % Enter the set layer as the last layer of the model
    
    model = assembleNetwork(replaced_model);                             % Loaded Neural Network Combination
end