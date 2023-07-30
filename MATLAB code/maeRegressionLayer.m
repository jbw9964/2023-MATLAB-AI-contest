
% Function to create mae layer
classdef maeRegressionLayer < nnet.layer.RegressionLayer ...
    
    methods
        function layer = maeRegressionLayer(name)           
            layer.Name = name;
            layer.Description = 'Check output layer';           % Layer Description
        end

        function loss = forwardLoss(~,Y,T)                      % mean absolute error function
            R = size(Y,3);
            meanAbsoluteError = sum(abs(Y-T),3)/R;

            N = size(Y,4);
            loss = sum(meanAbsoluteError)/N;
        end
        
    end
end