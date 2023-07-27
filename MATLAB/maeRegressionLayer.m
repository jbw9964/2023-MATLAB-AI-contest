classdef maeRegressionLayer < nnet.layer.RegressionLayer ...
        
   methods
        function layer = maeRegressionLayer(name)           
            layer.Name = name;
            layer.Description = 'Mean absolute error';
        end

        function loss = forwardLoss(layer,Y,T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.

            % Calculate MAE.
            R = size(Y,3);
            meanAbsoluteError = sum(abs(Y-T),3)/R;
    
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = sum(meanAbsoluteError)/N;
        end
        
    end
end