classdef new_output_layer < nnet.layer.ClassificationLayer
    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end
 
    methods
        function layer = new_output_layer()           
            % (Optional) Create a myClassificationLayer.

            % Layer constructor function goes here.
        end

        function [] = predict(layer, X)
            % (Optional) Initialize layer learnable and state parameters.
            %
            % Inputs:
            %         layer  - Layer to initialize
            %         layout - Data layout, specified as a networkDataLayout
            %                  object
            %
            % Outputs:
            %         layer - Initialized layer
            %
            %  - For layers with multiple inputs, replace layout with 
            %    layout1,...,layoutN, where N is the number of inputs.
            
            % Define layer initialization function here.
        end

    end
end
