function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
    % convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
    % pooledFeatures(poolRow, poolCol, featureNum, imageNum)
    for numImage = 1:numImages
        for numFeature = 1:numFilters
            for poolRow = 1:convolvedDim / poolDim
                offsetRow = 1+(poolRow-1)*poolDim;
                for poolCol = 1:convolvedDim / poolDim
                    offsetCol = 1+(poolCol-1)*poolDim;
                    patch = convolvedFeatures(offsetRow:offsetRow+poolDim-1, ...
                        offsetCol:offsetCol+poolDim-1,numFeature,numImage); %取出一个patch
                    pooledFeatures(poolRow,poolCol,numFeature,numImage) = mean(patch(:));
                end
            end            
        end
    end
    
end