function theta = cnnInitParams(imageDim,filterDim,numFilters,...
                                poolDim,numClasses)
% Initialize parameters for a single layer convolutional neural
% network followed by a softmax layer.
%                            
% Parameters:
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  theta      -  unrolled parameter vector with initialized weights

%% Initialize parameters randomly based on layer sizes.
assert(filterDim < imageDim,'filterDim must be less that imageDim');

Wc = 1e-1 * randn(filterDim, filterDim, numFilters); % 9 * 9 * 20

outDim = imageDim - filterDim + 1; % dimension of convolved image 20

% assume outDim is multiple of poolDim
assert(mod(outDim,poolDim)==0,...
       'poolDim must divide imageDim - filterDim + 1');

outDim = outDim / poolDim; % 20 / 2 = 10
hiddenSize = outDim^2 * numFilters; % 10 * 10 * 20 全连接层

% we'll choose weights uniformly from the interval [-r, r]
r  = sqrt(6) / sqrt(numClasses + hiddenSize + 1); % sqrt(20 + 2000 + 1)
Wd = rand(numClasses, hiddenSize) * 2 * r - r; % 10 * 2000

bc = zeros(numFilters, 1); %初始化为0 20
bd = zeros(numClasses, 1); % 10

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [Wc(:) ; Wd(:) ; bc(:) ; bd(:)];

end