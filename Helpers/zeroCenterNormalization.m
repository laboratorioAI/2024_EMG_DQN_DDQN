function newSpec = zeroCenterNormalization(spec)
%newSpec = spec - repmat(initMean, [13, 24, 1]);
initMean = double(load("init_mean.mat").initMean);
newSpec = spec - initMean;
% std_vector = std(spec, 0, 3);
% newSpec = spec ./ repmat(std_vector, [1, 1, 8]);
end