function lgraph = targetCNN()

lgraph = layerGraph();

% Add layer branches
tempLayers = imageInputLayer([13 24 8],"Name","state","Normalization","none");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Inception_1a-3x3_reduce")
    reluLayer("Name","Inception_1a-3x3_relu_reduce")
    convolution2dLayer([3 3],18,"Name","Inception_1a-3x3","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Inception_1a-5x5_reduce")
    reluLayer("Name","Inception_1a-5x5_relu_reduce_2")
    convolution2dLayer([5 5],18,"Name","Inception_1a-5x5","Padding",[2 2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","Inception_1a-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],18,"Name","Inception_1a-pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1a-1x1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","depthcat_1a")
    reluLayer("Name","Inception_1a_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Inception_1b-3x3_reduce")
    reluLayer("Name","Inception_1b-3x3_relu_reduce")
    convolution2dLayer([3 3],18,"Name","Inception_1b-3x3","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Inception_1b-5x5_reduce")
    reluLayer("Name","Inception_1b-5x5_relu_reduce_2")
    convolution2dLayer([5 5],18,"Name","Inception_1b-5x5","Padding",[2 2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","Inception_1b-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],18,"Name","Inception_1b-pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1b-1x1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","depthcat_1b")
    reluLayer("Name","Inception_1b")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","Inception_1c-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],18,"Name","Inception_1c-pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1c-1x1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Inception_1c-5x5_reduce")
    reluLayer("Name","Inception_1c-5x5_relu_reduce_2")
    convolution2dLayer([5 5],18,"Name","Inception_1c-5x5","Padding",[2 2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Inception_1c-3x3_reduce")
    reluLayer("Name","Inception_1c-3x3_relu_reduce")
    convolution2dLayer([3 3],18,"Name","Inception_1c-3x3","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","depthcat_1c");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1ac")
    reluLayer("Name","Inception_1c")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1d-1x1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Inception_1d-3x3_reduce")
    reluLayer("Name","Inception_1d-3x3_relu_reduce")
    convolution2dLayer([3 3],18,"Name","Inception_1d-3x3","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","Inception_1d-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],18,"Name","Inception_1d-pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Inception_1d-5x5_reduce")
    reluLayer("Name","Inception_1d-5x5_relu_reduce_2")
    convolution2dLayer([5 5],18,"Name","Inception_1d-5x5","Padding",[2 2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","depthcat_1d")
    reluLayer("Name","Inception_1d")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Inception_1e-5x5_reduce")
    reluLayer("Name","Inception_1e-5x5_relu_reduce_2")
    convolution2dLayer([5 5],18,"Name","Inception_1e-5x5","Padding",[2 2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1e-1x1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Inception_1e-3x3_reduce")
    reluLayer("Name","Inception_1e-3x3_relu_reduce")
    convolution2dLayer([3 3],18,"Name","Inception_1e-3x3","Padding",[1 1 1 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","Inception_1e-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],18,"Name","Inception_1e-pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","depthcat_1e");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1ce")
    reluLayer("Name","Inception_1e")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Inception_1f-5x5_reduce")
    reluLayer("Name","Inception_1f-5x5_relu_reduce_2")
    convolution2dLayer([5 5],18,"Name","Inception_1f-5x5","Padding",[2 2 2 2])
    reluLayer("Name","Inception_1f-5x5_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],18,"Name","Inception_1f-1x1")
    reluLayer("Name","Inception_1f-1x1_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","Inception_1f-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],18,"Name","Inception_1f-pool_proj")
    reluLayer("Name","Inception_1f-relu-pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Inception_1f-3x3_reduce")
    reluLayer("Name","Inception_1f-3x3_relu_reduce")
    convolution2dLayer([3 3],18,"Name","Inception_1f-3x3","Padding",[1 1 1 1])
    reluLayer("Name","Inception_1f-3x3_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","depthcat_1f")
    crossChannelNormalizationLayer(5,"Name","crossnorm_1")
    fullyConnectedLayer(6,"Name","output")];
lgraph = addLayers(lgraph,tempLayers);

% Clean helper var
clear tempLayers;

lgraph = connectLayers(lgraph,"state","Inception_1a-3x3_reduce");
lgraph = connectLayers(lgraph,"state","Inception_1a-5x5_reduce");
lgraph = connectLayers(lgraph,"state","Inception_1a-pool");
lgraph = connectLayers(lgraph,"state","Inception_1a-1x1");

lgraph = connectLayers(lgraph,"Inception_1a-5x5","depthcat_1a/in2");
lgraph = connectLayers(lgraph,"Inception_1a-pool_proj","depthcat_1a/in3");
lgraph = connectLayers(lgraph,"Inception_1a-1x1","depthcat_1a/in4");
lgraph = connectLayers(lgraph,"Inception_1a-3x3","depthcat_1a/in1");
lgraph = connectLayers(lgraph,"Inception_1a_relu","Inception_1b-3x3_reduce");
lgraph = connectLayers(lgraph,"Inception_1a_relu","Inception_1b-5x5_reduce");
lgraph = connectLayers(lgraph,"Inception_1a_relu","Inception_1b-pool");
lgraph = connectLayers(lgraph,"Inception_1a_relu","Inception_1b-1x1");
lgraph = connectLayers(lgraph,"Inception_1a_relu","addition_1ac/in1");
lgraph = connectLayers(lgraph,"Inception_1b-1x1","depthcat_1b/in2");
lgraph = connectLayers(lgraph,"Inception_1b-5x5","depthcat_1b/in1");
lgraph = connectLayers(lgraph,"Inception_1b-pool_proj","depthcat_1b/in4");
lgraph = connectLayers(lgraph,"Inception_1b-3x3","depthcat_1b/in3");
lgraph = connectLayers(lgraph,"Inception_1b","Inception_1c-pool");
lgraph = connectLayers(lgraph,"Inception_1b","Inception_1c-1x1");
lgraph = connectLayers(lgraph,"Inception_1b","Inception_1c-5x5_reduce");
lgraph = connectLayers(lgraph,"Inception_1b","Inception_1c-3x3_reduce");
lgraph = connectLayers(lgraph,"Inception_1c-pool_proj","depthcat_1c/in4");
lgraph = connectLayers(lgraph,"Inception_1c-1x1","depthcat_1c/in1");
lgraph = connectLayers(lgraph,"Inception_1c-5x5","depthcat_1c/in3");
lgraph = connectLayers(lgraph,"Inception_1c-3x3","depthcat_1c/in2");
lgraph = connectLayers(lgraph,"depthcat_1c","addition_1ac/in2");
lgraph = connectLayers(lgraph,"Inception_1c","Inception_1d-1x1");
lgraph = connectLayers(lgraph,"Inception_1c","Inception_1d-3x3_reduce");
lgraph = connectLayers(lgraph,"Inception_1c","Inception_1d-pool");
lgraph = connectLayers(lgraph,"Inception_1c","Inception_1d-5x5_reduce");
lgraph = connectLayers(lgraph,"Inception_1c","addition_1ce/in2");
lgraph = connectLayers(lgraph,"Inception_1d-1x1","depthcat_1d/in1");
lgraph = connectLayers(lgraph,"Inception_1d-3x3","depthcat_1d/in2");
lgraph = connectLayers(lgraph,"Inception_1d-pool_proj","depthcat_1d/in4");
lgraph = connectLayers(lgraph,"Inception_1d-5x5","depthcat_1d/in3");
lgraph = connectLayers(lgraph,"Inception_1d","Inception_1e-5x5_reduce");
lgraph = connectLayers(lgraph,"Inception_1d","Inception_1e-1x1");
lgraph = connectLayers(lgraph,"Inception_1d","Inception_1e-3x3_reduce");
lgraph = connectLayers(lgraph,"Inception_1d","Inception_1e-pool");
lgraph = connectLayers(lgraph,"Inception_1e-1x1","depthcat_1e/in1");
lgraph = connectLayers(lgraph,"Inception_1e-3x3","depthcat_1e/in2");
lgraph = connectLayers(lgraph,"Inception_1e-5x5","depthcat_1e/in3");
lgraph = connectLayers(lgraph,"Inception_1e-pool_proj","depthcat_1e/in4");
lgraph = connectLayers(lgraph,"depthcat_1e","addition_1ce/in1");
lgraph = connectLayers(lgraph,"Inception_1e","Inception_1f-5x5_reduce");
lgraph = connectLayers(lgraph,"Inception_1e","Inception_1f-1x1");
lgraph = connectLayers(lgraph,"Inception_1e","Inception_1f-pool");
lgraph = connectLayers(lgraph,"Inception_1e","Inception_1f-3x3_reduce");
lgraph = connectLayers(lgraph,"Inception_1f-1x1_relu","depthcat_1f/in1");
lgraph = connectLayers(lgraph,"Inception_1f-5x5_relu","depthcat_1f/in3");
lgraph = connectLayers(lgraph,"Inception_1f-relu-pool_proj","depthcat_1f/in4");
lgraph = connectLayers(lgraph,"Inception_1f-3x3_relu","depthcat_1f/in2");

end