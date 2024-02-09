addpath("Helpers\", "Plots\");

user = load('Data\Training1\user1\userData.mat');

dagnet = layerGraph(coder.loadDeepLearningNetwork('model_s30_e10.mat'));
initMean = dagnet.Layers(1, 1).Mean;
newInputLayer = imageInputLayer([13 24 8], 'Normalization', 'none', 'Name', 'state');
dagnet = replaceLayer(dagnet, 'data', newInputLayer);
net = assembleNetwork(dagnet);

% [fist, noGesture, open, pinch, waveIn, waveOut]
%% noGesture
[~,EMGGT,g1,~,gestureNameGT,gtSize,groundTruthGT] = readEMG(user, 2, 1, 40, 300, true);
g1 = zeroCenterNormalization(g1);
predict(net, g1)
disp(gestureNameGT)

%% fist
[~,EMGGT,g2,~,gestureNameGT,gtSize,groundTruthGT] = readEMG(user, 28, 15, 40, 300, true);
predict(net, g2)
disp(gestureNameGT)
plotEMG(repmat(groundTruthGT,1, 8), gestureNameGT);
plotEMG(EMGGT, gestureNameGT);

%% open
[~,EMGGT,g3,~,gestureNameGT,gtSize,groundTruthGT] = readEMG(user, 53, 8, 40, 300, true);
predict(net, g3)
disp(gestureNameGT)
%plotEMG(repmat(groundTruthGT,1, 8), gestureNameGT);

%% pinch
[~,EMGGT,g4,~,gestureNameGT,gtSize,groundTruthGT] = readEMG(user, 77, 13, 40, 300, true);
predict(net, g4)
disp(gestureNameGT)
%plotEMG(repmat(groundTruthGT,1, 8), gestureNameGT);

%% waveIn
[~,EMGGT,g5,~,gestureNameGT,gtSize,groundTruthGT] = readEMG(user, 104, 15, 40, 300, true);
predict(net, g5)
disp(gestureNameGT)
%plotEMG(repmat(groundTruthGT,1, 8), gestureNameGT);

%% waveOut
[~,EMGGT,g6,~,gestureNameGT,gtSize,groundTruthGT] = readEMG(user, 129, 15, 40, 300, true);
predict(net, g6)
disp(gestureNameGT)

% plotEMG(repmat(groundTruthGT,1, 8), gestureNameGT);
