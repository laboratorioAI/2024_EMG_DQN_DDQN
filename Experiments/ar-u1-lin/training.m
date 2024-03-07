%% Set-Up

addpath("Helpers\", "Data\", "Reset\", "Step\", "Reward\")

obsInfo = rlNumericSpec([13 24 8], Name='emgFeatures');
actInfo = rlFiniteSetSpec([1 2 3 4 5 6], Name='gestures');

obsInfos = {obsInfo, obsInfo};
actInfos = {actInfo, actInfo};

env = rlMultiAgentFunctionEnv(obsInfos, actInfos, @stepFnc, @resetFnc);

NNOptions = rlRepresentationOptions(...
    LearnRate=1e-5,...
    GradientThreshold=1, ...
    Optimizer="adam",...
    GradientThresholdMethod="l2norm",...
    UseDevice="gpu");

cnnDQN = targetCNN();
save("init-cnnDQN-" + string(datetime("today")) + ".mat", "cnnDQN");
criticDQN = rlQValueRepresentation( ...
    cnnDQN, ...
    obsInfo, ...
    actInfo,...
    'Observation','state', ...
    NNOptions);

cnnDDQN = targetCNN();
save("init-cnnDDQN-" + string(datetime("today")) + ".mat", "cnnDDQN");
criticDDQN = rlQValueRepresentation( ...
    cnnDDQN, ...
    obsInfo, ...
    actInfo,...
    'Observation','state', ...
    NNOptions);

%% Params

targetSmoothFactor = 1e-3;
miniBatchSize = 32;
numStepsLookAhead = 1;
discountFactor = 0.98;
experienceBufferLength = 50;
epsilonDecay = 1e-4;

%% DQN

agentOptionsDQN = rlDQNAgentOptions( ...
    UseDoubleDQN=false, ...
    TargetSmoothFactor=targetSmoothFactor, ...
    MiniBatchSize=miniBatchSize, ...
    NumStepsToLookAhead=numStepsLookAhead, ...
    DiscountFactor=discountFactor, ...
    SaveExperienceBufferWithAgent=true, ...
    ExperienceBufferLength=experienceBufferLength);

agentOptionsDQN.EpsilonGreedyExploration.EpsilonDecay = epsilonDecay;
agentDQN = rlDQNAgent(criticDQN, agentOptionsDQN);

%% DDQN

agentOptionsDDQN = rlDQNAgentOptions( ...
    UseDoubleDQN=true, ...
    TargetSmoothFactor=targetSmoothFactor, ...
    MiniBatchSize=miniBatchSize, ...
    NumStepsToLookAhead=numStepsLookAhead, ...
    DiscountFactor=discountFactor, ...
    SaveExperienceBufferWithAgent=true, ...
    ExperienceBufferLength=experienceBufferLength);

agentOptionsDDQN.EpsilonGreedyExploration.EpsilonDecay = epsilonDecay;
agentDDQN = rlDQNAgent(criticDDQN, agentOptionsDDQN);

%% Training
maxEpisodes = prod([150 1 100]); % maxSamples, maxUsers, maxIterations

trainOpts = rlMultiAgentTrainingOptions(...
    AgentGroups={[1 2]},...
    LearningStrategy=("decentralized"),...
    MaxEpisodes=maxEpisodes,...
    ScoreAveragingWindowLength=150,...
    StopTrainingCriteria="EpisodeCount",...
    StopTrainingValue=maxEpisodes,...
    SaveAgentCriteria="EpisodeFrequency",...
    SaveAgentValue=1000,...
    Verbose=true,...
    SaveAgentDirectory=pwd + "\Snapshots",...
    Plots="training-progress");

clear resetFnc;
clear stepFnc;

results = train([agentDQN, agentDDQN], env, trainOpts);

%% Save agents

save("DQN-" + string(datetime("today")) + ".mat", "agentDQN");
save("DDQN-" + string(datetime("today")) + ".mat", "agentDDQN");
save("trainStats-" + string(datetime("today")) + ".mat", "results")
save("cnnDQN-" + string(datetime("today")) + ".mat", "cnnDQN");
save("cnnDDQN-" + string(datetime("today")) + ".mat", "cnnDDQN");

