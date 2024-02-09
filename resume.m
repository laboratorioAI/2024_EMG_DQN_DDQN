%% Set-Up

addpath("Helpers\", "Data\", "Reset\", "Step\", "Reward\")

obsInfo = rlNumericSpec([13 24 8], Name='emgFeatures');
actInfo = rlFiniteSetSpec([1 2 3 4 5 6], Name='gestures');

obsInfos = {obsInfo, obsInfo};
actInfos = {actInfo, actInfo};

env = rlMultiAgentFunctionEnv(obsInfos, actInfos, @stepFnc, @resetFnc);

% Set to true if you want to keep the Experience Buffer.
USE_PRE_TRAINED_MODEL = true;

agent = load('agentDQN.mat');
agentDQN = agent.agentDQN;
agentDQN.AgentOptions.ResetExperienceBufferBeforeTraining = not(USE_PRE_TRAINED_MODEL);

agent = load("agentDDQN.mat");
agentDDQN = agent.agentDDQN;
agentDDQN.AgentOptions.ResetExperienceBufferBeforeTraining = not(USE_PRE_TRAINED_MODEL);

%% Training

prevStats = load("results.mat");

prevStats.results(1, 1).TrainingOptions.MaxEpisodes = 30000;
prevStats.results(1, 1).TrainingOptions.StopTrainingValue = 30000;

prevStats.results(1, 2).TrainingOptions.MaxEpisodes = 30000;
prevStats.results(1, 2).TrainingOptions.StopTrainingValue = 30000;

results = train([agentDQN, agentDDQN], env, prevStats.results);


%% Save agents

save("agentDQN.mat", "agentDQN");
save("agentDDQN.mat", "agentDDQN");
save("results.mat", "results");
