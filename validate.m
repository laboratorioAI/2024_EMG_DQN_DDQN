%% Set-Up
clear all;
clc;
addpath("Helpers\", "Data\", "Reset\", "Step\", "Reward\", "Experiments\", genpath("Eval\"))

obsInfo = rlNumericSpec([13 24 8], Name='emgFeatures');
actInfo = rlFiniteSetSpec([1 2 3 4 5 6], Name='gestures');

obsInfos = {obsInfo, obsInfo};
actInfos = {actInfo, actInfo};

env = rlMultiAgentFunctionEnv(obsInfos, actInfos, @stepFnc, @resetFnc);
experimentName = 'gtr-u306-91700';

agent = load(['Experiments\', experimentName, '\agentDQN.mat']);
agentDQN = agent.agentDQN;
getAction(agentDQN, rand(obsInfo.Dimension));

agent = load(['Experiments\', experimentName, '\agentDDQN.mat']);
agentDDQN = agent.agentDDQN;
getAction(agentDDQN, rand(obsInfo.Dimension));

usersCount = length(dir(fullfile('Data\Training\', 'user*')));
samplesPerUser = 150;

simOpts = rlSimulationOptions(NumSimulations=samplesPerUser * usersCount);

clear resetFncVal;
clear stepFncVal;

sim(env, [agentDQN, agentDDQN], simOpts);

pathResults = ['Experiments\', experimentName, '\results.mat'];
results = load(pathResults).results;
tp2csv(results, experimentName);
