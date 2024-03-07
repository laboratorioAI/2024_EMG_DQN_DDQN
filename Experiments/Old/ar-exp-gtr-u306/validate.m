%

%% Set-Up

addpath("Helpers\", "Data\", "Reset\", "Step\", "Reward\", "Experiments\")

obsInfo = rlNumericSpec([13 24 8], Name='emgFeatures');
actInfo = rlFiniteSetSpec([1 2 3 4 5 6], Name='gestures');

obsInfos = {obsInfo, obsInfo};
actInfos = {actInfo, actInfo};

env = rlMultiAgentFunctionEnv(obsInfos, actInfos, @stepFnc, @resetFnc);

agents = load('Experiments\gtr-u1-s150-15000\Snapshots\Agents14999.mat');
getAction(agents.saved_agent(1,1), rand(obsInfo.Dimension));
getAction(agents.saved_agent(1,2), rand(obsInfo.Dimension));

usersCount = length(dir(fullfile('Data\Training1\', 'user*')));
samplesPerUser = 150;

% stridesPerSample = 25;
% MaxSteps=25

simOpts = rlSimulationOptions(NumSimulations=samplesPerUser * usersCount);

clear resetFncVal;
clear stepFncVal;

experience = sim(env, agents.saved_agent, simOpts);

save("gtr-u1-s150-15000-" + string(datetime("today")) + ".mat", "experience", '-v7.3');
