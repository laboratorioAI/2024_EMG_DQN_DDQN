function [initialObs, info] = resetFnc()
% RESETFUN sets the default state of the environment.
%
% - INITIALOBS is a 1xN cell array (N is the total number of agents).
% - INFO contains any data that you want to pass between steps.
%
% To pass information from one step to the next, such as the environment
% state, use INFO.

% For this example, initialize the agent observations randomly
% (but set to 1 the value carried by the second observation channel
%  of the second agent).

%% Set-up

% Set to false if your are validating
info.isTraining = true;

% basicReward | groundTruthReward
info.rewardFnc = @groundTruthReward;

% Change it as required if you are validating
info.experimentName = "lin-gtr-u306-";

% The dir where Data will be loaded
dataDir = 'Data\Training\';

% How many users are there in Data?
usersCount = length(dir(fullfile(dataDir, 'user*')));
maxSamples = 150;

%{
Long-term reward method
    0 -> no long-term reward
    1 -> compute linear long-term reward.
        [0.04, 0.0] [0.20, 0.00] [0.08, 0.00]
    2 -> compute exponential long-term reward.
        [0.01, 0.2] [0.01, 0.25] [0.01, 0.28]
%}
info.longtermType = 1;
info.alpha = 0.04;
info.beta = 0;

%% Reset function

persistent currentSampleIdx
persistent allUserSamples

if isempty(currentSampleIdx)
    currentSampleIdx = 1;
    allUserSamples = trueDataShuffle(usersCount, maxSamples);
    disp("Users and samples have been shuffled.")
elseif currentSampleIdx < maxSamples * usersCount
    currentSampleIdx = currentSampleIdx + 1;
else
    currentSampleIdx = 1;
    allUserSamples = trueDataShuffle(usersCount, maxSamples);
    disp("Users and samples have been shuffled again.")
end

sampleUser = allUserSamples(:,currentSampleIdx);
info.currentUser = load(getUserDir(dataDir, sampleUser(2)));
info.currentSample = sampleUser(1);
info.currentSampleWindow = 1;
info.stride = 40;
info.windowSize = 300;

info.longtermA = 0;
info.longtermB = 0;

[info.maxWindows,~,obs,info.username,info.gestureName,info.gtSize,info.gt] = readEMG( ...
    info.currentUser, ...
    info.currentSample, ...
    info.currentSampleWindow, ...
    info.stride, ...
    info.windowSize, ...
    info.isTraining);

initialObs = {obs, obs};
fprintf( ...
    'RESET -> user: %s, sample: %i, gesture: %s\n', ...
    info.username, ...
    info.currentSample, ...
    info.gestureName);
end

%% Random data selection

function allUserSamples = trueDataShuffle(usersCount, userSamples)
row = 1:userSamples;
allUserSamples = [repmat(row, 1, usersCount); kron(1:usersCount, ones(1, userSamples))];
idx = randperm(usersCount*userSamples);
allUserSamples = allUserSamples(:,idx);
end
