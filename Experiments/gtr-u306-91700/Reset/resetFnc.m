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
info.experimentName = "pre-ft-gtr-";

% The dir where Data will be loaded
dataDir = 'D:\MATLAB\Cristian&Kevin\DQN-DDQN306\Data\Training\';

%% Reset function

% How many users are there in Data?
usersCount = length(dir(fullfile(dataDir, 'user*')));
maxSamples = 150;

persistent shuffledUsers
persistent sampleIndices
persistent currentSampleIdx
persistent userIdx

if isempty(currentSampleIdx)
    currentSampleIdx = 1;
    userIdx = 1;

    % Shuffle users and samples when training starts
    [shuffledUsers, sampleIndices] = shuffleData(usersCount, maxSamples);
elseif currentSampleIdx < maxSamples % Max samples per user for training
    currentSampleIdx = currentSampleIdx + 1;
elseif userIdx < usersCount % Max number of users
    currentSampleIdx = 1;
    userIdx = userIdx + 1;
else
    currentSampleIdx = 1;
    userIdx = 1;

    % Shuffle users and samples when there is no more users
    [shuffledUsers, sampleIndices] = shuffleData(usersCount, maxSamples);
end

% Loads tha user Data and saves it in info for avoiding disk readings every step.
info.currentUser = load(getUserDir(dataDir, shuffledUsers(userIdx)));
info.currentSample = sampleIndices(currentSampleIdx);
info.currentSampleWindow = 1;
info.stride = 40;
info.windowSize = 300;

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

function [shuffledUsers, sampleIndices] = shuffleData(usersCount, samples)
% SHUFFLEDATA - Shuffles users indices and samples in order to train in a
% homogeneous way.
%
% - shuffledUsers: A permutated array from 1 to usersCount.
% - sampleIndices: A permutated array from 1 to samples.
%
% Usage: [shuffledUsers, sampleIndices] = shuffleData(306, 150)
shuffledUsers = randperm(usersCount);
sampleIndices = randperm(samples);
end
