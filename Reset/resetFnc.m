function [initialObs, info] = resetFnc()
% Set to false if your are validating
info.isTraining = false;

% basicReward | groundTruthReward | groundTruthRewardFT | basicRewardFT
info.rewardFnc = @groundTruthReward;

% Change it as required if you are validating
info.outputFile = "gtr-u306-91700.csv";

% The dir where Data will be loaded
dataDir = 'Data\Training\';

% How many users are there in Data?
usersCount = length(dir(fullfile(dataDir, 'user*')));
maxSamples = 150;

% 0 = no long term reward, 1 = linear, 2 = exponential
info.longtermType = 2;
info.alpha = 1e-3;
info.beta = 0.20;

% Could save up to 3 hours
loadToRam = false;
%% Reset function88
persistent currentSampleIdx
persistent allUserSamples
persistent allUsers

if isempty(currentSampleIdx)
    currentSampleIdx = 1;
    allUserSamples = trueDataShuffle(usersCount, maxSamples);
    if loadToRam
        allUsers = loadAll(dataDir);
    end
    disp("Users and samples have been shuffled.")
elseif currentSampleIdx < maxSamples * usersCount
    currentSampleIdx = currentSampleIdx + 1;
else
    currentSampleIdx = 1;
    allUserSamples = trueDataShuffle(usersCount, maxSamples);
    disp("Users and samples have been shuffled again.")
end

sampleUser = allUserSamples(:,currentSampleIdx);
info.currentSample = sampleUser(1);
info.currentSampleWindow = 1;
info.FT = isequal(info.rewardFnc, @groundTruthRewardFT) || isequal(info.rewardFnc, @basicRewardFT);
info.rewardPerStep = isequal(info.rewardFnc, @groundTruthReward) || isequal(info.rewardFnc, @groundTruthRewardFT);

if loadToRam
    info.currentUser = allUsers{1, sampleUser(2)};
else
    info.currentUser = load(getUserDir(dataDir, sampleUser(2)));
end

info.stride = 40;
info.windowSize = 300;

info.longtermA = 0.0;
info.longtermB = 0.0;

info.evaluation = table();

[info.maxWindows,~,obs,info.username,info.gestureName,info.gtSize,info.gt] = readEMG( ...
    info.currentUser, ...
    info.currentSample, ...
    info.currentSampleWindow, ...
    info.stride, ...
    info.windowSize, ...
    info.isTraining);

if info.FT
    obs = zeroCenterNormalization(obs);
end

fprintf( ...
    'RESET -> user: %s, sample: %i, gesture: %s\n', ...
    info.username, ...
    info.currentSample, ...
    info.gestureName);

initialObs = {obs, obs};
tic
end

%% Random data selection

function allUserSamples = trueDataShuffle(usersCount, userSamples)
row = randperm(userSamples);
allUserSamples = [repmat(row, 1, usersCount); kron(1:usersCount, ones(1, userSamples))];
idx = randperm(usersCount*userSamples);
allUserSamples = allUserSamples(:,idx);
end
