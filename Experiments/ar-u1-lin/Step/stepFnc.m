function [nextObs, reward, isdone, info] = stepFnc(action, info)
% Next window from current user will be the next observation
info.currentSampleWindow = info.currentSampleWindow + 1;

[info.maxWindows,~,obs,info.username,info.gestureName,info.gtSize,info.gt] = readEMG( ...
    info.currentUser, ...
    info.currentSample, ...
    info.currentSampleWindow, ...
    info.stride, ...
    info.windowSize, ...
    info.isTraining);

nextObs = {obs, obs};

% Get reward from reward function
[R, actualGesture] = info.rewardFnc(action, info.gestureName, info.gtSize, info.gt);

rewardA = R(1);
rewardB = R(2);

switch info.longtermType
    case 1
        rewardA = linLongtermReward(info.alpha, info.beta, info.longtermA, rewardA);
        rewardB = linLongtermReward(info.alpha, info.beta, info.longtermB, rewardB);
    case 2
        rewardA = expLongtermReward(info.alpha, info.beta, info.longtermA, rewardA);
        rewardB = expLongtermReward(info.alpha, info.beta, info.longtermB, rewardB);
end

if rewardA > 0
    info.longtermA = info.longtermA + 1;
else
    info.longtermA = 0;
end

if rewardB > 0
    info.longtermB = info.longtermB + 1;
else
    info.longtermB = 0;
end

reward = [rewardA, rewardB];

% Creates a file with validation data
if ~info.isTraining
    fileID = fopen(info.experimentName + string(datetime("today")) + ".csv",'a');

    fprintf(fileID, '%i,%i,%f,%i,%f,%i,%i\n', ...
        actualGesture, ...
        cell2mat(action(1)), reward(1), ...
        cell2mat(action(2)), reward(2), ...
        info.currentSample, info.currentSampleWindow - 1);

    fclose(fileID);

    fprintf('STEP -> window: %i, gesture: %s, DQNPred: %s, DDQNPred: %s\n', ...
        info.currentSampleWindow, ...
        info.gestureName, ...
        mapAction(cell2mat(action(1))), ...
        mapAction(cell2mat(action(2))));
end

isdone = info.currentSampleWindow >= info.maxWindows;
end

