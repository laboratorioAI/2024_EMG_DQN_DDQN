function [nextObs, reward, isdone, info] = stepFnc(action, info)
% STEPFUN specifies how the environment advances to the next state given
% the actions from all the agents.
%
% If N is the total number of agents, then the arguments are as follows.
% - NEXTOBS is a 1xN cell array (s).
% - ACTION is a 1xN cell array.
% - REWARD is a 1xN numeric array.
% - ISDONE is a logical or numeric scalar.
% - INFO contains any data that you want to pass between steps.

% Get reward from reward function
[reward, actualGesture] = info.rewardFnc(action, info.gestureName, info.gtSize, info.gt);

% Creates a file with validation data
if ~info.isTraining
    fileID = fopen(info.experimentName + string(datetime("today")) + ".csv",'a');
    
    fprintf(fileID, '%i,%i,%f,%i,%f\n', ...
        actualGesture, ...
        cell2mat(action(1)), reward(1), ...
        cell2mat(action(2)), reward(2));

    fclose(fileID);

    fprintf('STEP -> window: %i, gesture: %s\n', info.currentSampleWindow, info.gestureName);
    disp(reward);
end

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

isdone = info.currentSampleWindow >= info.maxWindows;
end

