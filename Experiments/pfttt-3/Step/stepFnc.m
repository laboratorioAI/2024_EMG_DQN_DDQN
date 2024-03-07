function [nextObs, reward, isdone, info] = stepFnc(action, info)
info.currentSampleWindow = info.currentSampleWindow + 1;

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
nextObs = {obs, obs};

[R, actualGesture] = info.rewardFnc(action, info.gestureName, info.gtSize, info.gt);
rewardA = R(1);
rewardB = R(2);

if rewardA < 0
    info.longtermA = 0;
else
    info.longtermA = info.longtermA + 1;
end

if rewardB < 0
    info.longtermB = 0;
else
    info.longtermB = info.longtermB + 1;
end

switch info.longtermType
    case 1
        rewardA = linLongtermReward(info.alpha, info.beta, info.longtermA, rewardA);
        rewardB = linLongtermReward(info.alpha, info.beta, info.longtermB, rewardB);
    case 2
        rewardA = expLongtermReward(info.alpha, info.beta, info.longtermA, rewardA);
        rewardB = expLongtermReward(info.alpha, info.beta, info.longtermB, rewardB);
end

reward = [rewardA, rewardB];

if ~info.isTraining
    newRow = table( ...
        str2double(strrep(info.username,'user','')), info.currentSample, info.currentSampleWindow -1, ...
        actualGesture, cell2mat(action(1)), cell2mat(action(2)), rewardA, rewardB, ...
        'VariableNames', {'User', 'Sample', 'Window', 'ActualGesture', 'predDQN', 'predDDQN', 'DQNReward', 'DDQNReward'} ...
        );

    info.evaluation = [info.evaluation; newRow];
end

isdone = info.currentSampleWindow >= info.maxWindows;

if isdone && ~info.isTraining
    dqnValues = info.evaluation.predDQN;
    ddqnValues = info.evaluation.predDDQN;

    mapActionInverseFn = @mapActionInverse;
    if info.FT
        mapActionInverseFn = @mapActionInverseFT;
    end

    postDQNValues = emgPostProcessing(dqnValues, mapActionInverseFn);
    postDDQNValues = emgPostProcessing(ddqnValues, mapActionInverseFn);

    info.evaluation.postDQN = postDQNValues;
    info.evaluation.postDDQN = postDDQNValues;

    writetable( ...
        info.evaluation, ...
        info.outputFile, ...
        'WriteMode', 'append', ...
        'Delimiter', ',', ...
        'WriteVariableNames', ~isfile(info.outputFile) ...
        );

    disp(info.evaluation);
end
end

