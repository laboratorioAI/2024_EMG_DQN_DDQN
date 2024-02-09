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

if info.rewardPerStep
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
else
    reward = [0, 0];
end

newRow = table(cell2mat(action(1)), cell2mat(action(2)), VariableNames = {'predDQN', 'predDDQN'});
info.evaluation = [info.evaluation; newRow];

isdone = info.currentSampleWindow >= info.maxWindows;

mapActionInverseFn = @mapActionInverse;
mapActionFn = @mapAction;
if info.FT
    mapActionInverseFn = @mapActionInverseFT;
    mapActionFn = @mapActionFT;
end

if isdone && ~info.rewardPerStep
    dqnValues = info.evaluation.predDQN;
    postDQNValues = emgPostProcessing(dqnValues, mapActionInverseFn);
    postDQNValues = arrayfun(mapActionFn, postDQNValues, 'UniformOutput',false);
    postDQNValues = categorical([postDQNValues{:}]);

    ddqnValues = info.evaluation.predDQN;
    postDDQNValues = emgPostProcessing(ddqnValues, mapActionInverseFn);
    postDDQNValues = arrayfun(mapActionFn, postDDQNValues, 'UniformOutput',false);
    postDDQNValues = categorical([postDDQNValues{:}]);
    
    reward = info.rewardFnc([getPredClass(postDQNValues), getPredClass(postDDQNValues)] , info.gestureName, info.gtSize, info.gt);
end

if isdone && ~info.isTraining
    elapsedTime = toc;
    repInfo.gestureName = info.gestureName;

    userGt = info.currentUser.userData.testing{info.currentSample};
    repInfo.groundTruth = false(1,1000);
    if isfield(userGt, 'groundTruth')
        repInfo.groundTruth = userGt.groundTruth;
    end

    %% DQN Values
    dqnValues = info.evaluation.predDQN;
    postDQNValues = emgPostProcessing(dqnValues, mapActionInverseFn);

    dqnValues = arrayfun(mapActionFn, dqnValues, 'UniformOutput',false);
    dqnValues = categorical([dqnValues{:}]);

    postDQNValues = arrayfun(mapActionFn, postDQNValues, 'UniformOutput',false);
    postDQNValues = categorical([postDQNValues{:}]);

    %% DDQN Values
    ddqnValues = info.evaluation.predDDQN;
    postDDQNValues = emgPostProcessing(ddqnValues, mapActionInverseFn);

    ddqnValues = arrayfun(mapActionFn, ddqnValues, 'UniformOutput',false);
    ddqnValues = categorical([ddqnValues{:}]);

    postDDQNValues = arrayfun(mapActionFn, postDDQNValues, 'UniformOutput',false);
    postDDQNValues = categorical([postDDQNValues{:}]);

    %% Responses
    responseDQN = customResponse(dqnValues, info.stride);
    responseDDQN = customResponse(ddqnValues, info.stride);

    responsePostDQN = customResponse(postDQNValues, info.stride);
    responsePostDDQN = customResponse(postDDQNValues, info.stride);

    %% Evaluation
    dqn = evalRecognition(repInfo, responseDQN);
    ddqn = evalRecognition(repInfo, responseDDQN);

    post_dqn = evalRecognition(repInfo, responsePostDQN);
    post_ddqn = evalRecognition(repInfo, responsePostDDQN);

    if info.gestureName == "noGesture"
        dqn.recogResult = dqn.classResult;
        dqn.overlappingFactor = NaN;

        ddqn.recogResult = ddqn.classResult;
        ddqn.overlappingFactor = NaN;

        post_dqn.recogResult = post_dqn.classResult;
        post_dqn.overlappingFactor = NaN;

        post_ddqn.recogResult = post_ddqn.classResult;
        post_ddqn.overlappingFactor = NaN;
    end

    %% Save data
    userNumber = str2double(strrep(info.username,'user',''));

    allEvals = table( ...
        userNumber,                 info.currentSample,         repInfo.gestureName,            elapsedTime, ...
        responseDQN.class,          responseDDQN.class,         responsePostDQN.class,          responsePostDDQN.class, ...
        dqn.classResult,            ddqn.classResult,           post_dqn.classResult,           post_ddqn.classResult, ...
        dqn.recogResult,            ddqn.recogResult,           post_dqn.recogResult,           post_ddqn.recogResult, ...
        dqn.overlappingFactor,      ddqn.overlappingFactor,     post_dqn.overlappingFactor,     post_ddqn.overlappingFactor, ...
        VariableNames={ ...
        'user',                     'sample',                   'actual_class',                 'elapsed_time' ...
        'dqn_predicted_class',      'ddqn_predicted_class',     'post_dqn_predicted_class',     'post_ddqn_predicted_class', ...
        'dqn_classification',       'ddqn_classification',      'post_dqn_classification',      'post_ddqn_classification', ...
        'dqn_recognition',          'ddqn_recognition',         'post_dqn_recognition',         'post_ddqn_recognition' ...
        'dqn_overlapping_factor',   'ddqn_overlapping_factor',  'post_dqn_overlapping_factor',  'post_ddqn_overlapping_factor'});

    writetable( ...
        allEvals, ...
        info.outputFile, ...
        'WriteMode', 'append', ...
        'Delimiter', ',', ...
        'WriteVariableNames', ~isfile(info.outputFile) ...
        );
end
end

function response = customResponse(prediction, stride)
response.vectorOfLabels = prediction;
pointTimes = 1:stride:1100;
response.vectorOfTimePoints = pointTimes(1:length(prediction));
response.vectorOfProcessingTimes = 0.02 * ones(1, length(prediction));
response.class = getPredClass(prediction);
end

function predictedFullClass = getPredClass(prediction)
idx = prediction ~= 'noGesture';
allMode = mode(prediction);
if isempty(prediction(idx))
    predictedFullClass = allMode;
else
    predictedFullClass = mode(prediction(idx));
end
end
