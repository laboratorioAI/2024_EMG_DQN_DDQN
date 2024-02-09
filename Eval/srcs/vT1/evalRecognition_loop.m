function [results, confusion, tiempos] = evalRecognition_loop(responses,...
    privInfo, userGroup, repGroup, opts)
%evalRecognition_loop evaluates a set of users with their repetitions. It returns the
%results in a struct per user. Change this file only when the format of the
%submission file is changed.
%
% Inputs
%	responses           struct with fields
%                       user->repGroup->repNum->responseField.
%   privInfo            private struct with the hidden info.
%   userGroup           char (options 'training', 'testing')
%   repGroup            char (options 'training', 'testing')
%
% Outputs
% 	results             struct with user. Every user field has a matrix
%                       150-by-3, every row is [classResult, recogResult,
%                       overlappingFactor].
%   confusion           struct by user. Every user field has a 150-by-2
%                       cell, where 1st column is target and 2nd column,
%                       predicted class, both as categoricals.
%   tiempos             m-by-1 vector with processing times
% Ejemplo
% = evalRecognition_loop()
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org
Cuando escribí este código, solo dios y yo sabíamos como funcionaba.
Ahora solo lo sabe dios.

07 May 2020
Matlab 9.7.0.1261785 (R2019b) Update 3.
%}

%%
switch nargin
    case 2
        userGroup = 'testing';
        repGroup = 'testing';
        opts = struct();
    case 3
        repGroup = 'testing';
        opts = struct();
    case 4
        opts = struct();
end

usersTestList = fieldnames(responses.(repGroup));

% user loop
for kI = usersTestList'
    kUser = kI{1};
    
    fprintf('%s:\tAnalyzing %s\n',datestr(datetime), kUser)
    % user info (only for GT)
    userRepInfo = privInfo.(userGroup).(kUser).repInfo.(repGroup);
    
    numReps =  numel(responses.(repGroup).(kUser).class);
    
    
    % class, recog, overla, true, pred
    results.(kUser) = nan(numReps, 3);
    confusion.(kUser) = cell(numReps, 2);
    
    % tiempos
    tiempos.(kUser) = cat(2, ...
        responses.(repGroup).(kUser).vectorOfProcessingTime{:});
    
    % user rep loop
    for kRep = 1:numReps
        
        % get response. p1
        resp.class = responses.(repGroup).(kUser).class{kRep};
        if isempty(resp.class)
            continue
        end
        % get response. p2
        resp.vectorOfLabels = responses.(repGroup).(kUser...
            ).vectorOfLabels{kRep};
        resp.vectorOfTimePoints = responses.(repGroup).(kUser...
            ).vectorOfTimePoints{kRep};
        resp.vectorOfProcessingTimes = responses.(repGroup).(kUser...
            ).vectorOfProcessingTime{kRep};
        
        % get gt
        repInfo.gestureName = userRepInfo{kRep, 2};
        repInfo.groundTruth = userRepInfo{kRep, 4};
        
        repEval = evalRecognition(repInfo, resp, opts);
        
        results.(kUser)(kRep, :) = [repEval.classResult, ...
            repEval.recogResult, repEval.overlappingFactor];
        
        confusion.(kUser){kRep, 1} = repInfo.gestureName;
        confusion.(kUser){kRep, 2} = resp.class;
    end
end

tiempos = struct2array(tiempos);

