function [respXUser, ...
    classAvg, classStd, recogAvg, recogStd, timeAvg, timeStd,...
    targets, outputs] = ...
    analyseResults(evaluationXUser, confusionXUser, tiempos, ...
    options, dataPersons)
%analyseResults runs analysis over a set of responses. Its
%functionality was divided so that can be called from the results file.
%
% Inputs
%   evaluationXUser strict with field by user. Each of these fields have a
%                   150-by-3 double matrix with cols class, recog, overlap.
%   confusionXUser
%   confusionXUser  struct by user. Every user field has a 150-by-2 cell,
%                   where 1st column is target and 2nd column, predicted
%                   class, both as categoricals.
%   tiempos         -(1, m) double of processing times.
%   options         struct with fields save, outputFolder,
%                   genPersonalStats. save is boolean, outputFolder is
%                   char, genPersonalStats boolean.
%
%   dataPersons     -(306, 6) table, with age, gender, ehtnic of the users.

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: pij1613
jonathan.a.zea@ieee.org
Cuando escribí este código, solo dios y yo sabíamos como funcionaba.
Ahora solo lo sabe dios.

14 May 2020
Matlab 9.8.0.1323502 (R2020a).
%}

%% input
if nargin == 3
    options.save = false;    
end

if nargin == 5
    options.genPersonalStats = true;
else
    options.genPersonalStats = false;
end

%% tiempos
tiempos(tiempos == 0) = []; % removiendo ceros
timeAvg = mean(tiempos);
timeStd = std(tiempos);

%% means by user
userNames = fieldnames(evaluationXUser);

avgsXUser = struct();

for kUser = userNames'
    avgsXUser.(kUser{1}) = [mean(evaluationXUser.(kUser{1}), 'omitnan') ...
        std(evaluationXUser.(kUser{1}), 'omitnan')];
end

% order
getV = @(x)str2double(cell2mat(regexp(x, '\d+', 'match')));
% conversion
avgsXUser = struct2cell(avgsXUser);
avgsXUser = cat(1, avgsXUser{:})*100;
avgsXUser = [cellfun(getV, userNames) avgsXUser];
respXUser = array2table(avgsXUser, 'RowNames', userNames, ...
    'VariableNames', {'u','class', 'recog', 'overlapF','classStd', ...
    'recogStd', 'overlapFStd'});

respXUser = sortrows(respXUser);
% general results
classAvg = mean(respXUser.class);
recogAvg = mean(respXUser.recog);
recogStd = std(respXUser.recog);
classStd = std(respXUser.class);



%% export summary
exportSummary(respXUser, [], options, classAvg, classStd, recogAvg, ...
    recogStd, timeAvg, timeStd);

%% for stats
if options.genPersonalStats
    respXUser = [respXUser dataPersons(respXUser.u, :)];
end

%% confusion matrix
confs = struct2cell(confusionXUser);
confsAll = cat(1, confs{:});

targets = cat(1, confsAll{:, 1});
outputs = cat(1, confsAll{:, 2});


if options.save
    exportConfusionEMG(targets, outputs, options);
end


end


