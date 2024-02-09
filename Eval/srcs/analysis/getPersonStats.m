function [genStats, ageStats, handStats, damStats] = getPersonStats(...
    fullData, options)
%getPersonStats segmenta resultados por age, gender,etc, crea un archivo
%log.
%
% Inputs
%   fullData        (306, 12) table, accuracies, gender, age, etc...
%
% Output
%   genStats        summary table by segment gender.
%   ageStats        summary table by segment age.
%   handStats        summary table by segment handedness.
%   damStats        summary table by segment reported arm damage.
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org
Cuando escribí este código, solo dios y yo sabíamos como funcionaba.
Ahora solo lo sabe dios.

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

03 July 2020
Matlab 9.7.0.1261785 (R2019b) Update 3.
%}

%% conversions
fullData.gender = categorical(fullData.gender);
fullData.handedness = categorical(fullData.handedness);
fullData.handedness(fullData.handedness == 'left_Handed') = 'left-handed';
fullData.handedness(fullData.handedness == 'right_Handed') = 'right-handed';
% fullData.hasSufferedArmDamage(fullData.hasSufferedArmDamage ==1)={'yes'};
% fullData.hasSufferedArmDamage(fullData.hasSufferedArmDamage ==0)={'no'};

fullData.hasSufferedArmDamage = logical(fullData.hasSufferedArmDamage);

%% gender
genStats = grpstats(fullData, {'gender'}, {'mean', 'std'}, 'DataVars', ...
    {'class', 'recog', 'overlapF'});
exportSummary(genStats, 'gender', options)

%% age
ageStats = grpstats(fullData, {'age'}, {'mean', 'std'}, 'DataVars',...
    {'class', 'recog', 'overlapF'});
exportSummary(ageStats, 'age', options)


%%  handedness
% close all
handStats = grpstats(fullData, {'handedness'}, {'mean', 'std'}, ...
    'DataVars', {'class', 'recog', 'overlapF'});

exportSummary(handStats, 'handedness', options)
%% hasSufferedArmDamage
damStats = grpstats(fullData, {'hasSufferedArmDamage'}, ...
    {'mean', 'std'}, 'DataVars', {'class', 'recog', 'overlapF'});

exportSummary(damStats, 'hasSufferedArmDamage', options)


