%example_showConfusion evaluates a set of repetitions and plots several
%results. Uses a small set of synthetic data. Note charts will make more
%sense with more users and more reps. 

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

18 May 2022
Matlab 9.11.0.1873467 (R2021b) Update 3.
%}

clear all
close all
clc

%% options
userGroup = 'training';
repGroup = 'testing';

options.save = false;
evalOpts.defaultGesture = 'relax';

%% Creating synthetic samples
%------- 4 repetitions for user 1

repsU1.class = num2cell(categorical(["waveIn"; "relax"; "up"; "up"]));

repsU1.vectorOfLabels = ...
    {categorical({'relax', 'relax', 'waveIn', 'waveIn', 'waveIn'})
    categorical({'relax', 'relax', 'relax', 'relax', 'relax'})
    categorical({'relax', 'relax', 'up', 'up', 'relax'})
    categorical({'relax', 'relax', 'waveIn', 'up', 'up'})};

% all samples may have the same window size and stride
repsU1.vectorOfTimePoints = ...
    {[50 200 400 600 800]
    [50 200 400 600 800]
    [50 200 400 600 800]
    [50 200 400 600 800]
    };

% assuming that all windows process with the same processing time
repsU1.vectorOfProcessingTime = ...
    {[0.1 0.1 0.1 0.1 0.1]
    [0.1 0.1 0.1 0.1 0.1]
    [0.1 0.1 0.1 0.1 0.1]
    [0.1 0.1 0.1 0.1 0.1]
    };

responses.(repGroup).user_1 = repsU1;
%_------- Creating dataset repInfo
gt = false(1, 1000); % assuming all reps have the same gt.
gt(400:600) = true;
repInfo1 = {[], categorical({'waveIn'}), [], gt};
repInfo2 = {[], categorical({'relax'}), [], gt};
repInfo3 = {[], categorical({'up', }), [], gt};
repInfo4 = {[], categorical({'up', }), [], gt};

repsInfo.(userGroup).user_1.repInfo.(repGroup) = [repInfo1
    repInfo2
    repInfo3
    repInfo4];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------- 4 repetitions for user 2
repsU2.class = num2cell(categorical(["up"; "relax"; "up"; "waveIn"]));
repsU2.vectorOfLabels = ...
    {categorical({'relax', 'relax', 'waveIn', 'up', 'up'})
    categorical({'relax', 'relax', 'relax', 'relax', 'relax'})
    categorical({'up', 'relax', 'up', 'up', 'relax'})
    categorical({'relax', 'relax', 'waveIn', 'up', 'relax'})};

% all samples may have the same window size and stride
repsU2.vectorOfTimePoints = ...
    {[50 200 400 600 800]
    [50 200 400 600 800]
    [50 200 400 600 800]
    [50 200 400 600 800]
    };

% assuming that all windows process with the same processing time
repsU2.vectorOfProcessingTime = ...
    {[0.1 0.1 0.1 0.1 0.1]
    [0.1 0.1 0.1 0.1 0.1]
    [0.1 0.1 0.1 0.1 0.1]
    [0.1 0.1 0.1 0.1 0.1]
    };

%_------- Creating dataset repInfo
gt = false(1, 1000); % assuming all reps have the same gt.
gt(500:700) = true;
repInfo1 = {[], categorical({'waveIn'}), [], gt};
repInfo2 = {[], categorical({'relax'}), [], gt};
repInfo3 = {[], categorical({'up', }), [], gt};
repInfo4 = {[], categorical({'up', }), [], gt};

responses.(repGroup).user_2 = repsU2;



repsInfo.(userGroup).user_2.repInfo.(repGroup) = [repInfo1
    repInfo2
    repInfo3
    repInfo4];


%% Running the evalRecognition Loop
[evaluation, confusion, tiemps] = evalRecognition_loop(responses, ...
    repsInfo, userGroup, repGroup, evalOpts);



%% Analyse results
[sumTable, class, classStd, recog, recogStd, time, timeStd, ...
    targets, outputs] = ...
    analyseResults(evaluation, confusion, tiemps, options); % dataPersons
beep

%% charts Results

plotSummaryResults(sumTable, class, classStd, recog, recogStd ...
    ,tiemps, targets, outputs, options)



%% output
results.class = class;
results.class_std = classStd;
results.recog = recog;
results.recog_std = recogStd;
results.time = time;
results.time_std = timeStd;

results
