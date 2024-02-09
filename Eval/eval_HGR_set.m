function results = eval_HGR_set(experimentName, options)
%eval_HGR_set is the main function of the eval_HGR_set toolbox.
%This function evaluates the responses of a Hand Gesture Recognition Model
%saved in the file experimentName. Evaluation of classification accuracy,
%recognition accuracy and processing time is carried on. The responses .mat
%file must be in the correct format (Check documentation). When options for
%saving is enable, a folder is created where figs, pngs, txt summary and
%a confusion tex template are generated. This output folder is configurable
%through options, by default is "resultsHGR". NOTE: your submission file
%must be accesible from matlab, or options.inputFolder must be specified.
%This function loads the required files (i.e. responses, personalStats) and
%just calls methods for plotting and analising.
%NOTE: only a rep Group is evaluated.
%
% Inputs
%   experimentName  () char with the name of the responses *.mat file. Note
%                   that this name will be used as name of the experiment,
%                   and a folder with results will be created with it.
%   options         () struct with fields. Options are:
%                   .repGroup- char of the repetition group to evaluate.
%                   'training' or 'testing'.
%                   .privFile- char of the .mat file with the hidden
%                   information.
%                   .userInfoFile- char of the .mat file with the stats of
%                   the users (i.e. age, gender, etc.).
%                   .inputFolder- char of the folder where the submission
%                   is allocated.
%                   .publicFolder- char of the folder where images,
%                   summary, latex table will be saved.
%                   .evalResultsFolder- char of the folder where
%                   the summary mat evaluation will be saved. In testing
%                   this file is hidden from public.
%                   .userGroup- char training or testing with the user
%                   group corresponding to the responses.
%                   .save- bool when true saves the results in the output
%                   folder.
%                   .generateFigs- bool when true creates the graphs.
%
% Outputs
%  results          () struct with fields: class, class_std, recog,
%                   recog_std, time, time_std
%
% Ejemplo
%    = analyze_HGRResponses('experimentTrain_specific')
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

29 July 2020
Matlab 9.7.0.1261785 (R2019b) Update 3.
%}


%% Input Validation
arguments
    experimentName        (1, :) char
    options
end

% defaults
if ~isfield(options, 'privFile')
    options.privFile = 'repData_training';
end

if ~isfield(options, 'userInfoFile')
    options.userInfoFile = 'personalInfo_training';
end
if ~isfield(options, 'inputFolder')
    options.inputFolder = '';
end
if ~isfield(options, 'publicFolder')
    options.publicFolder = '.\resultsHGR\';
end
if ~isfield(options, 'evalResultsFolder')
    options.evalResultsFolder = '.\resultsHGR\';
end
if ~isfield(options, 'userGroup')
    options.userGroup = 'training';
end
if ~isfield(options, 'repGroup')
    options.repGroup = 'testing';
end
if ~isfield(options, 'generateFigs')
    options.generateFigs = true;
end
if ~isfield(options, 'save')
    options.save = true;
end

if ~isfield(options, 'genPersonalStats')
    options.genPersonalStats = true;
end

%%  extractions
privFile = options.privFile;
publicFolder = options.publicFolder;
userGroup = options.userGroup;
userInfoFile = options.userInfoFile;
repGroup = options.repGroup;

%% create output folder
options.outputFolder = [publicFolder experimentName];
[~, ~, ~] = mkdir(options.outputFolder);

%% load files
responsesFile = [options.inputFolder experimentName '.mat'];

% responses
responses = load(responsesFile);
responses = responses.response;

%% data persons
if options.genPersonalStats
    
    dataPersons = load(userInfoFile);
    dataPersons = dataPersons.([userGroup 'Users']);
else
    dataPersons = [];
end

%% priv
privInfo = load(privFile);
privInfo = privInfo.hiddenInfo;

%% Running the evalRecognition Loop
% by user, by rep, etc.
resultsFile = [options.evalResultsFolder '\' experimentName '.mat'];
[evaluation, confusion, tiemps] = evalRecognition_loop(responses, ...
    privInfo, userGroup, repGroup, options);

fecha = datestr(datetime);
save(resultsFile, 'evaluation', 'confusion', 'tiemps', 'fecha')

%% Analyse results
fprintf('%s____________\n', experimentName);

[sumTable, class, classStd, recog, recogStd, time, timeStd, ...
    targets, outputs] = ...
    analyseResults(evaluation, confusion, tiemps, options, dataPersons);
beep

%% charts Results
if options.generateFigs
    plotSummaryResults(sumTable, class, classStd, recog, recogStd ...
        ,tiemps, targets, outputs, options)
end

%% Results by segments
if options.genPersonalStats
    [genStats, ageStats, handStats, damStats] = getPersonStats(...
        sumTable, options);
    
    %% plot results
    if options.generateFigs
        plotBySegments(genStats, ageStats, handStats, damStats, options);
    end
end


%% output
results.class = class;
results.class_std = classStd;
results.recog = recog;
results.recog_std = recogStd;
results.time = time;
results.time_std = timeStd;

