%evaluatingATrainingSet is an example of the usage of this toolbox for
%evaluating a set of repetitions.
% This example evaluates 4 users of the training user group, testing
% rep group. To evaluate over the testing set, please submit your responses
% to the publicEvaluator website.
% Some of the options fields are used in this example, for more info type:
%
% doc eval_HGRResponses

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

29 July 2020
Matlab 9.7.0.1261785 (R2019b) Update 3.
%}

%%
% name of the .mat file with your responses
experimentName = 'experimentTrain_specific';

%% options
% Name of the folder where the figs will be generated.
options.publicFolder  = '.\resultsHGR\';
%
options.evalResultsFolder  = '.\resultsHGR\';

% only training userGroup is available, the testing userGroup is only
% available through submmisions to the Web App HGR Public evaluator!!!
options.userGroup  = 'training';

% Repetitions of training or testing. Check that your responses file is in
% the correct format.
options.repGroup = 'testing';

options.generateFigs  = true; % when false, figures are not generated.
options.save = true; % saves result charts and summary in folder.

% when true generates charts by segment (i.e. by gender, age, etc.)
options.genPersonalStats = true; 
% Need help for more options? Type
% doc eval_HGRResponses

%% calling the library function
results = eval_HGR_set(experimentName, options);
disp(results)
