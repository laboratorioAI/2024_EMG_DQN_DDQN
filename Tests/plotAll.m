addpath("Helpers\", "Plots\");

%% All experiments plots
experimentName = 'ar-lin-gtr-u75';
pathResults = ['Experiments\', experimentName, '\results.mat'];
results = load(pathResults).results;
tp2csv(results, experimentName);

% pathDQN = 'Experiments\ar-u1-exp-effort-lr\agentDQN.mat';
% pathDDQN = 'Experiments\ar-u1-exp-effort-lr\agentDDQN.mat'; 
% pathVal = 'Experiments\ar-u1-exp-effort-lr\ar-u1-exp-effort-lr.csv';
% agent = load(pathDQN);
% agentDQN = agent.agentDQN;
% 
% agent = load(pathDDQN);
% agentDDQN = agent.agentDDQN;
% plotDQNTrainingProgress(results, experimentName);
% saveas(gcf, fullfile('Experiments', experimentName, ['tp-' experimentName '.png']));
% 
% plotCombinedDQNTrainingProgress(results, experimentName);
% saveas(gcf, fullfile('Experiments', experimentName, ['ctp-' experimentName '.png']));
% 
% plotConfusionMat(pathVal, experimentName);
% saveas(gcf, fullfile('Experiments', experimentName, ['cf-ng-' experimentName '.png']));
% 
% plotConfusionMatNoGesture(pathVal, experimentName);
% saveas(gcf, fullfile('Experiments', experimentName, ['cf-' experimentName '.png']));
