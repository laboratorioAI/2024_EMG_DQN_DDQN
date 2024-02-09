function plotConfusionMatNoGesture(csvFilePath, customTitle)
data = readmatrix(csvFilePath);

actualValues = data.ActualGesture;
predictedDQN = data.predDQN;
predictedDDQN = data.predDDQN;

postActualValues = data.ActualGesture;
postPredictedDQN = data.postDQN;
postPredictedDDQN = data.postDDQN;

%% Filter

filterIdx = actualValues ~= 6;
actualValues = actualValues(filterIdx);
predictedDQN = predictedDQN(filterIdx);
predictedDDQN = predictedDDQN(filterIdx);

filterIdx = predictedDQN ~= 6;
actualValues = actualValues(filterIdx);
predictedDQN = predictedDQN(filterIdx);
predictedDDQN = predictedDDQN(filterIdx);

filterIdx = predictedDDQN ~= 6;
actualValues = actualValues(filterIdx);
predictedDQN = predictedDQN(filterIdx);
predictedDDQN = predictedDDQN(filterIdx);

%% Post-processing filter

filterIdx = postActualValues ~= 6;
postActualValues = postActualValues(filterIdx);
postPredictedDQN = postPredictedDQN(filterIdx);
postPredictedDDQN = postPredictedDDQN(filterIdx);

filterIdx = postPredictedDQN ~= 6;
postActualValues = postActualValues(filterIdx);
postPredictedDQN = postPredictedDQN(filterIdx);
postPredictedDDQN = postPredictedDDQN(filterIdx);

filterIdx = postPredictedDDQN ~= 6;
postActualValues = postActualValues(filterIdx);
postPredictedDQN = postPredictedDQN(filterIdx);
postPredictedDDQN = postPredictedDDQN(filterIdx);

%% Confusion matrices

[confusionDQN, orderDQN] = confusionmat(actualValues, predictedDQN);
[confusionDDQN, orderDDQN] = confusionmat(actualValues, predictedDDQN);

[cfPostDQN, orderPostDQN] = confusionmat(postActualValues, postPredictedDQN);
[cfPostDDQN, orderPostDDQN] = confusionmat(postActualValues, postPredictedDDQN);

uniqueValues = unique(vertcat(actualValues, predictedDQN, predictedDDQN));
actionLabels = cell(size(uniqueValues));

for i = 1:length(uniqueValues)
    actionLabels{i} = mapAction(uniqueValues(i));
end

allLabels = cellfun(@(x) x{1}, actionLabels, 'UniformOutput', false);

classMetrics(confusionDQN, orderDQN, allLabels, "cmNgDQN-" + customTitle + ".csv");
classMetrics(confusionDDQN, orderDDQN, allLabels, "cmNgDDQN-" + customTitle + ".csv");

classMetrics(cfPostDQN, orderPostDQN, allLabels, "cmNgPostDQN-" + customTitle + ".csv");
classMetrics(cfPostDDQN, orderPostDDQN, allLabels, "cmNgPostDDQN-" + customTitle + ".csv");

figure('WindowState', 'maximized');

subplot(2, 2, 1);
confusionchart( ...
    cfDQN, ...
    allLabels, ...
    'Title', "DQN " + customTitle, ...
    'FontSize', 24, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');

set(gca, 'FontSize', 16);

subplot(2, 2, 2);
confusionchart( ...
    cfDDQN, ...
    allLabels, ...
    'Title', "DDQN " + customTitle, ...
    'FontSize', 24, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');

set(gca, 'FontSize', 16);

subplot(2, 2, 3);
confusionchart( ...
    cfPostDQN, ...
    allLabels, ...
    'Title', "Post-processing DQN " + customTitle, ...
    'FontSize', 24, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');

set(gca, 'FontSize', 16);

subplot(2, 2, 4);
confusionchart( ...
    cfPostDDQN, ...
    allLabels, ...
    'Title', "Post-processing DDQN " + customTitle, ...
    'FontSize', 24, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');

set(gca, 'FontSize', 16);
end
