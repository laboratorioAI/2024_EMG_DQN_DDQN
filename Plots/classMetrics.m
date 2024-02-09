function classMetrics(C, order, labels, filename)

precision = zeros(length(order), 1);
recall = zeros(length(order), 1);
F1 = zeros(length(order), 1);

for i = 1:length(order)
    TP = C(i, i);
    FP = sum(C(:, i)) - TP;
    FN = sum(C(i, :)) - TP;

    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    F1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

acc = trace(C) / sum(C, 'all');

metricTable = table(char(labels(order)), precision, recall, F1, 'VariableNames', {'Class', 'Precision', 'Recall', 'F1'});
writetable(metricTable, filename);

accTable = table(char(labels(order)), acc, 'VariableNames', {'Accuracy'});
writetable(accTable, filename, "WriteMode","append");
end
