function [reward, actualGesture] = groundTruthReward(action, gestureName, groundTruthSize, groundTruth)
actionA = mapAction(cell2mat(action(1)));
actionB = mapAction(cell2mat(action(2)));

gtZeros = groundTruth == 0;

if all(gtZeros)
    rewardA = getReward(actionA, "noGesture", +0.25);
    rewardB = getReward(actionB, "noGesture", +0.25);
    actualGesture = mapActionInverse("noGesture");
else
    gtOnes = groundTruth == 1;
    actualGesture = mapActionInverse(gestureName);
    if all(gtOnes)
        futureReward = +1;
    elseif groundTruthSize == 0
        futureReward = +0.25;
        gestureName = "noGesture";
        actualGesture = mapActionInverse(gestureName);
    else
        futureReward = sum(gtOnes) / groundTruthSize;
    end
    rewardA = getReward(actionA, gestureName, futureReward);
    rewardB = getReward(actionB, gestureName, futureReward);
end

reward = [rewardA, rewardB];
end

function reward = getReward(action, expectedAction, defaultReward)
if action == expectedAction
    reward = defaultReward;
else
    reward = -1;
end
end
