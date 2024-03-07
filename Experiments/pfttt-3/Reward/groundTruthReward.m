function [reward, actualGesture] = groundTruthReward(action, gestureName, groundTruthSize, groundTruth)
actionA = mapAction(cell2mat(action(1)));
actionB = mapAction(cell2mat(action(2)));

gtZeros = groundTruth == 0;

if groundTruthSize == 0
    rewardA = getReward(actionA, "noGesture", +0.50);
    rewardB = getReward(actionB, "noGesture", +0.50);
    actualGesture = mapActionInverse("noGesture");
elseif all(gtZeros, "all")
    rewardA = getReward(actionA, "noGesture", +0.25);
    rewardB = getReward(actionB, "noGesture", +0.25);
    actualGesture = mapActionInverse("noGesture");
else
    gtOnes = groundTruth == 1;
    actualGesture = mapActionInverse(gestureName);
    if all(gtOnes, "all")
        futureReward = +1;  
    else
        futureReward = 2 - (sum(gtOnes) / groundTruthSize);
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
