function [reward, actualGesture] = basicReward(action, gestureName, groundTruthSize, groundTruth)
actionA = matpAction(cell2mat(action(1)));
actionB = mapAction(cell2mat(action(2)));

rewardA = -1;
rewardB = -1;

if actionA == gestureName
    rewardA = +1;
end

if actionB == gestureName
    rewardB = +1;
end

actualGesture = mapActionInverse(gestureName);
reward = [rewardA, rewardB];
end

