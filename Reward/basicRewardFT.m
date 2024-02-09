function [reward, actualGesture] = basicRewardFT(action, gestureName, groundTruthSize, groundTruth)
actionA = action(1);
actionB = action(2);

rewardA = -1;
rewardB = -1;

if actionA == gestureName
    rewardA = +1;
end

if actionB == gestureName
    rewardB = +1;
end

actualGesture = mapActionInverseFT(gestureName);
reward = [rewardA, rewardB];
end

