function [reward, actualGesture] = groundTruthReward(action, gestureName, groundTruthSize, groundTruth)
% groundTruthReward - Calculate reward based on ground truth and actions.
%
%   reward = groundTruthReward(action, gestureName, groundTruthSize, groundTruth)
%
%   Inputs:
%       - action: A cell array containing two actions.
%       - gestureName: Name of the gesture to compare with actions.
%       - groundTruthSize: How many 1's are there in the original signal?
%         NOTE: what if the number of 1's is greater than the window?
%         NOTE: what if the number os 1's uses two or more windows?
%       - groundTruth: Ground truth vector indicating gestures.
%
%   Output:
%       - reward: A vector containing rewards for each action.
%
actionA = mapAction(cell2mat(action(1)));
actionB = mapAction(cell2mat(action(2)));

gtZeros = groundTruth == 0;

if all(gtZeros)
    rewardA = getReward(actionA, "noGesture", +1);
    rewardB = getReward(actionB, "noGesture", +1);
    actualGesture = mapActionInverse("noGesture");
else
    gtOnes = groundTruth == 1;
    if all(gtOnes)
        futureReward = +1;
    else
        futureReward = sum(gtOnes) / groundTruthSize;
    end
    rewardA = getReward(actionA, gestureName, futureReward);
    rewardB = getReward(actionB, gestureName, futureReward);
    actualGesture = mapActionInverse(gestureName);
end

reward = [rewardA, rewardB];
end

function reward = getReward(action, expectedAction, defaultReward)
% getReward - Calculate reward based on the match between action and expected action.
%
%   reward = getReward(action, expectedAction, defaultReward)
%
%   Inputs:
%       - action: Actual action to compare.
%       - expectedAction: Expected action for comparison.
%       - defaultReward: Default reward if actions do not match.
%
%   Output:
%       - reward: Calculated reward based on the match.
%
if action == expectedAction
    reward = defaultReward;
else
    reward = -1;
end
end
