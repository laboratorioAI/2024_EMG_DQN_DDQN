function actionLabel = mapActionFT(action)
% mapAction - Map numeric action code to corresponding action label.
%
%   actionLabel = mapAction(action)
%
%   Input:
%       - action: Numeric code representing an action.
%
%   Output:
%       - actionLabel: Corresponding action label.
%
switch action
    case 1
        actionLabel = "fist";
    case 2
        actionLabel = "noGesture";
    case 3
        actionLabel = "open";
    case 4
        actionLabel = "pinch";
    case 5
        actionLabel = "waveIn";
    case 6
        actionLabel = "waveOut";
    otherwise
        error("There is something wrong!");
end
end