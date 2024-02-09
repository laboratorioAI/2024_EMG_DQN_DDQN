function actionLabel = mapAction(action)
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
        actionLabel = "waveIn";
    case 2
        actionLabel = "waveOut";
    case 3
        actionLabel = "fist";
    case 4
        actionLabel = "open";
    case 5
        actionLabel = "pinch";
    case 6
        actionLabel = "noGesture";
    otherwise
        error('There is something wrong!');
end
end