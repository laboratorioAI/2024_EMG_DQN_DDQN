function action = mapActionInverse(actionLabel)
% mapActionInverse - Map action label to corresponding numeric action code.
%
%   action = mapActionInverse(actionLabel)
%
%   Input:
%       - actionLabel: Action label.
%
%   Output:
%       - action: Numeric code representing the action.
%

switch actionLabel
    case "waveIn"
        action = 1;
    case "waveOut"
        action = 2;
    case "fist"
        action = 3;
    case "open"
        action = 4;
    case "pinch"
        action = 5;
    case "noGesture"
        action = 6;
    otherwise
        error('Invalid action label!');
end
end