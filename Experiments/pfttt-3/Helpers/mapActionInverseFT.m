function action = mapActionInverseFT(actionLabel)
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
    case "fist"
        action = 1;
    case "noGesture"
        action = 2;
    case "open"
        action = 3;
    case "pinch"
        action = 4;
    case "waveIn"
        action = 5;
    case "waveOut"
        action = 6;
    otherwise
        error('Invalid action label!');
end
end