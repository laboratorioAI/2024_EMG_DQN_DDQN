function plotResults(obj)
%plotResults is method of the class evalRecognition.
%
% Inputs
%   obj             an evalRecognition object.
%
% Example
%   r1 = evalRecognition(repOrgInfo,response);
%   r1.plotResults()
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

12 January 2020
Matlab 9.7.0.1261785 (R2019b) Update 3.
%}

defaultGesture = obj.defaultGesture;
responsePointVector = obj.response.vectorOfTimePoints;
predictionsVector = obj.response.vectorOfLabels;
classPrediction = obj.response.class;
trueClass = obj.repInfo.gestureName;


% groundTruth in the case of relax! (just for plot)
if ~isfield(obj.repInfo,'groundTruth')
    
    groundTruth = false(responsePointVector(end),1);
        
else
    groundTruth = obj.repInfo.groundTruth;    
end

equivalentVector = evalRecognition.getEquivalentPredictionVector(...
        groundTruth,predictionsVector,responsePointVector);
    
lengthGroundTruth = numel(groundTruth);
gt2 = repmat(defaultGesture,1,lengthGroundTruth);
gt2(~groundTruth) = defaultGesture;
gt2(groundTruth) = trueClass;
groundTruth = gt2;

%% set up
myCanva = figure;
responseAxes = axes('Parent', myCanva);
responseAxes.XGrid = 'on';
hold(responseAxes, 'on');

xlabel(responseAxes,'Time [points]')
ylabel(responseAxes,'Classes')



%% basic Plots
% predictions Vector
%-----------------------------------------
plot(responseAxes, equivalentVector, 'linewidth', 2,...
    'Color', [0.85 0.325 0.098])


% ground Truth
%-----------------------------------------
plot(responseAxes, groundTruth, 'linewidth', 2,...
    'Color', [102 204 255] / 255)

% true class!
%-----------------------------------------
text(0, defaultGesture, sprintf('trueClass %s', trueClass),...
    'fontsize', 20,'fontweight', 'bold', 'HorizontalAlignment', 'left',...
    'Color', [102 204 255] / 255)


% predicted classes!
%-----------------------------------------
text(0, classPrediction, sprintf('classPredicted: %s',classPrediction),...
    'fontsize', 20,'fontweight', 'bold', 'HorizontalAlignment', 'left',...
    'color',[0.85 0.325 0.098])


%% Classification and recognition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if obj.classResult
    % text correct clasification
    text(lengthGroundTruth, classPrediction, 'Classification: correct!',...
        'fontsize', 15, 'fontweight', 'bold',...
        'HorizontalAlignment', 'right', 'Color', [051 204 051]/ 255)
    
else
    % text wrong classification
    text(lengthGroundTruth, classPrediction, 'Classification: wrong!',...
        'fontsize', 15, 'fontweight', 'bold',...
        'Color', [204 051 051]/ 255, 'HorizontalAlignment', 'right')
end



% recognition
%-----------------------------------------
if isempty(obj.recogResult)
    % text correct recognition
    cColor = [0 0 1];
    msgRecog = 'notCalculated';
    oFText = 'notCalculated';
    
elseif obj.recogResult
    % text correct recognition
    cColor = [051 204 051] / 255;
    msgRecog = 'correct!';
    oFText = num2str(obj.overlappingFactor);
else
    % text wrong recognition
    cColor = [204 051 051] / 255;
    msgRecog = 'wrong!';
    
    oFText = '0. notCalculated';
end

% overlapping factor
%-----------------------------------------
title(['Recognition: ' msgRecog ', overlapping factor: ', oFText],...
    'FontSize', 17, 'FontWeight', 'bold',...
    'Color', cColor)

legend('equivalentVector', 'groundTruth')

end