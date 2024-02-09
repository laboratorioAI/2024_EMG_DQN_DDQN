function [maxWindows,EMG,spectrogram,username,gestureName,groundTruthSize,groundTruth] = readEMG( ...
    EMGUserInfo, ...
    sample, ...
    windowNum, ...
    stride, ...
    windowSize, ...
    isTraining)

username = EMGUserInfo.userData.userInfo.username;
groundTruthSize = 0;
groundTruth = zeros(1000, 1);

if isTraining
    EMGFull = EMGUserInfo.userData.training{sample}.emg;
    gestureName = EMGUserInfo.userData.training{sample}.gestureName;

    if isfield(EMGUserInfo.userData.training{sample}, 'groundTruth')
        %groundTruthIdx = EMGUserInfo.userData.training{sample}.groundTruthIndex;
        groundTruth = EMGUserInfo.userData.training{sample}.groundTruth'; % Transpose just to make it similar to EMG signal :)
        %groundTruthSize = groundTruthIdx(2) - groundTruthIdx(1);
        groundTruthSize = sum(groundTruth);
    end

else
    EMGFull = EMGUserInfo.userData.testing{sample}.emg;
    gestureName = EMGUserInfo.userData.testing{sample}.gestureName;

    if isfield(EMGUserInfo.userData.testing{sample}, 'groundTruth')
        %groundTruthIdx = EMGUserInfo.userData.testing{sample}.groundTruthIndex;
        groundTruth = EMGUserInfo.userData.testing{sample}.groundTruth';
        %groundTruthSize = groundTruthIdx(2) - groundTruthIdx(1);
        groundTruthSize = sum(groundTruth);
    end
end

gtSize = size(groundTruth);
gtPoints = gtSize(1);

emgSize = size(EMGFull);
emgPoints = emgSize(1);

maxWindows = ceil(gtPoints/stride);

if windowNum > maxWindows
    error('windowNum is greater than maxWindow')
end

startValue = (windowNum - 1)*stride + 1;
endValue = startValue + windowSize - 1;

if endValue > gtPoints
    newGroudTruth = [groundTruth;zeros(endValue - gtSize(1,1), 1)];
    groundTruth = newGroudTruth(startValue:endValue);
else
    groundTruth = groundTruth(startValue:endValue);
end

if endValue > emgPoints
    newEMG = [EMGFull;zeros(endValue - emgSize(1, 1), 8)] ;
    EMG = newEMG(startValue:endValue,:);
else
    EMG = EMGFull(startValue:endValue,:);
end

assert(~any(isnan(EMGFull(:))), 'EMGFull contains NaN values.');
assert(~any(isnan(groundTruth(:))), 'groundTruth contains NaN values.');

spectrogram = computeFeatures(EMG, 200, (0:12), 24, 0.5);
assert(~any(isnan(spectrogram(:))), 'Spectrogram contains NaN values.');
end

function featureMat = computeFeatures(EMG, sampleFreq, frequencies, window, overlapping)
% Rectification
signal = abs(EMG);

% Lower-pass filter
[Fb, Fa] = butter(5, 0.1, 'low');
signal = filtfilt(Fb, Fa, signal);

% Spectrogram
featureMat = zeros (13,window,8);
for i = 1:size(signal, 2)
    [s,~,~,~] = spectrogram(signal(:,i), window, floor(window * overlapping), frequencies, sampleFreq, 'yaxis');
    featureMat(:,:,i) = abs(s).^2;
end
end