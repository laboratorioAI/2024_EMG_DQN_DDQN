function [] = json2mat()
% JSON2MAT - Converts gesture data from JSON to MAT format for both training and testing data.
%
%   This function processes JSON files containing gesture data, converts them
%   into a structured MATLAB format, and saves the resulting data in separate
%   folders for training (Data/Training) and testing (Data/Testing).
%   WARNING: It uses parallelization.
%
%   Usage:
%       json2mat()
%
%   Note: This function assumes the existence of 'trainingJSON' and 'testingJSON'
%   folders containing JSON data in the current directory.
%
%   See also processData, processUserData.

clc
close all

% Enable parallel pool if not already started
if isempty(gcp('nocreate'))
    parpool();
end

% Define folder types
folderTypes = ["training", "testing"];

% Process data in parallel for both training and testing
parfor folderIndex = 1:length(folderTypes)
    userFolder = char(folderTypes(folderIndex));
    processData(userFolder);
    disp(userFolder + " data conversion done!");
end

% If you started the parallel pool in this function, you may want to shut it down
delete(gcp('nocreate'));
end

function processData(userFolder)
% PROCESSDATA - Process JSON files in the specified user folder and convert to MAT format.
%
%   This function iterates through JSON files in the specified user folder,
%   converts the data to MATLAB format, and saves it in a new folder.
%
%   Usage:
%       processData(userFolder)
%
%   Input:
%       - userFolder: 'training' or 'testing'
%
%   See also json2mat, processUserData.

folderData = [userFolder 'JSON'];
filesInFolder = dir(folderData); 
numFiles = length(filesInFolder);
userProcessed = 0;

disp("Files to be processed: " + numFiles);
for user_i = 1:numFiles
    if ~(strcmpi(filesInFolder(user_i).name, '.') || strcmpi(filesInFolder(user_i).name, '..') || strcmpi(filesInFolder(user_i).name, '.DS_Store'))
        userProcessed = userProcessed + 1;
        file = [folderData '/' filesInFolder(user_i).name '/' filesInFolder(user_i).name '.json'];
        current_file = horzcat(file);
        disp("Current file: " + current_file);
        userName=filesInFolder(user_i).name;
        text = fileread(file);
        
        processUserData(userFolder, userName, text, strcmpi(userFolder, 'training'));
    end
end
end

function processUserData(userFolder, userName, text, forTraining)
% PROCESSUSERDATA - Convert JSON data to MAT format and save in a new folder.
%
%   This function decodes JSON data, organizes it into a MATLAB data
%   structure, and saves the resulting data in a new folder.
%
%   Usage:
%       processUserData(userFolder, userName, text, forTraining)
%
%   Input:
%       - userFolder: 'training' or 'testing'
%       - userName: Name of the user
%       - text: JSON data in string format
%       - forTraining: Boolean indicating whether the data is for training
%
%   See also json2mat, processData.

user = jsondecode(text);
userData = user;

folder=[pwd,'\','Data','\', upper(userFolder(1)), userFolder(2:end), '\', userName];
saveFile = horzcat(folder, '\', 'userData.mat');
mkdir(folder)
save(saveFile,'userData');

userData.sync        = userData.synchronizationGesture;
userData.training    = userData.trainingSamples;
userData.testing     = userData.testingSamples;

userData  = rmfield(userData,'synchronizationGesture');
userData  = rmfield(userData,'trainingSamples');
userData  = rmfield(userData,'testingSamples');
userData  = rmfield(userData,'generalInfo');

userData.userInfo.username = userData.userInfo.name;

userData.sync     = struct2cell(userData.sync);
aux               = length(fieldnames(userData.sync{1,1}));

for o=1:aux
    index=char(strcat('idx_',string(o)));
    rx=userData.sync{1,1}.(index);
    emg_=[rx.emg.ch1,rx.emg.ch2,rx.emg.ch3,rx.emg.ch4,rx.emg.ch5,rx.emg.ch6,rx.emg.ch7,rx.emg.ch8]/128;

    userData.sync_{o,1}.emg                = emg_;
    userData.sync_{o,1}.pointGestureBegins = rx.startPointforGestureExecution;
    userData.sync_{o,1}.pose_myo           = rx.myoDetection;
    userData.sync_{o,1}.gyro               = [rx.gyroscope.x,rx.gyroscope.y,rx.gyroscope.z];
    userData.sync_{o,1}.accel              = [rx.accelerometer.x,rx.accelerometer.y,rx.accelerometer.z];
end

userData.sync        = userData.sync_;
userData  = rmfield(userData,'sync_');

userData.training = struct2cell(userData.training);

for o=1:150
    rx=userData.training{o,1};
    emg_=[rx.emg.ch1,rx.emg.ch2,rx.emg.ch3,rx.emg.ch4,rx.emg.ch5,rx.emg.ch6,rx.emg.ch7,rx.emg.ch8]/128;

    userData.train_{o,1}.emg                = emg_;
    userData.train_{o,1}.pointGestureBegins = rx.startPointforGestureExecution;
    userData.train_{o,1}.pose_myo           = rx.myoDetection;
    userData.train_{o,1}.gyro               = [rx.gyroscope.x,rx.gyroscope.y,rx.gyroscope.z];
    userData.train_{o,1}.accel              = [rx.accelerometer.x,rx.accelerometer.y,rx.accelerometer.z];
    userData.train_{o,1}.gestureName        = categorical(string(rx.gestureName));

    if string(rx.gestureName)=="noGesture"
    else
        userData.train_{o,1}.groundTruthIndex   = (rx.groundTruthIndex)';
        userData.train_{o,1}.groundTruth        = (logical(rx.groundTruth))';
    end
end

userData.training   = userData.train_;
userData            = rmfield(userData,'train_');

userData.testing  = struct2cell(userData.testing);

for o=1:150
    rx=userData.testing{o,1};
    emg_=[rx.emg.ch1,rx.emg.ch2,rx.emg.ch3,rx.emg.ch4,rx.emg.ch5,rx.emg.ch6,rx.emg.ch7,rx.emg.ch8]/128;

    userData.test_{o,1}.emg                = emg_;
    userData.test_{o,1}.pointGestureBegins = rx.startPointforGestureExecution;
    userData.test_{o,1}.pose_myo           = rx.myoDetection;
    userData.test_{o,1}.gyro               = [rx.gyroscope.x,rx.gyroscope.y,rx.gyroscope.z];
    userData.test_{o,1}.accel              = [rx.accelerometer.x,rx.accelerometer.y,rx.accelerometer.z];

    % NOTE: The original code skips this when the dataset is for testing.
    if forTraining
        userData.test_{o,1}.gestureName        = categorical(string(rx.gestureName));
        if string(rx.gestureName)=="noGesture"
        else
            userData.test_{o,1}.groundTruthIndex   = (rx.groundTruthIndex)';
            userData.test_{o,1}.groundTruth        = (logical(rx.groundTruth))';
        end
    end
end

userData.testing   = userData.test_;
userData           = rmfield(userData,'test_');

save(saveFile,'userData');
end