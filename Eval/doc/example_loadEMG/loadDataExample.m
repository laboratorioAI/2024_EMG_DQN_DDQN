%% user data 
% default folder
nameUser = 'user59';
[userData, userGroup] = getUser(nameUser);

% userGroup is a string either 'training' or 'testing'. The training user
% group contains users from 1 to 60. The testing user group contains users
% from 61 to 120.
userGroup = 'training';
% change this directory to the full 612 user dataset.
datasetFolder = '.\incompleteDataset\'; 
myUser = ReadUser(28, userGroup, datasetFolder);

% user info. 
% ---------------
% Fields available: age, gender, handedness, ethnicGroup,
% hasSufferedArmDamage, distances fromElbowToMyo, fromElbowToUlna,
% armPerimeter.
age = userData.userInfo.age
gender = userData.userInfo.gender

% data info. 
% ---------------
% Fields available: myoName (3 different myos were used for data
% acquisition), acquisitionDate, secondsPerSample, classesLabels,
% samplesPerClass.

myoName = userData.dataInfo.myoName
dateMeasure = userData.dataInfo.acquisitionDate


% user data with given folder
% ---------------
folderData = '.\data\';
nameUser = 'user1';

[userData, userGroup] = getUser(nameUser, folderData);


%% Gesture data
nameUser = 'user62';
nameGesture = 'waveOut';
repGroup = 'training';

allWaveOuts = getUserGesture(nameUser, nameGesture, repGroup)

thirdWaveOut = allWaveOuts{3}


%% rep
% known data info
%------------------

nameUser = 'user1';
repGroup = 'testing';
kRep = 45;

repData = getUserRep(nameUser, repGroup, kRep) 



% unknown data info
%------------------
nameUser = 'user105';
repGroup = 'testing';
kRep = 45;

repData = getUserRep(nameUser, repGroup, kRep) 



% repetition info available: emg, poseMyo, rot, label



