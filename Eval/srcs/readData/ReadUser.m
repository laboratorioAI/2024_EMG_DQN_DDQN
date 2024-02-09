classdef ReadUser
    %ReadUser is a class for reading the data of certain user. It contains
    %the userData property with all the information. It also has methods
    %for accesing the gesture repetitions.
    %
    %   myUser = ReadUser(28, 'training','.\data\')
    %      ReadUser with properties:
    %
    %     userData: [1×1 struct]
    %     gestureNames: {1×6 cell}
    %
    %   myUser.userData
    %      struct with fields:
    %
    %      extraInfo: [1×1 struct]
    %     university: 'EPN'
    %       userInfo: [1×1 struct]
    %           sync: {5×1 cell}
    %       training: {150×1 cell}
    %        testing: {150×1 cell}
    %
    %
    %   myUser.getRepetitions();
    %
    % Constructor ----------------
    %  ReadUser(userNum, userGroup, dataFolder)
    %   userNum         int between 1 and 306.
    %   userGroup       'traininig' or 'testing'
    %   dataFolder      relative or absolute path to the folder.
    %
    %
    % Properties ----------------
    %   userData        structure with fields as specified in readme.txt.
    %   gestureNames    1×6 cell with the name of the gestures
    %                   waveIn, waveOut, fist, open, pinch, sync,noGesture.
    %
    % Methods ----------------
    %  ReadUser.getRepetitionsOfGesture(gestureName) returns a 25x1 cell
    %  with the repetitions of the specified gestureName {waveIn, waveOut,
    %  fist, open, pinch, noGesture} ONLY of the training repetition group.
    %
    %  ReadUser.getAllRepetitions(repetitionGroup) returns a 150x1 cell
    %  with the repetitions of the specified repetitionGroup ('traininig'
    %  or 'testing').
    %
    %  ReadUser.getTestingRepetitions() returns a 150x1 cell with the
    %  repetitions of the testing repetition group.
    
    %{
    Laboratorio de Inteligencia y Visión Artificial
    ESCUELA POLITÉCNICA NACIONAL
    Quito - Ecuador

    7 January 2020
    Matlab 9.7.0.1261785 (R2019b) Update 3.
    v1
    %}
    
    properties
        userData
        gestureNames = {'waveIn','waveOut','fist','open','pinch',...
            'noGesture','sync'};
    end
    properties (Access = private)
        numRepsInRepGroup = 25;
    end
    
    methods
        %% Constructor
        function obj = ReadUser(userNum, userGroup, dataFolder)
            %-------------% data validation
            % userNum range from 1 to 306.
            assert(userNum == floor(userNum) ...
                & userNum > 0 & userNum <= 306, ...
                'userNum must be integer between 1 and 306, got %d', ...
                userNum)
            
            % training or testing
            assert(isequal(userGroup, 'training') ...
                | isequal(userGroup, 'testing'), ...
                'userGroup must be either training or testing')
            
            % file
            fileName = sprintf('%s\\%s\\user%d\\userData.mat', ...
                dataFolder, userGroup, userNum);
            assert(isfile(fileName), ...
                'file %s could not be found', fileName);
            
            %------------% read file
            f = load(fileName);
            obj.userData = f.userData;
        end
        
        %% methods
        function classReps = getRepetitionsOfGesture(obj, gestureName)
            switch gestureName
                case 'sync'
                    classReps = obj.userData.sync;
                    
                case {'waveIn', 'waveOut','fist', 'open', 'pinch','noGesture'}
                    classReps = cell(obj.numRepsInRepGroup, 1);
                    it = 0;
                    for kRep = obj.userData.training'
                        if isequal(kRep{:}.gestureName, gestureName)
                            it = it + 1;
                            classReps{it} = kRep{:};
                        end
                    end
                    
                otherwise
                    error('Invalid gestureName %s.', gestureName)
            end
        end
        
        
        function repetitions = getAllRepetitions(obj, repetitionGroup)
            % training or testing
            assert(isequal(repetitionGroup, 'training') ...
                | isequal(repetitionGroup, 'testing'), ...
                'repetitionGroup must be either training or testing')
            
            % get reps
            repetitions = obj.userData.(repetitionGroup);
        end
        
        function repetitions = getTestingRepetitions(obj)
            % get reps
            repetitions = obj.userData.testing;
        end        
    end
end