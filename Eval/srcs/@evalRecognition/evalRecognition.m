classdef evalRecognition
    %evalRecognition is the lib for classification and recognition eval (of
    %one repetition).
    %
    % classResult is true when trueClass is equal to predictedClass.
    % Recognition analysis (calculating the overlapping factor) is carried
    % on only when the equivalent class (got from the predictions vector)
    % is equal to the trueClass, otherwise is zero.
    % In the cases that trueClass is the default gesture, recognition and
    % overlapping factor are empty.
    %
    % Recognition is correct when the overlapping factor is greater than an
    % threshold (default is 25%). The overlapping factor is calculated as
    % follows:
    % overlappingFactor = 2 * A intersection B / (A + B);
    % where: A is the length of the ground truth's area with muscle
    % activity, B is the length of the equivalent vector's area with a
    % gesture detected.
    %
    % Inputs
    %   response        struct with fields:
    %                   vectorOfLabels  1xw categorical
    %                   vectorOfTimePoints 1xw double
    %                   vectorOfProcessingTimes  1xw double
    %                   class           categorical predictedClass.
    %
    %   repInfo         struct with fields:
    %                   gestureName     categorical
    %                   groundTruth     1xw logical set where is muscle
    %                                   activity. The noGesture doesnot
    %                                   have this field.
    %
    %   options         struct with optional fields:
    %                   thresholdRecognition    double, default 0.25.
    %                   defaultGesture          char, name of default
    %                                   gesture, default is "noGesture".
    %                   showWarning     bool, when true shows the message
    %                                   of no default gesture found in
    %                                   vector of labels.
    %
    % Outputs
    %   obj             structh with fields:
    %                   classResult     true when trueClass is equal to
    %                                   predictedClass.
    %                   recogResult     true when the vector of predictions
    %                                   is valid and overlapping factor is
    %                                   greater than threshold.
    %                   overlappingFactor double
    %
    % Example
    %   r1 = evalRecognition(repOrgInfo,response)
    %   r1 =
    %   evalRecognition with properties:
    %
    %              classResult: 1
    %              recogResult: 1
    %        overlappingFactor: 0.7470
    %     thresholdRecognition: 0.2500

    %{
        Laboratorio de Inteligencia y Visión Artificial
        ESCUELA POLITÉCNICA NACIONAL
        Quito - Ecuador

        % V2. 10 January 2020
        Matlab 9.7.0.1261785 (R2019b) Update 3.
    %}
    properties
        classResult
        recogResult
        overlappingFactor

    end
    properties (SetAccess=immutable)
        defaultGesture = categorical({'noGesture'});
        thresholdRecognition = 0.25;
        showWarning = true;
    end
    properties (Hidden=true)
        repInfo;
        response;
        equivalentVector;
    end
    methods
        plotResults(obj)
        %% Constructor
        function obj = evalRecognition(repInfo,response,evalOptions)
            %----% input validation
            % repInfo
            assert(isfield(repInfo, 'gestureName'), "invalid repInfo" + ...
                ", must have the field gestureName")

            % response
            responseFields = {'class', 'vectorOfLabels', ...
                'vectorOfTimePoints','vectorOfProcessingTimes'};
            for kField = responseFields
                assert(isfield(response, kField{:}), "invalid response"+...
                    ", must have the field %s.", kField{:})
            end
            obj.repInfo = repInfo;
            obj.response = response;

            %----% Unpacking
            % ground truth
            if ~isequal(repInfo.gestureName, obj.defaultGesture)
                % transient gestures, should have groundtruth
                assert(isfield(repInfo, 'groundTruth'), ...
                    "invalid repInfo, must have the field groundTruth")
                groundTruth = repInfo.groundTruth;
            end
            % threshold
            if nargin == 3 && isfield(evalOptions,'thresholdRecognition')
                obj.thresholdRecognition =evalOptions.thresholdRecognition;
            end

            % default gesture name
            if nargin == 3 && isfield(evalOptions,'defaultGesture')
                obj.defaultGesture = ...
                    categorical({evalOptions.defaultGesture});
            end

            % showWarning
            if nargin == 3 && isfield(evalOptions,'showWarning')
                obj.showWarning = evalOptions.showWarning;
            end

            % checking
            if ~any(response.vectorOfLabels == obj.defaultGesture)
                if obj.showWarning
                    warning(['Not default gesture named "%s" '...
                        'found in vector of labels.\n' ...
                        'Try changing the default name with the ' ...
                        'defaultGesture option in ' ...
                        'evalRecognition(...,...,evalOptions).\n' ...
                        'Common default names are "relax" and "noGesture".\n' ...
                        'If you want to stop viewing this warning, reset ' ...
                        'the showWarning option, view help.'], ...
                        obj.defaultGesture)
                end
            end
            predictionsVector = response.vectorOfLabels;
            responsePointVector = response.vectorOfTimePoints;

            trueClass = repInfo.gestureName;
            classPrediction = response.class;

            %----% Block representation
            blockClassesPredicted = ...
                evalRecognition.blockRepresentation(predictionsVector);


            %----% Response validity, based on the block representation
            [isPredictionVectorValid,classEquivalent] = ...
                evalRecognition.validateBlock(blockClassesPredicted, ...
                obj.defaultGesture);


            %----% Evaluation
            %------------------ Class eval
            obj.classResult = classPrediction == trueClass;

            %----------------- Recog eval
            %---% noGesture considerations
            if trueClass == obj.defaultGesture
                % noGesture gesture! XD
                obj.overlappingFactor = [];
                obj.recogResult = [];

            else
                % some gesture gesture
                if isPredictionVectorValid && (classEquivalent==trueClass)
                    % do the recognition

                    %---% get equilvant
                    equivalentVector = ...
                        evalRecognition.getEquivalentPredictionVector(...
                        groundTruth,predictionsVector,responsePointVector);


                    %---% overlapping factor
                    obj.overlappingFactor = ...
                        evalRecognition.calculateOverlappingFactor(...
                        groundTruth,equivalentVector,trueClass);

                    obj.recogResult = ...
                        obj.overlappingFactor >= obj.thresholdRecognition;
                    obj.equivalentVector = equivalentVector;
                else
                    % wrong recognition
                    % class equivalent does n't match true Class, or vector
                    % of predictions not valid.
                    obj.overlappingFactor = nan;
                    obj.recogResult = false;
                end
            end
        end
    end

    methods (Static)
        %%
        function [blockClassesPredicted,freqClass] = ...
                blockRepresentation(predictionsVector)
            %blockRepresentation returns freqClass an array with the number
            %of windows a class appeared. blockClassesPredicted is a vector
            %of contiguous classes.
            %
            % Inputs
            %   predictionsVector   (categorical Wx1)

            %---% validation
            numPredictions = numel(predictionsVector);

            if numPredictions < 1 % validation
                error("Error in blockRepresentation." + ...
                    "Not valid size of predictions!")
            end

            % counts the number of times a class appeared
            freqClass = 1;
            blockClassesPredicted = predictionsVector(1);

            for kPrediction = predictionsVector(2:end)
                if kPrediction == blockClassesPredicted(end)
                    % add to freq vector
                    freqClass(end) = freqClass(end) + 1;
                else
                    % create new block
                    freqClass(end + 1) = 1;
                    blockClassesPredicted(end + 1) = kPrediction;
                end
            end
        end

        function [isPredictionVectorValid, classEquivalent] = ...
                validateBlock(blockClassesPredicted, defaultGesture)
            %validateBlock return if is valid the blockClassesPredicted, in
            %the case it is, returns the classEquivalent.
            %
            % Inputs
            %   blockClassesPredicted   (categorical wx1)
            %   defaultGesture          (char) name of default gesture

            classEquivalent = [];
            if nargin == 1
                % for backwards compatibility
                defaultGesture = categorical({'noGesture'});
            end
            if ~iscategorical(defaultGesture)
                defaultGesture = categorical({defaultGesture});
            end

            switch numel(blockClassesPredicted)
                case 1
                    %-------------------------------
                    % 1 element in the block representation
                    % Case 1 (all predictions correspond to one class)
                    classEquivalent = blockClassesPredicted(1);
                    isPredictionVectorValid = true;

                case 2
                    %-------------------------------
                    % 2 elements in the block representation

                    classesPred = blockClassesPredicted(...
                        blockClassesPredicted ~= defaultGesture);
                    if numel(classesPred) == 1
                        % one of the 2 predictions is noGesture, other is
                        % class.
                        classEquivalent = classesPred;
                        isPredictionVectorValid = true;

                    else
                        % two no noGesture classes, invalid
                        isPredictionVectorValid = false;
                    end


                case 3
                    %-------------------------------
                    % 3 elements in the block representation
                    if blockClassesPredicted(1)==defaultGesture && ...
                            blockClassesPredicted(3) == defaultGesture
                        % gesture in the middle with relaxes at the sides
                        classEquivalent = blockClassesPredicted(2);
                        isPredictionVectorValid = true;

                    else
                        isPredictionVectorValid = false;
                    end


                otherwise
                    %-------------------------------
                    % more elements in the block representation!
                    % Case 7. a lot of stuff, really, really wrong
                    isPredictionVectorValid = false;
            end
        end

        function equivalentVector = getEquivalentPredictionVector(...
                groundTruth,predictionsVector,responsePointVector, ...
                defaultGesture)
            %getEquivalentPredictionVector(...) returns a vector of the
            %same size as groundTruth equivalent to predictionsVector and
            %responsePointVector.
            
            if nargin == 3
                % for backwards compatibility
                defaultGesture = categorical({'noGesture'});
            end
            if ~iscategorical(defaultGesture)
                defaultGesture = categorical({defaultGesture});
            end

            numPredictions = numel(predictionsVector);
            assert(numPredictions == numel(responsePointVector), ...
                "predictionsVector and responsePointVector must have" + ...
                "the same size!")

            tamGT = numel(groundTruth);

            equivalentVector = repmat(defaultGesture,1,tamGT);

            kPrediction = 1;
            for kGT = 1:tamGT
                % point alignment
                if kGT > responsePointVector(kPrediction)
                    % saturate! GT larger than predictions.
                    if kPrediction < numPredictions
                        kPrediction = kPrediction + 1;
                    end
                end
                equivalentVector(kGT) = predictionsVector(kPrediction);
            end
        end

        function overlappingFactor = calculateOverlappingFactor(...
                groundTruth,equivalentVector,trueClass)
            %-----------------------------------------
            % overlappingFactor = 2 * A intersection B / (A + B);
            % lengthCorrectGT <= A
            % lengthResp <= B
            % intersectionArea <= A intersection B

            lengthCorrectGT = sum(groundTruth);
            lengthResp = sum(equivalentVector == trueClass);
            intersectionArea = sum(groundTruth & ...
                equivalentVector == trueClass);

            overlappingFactor = 2*intersectionArea/...
                (lengthCorrectGT+lengthResp);
        end
    end
end