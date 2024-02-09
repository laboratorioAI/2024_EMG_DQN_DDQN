%example_evalRecognition es un ejemplo básico del uso de la librería para
%evaluación de reconocimiento

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

21 September 2020
Matlab 9.8.0.1417392 (R2020a) Update 4.
%}

%% NOTAS
% evaluamos 3 métricas:
% clasificación
% es 1 etiqueta!
%escalar

% reconocimiento
% es las etiquetas en los instantes de tiempo
% vector con estampa de tiempo

% tiempo de procesamiento
% el tiempo desde que recibió la señal (medición) hasta q imprimió en
% pantalla.
% vector para cada predicción.
% Tiempo real (opciones menor 300ms) (opciónes menor 100ms) con respecto a
% al percepción de la persona.


%% Configuración



%%
% información original (no depende del modelo) (es parte de los datos)
repInfo.gestureName = categorical({'noGesture'});
% categorical es un tipo de dato


repInfo.groundTruth = true(1, 1000); % 1000 xq 200 Hz por 5 segundos


%% predicción
response.vectorOfLabels = categorical({'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture', 'noGesture'});
response.vectorOfTimePoints = 1:40:1000; %1xw double (entero)


% con un tic toc.
% tiempo de procesamiento (segundos)
response.vectorOfProcessingTimes = 0.1 * ones(1, 25); % 1xw double

% no necesariamente depende del vector de arriba
response.class = categorical({'noGesture'}); % adivinamos q es waveIn

%%
r1 = evalRecognition(repInfo, response)

if isempty(r1.recogResult)
    if r1.classResult
    r1.recogResult = true;
    r1.overlappingFactor = 0.25;
    end
end

t = table( ...
    repInfo.gestureName, response.class, r1.classResult, r1.recogResult, r1.overlappingFactor, ...
    'VariableNames', {'actual_class','predicted_class','classification', 'recognition', 'overlapping_factor'})
