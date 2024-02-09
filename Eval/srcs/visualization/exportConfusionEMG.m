function mat = exportConfusionEMG(targets, outputs, options)
%exportConfusionEMG exports the confusion matrix to latex format.
%
% Inputs
%   targets         (m, 1) categorical of true classes.
%   outputs         (m, 1) categorical of predicted classes.
%   options         struct with options: outputFolder.
%
% Outputs
%   nameUser        char con el nombre de la carpeta del usuario.
%
% Ejemplo
%    = exportConfusionEMG()
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org
Cuando escribí este código, solo dios y yo sabíamos como funcionaba.
Ahora solo lo sabe dios.

"When a measure becomes a target, it ceases to be a good measure"
-Charles Goodhart.

04 June 2020
Matlab 9.7.0.1261785 (R2019b) Update 3.
%}


%%
classes = {'waveIn', 'waveOut', 'fist', 'open', 'pinch', 'noGesture'};

mat = zeros(6, 6);
for idxOutput = 1:6
    kOutput = classes{idxOutput};
        
    predsG = outputs == kOutput;
    
    for idxTarget = 1:6
        kTarget = classes{idxTarget};
        
        truesG = targets == kTarget;
        
        mat(idxOutput, idxTarget) = sum(predsG & truesG);
    end
end

%% metrics
% sensitivity
totalCols = sum(mat, 1);
porcentCols = diag(mat)'./totalCols*100;

% precision
totalRows = sum(mat, 2);
porcentRows = diag(mat)./totalRows*100;

total = sum(mat, 'all');
percentTotal = sum(diag(mat))/total*100;


%% send latex
outputFile = [options.outputFolder '\confusion.tex'];

confusion2latex(outputFile, mat, totalCols, porcentCols...
    , totalRows, porcentRows, total, percentTotal)
