function plotSummaryResults(respXUser, ...
    classAvg, classStd, recogAvg, recogStd, tiempos, ...
    targets, outputs, options)
%plotSummaryResults creates the figures of the results for a given HGR
%model.
%
% Inputs
%   nameUser        char con el nombre de la carpeta del usuario.
%
% Outputs
%   nameUser        char con el nombre de la carpeta del usuario.
% 
% Ejemplo
%    = plotSummaryResults()
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org
Cuando escribí este código, solo dios y yo sabíamos como funcionaba.
Ahora solo lo sabe dios.

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

17 August 2020
Matlab 9.8.0.1417392 (R2020a) Update 4.
%}


%% plotting
[ax, fClass] = prettyFig();
plotAccXUser(ax, respXUser.class, classAvg, classStd, ...
    'Classification Accuracy');

[ax, fClassHist] = prettyFig();
plotHistUser(ax, respXUser.class, ...
    'Histogram of classification accuracy per user');

%
[ax, fRecog] = prettyFig();
plotAccXUser(ax, respXUser.recog, recogAvg, recogStd, ...
    'Recognition Accuracy');

[ax, fRecogHist] = prettyFig();
plotHistUser(ax, respXUser.recog, ...
    'Histogram of recognition accuracy per user');


%% tiempos
% hacer aparte...
[ax, fTiempo] = prettyFig();
plotTiempo(ax, tiempos);

%%  confusion
[~, fConfusion] = prettyFig();

plotconfusion(targets, outputs);


%%
if options.save
    savefig(fClass, [options.outputFolder '\class.fig'])
    savefig(fClassHist, [options.outputFolder '\classHist.fig'])
    savefig(fRecog, [options.outputFolder '\recog.fig'])
    savefig(fRecogHist, [options.outputFolder '\recogHist.fig'])
    
    saveas(fClass, [options.outputFolder '\class.png'])
    saveas(fClassHist, [options.outputFolder '\classHist.png'])
    saveas(fRecog, [options.outputFolder '\recog.png'])
    saveas(fRecogHist, [options.outputFolder '\recogHist.png'])
    
    saveas(fConfusion, [options.outputFolder '\confusion.png'])
    savefig(fConfusion, [options.outputFolder '\confusion.fig'])
    
    savefig(fTiempo, [options.outputFolder '\tiempos.fig'])    
    saveas(fTiempo, [options.outputFolder '\tiempos.png'])
end