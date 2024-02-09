function [meanVal, stdVal] = plotTiempo(ax, tiempos, options)
%plotTiempo plot the histogram of processing times. Considering +- 3 sigma.
%
% Inputs
%   ax              axes object
%   tiempos         (m, 1) double with times in seconds.
%   options         struct with fields: sigmaWidth,numBins.
%
% Outputs
%   nameUser        char con el nombre de la carpeta del usuario.
%
% Ejemplo
%    = plotTiempo()
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

%% input
switch nargin
    case 1
        ax = prettyFig();
        options.sigmaWidth = 3;
        options.numBins = 50;
    case 2
        options.sigmaWidth = 3;
        options.numBins = 50;
end

%%
tiempos(tiempos == 0) = []; % removiendo ceros
meanVal = mean(tiempos);
stdVal = std(tiempos);

minT = meanVal - options.sigmaWidth * stdVal;
maxT = meanVal + options.sigmaWidth * stdVal;

if minT < 0
    minT = 0;
end


%% limiting
timeVector = tiempos(tiempos > minT & tiempos <= maxT);

h2 = histogram(timeVector, 'Normalization', 'probability');
h2.DataTipTemplate.FontSize = 12;
h2.DataTipTemplate.DataTipRows(1).Label = 'frequency';
h2.DataTipTemplate.DataTipRows(2).Label = 'range';

h2.NumBins = options.numBins;

%%
% std
r = area(ax,[meanVal - stdVal, meanVal + stdVal], max(h2.Values)*[1 1],...
    'FaceAlpha', 0.3);
r.LineStyle = 'none';
r.BaseLine.LineStyle = 'none';
r.PickableParts = 'none';

% mean
xl = xline(ax, meanVal, ':', 'LineWidth', 2);


% legend
legend([r xl], {'avg \pm \sigma', 'avg'}, 'Location', 'best');


%%
xlabel('Processing time [ms]')
ylabel('Normalized frequency')
title('Histogram of processing times')


%%
ax.YAxis.MinorTickValues = unique(h2.Values);
ax.YMinorGrid = 'on';
ax.YMinorTick = 'on';

%%
xlim([minT maxT])
ylim([0 max(h2.Values)])

%%
ax.XTick = sort([minT maxT meanVal meanVal+stdVal meanVal-stdVal]);

%%
tickFun = @(x) sprintf('%g%%', x*100);
ax.YTickLabel = arrayfun(tickFun, ax.YTick, 'UniformOutput', false);

%%
tickFun = @(x) sprintf('%g', x*1000);
ax.XTickLabel = arrayfun(tickFun, ax.XTick, 'UniformOutput', false);
