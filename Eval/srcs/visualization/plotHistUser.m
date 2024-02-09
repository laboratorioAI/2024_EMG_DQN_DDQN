function plotHistUser(ax, vals, titulo)
% plotAccXUser returns a fig and an axes object with font 20 and holded on.
%
% Outputs
% ax        axes object.
% f         figure object.
%
% Ejemplo
%     = prettyFig(options)
%
%%
meanVal = mean(vals);
stdVal = std(vals);

h1 = histogram(ax, vals, 'Normalization', 'probability');
h1.DataTipTemplate.FontSize = 12;
if h1.NumBins < 10 % too few bins...
    h1.NumBins = 10;
end
h1.DataTipTemplate.DataTipRows(1).Label = 'frequency';
h1.DataTipTemplate.DataTipRows(2).Label = 'range';

h1.BinLimits = [min(h1.BinLimits) 100];

% std
r = area(ax,[sat(meanVal - stdVal), sat(meanVal + stdVal)], ...
    max(h1.Values)*[1 1], 'FaceAlpha', 0.3);
r.LineStyle = 'none';
r.BaseLine.LineStyle = 'none';
r.PickableParts = 'none';

% mean
xl = xline(ax, meanVal, ':', 'LineWidth', 2);


% ticks
ax.YAxis(1).MinorTickValues = unique(h1.Values);
ax.YMinorGrid = 'on';
ax.YMinorTick = 'on';

limsX = [h1.BinLimits, meanVal, sat(meanVal + stdVal), ...
    sat(meanVal - stdVal)];

limsX = unique(limsX);
ax.XTick = limsX;

%config

xtickformat(ax, '%.2g%%')
ylabel(ax, 'Normalized frequency')
xlabel(ax, 'Accuracy')
title(ax, titulo)

drawnow
% cumulative
ax.ColorOrderIndex = 1; % changing color in line colormap
yyaxis(ax, 'right')
% ax.YColor = [0 0 0];
[B, GB] = groupcounts(vals);
cumuls = cumsum(B)/numel(vals) * 100;

p1 = plot(ax, GB, cumuls, 'LineWidth', 2);
p1.DataTipTemplate.FontSize = 12;
p1.DataTipTemplate.DataTipRows(1).Label = 'Accuracy';
p1.DataTipTemplate.DataTipRows(1).Format = '%.2g%%';
p1.DataTipTemplate.DataTipRows(2).Label = 'Cumulative density';
p1.DataTipTemplate.DataTipRows(2).Format = '%.2g%%';
p1.Color(4) = 0.75;
ytickformat(ax, '%d%%')
% ylabel(ax, 'Cumulative density')

ax.YAxis(2).MinorTickValues = cumuls;
ax.YAxis(2).MinorTick = 'on';

% legend
% drawnow
yyaxis(ax, 'left')
legend([r xl p1], {'avg \pm \sigma', 'avg', 'cumulative'}, ...
    'Location', 'best');

%%
tickFun = @(x) sprintf('%g%%', x*100);
ax.YTickLabel = arrayfun(tickFun, ax.YTick, 'UniformOutput', false);

end