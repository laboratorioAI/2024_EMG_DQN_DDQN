
function plotAccXUser(ax, vals, meanVal, stdVal, titulo)
% plotAccXUser returns a fig and an axes object with font 20 and holded on.
%
% Outputs
% ax        axes object.
% f         figure object.
%
% Ejemplo
%     = prettyFig(options)
%

numUsers = numel(vals);

% users
s = plot(ax, vals, '.');
s.DataTipTemplate.FontSize = 12;
s.DataTipTemplate.DataTipRows(1).Label = 'user';
s.DataTipTemplate.DataTipRows(2).Label = 'accuracy';
s.DataTipTemplate.DataTipRows(2).Format = '%.2g%%';
s.MarkerSize = 12;

% std
r = area(ax, [1 numUsers], (sat(meanVal + stdVal)) *[1 1], ...
    sat(meanVal - stdVal), 'FaceAlpha', 0.3); %  'LineWidth' , 2,
r.LineStyle = 'none';
r.BaseLine.LineStyle = 'none';
r.PickableParts = 'none';
% mean
yline(ax, meanVal, ':', 'LineWidth', 2);

% ticks y
ytickformat(ax, '%.2g%%')
limsY = [minmax(vals'), meanVal, sat(meanVal + stdVal),...
    sat(meanVal - stdVal)];
limsY = unique(limsY);
ax.YTick = limsY;

% legend
drawnow
legend(ax, {'user', 'avg \pm \sigma', 'avg'}, 'Location', 'best');

%config
ax.XAxis.MinorTickValues = 1:numUsers;
ax.XMinorGrid = 'on';
ax.XTick = [];
xlim(ax, [1 numUsers])

% xlabel(ax, 'User')
title(ax, titulo)
drawnow
end