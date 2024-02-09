function [ax, f] = prettyFig()
% prettyFig returns a fig and an axes object with font 20 and holded on.
%
% Outputs
% ax        axes object.
% f         figure object.
%
% Ejemplo
%     = prettyFig()
%

%%
set(0,'defaultAxesFontSize', 20)
f = figure();

% if options.visibleFig
%     f = figure();
% else
%     f = figure('Visible', 0);
% end

f.WindowState = 'maximized';
ax = axes(f, 'FontSize', 20);
drawnow
hold(ax, 'on');
f.PaperPositionMode = 'auto';
drawnow
end
