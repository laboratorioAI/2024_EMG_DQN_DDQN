function plotBySegments(genStats, ageStats, handStats, damStats, options)
%plotBySegments generates charts for age, gender, etc.
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

%% plot gender
[~, fGender] = prettyFig();
ax1G = subplot(1, 2, 1, 'replace');

ax2G = subplot(1, 2, 2);
hold(ax2G, 'on')

pie(ax1G, genStats.GroupCount)
legend(ax1G, genStats.Row, 'Location', 'bestoutside')
colormap(ax1G, 'cool')
title(ax1G, 'Gender distribution in dataset')
% ax1G.FontSize = 25;

h1 = bar(ax2G, genStats.gender, [genStats.mean_class genStats.mean_recog]);
yPoints = cat(1, h1.YEndPoints);
% xPoints = cat(1, h1.XEndPoints);

ax2G.YLim = [min(yPoints , [],'all')*0.995 max(yPoints , [],'all')*1.005];
ax2G.YTick = unique(yPoints);
ax2G.YGrid = 'on';
title(ax2G, 'Accuracy by gender')
ytickformat(ax2G, '%.2g\%')

legend(h1,'classification', 'recognition', 'Location', 'best')


%% plot age
[~, fAge] = prettyFig();
ax1A = subplot(2, 1, 1, 'replace');

ax2A = subplot(2, 1, 2);
hold(ax2A, 'on')
bar(ax1A, ageStats.age, ageStats.GroupCount / 306*100);
% xlabel(ax1A, 'Age [years]')
% xtickformat(ax1A, '%g years')
ytickformat(ax1A, '%.2g%%')

ylabel(ax1A, 'Age distribution in dataset')
ax1A.YGrid = 'on';

%
errorbar(ax2A, [ageStats.age ageStats.age], ...
    [ageStats.mean_class ageStats.mean_recog], ...
    [ageStats.std_class ageStats.std_recog], '*');

ylabel(ax2A, 'Accuracy by age')
ax2A.YGrid = 'on';
ytickformat(ax2A, '%.2g%%')
xlabel(ax2A, 'Age [years]')
ax2A.XTick = ax1A.XTick;
ax2A.XLim = ax1A.XLim;
ax2A.XGrid = 'on';
legend(ax2A,'classification', 'recognition', 'Location', 'best')
if ax2A.YLim(2) > 100
    ax2A.YLim(2) = 100;
end

%%  plot handedness
[~, fHand] = prettyFig();
ax1H = subplot(1, 2, 1, 'replace');

ax2H = subplot(1, 2, 2);
hold(ax2H, 'on')

pie(ax1H, handStats.GroupCount)
legend(ax1H, handStats.Row, 'Location', 'bestoutside')
colormap(ax1H, 'summer')
title(ax1H, 'handedness distribution in dataset')

hH1 = bar(ax2H, handStats.handedness, ...
    [handStats.mean_class handStats.mean_recog]);
yPoints = cat(1, hH1.YEndPoints);
% xPoints = cat(1, hH1.XEndPoints);
if numel(handStats.Row) > 1
ax2H.XLim = handStats.Row;
end

ax2H.YLim = [min(yPoints , [],'all')*0.995 max(yPoints , [],'all')*1.005];
ax2H.YTick = unique(yPoints);
ax2H.YGrid = 'on';
title(ax2H, 'Accuracy by handedness')
ytickformat(ax2H, '%.2g\%')

legend(hH1,'classification', 'recognition', 'Location', 'best')

%% ethnicGroup (muy poquitos)
%ageStats = grpstats(fullData, ...
%{'ethnicGroup'}, {'mean', 'std'},'DataVars',{'class','recog','overlapF'});
% 303 de los 306 son latinos, 2 indígenas y 1 native

%% plot hasSufferedArmDamage

[~, fDamage] = prettyFig();
ax1D = subplot(1, 2, 1, 'replace');

ax2D = subplot(1, 2, 2);
hold(ax2D, 'on')

pie(ax1D, damStats.GroupCount)
if damStats.Row{1} == '0'
    leg = {'no', 'yes'};
else
    leg = {'yes', 'no'};
end
legend(ax1D, leg, 'Location', 'bestoutside')
colormap(ax1D, 'copper')
title(ax1D, 'Reported arm damage distribution in dataset')

hH1 = bar(ax2D, damStats.hasSufferedArmDamage, ...
    [damStats.mean_class damStats.mean_recog]);
yPoints = cat(1, hH1.YEndPoints);
% xPoints = cat(1, hH1.XEndPoints);
% ax2D.XLim = damStats.Row;

ax2D.YLim = [min(yPoints , [],'all')*0.995 max(yPoints , [],'all')*1.005];
ax2D.YTick = unique(yPoints);
ax2D.YGrid = 'on';
title(ax2D, 'Accuracy by reported arm damage')
ytickformat(ax2D, '%.2g\%')
ax2D.XTick = [0 1];
ax2D.XTickLabel = {'no', 'yes'};
legend(hH1,'classification', 'recognition', 'Location', 'best')

drawnow
%%
if options.save
    savefig(fGender, [options.outputFolder '\gender.fig'])
    savefig(fAge, [options.outputFolder '\age.fig'])
    savefig(fHand, [options.outputFolder '\handedness.fig'])
    savefig(fDamage, [options.outputFolder '\armDamage.fig'])
    
    saveas(fGender, [options.outputFolder '\gender.png'])
    saveas(fAge, [options.outputFolder '\age.png'])
    saveas(fHand, [options.outputFolder '\handedness.png'])
    saveas(fDamage, [options.outputFolder '\armDamage.png'])
end