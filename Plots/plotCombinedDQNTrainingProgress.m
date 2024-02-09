function plotCombinedDQNTrainingProgress(savedAgentsResult, experimentName)
if ~isequal(size(savedAgentsResult), [1, 2])
    error('savedAgentsResult must be 1x2.');
end

%% DQN Data
agent1 = savedAgentsResult(1, 1);
episodeIndex1 = agent1.EpisodeIndex;
%episodeReward1 = agent1.EpisodeReward;
averageReward1 = agent1.AverageReward;
%episodeQ01 = agent1.EpisodeQ0;

%% DDQN Data
agent2 = savedAgentsResult(1, 2);
episodeIndex2 = agent2.EpisodeIndex;
%episodeReward2 = agent2.EpisodeReward;
averageReward2 = agent2.AverageReward;
%episodeQ02 = agent2.EpisodeQ0;

%% Stride

stride = 100;
filteredIndex1 = episodeIndex1(1:stride:end);
%filteredEpisodeQ01 = episodeQ01(1:stride:end);
%filteredEpisodeReward1 = episodeReward1(1:stride:end);
filteredAverageReward1 = smooth(averageReward1(1:stride:end), 10);

filteredIndex2 = episodeIndex2(1:stride:end);
%filteredEpisodeQ02 = episodeQ02(1:stride:end);
%filteredEpisodeReward2 = episodeReward2(1:stride:end);
filteredAverageReward2 = smooth(averageReward2(1:stride:end), 10);

figure('WindowState', 'maximized');

%% Plot

%plot(filteredIndex1, filteredEpisodeQ01, 'b--', 'LineWidth', 2, 'DisplayName', 'Agent 1 - EpisodeQ0');
hold on;
%plot(filteredIndex1, filteredEpisodeReward1, 'b-.', 'LineWidth', 2, 'DisplayName', 'Agent 1 - EpisodeReward');
plot(filteredIndex1, filteredAverageReward1, 'LineWidth', 3, 'DisplayName', 'DQN');

%plot(filteredIndex2, filteredEpisodeQ02, 'g--', 'LineWidth', 2, 'DisplayName', 'Agent 2 - EpisodeQ0');
%plot(filteredIndex2, filteredEpisodeReward2, 'g-.', 'LineWidth', 2, 'DisplayName', 'Agent 2 - EpisodeReward');
plot(filteredIndex2, filteredAverageReward2, 'LineWidth', 3, 'DisplayName', 'DDQN');

xlabel('Episode', FontSize=24);
ylabel('Average Reward', FontSize=24);
title(['DQN Vs. DDQN Training Progress - ' experimentName], FontSize=24);
legend('show', fontSize=16);
grid on;
end
