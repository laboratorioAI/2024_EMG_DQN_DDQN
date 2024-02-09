function plotDQNTrainingProgress(savedAgentsResult, experimentName)
if ~isequal(size(savedAgentsResult), [1, 2])
    error('savedAgentsResult must be 1x2.');
end

%% DQN Data

agent1 = savedAgentsResult(1, 1);
episodeIndex1 = agent1.EpisodeIndex;
episodeReward1 = agent1.EpisodeReward;
averageReward1 = agent1.AverageReward;
episodeQ01 = agent1.EpisodeQ0;

%% DDQN Data

agent2 = savedAgentsResult(1, 2);
episodeIndex2 = agent2.EpisodeIndex;
episodeReward2 = agent2.EpisodeReward;
averageReward2 = agent2.AverageReward;
episodeQ02 = agent2.EpisodeQ0;

%% Stride

stride = 100;
filteredIndex1 = episodeIndex1(1:stride:end);
filteredEpisodeReward1 = smooth(episodeReward1(1:stride:end), 10);
filteredAverageReward1 = smooth(averageReward1(1:stride:end), 10);
% filteredEpisodeQ01 = smooth(episodeQ01(1:stride:end), 10);

filteredIndex2 = episodeIndex2(1:stride:end);
filteredEpisodeReward2 = smooth(episodeReward2(1:stride:end), 10);
filteredAverageReward2 = smooth(averageReward2(1:stride:end), 10);
% filteredEpisodeQ02 = smooth(episodeQ02(1:stride:end), 10);

figure('WindowState', 'maximized');

%% DQN

subplot(1, 2, 1);
plot(filteredIndex1, filteredEpisodeReward1, 'LineWidth', 3, 'DisplayName', 'EpisodeReward');
hold on;
plot(filteredIndex1, filteredAverageReward1, '--', 'LineWidth', 2, 'DisplayName', 'AverageReward');
% plot(filteredIndex1, filteredEpisodeQ01, '-.', 'LineWidth', 2, 'DisplayName', 'EpisodeQ0');
xlabel('Episode', FontSize=24);
ylabel('Episode Reward', FontSize=24);
title('DQN Training Progress', FontSize=24);
legend('show', fontsize=16);
grid on;

%% DDQN

subplot(1, 2, 2);
plot(filteredIndex2, filteredEpisodeReward2, 'LineWidth', 3, 'DisplayName', 'EpisodeReward');
hold on;
plot(filteredIndex2, filteredAverageReward2, '--', 'LineWidth', 2, 'DisplayName', 'AverageReward');
% plot(filteredIndex2, filteredEpisodeQ02, '-.', 'LineWidth', 2, 'DisplayName', 'EpisodeQ0');
xlabel('Episode', FontSize=24);
ylabel('Episode Reward', FontSize=24);
title(['DDQN Training Progress - ' experimentName], FontSize=24);
legend('show', fontsize=16);
grid on;
end
