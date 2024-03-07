function tp2csv(savedAgentsResult, experimentName)
if ~isequal(size(savedAgentsResult), [1, 2])
    error('savedAgentsResult must be 1x2.');
end

outTable = table();

%% DQN Data

agentDQN = savedAgentsResult(1, 1);
agentDDQN = savedAgentsResult(1, 2);

outTable.EpisodeIndex = agentDQN.EpisodeIndex;

outTable.DQNEpisodeReward = agentDQN.EpisodeReward;
outTable.DQNAverageReward = agentDQN.AverageReward;
outTable.DQNEpisodeQ0 = agentDQN.EpisodeQ0;

outTable.DDQNEpisodeReward = agentDDQN.EpisodeReward;
outTable.DDQNAverageReward = agentDDQN.AverageReward;
outTable.DDQNEpisodeQ0 = agentDDQN.EpisodeQ0;

writetable(outTable, experimentName + "-tp.csv", 'WriteMode', 'append');

end