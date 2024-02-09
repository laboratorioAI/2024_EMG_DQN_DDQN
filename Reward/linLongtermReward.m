function lreward = linLongtermReward(alpha, beta, x, currentReward)
lreward = currentReward + (alpha * x - beta);
end