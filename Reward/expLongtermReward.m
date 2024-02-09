function lreward = expLongtermReward(alpha, beta, x, currentReward)
lreward = currentReward + (alpha * exp(beta*x) - alpha);
end