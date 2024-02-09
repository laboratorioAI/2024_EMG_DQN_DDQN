function plotEMG(emgMatrix, ptitle)
[n, numChannels] = size(emgMatrix);

if numChannels ~= 8
    error('It must has 8 columns.');
end

figure;

for channel = 1:numChannels
    subplot(4, 2, channel);
    plot(1:n, emgMatrix(:, channel));
    ylim([-1, 1]);
    title(['Ch ' num2str(channel)]);
    xlabel('Samples');
    ylabel('Amplitude');
end

sgtitle(ptitle);
end
