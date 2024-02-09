function plotSpectrograms(featureMat, ptitle)
if size(featureMat) ~= [13, 300, 8]
    error('Invalid input dimensions. Expected [13 24 8] matrix.');
end

figure;

for i = 1:8
    subplot(2, 4, i);
    imagesc(featureMat(:, :, i));
    colormap('jet');
    title(['Ch ' num2str(i)]);
    xlabel('Samples');
    ylabel('Frequency');
    colorbar;
end

sgtitle(ptitle);
end
