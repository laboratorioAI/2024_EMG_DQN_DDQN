function signal = emgPostProcessing(signal, mapActionInverseFn)
i = -1;
j = -1;
ignoredGesture = mapActionInverseFn('noGesture');
signal = removeIsolated(signal, ignoredGesture);
for k = 1:length(signal)
    if i ~= -1 && (j ~= -1 || k == length(signal))
        if j == -1 && k == length(signal)
            j = k;
        end

        if i == j
            signal(i) = ignoredGesture;
            continue;
        end

        most_common = mode(signal(i:j));
        signal(i:j) = most_common;

        i = -1;
        j = -1;
    end

    if signal(k) == ignoredGesture
        if i ~= -1 && j == -1
            j = k - 1;
        end
        continue;
    end

    if i == -1
        i = k;
        continue;
    end
end
signal(signal == -1) = mapActionInverseFn('noGesture');
end

function signal = removeIsolated(signal, gesture)
for i = 1:length(signal)
    if signal(i) == gesture
        if i == 1
            if signal(i + 1) ~= gesture
                signal(i) = -1;
            end
        elseif i == length(signal)
            if signal(i - 1) ~= gesture
                signal(i) = -1;
            end
        else
            if signal(i - 1) ~= gesture && signal(i + 1) ~= gesture
                signal(i) = -1;
            end
        end
    end
end
end
