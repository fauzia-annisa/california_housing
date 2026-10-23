% California housing: least-squares regression and prediction error map
% Input: cali.csv (5,000-row sample, 8 features + MedHouseVal in column 9)

T = readtable('cali.csv');

% Remove extreme rows (outliers in occupancy and rooms)
T2 = T(T.AveOccup < 10 & T.AveRooms < 20, :);
n2 = height(T2);

X2 = T2{:, 1:8};   % features
y2 = T2{:, 9};     % target: median house value

% 80/20 random split
rng(42);
id = randperm(n2);
c  = round(0.8 * n2);
t  = id(1:c);        % training indices
s  = id(c+1:end);    % test indices

% Standardize using training statistics only
mu = mean(X2(t, :));
sd = std(X2(t, :));
A  = [ones(n2, 1), (X2 - mu) ./ sd];

% Least-squares fit
b = A(t, :) \ y2(t);

% Test MSE
disp(mean((y2(s) - A(s, :) * b).^2))

% Error map: longitude vs latitude, coloured by absolute error
figure;
scatter(T2.Longitude(s), T2.Latitude(s), 10, abs(y2(s) - A(s, :) * b), 'filled');
colorbar;
title('Prediction error map (outliers removed)');
exportgraphics(gcf, 'error_map_filtered.png')
