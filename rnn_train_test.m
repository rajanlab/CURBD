
dtData = 0.0641
number_units = 500
steps = 165
[tData, xBump, hBump] = bumpgen(number_units, dtData, steps);

model = trainMultiRegionRNN(hBump, struct( ...
    'dtData', dtData, ...
    'dtFactor',10, ...
    'trainType','currents', ...
    'nRunTrain',120, ...
    'tauRNN', 0.01, ...
    'g', 1.5, ...
    'ampInWN', 0.1, ...
    'nRunFree',5));


function [tData, xBump, hBump] = bumpgen(number_units, dt_data, steps)
% generate a simple bump for testing trainMultiRegionRNN
    tData = 0:dt_data:steps*dt_data;
    xBump = zeros(number_units, length(tData));
    sig = 0.0343 * number_units; % scaled correctly in neuron space!!!
    norm_by = 2 * sig ^ 2;
    for i=1:number_units
        xBump(i, :) = exp(-(i-number_units*tData/tData(end)).^2/(norm_by));
    end
%    hBump = log((xBump+0.01)./(1-xBump+0.01)); % current from rate
    xBump = xBump/max(xBump, [], 'all');
    xBump = min(xBump, 0.999);
    xBump = max(xBump, -0.999);
    hBump = atanh(xBump);
end
