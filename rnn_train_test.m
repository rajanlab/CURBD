
dtData = 0.0641
number_units = 500
steps = 165
[tData, bump_rates, bump_current] = bumpgen(number_units, dtData, steps);

model = trainMultiRegionRNN(bump_rates, struct( ...
    'dtData', dtData, ...
    'dtFactor',10, ...
    'trainType','currents', ...
    'nRunTrain',120, ...
    'tauRNN', 0.01, ...
    'g', 1.5, ...
    'ampInWN', 0.1, ...
    'nRunFree',5));


function [tData, bump_rates, bump_current] = bumpgen(number_units, dt_data, steps)
% generate a simple bump for testing trainMultiRegionRNN, 
% calculates current assuming the activation function is tanh
    tData = 0:dt_data:steps*dt_data;
    bump_rates = zeros(number_units, length(tData));
    sig = 0.0343 * number_units; % scaled correctly in neuron space!!!
    norm_by = 2 * sig ^ 2;
    for i=1:number_units
        bump_rates(i, :) = exp(-(i-number_units*tData/tData(end)).^2/(norm_by));
    end
    bump_rates = bump_rates/max(bump_rates, [], 'all');
    bump_rates = min(bump_rates, 0.999);
    bump_rates = max(bump_rates, -0.999);
    bump_current = atanh(bump_rates);
end
