%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% example script to generate simulated interacting brain regions and
% perform Current-Based Decomposition (CURBD). Ref:
%
% Perich MG et al. Inferring brain-wide interactions using data-constrained
% recurrent neural network models. bioRxiv. DOI: https://doi.org/10.1101/2020.12.18.423348
%
% Written by Matthew G. Perich. Updated December 2020.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;

% simulate three interacting regions with external inputs as in Ref. above
sim = threeRegionSim(struct('N',100));

activity = cat(1,sim.Ra,sim.Rb,sim.Rc);
regions = { ...
    'Region A',1:sim.params.Na; ...
    'Region B',sim.params.Na+1:sim.params.Na+sim.params.Nb; ...
    'Region C',sim.params.Na+sim.params.Nb+1:sim.params.Na+sim.params.Nb+sim.params.Nc; ...
    };


% train Model RNN targeting these three regions
model = trainMultiRegionRNN(activity, struct( ...
    'dtData',sim.params.dtData, ...
    'dtFactor',5, ...
    'regionNames',{regions(:,1)}, ...
    'regionIDs',{regions(:,2)}, ...
    'tauRNN',sim.params.tau/2, ...
    'nRunTrain',500, ...
    'nRunFree',5));


% do CURBD
CURBD = computeCURBD(model);


% plot heatmaps of currents
figure('Position',[100 100 900 900]);
count = 1;
for iTarget = 1:size(CURBD,1)
    for iSource = 1:size(CURBD,2)
        subplot(size(CURBD,1),size(CURBD,2),count); hold all; count = count + 1;
        imagesc(model.tRNN,1:sim.params.Na,CURBD{iTarget,iSource});
        axis tight;
        set(gca,'Box','off','TickDir','out','FontSIze',14,'CLim',[-1 1]);
        xlabel('Time (s)');
        ylabel(['Neurons in ' regions{iTarget,1}]);
        title([regions{iSource,1} ' to ' regions{iTarget,1}]);
    end
end
colormap winter;
