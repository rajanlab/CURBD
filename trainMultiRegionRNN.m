function [out] = trainMultiRegionRNN(activity,params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Trains a data-constrained multi-region RNN. The RNN can be used for,
% among other things, Current-Based Decomposition (CURBD). Ref:
%
% Perich MG et al. Inferring brain-wide interactions using data-constrained
% recurrent neural network models. bioRxiv. DOI: https://doi.org/10.1101/2020.12.18.423348
%
% out = trainMultiRegionRNN(activity,params)
%
% Note that region identity is not used in training unless the normByRegion
% option is true, in which case the identities are only used to scale the
% target data by region appropriately. So you can train without telling
% this code what the regionIDs are, so long as normByRegion is false.
%
% INPUTS:
%   activity : N x T matrix where N is number of neurons and T is time
%   params   : struct with fields corresponding to parameters to overwrite
%          See comments in code for list and descriptions
%
% OUTPUTS:
%   out      : struct containing the results of the simulation
%
%
% Written by Matthew G. Perich. Updated December 2020.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data parameters
dtData         = NaN;     % time step (in s) of the training data
dtFactor       = 1;       % number of interpolation steps for RNN
normByRegion   = false;   % normalize activity by region or globally
regionNames    = {};      % cell of strings with region names, if desired
regionIDs      = [];      % used for the region designations (see above)
%          Note that regionIDs can be either:
%            1) cell array with M entries with indices to each of M regions
%            2) vector of ints with N entries with numerical labels for
%               each neuron corresponding to the M possible regions
% RNN parameters
g              = 1.5;     % instability (chaos); g<1=damped, g>1=chaotic
tauRNN         = 0.01;    % decay costant of RNN units
tauWN          = 0.1;     % decay constant on filtered white noise inputs
ampInWN        = 0.01;    % input amplitude of filtered white noise
% training parameters
nRunTrain      = 2000;    % number of training runs
nRunFree       = 10;      % number of untrained runs at end
P0             = 1;       % learning rate
nonlinearity   = @tanh;   % inline function for nonlinearity
resetPoints    = 1;       % default to only set initial state at time 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotStatus     = true;    % whether to plot data fits during training
verbose        = true;    % whether to print status updates
% overwrite defaults based on inputs
assignParams(who,params);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% finish setting up default parameters
nLearn = size(activity,1); % number of learning steps
nUnits = size(activity,1); % number of units
if isnan(dtData)
    disp('WARNING: The dt of the data was not specified. Defaulting to 1.');
    dtData = 1;
end
dtRNN = dtData / dtFactor;
nRunTot = nRunTrain + nRunFree;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% convert labels into a regions list
%   cell array, column 1 is region name, column 2 is indices in RNN
%   if no region names are given, defaults to "region 1", etc
if ~isempty(regionIDs)
    if iscell(regionIDs)
        % get number of regions
        nRegions = length(regionIDs);
        idxRegion = regionIDs;
    elseif numel(regionIDs) == nUnits
        regVals = unique(regionIDs);
        nRegions = length(regVals);
        idxRegion = cell(nRegions,1);
        for iReg = 1:nRegions
            idxRegion{iReg} = find(regionIDs == regVals(iReg));
        end
    else
        error('ERROR: cannot parse the regionIDs input...');
    end
    % now get region names
    if isempty(regionNames)
        regionNames = cell(nRegions,1);
        for iReg = 1:nRegions
            regionNames{iReg} = ['Region' num2str(iReg)];
        end
    end
    if ~iscell(regionNames)
        error('ERROR: cannot parse the region names...');
    end
    % now put it all together
    regions = cell(nRegions,2);
    regions(:,1) = regionNames;
    regions(:,2) = idxRegion;
else
    if verbose, disp('FYI: no region info provided.'); end
    regions = {};
    nRegions = NaN;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set up everything for training

% if the RNN is bigger than training neurons, pick the ones to target
learnList = randperm(nUnits);
iTarget = learnList(1:nLearn);
iNonTarget = learnList(nLearn:end);

% set up data vectors
tData = dtData*(0:size(activity,2)-1);
tRNN = 0:dtRNN:tData(end);

% set up white noise inputs
ampWN = sqrt(tauWN/dtRNN);
iWN = ampWN*randn(nUnits, length(tRNN));
inputWN = ones(nUnits, length(tRNN));
for tt = 2: length(tRNN)
    inputWN(:, tt) = iWN(:, tt) + (inputWN(:, tt - 1) - iWN(:, tt))*exp(-(dtRNN/tauWN));
end
inputWN = ampInWN*inputWN;

% initialize directed interaction matrix J
J = g * randn(nUnits,nUnits) / sqrt(nUnits);
J0 = J;

% set up target training data
Adata = activity;
% normalize
if normByRegion
    if ~isnan(nRegions)
        for iReg = 1:nRegions
            Adata(regions{iReg,2},:) = Adata(regions{iReg,2},:) / max(max(Adata(regions{iReg,2},:)));
        end
    else
        error('ERROR: no regions were defined. Cannot normalize by region.');
    end
else
    Adata = Adata/max(max(Adata));
end
Adata = min(Adata, 0.999);
Adata = max(Adata, -0.999);


% get standard deviation of entire data
stdData = std(reshape(Adata(iTarget,:), length(iTarget)*length(tData), 1));

% get indices for each sample of model data
iModelSample = zeros(length(tData), 1);
for i=1:length(tData)
    [~, iModelSample(i)] = min(abs(tData(i)-tRNN));
end

% initialize some others
RNN = zeros(nUnits, length(tRNN));
chi2 = zeros(1,nRunTot);
pVars = zeros(1,nRunTot);

% initialize learning update matrix (see Sussillo and Abbot, 2009)
PJ = P0*eye(nLearn, nLearn);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% start training
if plotStatus
    f = figure('Position',[100 100 1800 600]);
end

% loop along training runs
for nRun=1:nRunTot
    % set initial condition to match target data
    H = Adata(:, 1);

    % convert to currents through nonlinearity
    RNN(:, 1) = nonlinearity(H);

    % variables to track when to update the J matrix since the RNN and
    % data can have different dt values
    tLearn = 0; % keeps track of current time
    iLearn = 1; % keeps track of last data point learned
    for tt=2:length(tRNN)
        % update current learning time
        tLearn = tLearn + dtRNN;

        % check if the current index is a reset point. Typically this won't
        % be used, but it's an option for concatenating multi-trial data
        if ismember(tt,resetPoints)
            H = Adata(:,floor(tt/dtFactor)+1);
        end

        % compute next RNN step
        RNN(:, tt) = nonlinearity(H);
        JR = J*RNN(:, tt) + inputWN(:, tt);
        H = H + dtRNN*(-H + JR)/tauRNN;

        % check if the RNN time coincides with a data point to update J
        if (tLearn>=dtData)
            tLearn = 0;
            % compute error
            err = RNN(1:nUnits, tt) - Adata(1:nUnits, iLearn);

            % update chi2 using this error
            chi2(nRun) = chi2(nRun) + mean(err.^2);

            % update learning index
            iLearn = iLearn + 1;

            % check if it's a training run
            if  (nRun<=nRunTrain)
                % update J based on error gradient
                k = PJ*RNN(iTarget, tt);
                rPr = RNN(iTarget, tt)'*k;
                c = 1.0/(1.0 + rPr);
                PJ = PJ - c*(k*k');
                J(1:nUnits, iTarget) = J(1:nUnits, iTarget) - c*err(1:nUnits, :)*k';
            end
        end
    end

    rModelSample = RNN(iTarget, iModelSample);

    % compute variance explained of activity by units
    pVar = 1 - (norm(Adata(iTarget,:) - rModelSample, 'fro')/(sqrt(length(iTarget)*length(tData))*stdData)).^2;
    pVars(nRun) = pVar;

    % print status
    if verbose, fprintf('trial=%d pVar=%f chi2=%f\n', nRun, pVar, chi2(nRun)); end

    % plot the data, chi2, error, and a random unit
    if plotStatus
        clf(f);
        idx = randi(nUnits);
        subplot(2,4,1);
        hold on;
        imagesc(Adata(iTarget,:));
        axis tight;
        title('real');
        set(gca,'Box','off','TickDir','out','FontSize',14);
        subplot(2,4,2);
        hold on;
        imagesc(RNN(iTarget,:));
        axis tight;
        title('model');
        set(gca,'Box','off','TickDir','out','FontSize',14);
        subplot(2,4,[3 4 7 8]);
        hold all;
        plot(tRNN,RNN(iTarget(idx),:));
        plot(tData,Adata(iTarget(idx),:));
        title(nRun)
        set(gca,'Box','off','TickDir','out','FontSize',14);
        subplot(2,4,5);
        hold on;
        plot(pVars(1:nRun));
        ylabel('pVar');
        set(gca,'Box','off','TickDir','out','FontSize',14);
        subplot(2,4,6);
        hold on;
        plot(chi2(1:nRun))
        ylabel('chi2');
        set(gca,'Box','off','TickDir','out','FontSize',14);
        drawnow;
    end
end

% package up outputs
outParams = struct( ...
    'dtFactor',dtFactor, ...
    'normByRegion',normByRegion, ...
    'g',g, ...
    'P0',P0, ...
    'tauRNN',tauRNN, ...
    'tauWN',tauWN, ...
    'ampInWN',ampInWN, ...
    'nRunTot',nRunTot, ...
    'nRunTrain',nRunTrain, ...
    'nRunFree',nRunFree, ...
    'nonlinearity',nonlinearity, ...
    'resetPoints',resetPoints, ...
    'nUnits',nUnits, ...
    'nRegions',nRegions);
out = struct( ...
    'regions',{regions}, ...
    'RNN',RNN, ...
    'tRNN',tRNN, ...
    'dtRNN',dtRNN, ...
    'Adata',Adata, ...
    'tData',tData, ...
    'dtData',dtData, ...
    'J',J, ...
    'J0',J0, ...
    'chi2',chi2, ...
    'pVar',pVar, ...
    'stdData',stdData, ...
    'inputWN',inputWN, ...
    'iTarget',iTarget, ...
    'iNonTarget',iNonTarget, ...
    'params',outParams);
end
