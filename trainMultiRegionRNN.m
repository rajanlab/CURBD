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
% Note that region/group identity is not used in training unless the normByGroup
% option is true, in which case the identities are only used to scale the
% target data by region appropriately. So you can train without telling
% this code what the groupIDs are, so long as normByGroup is false.
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
normByGroup    = false;   % normalize activity by group (e.g. regions) or globally
groupNames    = {};      % cell of strings with group names, if desired
groupIDs      = [];      % used for the group designations (see above)
%          Note that groupIDs can be either:
%            1) cell array with M entries with indices to each of M groups (e.g. regions)
%            2) vector of ints with N entries with numerical labels for
%               each neuron corresponding to the M possible groups
% RNN parameters
g              = 1.5;     % instability (chaos); g<1=damped, g>1=chaotic
tauRNN         = 0.01;    % decay costant of RNN units
tauWN          = 0.1;     % decay constant on filtered white noise inputs
ampInWN        = 0.01;    % input amplitude of filtered white noise
% training parameters
J0             = [];      % if empty, intializes randomly
nRunTrain      = 2000;    % number of training runs
nRunFree       = 1;      % number of untrained runs at end
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
% convert labels into a group list
%   cell array, column 1 is group name, column 2 is indices in RNN
%   if no group names are given, defaults to "group 1", etc
if ~isempty(groupIDs)
    if iscell(groupIDs)
        % get number of groups
        nGroups = length(groupIDs);
        idxGroup = groupIDs;
    elseif numel(groupIDs) == nUnits
        groupVals = unique(groupIDs);
        nGroups = length(groupVals);
        idxGroup = cell(nGroups,1);
        for iGroup = 1:nGroups
            idxGroup{iGroup} = find(groupIDs == groupVals(iGroup));
        end
    else
        error('ERROR: cannot parse the groupIDs input...');
    end
    % now get group names
    if isempty(groupNames)
        groupNames = cell(nGroups,1);
        for iGroup = 1:nGroups
            groupNames{iGroup} = ['Group' num2str(iGroup)];
        end
    end
    if ~iscell(groupNames)
        error('ERROR: cannot parse the group names...');
    end
    % now put it all together
    groups = cell(nGroups,2);
    groups(:,1) = groupNames;
    groups(:,2) = idxGroup;
else
    if verbose, disp('FYI: no group info provided.'); end
    groups = {};
    nGroups = NaN;
end

if verbose, disp('Inputs processed.'); end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set up everything for training

if verbose, disp('Initializing simulation.'); end

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
if isempty(J0)
    J = g * randn(nUnits,nUnits) / sqrt(nUnits);
    J0 = J;
else
    J = J0;
end

% set up target training data
Adata = activity;
% normalize
if normByGroup
    if ~isnan(nGroups)
        for iGroup = 1:nGroups
            Adata(groups{iGroup,2},:) = Adata(groups{iGroup,2},:) / max(max(Adata(groups{iGroup,2},:)));
        end
    else
        error('ERROR: no groups were defined. Cannot normalize by group.');
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
if verbose, disp('Beginning training.'); end
for nRun=1:nRunTot
    tic;
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
        JR = J*RNN(:, tt-1) + inputWN(:, tt);
        H = H + dtRNN*(-H + JR)/tauRNN;
        RNN(:, tt) = nonlinearity(H);

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
                k = PJ*RNN(iTarget, tt-1);
                rPr = RNN(iTarget, tt-1)'*k;
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
        legend({'Model','Data'});
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
    toc;
end

% package up outputs
outParams = struct( ...
    'dtFactor',dtFactor, ...
    'normByGroup',normByGroup, ...
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
    'nGroups',nGroups);
out = struct( ...
    'groups',{groups}, ...
    'RNN',RNN, ...
    'tRNN',tRNN, ...
    'dtRNN',dtRNN, ...
    'Adata',Adata, ...
    'tData',tData, ...
    'dtData',dtData, ...
    'J',J, ...
    'J0',J0, ...
    'chi2',chi2, ...
    'pVar',pVars, ...
    'stdData',stdData, ...
    'inputWN',inputWN, ...
    'iTarget',iTarget, ...
    'iNonTarget',iNonTarget, ...
    'params',outParams);
end
