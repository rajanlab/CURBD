function [CURBD,CURBDLabels] = computeCURBD(varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Performs Current-Based Decomposition (CURBD) of multi-region data. Ref:
%
% Perich MG et al. Inferring brain-wide interactions using data-constrained
% recurrent neural network models. bioRxiv. DOI: https://doi.org/10.1101/2020.12.18.423348
%
% Two input options:
%   1) out = computeCURBD(model, params)
%       Pass in the output struct of trainMultiRegionRNN and it will do the
%       current decomposition. Note that regions has to be defined.
%
%   2) out = computeCURBD(RNN, J, regions, params)
%       Only needs the RNN activity, region info, and J matrix
%
%   Only parameter right now is current_type, to isolate excitatory or
%   inhibitory currents.
%
% OUTPUTS:
%   CURBD: M x M cell array containing the decomposition for M regions.
%       Target regions are in rows and source regions are in columns.
%   CURBDLabels: M x M cell array with string labels for each current
%
%
% Written by Matthew G. Perich. Updated December 2020.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% available parameters
current_type = 'all'; % 'excitatory', 'inhibitory', or 'all'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
params = struct();
if nargin <=2 % it's a trained model RNN output struct
    RNN = varargin{1}.RNN;
    J = varargin{1}.J;
    regions = varargin{1}.regions;
    if nargin == 2
        params = varargin{2};
    end
elseif nargin == 3 || nargin == 4
    RNN = varargin{1};
    J = varargin{2};
    regions = varargin{3};
    if nargin == 4
        params = varargin{4};
    end
else
    error('ERROR: cannot parse inputs');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
assignParams(who,params);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(regions) || ~iscell(regions)
    error('ERROR: not sure how to interpret regions.');
end
nRegions = size(regions,1);

switch lower(current_type(1:3))
    case 'exc' % take only positive J weights
        J(J < 0) = 0;
    case 'inh' % take only negative J weights
        J(J > 0) = 0;
end

% loop along all bidirectional pairs of regions
CURBD = cell(nRegions,nRegions);
CURBDLabels = cell(nRegions,nRegions);
for iRegionTarget = 1:nRegions
    in_target = regions{iRegionTarget,2};
    for iRegionSource = 1:nRegions
        in_source = regions{iRegionSource,2};

        CURBD{iRegionTarget,iRegionSource} = J(in_target,in_source) * RNN(in_source,:);
        CURBDLabels{iRegionTarget,iRegionSource} = [regions{iRegionSource,1} ' to ' regions{iRegionTarget,1}];
    end
end
