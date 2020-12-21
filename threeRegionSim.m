function out = threeRegionSim(params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% out = threeRegionSim(params)
%
% Generates a simulated dataset with three interacting regions. Ref:
%
% Perich MG et al. Inferring brain-wide interactions using data-constrained
% recurrent neural network models. bioRxiv. DOI: https://doi.org/10.1101/2020.12.18.423348
%
% INPUTS:
%   params : (optional) parameter struct. See code below for options.
%
% OUTPUTS:
%   out : output struct with simulation results and parameters
%
% Written by Matthew G. Perich. Updated December 2020.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize variables
N            = 100;  % number of units in each region
ga           = 1.8;  % chaos parameter for Region A
gb           = 1.5;  % chaos parameter for Region B
gc           = 1.5;  % chaos parameter for Region C
tau          = 0.1;  % decay time constant of RNNs
fracInterReg = 0.05; % fraction of inter-region connections
ampInterReg  = 0.2; % amplitude of inter-region connections
fracExternal = 0.5;  % fraction of external inputs to B/C
ampInB       = 1;    % amplitude of external inputs to Region B
ampInC       = -1;   % amplitude of external inputs to Region C
dtData       = 0.01; % time step (s) of the simulation
T            = 10;   % total simulation time
leadTime     = 2;    % time before sequence starts and after FP moves
bumpStd      = 0.2;  % width (in frac of population) of sequence/FP
plotSim      = true; % whether to plot the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin == 0, params = struct(); end
assignParams(who,params);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% configure everything

% set up time vector for data
tData = 0:dtData:T;

% for now it only works if the networks are the same size
Na = N;
Nb = N;
Nc = N;

% set up RNN A (chaotic responder)
Ja = randn(Na, Na);
Ja = ga/sqrt(Na) * Ja;
hCa = 2*rand(Na, 1)-1; % start from random state

% set up RNN B (driven by sequence)
Jb = randn(Nb, Nb);
Jb = gb/sqrt(Nb) * Jb;
hCb = 2*rand(Nb, 1)-1;

% set up RNN C (driven by fixed point)
Jc = randn(Nc, Nc);
Jc = gc/sqrt(Nb) * Jc;
hCc = 2*rand(Nc, 1)-1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate external inputs

% set up sequence-driving network
xBump = zeros(Nb, length(tData));
sig = bumpStd*Nb; %width of bump in N units
for i = 1:Nb
    if 0
        xBump(i, :) = exp(-(i-sig-Nb*tData/tData(end)).^2/(2*sig^2));
    else
        xBump(i, :) = exp(-(i-sig-Nb*tData/(tData(end)/2)).^2/(2*sig^2));
        xBump(i,ceil(length(tData)/2)-100:end) = xBump(i,ceil(length(tData)/2)-100);
    end
end
hBump = log((xBump+0.01)./(1-xBump+0.01));
hBump = hBump-min(min(hBump));
hBump = hBump/max(max(hBump));

% set up fixed points driving network
xFP = zeros(Nc, length(tData));
for i = 1:Nc
    xFP(i, :) = [xBump(i,10) * ones(1, ceil(length(tData)/2)+100), ...
        xBump(i,300) * ones(1, length(tData)-(ceil(length(tData)/2)+100))];
end
hFP = log((xFP+0.01)./(1-xFP+0.01));
hFP = hFP-min(min(hFP));
hFP = hFP/max(max(hFP));

% add the lead time
tData = [tData, tData(end)+dtData:dtData:T+leadTime];
hBump = cat(2,repmat(hBump(:,1),1,leadTime / dtData),hBump);
hFP = cat(2,repmat(hFP(:,1),1,leadTime / dtData),hFP);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% build connectivity between RNNs
Nfrac = round(fracInterReg*N);

[w_A2B,w_A2C,w_B2A,w_B2C,w_C2A,w_C2B,w_Seq2B,w_Fix2C]=deal(zeros(N,1));
rand_idx = randperm(N);
w_A2B(rand_idx(1:Nfrac)) = 1;
rand_idx = randperm(N);
w_A2C(rand_idx(1:Nfrac)) = 1;
rand_idx = randperm(N);
w_B2A(rand_idx(1:Nfrac)) = 1;
rand_idx = randperm(N);
w_B2C(rand_idx(1:Nfrac)) = 1;
rand_idx = randperm(N);
w_C2A(rand_idx(1:Nfrac)) = 1;
rand_idx = randperm(N);
w_C2B(rand_idx(1:Nfrac)) = 1;

% Sequence only projects to B
Nfrac = round(fracExternal*N);
rand_idx = randperm(N);
w_Seq2B(rand_idx(1:Nfrac)) = 1;

% Fixed point only projects to A
Nfrac = round(fracExternal*N);
rand_idx = randperm(N);
w_Fix2C(rand_idx(1:Nfrac)) = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate time series simulated data
Ra = NaN(Na, length(tData));
Rb = NaN(Nb, length(tData));
Rc = NaN(Nc, length(tData));
for tt = 1:length(tData)
    Ra(:, tt) = tanh(hCa);
    Rb(:, tt) = tanh(hCb);
    Rc(:, tt) = tanh(hCc);

    % chaotic responder
    JRa = Ja * Ra(:, tt) + ...
        ampInterReg * w_B2A .* Rb(:,tt) + ...
        ampInterReg * w_C2A .* Rc(:,tt);
    hCa = hCa + dtData*(-hCa + JRa) / tau;

    % sequence driven
    JRb = Jb * Rb(:, tt) + ...
        ampInterReg * w_A2B .* Ra(:,tt) + ...
        ampInterReg * w_C2B .* Rc(:,tt) + ...
        ampInB * w_Seq2B .* hBump(:,tt);
    hCb = hCb + dtData * (-hCb + JRb) / tau;

    % fixed point driven
    JRc = Jc * Rc(:, tt) + ...
        ampInterReg * w_B2C .* Rb(:,tt) + ...
        ampInterReg * w_A2C .* Ra(:,tt) + ...
        ampInC * w_Fix2C .* hFP(:,tt);
    hCc = hCc + dtData * (-hCc + JRc) / tau;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% package up outputs
Rseq = hBump;
Rfp  = hFP;
% normalize
Ra = Ra./max(max(Ra));
Rb = Rb./max(max(Rb));
Rc = Rc./max(max(Rc));
Rseq = Rseq./max(max(Rseq));
Rfp = Rfp./max(max(Rfp));

out_params = struct( ...
    'Na',Na, ...
    'Nb',Nb, ...
    'Nc',Nc, ...
    'ga',ga, ...
    'gb',gb, ...
    'gc',gc, ...
    'tau',tau, ...
    'fracInterReg',fracInterReg, ...
    'ampInterReg',ampInterReg, ...
    'fracExternal',fracExternal, ...
    'ampInB',ampInB, ...
    'ampInC',ampInC, ...
    'dtData',dtData, ...
    'T',T, ...
    'leadTime',leadTime, ...
    'bumpStd',bumpStd);
out = struct( ...
    'Ra',Ra, ...
    'Rb',Rb, ...
    'Rc',Rc, ...
    'Rseq',Rseq, ...
    'Rfp',Rfp, ...
    'tData',tData, ...
    'Ja',Ja, ...
    'Jb',Jb, ...
    'Jc',Jc, ...
    'w_A2B',w_A2B, ...
    'w_A2C',w_A2C, ...
    'w_B2A',w_B2A, ...
    'w_B2C',w_B2C, ...
    'w_C2A',w_C2A, ...
    'w_C2B',w_C2B, ...
    'w_Fix2C',w_Fix2C, ...
    'w_Seq2B',w_Seq2B, ...
    'params',out_params);

if plotSim
    % plot simulation
    c_lim = 1.5*[-1 1];

    figure('Position',[100 100 900 700]);
    subplot(4,3,1)
    imagesc(tData, 1:Na, Ra);
    axis square;
    colorbar;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    set(gca,'CLim',c_lim);
    title(['RNN A - g=' num2str(ga)]);
    subplot(4,3,2)
    imagesc(1:Na, 1:Na, Ja);
    axis square;
    colorbar;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    title('DI matrix A');
    subplot(4,3,3); hold all;
    idx = randperm(Na);
    plot(tData, Ra(idx(1), :),'LineWidth',2);
    plot(tData, Ra(idx(2), :),'LineWidth',2);
    plot(tData, Ra(idx(3), :),'LineWidth',2);
    ylim([-1 1])
    title('units from RNN A');
    axis square;
    set(gca,'Box','off','TickDir','out','FontSize',14);


    subplot(4,3,4)
    imagesc(tData, 1:Nb, Rb);
    axis square;
    colorbar;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    set(gca,'CLim',c_lim);
    title(['RNN B - g=' num2str(gb)]);
    subplot(4,3,5)
    imagesc(1:Nb, 1:Nb, Jb);
    axis square;
    colorbar;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    title('DI matrix B');
    subplot(4,3,6); hold all;
    idx = randperm(Nb);
    plot(tData, Rb(idx(1), :),'LineWidth',2);
    plot(tData, Rb(idx(2), :),'LineWidth',2);
    plot(tData, Rb(idx(3), :),'LineWidth',2);
    ylim([-1 1])
    title('units from RNN B');
    axis square;
    set(gca,'Box','off','TickDir','out','FontSize',14);


    subplot(4,3,7)
    imagesc(tData, 1:Nc, Rc);
    axis square;
    colorbar;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    set(gca,'CLim',c_lim);
    title(['RNN C - g=' num2str(gc)]);
    subplot(4,3,8)
    imagesc(1:Nc, 1:Nc, Jc);
    axis square;
    colorbar;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    title('DI matrix C');
    subplot(4,3,9); hold all;
    idx = randperm(Nc);
    plot(tData, Rc(idx(1), :),'LineWidth',2);
    plot(tData, Rc(idx(2), :),'LineWidth',2);
    plot(tData, Rc(idx(3), :),'LineWidth',2);
    ylim([-1 1])
    title('units from RNN C');
    axis square;
    set(gca,'Box','off','TickDir','out','FontSize',14);


    subplot(4,3,10)
    imagesc(tData, 1:Nc, Rfp);
    axis square;
    colorbar;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    title('Fixed Point Driver');

    subplot(4,3,11)
    imagesc(tData, 1:Nc, Rseq);
    axis square;
    colorbar;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    title('Sequence Driver');

    drawnow;
end

end
