% Samuel Taylor and Ryan Smith, 2021

% Model inversion script
function [DCM] = TOM_inversion_simple(DCM)

% MDP inversion using Variational Bayes
% FORMAT [DCM] = spm_dcm_mdp(DCM)

% If simulating - comment out section on line 196
% If not simulating - specify subject data file in this section 

%
% Expects:
%--------------------------------------------------------------------------
% DCM.MDP   % MDP structure specifying a generative model
% DCM.field % parameter (field) names to optimise
% DCM.U     % cell array of outcomes (stimuli)
% DCM.Y     % cell array of responses (action)
%
% Returns:
%--------------------------------------------------------------------------
% DCM.M     % generative model (DCM)
% DCM.Ep    % Conditional means (structure)
% DCM.Cp    % Conditional covariances
% DCM.F     % (negative) Free-energy bound on log evidence
% 
% This routine inverts (cell arrays of) trials specified in terms of the
% stimuli or outcomes and subsequent choices or responses. It first
% computes the prior expectations (and covariances) of the free parameters
% specified by DCM.field. These parameters are log scaling parameters that
% are applied to the fields of DCM.MDP. 
%
% If there is no learning implicit in multi-trial games, only unique trials
% (as specified by the stimuli), are used to generate (subjective)
% posteriors over choice or action. Otherwise, all trials are used in the
% order specified. The ensuing posterior probabilities over choices are
% used with the specified choices or actions to evaluate their log
% probability. This is used to optimise the MDP (hyper) parameters in
% DCM.field using variational Laplace (with numerical evaluation of the
% curvature).
%
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_dcm_mdp.m 7120 2017-06-20 11:30:30Z spm $

% OPTIONS
%--------------------------------------------------------------------------
ALL = false;

% prior expectations and covariance
%--------------------------------------------------------------------------
prior_variance = 2^-2;

for i = 1:length(DCM.field)
    field = DCM.field{i};
    try
        param = DCM.MDP.(field);
        param = double(~~param);
    catch
        param = 1;
    end
    if ALL
        pE.(field) = zeros(size(param));
        pC{i,i}    = diag(param);
    else
        if ismember(field,{'lambdaS','lambdaO'})
            pE.(field) = log((DCM.MDP.(field)+1)/(1-DCM.MDP.(field)));  % in logit-space - bounded between -1 and 1!
            pC{i,i}    = prior_variance;
        elseif ismember(field,{'alphaS_solo','alphaS_shared','alphaO_solo', ...
            'alphaO_shared','alphaS','alphaO','deltaS','deltaO'})
            pE.(field) = log(DCM.MDP.(field)/(1-DCM.MDP.(field)));      % in logit-space - bounded between 0 and 1!
            pC{i,i}    = prior_variance;
        elseif ismember(field, {'tauS','tauO'})                         % in logit-space - bounded between 0.0001 and .08!
            pE.(field) = log((DCM.MDP.(field)-0.0001)/(0.08-DCM.MDP.(field)));
            pC{i,i}   = prior_variance;     
        else
            errorMessage = sprintf('Error: %s not transformed.', field);
            error(errorMessage);
        end
    end
end

pC      = spm_cat(pC);

% model specification
%--------------------------------------------------------------------------
M.L     = @(P,M,U,Y)spm_mdp_L(P,M,U,Y);  % log-likelihood function
M.pE    = pE;                            % prior means (parameters)
M.pC    = pC;                            % prior variance (parameters)
M.mdp   = DCM.MDP;                       % MDP structure

% Variational Laplace
%--------------------------------------------------------------------------
[Ep,Cp,F] = spm_nlsi_Newton(M,DCM.U,DCM.Y);

% Store posterior densities and log evidence (free energy)
%--------------------------------------------------------------------------
DCM.M   = M;
DCM.Ep  = Ep;
DCM.Cp  = Cp;
DCM.F   = F;


return

function L = spm_mdp_L(P,M,U,Y)
% log-likelihood function
% FORMAT L = spm_mdp_L(P,M,U,Y)
% P    - parameter structure
% M    - generative model
% U    - inputs
% Y    - observed repsonses
%__________________________________________________________________________

if ~isstruct(P); P = spm_unvec(P,M.pE); end

% multiply parameters in MDP
%--------------------------------------------------------------------------
mdp   = M.mdp;
field = fieldnames(M.pE);
for i = 1:length(field)
    if ismember(field{i},{})
        mdp.(field{i}) = exp(P.(field{i}));
    elseif ismember(field{i},{})
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    else
        mdp.(field{i}) = (P.(field{i}));
    end
end


% discern whether learning is enabled - and identify unique trials if not
%--------------------------------------------------------------------------
if any(ismember(fieldnames(mdp),{'a','b','d','c','d','e'}))
    j = 1:numel(U);
    k = 1:numel(U);
else
    % find unique trials (up until the last outcome)
    %----------------------------------------------------------------------
    u       = spm_cat(U');
    [i,j,k] = unique(u(:,1:(end - 1)),'rows');
end

U_Block = U{:}-1;
rewards = reshape(U_Block,mdp.T,mdp.NB)';

Y_Block = Y{:}-1;
choices = reshape(Y_Block,mdp.T,mdp.NB)';

L = 0;
force_choice = mdp.force_choice;
force_outcome = mdp.force_outcome;

% Each block is separate -- effectively resetting beliefs at the start of
% each block. 
for idx_block = 1:mdp.NB

    p1 = .5;
    p2 = .5;
    p3 = .5;

    true_probs = [p1   p2   p3   ;
                  1-p1 1-p2 1-p3];

    % simulating or fitting
    sim = 0; %1 = simulating, 0 = fitting
    mdp.force_choice = force_choice(idx_block,:);
    mdp.force_outcome = force_outcome(idx_block,:);


        % solve MDP and accumulate log-likelihood
        %--------------------------------------------------------------------------
    MDP_Block{idx_block} = Simple_TAB_model_v2(mdp, rewards(idx_block,:), choices(idx_block,:), sim);

    for j = 4:mdp.T
        L = L + log(MDP_Block{idx_block}.action_probabilities(MDP_Block{idx_block}.choices(j),j) + eps);
    end


end

fprintf('LL: %f \n',L)