clear all;
close all;
rng(23);
dbstop if error

FIT = true;
SIM = true;
doparallel = 0;
laplace = false;
options.fit='data';
options.doem=0;
options.doprior_init=1;
options.fitsjs='all';

experiment_mode = "prolific";
options.experiment_mode = experiment_mode;

if ispc
    root = 'L:';
    if experiment_mode == "prolific"
        subject = '5590a34cfdf99b729d4f69dc'; % 5c4ea6cc889752000156dd8e 5590a34cfdf99b729d4f69dc 66368ac547b8824e50cfa854 5fadd628cd4e9e1c42dab969 5fc58cd91b53521031a2d369 5fd5381b5807b616d910c586
        result_dir = 'L:/rsmith/lab-members/osanchez/wellbeing/tom/model_output/';
    elseif experiment_mode == "local"
        subject = 'AA003'; % example id, it is helpful to have a couple ones commented to check
        results_dir = nan; % add the desired path to this!
    end
    % Parameter List
    
    % The priors are written in the following order:alphaS_solo,
    % alphaS_shared, alphaO_solo, alphaO_shared, tauS, tauO, deltaS, deltaO, lambdaS, lambdaO
    % IN CASES WHERE FITTING NON-SPLIT PARAMETERS you may call it
    % 'alpha','tau', etc
    % if field contains one "alpha" or "lambda", one value of that parameter is fit
    DCM.field = {'alpha','tau','delta'};    
    % this should be changed in the python script when running multiple models
    DCM.model_n = '4';

    % three possible modes for single leak parameters:
    
    % PEs_FB only updates false beliefs based on self prediction error
    % PEo_B only updates self beliefbased on other prediction error
    % BI indicates that the leak is bi-directional
    DCM.leak_mode = {'NaN'};

elseif isunix
    root = '/media/labs/';
    subject = getenv('SUBJECT')
    result_dir = getenv('RESULTS')
    DCM.field = cellstr(strsplit(getenv('FIELD'),','))
    DCM.model_n = getenv('MODEL_N')
    DCM.leak_mode = cellstr(getenv('LEAK_MODE'))
    experiment_mode = getenv('EXPERIMENT')
end

addpath([root '/rsmith/all-studies/util/spm12/']);
addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);


% set up priors and variance based on the estimates of previous paper
if any(contains(DCM.field, 'alphaS'))
    DCM.priors      = [.2 .2 .2 .2];
    DCM.prior_std   = [.02 .01 .02 .01];
else
    DCM.priors      = [.19 .18 .19 .18];
    DCM.prior_std   = [.02 .01 .02 .01];
end
if any(contains(DCM.field, 'tauS'))
    DCM.priors      = [DCM.priors .02 .02];
    DCM.prior_std   = [DCM.prior_std .002 .001];
else
    DCM.priors      = [DCM.priors .02 .02];
    DCM.prior_std   = [DCM.prior_std .001 .001];
end
if any(contains(DCM.field, 'deltaS'))
    DCM.priors      = [DCM.priors .15 .1];
    DCM.prior_std   = [DCM.prior_std .015 .01];
else
    DCM.priors      = [DCM.priors .1 .1];
    DCM.prior_std   = [DCM.prior_std .02 .02];
end
if any(contains(DCM.field, 'lambdaS'))
    DCM.priors      = [DCM.priors .07 .04];
    DCM.prior_std   = [DCM.prior_std .03 .008];
else
    DCM.priors      = [DCM.priors .04 .04];
    DCM.prior_std   = [DCM.prior_std .008 .008];
end

if experiment_mode == "prolific"
    [R] = TOM_fit_prolific(DCM,options,subject,doparallel,laplace);
elseif experiment_model == "local"
    [R] = TOM_fit_prolific(DCM,options,subject,doparallep,laplace); % create a copy of TOM_fit_prolific and change the name to local, update accordingly
end

% results table %
results_table = table;
results_table.model_n = DCM.model_n;
results_table.ID = subject;
% results_table.avg_action_prob = R.DCM.avg_action_prob;
% results_table.model_acc = R.DCM.model_acc;
results_table.OtherPr = mean(R.DCM.OtherPr);
results_table.SelfPr = mean(R.DCM.SelfPr);
results_table.abs_error = R.DCM.abs_error;
% if any(contains(DCM, 'leak_mode'), 'BI', 'PEo_B','PEs_FB')
if isfield(DCM,'leak_mode')
    if any(contains(DCM.leak_mode,{'BI', 'PEo_B', 'PEs_FB'}))
        results_table.leak_type = DCM.leak_mode{1};
    end
end
for i = 1:length(DCM.field)
    results_table.(['prior_' DCM.field{i}]) = DCM.priors(R.r.opt_idx(i));
end
if laplace == true
    results_table.F = R.DCM.F;
else
    results_table.bic = R.bic;
end
results_table.LL = R.DCM.LL;
for i = 1:length(DCM.field)
    results_table.(['posterior_' DCM.field{i}]) = R.DCM.pE(i);
end
results_table.timeout_percent = R.r.subjects.timeout_percent;
results_table.corr_true_and_subj_self_prob = R.DCM.corr_true_and_subj_self_prob;
results_table.corr_true_and_subj_other_prob = R.DCM.corr_true_and_subj_other_prob;
results_table.corr_true_other_and_subj_self = R.DCM.corr_true_other_and_subj_self;
results_table.corr_true_self_and_subj_other = R.DCM.corr_true_self_and_subj_other;
results_table.corr_true_all_and_subj_other = R.DCM.corr_true_all_and_subj_other;
results_table.corr_true_all_and_subj_self = R.DCM.corr_true_all_and_subj_self;

writetable(results_table, [result_dir '/tom_fit_' char(subject) '.csv'])
