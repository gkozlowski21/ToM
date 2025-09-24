function [R] = TOM_fit_prolific(DCM,options,subject,doparallel,laplace)

% %Todays date
% [y,m,d] = ymd(datetime);
% dt = strcat(num2str(m), '-', num2str(d), '-', num2str(y));
% model_parts = strsplit(strtrim(num2str(model)));
% model_string = strjoin(model_parts, '_');
% ld = [dt '-' num2str(model_string) ];

%Configure data
[r] = TOM_config(DCM,options,subject,laplace);
% CMG edit
r.doparallel = doparallel;
%Run models
Np=length(r.opt_idx);
if laplace == 1
    [DCM] = emfit_OPS(r,Np,cell(Np,1),2000,0,1,r.maxit,0,'','',1,options.doprior_init); % note that I am calling emfit_CMG

    % go into the model to retrieve model acc and avg. action probability

    [l,model_acc, avg_action_prob,OtherPr,SelfPr,abs_error]=FBT_llfun(DCM.Ep,r,r.nsjs);

    % re-transform parameters
    DCM.pE = sigmtr(DCM.Ep',r.LB(r.opt_idx),r.UB(r.opt_idx),50);
    % DCM.pC = sigmtr(DCM.Cp,r.LB(r.opt_idx),r.UB(r.opt_idx),1);
    DCM.LL = -l;
    DCM.abs_error = mean(abs_error);
    %Save in results structure
%     R.r=r;  %Input data
%     R.DCM = DCM; %results
%     R.priors = r.subjects.priors;  %Best fitting parameters for each subject
%     R.fields = r.model;
else
    [E,V,~,stats,bf] = emfit_CMG(r,Np,cell(Np,1),2000,0,1,r.maxit,0,'','',1,options.doprior_init);

    [~,model_acc, avg_action_prob,OtherPr,SelfPr,abs_error]=FBT_llfun(E,r,r.nsjs);

    DCM.pE = stats.subjectmeans;
    DCM.abs_error = mean(abs_error);
    R.bic = bf.bic;
    DCM.LL = -stats.LL;
    DCM.corr_true_and_subj_self_prob = stats.corr_true_and_subj_self_prob;
    DCM.corr_true_and_subj_other_prob = stats.corr_true_and_subj_other_prob;
    DCM.corr_true_other_and_subj_self = stats.corr_true_other_and_subj_self;
    DCM.corr_true_self_and_subj_other = stats.corr_true_self_and_subj_other;
    DCM.corr_true_all_and_subj_other = stats.corr_true_all_and_subj_other;
    DCM.corr_true_all_and_subj_self = stats.corr_true_all_and_subj_self;
end

% calculate model accuracy
% DCM.model_acc = model_acc;
% calculate avg. action probability
% DCM.avg_action_prob = avg_action_prob;

% Save the probability density of each choice they make
DCM.OtherPr = OtherPr;
DCM.SelfPr = SelfPr;
%Save results in structure R
R.r=r; % input data
R.DCM = DCM;
R.priors = r.subjects.priors;
R.fields = r.model;
end