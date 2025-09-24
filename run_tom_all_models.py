import sys, os, re, subprocess

#subject_list_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/theory_of_mind/TOM_subject_IDs_prolific.csv'
subject_list_path = '/media/labs/rsmith/lab-members/osanchez/wellbeing/tom/updated_id_list.csv'
results = sys.argv[1]
experiment_mode = sys.argv[2] # indicate inperson, mturk, or prolific

models = [
    {'field': 'tau','leak_mode': 'NA','model_n':1},
    {'field': 'tau','leak_mode': 'NA','model_n':2},
    {'field': 'alpha,tau','leak_mode': 'NA','model_n':3},
    {'field': 'alpha,tau,delta','leak_mode': 'NA','model_n':4},
    {'field': 'alphaS,alphaO,tau,delta','leak_mode': 'NA','model_n':5},
    {'field': 'alphaS,alphaO,tauS,tauO,delta','leak_mode': 'NA','model_n':6},
    {'field': 'alphaS,alphaO,tauS,tauO,deltaS,deltaO','leak_mode': 'NA','model_n':7},
    {'field': 'alpha,tauS,tauO,delta','leak_mode': 'NA','model_n':8},
    {'field': 'alpha,tau,deltaS,deltaO','leak_mode': 'NA','model_n':9},
    {'field': 'alphaS,alphaO,tau,deltaS,deltaO','leak_mode': 'NA','model_n':10},
    {'field': 'alphaS,alphaO,tauS,tauO,deltaS,deltaO,lambdaS,lambdaO','leak_mode': 'NA','model_n':11}, # remember the comma when running all models
    {'field': 'alphaS,alphaO,tauS,tauO,deltaS,deltaO,lambda','leak_mode': 'BI','model_n':12},
    {'field': 'alphaS,alphaO,tauS,tauO,deltaS,deltaO,lambda','leak_mode': 'PEo_B','model_n':13},
    {'field': 'alphaS,alphaO,tauS,tauO,deltaS,deltaO,lambda','leak_mode': 'PEs_FB','model_n':14},
    {'field': 'alphaS,alphaO,tau,delta,lambdaS,lambdaO','leak_mode': 'NA','model_n':15},
    {'field': 'alpha,tau,delta,lambda','leak_mode': 'BI','model_n':16},
    {'field': 'alpha,tau,delta,lambda','leak_mode': 'PEo_B','model_n':17},
    {'field': 'alpha,tau,delta,lambda','leak_mode': 'PEs_FB','model_n':18},
    {'field': 'alpha,tau,delta,lambdaS,lambdaO','leak_mode': 'NA','model_n':19},
    {'field': 'alpha,tau,delta','leak_mode': 'NA','model_n':20},
    {'field': 'alpha,tau,delta','leak_mode': 'NA','model_n':21}
]

if not os.path.exists(results):
    os.makedirs(results)
    print(f"Created results directory {results}")

if not os.path.exists(f"{results}/logs"):
    os.makedirs(f"{results}/logs")
    print(f"Created results-logs directory {results}/logs")

#subjects = []
#with open(subject_list_path) as infile:
#    next()    
#    for line in infile:
#        subjects.append(line.strip())

subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        if 'id' not in line:
            subjects.append(line.strip())


ssub_path = '/media/labs/rsmith/lab-members/osanchez/wellbeing/tom/TOM_new_model_v2/run_tom_all_models.ssub'

    
for index, model in enumerate(models, start=1):
    combined_results_dir = os.path.join(results, f"model{index}")
    field = model['field']
    model_n = model['model_n']
    leak_mode = model['leak_mode']

    if not os.path.exists(f"{combined_results_dir}/logs"):
        os.makedirs(f"{combined_results_dir}/logs")
        print(f"Created results-logs directory {combined_results_dir}/logs")
    
    for subject in subjects:
        ssub_path = '/media/labs/rsmith/lab-members/osanchez/wellbeing/tom/TOM_new_model_v2/run_tom_all_models.ssub'
        stdout_name = f"{combined_results_dir}/logs/{subject}-%J.stdout"
        stderr_name = f"{combined_results_dir}/logs/{subject}-%J.stderr"
    
        jobname = f'TOM_model-{model_n}-{subject}'
        os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} \"{subject}\" \"{combined_results_dir}\" \"{field}\" \"{model_n}\" \"{leak_mode}\" \"{experiment_mode}\"")
        #os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {subject_list} {combined_results_dir} {fit_hierarchical} {field} {drift_mapping} {bias_mapping} {thresh_mapping} {use_parfor} {use_ddm}")
    
        print(f"SUBMITTED JOB [{jobname}]")

    ###python3 /media/labs/rsmith/lab-members/osanchez/wellbeing/tom/TOM_new_model_v2/run_tom_all_models.py  /media/labs/rsmith/lab-members/osanchez/wellbeing/tom/model_output/all_models_without_nonresponsive_ids "prolific"


    ## joblist | grep coop | grep -Po 98.... | xargs scancel
    #OR
    ## joblist | grep coop | grep -Po 43... |xargs -n1 scancel
    #scancel -u osanchez