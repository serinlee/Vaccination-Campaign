#%% import settings

from alloc import Alloc
import numpy as np
import pandas as pd

def get_full_way_sa(fips):
    # Upper, lower bound, and number of trials
    fips_list = [53047, 53033]
    fips_ind = fips_list.index(fips)
    vr_list = np.linspace(0, 0.0001, 5)
    kr_list = np.linspace(5000, 10000, 5)
    ke_list = [np.linspace(1,5,5), np.linspace(5,15,5)][fips_ind]
    p_list = np.linspace(0,1,5)

    sa_list = []
    for p in p_list:
        for k_R in kr_list:
            for k_E in ke_list:
                for vr in vr_list:
                    init_param_list = [("p1", p),("p2", p),("p3",p),("p4", p),("p5", p), 
                                       ("vaccine_risk", vr), ("k_R",k_R), ("k_E", k_E)]
                    sa_list.append(init_param_list)
    return sa_list

def get_combined_param_update_bounds():
    # Upper, lower bound, and number of trials
    combined_param_update_bounds = [
        [("vaccine_risk", 0, 0.0001, 11)],
        [("k_R", 0.1*5000, 30*5000, 11)],
        [("k_E", 0.1, 30, 11)],
        [("overall_alpha", 0.0001, 0.001, 11)],
        [("beta", 1, 10, 11)],
        [("p_online", 0, 1, 11)],
        [("lam", 0, 0.2, 11)],
        [("O_m", 1, 10, 11)],
        [("VE_beta", 0.0, 1.0, 11)],
        [("VE_death", 0.0, 1.0, 11)],
        [("rae", 100, 1000, 11)],
    ]
    return combined_param_update_bounds

def get_sa_list(param_bounds):
    param_name, lower_bound, upper_bound, num_trials = param_bounds[0]
    step = (upper_bound - lower_bound) / (num_trials - 1)
    sa_list = [(param_name, round(lower_bound + i * step, 5)) for i in range(num_trials)]
    return sa_list

def extract_values_from_filepath(filepath):
    pattern = r"SA_Result/result_sa_(\d+)_p_(\d+)_B_(\d+)_date_\d+"
    match = re.match(pattern, filepath)
    if match:
        sa_index = int(match.group(1))
        point_index = int(match.group(2))
        B = int(match.group(3))
        return point_index, sa_index, B
    else:
        return None
#%% 
date = '1104'
num_alloc = 1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fips_num', type=int, help='FIPS number (region)')
parser.add_argument('-p', '--point_index', type=int, help='Point index (calibration)')
parser.add_argument('-B', '--B', type=int, help='Budget')
parser.add_argument('-i', '--sa_index', type=int, help='SA index')
args = parser.parse_args()

global_top_df = pd.DataFrame()
# param_bounds = get_combined_param_update_bounds()[args.sa_index]
# sa_list = get_sa_list(param_bounds)
full_sa_list = get_full_way_sa(args.fips_num)
sa_list = full_sa_list[args.sa_index::10]

for sa in sa_list[:1]:
    alc = Alloc(fips_num = args.fips_num, obj_type = 'all', alg='reg_age', B=args.B, num_alloc = num_alloc, point_index = args.point_index)
    alc.init_param_list  = sa
    print(sa)
    alloc_test = alc.get_alloc_list()
    alc.run_code(parallel=True, alloc_list = alloc_test, save_result = False)
    #Now get the best results
    outcome_list = ['cost_deaths_0','disparity_deaths_0','cost_vacc_0','disparity_vacc_0']
    for outcome_metric in outcome_list:
        (top_reg, top_age, alloc_test) = alc.get_alloc_top_age_reg(alc.sol_history, outcome_metric)
        alc = Alloc(fips_num = args.fips_num, obj_type = outcome_metric, alg='reg_age', B=args.B, num_alloc = num_alloc, point_index = args.point_index)
        alc.init_param_list  = sa
        alc.run_code(parallel=True, alloc_list = alloc_test)
        final_df = pd.concat([top_reg, top_age, alc.sol_history], ignore_index = True)
        top_df = final_df.sort_values(outcome_metric, ascending = False if 'cost' in outcome_metric else True).iloc[:1]
        top_df['obj'] = outcome_metric
        top_df['param_update'] = str(sa)
        global_top_df = pd.concat([global_top_df, top_df], ignore_index = True)

file_name = f'SA_Result/top_{args.fips_num}_sa_{args.sa_index}_p_{args.point_index}_B_{args.B}_date_{date}'
with np.printoptions(linewidth=10000):
    global_top_df.to_csv(file_name+'.csv')
global_top_df.to_pickle(file_name+'.pkl')

# %% For testing purpose
# import plot

# sa_index = 0
# fips_num = 53047
# point_index = 8
# B = 100
# date = '1025'

# param_bounds = get_combined_param_update_bounds() [sa_index]
# param_name, lower_bound, upper_bound, num_trials = param_bounds[0]
# step = (upper_bound - lower_bound) / (num_trials - 1)
# sa_list = [(param_name, round(lower_bound + i * step, 1)) for i in range(num_trials)]
# global_sol_history = pd.DataFrame()

# for sa in sa_list:
#     alc = Alloc(fips_num = fips_num, obj_type = 'all', alg='reg_age', B=B, num_alloc = 1, point_index = point_index)
#     alc.param_update_list  = [sa]
#     alloc_test = np.stack((np.zeros(25), alc.get_alloc_list()[1]))
#     ret_list = alc.run_code(parallel=False, alloc_list = alloc_test, save_result = True)
#     alc.sol_history['param_update'] = str(sa)
#     global_sol_history = pd.concat([global_sol_history, alc.sol_history])
#     plot.plot_results_with_calib(alc.model, alc.model.t_f, ret_list, lw=0.5, error_bar = True)

#%% additional steps
# import ast
# B = 100
# file_path = f'SA_Result/sa_53047_VE_beta_p_8_B_100_date_1025'
# df = pd.read_pickle(file_path+'.pkl')
# grouped = df.groupby('param_update')
# grouped_dfs = {group_name: group for group_name, group in grouped}
# outcome_list = ['cost_deaths_0','disparity_deaths_0']
# row = df.iloc[0]

# global_top_df = pd.DataFrame()
# for param_update, group_df in grouped_dfs.items():
#     for outcome_metric in outcome_list:
#         print(f"Param: {[ast.literal_eval(param_update)]}, outcome: {outcome_metric}")
#         alc = Alloc(fips_num = row['fips'], obj_type = outcome_metric, alg='reg_age', 
#                     B=B, num_alloc = 20, point_index = row['point_index'])
#         alc.param_update_list = [ast.literal_eval(param_update)]
#         (top_reg, top_age, alloc_test) = alc.get_alloc_top_age_reg(group_df, outcome_metric)
#         alc.run_code(parallel=True, alloc_list = alloc_test)
#         final_df = pd.concat([top_reg, top_age, alc.sol_history], ignore_index = True)
#         final_df['obj'] = outcome_metric
#         final_df['param_update'] = param_update
#         top_df = final_df.sort_values(outcome_metric, ascending = False if 'cost' in outcome_metric else True).iloc[:1]
#         global_top_df = pd.concat([global_top_df, top_df], ignore_index = True)
# with np.printoptions(linewidth=10000):
#     global_top_df.to_csv(file_path+"_top.csv")

# %%
