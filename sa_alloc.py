#%% import settings

from alloc import Alloc
import numpy as np
import pandas as pd
import glob
import ast
import seaborn as sns
import matplotlib.pyplot as plt

import plot
from itertools import product
from scipy.interpolate import interp1d

def get_full_way_sa():
    vr_list = np.linspace(0.0, 0.0003, 3)
    p_list = np.linspace(0,1,3)
    kr_list = np.linspace(15000, 25000, 5)
    ke_list = np.linspace(14, 16, 5)

    sa_list = []
    for p in p_list:
        for k_R in kr_list:
            for k_E in ke_list:
                for vr in vr_list:
                    init_param_list = [("p1", p),("p2", p),("p3",p),("p4", p),("p5", p), 
                                       ("vaccine_risk", vr), ("k_R",k_R), ("k_E", k_E)]
                    sa_list.append(init_param_list)
    return sa_list

def get_mid_sa():
    vr_list = np.linspace(0.0, 0.0003, 3)
    p_list = np.linspace(0,1,3) # Fix!!! 
    kr_list = [20000]
    ke_list = [15]

    sa_list = []

    for vr in vr_list:
        for k_R in kr_list:
            for k_E in ke_list:
                for p in p_list:
                    init_param_list = [("p1", p),("p2", p),("p3",p),("p4", p),("p5", p), 
                                    ("vaccine_risk", vr), ("k_R",k_R), ("k_E", k_E)]
                    sa_list.append(init_param_list)
    return sa_list

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
#%% Actually run
if __name__ == '__main__':
    date = '0220'
    num_alloc = 1

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fips_num', type=int, help='FIPS number (region)')
    parser.add_argument('-p', '--point_index', type=int, help='Point index (calibration)')
    parser.add_argument('-B', '--B', type=int, help='Budget')
    parser.add_argument('-i', '--sa_index', type=int, help='SA index')
    args = parser.parse_args()

    global_top_df = pd.DataFrame()
    full_sa_list = get_full_way_sa()
    sa_list = full_sa_list[args.sa_index::1]

    for sa in sa_list:
        alc = Alloc(fips_num = args.fips_num, obj_type = 'all', alg='reg_age', 
                    B=args.B, num_alloc = num_alloc, point_index = 0,
                    init_param_list  = sa.copy())
        alloc_test = alc.get_alloc_list()
        alc.run_code(parallel=True, alloc_list = alloc_test, save_result = False)
        outcome_list = ['cost_deaths_0','cost_vacc_0']
        outcome_list = ['cost_vacc_0']
        for outcome_metric in outcome_list:
            (top_reg, top_age, alloc_test) = alc.get_alloc_top_age_reg(alc.sol_history, outcome_metric)
            alc = Alloc(fips_num = args.fips_num, init_param_list  = sa.copy(), obj_type = outcome_metric, 
                        alg='reg_age', B=args.B, num_alloc = num_alloc, point_index = args.point_index)
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
