#%% import settings

from alloc import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fips_num', type=int, help='FIPS number (region)')
parser.add_argument('-p', '--point_index', type=int, help='Point index (calibration)')
parser.add_argument('-B', '--B', type=int, help='Budget')
parser.add_argument('-i', '--sa_index', type=int, help='SA index')
args = parser.parse_args()

alc = Alloc(fips_num = args.fips_num, obj_type = 'cost_unit', alg='reg_age', B=args.B, num_alloc = 20, point_index = args.point_index)


def get_combined_param_update_list(alc):
    combined_param_update_list = [
        ("beta", alc.model.beta),
        ("beta", alc.model.beta / 5),
        ("beta", alc.model.beta * 5),
        ("overall_alpha", alc.model.overall_alpha / 5),
        ("overall_alpha", alc.model.overall_alpha * 5),
        ("rae", alc.model.rae / 5),
        ("rae", alc.model.rae * 5),
        ("lam", alc.model.lam / 5),
        ("lam", alc.model.lam * 5),
        ("O", alc.model.O / 5),
        ("O", alc.model.O * 5),
        ("O", np.ones(np.shape(alc.model.O)) / np.sum(np.ones(np.shape(alc.model.O))) * np.sum(alc.model.O)),
        ("O", np.eye(alc.model.num_group) / alc.model.num_group * np.sum(alc.model.O)),
        ("VE_death", 0.01),
        ("VE_death", 0.9),
        ("VE_beta", 0.1),
        ("VE_beta", 0.9)
    ]
    return combined_param_update_list

def extract_values_from_filepath(filepath):
    pattern = r"Result/result_sa_(\d+)_p_(\d+)_B_(\d+)_date_\d+"
    match = re.match(pattern, filepath)
    if match:
        sa_index = int(match.group(1))
        point_index = int(match.group(2))
        B = int(match.group(3))
        return point_index, sa_index, B
    else:
        return None
#%% 
date = '0723'

alc.param_update_list = [combined_param_update_list[args.sa_index]]
param_update = f'("{alc.param_update_list[0][0]}", {alc.param_update_list[0][1]})'
alloc_test = alc.get_alloc_list()[:2]
alc.run_code(parallel=True, alloc_list = alloc_test)

alc.sol_history['param_update'] = param_update
file_name = 'sa_'+str(args.sa_index)+'_p_'+str(args.point_index)+'_B_'+str(args.B)
result_file_path = f'Result/result_{file_name}_date_{date}'
alc.sol_history.to_pickle(result_file_path)


# %% Read results and identify top results
import pandas as pd
import glob
import re

date = '0723'
pattern = f'Result/result_sa_*_{date}'
file_paths = glob.glob(pattern)

for file_path in file_paths:
    point_index, sa_index, B = extract_values_from_filepath(file_path)
    alc = Alloc(fips_num = fips_num, B = B, point_index = point_index)
    combined_param_update_list = get_combined_param_update_list(alc)
    alc.param_update_list = [combined_param_update_list[sa_index]]
    alloc_list = alc.get_alloc_top_age_reg(file_path)
    alc.run_code(parallel=True, alloc_list = alloc_list)
    param_update = f'("{alc.param_update_list[0][0]}", {alc.param_update_list[0][1]})'
    alc.sol_history['param_update'] = param_update
    existing_data = pd.read_pickle(file_path)
    combined_data = pd.concat([existing_data, alc.sol_history])
    combined_data.to_pickle(file_path+"_final")

#  %% Plot results
from plot import *
from alloc import *
import numpy as np
import pandas as pd
from scipy.integrate import odeint 
import glob
import re

date = '0723'
fips_num = '53011'
pattern = f'Result/result_sa_*_{date}_final'
file_paths = glob.glob(pattern)
file_paths = sorted(file_paths, key = extract_values_from_filepath)

outcome_df = pd.DataFrame(columns = ['point_index', 'sa_index','B','best_alloc', 'worst_alloc', 'best_alloc_ben', 'worst_alloc_ben','best_alloc_tot', 'worst_alloc_tot'])
[best_alloc, worst_alloc, best_alloc_ben, worst_alloc_ben] = [np.zeros(model.num_group)]*4
for file_path in file_paths:
    point_index, sa_index, B = extract_values_from_filepath(file_path)
    print(point_index, sa_index, B)
    alc = Alloc(fips_num = fips_num, B = B, point_index = point_index)
    combined_param_update_list = get_combined_param_update_list(alc)
    param_update_list = [combined_param_update_list[sa_index]]

    result_df = pd.read_pickle(file_path)
    top_alloc = result_df.nlargest(1, 'cost_unit')
    least_alloc = result_df.nsmallest(1, 'cost_unit')

    interest_df = pd.concat([top_alloc, least_alloc])

    ret_list = []
    for index, case in interest_df.iterrows():
        # model = VaccineModel(fips_num, param_update_list)
        # for param_name in alc.point.index.tolist():
        #     param_value = alc.point[param_name]
        #     model.update_param(param_name, param_value)
        # t = model.t_f
        # model.update_param("U", case['alloc']) 
        # ret = odeint(model.run_model, model.get_y0(), t)
        # ret_list.append(ret)
        print("Benefit ", case.cost_unit)
        if index==top_alloc.index: best_alloc = case.alloc; best_alloc_ben = case.benefits
        if index==least_alloc.index: worst_alloc = case.alloc; worst_alloc_ben = case.benefits
    
    new_row = pd.DataFrame([[point_index, sa_index, B, best_alloc, worst_alloc, best_alloc_ben, worst_alloc_ben, np.sum(best_alloc_ben), np.sum(worst_alloc_ben)]], columns = outcome_df.columns)
    outcome_df = pd.concat([outcome_df, new_row])
outcome_df.to_pickle("final_result")
    # model = VaccineModel(fips_num, param_update_list)
    # for param_name in alc.point.index.tolist():
    #     param_value = alc.point[param_name]
    #     model.update_param(param_name, param_value)
    # t = model.t_f
    # ret = odeint(model.run_model, model.get_y0(), t)
    # ret_list.append(ret)
    # file_name = f'p_{point_index}_sa_{sa_index}_B_{B}'
    # # plot_deaths_with_calib(t, ret_list, lw=2)
    # plot_deaths_with_calib(t, ret_list, lw=2, filename = file_name)

# %%
