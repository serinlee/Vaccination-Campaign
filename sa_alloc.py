#####################################################################
# A Module that runs allocation problem with varying model parameters
#####################################################################
#%% import settings

from alloc import Alloc
import numpy as np
import pandas as pd

def get_sa_ranges():
    vr_list = np.linspace(0.0, 0.0003, 3)
    p_list = np.linspace(0, 1, 3)
    kr_list = np.linspace(15000, 25000, 3)
    ke_list = np.linspace(9, 15, 3)
    return (vr_list, p_list, kr_list, ke_list)

def get_full_way_sa():
    (vr_list, p_list, kr_list, ke_list) = get_sa_ranges()

    sa_list = []
    for p in p_list:
        for k_R in kr_list:
            for k_E in ke_list:
                for vr in vr_list:
                    init_param_list = [("p_emotional", p), ("vaccine_risk", vr), ("k_R",k_R), ("k_E", k_E)]
                    sa_list.append(init_param_list)
    return sa_list

def get_mid_sa():
    (vr_list, p_list, kr_list, ke_list) = get_sa_ranges()
    kr_list = [kr_list[1]]
    ke_list =[ke_list[1]]

    sa_list = []
    for vr in vr_list:
        for k_R in kr_list:
            for k_E in ke_list:
                for p in p_list:
                    init_param_list = [("p_emotional", p), ("vaccine_risk", vr), ("k_R",k_R), ("k_E", k_E)]
                    sa_list.append(init_param_list)
    return sa_list

#%% Actually run
if __name__ == '__main__':
    date = '0320'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fips_num', type=int, help='FIPS number (region)')
    parser.add_argument('-p', '--point_index', type=int, help='Point index (calibration)')
    parser.add_argument('-B', '--B', type=int, help='Budget')
    parser.add_argument('-i', '--sa_index', type=int, help='SA index')
    args = parser.parse_args()

    global_top_df = pd.DataFrame()
    full_sa_list = get_mid_sa()
    sa_list = full_sa_list[args.sa_index::3]

    num_alloc = 1
    for sa in sa_list:
        print(sa)
        init_param_list  = sa.copy()
        alc = Alloc(fips_num = args.fips_num, obj_type = 'all', alg='reg_age', 
                    B=args.B, num_alloc = num_alloc, point_index = 0,
                    init_param_list  = init_param_list)
        alloc_test = alc.get_alloc_list()
        alc.run_code(parallel=True, alloc_list = alloc_test, save_result = False)
        outcome_list = ['tot_benefits_vacc']
        for outcome_metric in outcome_list:
            (top_reg, top_age, alloc_test) = alc.get_alloc_top_age_reg(alc.sol_history, outcome_metric)
            alc = Alloc(fips_num = args.fips_num, init_param_list  = init_param_list, obj_type = outcome_metric, 
                        alg='reg_age', B=args.B, num_alloc = num_alloc, point_index = args.point_index)
            alc.run_code(parallel=True, alloc_list = alloc_test)
            final_df = pd.concat([top_reg, top_age, alc.sol_history], ignore_index = True)
            top_df = final_df.sort_values(outcome_metric, ascending = False).iloc[:1]
            top_df['obj'] = outcome_metric
            top_df['param_update'] = str(sa)
            global_top_df = pd.concat([global_top_df, top_df], ignore_index = True)
            with np.printoptions(linewidth=10000):
                    final_df.to_csv('test.csv')

    file_name = f'Results/SA_Result/top_{args.fips_num}_sa_{args.sa_index}_p_{args.point_index}_B_{args.B}_date_{date}'
    with np.printoptions(linewidth=10000):
        global_top_df.to_csv(file_name+'.csv')
    global_top_df.to_pickle(file_name+'.pkl')

