#%%
from vaccinemodel import *
import warnings

class Alloc:
    def __init__(self, fips_num, alloc=None, init_param_list = [], point_index=-1, obj_type='', alg='', B=1000, num_alloc = 1, n_iter=0, param_update_list=[], debug=False):
        self.points = pd.read_csv(f'Data/{fips_num}_points_new.csv')
        self.point_index = point_index
        self.point = self.points.iloc[self.point_index]
        self.init_param_list = init_param_list
        for i in range(len(self.point)):
            self.init_param_list.append((self.point.index[i], self.point[i]))
        self.fips_num = fips_num
        self.model = VaccineModel(self.fips_num)
        self.no_policy_outcome_deaths = np.ones(self.model.num_group)*-1
        self.no_policy_outcome_vacc = np.ones(self.model.num_group)*-1
        self.alg = alg
        self.obj_type = obj_type
        self.sol_history_columns = ['fips', 'point_index', 'n_iter', 'alg', 'alloc', 'obj', 'cost_deaths', 'disparity_deaths',
                                     'cost_vacc', 'disparity_vacc',  'benefits_deaths', 'benefits_vacc' ]
        self.sol_history = pd.DataFrame(columns =  self.sol_history_columns)
        self.alloc = np.zeros(self.model.num_group)
        if alloc is not None:
            self.alloc = alloc
        self.B = B
        self.num_alloc = num_alloc
        self.n_iter = n_iter
        self.param_update_list = param_update_list
        self.ret_list = []
        self.eta_list = []
        self.all_eta_list = []
        self.debug = debug
        
    # get outcomes
    def get_no_policy(self, save_result = False):
        self.model = VaccineModel(self.fips_num, init_param_list = self.init_param_list, 
                                  param_update_list=self.param_update_list, debug = self.debug)
        t = self.model.t_f
        self.model.update_param("U", np.zeros(self.model.num_group))
        ret = odeint(self.model.run_model, self.model.get_y0(), t)
        # print("No U init eta ", np.round(self.model.mean_eta[0],2), "last eta ", np.round(self.model.mean_eta[-1],2),
        #       " init p_emo ", np.round(self.model.mean_emo[0],2), "last p_emo ", np.round(self.model.mean_emo[-1],2),
        #                 " init p_rat ", np.round(self.model.mean_rat[0],2)," last p_rat ", np.round(self.model.mean_rat[-1],2))
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), self.model.num_group, self.model.num_comp)))
        self.no_policy_outcome_deaths = DA[:, -1] + DP[:, -1]
        self.no_policy_outcome_vacc = SP[:,-1] + IP[:,-1] + RP[:,-1]
        if save_result:
            return ret
 
    def calculate_outcome_metrics(self, benefits, alloc):
        # policy_cost = np.array([1, np.sum(np.square(alloc)), np.sum(alloc / self.model.N_by_group)])    
        outcome_cost_fair = [sum(benefits)]
        avg_benefit = np.mean(benefits)
        outcome_disparity = np.abs(benefits / avg_benefit - 1)
        outcome_disparity_fair = np.array([np.max(outcome_disparity), np.sum(outcome_disparity)])
        return outcome_cost_fair, outcome_disparity_fair

    def run_alloc(self, alloc):
        self.model = VaccineModel(self.fips_num, init_param_list = self.init_param_list, 
                                  param_update_list=self.param_update_list, debug = self.debug)
        t = self.model.t_f
        self.model.update_param("U", alloc)
        ret = odeint(self.model.run_model, self.model.get_y0(), t)
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), self.model.num_group, self.model.num_comp)))
        self.ret_list.append(ret)
        self.eta_list.append(self.model.mean_eta)
        self.all_eta_list.append(self.model.eta_all)
        benefits_deaths = (self.no_policy_outcome_deaths -  (DA[:, -1] + DP[:, -1])).round(4)
        outcome_cost_fair_deaths, outcome_disparity_fair_deaths = self.calculate_outcome_metrics(benefits_deaths, alloc)
        benefits_vacc = ((SP[:,-1] + IP[:,-1] + RP[:,-1]) - self.no_policy_outcome_vacc).round(4)
        outcome_cost_fair_vacc, outcome_disparity_fair_vacc = self.calculate_outcome_metrics(benefits_vacc, alloc)

        # benefits_deaths_by_pop = benefits_deaths/self.model.N_by_group
        # outcome_cost_fair_deaths_by_pop, outcome_disparity_fair_deaths_by_pop = self.calculate_outcome_metrics(benefits_deaths_by_pop, alloc)
        # benefits_vacc_by_pop = benefits_vacc/self.model.N_by_group
        # outcome_cost_fair_vacc_by_pop, outcome_disparity_fair_vacc_by_pop = self.calculate_outcome_metrics(benefits_vacc_by_pop, alloc)

        new_row = pd.DataFrame([[self.fips_num, self.point_index, self.n_iter, self.alg, alloc, self.obj_type, outcome_cost_fair_deaths, outcome_disparity_fair_deaths,
                                 outcome_cost_fair_vacc, outcome_disparity_fair_vacc, benefits_deaths, benefits_vacc]], columns=self.sol_history_columns)
        if self.alg != 'ga':
            return new_row
        else:
            self.sol_history = pd.concat([self.sol_history, new_row], ignore_index=False)
            value_mapping = {
                'cost_deaths': -outcome_cost_fair_deaths[0],
                'max_disparity_deaths': outcome_disparity_fair_deaths[0],
                # 'sum_disparity_deaths': outcome_disparity_fair_deaths[1],
                # 'cost_deaths_by_pop': -outcome_cost_fair_deaths_by_pop[0],
                # 'max_disparity_deaths_by_pop': outcome_disparity_fair_deaths_by_pop[0],
                # 'sum_disparity_deaths_by_pop': outcome_disparity_fair_deaths_by_pop[1],
                'cost_vacc': -outcome_cost_fair_vacc[0],
                'max_disparity_vacc': outcome_disparity_fair_deaths[0],
                # 'sum_disparity_vacc': outcome_disparity_fair_deaths[1],
                # 'cost_vacc_by_pop': -outcome_cost_fair_vacc_by_pop[0],
                # 'max_disparity_vacc_by_pop': outcome_disparity_fair_deaths_by_pop[0],
                # 'sum_disparity_vacc_by_pop': outcome_disparity_fair_deaths_by_pop[1]
            }
            # print(value_mapping.get(self.obj_type, 0) , ' alloc ', alloc)
            return value_mapping.get(self.obj_type, 0) 
    
    def get_alloc_by_lhs(self, tot_samp, B, n_iter):
        import pyDOE
        import numpy as np
        np.random.seed(1234+n_iter)
        lhs_samp = pyDOE.lhs(self.model.num_group-1, tot_samp)
        sorted_lhs = np.sort(lhs_samp, axis=1)
        sorted_lhs = np.hstack((np.zeros((tot_samp, 1)), sorted_lhs, np.ones((tot_samp, 1))))
        diff_lhs = sorted_lhs[:, 1:] - sorted_lhs[:, :-1]
        return (diff_lhs)*B

    def get_alloc_by_reg(self, num_alloc, tot_unit):
        from itertools import combinations_with_replacement
        alloc_comb = list(combinations_with_replacement(range(self.model.num_reg_group), num_alloc))
        alloc = np.zeros((len(alloc_comb),self.model.num_group))
        for k in range(len(alloc_comb)):
            for j in range(self.model.num_group):
                alloc[k][j] = alloc_comb[k].count(j//self.model.num_age_group)*tot_unit/num_alloc/self.model.num_age_group
        return alloc

    def get_alloc_by_age(self, num_alloc, tot_unit):
        from itertools import combinations_with_replacement
        alloc_comb = list(combinations_with_replacement(range(self.model.num_reg_group), num_alloc))
        alloc = np.zeros((len(alloc_comb),self.model.num_group))
        for k in range(len(alloc_comb)):
            for j in range(self.model.num_group):
                alloc[k][j] = alloc_comb[k].count(j%self.model.num_age_group)*tot_unit/num_alloc/self.model.num_age_group
        return alloc
    
    def get_vertex_alloc(self, num_group, tot_alloc):
        alloc_list= list(self.get_alloc_by_reg(1, tot_alloc))
        alloc_list += list(self.get_alloc_by_age(1, tot_alloc))
        for i in range(num_group):
            alloc = np.zeros(num_group)
            alloc[i] = tot_alloc
            alloc_list.append(alloc)
        return alloc_list

    def get_alloc_top_age_reg(self, result_df, outcome_metric):
        def mark_based_on_first_n_elements(row):
            if len(np.unique(row['alloc'][:5]))== 1: return 'reg_based'
            else: return 'age_based'
        
        alloc_columns = [col for col in result_df.columns if col.startswith("alloc_")]
        result_df['alloc'] = (result_df[alloc_columns].values.tolist())
        result_df['mark'] = result_df.apply(mark_based_on_first_n_elements, axis=1)
        reg_df = result_df[result_df['mark']=='reg_based']
        age_df = result_df[result_df['mark']=='age_based']
        top_n = 10
        top_reg = reg_df.nlargest(top_n, outcome_metric)
        top_age = age_df.nlargest(top_n, outcome_metric) 

        if 'disparity' in outcome_metric:
            top_reg = reg_df.nsmallest(top_n, outcome_metric)
            top_age = age_df.nsmallest(top_n, outcome_metric)

        num_group = len(top_reg.iloc[0].alloc)
        tot_alloc = sum(top_reg.iloc[0].alloc)

        alloc_list = []
        for i in range(len(top_reg)):
            for j in range(len(top_age)):
                alloc = np.array(top_reg.iloc[i].alloc)/tot_alloc * np.array(top_age.iloc[j].alloc)/tot_alloc
                if sum(alloc) >0: 
                    alloc = alloc/sum(alloc)*sum(top_reg.iloc[i].alloc)
                    alloc_list.append(alloc)

        # Add vertex points as well
        alloc_list += self.get_vertex_alloc(num_group, tot_alloc)
        alloc_list = np.unique(np.array(alloc_list), axis=0)
        return (top_reg, top_age, alloc_list)
    
    def get_alloc_list(self):
        alg, B, num_alloc, n_iter = self.alg, self.B, self.num_alloc, self.n_iter
        alloc_list = []
        if alg=='lhs':
            top_n = 5
            tot_samp = len(self.get_alloc_by_reg(num_alloc, B))*2+top_n**2
            alloc_list = self.get_alloc_by_lhs(tot_samp, B, n_iter)
        elif alg=='reg_age':
            alloc_list_reg = self.get_alloc_by_reg(num_alloc, B)
            alloc_list_age = self.get_alloc_by_age(num_alloc, B)
            alloc_list = np.concatenate((alloc_list_reg, alloc_list_age), axis=0)
        elif alg=='vertex':
            alloc_list = self.get_vertex_alloc(self.model.num_group,B)
        return(alloc_list)

    def run_code_parallel(self, alloc_list):
        import multiprocess as mp
        global sol_history
        manager = mp.Manager()
        sol_history = manager.list()
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(self.run_alloc, alloc_list, chunksize=1)
        self.sol_history = pd.concat([self.sol_history] + results, ignore_index=True)
        return None  # Or return something meaningful
    
    def run_ga(self, n_iter=0):
        from scipy.optimize import Bounds, LinearConstraint, differential_evolution
        import multiprocess as mp
        bounds = Bounds([0]*self.model.num_group, [self.B]*self.model.num_group)
        constraint = LinearConstraint(np.ones((1, self.model.num_group)), [self.B], [self.B])
        self.get_no_policy()
        # res = differential_evolution(self.run_alloc_ga, bounds, constraints=constraint, seed=1234+n_iter, workers=mp.cpu_count(), maxiter=5000, popsize=15, tol= 10)        res = differential_evolution(self.run_alloc_ga, bounds, constraints=constraint, seed=1234+n_iter, workers=mp.cpu_count(), maxiter=5000, popsize=15, tol= 10)
        res = differential_evolution(self.run_alloc, bounds, constraints=constraint, 
                                     x0 = np.zeros(self.model.num_group)*(self.B/self.model.num_group-1),
                                     seed=1234+n_iter, workers=1, maxiter=50000, popsize=15, tol= 10)
        return (res)

    def run_code(self, parallel=False, alloc_list=None, save_result=False):
        self.get_no_policy(save_result)
        if alloc_list is None:
            alloc_list = self.get_alloc_list()
        if parallel:
            self.run_code_parallel(alloc_list)
            
        else:
            for alloc in alloc_list:
                results = self.run_alloc(alloc)
                self.sol_history = pd.concat([self.sol_history, results], ignore_index=True)
        
        self.sol_history = organize_df(self.sol_history)
        
        if save_result:
            if parallel: 
                warnings.warn("The 'save_result' option is not supported in parallel mode.")
            return (self.ret_list, self.eta_list, self.all_eta_list)

def organize_df(df, sort_by='disparity_deaths_0'):
    list_outcome = ['alloc', 'cost_deaths', 'disparity_deaths',  'cost_vacc', 'disparity_vacc', 'benefits_deaths', 'benefits_vacc']
    # list_outcome = ['alloc']    
    for column in list_outcome:
        new_columns =  [f'{column}_{i}' for i in range(len(df[column].iloc[0]))]
        new_df = pd.DataFrame(df[column].tolist(), columns=new_columns)
        # df.drop(column, axis=1, inplace=True)
        df = pd.concat([df, new_df], axis=1)
    df = df.sort_values(sort_by, ascending = False if 'cost' in sort_by else True)
    return df

#%% User can specify point_index, obj_type, alg, B, num_alloc, n_iter
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fips_num', type=int, help='FIPS number (region)')
    parser.add_argument('-p', '--point_index', type=int, help='Point index (calibration)')
    parser.add_argument('-B', '--B', type=int, help='Total Budget')
    args = parser.parse_args()

    alc = Alloc(fips_num = args.fips_num, obj_type = 'all', alg='reg_age', B=args.B, num_alloc = 20, point_index = args.point_index)
    # alc = Alloc(fips_num = 53033, obj_type = 'all', alg='reg_age', B=5000, num_alloc = 1, point_index = 0)

    date = '1030'
    alloc_test = alc.get_alloc_list()
    alc.run_code(parallel=True, alloc_list = alloc_test)

    result_file_path = f'Result/{alc.fips_num}_obj_{alc.obj_type}_alg_{alc.alg}_B_{alc.B}_point_{alc.point_index}_date_{date}'
    alc.sol_history.to_pickle(result_file_path)
    with np.printoptions(linewidth=1000):
        (alc.sol_history).to_csv(result_file_path)


# %%
# file_path = 'refined_result.csv'
# df = pd.read_csv(file_path)
# df = organize_df(df)
# df.alloc.values[:5]

# # %%
# ret_list = []
# for alloc in ([np.zeros(25)] + df.iloc[[0, -1]]['alloc'].tolist()):
#     alc.model = VaccineModel(alc.fips_num, init_param_list = alc.init_param_list, 
#                                   param_update_list=alc.param_update_list)
#     t = alc.model.t_f
#     alc.model.update_param("U", alloc)
#     ret = odeint(alc.model.run_model, alc.model.get_y0(), t)
#     ret_list.append(ret)
# plot.plot_results_with_calib(alc.model, t, ret_list, lw=0.5, error_bar = True)
# %%
# import seaborn as sns
# import matplotlib.pyplot as plt

# num = 6
# plt.figure(figsize = (16,2))
# sns.heatmap([df.alloc[num]/np.sum(df.alloc[num]), df.benefits_deaths[num]/np.sum(df.benefits_deaths[num]), df.benefits_vacc[num]/np.sum(df.benefits_vacc[num])], cmap="Blues", xticklabels=5, yticklabels=1, cbar=True, annot=True, fmt = '.2f',annot_kws={"fontsize": 8})
# plt.show()
# plt.figure(figsize = (16,2))
# sns.heatmap([df.alloc[num], df.benefits_deaths[num], df.benefits_vacc[num]], cmap="Blues", xticklabels=5, yticklabels=1, cbar=True, annot=True, fmt = '.0f',annot_kws={"fontsize": 8})
# plt.show()
# %%
#Now get the best results
# date = '1030'
# fips_list = [53047,53033]
# global_top_df = pd.DataFrame()
# for fips_num in fips_list:
#     for p in range(5):
#         B = [200,10000][fips_list.index(fips_num)]
#         result_file_path = f'Result/{fips_num}_obj_all_alg_reg_age_B_{B}_point_{p}_date_{date}'
#         result_df = pd.read_csv(result_file_path)
#         outcome_list = ['cost_deaths_0','disparity_deaths_0','cost_vacc_0','disparity_vacc_0']
#         # outcome_list = ['disparity_deaths_1','disparity_vacc_1']
#         for outcome_metric in outcome_list:
#             alc = Alloc(fips_num = fips_num, obj_type = outcome_metric, alg='reg_age', B=B, num_alloc = 20, point_index = p)
#             (top_reg, top_age, alloc_test) = alc.get_alloc_top_age_reg(result_df, outcome_metric)
#             alc.run_code(parallel=True, alloc_list = alloc_test)
#             final_df = pd.concat([top_reg, top_age, alc.sol_history], ignore_index = True)
#             final_df['obj'] = outcome_metric
#             top_df = final_df.sort_values(outcome_metric, ascending = False if 'cost' in outcome_metric else True).iloc[:1]
#             global_top_df = pd.concat([global_top_df, top_df], ignore_index = True)
# # %%
# with np.printoptions(linewidth=10000):
#     global_top_df.to_csv(f'top_results_final_{date}_new.csv')
# # global_top_df.to_pickle(f'top_results_final_{date}.pkl')

# # %%
