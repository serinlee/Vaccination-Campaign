#%%
from vaccinemodel import *
import plot

class Alloc:
    def __init__(self, fips_num, alloc=None, point_index=-1, obj_type='', alg='', B=1000, num_alloc = 1, n_iter=0, param_update_list=[]):
        self.points = pd.read_csv(f'Data/{fips_num}_points.csv')
        self.point_index = point_index
        self.point = self.points.iloc[self.point_index]
        self.fips_num = fips_num
        self.model = VaccineModel(self.fips_num)
        self.no_policy_outcome_deaths = np.ones(self.model.num_group)*-1
        self.no_policy_outcome_vacc = np.ones(self.model.num_group)*-1
        self.alg = alg
        self.obj_type = obj_type
        self.sol_history_columns = ['point_index', 'n_iter', 'alg', 'alloc', 'outcome_cost_fair_deaths', 'outcome_disparity_fair_deaths', 'benefits_deaths', 'outcome_cost_fair_vacc', 'outcome_disparity_fair_vacc', 'benefits_vacc']
        self.sol_history = pd.DataFrame(columns =  self.sol_history_columns)
        self.alloc = np.zeros(self.model.num_group)
        if alloc is not None:
            self.alloc = alloc
        self.B = B
        self.num_alloc = num_alloc
        self.n_iter = n_iter
        self.param_update_list = param_update_list
        print("Update initial param")
        for param_name in self.point.index.tolist():
            param_value = self.point[param_name]
            self.model.update_param(param_name, param_value)
        
    # get outcomes
    def get_no_policy(self):
        self.model = VaccineModel(self.fips_num, self.param_update_list)
        print("Update initial param in no policy")
        for param_name in self.point.index.tolist():
            param_value = self.point[param_name]
            self.model.update_param(param_name, param_value)
        t = self.model.t_f
        print("Update param U in no policy")
        self.model.update_param("U", np.zeros(self.model.num_group))
        ret = odeint(self.model.run_model, self.model.get_y0(), t)
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), self.model.num_group, self.model.num_comp)))
        self.no_policy_outcome_deaths = DA[:, -1] + DP[:, -1]
        self.no_policy_outcome_vacc = SP[:,-1] + IP[:,-1] + RP[:,-1]

    def calculate_outcome_metrics(self, benefits, alloc):
        policy_cost = np.array([1, np.sum(np.square(alloc)), np.sum(alloc / self.model.N_by_group)])    
        outcome_cost_fair = sum(benefits) / policy_cost
        avg_benefit = np.mean(benefits)
        outcome_disparity = np.abs(benefits / avg_benefit - 1)
        outcome_disparity_fair = np.array([np.max(outcome_disparity), np.sum(outcome_disparity)])
        return outcome_cost_fair, outcome_disparity_fair

    def run_alloc(self, alloc):
        self.model = VaccineModel(self.fips_num, self.param_update_list, debug=True)
        print("Update initial param in run_alloc")
        for param_name in self.point.index.tolist():
            param_value = self.point[param_name]
            self.model.update_param(param_name, param_value)
        t = self.model.t_f
        print("Update param U in run_alloc")
        self.model.update_param("U", alloc)
        ret = odeint(self.model.run_model, self.model.get_y0(), t)
        plot.plot_results_with_calib(self.model, self.model.t_f, [ret], error_bar=True)
        
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), self.model.num_group, self.model.num_comp)))

        benefits_deaths = (self.no_policy_outcome_deaths -  (DA[:, -1] + DP[:, -1])).round(4)
        outcome_cost_fair_deaths, outcome_disparity_fair_deaths = self.calculate_outcome_metrics(benefits_deaths, alloc)

        benefits_vacc = ((SP[:,-1] + IP[:,-1] + RP[:,-1]) - self.no_policy_outcome_vacc).round(4)
        outcome_cost_fair_vacc, outcome_disparity_fair_vacc = self.calculate_outcome_metrics(benefits_vacc, alloc)

        new_row = pd.DataFrame([[self.point_index, self.n_iter, self.alg, alloc, outcome_cost_fair_deaths, outcome_disparity_fair_deaths, 
                                 benefits_deaths, outcome_cost_fair_vacc, outcome_disparity_fair_vacc, benefits_vacc]], columns=self.sol_history_columns)            
        return new_row

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

    def get_alloc_top_age_reg(self, result_file_path):
        result_df = pd.read_pickle(result_file_path)
        reg_df, age_df = np.array_split(result_df, 2)
        top_reg = reg_df.nlargest(5, 'cost_unit')
        top_age = age_df.nlargest(5, 'cost_unit')
        alloc_list = []
        for i in range(len(top_reg)):
            for j in range(len(top_age)):
                alloc = ((top_reg.iloc[i].alloc*top_age.iloc[j].alloc)/sum(top_reg.iloc[i].alloc*top_age.iloc[j].alloc)*sum(top_reg.iloc[i].alloc))
                alloc_list.append(alloc)
        return alloc_list
    
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

    def run_code(self, parallel=False, alloc_list=None):
        self.get_no_policy()
        if alloc_list is None:
            alloc_list = self.get_alloc_list()
        if parallel:
            self.run_code_parallel(alloc_list)
        else:
            for alloc in alloc_list:
                results = self.run_alloc(alloc)
                self.sol_history = pd.concat([self.sol_history, results], ignore_index=True)

#%% User can specify popint_index, obj_type, alg, B, num_alloc, n_iter

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('-f', '--fips_num', type=int, help='FIPS number (region)')
# parser.add_argument('-p', '--point_index', type=int, help='Point index (calibration)')
# parser.add_argument('-B', '--B', type=int, help='Total Budget')
# args = parser.parse_args()

# alc = Alloc(fips_num = args.fips_num, obj_type = 'all', alg='reg_age', B=args.B, num_alloc = 1, point_index = args.point_index)
alc = Alloc(fips_num = 53033, obj_type = 'all', alg='reg_age', B=5000, num_alloc = 1, point_index = 1)

date = '0918'
# param_update = f'("{param_update_list[0][0]}", {param_update_list[0][1]})'
param_update = None
alloc_test = alc.get_alloc_list()[:1]
alc.run_code(parallel=True, alloc_list = alloc_test)

# result_file_path = f'Result/{alc.fips_num}_obj_{alc.obj_type}_alg_{alc.alg}_B_{alc.B}_date_{date}_point_{point_index}'
# if param_update is not None:
#     result_file_path += f"_param_{param_update[0][0]}_{param_update[0][1]}"
# alc.sol_history.to_pickle(result_file_path)


# %%
df = alc.sol_history
outcome_cost_fair_deaths_df = pd.DataFrame(df['outcome_cost_fair_deaths'].tolist(), columns=[f'cost_deaths_{i}' for i in range(len(df['outcome_cost_fair_deaths'].iloc[0]))])
outcome_disparity_fair_deaths_df = pd.DataFrame(df['outcome_disparity_fair_deaths'].tolist(), columns=[f'disparity_deaths_{i}' for i in range(len(df['outcome_disparity_fair_deaths'].iloc[0]))])
outcome_cost_fair_vacc_df = pd.DataFrame(df['outcome_cost_fair_vacc'].tolist(), columns=[f'cost_vacc_{i}' for i in range(len(df['outcome_cost_fair_vacc'].iloc[0]))])
outcome_disparity_fair_vacc_df = pd.DataFrame(df['outcome_disparity_fair_vacc'].tolist(), columns=[f'disparity_vacc_{i}' for i in range(len(df['outcome_disparity_fair_vacc'].iloc[0]))])
df = pd.concat([df, outcome_cost_fair_deaths_df, outcome_disparity_fair_deaths_df, outcome_cost_fair_vacc_df, outcome_disparity_fair_vacc_df], axis=1)
df.drop(['outcome_cost_fair_deaths', 'outcome_disparity_fair_deaths', 'outcome_cost_fair_vacc', 'outcome_disparity_fair_vacc'], axis=1, inplace=True)
df = df[[col for col in df.columns if col not in ['benefits_deaths', 'benefits_vacc']] + ['benefits_deaths', 'benefits_vacc']]

df = df.sort_values('cost_deaths_0', ascending = False)
# %%