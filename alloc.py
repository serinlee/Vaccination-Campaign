#####################################################################
# A Module that solves allocation problem with set of allocation lists or optimization algorithm
# 1. Generates list of allocation strategies 'alg' - two-stage coarse grid ('reg_age'), latin hypercube sampling('lhs'), allocation to single group ('vertex'),
#    or using global optimization algorithms such as genetic algorithm ('ga') and trust-region ('trust_constr') method.
# 2. Measures the outcomes given health benefits (base: 'tot_benefits_vacc') that maximizes the overall vaccination uptake increase compared to no campaign
# 3. Saves the result as csv and pkl file
#####################################################################

#%%
from vaccinemodel import *
import warnings

class Alloc:
    def __init__(self, fips_num=53033, alloc=None, init_param_list = [], point_index=0, obj_type='', alg='', B=20000, num_alloc = 1, n_iter=0, param_update_list=[], debug=False):
        # set initial parameter sets based on best calibration parameter set
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
        self.sol_history_columns = ['fips', 'point_index', 'n_iter', 'alg', 'alloc', 'obj', 
                                    'tot_benefits_vacc', 'min_benefits_vacc_per_pop', 'tot_benefits_deaths','min_benefits_deaths_per_pop',
                                    'benefits_vacc_per_pop','benefits_deaths_per_pop']
        self.sol_history = pd.DataFrame(columns =  self.sol_history_columns)
        self.alloc = np.zeros(self.model.num_group)
        if alloc is not None:
            self.alloc = alloc

        self.B = B # total budget
        self.num_alloc = num_alloc # number of unit that divide the budget (if num_alloc=20, assign total of 20 units (5%) of the budget)
        self.n_iter = n_iter # used for random seeding
        self.param_update_list = param_update_list
        self.ret_list = [] # additional outcomes
        self.eta_list = []
        self.all_eta_list = []
        self.debug = debug
        
    # get outcomes without any policy
    def get_no_policy(self, save_result = False):
        self.model = VaccineModel(self.fips_num, init_param_list = self.init_param_list, 
                                  param_update_list=self.param_update_list, debug = self.debug)
        t = self.model.t_f
        self.model.update_param("U", np.zeros(self.model.num_group)) # No allocation given in this case
        ret = odeint(self.model.run_model, self.model.get_y0(), t) # run the model
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), self.model.num_group, self.model.num_comp)))
        self.no_policy_outcome_deaths = DA[:, -1] + DP[:, -1]
        self.no_policy_outcome_vacc = SP[:,-1] + IP[:,-1] + RP[:,-1]
        if save_result:
            return ret
 
    def calculate_outcome_metrics(self, ret, alloc):    
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(self.model.t_f), self.model.num_group, self.model.num_comp)))
        self.ret_list.append(ret)
        self.eta_list.append(self.model.mean_eta)
        self.all_eta_list.append(self.model.eta_all)
        N = SA[:,-1]+IA[:,-1]+RA[:,-1]+IP[:,-1]+RP[:,-1]+DP[:,-1]
        
        benefits_deaths = (self.no_policy_outcome_deaths -  (DA[:, -1] + DP[:, -1]))
        benefits_vacc = ((SP[:,-1] + IP[:,-1] + RP[:,-1]) - self.no_policy_outcome_vacc).round(4)

        tot_benefits_deaths = sum(benefits_deaths)
        tot_benefits_vacc = sum(benefits_vacc)

        benefits_deaths_per_pop = benefits_deaths/N
        benefits_vacc_per_pop = benefits_vacc/N
        
        min_benefits_deaths_per_pop = np.min(benefits_deaths_per_pop)
        min_benefits_vacc_per_pop = np.min(benefits_vacc_per_pop)

        return (tot_benefits_vacc, min_benefits_vacc_per_pop, tot_benefits_deaths, min_benefits_deaths_per_pop, benefits_vacc_per_pop, benefits_deaths_per_pop)

    def run_alloc(self, alloc):
        if self.alg in ['trust_constr' , 'ga']:
            alloc *= self.B
        self.model = VaccineModel(self.fips_num, init_param_list = self.init_param_list, 
                                  param_update_list=self.param_update_list, debug = self.debug)
        t = self.model.t_f
        self.model.update_param("U", alloc)
        ret = odeint(self.model.run_model, self.model.get_y0(), t) # run model
        
        (tot_benefits_vacc, min_benefits_vacc_per_pop, tot_benefits_deaths, min_benefits_deaths_per_pop, benefits_vacc_per_pop, benefits_deaths_per_pop) = self.calculate_outcome_metrics(ret, alloc)

        new_row = pd.DataFrame([[self.fips_num, self.point_index, self.n_iter, self.alg, alloc, self.obj_type, 
                                 tot_benefits_vacc, min_benefits_vacc_per_pop, tot_benefits_deaths, min_benefits_deaths_per_pop, benefits_vacc_per_pop, benefits_deaths_per_pop]], columns=self.sol_history_columns)
        if self.alg not in ['ga','trust_constr']:
            return new_row
        else:
            self.sol_history = pd.concat([self.sol_history, new_row], ignore_index=True)
            value_mapping = {
                'tot_benefits_vacc': -tot_benefits_vacc,
                'min_benefits_vacc_per_pop': -min_benefits_vacc_per_pop,
                'tot_benefits_deaths': -tot_benefits_deaths,
                'min_benefits_deaths_per_pop': -min_benefits_deaths_per_pop,
            }
            if self.debug:
                print(value_mapping.get(self.obj_type, 0) , ' alloc ', alloc)
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
    
    def get_vertex_alloc(self):
        alloc_list = []
        for i in range(self.model.num_group):
            alloc = np.zeros(self.model.num_group)
            alloc[i] = self.B
            alloc_list.append(alloc)
        return alloc_list
    
    def get_top_region_vertex_alloc(self):
        alloc_list = []
        for i in range(self.model.num_group):
            alloc = np.zeros(self.model.num_group)
            alloc[i] = self.B
            alloc_list.append(alloc)
        return alloc_list[15:20]

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

        tot_alloc = sum(top_reg.iloc[0].alloc)

        alloc_list = []
        for i in range(len(top_reg)):
            for j in range(len(top_age)):
                alloc = np.array(top_reg.iloc[i].alloc)/tot_alloc * np.array(top_age.iloc[j].alloc)/tot_alloc
                if sum(alloc) >0: 
                    alloc = alloc/sum(alloc)*sum(top_reg.iloc[i].alloc)
                    alloc_list.append(alloc)

        # Add vertex points as well
        alloc_list += self.get_vertex_alloc()
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
            alloc_list = self.get_vertex_alloc()
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
        bounds = Bounds([0]*self.model.num_group, [1]*self.model.num_group)
        constraint = LinearConstraint(np.ones((1, self.model.num_group)), [1], [1])
        self.get_no_policy()
        res = differential_evolution(self.run_alloc, bounds, constraints=constraint, 
                                     x0 = self.model.N_by_group / np.sum(self.model.N_by_group),
                                     seed=1234+n_iter, workers=1, maxiter=10000, popsize=15, tol= 10)
        return (res)
    
    def run_trust_constr(self, n_iter=0):
        import scipy
        from scipy.optimize import Bounds
        from scipy.optimize import LinearConstraint
        from scipy.optimize import minimize
        results = []
        np.random.seed(1234 + n_iter)

        lb = [0] * self.model.num_group + [1]
        ub = [1] * self.model.num_group + [1]
        A = np.concatenate((np.eye(self.model.num_group), np.ones((1, self.model.num_group))), axis=0)

        x0 = self.model.N_by_group / np.sum(self.model.N_by_group) # normalize the distribution so that sum is 1

        constraint = LinearConstraint(A, lb=lb, ub=ub, keep_feasible=True)
        selected_eval_number = 2000

        def callback(xk, state):
            current_eval_number = state.nfev
            if current_eval_number >= selected_eval_number:
                return True
            return False
        
        res = minimize(self.run_alloc, x0, method='trust-constr', constraints=constraint, 
                       options={'verbose': 2, 'disp': True, 'maxiter': 2000,'initial_constr_penalty':1e7},
                       callback=callback)
        return(res)
    
    def run_code(self, parallel=False, alloc_list=None, save_result=False):
        self.get_no_policy(save_result)
        if self.alg=='ga':
            self.run_ga()
        if self.alg=='trust_constr':
            self.run_trust_constr()
            self.sol_history = self.sol_history.reset_index()
        else:
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

def organize_df(df, sort_by='tot_benefits_vacc'):
    list_outcome = ['alloc', 'benefits_vacc_per_pop', 'benefits_deaths_per_pop']
    for column in list_outcome:
        if column in df.columns and isinstance(df[column].iloc[0], (list, np.ndarray)):
            new_columns =  [f'{column}_{i}' for i in range(len(df[column].iloc[0]))]
            new_df = pd.DataFrame(df[column].tolist(), columns=new_columns)
            # df.drop(column, axis=1, inplace=True)
            if len(new_df)>1:
                df = pd.concat([df, new_df], axis=1)
    return df

#%% User can specify point_index, obj_type, alg, B, num_alloc, n_iter
if __name__ == '__main__':
    # When running from terminal
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fips_num', type=int, help='FIPS number (region)')
    parser.add_argument('-p', '--point_index', type=int, help='Point index (calibration)') # used when there are more than 1 parameter sets.
    parser.add_argument('-B', '--B', type=int, help='Total Budget')
    args = parser.parse_args()

    alc = Alloc(fips_num = args.fips_num, obj_type = 'all', alg='reg_age', B=args.B, num_alloc = 20, point_index = args.point_index)

    # When running directly
    # alc = Alloc(fips_num = 53033, obj_type = 'all', alg='reg_age', B=20000, num_alloc = 1, point_index = 0)

    date = '1030'
    alloc_test = alc.get_alloc_list()
    alc.run_code(parallel=True, alloc_list = alloc_test)

    # save the result
    result_file_path = f'Result/{alc.fips_num}_obj_{alc.obj_type}_alg_{alc.alg}_B_{alc.B}_point_{alc.point_index}_date_{date}'
    alc.sol_history.to_pickle(result_file_path+'.pkl')
    with np.printoptions(linewidth=1000):
        (alc.sol_history).to_csv(result_file_path+'.csv') 

# %%
