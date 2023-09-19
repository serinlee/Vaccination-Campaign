
#%%
from vaccinemodel import *
from plot import *

def get_age_calib_val(model, X):
    X_sum = X.reshape(model.num_reg_group, model.num_age_group, int(X.size/(model.num_age_group*model.num_reg_group)))
    X_sum = np.transpose(X_sum,(1,0,2))
    X_sum = np.sum(X_sum, axis=1)
    X_interest = [X_sum[0],sum(X_sum[1:4]),X_sum[-1]]
    return(X_interest)

def get_calib_score_MSE(model, ret):
    [SA, IA, RA, DA, SP, IP, RP, DP] = np.array(np.transpose(np.reshape(np.array(ret), (len(model.t_f), model.num_group,model.num_comp))))[:,:,model.data_date]
    I = IA+IP
    A = SA+IA+RA
    P = SP+IP+RP
    N = A+P
    D = DA+DP
    A_int_by_age = get_age_calib_val(model, A)
    N_int_by_age = get_age_calib_val(model, N)
    I_int = sum(get_age_calib_val(model, I))
    D_int = sum(get_age_calib_val(model, D))
    score_vacc = 0
    for i in range(len(model.data_anti_prop)):
        values = A_int_by_age[i]/N_int_by_age[i]
        lower_errs = model.data_anti_prop[i]-0.01
        upper_errs = model.data_anti_prop[i]+0.01
        score_vacc += np.sum(np.where(values > upper_errs, values - upper_errs, np.maximum(lower_errs - values,np.zeros(len(values)))))/np.mean((1-model.data_anti_prop)*model.vacc_rate_range[0])

    values = I_int/sum(N_int_by_age)
    lower_errs = model.data_inf_prop/model.inf_rate_range[1]
    upper_errs = model.data_inf_prop/model.inf_rate_range[2]
    score_inf = np.sum(np.where(values > upper_errs, values - upper_errs, np.maximum(lower_errs - values,np.zeros(len(values)))))/np.mean(model.data_inf_prop/model.inf_rate_range[0])

    values = D_int
    lower_errs = model.data_death*model.death_rate_range[1]
    upper_errs = model.data_death*model.death_rate_range[2]
    score_death = np.sum(np.where(values > upper_errs, values - upper_errs, np.maximum(lower_errs - values,np.zeros(len(values)))))/np.mean(model.data_death*model.death_rate_range[0])

    score = np.array([score_vacc, score_death, score_inf])
    return (score)
    # print(score)

    return np.sum(score)

def vacc_calib_MSE(fips_num, samp_list, plot=False):
    param_update_list = samp_list
    model = VaccineModel(fips_num, param_update_list, param_update_mode = 'Calibration', t_f= np.linspace(0, 58, 59))
    ret = odeint(model.run_model, model.get_y0(), model.t_f)
    score = get_calib_score_MSE(model, ret)
    if plot: plot_results_with_calib(model, model.t_f, [ret],error_bar=True, filename=f'Calib_{fips_num}')
    return(score)

def get_samp_by_lhs(n_trials, nrep, x_dim, x_u, x_l, x_name):
    from pyDOE import lhs
    import numpy as np
    np.random.seed(nrep)
    sample_lhs = lhs(x_dim, n_trials)
    x = x_l + sample_lhs*(x_u-x_l)
    param_update_lists = []
    for i in range(n_trials):
        param_tuple_list = []
        for j in range(x_dim):
            param_tuple_list.append((x_name[j], x[i, j]))
        param_update_lists.append(param_tuple_list)
    return param_update_lists


def run_code_parallel(obj_function, fips_num, param_update_lists):
    import multiprocess as mp
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(obj_function, [(fips_num, param_list) for param_list in param_update_lists])
    return results 

def run_calibration(fips_num, n_trials, nrep):
    x_dim = 13
    x_name = ['overall_alpha','beta','prop_sus','O_m','p1','p2','p3','p4','p5','rae','k_R','k_E','lam']
    overall_alpha_range = [0.0001,0.0003] 
    beta_range = [1.0, 3.0] # default: 2.0
    prop_sus_range = [0.2, 0.8] # default: 0.6
    O_m_range = [0.1, 3] # default: 0.5
    p_range = [0.01, 0.99] # default: 0.5
    rae_range = [150, 250] # default: 200
    k_R_range = [0.1*1e6, 1.0*1e6] # default: 0.5 k_R = 1/(k_R_range*overall_alpha)
    k_E_range = [0.1, 1.5] # default: 0.5 
    lam_range = [0.01, 0.1] # default: 0.25
    x_list = np.array([overall_alpha_range,beta_range,prop_sus_range,O_m_range,
                    p_range,p_range,p_range,p_range,p_range,rae_range,k_R_range,k_E_range,lam_range])
    x_l = x_list[:,0]
    x_u = x_list[:,1]
    obj_function = vacc_calib_MSE
    func_name = f"calib_{fips_num}"
    param_update_lists = get_samp_by_lhs(n_trials, nrep, x_dim, x_u, x_l, x_name)
    results = np.array(run_code_parallel(obj_function, fips_num, param_update_lists))
    data_list = []
    for param_list in param_update_lists:
        param_dict = dict(param_list)
        data_list.append(param_dict)
    df_final = pd.DataFrame(data_list, columns = x_name)
    col_list = ['vacc_score','dead_score','inf_score']
    for i in range(len(col_list)):
        df_final.insert(0, col_list[i], results[:,i])  # None can be replaced with your desired default values
    df_final.insert(0, 'final_score', np.sum(results, axis=1)) 
    df_final = df_final.sort_values(by='final_score')
    return df_final
# %%
import argparse
n_trials = 500
nrep = 1234
# parser = argparse.ArgumentParser()
# parser.add_argument('-f', '--fips_num', type=int, help='FIPS number (region)')
# args = parser.parse_args()
df_final = run_calibration(args.fips_num, n_trials, nrep)
# df_final.to_csv(f'Calibration_Result/{args.fips_num}_calib_result_n_{n_trials}_nrep_{nrep}_test.csv')

# %%
def get_best_result(fips_num, n_trials, nrep, top_n, sort_by):
    result_df = pd.read_csv(f'Calibration_Result/{fips_num}_calib_result_n_{n_trials}_nrep_{nrep}_test.csv')
    x_name = ['overall_alpha','beta','prop_sus','O_m','p1','p2','p3','p4','p5','rae','k_R','k_E','lam']
    result_df = result_df.sort_values(sort_by)
    best_df = result_df.iloc[:top_n]
    best_df.to_csv(f'Data/{fips_num}_points.csv', index=False, columns = x_name)
    rets = []
    for i in range(top_n):
        param_tuple_list = []
        for j in range(13):
            param_tuple_list.append((x_name[j], result_df.iloc[i][x_name[j]]))
        model = VaccineModel(fips_num, param_tuple_list, param_update_mode = 'Calibration', t_f= np.linspace(0, 58, 59))
        rets.append(odeint(model.run_model, model.get_y0(), model.t_f))
    plot_results_with_calib(model, model.t_f, rets, error_bar=True, filename=f'Calib_{fips_num}_test')


# %%
