#####################################################################
# A Module to calibrate parameters
#####################################################################

from vaccinemodel import *
from plot import *

def get_age_calib_val(model, X):
    X_sum = X.reshape(model.num_reg_group, model.num_age_group, int(X.size/(model.num_age_group*model.num_reg_group)))
    X_sum = np.transpose(X_sum,(1,0,2))
    X_sum = np.sum(X_sum, axis=1)
    X_interest = [X_sum[0],sum(X_sum[1:4]),X_sum[-1]]
    return(X_interest)

def get_calib_score_MSE(model, ret):
    calib_date = [i * 7 for i in range(int(len(model.t_c)/7))]
    [SA, IA, RA, DA, SP, IP, RP, DP] = np.array(np.transpose(np.reshape(np.array(ret), (len(model.t_f), model.num_group,model.num_comp))))[:,:,calib_date]
    I = IA+IP
    A = SA+IA+RA
    P = SP+IP+RP
    N = A+P
    D = DA+DP
    A_int_by_age = get_age_calib_val(model, A)
    N_int_by_age = get_age_calib_val(model, N)
    I_int = sum(get_age_calib_val(model, I))
    D_int = sum(get_age_calib_val(model, D))
    score_vacc = []
    for i in range(len(model.data_anti_prop)):
        values = A_int_by_age[i]/N_int_by_age[i]
        lower_errs = model.data_anti_prop[i]*model.vacc_rate_range[1]
        upper_errs = model.data_anti_prop[i]*model.vacc_rate_range[2]
        score_vacc.append(np.sum(np.where(values > upper_errs, values - upper_errs, np.maximum(lower_errs - 
values,np.zeros(len(values)))))/np.mean((1-model.data_anti_prop)*model.vacc_rate_range[0]))

    values = I_int/sum(N_int_by_age)
    lower_errs = model.data_inf_prop/model.inf_rate_range[1]
    upper_errs = model.data_inf_prop/model.inf_rate_range[2]
    score_inf = np.sum(np.where(values > upper_errs, values - upper_errs, np.maximum(lower_errs - 
values,np.zeros(len(values)))))/np.mean(model.data_inf_prop/model.inf_rate_range[0])

    values = D_int
    lower_errs = model.data_death*model.death_rate_range[1]
    upper_errs = model.data_death*model.death_rate_range[2]
    score_death = np.sum(np.where(values > upper_errs, values - upper_errs, np.maximum(lower_errs - 
values,np.zeros(len(values)))))/np.mean(model.data_death*model.death_rate_range[0])

    score = ([score_vacc, score_death, score_inf])
    return (score)

def vacc_calib_MSE(fips_num, samp_list, plot=False):
    model = VaccineModel(fips_num, init_param_list = samp_list, t_f= np.linspace(0, 58, 59), debug=False)
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
    x_name = ['overall_alpha','beta','prop_sus','O_m','rae','lam']
    x_dim = len(x_name)
    overall_alpha_range = [0.0001,0.0002] 
    beta_range = [1.5, 3.5]
    prop_sus_range = [0.4, 0.8]
    O_m_range = [1, 10]
    lam_range = [0.01, 0.05]
    rae_range = [150, 250]

    x_list = np.array([overall_alpha_range, beta_range, prop_sus_range, O_m_range, rae_range, lam_range])

    x_l = x_list[:,0]
    x_u = x_list[:,1]
    obj_function = vacc_calib_MSE
    param_update_lists = get_samp_by_lhs(n_trials, nrep, x_dim, x_u, x_l, x_name)
    results = (run_code_parallel(obj_function, fips_num, param_update_lists))
    data_list = []
    for param_list in param_update_lists:
        param_dict = dict(param_list)
        data_list.append(param_dict)
    df_final = pd.DataFrame(data_list, columns = x_name)

    col_list = ['vacc_score','dead_score','inf_score']
    score_df = pd.DataFrame(results, columns = col_list)
    expanded_df = pd.concat([score_df['vacc_score'].apply(pd.Series), score_df.drop('vacc_score', axis=1)], axis=1)
    expanded_df.columns = [f'vacc_score_{i}' for i in range(expanded_df.shape[1] - 2)] + list(score_df.columns[1:])
    expanded_df['final_score'] = np.sum(expanded_df, axis=1)
    expanded_df['vacc_score'] = expanded_df[[col for col in expanded_df.columns if 'vacc_score' in col]].sum(axis=1)
    df_final = pd.concat([df_final, expanded_df], axis=1)
    df_final = df_final.sort_values(by='final_score')
    return df_final

def get_best_result(filename, fips_num=53033, index_list=[0], t = np.linspace(0, 59, 60)):
    result_df = pd.read_csv(filename)
    x_name = ['overall_alpha','beta','prop_sus','O_m','rae','lam']
    best_df = result_df.iloc[index_list]
    rets = []
    # Show calibration results
    for i in (index_list):
        param_tuple_list = []
        for j in range(len(x_name)):
            x_set = (x_name[j], result_df.iloc[i][x_name[j]])
            param_tuple_list.append(x_set)
        model = VaccineModel(fips_num, init_param_list = param_tuple_list, t_f= t)
        ret = odeint(model.run_model, model.get_y0(), t)
        rets.append(ret)
    County_name = {53011: 'Clark County', 53033: 'King County', 53047: 'Okanogan County'}
    plot_results_with_calib(model, t, rets, error_bar=True, 
                            lw=1.5, filename=f'{fips_num}_calib', title=f"Best calibration result in {County_name.get(fips_num)}, WA")
    # best_df.to_csv(f'Data/{fips_num}_points_new.csv', index=False, columns = x_name)
    return model
    

#  %%
if __name__ == '__main__':
    n_trials = 100
    date = '0307'
    filename = f'Results/Calibration_Result/calib_result_n_{n_trials}_date_{date}.csv'
    nrep = 1234
    df_final = run_calibration(53033, n_trials, nrep)
    df_final.to_csv(filename)
    get_best_result(filename)
# %%
