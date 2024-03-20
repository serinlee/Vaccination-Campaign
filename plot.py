#####################################################################
# A Module that plots vaccinemodel results
#####################################################################

#%% Import settings
import seaborn as sns
import matplotlib.pyplot as plt
from vaccinemodel import VaccineModel
import numpy as np

plt.rcdefaults()

def plot_data_points(model):
    data_anti_prop = model.data_anti_prop
    death_rate_range = model.death_rate_range
    data_death = model.data_death
    data_date = [i * 7 for i in range(int(len(model.t_c)/7))]
    data_inf_prop = model.data_inf_prop
    inf_rate_range = model.inf_rate_range
    vacc_rate_range = model.vacc_rate_range
    c_list = ['r', 'b', 'g', 'grey', 'orange']
    anti_label = ['(Age 0-17)', '(Age 18-64)', '(Age 65+)']

    fig, axes = plt.subplots(1, 3, figsize=(5*3, 5))

    for i in range(len(data_anti_prop)):
        axes[0].errorbar(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0],
                            yerr=[100 * (1 - data_anti_prop[i]) * (vacc_rate_range[0] - vacc_rate_range[1]),
                                  100 * (1 - data_anti_prop[i]) * (vacc_rate_range[2] - vacc_rate_range[0])],
                            fmt='o', ecolor=c_list[i], color=c_list[i], capsize=5, markersize=3,label='Target data ' + anti_label[i])

    axes[2].errorbar(data_date, data_death * death_rate_range[0],
                        yerr=[data_death * (death_rate_range[0] - death_rate_range[1]),
                            data_death * (death_rate_range[2] - death_rate_range[0])],
                        fmt='o', capsize=5, markersize=3,label='Target data')
    
    axes[1].errorbar(data_date, 100 * data_inf_prop / inf_rate_range[0],
                        yerr=[100 * data_inf_prop * (1 / inf_rate_range[2] - 1 / inf_rate_range[0]),
                            100 * data_inf_prop * (1 / inf_rate_range[0] - 1 / inf_rate_range[1])],
                        fmt='o', capsize=5, markersize=3,label='Target data')

    for ax in axes:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=2)
        date = [i*7 for i in range(int(model.t_c[-1]/7))]
        date_label_full = [i+1 for i in range(int(model.t_c[-1]/7))]
        ax.set_xlabel("Week", fontsize=14)
        ax.set_xticks(date)
        ax.set_xticklabels(date_label_full[:len(date)], rotation=0)
        ax.tick_params(axis='both', which='both', labelsize=10)  # Adjust

    axes[0].set_title('Vaccinated population by age group', fontsize=14)
    axes[1].set_title('Infectious population', fontsize=14)
    axes[2].set_title('Deceased population', fontsize=14)
    
    axes[0].set_ylabel("Percentage (%)", fontsize=14)
    axes[1].set_ylabel("Percentage (%)", fontsize=14)
    axes[2].set_ylabel("Person", fontsize=14)

    axes[0].set_ylim([0, 100])
    axes[1].set_ylim([0.0, 0.05*100])
    plt.suptitle("Target Data in King County, WA", fontsize=18)
    fig.tight_layout()
    plt.savefig(f"Results/Plot/Target_{str(model.reg)}.png")
    plt.show()

def get_age_calib_val(model, X):
    X_sum = X.reshape(model.num_reg_group, model.num_age_group, int(X.size / (model.num_age_group * model.num_reg_group)))
    X_sum = np.transpose(X_sum, (1, 0, 2))
    X_sum = np.sum(X_sum, axis=1)
    X_interest = [X_sum[0], sum(X_sum[1:4]), X_sum[-1]]
    return X_interest

def plot_results_with_calib(model, t, ret_list, error_bar=False, lw=0.5, filename=None, title='', includedata=True):
    data_anti_prop = model.data_anti_prop
    death_rate_range = model.death_rate_range
    data_death = model.data_death
    data_inf_prop = model.data_inf_prop
    data_date = [i * 7 for i in range(int(len(model.t_c)/7))]
    inf_rate_range = model.inf_rate_range
    vacc_rate_range = model.vacc_rate_range
    num_group = model.num_group

    l_list = ['-',(0,(5,1)), '--',':','-.','dashed']*10
    label = ["0-17", "18-64", "65+"]
    color = ['r','b','g']

    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5))

    for idx, ret in enumerate(ret_list):
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), num_group, model.num_comp)))
        I = IA + IP
        A = SA + IA + RA
        P = SP + IP + RP
        N = A + P
        D = DA + DP
        # Get values to match with calibration data
        A_int_by_age = get_age_calib_val(model, A)
        N_int_by_age = get_age_calib_val(model, N)

        for i in range(len(data_anti_prop)):
            axes[0].plot(t, 100 * (1 - A_int_by_age[i] / N_int_by_age[i]),
                         label=f'Simulation Results (Age {label[i]})' if idx == 0 else "", 
                         color='grey', linestyle = l_list[2-i], linewidth=lw, alpha=1.0)
            if error_bar and idx == 0 and includedata:
                axes[0].errorbar(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0],
                                 yerr=[100 * (1 - data_anti_prop[i]) * (vacc_rate_range[0] - vacc_rate_range[1]),
                                       100 * (1 - data_anti_prop[i]) * (vacc_rate_range[2] - vacc_rate_range[0])],
                                 fmt='o', ecolor=color[i], color=color[i], capsize=5, markersize=3,
                                 label=f'Observed data (Age {label[i]})')
            elif not error_bar and idx == 0 and includedata:
                axes[0].plot(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0], color=color[i], marker='o', linestyle='', label=f'Observed data (Age {label[i]})')

        axes[1].plot(t, 100 * sum(I) / sum(N), color='grey', linewidth=lw, alpha=1.0, linestyle = l_list[idx],marker = '',markevery=60+idx,
                    label="Simulation Results" if idx == 0 else "")

        axes[2].plot(t, sum(D), color='grey', linewidth=lw, alpha=1.0, linestyle = l_list[idx], marker = '', markevery=60+idx,
                    label="Simulation Results" if idx == 0 else "")

        if error_bar and idx == 0 and includedata:
            axes[1].errorbar(data_date, 100 * data_inf_prop / inf_rate_range[0],
                             yerr=[100 * data_inf_prop * (1 / inf_rate_range[2] - 1 / inf_rate_range[0]),
                                   100 * data_inf_prop * (1 / inf_rate_range[0] - 1 / inf_rate_range[1])],
                             fmt='o', capsize=5, markersize=3, label='Observed data')
            axes[2].errorbar(data_date, data_death * death_rate_range[0],
                             yerr=[data_death * (death_rate_range[0] - death_rate_range[1]),
                                   data_death * (death_rate_range[2] - death_rate_range[0])],
                             fmt='o', capsize=5, markersize=3, label='Observed data')
            
        elif not error_bar and idx == 0 and includedata:
            axes[1].plot(data_date, 100 * data_inf_prop / inf_rate_range[0], marker='o', linestyle='')
            axes[2].plot(data_date, data_death * death_rate_range[0], marker='o', linestyle='')

    # Set legend and xticks by simulation period
    for ax in axes:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=2, fontsize=11)    
        if t[-1] <= 100:
            date = [i*7 for i in range(int(t[-1]/7))]
            date_label_full = [i+1 for i in range(int(t[-1]/7))]
            ax.set_xlabel("Week", fontsize=14)
            ax.set_xticks(date)
            ax.set_xticklabels(date_label_full[:len(date)], rotation=0)
            ax.tick_params(axis='both', which='both', labelsize=10)
        else:
            date = [i*30.5 for i in range(int(t[-1]/30.5))]
            date_label_full = [i+1 for i in range(int(t[-1]/30.5))]
            ax.set_xlabel("Month", fontsize=14)
            ax.set_xticks(date)
            ax.set_xticklabels(date_label_full[:len(date)], rotation=0)
            ax.tick_params(axis='both', which='both', labelsize=10)

    axes[0].set_title('Vaccinated population by age group', fontsize=14)
    axes[1].set_title('Infectious population', fontsize=14)
    axes[2].set_title('Deceased population', fontsize=14)
    
    axes[0].set_ylabel("Percentage (%)", fontsize=14)
    axes[1].set_ylabel("Percentage (%)", fontsize=14)
    axes[2].set_ylabel("Person", fontsize=14)

    axes[0].set_ylim([0, 100])

    plt.suptitle(title, fontsize=18)
    fig.tight_layout()
    if filename is None:
        plt.show()
    else: plt.savefig(f"Results/Plot/{filename}.png",  transparent=False, dpi=300, bbox_inches='tight')

def plot_results(model, t, ret_list, to_plot='vacc', lw=0.5, filename=None, title='', label=None):
    if to_plot not in ['vacc', 'vacc_all','inf','deceased']:
        raise ValueError("Current plot can show vaccination by age ('vacc'), overall vaccination rate('vacc_all'), infection rate('inf'), deceased population ('deceased') only. Use these keywords to plot or customize plot_results function")
        
    c_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
              '#bcbd22', '#17becf', '#1a9850', '#66a61e', '#a6cee3', '#fdbf6f', '#fb9a99', '#e31a1c', 
              '#fb9a99', '#33a02c', '#b2df8a', '#a6cee3']*300
    num_group = model.num_group

    fig, ax = plt.subplots(figsize=(6, 4))

    for idx, ret in enumerate(ret_list):
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), num_group, model.num_comp)))
        I = IA + IP
        A = SA + IA + RA
        P = SP + IP + RP
        N = A + P
        D = DA + DP
        A_int_by_age = get_age_calib_val(model, A)
        N_int_by_age = get_age_calib_val(model, N)

        if to_plot == 'vacc':
            ax.set_title('Vaccinated population', fontsize=14)
            ax.set_ylabel('Percentage (%)', fontsize=14)
            for i in range(len(A_int_by_age)):
                ax.plot(t, 100 * (1 - A_int_by_age[i] / N_int_by_age[i]),
                        color=c_list[idx//len(A_int_by_age)], label=label[idx] if i==0 and label is not None else '', 
                        linewidth=lw, alpha=1.0, linestyle='-', marker='')

        elif to_plot == 'vacc_all':
            ax.set_title('Vaccinated population', fontsize=14)
            ax.set_ylabel('Percentage (%)', fontsize=14)
            ax.plot(t, 100 * (1 - sum(A_int_by_age) / sum(N_int_by_age)), color=c_list[idx], label=label[idx] if label is not None else '', linewidth=lw, alpha=1.0, linestyle='-', marker='')

        elif to_plot == 'inf':
            ax.set_title('Infectious population', fontsize=14)
            ax.set_ylabel("Percentage (%)", fontsize=14)
            ax.plot(t, 100 * sum(I) / sum(N), color=c_list[idx], label=label[idx] if label is not None else '', linewidth=lw, alpha=1.0, linestyle='-', marker='')
            
        elif to_plot == 'deceased':
            ax.set_title('Deceased population', fontsize=14)
            ax.set_ylabel("Person", fontsize=14)
            ax.plot(t, sum(D), color=c_list[idx], label=label[idx] if label is not None else '',linewidth=lw, alpha=1.0, linestyle='-', marker='')
            

        ax.set_xlabel("Month")
        if t[-1] <= model.t_c[-1]:
            date = [i*7 for i in range(int(t[-1]/7))]
            date_label_full = [i+1 for i in range(int(t[-1]/7))]
            ax.set_xlabel("Week", fontsize=14)
            ax.set_xticks(date)
            ax.set_xticklabels(date_label_full[:len(date)], rotation=0)
            ax.tick_params(axis='both', which='both', labelsize=10)
        else:
            date = [i*30.5 for i in range(int(t[-1]/30.5))]
            date_label_full = [i+1 for i in range(int(t[-1]/30.5))]
            ax.set_xlabel("Month", fontsize=14)
            ax.set_xticks(date)
            ax.set_xticklabels(date_label_full[:len(date)], rotation=0)
            ax.tick_params(axis='both', which='both', labelsize=10)
    
    if label is not None:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=2)
        
    fig.tight_layout()
    if filename is None:
        plt.show()
    else: 
        plt.savefig(f"Results/Plot/{filename}.png")

# %%
if __name__ == '__main__':
    from scipy.integrate import odeint
    model = VaccineModel()
    ret = odeint(model.run_model, model.get_y0(), model.t_f)
    model2 = VaccineModel(init_param_list = [('beta', 3)]) # Model 2 changes the beta value
    ret2 = odeint(model2.run_model, model2.get_y0(), model2.t_f)
    plot_results_with_calib(model, model.t_f, [ret, ret2])
    plot_results(model, model.t_f, [ret, ret2],'vacc_all', label=['Model', 'Model 2'])

# %%
