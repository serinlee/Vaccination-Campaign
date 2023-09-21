#%% import
import seaborn as sns
import matplotlib.pyplot as plt
from vaccinemodel import VaccineModel
import numpy as np

def plot_data_points(model):
    data_date = model.data_date
    data_anti_prop = model.data_anti_prop
    death_rate_range = model.death_rate_range
    data_death = model.data_death
    date_label = model.date_label
    data_inf_prop = model.data_inf_prop
    inf_rate_range = model.inf_rate_range
    vacc_rate_range = model.vacc_rate_range

    fig, axes = plt.subplots(1, 3, figsize=(5*3, 5))
    c_list = ['r', 'b', 'g', 'grey', 'orange']
    anti_label = ['(Age 0-17)', '(Age 18-64)', '(Age 65+)']
    # Access the data_anti_prop attribute from the VaccineModel instance
    for i in range(len(data_anti_prop)):
        axes[0].plot(data_date, 100*(1-data_anti_prop[i])*vacc_rate_range[0],
                     color=c_list[i], marker='o', linestyle='', label='Target data ' + anti_label[i])
    axes[1].plot(data_date,100*data_inf_prop/inf_rate_range[0], marker='o',linestyle='',label='Target data')
    axes[2].plot(data_date, data_death*death_rate_range[0], marker='o',linestyle='',label='Target data')
    
    for ax in axes:
        ax.set_xlabel("Days")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=2)
        ax.set_xticks(data_date)
        ax.set_xticklabels(date_label, rotation=40)

    # axes[2].set_title('Dead population')
    # axes[0].set_ylabel('Vaccinated population (%)')
    # axes[1].set_ylabel("Infectious population (%)")
    
    axes[0].set_ylabel("Vaccinated population (%)")
    axes[1].set_ylabel("Infectious population (%)")
    axes[2].set_ylabel("Dead population")
    axes[0].set_ylim([0, 100])
    axes[1].set_ylim([0.0, 0.05*100])
    plt.suptitle("Target Data by Health Outcomes")
    fig.tight_layout()
    plt.savefig("Plot/Target.png")
    plt.show()

def get_age_calib_val(model, X):
    X_sum = X.reshape(model.num_reg_group, model.num_age_group, int(X.size / (model.num_age_group * model.num_reg_group)))
    X_sum = np.transpose(X_sum, (1, 0, 2))
    X_sum = np.sum(X_sum, axis=1)
    X_interest = [X_sum[0], sum(X_sum[1:4]), X_sum[-1]]
    return X_interest

def plot_results_with_calib(model, t, ret_list, error_bar=False, lw=0.5, filename=None):
    data_date = model.data_date
    data_anti_prop = model.data_anti_prop
    death_rate_range = model.death_rate_range
    data_death = model.data_death
    data_inf_prop = model.data_inf_prop
    inf_rate_range = model.inf_rate_range
    vacc_rate_range = model.vacc_rate_range
    num_group = model.num_group

    c_list = ['r', 'b', 'g', 'grey', 'orange']
    label = ["0-17", "18-64", "65+"]
    data_date_full = t
    date_label_full = ['Jan/23', 'Feb/23', 'Mar/23', 'Apr/23', 'May/23', 'Jun/23', 'Jul/23', 'Aug/23', 'Sep/23', 'Oct/23', 'Nov/23', 'Dec/23']

    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 6))

    for idx, ret in enumerate(ret_list):
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), num_group, model.num_comp)))
        I = IA + IP
        A = SA + IA + RA
        P = SP + IP + RP
        N = A + P
        D = DA + DP
        A_int_by_age = get_age_calib_val(model, A)
        N_int_by_age = get_age_calib_val(model, N)
        for i in range(len(data_anti_prop)):
            axes[0].plot(t, 100 * (1 - A_int_by_age[i] / N_int_by_age[i]),
                         label=f'Simulated-Age {label[i]}' if idx == 0 else "", linewidth=lw, color=c_list[i], alpha=0.3)
            if error_bar and idx == 0:
                axes[0].errorbar(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0],
                                 yerr=[np.ones(len(data_date)), np.ones(len(data_date))],
                                #  yerr=[100 * (1 - data_anti_prop[i]) * (vacc_rate_range[0] - vacc_rate_range[1]),
                                #        100 * (1 - data_anti_prop[i]) * (vacc_rate_range[2] - vacc_rate_range[0])],
                                 fmt='o', ecolor=c_list[i], color=c_list[i], capsize=5, markersize=3,
                                 label='Observed data')
            elif not error_bar and idx == 0:
                axes[0].plot(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0], color=c_list[i], marker='o',
                             linestyle='', label='Observed data')

        axes[1].plot(t, sum(D), color='grey', linewidth=lw, alpha=0.5,
                     label="Simulation Results" if idx == 0 else "")
        axes[2].plot(t, 100 * sum(I) / sum(N), color='grey', linewidth=lw, alpha=0.5,
                     label="Simulation Results" if idx == 0 else "")

        if error_bar and idx == 0:
            axes[1].errorbar(data_date, data_death * death_rate_range[0],
                             yerr=[data_death * (death_rate_range[0] - death_rate_range[1]),
                                   data_death * (death_rate_range[2] - death_rate_range[0])],
                             fmt='o', capsize=5, markersize=3, label='Observed data')
            axes[2].errorbar(data_date, 100 * data_inf_prop / inf_rate_range[0],
                             yerr=[100 * data_inf_prop * (1 / inf_rate_range[2] - 1 / inf_rate_range[0]),
                                   100 * data_inf_prop * (1 / inf_rate_range[0] - 1 / inf_rate_range[1])],
                             fmt='o', capsize=5, markersize=3, label='Observed data')
        elif not error_bar and idx == 0:
            axes[1].plot(data_date, data_death * death_rate_range[0], marker='o', linestyle='',
                         label='Observed data')
            axes[2].plot(data_date, 100 * data_inf_prop / inf_rate_range[0], marker='o', linestyle='',
                         label='Estimated data')

    # for ax in axes:
    #     ax.set_xlabel("Month")
    #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=2)
    #     ax.set_xticks(data_date_full)
    #     ax.set_xticklabels(date_label_full, rotation=40)

    axes[0].set_title('Vaccinated population')
    axes[2].set_title('Infectious population')
    axes[1].set_title('Dead population')
    axes[0].set_ylabel('Percentage (%)')
    axes[2].set_ylabel("Percentage (%)")
    axes[1].set_ylabel("Person")
    # axes[0].set_ylim([50, 60])
    # axes[2].set_ylim([0.0, 0.05 * 100])
    plt.suptitle(f"Model Validation to {model.reg} County")
    fig.tight_layout()
    if filename is None:
        plt.show()
    else: plt.savefig(f"Plot/{filename}.png")

def plot_deaths_with_calib(model, t, ret_list, error_bar=False, lw=0.5, filename=None):
    c_list = ['r', 'b', 'g', 'grey', 'orange']
    label = ["Best policy", "Worst policy", "No policy"]
    # data_date_full = np.linspace(0, 365, 12)
    # date_label_full = ['Jan/23', 'Feb/23', 'Mar/23', 'Apr/23', 'May/23', 'Jun/23', 'Jul/23', 'Aug/23', 'Sep/23', 'Oct/23', 'Nov/23', 'Dec/23']

    fig, ax = plt.subplots(figsize=(5, 4))
    for idx, ret in enumerate(ret_list):
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), model.num_group, model.num_comp)))
        I = IA + IP
        A = SA + IA + RA
        P = SP + IP + RP
        N = A + P
        D = DA + DP

        ax.plot(t, sum(D), color=c_list[idx], linewidth=lw, alpha=0.5, label=label[idx])
        # if idx == len(ret_list)-1:
        #     ax.plot(data_date, data_death * death_rate_range[0], marker='o', linestyle='', label='Observed data')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=3)
        # ax.set_xlabel("Month")
        # ax.set_xticks(data_date_full)
        # ax.set_xticklabels(date_label_full, rotation=40)

    ax.set_ylabel("Person")
    # ax.set_ylim(bottom = 60)
    fig.tight_layout()
    if filename is None:
        plt.show()
    else: plt.savefig(f"Plot/{filename}.png")


def plot_results(model, t, ret_list, lw=0.5):
    data_anti_prop = model.data_anti_prop
    num_group = model.num_group

    c_list = ['r', 'b', 'g', 'grey', 'orange']
    label = ["0-17", "18-64", "65+"]
    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 6))
    count = 0
    for ret in ret_list:
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), num_group, model.num_comp)))
        I = IA + IP
        A = SA + IA + RA
        P = SP + IP + RP
        N = A + P
        D = DA + DP
        A_int_by_age = get_age_calib_val(model, A)
        N_int_by_age = get_age_calib_val(model, N)
        for i in range(len(data_anti_prop)):
            axes[0].plot(t, 100 * (1 - A_int_by_age[i] / N_int_by_age[i]),
                        #  label='Simulated-Age ' + str(label[i]),
                         label = 'Alloc '+str(count),
                        linewidth=lw, color=c_list[i], alpha=0.3)
        axes[1].plot(t, sum(D), color='grey', linewidth=lw, alpha=0.5, 
                    #  label="Simulation Results"
                     label = 'Alloc '+str(count)
                     )
        axes[2].plot(t, 100 * sum(I) / sum(N), color='grey', linewidth=lw, alpha=0.5, 
                    #  label="Simulation Results",
                     label = 'Alloc '+str(count)
                     )
        count+=1
    for ax in axes:
        ax.set_xlabel("Days")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=2)
        # ax.set_xticks(data_date)
        # ax.set_xticklabels(date_label, rotation=40)

    axes[0].set_title('Vaccinated population')
    axes[2].set_title('Infectious population')
    axes[1].set_title('Dead population')
    axes[0].set_ylabel('Percentage (%)')
    axes[2].set_ylabel("Percentage (%)")
    axes[1].set_ylabel("Person")
    axes[0].set_ylim([0, 100])
    axes[2].set_ylim([0.0, 0.05 * 100])
    fig.tight_layout()
    plt.show()

# %%
# from scipy.integrate import odeint
# model = VaccineModel()
# y0 = model.get_y0()
# t = model.t_c
# ret = odeint(model.run_model, y0, t)
# plot_results_with_calib(t, [ret])
# plot_results(t, [ret])
# %%
