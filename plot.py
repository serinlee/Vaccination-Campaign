#%% import
import seaborn as sns
import matplotlib.pyplot as plt
from vaccinemodel import VaccineModel
import numpy as np
plt.rcdefaults()

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
        # axes[0].plot(data_date, 100*(1-data_anti_prop[i])*vacc_rate_range[0],
        #              color=c_list[i], marker='o', linestyle='', label='Target data ' + anti_label[i])
        axes[0].errorbar(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0],
                            # yerr=[np.ones(len(data_date)), np.ones(len(data_date))],
                            yerr=[100 * (1 - data_anti_prop[i]) * (vacc_rate_range[0] - vacc_rate_range[1]),
                                  100 * (1 - data_anti_prop[i]) * (vacc_rate_range[2] - vacc_rate_range[0])],
                            fmt='o', ecolor=c_list[i], color=c_list[i], capsize=5, markersize=3,label='Target data ' + anti_label[i])

    # axes[1].plot(data_date,100*data_inf_prop/inf_rate_range[0], marker='o',linestyle='',label='Target data')
    # axes[2].plot(data_date, data_death*death_rate_range[0], marker='o',linestyle='',label='Target data')

    axes[2].errorbar(data_date, data_death * death_rate_range[0],
                        yerr=[data_death * (death_rate_range[0] - death_rate_range[1]),
                            data_death * (death_rate_range[2] - death_rate_range[0])],
                        fmt='o', capsize=5, markersize=3,label='Target data')
    # label='Observed data')
    axes[1].errorbar(data_date, 100 * data_inf_prop / inf_rate_range[0],
                        yerr=[100 * data_inf_prop * (1 / inf_rate_range[2] - 1 / inf_rate_range[0]),
                            100 * data_inf_prop * (1 / inf_rate_range[0] - 1 / inf_rate_range[1])],
                        fmt='o', capsize=5, markersize=3,label='Target data')

    for ax in axes:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=2)
        # ax.set_xlabel("Days")
        # ax.set_xticks(data_date)
        # ax.set_xticklabels(date_label, rotation=40)
        date = [i*7 for i in range(int(model.t_c[-1]/7))]
        date_label_full = [i+1 for i in range(int(model.t_c[-1]/7))]
        ax.set_xlabel("Week", fontsize=14)
        ax.set_xticks(date)
        ax.set_xticklabels(date_label_full[:len(date)], rotation=0)
        ax.tick_params(axis='both', which='both', labelsize=10)  # Adjust

    axes[0].set_title('Vaccinated population by age group', fontsize=14)
    axes[1].set_title('Infectious population', fontsize=14)
    axes[2].set_title('Dead population', fontsize=14)
    
    axes[0].set_ylabel("Percentage (%)", fontsize=14)
    axes[1].set_ylabel("Percentage (%)", fontsize=14)
    axes[2].set_ylabel("Person", fontsize=14)

    axes[0].set_ylim([0, 100])
    axes[1].set_ylim([0.0, 0.05*100])
    plt.suptitle("Target Data in King County, WA", fontsize=18)
    fig.tight_layout()
    plt.savefig(f"Plot/Target_{str(model.reg)}.png")
    # plt.show()

def get_age_calib_val(model, X):
    X_sum = X.reshape(model.num_reg_group, model.num_age_group, int(X.size / (model.num_age_group * model.num_reg_group)))
    X_sum = np.transpose(X_sum, (1, 0, 2))
    X_sum = np.sum(X_sum, axis=1)
    X_interest = [X_sum[0], sum(X_sum[1:4]), X_sum[-1]]
    return X_interest

def plot_results_with_calib(model, t, ret_list, error_bar=False, lw=0.5, filename=None, title='', includedata=True):
    import plot
    data_date = model.data_date
    data_anti_prop = model.data_anti_prop
    death_rate_range = model.death_rate_range
    data_death = model.data_death
    data_inf_prop = model.data_inf_prop
    inf_rate_range = model.inf_rate_range
    vacc_rate_range = model.vacc_rate_range
    num_group = model.num_group

    l_list = ['-',(0,(5,1)), '--',':','-.','dashed']*100
    marker_list = ['o','s','D','^','v']
    marker_list = ['']*300
    label = ["0-17", "18-64", "65+"]
    
    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5))

    for idx, ret in enumerate(ret_list):
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), num_group, model.num_comp)))
        I = IA + IP
        A = SA + IA + RA
        P = SP + IP + RP
        N = A + P
        D = DA + DP
        A_int_by_age = plot.get_age_calib_val(model, A)
        N_int_by_age = plot.get_age_calib_val(model, N)
        print(round((1-sum(A[:,0])/sum(N[:,0]))*100,3), round((1-sum(A[:,56])/sum(N[:,56]))*100,3), round((1-sum(A[:,-1])/sum(N[:,-1]))*100,3), int(sum(D[:,-1])))
        for i in range(len(data_anti_prop)):
            color = ['r','b','g']
            axes[0].plot(t, 100 * (1 - A_int_by_age[i] / N_int_by_age[i]),
                         label=f'Simulation Results (Age {label[i]})' if idx == 0 else "", 
                        #  color=color[i],
                        #  label= policy_label[idx] if i == 0 else "", 
                        #  label= idx if i == 0 else "", 
                         color='grey',  
                         linestyle = l_list[2-i], marker = marker_list[idx], markevery=60+idx,
                         linewidth=lw, alpha=1.0)
            if error_bar and idx == 0 and includedata:
                axes[0].errorbar(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0],
                                 yerr=[np.ones(len(data_date)), np.ones(len(data_date))],
                                #  yerr=[100 * (1 - data_anti_prop[i]) * (vacc_rate_range[0] - vacc_rate_range[1]),
                                #        100 * (1 - data_anti_prop[i]) * (vacc_rate_range[2] - vacc_rate_range[0])],
                                 fmt='o', ecolor=color[i], color=color[i], capsize=5, markersize=3,
                                 label=f'Observed data (Age {label[i]})')
            elif not error_bar and idx == 0 and includedata:
                axes[0].plot(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0], color='grey', marker='o',
                             linestyle='', label=f'Observed data (Age {label[i]})')

        axes[2].plot(t, sum(D), color='grey', linewidth=lw, alpha=1.0, linestyle = l_list[idx], marker = marker_list[idx], markevery=60+idx,
                    label="Simulation Results" if idx == 0 else "")
        axes[1].plot(t, 100 * sum(I) / sum(N), color='grey', linewidth=lw, alpha=1.0, linestyle = l_list[idx],marker = marker_list[idx],markevery=60+idx,
                    label="Simulation Results" if idx == 0 else "")

        if error_bar and idx == 0 and includedata:
            axes[2].errorbar(data_date, data_death * death_rate_range[0],
                             yerr=[data_death * (death_rate_range[0] - death_rate_range[1]),
                                   data_death * (death_rate_range[2] - death_rate_range[0])],
                             fmt='o', capsize=5, markersize=3,
            label='Observed data')
            axes[1].errorbar(data_date, 100 * data_inf_prop / inf_rate_range[0],
                             yerr=[100 * data_inf_prop * (1 / inf_rate_range[2] - 1 / inf_rate_range[0]),
                                   100 * data_inf_prop * (1 / inf_rate_range[0] - 1 / inf_rate_range[1])],
                             fmt='o', capsize=5, markersize=3,
                             label='Observed data')
        elif not error_bar and idx == 0 and includedata:
            axes[2].plot(data_date, data_death * death_rate_range[0], marker='o', linestyle='')
                        #  label='Observed data')
            axes[1].plot(data_date, 100 * data_inf_prop / inf_rate_range[0], marker='o', linestyle='')
                        #  label='Estimated data')

    for ax in axes:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=2, fontsize=11)    
        if t[-1] <= 100:
            date = [i*7 for i in range(int(t[-1]/7))]
            date_label_full = [i+1 for i in range(int(t[-1]/7))]
            ax.set_xlabel("Week", fontsize=14)
            ax.set_xticks(date)
            ax.set_xticklabels(date_label_full[:len(date)], rotation=0)
            ax.tick_params(axis='both', which='both', labelsize=10)  # Adjust the font size (12 is just an example)
        else:
            date = [i*30.5 for i in range(int(t[-1]/30.5))]
            date_label_full = [i+1 for i in range(int(t[-1]/30.5))]
            # date_label_full = ['Jan/23', 'Feb/23', 'Mar/23', 'Apr/23', 'May/23', 'Jun/23', 'Jul/23', 'Aug/23', 'Sep/23', 'Oct/23', 'Nov/23', 'Dec/23']
            ax.set_xlabel("Month", fontsize=14)
            ax.set_xticks(date)
            ax.set_xticklabels(date_label_full[:len(date)], rotation=0)
            ax.tick_params(axis='both', which='both', labelsize=10)  # Adjust the font size (12 is just an example)

    axes[0].set_title('Vaccinated population by age group', fontsize=14)
    axes[1].set_title('Infectious population', fontsize=14)
    axes[2].set_title('Dead population', fontsize=14)
    
    axes[0].set_ylabel("Percentage (%)", fontsize=14)
    axes[1].set_ylabel("Percentage (%)", fontsize=14)
    axes[2].set_ylabel("Person", fontsize=14)

    axes[0].set_ylim([0, 100])
    # axes[2].set_ylim([0.0, 0.05 * 100])

    # plt.legend(policy_label, title='Best campaign by objective', loc='upper center', bbox_to_anchor=(1.5, 0.8), fancybox=True, shadow=True, ncol=1) 
    # plt.legend([i for i in range(len(ret_list))], title='Best campaign by objective', loc='upper center', bbox_to_anchor=(1.5, 0.8), fancybox=True, shadow=True, ncol=1) 
    
    plt.suptitle(title, fontsize=18)
    fig.tight_layout()
    if filename is None:
        plt.show()
    else: plt.savefig(f"Plot/{filename}.png",  transparent=False, dpi=300, bbox_inches='tight')


def plot_results_with_calib_one_plot(model, t, ret_list,to_plot='vacc', error_bar=False, lw=0.5, filename=None, title='', label=[], includedata=True):
    import plot
    data_date = model.data_date
    data_anti_prop = model.data_anti_prop
    death_rate_range = model.death_rate_range
    data_death = model.data_death
    data_inf_prop = model.data_inf_prop
    inf_rate_range = model.inf_rate_range
    vacc_rate_range = model.vacc_rate_range
    num_group = model.num_group

    c_list = ['grey', 'r', 'b', 'g', 'orange']
    l_list = ['-',(0,(5,1)), '--',':','-.','dashed']
    l_list = ['-']*300
    marker_list = ['o','s','D','^','v']
    marker_list = ['']*300

    c_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
              '#bcbd22', '#17becf', '#1a9850', '#66a61e', '#a6cee3', '#fdbf6f', '#fb9a99', '#e31a1c', 
              '#fb9a99', '#33a02c', '#b2df8a', '#a6cee3']*300

    age_label = ["0-17", "18-64", "65+"]
    
    fig, ax = plt.subplots(figsize=(6, 4))

    for idx, ret in enumerate(ret_list):
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), num_group, model.num_comp)))
        I = IA + IP
        A = SA + IA + RA
        P = SP + IP + RP
        N = A + P
        D = DA + DP
        A_int_by_age = plot.get_age_calib_val(model, A)
        N_int_by_age = plot.get_age_calib_val(model, N)
        print(round((1-sum(A[:,-1])/sum(N[:,-1]))*100,3), int(sum(D[:,-1])))

        if to_plot == 'vacc':
            ax.set_title('Vaccinated population', fontsize=14)
            ax.set_ylabel('Percentage (%)', fontsize=14)
            for i in range(len(data_anti_prop)):
                ax.plot(t, 100 * (1 - A_int_by_age[i] / N_int_by_age[i]),
                            # label= label[idx] if i==0 else None, 
                            # color=c_list[idx], linestyle = l_list[idx], marker = marker_list[idx], markevery=60+idx,
                            linewidth=lw, alpha=1.0)
                if error_bar and idx == 0 and includedata:
                    color = ['r','b','g']
                    ax.errorbar(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0],
                                    yerr=[np.ones(len(data_date)), np.ones(len(data_date))],
                                    fmt='o', ecolor=color[i], color=color[i], capsize=5, markersize=3)
                elif not error_bar and idx == 0 and includedata:
                    ax.plot(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0], color=c_list[i], marker='o')
        elif to_plot == 'vacc_all':
            ax.set_title('Vaccinated population', fontsize=14)
            ax.set_ylabel('Percentage (%)', fontsize=14)
            # ax.set_ylim([30,85])
            ax.plot(t, 100 * (1 - sum(A_int_by_age) / sum(N_int_by_age)),
                        label= label[idx] , 
                        color = c_list[idx//2], alpha = 0.5 if idx%2==1 else 1,
                        linestyle = 'dashed' if idx%2==0 else '-',
                        marker = marker_list[idx], markevery=60+idx,
                        linewidth=lw)
            if idx%2==0:
                base = 100 * (1 - sum(A_int_by_age) / sum(N_int_by_age))
            elif idx%2==1:
                best = 100 * (1 - sum(A_int_by_age) / sum(N_int_by_age))
                ax.fill_between(t, base, best, 
                                color=c_list[(idx) // 2], alpha=0.3)

            if error_bar and idx == 0 and includedata:
                color = ['r','b','g']
                for i in range(len(data_anti_prop)):
                    ax.errorbar(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0],
                                    yerr=[np.ones(len(data_date)), np.ones(len(data_date))],
                                    fmt='o', ecolor=color[i], color=color[i], capsize=5, markersize=3)
            elif not error_bar and idx == 0 and includedata:
                ax.plot(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0], color=c_list[i], marker='o')

        elif to_plot == 'inf':
            ax.set_title('Infectious population', fontsize=14)
            ax.set_ylabel("Percentage (%)", fontsize=14)
            ax.plot(t, 100 * sum(I) / sum(N), color=c_list[idx], linewidth=lw, alpha=1.0, linestyle = l_list[idx],marker = marker_list[idx],markevery=60+idx,
                    label= label[idx])
            if error_bar and idx == 0 and includedata:
                ax.errorbar(data_date, 100 * data_inf_prop / inf_rate_range[0],
                             yerr=[100 * data_inf_prop * (1 / inf_rate_range[2] - 1 / inf_rate_range[0]),
                                   100 * data_inf_prop * (1 / inf_rate_range[0] - 1 / inf_rate_range[1])],
                             fmt='o', capsize=5, markersize=3)
            elif not error_bar and idx == 0 and includedata:
                ax.plot(data_date, 100 * data_inf_prop / inf_rate_range[0], marker='o', linestyle='')
                    
        elif to_plot == 'dead':
            ax.set_title('Dead population', fontsize=14)
            ax.set_ylabel("Person", fontsize=14)
            ax.plot(t, sum(D), color=c_list[idx], linewidth=lw, alpha=1.0, linestyle = l_list[idx],marker = marker_list[idx],markevery=60+idx,
                        label= label[idx])
            
            if error_bar and idx == 0 and includedata:
                ax.errorbar(data_date, data_death * death_rate_range[0],
                                yerr=[data_death * (death_rate_range[0] - death_rate_range[1]),
                                    data_death * (death_rate_range[2] - death_rate_range[0])],
                                fmt='o', capsize=5, markersize=3)
            
            elif not error_bar and idx == 0 and includedata:
                ax.plot(data_date, data_death * death_rate_range[0], marker='o', linestyle='')
                            #  label='Observed data')

        ax.set_xlabel("Month")
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=2)    
        if t[-1] <= model.t_c[-1]:
            date = [i*7 for i in range(int(t[-1]/7))]
            date_label_full = [i+1 for i in range(int(t[-1]/7))]
            ax.set_xlabel("Week", fontsize=14)
            ax.set_xticks(date)
            ax.set_xticklabels(date_label_full[:len(date)], rotation=0)
            ax.tick_params(axis='both', which='both', labelsize=10)  # Adjust the font size (12 is just an example)
        else:
            date = [i*30.5 for i in range(int(t[-1]/30.5))]
            date_label_full = [i+1 for i in range(int(t[-1]/30.5))]
            # date_label_full = ['Jan/23', 'Feb/23', 'Mar/23', 'Apr/23', 'May/23', 'Jun/23', 'Jul/23', 'Aug/23', 'Sep/23', 'Oct/23', 'Nov/23', 'Dec/23']
            ax.set_xlabel("Month", fontsize=14)
            ax.set_xticks(date)
            ax.set_xticklabels(date_label_full[:len(date)], rotation=0)
            ax.tick_params(axis='both', which='both', labelsize=10)  # Adjust the font size (12 is just an example)

    # plt.legend()
    # plt.legend(policy_label, title='Best campaign by objective', loc='upper center', bbox_to_anchor=(1.5, 0.8), fancybox=True, shadow=True, ncol=1) 
    # plt.legend([i for i in range(len(ret_list))], title='Best campaign by objective', loc='upper center', bbox_to_anchor=(1.5, 0.8), fancybox=True, shadow=True, ncol=1) 
    
    # plt.title(title, fontsize=18)
    fig.tight_layout()
    if filename is None:
        plt.show()
    else: plt.savefig(f"Plot/{filename}.png")

def plot_results_with_calib_old(model, t, ret_list, error_bar=False, lw=0.5, filename=None, title=''):
    data_date = model.data_date
    data_anti_prop = model.data_anti_prop
    death_rate_range = model.death_rate_range
    data_death = model.data_death
    data_inf_prop = model.data_inf_prop
    inf_rate_range = model.inf_rate_range
    vacc_rate_range = model.vacc_rate_range
    num_group = model.num_group

    c_list = ['r', 'b', 'g', 'grey', 'orange']
    
    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5.5))

    for idx, ret in enumerate(ret_list):

        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), num_group, model.num_comp)))
        I = IA + IP
        A = SA + IA + RA
        P = SP + IP + RP
        N = A + P
        D = DA + DP
        A_int_by_age = get_age_calib_val(model, A)
        N_int_by_age = get_age_calib_val(model, N)
        # if t[-1]>300:
        #     print(round((1-sum(A[:,0])/sum(N[:,0]))*100,3), round((1-sum(A[:,300])/sum(N[:,300]))*100,3), round((sum(I[:,300])/sum(N[:,300]))*100,3), sum(D[:,300]))
        # else: 
        print(round((1-sum(A[:,0])/sum(N[:,0]))*100,3), round((1-sum(A[:,-1])/sum(N[:,-1]))*100,3), round((sum(I[:,-1])/sum(N[:,-1]))*100,3), sum(D[:,-1]))
        for i in range(len(data_anti_prop)):
            age_label = ["0-17", "18-64", "65+"]
            axes[0].plot(t, 100 * (1 - A_int_by_age[i] / N_int_by_age[i]),
                         label=f'Simulated - Age {age_label[i]}' if idx == 0 else "", linewidth=lw, color=c_list[i], alpha=0.3)
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

        axes[2].plot(t, sum(D), color='grey', linewidth=lw, alpha=0.5,
                     label="Simulation Results" if idx == 0 else "")
        axes[1].plot(t, 100 * sum(I) / sum(N), color='grey', linewidth=lw, alpha=0.5,
                     label="Simulation Results" if idx == 0 else "")

        if error_bar and idx == 0:
            axes[2].errorbar(data_date, data_death * death_rate_range[0],
                             yerr=[data_death * (death_rate_range[0] - death_rate_range[1]),
                                   data_death * (death_rate_range[2] - death_rate_range[0])],
                             fmt='o', capsize=5, markersize=3, label='Observed data')
            axes[1].errorbar(data_date, 100 * data_inf_prop / inf_rate_range[0],
                             yerr=[100 * data_inf_prop * (1 / inf_rate_range[2] - 1 / inf_rate_range[0]),
                                   100 * data_inf_prop * (1 / inf_rate_range[0] - 1 / inf_rate_range[1])],
                             fmt='o', capsize=5, markersize=3, label='Observed data')
        elif not error_bar and idx == 0:
            axes[2].plot(data_date, data_death * death_rate_range[0], marker='o', linestyle='',
                         label='Observed data')
            axes[1].plot(data_date, 100 * data_inf_prop / inf_rate_range[0], marker='o', linestyle='',
                         label='Estimated data')

    for ax in axes:
        ax.set_xlabel("Month")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=2)    
        if t[-1] <= model.t_c[-1]:
            date = [i*7 for i in range(int(t[-1]/7))]
            date_label_full = [i+1 for i in range(int(t[-1]/7))]
            ax.set_xlabel("Week", fontsize=14)
            ax.set_xticks(date)
            ax.set_xticklabels(date_label_full[:len(date)], rotation=0)
            ax.tick_params(axis='both', which='both', labelsize=10)  # Adjust the font size (12 is just an example)
        else:
            date = [i*30.5 for i in range(int(t[-1]/30.5))]
            date_label_full = [i+1 for i in range(int(t[-1]/30.5))]
            # date_label_full = ['Jan/23', 'Feb/23', 'Mar/23', 'Apr/23', 'May/23', 'Jun/23', 'Jul/23', 'Aug/23', 'Sep/23', 'Oct/23', 'Nov/23', 'Dec/23']
            ax.set_xlabel("Month", fontsize=14)
            ax.set_xticks(date)
            ax.set_xticklabels(date_label_full[:len(date)], rotation=0)
            ax.tick_params(axis='both', which='both', labelsize=10)  # Adjust the font size (12 is just an example)

    axes[0].set_title('Vaccinated population', fontsize=14)
    axes[1].set_title('Infectious population', fontsize=14)
    axes[2].set_title('Dead population', fontsize=14)
    axes[0].set_ylabel('Percentage (%)', fontsize=14)
    axes[1].set_ylabel("Percentage (%)", fontsize=14)
    axes[2].set_ylabel("Person", fontsize=14)
    axes[0].set_ylim([0, 100])
    # axes[2].set_ylim([0.0, 0.05 * 100])
        
    plt.suptitle(title, fontsize=18)
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
# plot_results_with_calib(model, t, [ret])
# plot_results(t, [ret])
# %%
