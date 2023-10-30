#%% Import all settings
from vaccinemodel import *
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from alloc import *

king = VaccineModel(53033)
clark = VaccineModel(53011)
okanogan = VaccineModel(53047)
#%%
C_impact = {'C_king_in': np.sum(king.C, axis=1), 'C_king_out': np.sum(king.C, axis=0),
'C_clark_in': np.sum(clark.C, axis=1), 'C_clark_out': np.sum(clark.C, axis=0),
'C_okanogan_in': np.sum(okanogan.C, axis=1), 'C_okanogan_out': np.sum(okanogan.C, axis=0),
}

O_impact = {'O_king_in': np.sum(king.O, axis=1), 'O_king_out': np.sum(king.O, axis=0),
'O_clark_in': np.sum(clark.O, axis=1), 'O_clark_out': np.sum(clark.O, axis=0),
'O_okanogan_in': np.sum(okanogan.O, axis=1), 'O_okanogan_out': np.sum(okanogan.O, axis=0),
}

init_pop = {'init_pop_king': king.N_by_group, 
             'init_pop_clark':clark.N_by_group, 
             'init_pop_okanogan':okanogan.N_by_group,  }

init_anti = {'init_anti_king': king.prop_anti*king.N_by_group, 
             'init_anti_clark':clark.prop_anti*clark.N_by_group, 
             'init_anti_okanogan':okanogan.prop_anti*okanogan.N_by_group,  }

demo_df = pd.DataFrame(C_impact)
demo_df = pd.concat([demo_df, pd.DataFrame(O_impact)], axis=1)
demo_df = pd.concat([demo_df, pd.DataFrame(init_pop)], axis=1)
demo_df = pd.concat([demo_df, pd.DataFrame(init_anti)], axis=1)
# demo_df.to_pickle('final_demographics.pkl')

# %% Observe group information
plt.figure(figsize=(12, 6))
regions = ['reg1', 'reg2', 'reg3', 'reg4', 'reg5']
ages = ['age1', 'age2', 'age3', 'age4', 'age5']
x_tick_labels = [f'{region}-{age}' for region in regions for age in ages]

impact_values = [data * 100 for data in C_impact.values()]
sns.heatmap(impact_values, cmap='Blues', annot=True, fmt='.0f', linewidths=0.5)
plt.xticks(np.arange(len(x_tick_labels))+0.5, x_tick_labels, rotation=90)
plt.yticks(np.arange(len(C_impact))+0.5, C_impact.keys(), rotation=0)
plt.title('Physical contact impact')
plt.show()

plt.figure(figsize=(12, 6))
impact_values = [data * 100 for data in O_impact.values()]
sns.heatmap(impact_values, cmap='Blues', annot=True, fmt='.0f', linewidths=0.5)
plt.xticks(np.arange(len(x_tick_labels))+0.5, x_tick_labels, rotation=90)
plt.yticks(np.arange(len(O_impact))+0.5, O_impact.keys(), rotation=0)
plt.title('Opinion contact impact')
plt.show()

#%% 
plt.figure(figsize=(12, 3))
impact_values = [data * 100 for data in init_anti.values()]
sns.heatmap(impact_values, cmap='Blues', annot=True, fmt='.0f', linewidths=0.5)
plt.xticks(np.arange(len(x_tick_labels))+0.5, x_tick_labels, rotation=90)
plt.yticks(np.arange(len(init_anti))+0.5, init_anti.keys(), rotation=0)
plt.title('Initial anti-vaccination population proportion')
plt.show()

plt.figure(figsize=(12, 3))
impact_values = [data * 100 for data in init_pop.values()]
sns.heatmap(impact_values, cmap='Blues', annot=True, fmt='.0f', linewidths=0.5)
plt.xticks(np.arange(len(x_tick_labels))+0.5, x_tick_labels, rotation=90)
plt.yticks(np.arange(len(init_pop))+0.5, init_pop.keys(), rotation=0)
plt.title('Population proportion')
plt.show()

# %% Analyze allocation results
date = '1025'
df = pd.read_csv(f'top_results_final_{date}.csv')
grouped = df.groupby(['fips', 'obj'])
result = grouped.apply(lambda x: x.select_dtypes(include=['number']).mean())
result = result.drop(columns=['fips', 'Unnamed: 0','point_index','n_iter'])
result = result.reset_index()
result.sort_values(by=['fips','obj'])
result = result[result['obj'] != 'all']

def merge_column(df, name):
    columns = [col for col in df.columns if col.startswith(name)]
    df[name] = (df[columns].values.tolist())
    df[name+'_norm'] = df.apply(lambda x: np.array(x[name])/sum(x[name]), axis=1)
    # df = df.drop(columns=columns, axis=1)
    return df

result = merge_column(result, 'alloc')
result = merge_column(result, 'benefits_deaths')
result = merge_column(result, 'benefits_vacc')
result['disparity_deaths'] = result.apply(lambda x: (np.abs(np.array(x['benefits_deaths'])/np.average(np.array(x['benefits_deaths']))-1)), axis=1)
result['disparity_vacc'] = result.apply(lambda x: (np.abs(np.array(x['benefits_vacc'])/np.average(np.array(x['benefits_vacc']))-1)), axis=1)

result_final = result[~result['obj'].str.contains('by_pop')]
result_final.to_csv(f'refined_result_{date}.csv')
result_final.to_pickle(f'refined_result_{date}.pkl')

# %%
select_columns = 'disparity_vacc' #'alloc_norm' 'benefits_vacc_norm' 'benefits_deaths_norm'
selected_rows = result[~result['obj'].str.contains('by_pop')  & result['obj'].str.contains('deaths_0') ]
# selected_rows = selected_rows.sort_values('obj')
plt.figure(figsize=(12, len(selected_rows)/3))
sns.heatmap(np.vstack(selected_rows[select_columns]), cmap='Blues', annot=True, fmt='.1f', linewidths=0.5)
plt.xticks(np.arange(len(x_tick_labels))+0.5, x_tick_labels, rotation=90)
y_labels = [f"{row['fips']} - {row['obj']}: {round(sum(row[select_columns.rstrip('_norm')]),1)}" for _, row in selected_rows.iterrows()]
plt.yticks(np.arange(len(y_labels))+0.5, y_labels, rotation=0)
plt.title(select_columns)
plt.show()

# %%
reg_1 = '53011'
County_name = {'53011': 'Clark County', '53033': 'King County', '53047': 'Okanogan County'}

gpd_wa_cs = gpd.read_file("Data/cousub20/cousub20.shp") 
gpd_wa_ct = gpd.read_file("Data/tract20/tract20.shp")
gpd_wa_cs = gpd_wa_cs[gpd_wa_cs['COUNTYFP'].str.startswith(reg_1[2:])]

gpd_wa_merge = gpd_wa_cs.merge(demo_df, left_on='NAMELSAD', right_on = 'NAME', how='inner')
gpd_wa_merge.columns = [c.replace(' ', '_').replace('(','').replace(')','').strip() for c in gpd_wa_merge.columns]

# Further combine regions to reduce to 5 groups
reg_label, reg_bracket_ext,reg_bracket_num=[],[],[]
if reg_1=='53011':
    reg_label = ['Battle Ground', 'Camas', 'La Center-Ridgefield-Yacolt','Orchards', 'Vancouver']
    reg_bracket = [['Battle Ground CCD'], ['Camas CCD'], ['La Center CCD','Ridgefield CCD','Yacolt CCD'], ['Orchards CCD'], ['Vancouver CCD']]
    reg_bracket_num = [0,1,2,3,2,4,2]
elif reg_1=='53033':
    reg_label = ['Enumclaw Plateau-\nSnoqualmie Valley','Federal Way-Auburn-\nVashon Island','Issaquah Plateau-\nTahoma-Maple Valley','Seattle','Seattle East']
    reg_label = ['Enumclaw-\nSnoqualmie Valley','Federal Way-Auburn-\nVashon Island','Issaquah Plateau-\nTahoma-Maple Valley','Seattle','Seattle East']
    reg_bracket = [['Enumclaw Plateau CCD','Snoqualmie Valley CCD'],['Federal Way-Auburn CCD','Vashon Island CCD'],['Issaquah Plateau CCD','Tahoma-Maple Valley CCD'],['Seattle CCD'],['Seattle East CCD']]
    reg_bracket_num = [0,1,2,3,4,0,2,1]

elif reg_1=='53047':
    reg_bracket = [['Brewster-Wakefield CCD', 'Conconully-Riverside CCD','Tonasket CCD'],['Colville Reservation CCD','Okanogan CCD'],['Methow Valley CCD','Early Winters CCD'],['Omak CCD'], ['Oroville CCD']]
#     reg_label = ['Brewster-Wakefield-Conconully-Riverside-Tonasket','Colville Reservation-Okanogan','Methow Valley-Early Winters','Omak','Oroville']
    reg_label = ['B-W-C-R-T','C-O','MV-EW','Omak','Oroville']
#     reg_label = ['Mid region','Southeast region','West region','Omak','Oroville']
    reg_bracket_num = [0,1,0,2,2,1,3,4,0]
    
gpd_wa_merge['reg_bracket'] = reg_bracket_num
gpd_wa_merge_dissolve = gpd_wa_merge.dissolve(by='reg_bracket', as_index=False)
gpd_wa_merge_dissolve.index = reg_label

#%% Pareto optimal
obj_mapper = {
'cost_deaths_0':'Max_Deaths',
# 'disparity_deaths_1':'MMD_Deaths_2',
'cost_vacc_0':'Max_Vaccination',
'disparity_deaths_0':'MMD_Deaths',
'disparity_vacc_0':'MMD_Vaccination',
# 'disparity_vacc_1':'MMD_Vaccination_2',
}
custom_order = pd.CategoricalDtype(categories=obj_mapper.keys(), ordered=True)

result_final = pd.read_pickle(f'refined_result_{date}.pkl')

king_final = result_final[result_final['fips'] == 53033]
king_final['obj']  = king_final['obj'].astype(custom_order)
king_final = king_final.sort_values(by='obj')

okanogan_final = result_final[result_final['fips']== 53047]
okanogan_final['obj'] = okanogan_final['obj'].astype(custom_order)
okanogan_final = okanogan_final.sort_values(by='obj')

clark_final = result_final[result_final['fips']== 53011]
clark_final['obj'] = clark_final['obj'].astype(custom_order)
clark_final = clark_final.sort_values(by='obj')
#%%

def get_extended_line(two_points):
    x_coords, y_coords = zip(*two_points)
    slope = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0])
    intercept = y_coords[0] - slope * x_coords[0]
    extended_x = [0]+[x for x in x_coords] + [x_coords[-1]*1.5]  # Extend the x-coordinate
    extended_y = [slope * x + intercept for x in extended_x]
    return (extended_x, extended_y)

# df = clark_final
df = okanogan_final
# df = king_final
County_name = {'53011': 'Clark County', '53033': 'King County', '53047': 'Okanogan County'}
markers = ["s", "o", "^", "D"]*2
colors = ['blue', 'lightblue','red', 'orange']
outcome_metric = ['cost_deaths_0', 'disparity_deaths_0','cost_vacc_0', 'disparity_vacc_0']
titles = ['Total deaths averted\n Vs. Maximum disparity in deaths averted', 'Total vaccination increased\n vs. Maximum disparity in vaccination increased']
xlabels = ['Total deaths averted\n(Per 100K population)', 'Total vaccination increased\n(Per 100K population)']
ylabels = ['Maximum disparity in\ndeaths averted', 'Maximum disparity in\nvaccination increased']

county_name = {53011: 'clark', 53033: 'king', 53047: 'okanogan'}
pop_per_100k= demo_df[f'init_pop_{county_name.get((df.iloc[0].fips))}'].sum()/1e5
outcome_metric_per_unit = []
for metric in outcome_metric:
    outcome_metric_per_unit.append(f'{metric}_per_100k')
    if 'cost' in metric:
        df[f'{metric}_per_100k'] = df[metric] / pop_per_100k
    else:  df[f'{metric}_per_100k'] = df[metric] 

#%%
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# eff_deaths = np.array([[9.943238, 4.177221], [16.445820, 5.577099]])
# (extended_x, extended_y) = get_extended_line(eff_deaths)
# axs[0].plot(eff_deaths[:,0],eff_deaths[:,1], marker='', linestyle='--', color='green')

# eff_vacc = df[['cost_vacc_0', 'disparity_vacc_0']].iloc[:4].sort_values(by='cost_vacc_0').iloc[:4].values
# axs[1].plot(eff_vacc[:,0]/1000, eff_vacc[:,1], marker='', linestyle='--', color='green')

outcome_to_plot = outcome_metric_per_unit
for i in range(len(outcome_to_plot)):
    color = colors[i]
    axs[0].scatter(df.iloc[i][outcome_to_plot[0]], df.iloc[i][outcome_to_plot[1]],
        marker=markers[i], color=color, edgecolors='k', label=obj_mapper.get(df.iloc[i]['obj']), s=70)
    axs[1].scatter(df.iloc[i][outcome_to_plot[2]]/1000, df.iloc[i][outcome_to_plot[3]],
            marker=markers[i], color=color,edgecolors='k', label=obj_mapper.get(df.iloc[i]['obj']), s=70)

axs[0].set_xlim([0, 2])
axs[1].set_xlim([0, 8])
for ax, title, xlabel, yalbel in zip(axs, titles, xlabels, ylabels):
    ax.set_ylim([0,9])
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(yalbel, fontsize=11)
    ax.set_title(title,  fontsize=13)
plt.subplots_adjust(bottom=0.3, top=0.7, left=0.1, right=0.9)
plt.legend(bbox_to_anchor=(1.3, 0.5), loc='center', ncol=1, title='Best campaign by objective', prop={'size': 8})
# plt.suptitle(f'Health benefits by best allocation for each objective in {County_name.get(str(df.iloc[0].fips))}, WA', y=0.82)
# legend = plt.legend(bbox_to_anchor=(-0.1, -0.5), loc='center', ncol=4, 
#            title='Best campaign by objective', prop={'size': 12})
# legend.get_title().set_fontsize(16)
# plt.tight_layout()

# After creating your figure and subplots, add this line before saving:
fig = plt.gcf()
fig.set_size_inches(12, 5)  # Set your desired figsize
plt.savefig(f'Plot/Pareto_{County_name.get(str(df.iloc[0].fips))}.png', transparent=False, dpi=300, bbox_inches='tight')
plt.show()

# %% Get default policy
date = '1025'
result = pd.read_pickle(f'refined_result_{date}.pkl')
result = result[result['obj']!='no_policy']
df = result.copy()
fips_list = [53033, 53047]
for fips in fips_list:
    row = df[(df['fips']==fips) & (df['obj']=='cost_deaths_0')]
    ret_list =[]
    # deaths = []
    # vacc = []
    for i in range(5):
        alc = Alloc(fips_num = row['fips'].values[0], obj_type = 'all', alg='reg_age', B=[10000, 200][fips_list.index(fips)], num_alloc = 20, point_index = i)
        ret_list.append(alc.run_code(parallel=False, 
                                alloc_list = [np.zeros(25)], 
                                # alloc_list = [np.zeros(25)]+[row['alloc'].values[0]], 
                                save_result = True))
        # deaths.append(alc.no_policy_outcome_deaths)
        # vacc.append(alc.no_policy_outcome_vacc)
    plot.plot_results_with_calib(alc.model, alc.model.t_f, ret_list, lw=1.5, error_bar = True, filename=f'no_campaign_{fips}_prev')
    
    # ret_list = np.array(ret_list).reshape((10, 365, 200))

#     deaths = np.mean(deaths, axis=0)
#     vacc = np.mean(vacc, axis=0)
#     new_row = pd.DataFrame([[alc.fips_num, 'no_policy_0', np.sum(deaths), np.max(np.abs(deaths/np.mean(deaths)-1)),np.sum(vacc), np.max(np.abs(vacc/np.mean(vacc)-1)), np.zeros(25), deaths, vacc ]], columns = ['fips','obj','cost_deaths_0','disparity_deaths_0','cost_vacc_0','disparity_vacc_0','alloc','benefits_deaths','benefits_vacc'])
#     result = pd.concat([new_row, result])
# result.to_pickle(f'Result/refined_result_with_no_policy_{date}.pkl')
    
# %%

file_path = f'Result/refined_result_with_no_policy_{date}.pkl'
df = pd.read_pickle(file_path)
rows = df[(df['fips']==53047)]

glob_ret_best = []
for index, row in rows.iterrows():
    ret_list_no = []
    ret_list_best = []
    for i in [8]:
        for alloc in ([row['alloc']]):
        # for alloc in ([row['alloc'].values[0], np.zeros(25)]):
            alc = Alloc(fips_num = row['fips'], obj_type = 'all', alg='reg_age', B=100, num_alloc = 20, point_index = i)
            alc.model = VaccineModel(alc.fips_num, init_param_list = alc.init_param_list, 
                                        param_update_list=alc.param_update_list)
            t = alc.model.t_f
            alc.model.update_param("U", alloc)
            ret = odeint(alc.model.run_model, alc.model.get_y0(), t)
            ret_list_best.append(ret)
        # print(alc.model.min_rat, np.mean(alc.model.mean_rat), alc.model.max_rat)
        # print(alc.model.min_emo, np.mean(alc.model.mean_emo), alc.model.max_emo)
        # ret_mean_no = np.mean(ret_list_no, axis=0)
    ret_mean_best = np.mean(ret_list_best, axis=0)
    glob_ret_best.append(ret_mean_best)
    # plot.plot_results_with_calib(alc.model, t, [ret_mean_no, ret_mean_best], lw=1.5, error_bar = True)
plot_results_with_calib(alc.model, t, glob_ret_best, lw=1.5, error_bar = True, filename='Plot_53047')
# plot.plot_results_with_calib(alc.model, t, ret_list, lw=0.5, error_bar = True)
# %%

def plot_results_with_calib(model, t, ret_list, error_bar=False, lw=0.5, filename=None, title=''):
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
    l_list = ['-']*30
    marker_list = ['o','s','D','^','v']
    marker_list = ['']*10

    c_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
              '#bcbd22', '#17becf', '#1a9850', '#66a61e', '#a6cee3', '#fdbf6f', '#fb9a99', '#e31a1c', 
              '#fb9a99', '#33a02c', '#b2df8a', '#a6cee3']

    label = ["0-17", "18-64", "65+"]
    policy_label = ['No campaign','Max_Deaths','Max_Vaccination','MMD_Deaths','MMD_Vaccination']
    
    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 4))

    for idx, ret in enumerate(ret_list):
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(ret), (len(t), num_group, model.num_comp)))
        I = IA + IP
        A = SA + IA + RA
        P = SP + IP + RP
        N = A + P
        D = DA + DP
        A_int_by_age = plot.get_age_calib_val(model, A)
        N_int_by_age = plot.get_age_calib_val(model, N)
        print(round((1-sum(A[:,-1])/sum(N[:,-1]))*100,3))
        for i in range(len(data_anti_prop)):
            axes[0].plot(t, 100 * (1 - A_int_by_age[i] / N_int_by_age[i]),
                        #  label=f'Simulated-Age {label[i]}' if idx == 0 else "", color=c_list[i],
                        #  label= policy_label[idx] if i == 0 else "", 
                         label= idx if i == 0 else "", 
                         color=c_list[idx], linestyle = l_list[idx], marker = marker_list[idx], markevery=60+idx,
                         linewidth=lw, alpha=1.0)
            if error_bar and idx == 0:
                color = ['r','b','g']
                axes[0].errorbar(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0],
                                 yerr=[np.ones(len(data_date)), np.ones(len(data_date))],
                                #  yerr=[100 * (1 - data_anti_prop[i]) * (vacc_rate_range[0] - vacc_rate_range[1]),
                                #        100 * (1 - data_anti_prop[i]) * (vacc_rate_range[2] - vacc_rate_range[0])],
                                 fmt='o', ecolor=color[i], color=color[i], capsize=5, markersize=3)
                                #  label=f'Observed data (Age {label[i]})')
            elif not error_bar and idx == 0:
                axes[0].plot(data_date, 100 * (1 - data_anti_prop[i]) * vacc_rate_range[0], color=c_list[i], marker='o')
                            #  linestyle='', label=f'Observed data (Age {label[i]})')

        axes[2].plot(t, sum(D), color=c_list[idx], linewidth=lw, alpha=1.0, linestyle = l_list[idx],marker = marker_list[idx],markevery=60+idx,
                    label= idx)
                    # label= policy_label[idx])
                    #  label="Simulation Results" if idx == 0 else "")
        axes[1].plot(t, 100 * sum(I) / sum(N), color=c_list[idx], linewidth=lw, alpha=1.0, linestyle = l_list[idx],marker = marker_list[idx],markevery=60+idx,
                    label= idx)
                    # label= policy_label[idx])
                    #  label="Simulation Results" if idx == 0 else "")

        if error_bar and idx == 0:
            axes[2].errorbar(data_date, data_death * death_rate_range[0],
                             yerr=[data_death * (death_rate_range[0] - death_rate_range[1]),
                                   data_death * (death_rate_range[2] - death_rate_range[0])],
                             fmt='o', capsize=5, markersize=3)
            # label='Observed data')
            axes[1].errorbar(data_date, 100 * data_inf_prop / inf_rate_range[0],
                             yerr=[100 * data_inf_prop * (1 / inf_rate_range[2] - 1 / inf_rate_range[0]),
                                   100 * data_inf_prop * (1 / inf_rate_range[0] - 1 / inf_rate_range[1])],
                             fmt='o', capsize=5, markersize=3)
                            #  , label='Observed data')
        elif not error_bar and idx == 0:
            axes[2].plot(data_date, data_death * death_rate_range[0], marker='o', linestyle='')
                        #  label='Observed data')
            axes[1].plot(data_date, 100 * data_inf_prop / inf_rate_range[0], marker='o', linestyle='')
                        #  label='Estimated data')

    for ax in axes:
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

    axes[0].set_title('Vaccinated population', fontsize=14)
    axes[1].set_title('Infectious population', fontsize=14)
    axes[2].set_title('Dead population', fontsize=14)
    axes[0].set_ylabel('Percentage (%)', fontsize=14)
    axes[1].set_ylabel("Percentage (%)", fontsize=14)
    axes[2].set_ylabel("Person", fontsize=14)
    axes[0].set_ylim([0, 100])
    # axes[2].set_ylim([0.0, 0.05 * 100])

    # plt.legend(policy_label, title='Best campaign by objective', loc='upper center', bbox_to_anchor=(1.5, 0.8), fancybox=True, shadow=True, ncol=1) 
    plt.legend([i for i in range(len(ret_list))], title='Best campaign by objective', loc='upper center', bbox_to_anchor=(1.5, 0.8), fancybox=True, shadow=True, ncol=1) 
    
    plt.suptitle(title, fontsize=18)
    fig.tight_layout()
    if filename is None:
        plt.show()
    else: plt.savefig(f"Plot/{filename}.png")

# %%from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
data = pd.read_csv('test.csv')
data = data.applymap(strip_percentage_and_convert)
data = data.dropna()
X = data[['overall_alpha', 'beta', 'prop_sus', 'O_m', 'p1', 'p2', 'p3', 'p4','p5', 'rae', 'k_R', 'k_E', 'lam']]
Y = data[['p1_d', 'p2_d', 'p3_d', 'p4_d','p5_d', 'p1_v', 'p2_v', 'p3_v']] 

# Create interaction features for X
poly = PolynomialFeatures(degree=1, interaction_only=True, include_bias=False)
X_interactions = poly.fit_transform(X)

# List of target columns
target_column = ['p1_d', 'p2_d', 'p3_d', 'p4_d']
target_column += ['p1_v', 'p2_v', 'p3_v']

# Initialize a Random Forest model
model = RandomForestRegressor()

# Fit the model to X with interaction features and the selected target columns
model.fit(X_interactions, Y[target_column])

# Get feature importances
feature_importances = model.feature_importances_

# Sort features by importance
sorted_features = sorted(zip(poly.get_feature_names_out(X.columns), feature_importances), key=lambda x: x[1], reverse=True)

# Print feature importances
for feature, importance in sorted_features:
    print(f"{feature}: {importance}")
