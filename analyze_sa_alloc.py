#%% import settings

from alloc import Alloc
from sa_alloc import *

import numpy as np
import pandas as pd
import glob
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import product
from scipy.interpolate import interp1d
import plot
import json
import pickle

colors = sns.color_palette("tab10", n_colors= 5)
custom_cmap = ListedColormap(colors)
color = [custom_cmap(i) for i in range(5)]
marker = ['o','s','D','^','v']
title_font_size = 20
label_font_size = 16
tick_font_size = 14
age_label = ['0-17','18-34','35-49','50-64','65+']
reg_label = ['Seattle','Seattle\nEast','FAV','ITM', 'ES']
group_label  = ['1,1','1,2','1,3','1,4','1,5']
large_sa = ['vaccine_risk', 'p_emotional']
within_sa = ['k_R', 'k_E']
num_pop = 2195285.4364228635
# %% read all data and combine into one
re_order = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4]
def reorder_list(lst, re_order):
    return [lst[i] for i in re_order]

def clean_data(date = '0220'):
    import os
    folder_path = 'SA_Result'
    file_pattern = f'*{date}*.pkl'

    matching_files = glob.glob(os.path.join(folder_path, file_pattern))
    dfs = []
    for file_path in matching_files:
        print(file_path)
        df = pd.read_pickle(file_path)
        df['B'] = int(file_path.split('B')[-1].split('_')[1])
        df['source'] = file_path.split('_')[-1].split('.')[0]
        # df['B'] = (df['alloc'].apply(sum))
        dfs.append(df)
    glob_df = pd.concat(dfs, ignore_index=True)

    def unpack_tuples(row):
        return {key: value for key, value in row}

    glob_df['param_update'] = glob_df['param_update'].apply(lambda x: ast.literal_eval(x))
    pd.concat([glob_df['param_update'].apply(lambda x: unpack_tuples(x)).apply(pd.Series)
    , glob_df],  axis=1)
    glob_df = pd.concat([glob_df['param_update'].apply(lambda x: unpack_tuples(x)).apply(pd.Series)
    , glob_df],  axis=1)
    # glob_df = glob_df.drop(columns = ['p2','p3','p4','p5'])
    # glob_df = glob_df.rename(columns={'p1': 'p_emotional'})

    # Define the re_order alloc results
    # glob_df['alloc'] = glob_df.filter(like='alloc_').values.tolist()
    glob_df['org_alloc'] = glob_df['alloc'].copy()
    glob_df['alloc'] = glob_df['alloc'].apply(lambda x: reorder_list(x, re_order))

    # Get max_alloc
    glob_df['max_alloc'] = glob_df['alloc'].apply(lambda x: np.argmax(x))

    def set_max_alloc(row):
        max_value = max(row['alloc'])
        max_indices = [index for index, value in enumerate(row['alloc']) if value == max_value]
        return max_indices

    glob_df['max_alloc_list'] = glob_df.apply(set_max_alloc, axis=1)
    glob_df['max_alloc'] =  glob_df['max_alloc_list'].apply(lambda x: np.mean(x))
    glob_df.to_csv(f'Refined_result/refined_sa_{date}_king.csv')
    glob_df.to_pickle(f'Refined_result/refined_sa_{date}_king.pkl')

def format_annotation(value):
    quotient = value // 5+1
    remainder = value % 5+1
    return f"({int(quotient)},{int(remainder)})"

def format_annotation_list(list):
    text = ''
    if len(list) == 5 and all(list[i] == list[i-1] + 1 for i in range(1, 5)):
        return f'Reg {list[0] // 5 + 1}'
    elif len(list) == 5 and all(list[i] == list[i-1] + 5 for i in range(1, 5)):
        return f'Age {list[0] % 5 + 1}'
    
    for index, value in enumerate(list):
        quotient = value // 5 + 1
        remainder = value % 5 + 1
        if index > 0:
            text += ', '
        text += f"({int(quotient)},{int(remainder)})"
    return text

def custom_annot(x, pos, fontsize_base=10):
    # Adjust the fontsize dynamically based on the value in the cell
    fontsize = fontsize_base * (1 + x / pivot_data.values.max())
    return f'{x:.2f}', fontsize, 'black'

color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
              '#bcbd22', '#17becf', '#1a9850', '#66a61e', '#a6cee3', '#fdbf6f', '#fb9a99', '#e31a1c', 
              '#fb9a99', '#33a02c', '#b2df8a', '#a6cee3']*300

def plot_eta(eta_list, label=None, title=''):        
    fig, ax = plt.subplots(figsize=(3, 2.5))
    
    for i, eta in enumerate(eta_list):
        x, y = np.array(eta).T
        ax.plot(x, y, linewidth = 1, 
                color = color[i//2], alpha = 0.5 if i%2==1 else 1,
                linestyle = 'dashed' if i%2==0 else '-',
                label=label[i] if label is not None else None)
    for i in range(0, len(eta_list), 2):
        eta_arr1 = eta_list[i]
        eta_arr2 = eta_list[i + 1]        
        f1 = interp1d(np.array(eta_arr1)[:, 0], np.array(eta_arr1)[:, 1], kind='linear', fill_value='extrapolate')
        f2 = interp1d(np.array(eta_arr2)[:, 0], np.array(eta_arr2)[:, 1], kind='linear', fill_value='extrapolate')
        ax.fill_between(x, f1(x), f2(x), color=color[i // 2], alpha=0.3)

    ax.set_xlabel('Month', size=12)
    ax.set_xticks([i*30.4 for i in range(12)])
    ax.set_xticklabels([i+1 for i in range(12)], size=10)
    ax.set_ylabel(f'Opinion persuasiveness\n $\\overline{{\\eta}}(t$)', size=12)
    ax.set_ylim([0, 1.0])
    if label is not None:
        ax.legend(loc='center', bbox_to_anchor=(1.8, 0.5))
    # plt.tight_layout()
    plt.title(title)
    plt.show()

def plot_mid_results(glob_df, obj):
    mid_value = glob_df
    y = ['k_E', 'k_R']
    y = ['p_emotional', 'vaccine_risk']
    if 'k_E' in y:
        xticklabels = ['Low', '', '', '', 'High'][::-1]
        yticklabels = xticklabels
        xlabel = f'Emotional sensitivity ($k_E$)'
        ylabel = f'Rational sensitivity ($k_R$)'
    if 'p_emotional' in y:
        xticklabels = mid_value[y[0]].unique()
        yticklabels = mid_value[y[1]].unique()
        xlabel = f'Importance of emotional judgment ($\\rho$)'
        ylabel =f'Perceived vaccine risk ($\\nu$)'

    # mid_value = glob_df[(glob_df['k_R'] == 20000) & (glob_df['k_E'] == 12)]
    # mid_value = glob_df[(glob_df['k_R'] == 20000) & (glob_df['k_E'] == 12) & (glob_df['vaccine_risk'].isin([0, 1.5e-4, 3e-4])) & (glob_df['p_emotional'].isin([0, 0.5, 1]))]
    var_shape = (int(np.sqrt(len(mid_value))),int(np.sqrt(len(mid_value))))
    
    #First plot outcomes
    fig, ax = plt.subplots(1,1,figsize=(5, 4))
    val = mid_value[obj].values.reshape(var_shape).copy()
    if obj == 'cost_vacc_0':
        val /= num_pop
        title = 'Vaccination uptake increase (%)'
    if obj=='cost_deaths_0':
        title = 'Deaths averted'
    sns.heatmap(val, annot=True, cmap= 'Blues', ax=ax, fmt='.1%', linewidth=0.5, annot_kws={'fontsize': tick_font_size-3})
    if 'k_E' in y:
        ax.invert_xaxis()
    ax.set_title(title, fontsize=title_font_size-5)
    ax.set_xlabel(xlabel, fontsize=14)              
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticklabels(xticklabels,fontsize=10, rotation=0)      
    ax.set_yticklabels(yticklabels, fontsize=10, rotation=0)
    plt.show()

def plot_fourway_sa(glob_df, B_list, obj_outcome_list,outcome_list, nu, rho):
    for B in B_list:
    # for B in [2000, 10000, 50000]:
        for obj_outcome in obj_outcome_list:
            for outcome in outcome_list:
                # if outcome not in ['max_alloc', obj_outcome]: continue
                print(obj_outcome, outcome)
                df = glob_df[glob_df['B']==B].copy()
                vmin = df[outcome].min()
                vmax = df[outcome].max()
                if outcome == 'cost_vacc_0':  vmin/=1e4; vmax/=1e4; vmin=0; vmax=10
                elif outcome == 'max_alloc' and 'cost' in obj_outcome:  vmin=0; vmax=5
                num_row = len(glob_df[large_sa[0]].unique())
                num_col = len(glob_df[large_sa[1]].unique())
                fig, axes = plt.subplots(num_row, num_col, figsize=(16, 12))
                count = 0
                for i, (sa1, sa2) in enumerate(glob_df[large_sa].drop_duplicates().values):
                    ax = axes[count // num_col, count % num_col]
                    filtered_data = df[(df[large_sa[0]] == sa1) & (df[large_sa[1]] == sa2) & (df['obj'] == obj_outcome)]
                    pivot_data = filtered_data.pivot_table(index=within_sa[0], columns=within_sa[1], values=outcome, aggfunc='first')
                    if outcome == 'max_alloc':
                        annot_data = filtered_data.pivot_table(index=within_sa[0], columns=within_sa[1], values='max_alloc_list', aggfunc='first')
                        annot_data = annot_data.applymap(format_annotation_list)    
                        # print(annot_data, vmin, vmax)       
                        sns.heatmap(pivot_data, 
                                    # annot=False,
                                    annot=annot_data, 
                                    fmt='', cmap=custom_cmap, cbar=False, linewidths=1, 
                                    ax=ax, vmin=vmin, vmax=vmax,  annot_kws={'fontsize': 15})
                    else:
                        if outcome == 'cost_vacc_0': pivot_data /= 1e4
                        im = sns.heatmap(pivot_data, annot=True, fmt='.0f',
                                    cmap='Blues_r' if 'disparity' in outcome else 'Blues', cbar=False, linewidths=1,
                                    ax=ax, vmin=vmin, vmax=vmax, annot_kws={'fontsize': tick_font_size})
                    ax.set_title(f'({chr(97 + count)}) $\\rho={sa1}$, $\\nu={sa2*1e4:.1f}$e-4', fontsize=title_font_size)
                    ax.set_title(f'({chr(97 + count)}) $\\nu=${nu[i//3]}, $\\rho={sa2}$', fontsize=title_font_size)
                    ax.invert_xaxis()
                    # ax.set_xticklabels(['High', '', 'Low'], fontsize=tick_font_size, rotation=0)
                    # ax.set_yticklabels(['High', '','Low'], fontsize=tick_font_size)
                    ax.set_xlabel(f'Emotional sensitivity ($k_E$)', fontsize=label_font_size)
                    ax.set_ylabel(f'Rational sensitivity ($k_R$)', fontsize=label_font_size)

                    count += 1
                # sns.set(font_scale=2)
                # if outcome =='cost_vacc_0': title = f'Vaccination uptake increase $\\sum_{{i=1}}^{{n}} \\Delta P_i(b,x;T)$ (10,000s)'
                if outcome=='max_alloc':
                    cbar_ax = fig.add_axes([0.92, 0.3, 0.03, 0.4])  # Adjust the position and size as needed
                    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap.reversed()), cax=cbar_ax)
                    cbar.ax.set_title('Group (region, age)', fontsize=title_font_size, loc='left', pad=20)
                    cbar.ax.tick_params(labelsize=12)  # Adjust the fontsize of the tick labels
                    cbar.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9])
                    label = ['(1,'+str(g+1)+') (Seattle, Age '+age_label[g] +')' for g in range(5)][::-1]
                    cbar.set_ticklabels(label, fontsize = tick_font_size+3)
                    title = 'Optimal allocated group\n'
                if outcome == 'cost_vacc_0':
                    cbar_ax = fig.add_axes([0.92, 0.3, 0.03, 0.4])
                    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='Blues'), cax=cbar_ax)
                    tick_interval = (vmax - vmin) / 4 
                    tick_positions = np.arange(vmin, vmax + tick_interval, tick_interval)
                    tick_labels = [int(x) for x in tick_positions]
                    cbar.set_ticks((tick_positions - vmin) / (vmax - vmin))  # Normalize tick positions to range [0, 1]
                    cbar.set_ticklabels(tick_labels, fontsize = tick_font_size+3)
                    # cbar.ax.tick_params(labelsize=12) 
                    title = f'Vaccination uptake increase (%)\n'
                    # cbar.ax.set_label(title)

                fig.text(-0.06, 0.5, f'Perceived Vaccine risk ($\\nu$)', ha='center', va='center', rotation=90, weight='bold', fontsize=title_font_size+5)
                fig.text(-0.02, 0.17, f'High\n($\\nu=${nu[2]})', ha='center', va='center', rotation=90, weight='bold', fontsize=title_font_size)
                fig.text(-0.02, 0.5, f'Medium\n($\\nu=${nu[1]})', ha='center', va='center', rotation=90, weight='bold', fontsize=title_font_size)
                fig.text(-0.02, 0.8, f'Low\n($\\nu=${nu[0]})', ha='center', va='center', rotation=90, weight='bold', fontsize=title_font_size)

                fig.text(0.5, -0.08, f'Importance of emotional judgment ($\\rho$)', ha='center', va='center', rotation=0, weight='bold', fontsize=title_font_size+5)
                fig.text(0.17, -0.025, f'Rational\n($\\rho=0$)', ha='center', va='center', rotation=0, weight='bold', fontsize=title_font_size)
                fig.text(0.5, -0.025, f'Balanced\n($\\rho=0.5$)', ha='center', va='center', rotation=0, weight='bold', fontsize=title_font_size)
                fig.text(0.83, -0.025, f'Emotional\n($\\rho=1$)', ha='center', va='center', rotation=0, weight='bold', fontsize=title_font_size)

                plt.suptitle(title, fontsize = title_font_size+10, weight='bold')
                plt.tight_layout(rect=[0, 0, .9, 1])
                # plt.savefig(f'Plot/{obj_outcome}_{outcome}_B_{B}_2.png',bbox_inches='tight', dpi=200)
                plt.show()

def deserialize_arrays(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)

def run_model(B_list, sa_list, glob_df):

    columns_to_create = ['base_result', 'campaign_result', 'base_eta', 'campaign_eta', 'base_eta_all', 'campaign_eta_all']
    for col in columns_to_create:
        if col not in glob_df.columns:
            glob_df[col] = None  # or any default value suitable for your data type

    glob_ret_list = []
    glob_eta_list = []
    glob_all_eta_list = []
    glob_best_alloc_list = []
    for B in B_list:
        ret_list = []
        eta_list = []
        all_eta_list = []
        best_alloc_list = []
        for sa in sa_list:
            init_param_list = sa.copy()
            alc = Alloc(fips_num = 53033, obj_type = 'all', alg='reg_age', 
                        B=B, init_param_list  = init_param_list)
            row = glob_df[(glob_df['p_emotional'] == sa[0][1]) &
                                (glob_df['vaccine_risk'] == sa[1][1]) &
                                (glob_df['k_R'] == sa[2][1]) &
                                (glob_df['k_E'] == sa[3][1]) &
                                (glob_df['B'] == B) &
                                (glob_df['obj'] == obj)]['org_alloc']
            if len(row)>0:
                best_alloc = row.iloc[0]
                best_alloc_list.append(best_alloc)
                print(best_alloc[15:20])
                alloc_test = [np.zeros(25), np.array(best_alloc)]
                rets,etas,all_etas = alc.run_code(parallel=False, alloc_list = alloc_test, save_result = True)
                for ret in rets: ret_list.append(ret)
                for eta in etas: eta_list.append(eta)
                for all_eta in all_etas: all_eta_list.append(all_eta)
        glob_ret_list.append(ret_list)
        glob_eta_list.append(eta_list)
        glob_all_eta_list.append(all_eta_list)
        glob_best_alloc_list.append(best_alloc_list)
    return (glob_ret_list, glob_eta_list, glob_all_eta_list, glob_best_alloc_list)
    # plot_eta(glob_eta_list, label= B_list)
    # plot.plot_results_with_calib_one_plot(alc.model, alc.model.t_f, glob_ret_list,to_plot='vacc', 
    #                                       error_bar=True, lw=1.5, label = B_list, includedata=False)


#%% Read data
    
date='0225_base'
clean_data(date)
filename = f'Refined_result/refined_sa_{date}_king.pkl'

glob_df = pd.read_pickle(filename)
glob_df = glob_df[(glob_df['vaccine_risk'].isin([0, 1.5e-4, 3e-4])) & (glob_df['p_emotional'].isin([0, 0.5, 1]))]
obj = 'cost_vacc_0'
obj_outcome_list = ['cost_vacc_0']
outcome_list = obj_outcome_list+['max_alloc']
nu = [round(i,5) for i in glob_df['vaccine_risk'].unique()]
rho = glob_df['p_emotional'].unique()
glob_df = glob_df.sort_values(by=['B']+large_sa+within_sa, ascending=[True, True, True, True, True])
B_list = glob_df.B.unique()

#%%
glob_df = glob_df[glob_df['B']==20000]
# plot_mid_results(plot_df, obj)

# krke_list = [1,4,7]+[3,4,5]
# sa_list = get_mid_sa()
# sa_list = [sa_list[k] for k in krke_list]
sa_list = glob_df['param_update'].values
glob_ret_list, glob_eta_list, glob_all_eta_list, glob_best_alloc_list =  run_model(B_list, sa_list, glob_df)

#%%
# Plot eta
# num_row = len(plot_df[large_sa[0]].unique())
# num_col = len(plot_df[large_sa[1]].unique())

num_row = 3
num_col = num_row
fig, axes = plt.subplots(num_row, num_col, figsize=(4+3*num_col, 3*num_row))
B_ind = 0
# fig, axes = plt.subplots(3,3, figsize=(11,6.5))
plt.rcdefaults()
for k in range(1):
    for B_ind in range(len(B_list)):
        # print(sa_list[k][4:])
        eta_per_group = []
        best_alloc = glob_best_alloc_list[B_ind][k][15:20]
        for i in [0,3,6]:
            x = np.array([data[0] for data in glob_all_eta_list[B_ind][i]]).T
            y = np.array([data[1] for data in glob_all_eta_list[B_ind][i]])[:, 15:20]
            # y = np.array([data[1] for data in glob_all_eta_list[B_ind][2*k+i]])[:, 20:25]
            # y = np.array([data[1] for data in glob_eta_all_list[B_ind][2*k+i]])
            eta_per_group.append([x, y])
        ax = axes[B_ind//num_col, B_ind%num_col]
        # ax.set_title(f'({chr(97 + k)}) $\\nu=${nu[k//3]}, $\\rho={rho[k%3]}$', fontsize=15)

        for i, eta in enumerate(eta_per_group):
            if i%2==2: continue
            x, y = eta
            print( ["{:.0%}".format(val) for val in y[60]])
            for g in range(len(y.T)):
                ax.plot(x, y.T[g], linewidth = 2, color = color[g], alpha =1,
                        linestyle = 'dashed' if i%2==0 else '-')
                        # marker = marker[g] if i%2==1 else '', markersize=5, markevery=4000,
                        # label = '(1,'+str(g+1)+') (Seattle, Age '+age_label[g] +')' if (i%2==1 and k==0) else None)
                # desired_values = np.arange(0, 361, 60)
                # closest_indices = np.searchsorted(x, desired_values, side='left')
                # closest_indices = np.clip(closest_indices, 0, len(x) - 1)
                # if i%2==1:
                #     for mark_every in closest_indices:
                #             ax.plot(x[mark_every], y.T[g][mark_every], marker=marker[g], linestyle='None', 
                #                     markersize=4, color = color[g])
                # print(k, g, [int(y.T[g][idx]*100) for idx in closest_indices])
                            
            # if i%2==1:
            #     f0 = interp1d(eta_per_group[0][0], (eta_per_group[0][1]).T, kind='linear', fill_value='extrapolate')(x)
            #     f1 = interp1d(eta_per_group[1][0], (eta_per_group[1][1]).T,  kind='linear', fill_value='extrapolate')(x)
            #     for h in range(len(y.T)):
            #         ax.fill_between(x, f0[h], f1[h], color = color[h], alpha=0.5)
                    

        # plt.legend(bbox_to_anchor = [1.0,0.85])
        ax.set_xlabel('Month', size=12)
        ax.set_xticks([i*30.4 for i in range(12)])
        ax.set_xticklabels([i+1 for i in range(12)], size=10)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], size=10)
        ax.set_ylabel(f'${{\\eta}}_i(t)$', size=12)
        ax.set_ylim([0,1.05])
        # ax.axvline(x=365, color='grey', linestyle='-', linewidth = 3.5)
        title_font_size = 18  # Adjust the title font size
        label_font_size = 14  # Adjust the axis labels' font size

fig.legend(bbox_to_anchor=(1.25, 0.6), ncol=1, title='Group')

# fig.text(-0.06, 0.5, f'Perceived Vaccine risk ($\\nu$)', ha='center', va='center', rotation=90, fontsize = title_font_size, weight='bold')
# fig.text(-0.02, 0.16, f'High\n($\\nu=${nu[2]})', ha='center', va='center', rotation=90,  fontsize = label_font_size, weight='bold')
# fig.text(-0.02, 0.47, f'Medium\n($\\nu=${nu[1]})', ha='center', va='center', rotation=90, fontsize = label_font_size, weight='bold')
# fig.text(-0.02, 0.8, f'Low\n($\\nu=${nu[0]})', ha='center', va='center', rotation=90, fontsize = label_font_size, weight='bold')

# fig.text(0.5, -0.1, f'Importance of emotional judgment ($\\rho$)', ha='center', va='center', rotation=0, fontsize = title_font_size, weight='bold')
# fig.text(0.21, -0.03, f'Rational\n($\\rho=0$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')
# fig.text(0.53, -0.03, f'Balanced\n($\\rho=0.5$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')
# fig.text(0.86, -0.03, f'Emotional\n($\\rho=1$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')

plt.suptitle('Changes in opinion persuasiveness with optimal campaign', fontsize = title_font_size, weight='bold')

plt.tight_layout()
# plt.savefig(f'Plot/Eta_B_{B}_2.png', transparent=False, dpi=200, bbox_inches='tight')
plt.show()

#%% Plot vacc
# fig, axes = plt.subplots(3,3, figsize=(11,6.5))
num_row = 3
num_col = num_row
fig, axes = plt.subplots(num_row, num_col, figsize=(4+3*num_col, 3*num_row))
B_ind = 0
plt.rcdefaults()
day_shorten_list = []

for k in range(len(sa_list)):
    for B_ind in [1]:
        ax = axes[k//num_col, k%num_col]    
        # ax = axes[B_ind//num_col, B_ind%num_col]    
        # print(sa_list[k][4:])
        eta_per_group = []
        # best_alloc = glob_best_alloc_list[B_ind][k][15:20]

        herd_day_index = []
        for i in [0,1]:
            [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(glob_ret_list[B_ind][2*k+i]), (len(glob_ret_list[B_ind][0]), 25, 8)))
            A = SA + IA + RA
            P = SP + IP + RP
            N = A + P
            I = IA + IP
            
            y = (P/N)[15:20,:].T
            # y = (np.sum(P.reshape((5,5,365)), axis=1)/np.sum(N.reshape((5,5,365)), axis=1)).T
            # y = (np.sum(P.reshape((5,5,365)), axis=0)/np.sum(N.reshape((5,5,365)), axis=0)).T
            # y = (I/N)[15:20,:].T
            # y = ((DA+DP))[15:20,:].T
            x = np.array([i for i in range(len(y))])
            eta_per_group.append([x, y])

        for i, eta in enumerate(eta_per_group):
            # if i%2>1: continue
            x, y = eta
            # print( ["{:.0%}".format(val) for val in y[60]])
            group_label = [i for i in range(1,6)]
            for g in range(5):
                ax.plot(x, y.T[g], 
                        linewidth = 2, 
                        color = color[g], 
                        alpha =1,
                        linestyle = 'dashed' if i%2==0 else '-',
                        marker = marker[g] if i%2==1 else '',
                        markersize=5, markevery=4000,
                        label = group_label[g] if (i%2==1 and k==0) else None)
                desired_values = np.arange(0, 361, 60)
                closest_indices = np.searchsorted(x, desired_values, side='left')
                closest_indices = np.clip(closest_indices, 0, len(x) - 1)
                
                if i%2==1:
                    for mark_every in closest_indices:
                            ax.plot(x[mark_every], y.T[g][mark_every], marker=marker[g], linestyle='None', 
                                    markersize=4, color = color[g])
                # print(k, g, [int(y.T[g][idx]*100) for idx in closest_indices])
            if i%2==1:
                f0 = interp1d(eta_per_group[0][0], (eta_per_group[0][1]).T, kind='linear', fill_value='extrapolate')(x)
                f1 = interp1d(eta_per_group[1][0], (eta_per_group[1][1]).T,  kind='linear', fill_value='extrapolate')(x)
                cumul_benefit = np.zeros((f0[0]).shape)
                for h in range(5):
                    # ax.fill_between(x, -(f0[h]-f1[h]), color = color[h], alpha=0.5,
                    #         label = '(1,'+str(h+1)+') (Seattle, Age '+age_label[h] +')' if k==0 else None)                
                    # ax.fill_between(x, cumul_benefit, cumul_benefit-(f0[h]-f1[h]), color = color[h], alpha=0.5,
                    #         label = '(1,'+str(h+1)+') (Seattle, Age '+age_label[h] +')' if k==0 else None)                
                    # cumul_benefit -= f0[h]-f1[h]
                    # ax.plot(x, f0[k])
                    # ax.plot(x, f1[k])
                    ax.fill_between(x, f0[h], f1[h], color = color[h], alpha=0.5)
                    

        # plt.legend(bbox_to_anchor = [1.2,0.85])
        ax.set_xlabel('Month', size=12+5)
        interval = 30.4*2
        ax.set_xticks([i*interval for i in range(int(np.ceil(len(x)/interval)))])
        ax.set_xticklabels([int(interval/30.4*i) for i in range(int(np.ceil(len(x)/interval)))], size=10+5)
        ax.set_ylabel(r'$\frac{P(t)}{N(t)}$', size=12+5)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], size=10+5)
        ax.set_ylim([0,1.05])

    title_font_size = 18+5  # Adjust the title font size
    label_font_size = 14+5  # Adjust the axis labels' font size

fig.legend(bbox_to_anchor=(1.13, 0.57), ncol=1, title='Age ranges', fontsize=15, title_fontsize = 18)

# fig.text(-0.06, 0.5, f'Perceived Vaccine risk ($\\nu$)', ha='center', va='center', rotation=90, fontsize = title_font_size, weight='bold')
# fig.text(-0.02, 0.11, f'High\n($\\nu=${nu[2]})', ha='center', va='center', rotation=90,  fontsize = label_font_size, weight='bold')
# fig.text(-0.02, 0.49, f'Medium\n($\\nu=${nu[1]})', ha='center', va='center', rotation=90, fontsize = label_font_size, weight='bold')
# fig.text(-0.02, 0.87, f'Low\n($\\nu=${nu[0]})', ha='center', va='center', rotation=90, fontsize = label_font_size, weight='bold')

# fig.text(0.5, -0.1, f'Importance of emotional judgment ($\\rho$)', ha='center', va='center', rotation=0, fontsize = title_font_size, weight='bold')
# fig.text(0.13, -0.03, f'Rational\n($\\rho=0$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')
# fig.text(0.53, -0.03, f'Balanced\n($\\rho=0.5$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')
# fig.text(0.93, -0.03, f'Emotional\n($\\rho=1$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')

# plt.suptitle('Changes in pro-vaccination population with optimal campaign\n', fontsize = title_font_size, weight='bold')

plt.tight_layout()
# plt.savefig(f'Plot/Vacc_B_{B}.png', transparent=False, dpi=300, bbox_inches='tight')
plt.show()


# %% Tornado plot
colors = sns.color_palette("tab10", n_colors= 5)
custom_cmap = ListedColormap(colors)
color = [custom_cmap(i) for i in range(5)]
large_sa = ['vaccine_risk', 'p_emotional']
plot_df = glob_df[(glob_df['vaccine_risk'].isin([0, 1.5e-4, 3e-4])) & (glob_df['p_emotional'].isin([0, 0.5, 1])) 
                  & (glob_df['k_R'] == 20000) & (glob_df['k_E'] == 12) & (glob_df['source'] == 'cost_vacc')
                  & (glob_df['B'].isin([10000, 20000, 50000]))]
# obj_outcome = 'cost_vacc_0'
for obj_outcome in obj_outcome_list:
    outcome = obj_outcome
    df = plot_df[(plot_df['k_R'] == 20000) & (plot_df['k_E'] == 12) & (plot_df['obj'] == obj_outcome)]
    vmin = df[outcome].min()
    vmax = df[outcome].max()
    if outcome == 'cost_vacc_0':  
        vmin/=1e4; 
        vmax/=1e4
    num_to_plot = 1
    fig, ax = plt.subplots(num_to_plot, num_to_plot, figsize=(8, 6))
    count = 0
    glob_data_points = []
    yticklabels = []
    for i, (sa1, sa2) in enumerate(plot_df[large_sa].drop_duplicates().values[::-1]):
        # ax = axes[count // num_to_plot, count % num_to_plot]
        filtered_data = df[(df[large_sa[0]] == sa1) & (df[large_sa[1]] == sa2)].sort_values('B')[outcome].values/num_pop
        base = 0
        low = (filtered_data[0] - filtered_data[1])*100
        high = (filtered_data[2] - filtered_data[1])*100
        # low = (filtered_data[0] - filtered_data[1])/filtered_data[1]*100
        # high = (filtered_data[2] - filtered_data[1])/filtered_data[1]*100

        data_points = [low, base, high]
        print(sa1, sa2, data_points)
        glob_data_points.append(data_points)
        plt.barh([i,i,i], data_points, color=[color[1] if x < 0 else color[0] for x in data_points])
        count += 1
        yticklabels.append(f'({chr(97 + 8 - i)}) $\\nu={sa1}$, $\\rho={sa2}$')

    plt.yticks(np.arange(9), yticklabels, fontsize = title_font_size-5)
    plt.axvline(x=0, linewidth = 2.5, color='black')

    glob_data_points = np.array(glob_data_points)
    ax.text(glob_data_points.min()*1.1, 5, f'C = 10,000', ha='center', va='center', rotation=0, fontsize=title_font_size-7, color=color[1], weight='bold')
    ax.text(0, 9, f'C = 20,000', ha='center', va='center', rotation=0, fontsize=title_font_size-7, color='black', weight='bold')
    ax.text(glob_data_points.max()*1.15, 5, f'C = 50,000', ha='center', va='center', rotation=0, fontsize=title_font_size-7, color=color[0], weight='bold')
    ax.set_xlim(glob_data_points.min()*1.5, glob_data_points.max()*1.5)
    ax.set_xlabel('\n Changes in vaccinated population percentage (%)', size = label_font_size-2)
    # ax.set_xticks([-100, 0, 200, 400, 600, 800, 1000, 1200])
    # ax.set_xticklabels([-100, 0, 200, 400, 600, 800, 1000, 1200])
    # ax.text(420, 9.7, 'Budget Impact on Vaccination Uptake', fontsize = title_font_size, ha='center')
    plt.tight_layout()
    # plt.savefig(f'Plot/Tornado_Cost.png',bbox_inches='tight')
    plt.show()


# %%
# Plot eta
# B_list = [20000]
vary = ['k_R','k_E']
vary = ['vaccine_risk', 'p_emotional']

title_font_size = 14+2  # Adjust the title font size
label_font_size = 10+2  # Adjust the axis labels' font size
tick_font_size = 10

for B_to_plot in [20000]:
    plot_df = glob_df[glob_df['B']==B_to_plot]
    plot_df = plot_df[(plot_df['vaccine_risk'].isin([0, 1.5e-4, 3e-4])) & (plot_df['p_emotional'].isin([0, 0.5, 1]))]
    plot_df = plot_df.sort_values(vary)

    num_row = len(plot_df[vary[0]].unique())
    num_col = len(plot_df[vary[1]].unique())
    fig, axes = plt.subplots(num_row, num_col, figsize=(4+3.5*num_col, 1.5*num_row))

    plt.rcdefaults()
    for k in range(num_row*num_col):
        ax = axes[k//num_col, k%num_col]
        res = np.array(plot_df.alloc.iloc[k]).reshape((5,5))
        region = np.sum(res, axis=1)/B_to_plot
        age = np.sum(res, axis=0).T/B_to_plot
        sns.heatmap([np.concatenate((region, age), axis=0)], ax=ax, cmap='Blues', cbar=False, vmax=1, annot=True, fmt='.0%', 
                    annot_kws={'fontsize': tick_font_size+1})
        ax.axvline(x=5, color='black', linestyle='-', linewidth = 2.5)
        numbers  = [i for i in range(1,6)]
        x_tick_labels = [f'{i}\nRegion' if i == 3 else f'{i}' for i in numbers] + [f'{age_label[i-1]}\nAge' if i == 3 else f'{age_label[i-1]}' for i in numbers] 
        ax.set_xticks(np.arange(len(x_tick_labels))+0.5, x_tick_labels, rotation=0, fontsize = tick_font_size)
        ax.set_yticks([])

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f'({chr(97 + k)}) $\\nu={nu[k//num_col]}$, $\\rho={rho[k%num_col]}$', fontsize = label_font_size)

    # fig.legend(bbox_to_anchor=(1.25, 0.6), ncol=1, title='Group')

    fig.text(-0.06, 0.5, f'Perceived Vaccine risk ($\\nu$)', ha='center', va='center', rotation=90, fontsize = title_font_size, weight='bold')
    fig.text(-0.02, 0.18, f'High\n($\\nu=${nu[2]})', ha='center', va='center', rotation=90,  fontsize = label_font_size, weight='bold')
    fig.text(-0.02, 0.47, f'Medium\n($\\nu=${nu[1]})', ha='center', va='center', rotation=90, fontsize = label_font_size, weight='bold')
    fig.text(-0.02, 0.8, f'Low\n($\\nu=${nu[0]})', ha='center', va='center', rotation=90, fontsize = label_font_size, weight='bold')

    fig.text(0.5, -0.1, f'Importance of emotional judgment ($\\rho$)', ha='center', va='center', rotation=0, fontsize = title_font_size, weight='bold')
    fig.text(0.17, -0.03, f'Rational ($\\rho=0$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')
    fig.text(0.5, -0.03, f'Balanced ($\\rho=0.5$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')
    fig.text(0.83, -0.03, f'Emotional ($\\rho=1$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')

    plt.suptitle('Optimal Campaign Allocation by Region and by Age', fontsize = title_font_size, weight='bold')

    plt.tight_layout()
    plt.savefig(f'Plot/Alloc_B_{B_to_plot}.png', transparent=False, dpi=200, bbox_inches='tight')
    plt.show()
#%%
num_row = 3
num_col = num_row
# B_ind = 1
plt.rcdefaults()
day_shorten_list = []
# color = ['grey','']

title_font_size = 18+2  # Adjust the title font size
label_font_size = 14+2  # Adjust the axis labels' font size
for B in [20000]:
    B_ind = 1
    # B_ind = np.where(B_list == B)[0][0]
    # if B_ind is not 2: continue
    fig, axes = plt.subplots(num_row, num_col, figsize=(4+3*num_col, 3*num_row))
    for k in range(num_row*num_col):
        ax = axes[k//num_col, k%num_col]    
        # print(sa_list[k][4:])
        eta_per_group = []
        best_alloc = glob_best_alloc_list[B_ind][k][15:20]

        herd_day_index = []
        for i in range(2):
            [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(glob_ret_list[B_ind][2*k+i]), (len(glob_ret_list[B_ind][0]), 25, 8)))
            A = SA + IA + RA
            P = SP + IP + RP
            N = A + P
            I = IA + IP
            y = np.sum(P, axis=0)/np.sum(N, axis=0)
            x = np.array([i for i in range(len(y))])
            eta_per_group.append([x, y])
            labels = ['No campaign', 'Optimal campaign']

        color = ['grey','blue']
        for i, eta in enumerate(eta_per_group):
            x, y = eta
            ax.plot(x, y.T, 
                    linewidth = 2.5, 
                    color = color[i], 
                    alpha =1,
                    linestyle = 'dashed' if i%2==0 else '-',
                    label = labels[i%2] if k==0 else None)
            if i%2==1:
                f0 = interp1d(eta_per_group[0][0], (eta_per_group[0][1]).T, kind='linear', fill_value='extrapolate')(x)
                f1 = interp1d(eta_per_group[1][0], (eta_per_group[1][1]).T,  kind='linear', fill_value='extrapolate')(x)
                cumul_benefit = np.zeros((f0[0]).shape)
                ax.fill_between(x, f0, f1, color = 'skyblue', alpha=0.5)
                ax.text(x[-1], 0.1, f'Total increase: {round((f1-f0)[-1]*100,1)}%', fontsize=label_font_size, ha='right', color='red')
                ax.set_title(f'({chr(97 + k)}) $\\nu={nu[k//num_col]}$, $\\rho={rho[k%num_col]}$', fontsize=title_font_size-5)

        ax.set_xlabel('Month', size=12+5)
        interval = 30.4*2
        ax.set_xticks([i*interval for i in range(int(np.ceil(len(x)/interval)))])
        ax.set_xticklabels([int(interval/30.4*i) for i in range(int(np.ceil(len(x)/interval)))], size=10+5)
        # ax.set_ylabel(r'$\frac{P(t)}{N(t)}$', size=12+5)
        # ax.set_ylabel(r'$\frac{P(t)}{N(t)}$', size=label_font_size+5)
        ax.set_ylabel('P(t)/N(t)', size=label_font_size-3)
        # ax.set_ylabel(r'$\frac{\sum_{{i=1}}^{{n}} P_i(t)}{\sum_{{i=1}}^{{n}} N_i(t)}$', size=label_font_size+5)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], size=10+5)
        # ax.set_ylim([0,1.05])

    fig.legend(bbox_to_anchor=(1.23, 0.57), ncol=1, title='Campaign', fontsize=15, title_fontsize = 18)

    fig.text(-0.06, 0.5, f'Perceived Vaccine risk ($\\nu$)', ha='center', va='center', rotation=90, fontsize = title_font_size, weight='bold')
    fig.text(-0.02, 0.17, f'High\n($\\nu=${nu[2]})', ha='center', va='center', rotation=90,  fontsize = label_font_size, weight='bold')
    fig.text(-0.02, 0.49, f'Medium\n($\\nu=${nu[1]})', ha='center', va='center', rotation=90, fontsize = label_font_size, weight='bold')
    fig.text(-0.02, 0.78, f'Low\n($\\nu=${nu[0]})', ha='center', va='center', rotation=90, fontsize = label_font_size, weight='bold')

    fig.text(0.5, -0.1, f'Importance of emotional judgment ($\\rho$)', ha='center', va='center', rotation=0, fontsize = title_font_size, weight='bold')
    fig.text(0.2, -0.03, f'Rational\n($\\rho=0$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')
    fig.text(0.53, -0.03, f'Balanced\n($\\rho=0.5$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')
    fig.text(0.86, -0.03, f'Emotional\n($\\rho=1$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')

    plt.suptitle('Changes in vaccinated population with optimal campaign\n', fontsize = title_font_size, weight='bold')

    plt.tight_layout()
    # plt.savefig(f'Plot/Vacc_B_{B}.png', transparent=False, dpi=300, bbox_inches='tight')
    plt.show()
# %%
org = pd.read_pickle( f'Refined_result/refined_sa_0225_base_king.pkl')
krke = pd.read_pickle( f'Refined_result/refined_sa_0225_kr_ke_king.pkl')
deaths = pd.read_pickle( f'Refined_result/refined_sa_0225_deaths_king.pkl')
fb = pd.read_pickle( f'Refined_result/refined_sa_0225_fb_king.pkl')
disp = pd.read_pickle( f'Refined_result/refined_sa_0225_disparity_king.pkl')

org['source'] = 'base'
krke['source'] = 'krke'
fb['source'] = 'fb'
deaths['source'] = 'obj_deaths'
disp['source'] = 'obj_disp'

final_df = pd.concat([org, krke])
final_df = pd.concat([final_df, fb])
final_df = pd.concat([final_df, deaths])
final_df = pd.concat([final_df, disp])
final_df.reset_index()
final_df.to_pickle(f'Refined_result/refined_sa_0225_all_king.pkl')
with np.printoptions(linewidth=10000):
    final_df.to_csv(f'Refined_result/refined_sa_0225_all_king.csv')
# %%
    
def divide_alloc_results(alloc_list):
    final_list = []
    for alloc in alloc_list:
        res = np.array(alloc).reshape((5,5))
        B = np.sum(res)
        region = np.sum(res, axis=1)/B
        age = np.sum(res, axis=0).T/B
        final_list.append([np.concatenate((region, age), axis=0)])
    return (final_list)

def get_tornado(val_list):
    base = 0
    low = (val_list[0] - val_list[1])
    high = (val_list[2] - val_list[1])
    # return ([[low, base,0],[0,0,high]])
    return ([[low],[high]])

final_df = pd.read_pickle('Refined_result/refined_sa_0225_all_king.pkl')
sa_df = final_df[(final_df['vaccine_risk']==0.00015) & (final_df['p_emotional']==0.5) ]
base_row = sa_df[(sa_df['k_R']==20000) & (sa_df['k_E']==12) & (sa_df['source']=='base') & (sa_df['B']==20000)]
B_row = sa_df[(sa_df['k_R']==20000) & (sa_df['k_E']==12) & (sa_df['source']=='base')].sort_values('B')
kr_row = sa_df[(sa_df['k_E']==12) & (sa_df['source']=='krke')].sort_values('k_R')
ke_row = sa_df[(sa_df['k_R']==20000) & (sa_df['source']=='krke')].sort_values('k_E')
death_row = sa_df[(sa_df['source']=='obj_deaths')]
disp_row = sa_df[(sa_df['source']=='obj_disp')]
fb_row = sa_df[(sa_df['source']=='fb')]

dfs = [pd.DataFrame(row) for row in [base_row, B_row, kr_row, ke_row, death_row, disp_row]]
sa_df_final = pd.concat(dfs, ignore_index=True)
sa_df_final.to_pickle(f'Refined_result/refined_sa_0225_sa_king.pkl')
with np.printoptions(linewidth=10000):
    sa_df_final.to_csv(f'Refined_result/refined_sa_0225_sa_king.csv')

#%%
base_val = base_row['cost_vacc_0'].values[0]/num_pop
base_alloc = np.array(base_row['alloc'].values[0]).reshape((5,5))

# These have lower and upper bound
B_val = B_row['cost_vacc_0'].values/num_pop
kr_val = kr_row['cost_vacc_0'].values/num_pop
ke_val =ke_row['cost_vacc_0'].values/num_pop
death_val = death_row['cost_vacc_0'].values/num_pop
disp_val = disp_row['cost_vacc_0'].values/num_pop
fb_val = fb_row['cost_vacc_0'].values/num_pop

val_list = [[[0]]]
val_list.append(get_tornado(B_val))
val_list.append(get_tornado(kr_val))
val_list.append(get_tornado(ke_val))
val_list.append([[fb_val[0]-base_val]])
val_list.append([[death_val[0] - base_val]])
val_list.append([[disp_val[0] - base_val]])

val_list = np.concatenate([np.concatenate(sublist, axis=0) for sublist in val_list], axis=0).reshape((10,1))[::-1]*100

B_alloc = B_row.sort_values('B')['alloc'].values
kr_alloc = kr_row.sort_values('k_R')['alloc'].values
ke_alloc = ke_row.sort_values('k_E')['alloc'].values
fb_alloc = fb_row['alloc'].values
death_alloc = death_row['alloc'].values
disp_alloc = disp_row['alloc'].values

sa_alloc = []
sa_alloc.append(divide_alloc_results(B_alloc))
sa_alloc.append(divide_alloc_results(kr_alloc))
sa_alloc.append(divide_alloc_results(ke_alloc))
sa_alloc.append(divide_alloc_results(fb_alloc))
sa_alloc.append(divide_alloc_results(death_alloc))
sa_alloc.append(divide_alloc_results(disp_alloc))
sa_alloc = np.concatenate([np.concatenate(sublist, axis=0) for sublist in sa_alloc], axis=0)

#%%
import matplotlib.colors as mcolors

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
yticklabels = ['Base Case','C=10,000','C=50,000','High rational sensitivity (k_R = 1/15000)','Low rational sensitivity (k_R = 1/25000)',
               'High emotional sensitivity(k_E = 1/9)','Low emotional sensitivity (k_E = 1/15)'
               ,f'Use facebook data for $O_{{ij}}$', 'Objective: Avert deaths','Objective: Max-min vaccine increase (%)'][::-1]

for i, val in enumerate(val_list):
    print(i, val)
    plt.barh([i,i,i], val, color=[color[1] if x < 0 else color[0] for x in val])

plt.yticks(np.arange(len(yticklabels)), yticklabels, fontsize = title_font_size-10)
plt.axvline(x=0, linewidth = 2.5, color='black')

ax.set_xlim(val_list.min()*1.5, val_list.max()*1.5)
ax.set_xlabel('Absolute Percentage Change (%)', size = label_font_size-5)
ax.set_title('Impact of Parameter Changes on Vaccinated Population', size=title_font_size-5, pad=15)
plt.tight_layout()
plt.savefig(f'Plot/Tornado_Cost_SA.png',bbox_inches='tight')
plt.show()
#%%
colors = sns.color_palette("tab10", n_colors= 2)
custom_cmap = ListedColormap(colors)
color = [custom_cmap(i) for i in range(2)]

fig, axes = plt.subplots(len(yticklabels), 1, figsize=(10, 6))
# yticklabels = ['Base case', 'C=10,000','C=50,000','High rational sensitivity\n(k_R = 1/15000)','Low rational sensitivity\n(k_R = 1/25000)',
            #    'High emotional sensitivity\n(k_E = 1/9)','Low emotional sensitivity\n(k_E = 1/15)','Objective: Avert deaths', f'Use facebook data for $O_{{ij}}$']
plt.rcdefaults()
for k in range(len(yticklabels)):
    ax = axes[k]
    sns.heatmap([np.concatenate((val_list[len(yticklabels) -1-k]/100, sa_alloc[k]))] , ax=ax, cmap='Blues', cbar=False, vmin=0, vmax=1, annot=True, fmt='.1%')
    ax.axvline(x=5+1, color='black', linestyle='-', linewidth = 2.5)
    ax.axvline(x=1, color='white', linestyle='-', linewidth = 10)
    numbers  = [i for i in range(1,6)]
    # x_tick_labels = ['']
    x_tick_labels = [f'{i}\nRegion' if i == 3 else f'{i}' for i in numbers] + [f'{age_label[i-1]}\nAge' if i == 3 else f'{age_label[i-1]}' for i in numbers] 
    if k==len(yticklabels) -1: ax.set_xticks(np.arange(len(x_tick_labels))+1.5, x_tick_labels, rotation=0)
    else:ax.set_xticks([])
    ax.set_yticks([0.5])
    ax.set_yticklabels([yticklabels[len(yticklabels) -1 - k]], rotation=0)
   
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    color_val = val_list[len(yticklabels) -1-k]/100
    cmap = mcolors.LinearSegmentedColormap.from_list('coolwarm', ['orangered', 'white','#1E90FF'])
    cmap = mcolors.LinearSegmentedColormap.from_list('coolwarm', [color[1], 'white',color[0]])
    # cmap = plt.cm.get_cmap('coolwarm')
    rect_color = cmap(0.5 + color_val * 5)  # Adjust scaling for visual sensitivity
    rect = plt.Rectangle((x_min, y_min), 1, y_max - y_min, linewidth=2, edgecolor=None, facecolor=rect_color, alpha=1.0)
    ax.add_patch(rect)
    if k==0:
        ax.text(0.5, -0.75, 'Vaccination\npopulation (%)', ha='center', va='center', rotation=0, fontsize=label_font_size-7, color='black')
        ax.text(6, -0.5, 'Optimal allocation by Region and Age', ha='center', va='center', rotation=0, fontsize=label_font_size-3, color='black')

plt.suptitle('Sensitivity Analysis', fontsize = title_font_size-5, weight='bold')

plt.tight_layout()
# plt.savefig(f'Plot/SA_all.png', transparent=False, dpi=200, bbox_inches='tight')
plt.show()# %%

# %%
fig, axes = plt.subplots(len(yticklabels), 1, figsize=(8, 6))
plt.rcdefaults()
for k in range(len(yticklabels)):
    ax = axes[k]
    sns.heatmap([sa_alloc[k]], ax=ax, cmap='Blues', cbar=False, vmin=0, vmax=1, annot=True, fmt='.0%')
    ax.axvline(x=5, color='black', linestyle='-', linewidth = 2.5)
    # ax.axvline(x=1, color='white', linestyle='-', linewidth = 10)
    numbers  = [i for i in range(1,6)]
    # x_tick_labels = ['']
    x_tick_labels = [f'{i}\nRegion' if i == 3 else f'{i}' for i in numbers] + [f'{age_label[i-1]}\nAge' if i == 3 else f'{age_label[i-1]}' for i in numbers] 
    if k==len(yticklabels)-1: ax.set_xticks(np.arange(len(x_tick_labels))+0.5, x_tick_labels, rotation=0)
    else:ax.set_xticks([])
    ax.set_yticks([0.5])
    ax.set_yticklabels([yticklabels[len(yticklabels)-1-k]], rotation=0)
   
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=4, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

plt.suptitle('Sensitivity Analysis', fontsize = title_font_size-5, weight='bold')

plt.tight_layout()
plt.savefig(f'Plot/SA_alloc.png', transparent=False, dpi=200, bbox_inches='tight')
plt.show()# %%

# %%
base_case_ben_pop = [ 0.48451373,  1.17271271,  1.18432927,  2.33219056,  2.90206302,
        1.03852887,  2.51255127,  2.79855855,  8.65490584,  5.53445266,
        0.49688024,  1.17817695,  1.19118277,  2.32078399,  2.94551917,
        5.40342726, 52.37817862, 50.63518284, 15.29370277,  8.80442531,
        1.97154333,  5.49670695, 23.67500849, 18.2803904 ,  6.01915431]
disp_case_ben_pop = sa_df[(sa_df['source']=='obj_disp')]['benefits_vacc_per_pop'].values[0]*100
base_case_ben_pop= reorder_list(base_case_ben_pop, re_order)
disp_case_ben_pop= reorder_list(disp_case_ben_pop, re_order)
fig, ax = plt.subplots(figsize=(15, 1.5))
sns.heatmap([base_case_ben_pop, disp_case_ben_pop], cmap='Blues',annot=True, fmt='.1f', linewidth=1, ax=ax)
ax.set_yticklabels(['base','disparity'],rotation=0)
plt.tight_layout()
plt.savefig(f'Plot/disparity_base_comparison.png', transparent=False, dpi=200, bbox_inches='tight')
plt.show()# %%

# %% Policy comparison

B_list = [20000]
policy_list = [3,4,5]
sa_list = get_mid_sa()
sa_list = [sa_list[i] for i in policy_list]
glob_ret_list = []
glob_eta_list = []
glob_all_eta_list = []
best_alloc_list = []
for B in B_list:
    for sa in sa_list:
        init_param_list = sa.copy()
        alc = Alloc(fips_num = 53033, obj_type = 'all', alg='reg_age', 
                    B=B, init_param_list  = init_param_list)
        row = glob_df[(glob_df['p_emotional'] == sa[0][1]) &
                            (glob_df['vaccine_risk'] == sa[1][1]) &
                            (glob_df['k_R'] == sa[2][1]) &
                            (glob_df['k_E'] == sa[3][1]) &
                            (glob_df['B'] == B) &
                            (glob_df['source']== 'krke') &
                            (glob_df['obj'] == obj)]['org_alloc']
        if len(row)>0:
            print(len(row))
            best_alloc = row.iloc[0]
            best_alloc_list.append(best_alloc)
for B in B_list:
    ret_list = []
    eta_list = []
    all_eta_list = []          
    for sa in sa_list:
        init_param_list = sa.copy()
        alc = Alloc(fips_num = 53033, obj_type = 'all', alg='reg_age', 
                    B=B, init_param_list  = init_param_list)
        alloc_test = [np.zeros(25)] + [alloc / np.sum(alloc) * B for alloc in best_alloc_list]
        alloc_test = np.unique(np.array(alloc_test), axis=0)
        print(alloc_test)
        rets,etas,all_etas = alc.run_code(parallel=False, alloc_list = alloc_test, save_result = True)
        for ret in rets: ret_list.append(ret)
        for eta in etas: eta_list.append(eta)
        for all_eta in all_etas: all_eta_list.append(all_eta)
    glob_ret_list.append(ret_list)
    glob_eta_list.append(eta_list)
    glob_all_eta_list.append(all_eta_list)
# %%
