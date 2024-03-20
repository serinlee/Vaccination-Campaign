#####################################################################
# A Module that analyzes sa_alloc results
#####################################################################

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
from scipy.interpolate import interp1d
import plot
import json
import pickle

colors = sns.color_palette("tab10", n_colors= 5)
custom_cmap = ListedColormap(colors)
color = [custom_cmap(i) for i in range(5)]
marker = ['o','s','D','^','v']*5
title_font_size = 20
label_font_size = 16
tick_font_size = 14
age_label = ['0-17','18-34','35-49','50-64','65+']
reg_label = ['Seattle','Seattle\nEast','FAV','ITM', 'ES']
group_label  = ['1,1','1,2','1,3','1,4','1,5']
sa_param = ['vaccine_risk', 'p_emotional','k_R', 'k_E']
mapping = {'cost_vacc_0': 'tot_benefits_vacc', 'cost_deaths_0': 'tot_benefits_deaths', 'disparity_vacc_0':'min_benefits_vacc_per_pop' }
num_pop = 2195285
# %% read all data and combine into one
re_order = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4] # function needed to sort results by population size
def reorder_list(lst, re_order):
    return [lst[i] for i in re_order]

def unpack_tuples(row):
    return {key: value for key, value in row}

def clean_data(date = '0220'):
    import os
    folder_path = 'Results/SA_Result'
    file_pattern = f'*{date}*.pkl'

    matching_files = glob.glob(os.path.join(folder_path, file_pattern))
    dfs = []
    for file_path in matching_files:
        print(file_path)
        df = pd.read_pickle(file_path)
        df['B'] = int(file_path.split('B')[-1].split('_')[1])
        df['source'] = file_path.split('_')[-1].split('.')[0]
        dfs.append(df)
        df['obj'] = df['obj'].replace(mapping)
        df = df.rename(columns=mapping)
    glob_df = pd.concat(dfs, ignore_index=True)
    glob_df['param_update'] = glob_df['param_update'].apply(lambda x: ast.literal_eval(x))
    pd.concat([glob_df['param_update'].apply(lambda x: unpack_tuples(x)).apply(pd.Series)
    , glob_df],  axis=1)
    glob_df = pd.concat([glob_df['param_update'].apply(lambda x: unpack_tuples(x)).apply(pd.Series)
    , glob_df],  axis=1)

    glob_df['org_alloc'] = glob_df['alloc'].copy() # Keep the original alloc results
    glob_df['alloc'] = glob_df['alloc'].apply(lambda x: reorder_list(x, re_order))
    glob_df.to_csv(f'Results/Refined_result/refined_sa_{date}_king.csv')
    glob_df.to_pickle(f'Results/Refined_result/refined_sa_{date}_king.pkl')

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

#%% Read data
    
date='0225_base'
clean_data(date)
filename = f'Results/Refined_result/refined_sa_{date}_king.pkl'

glob_df = pd.read_pickle(filename)
glob_df = glob_df[(glob_df['vaccine_risk'].isin([0, 1.5e-4, 3e-4])) & (glob_df['p_emotional'].isin([0, 0.5, 1]))]
obj = 'tot_benefits_vacc'
obj_outcome_list = [obj]
outcome_list = obj_outcome_list+['max_alloc']
nu = [round(i,5) for i in glob_df['vaccine_risk'].unique()]
rho = glob_df['p_emotional'].unique()
glob_df = glob_df.sort_values(by=['B']+sa_param, ascending=True)
B_list = glob_df.B.unique()
sa_list = glob_df['param_update'].values

#%% Run model to get more detailed results
B_list = [20000]
glob_ret_list, glob_eta_list, glob_all_eta_list, glob_best_alloc_list =  run_model(B_list, sa_list, glob_df)

#%% Plot figure 3
num_row = 3
num_col = num_row
plt.rcdefaults()

title_font_size = 20
label_font_size = 15
labels = ['No campaign', 'Optimal campaign']
color = ['grey','blue']

for B in [20000]:
    B_ind = B_list.index(B)
    fig, axes = plt.subplots(num_row, num_col, figsize=(4+3*num_col, 3*num_row))
    for k in range(num_row*num_col):
        ax = axes[k//num_col, k%num_col]    
        eta_per_group = []
        for i in range(2):
            [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(glob_ret_list[B_ind][2*k+i]), (len(glob_ret_list[B_ind][0]), 25, 8)))
            A = SA + IA + RA
            P = SP + IP + RP
            N = A + P
            I = IA + IP
            y = np.sum(P, axis=0)/np.sum(N, axis=0)
            x = np.array([i for i in range(len(y))])
            eta_per_group.append([x, y])
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
        ax.set_xticklabels([int(interval/30.4*i) for i in range(int(np.ceil(len(x)/interval)))], size=15)
        ax.set_ylabel('P(t)/N(t)', size=label_font_size-3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], size=15)

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
    # plt.savefig(f'Results/Plot/Vacc_B_{B}.png', transparent=False, dpi=300, bbox_inches='tight')
    plt.show()

#%% Plot figure 4
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
    # plt.savefig(f'Results/Plot/Alloc_B_{B_to_plot}.png', transparent=False, dpi=200, bbox_inches='tight')
    plt.show()
#%%
################
# One-way sensitivity analysis
################

org = pd.read_pickle( f'Results/Refined_result/refined_sa_0225_base_king.pkl')
krke = pd.read_pickle( f'Results/Refined_result/refined_sa_0225_kr_ke_king.pkl')
deaths = pd.read_pickle( f'Results/Refined_result/refined_sa_0225_deaths_king.pkl')
fb = pd.read_pickle( f'Results/Refined_result/refined_sa_0225_fb_king.pkl')
disp = pd.read_pickle( f'Results/Refined_result/refined_sa_0225_disparity_king.pkl')

# Combine to single df
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
final_df = final_df.rename(columns=mapping)
final_df['obj'] = final_df['obj'].replace(mapping)
final_df.to_pickle(f'Results/Refined_result/refined_sa_0225_all_king.pkl')
with np.printoptions(linewidth=10000):
    final_df.to_csv(f'Results/Refined_result/refined_sa_0225_all_king.csv')

# Select rows
final_df = pd.read_pickle('Results/Refined_result/refined_sa_0225_all_king.pkl')
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
sa_df_final.to_pickle(f'Results/Refined_result/refined_sa_0225_sa_king.pkl')
with np.printoptions(linewidth=10000):
    sa_df_final.to_csv(f'Results/Refined_result/refined_sa_0225_sa_king.csv')

base_val = base_row['tot_benefits_vacc'].values[0]/num_pop
base_alloc = np.array(base_row['alloc'].values[0]).reshape((5,5))

B_val = B_row['tot_benefits_vacc'].values/num_pop
kr_val = kr_row['tot_benefits_vacc'].values/num_pop
ke_val =ke_row['tot_benefits_vacc'].values/num_pop
death_val = death_row['tot_benefits_vacc'].values/num_pop
disp_val = disp_row['tot_benefits_vacc'].values/num_pop
fb_val = fb_row['tot_benefits_vacc'].values/num_pop

# Prepare for tornado plot results (Figure 5)
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
    return ([[low],[high]])

val_list = [[[0]]]
val_list.append(get_tornado(B_val))
val_list.append(get_tornado(kr_val))
val_list.append(get_tornado(ke_val))
val_list.append([[fb_val[0]-base_val]])
val_list.append([[death_val[0] - base_val]])
val_list.append([[disp_val[0] - base_val]])

val_list = np.concatenate([np.concatenate(sublist, axis=0) for sublist in val_list], axis=0).reshape((10,1))[::-1]*100

# Prepare for allocation results (Figure 5)
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

#%% Plot tornado plot
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
plt.savefig(f'Results/Plot/Tornado_SA.png',bbox_inches='tight')
plt.show()

#%% Plot allocation results - version 1
colors = sns.color_palette("tab10", n_colors= 2)
custom_cmap = ListedColormap(colors)
color = [custom_cmap(i) for i in range(2)]

fig, axes = plt.subplots(len(yticklabels), 1, figsize=(10, 6))
plt.rcdefaults()
for k in range(len(yticklabels)):
    ax = axes[k]
    sns.heatmap([np.concatenate((val_list[len(yticklabels) -1-k]/100, sa_alloc[k]))] , ax=ax, cmap='Blues', cbar=False, vmin=0, vmax=1, annot=True, fmt='.0%')
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
plt.savefig(f'Results/Plot/SA_all.png', transparent=False, dpi=200, bbox_inches='tight')
plt.show()

# %% Plot allocation results - version 2
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
plt.savefig(f'Results/Plot/SA_alloc.png', transparent=False, dpi=200, bbox_inches='tight')
plt.show()
