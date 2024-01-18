#%% import settings

from alloc import Alloc
import numpy as np
import pandas as pd
import glob
import ast
import seaborn as sns
import matplotlib.pyplot as plt

import plot
from itertools import product
from scipy.interpolate import interp1d

def get_full_way_sa(fips=53033):
    vr_list = np.linspace(0.0, 0.0003, 3)
    p_list = np.linspace(0,1,3) # Fix!!! 
    kr_list = np.linspace(15000, 25000, 5)
    ke_list = np.linspace(14, 16, 5) # Fix!!!

    sa_list = []
    for p in p_list:
        for k_R in kr_list:
            for k_E in ke_list:
                for vr in vr_list:
                    init_param_list = [("p1", p),("p2", p),("p3",p),("p4", p),("p5", p), 
                                       ("vaccine_risk", vr), ("k_R",k_R), ("k_E", k_E)]
                    sa_list.append(init_param_list)
    return sa_list


def get_test_sa():
    vr_list = np.linspace(0.0, 0.0003, 3)
    p_list = np.linspace(0,1,3) # Fix!!! 
    kr_list = np.linspace(15000, 25000, 3)
    ke_list = np.linspace(14, 16, 3) # Fix!!!
    sa_list = []

    # vr_list = [0.00015]
    # p_list = [1]
    kr_list = [20000]
    ke_list = [15]

    for vr in vr_list:
        for k_R in kr_list:
            for k_E in ke_list:
                for p in p_list:
                    init_param_list = [("p1", p),("p2", p),("p3",p),("p4", p),("p5", p), 
                                    ("vaccine_risk", vr), ("k_R",k_R), ("k_E", k_E)]
                    sa_list.append(init_param_list)
    return sa_list

def get_test_sa_fb():
    vr_list = np.linspace(0.0, 0.0003, 3)
    p_list = np.linspace(0,1,3) # Fix!!! 
    kr_list = np.linspace(15000, 25000, 3)
    ke_list = np.linspace(14, 16, 3) # Fix!!!
    sa_list = []

    # vr_list = [0.00015]
    # p_list = [1]
    kr_list = [20000]
    ke_list = [15]
    p_online_list = [0, 0.5, 1]

    for vr in vr_list:
        for k_R in kr_list:
            for k_E in ke_list:
                for p in p_list:
                    for p_online in p_online_list:
                        init_param_list = [("p1", p),("p2", p),("p3",p),("p4", p),("p5", p), 
                                        ("vaccine_risk", vr), ("k_R",k_R), ("k_E", k_E), ("p_online", p_online)]
                        sa_list.append(init_param_list)
    return sa_list

def get_combined_param_update_bounds():
    # Upper, lower bound, and number of trials
    combined_param_update_bounds = [
        [("vaccine_risk", 0, 0.0001, 11)],
        [("k_R", 0.1*5000, 30*5000, 11)],
        [("k_E", 0.1, 30, 11)],
        [("overall_alpha", 0.0001, 0.001, 11)],
        [("beta", 1, 10, 11)],
        [("p_online", 0, 1, 11)],
        [("lam", 0, 0.2, 11)],
        [("O_m", 1, 10, 11)],
        [("VE_beta", 0.0, 1.0, 11)],
        [("VE_death", 0.0, 1.0, 11)],
        [("rae", 100, 1000, 11)],
    ]
    return combined_param_update_bounds

def get_sa_list(param_bounds):
    param_name, lower_bound, upper_bound, num_trials = param_bounds[0]
    step = (upper_bound - lower_bound) / (num_trials - 1)
    sa_list = [(param_name, round(lower_bound + i * step, 5)) for i in range(num_trials)]
    return sa_list

def extract_values_from_filepath(filepath):
    pattern = r"SA_Result/result_sa_(\d+)_p_(\d+)_B_(\d+)_date_\d+"
    match = re.match(pattern, filepath)
    if match:
        sa_index = int(match.group(1))
        point_index = int(match.group(2))
        B = int(match.group(3))
        return point_index, sa_index, B
    else:
        return None
#%% Actually run
date = '1115'
num_alloc = 20

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fips_num', type=int, help='FIPS number (region)')
parser.add_argument('-p', '--point_index', type=int, help='Point index (calibration)')
parser.add_argument('-B', '--B', type=int, help='Budget')
parser.add_argument('-i', '--sa_index', type=int, help='SA index')
args = parser.parse_args()

global_top_df = pd.DataFrame()
# param_bounds = get_combined_param_update_bounds()[args.sa_index]
# sa_list = get_sa_list(param_bounds)
full_sa_list = get_full_way_sa()
sa_list = full_sa_list[args.sa_index::1]

for sa in sa_list:
    alc = Alloc(fips_num = args.fips_num, obj_type = 'all', alg='reg_age', 
                B=args.B, num_alloc = num_alloc, point_index = 0,
                init_param_list  = sa.copy())
    print(sa)
    alloc_test = alc.get_alloc_list()
    # alloc_test = alc.get_vertex_alloc(25, args.B)
    alc.run_code(parallel=True, alloc_list = alloc_test, save_result = False)
    outcome_list = ['cost_deaths_0','disparity_deaths_0','cost_vacc_0','disparity_vacc_0']
    outcome_list = ['cost_deaths_0','cost_vacc_0']
    for outcome_metric in outcome_list:
        (top_reg, top_age, alloc_test) = alc.get_alloc_top_age_reg(alc.sol_history, outcome_metric)
        alc = Alloc(fips_num = args.fips_num, init_param_list  = sa.copy(), obj_type = outcome_metric, 
                    alg='reg_age', B=args.B, num_alloc = num_alloc, point_index = args.point_index)
        alc.run_code(parallel=True, alloc_list = alloc_test)
        final_df = pd.concat([top_reg, top_age, alc.sol_history], ignore_index = True)
        # final_df = alc.sol_history
        top_df = final_df.sort_values(outcome_metric, ascending = False if 'cost' in outcome_metric else True).iloc[:1]
        top_df['obj'] = outcome_metric
        top_df['param_update'] = str(sa)
        global_top_df = pd.concat([global_top_df, top_df], ignore_index = True)

file_name = f'SA_Result/top_{args.fips_num}_sa_{args.sa_index}_p_{args.point_index}_B_{args.B}_date_{date}'
with np.printoptions(linewidth=10000):
    global_top_df.to_csv(file_name+'.csv')
global_top_df.to_pickle(file_name+'.pkl')


# %% read all data and combine into one
date=1115
folder_path = 'SA_Result'
file_pattern = f'*{date}_p_online_100.pkl'

matching_files = glob.glob(os.path.join(folder_path, file_pattern))
dfs = []
for file_path in matching_files:
    print(file_path)
    df = pd.read_pickle(file_path)
    # df['B'] = int(file_path.split('_')[-3])
    df['B'] = 10000
    dfs.append(df)
glob_df = pd.concat(dfs, ignore_index=True)

def unpack_tuples(row):
    return {key: value for key, value in row}

glob_df['param_update'] = glob_df['param_update'].apply(lambda x: ast.literal_eval(x))
pd.concat([glob_df['param_update'].apply(lambda x: unpack_tuples(x)).apply(pd.Series)
, glob_df],  axis=1)
glob_df = pd.concat([glob_df['param_update'].apply(lambda x: unpack_tuples(x)).apply(pd.Series)
, glob_df],  axis=1)
glob_df = glob_df.drop(columns = ['p2','p3','p4','p5'])
glob_df = glob_df.rename(columns={'p1': 'p_emotional'})

# Define the re_order alloc results
re_order = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4]
def reorder_list(lst, re_order):
    return [lst[i] for i in re_order]
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
glob_df.to_csv(f'refined_sa_{date}_king_p_online_100.csv')
glob_df.to_pickle(f'refined_sa_{date}_king_p_online_100.pkl')

#%% Functions to plot
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

def plot_eta(eta_list, label=None, title=''):        
    fig, ax = plt.subplots(figsize=(3, 2.5))
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
              '#bcbd22', '#17becf', '#1a9850', '#66a61e', '#a6cee3', '#fdbf6f', '#fb9a99', '#e31a1c', 
              '#fb9a99', '#33a02c', '#b2df8a', '#a6cee3']*300
    for i, eta in enumerate(eta_list):
        x, y = np.array(eta).T
        ax.plot(x, y, linewidth = 1, 
                color = color[i//2], alpha = 0.5 if i%2==1 else 1,
                linestyle = 'dashed' if i%2==0 else '-',
                label=label[i] if label is not None else None)
    
    f0 = interp1d(np.array(eta_list[0])[:, 0], np.array(eta_list[0])[:, 1], kind='linear', fill_value='extrapolate')
    f1 = interp1d(np.array(eta_list[1])[:, 0], np.array(eta_list[1])[:, 1], kind='linear', fill_value='extrapolate')
    f2 = interp1d(np.array(eta_list[2])[:, 0], np.array(eta_list[2])[:, 1], kind='linear', fill_value='extrapolate')
    f3 = interp1d(np.array(eta_list[3])[:, 0], np.array(eta_list[3])[:, 1], kind='linear', fill_value='extrapolate')
    f4 = interp1d(np.array(eta_list[4])[:, 0], np.array(eta_list[4])[:, 1], kind='linear', fill_value='extrapolate')
    f5 = interp1d(np.array(eta_list[5])[:, 0], np.array(eta_list[5])[:, 1], kind='linear', fill_value='extrapolate')


    ax.fill_between(x, f0(x), f1(x), color=color[0], alpha=0.3)
    ax.fill_between(x, f2(x), f3(x), color=color[1], alpha=0.3)
    ax.fill_between(x, f4(x), f5(x), color=color[2], alpha=0.3)

    ax.set_xlabel('Month', size=12)
    ax.set_xticks([i*33.8 for i in range(12)])
    ax.set_xticklabels([i+1 for i in range(12)], size=10)
    ax.set_ylabel(f'Opinion persuasiveness\n $\\overline{{\\eta}}(t$)', size=12)
    ax.set_ylim([0, 1.0])
    if label is not None:
        ax.legend(loc='center', bbox_to_anchor=(1.8, 0.5))
    # plt.tight_layout()
    plt.title(title)
    plt.show()
#%% Read data

date = 1115
p_online = 0
filename = f'refined_sa_{date}_king_p_online_{int(p_online*100)}.pkl'

glob_df = pd.read_pickle(filename)
obj_outcome_list = ['cost_vacc_0']
outcome_list = ['max_alloc']
outcome_list+=obj_outcome_list

large_sa = ['vaccine_risk', 'p_emotional']
within_sa = ['k_R', 'k_E']
nu = [0, '0.0015', '0.0003']
rho = [0, 0.5, 1]
glob_df = glob_df.sort_values(by=large_sa+within_sa, ascending=[True, True, False, False])

#%% Plot Figure 1 and 2

for B in [10000]:
    for obj_outcome in obj_outcome_list[:1]:
        for outcome in outcome_list:
            # if outcome not in ['max_alloc', obj_outcome]: continue
            print(obj_outcome, outcome)
            df = glob_df[glob_df['B']==B].copy()
            vmin = df[outcome].min()
            vmax = df[outcome].max()
            if outcome == 'cost_vacc_0':  vmin/=1e4; vmax/=1e4
            elif outcome == 'max_alloc' and 'cost' in obj_outcome:  vmin=0; vmax=5

            # Create a 5x5 grid of subplots
            num_to_plot = 3
            fig, axes = plt.subplots(num_to_plot, num_to_plot, figsize=(16, 12))

            # Increase font size for various plot elements
            title_font_size = 20  # Adjust the title font size
            label_font_size = 16  # Adjust the axis labels' font size
            tick_font_size = 14   # Adjust the tick labels' font size

            # Iterate over unique combinations of p_emotional and vaccine_risk
            count = 0
            for i, (sa1, sa2) in enumerate(df[large_sa].drop_duplicates().values):
                ax = axes[count // num_to_plot, count % num_to_plot]

                filtered_data = df[(df[large_sa[0]] == sa1) & (df[large_sa[1]] == sa2) & (df['obj'] == obj_outcome)]
                pivot_data = filtered_data.pivot_table(index=within_sa[0], columns=within_sa[1], values=outcome, aggfunc='first')
                # pivot_data = pivot_data.loc[::-1, ::-1]

                # print(pivot_data)
                if outcome == 'max_alloc':
                    annot_data = filtered_data.pivot_table(index=within_sa[0], columns=within_sa[1], values='max_alloc_list', aggfunc='first')
                    annot_data = annot_data.applymap(format_annotation_list)
                    sns.heatmap(pivot_data, annot=annot_data, fmt='', cmap='Greens',
                                ax=ax, vmin=vmin, vmax=vmax,  annot_kws={'fontsize': 15})
                else:
                    if outcome == 'cost_vacc_0': pivot_data /= 1e4
                    sns.heatmap(pivot_data, annot=True, fmt='.0f',
                                cmap='Blues_r' if 'disparity' in outcome else 'Blues',
                                ax=ax, vmin=vmin, vmax=vmax, annot_kws={'fontsize': tick_font_size})
                # ax.set_title(f'({chr(97 + count)}) $\\rho={sa1}$, $\\nu={sa2*1e4:.1f}$e-4', fontsize=title_font_size)
                ax.set_title(f'({chr(97 + count)}) $\\nu=${nu[i//3]}, $\\rho={sa2}$', fontsize=title_font_size)
                ax.invert_xaxis()
                ax.set_xticklabels(['High', '', '', '', 'Low'], fontsize=tick_font_size, rotation=0)
                ax.set_yticklabels(['High', '', '', '', 'Low'], fontsize=tick_font_size)
                ax.set_xlabel(f'Emotional sensitivity ($1/k_E$)', fontsize=label_font_size)
                ax.set_ylabel(f'Rational sensitivity ($1/k_R$)', fontsize=label_font_size)
                count += 1
            # sns.set(font_scale=2)
            # if outcome =='cost_vacc_0': title = f'Vaccination uptake increase $\\sum_{{i=1}}^{{n}} \\Delta P_i(b,x;T)$ (10,000s)'
            if outcome =='cost_vacc_0': title = f'Vaccination uptake increase (10,000s)'
            if outcome =='max_alloc': title = 'Most allocated group'
            fig.text(-0.06, 0.5, f'Perceived Vaccine risk ($\\nu$)', ha='center', va='center', rotation=90, weight='bold', fontsize=title_font_size+5)
            fig.text(-0.02, 0.17, f'High\n($\\nu=${nu[2]})', ha='center', va='center', rotation=90, weight='bold', fontsize=title_font_size)
            fig.text(-0.02, 0.5, f'Medium\n($\\nu=${nu[1]})', ha='center', va='center', rotation=90, weight='bold', fontsize=title_font_size)
            fig.text(-0.02, 0.8, f'Low\n($\\nu=${nu[0]})', ha='center', va='center', rotation=90, weight='bold', fontsize=title_font_size)

            fig.text(0.5, -0.08, f'Importance of emotional judgment ($\\rho$)', ha='center', va='center', rotation=0, weight='bold', fontsize=title_font_size+5)
            fig.text(0.17, -0.025, f'Rational\n($\\rho=0$)', ha='center', va='center', rotation=0, weight='bold', fontsize=title_font_size)
            fig.text(0.5, -0.025, f'Balanced\n($\\rho=0.5$)', ha='center', va='center', rotation=0, weight='bold', fontsize=title_font_size)
            fig.text(0.83, -0.025, f'Emotional\n($\\rho=1$)', ha='center', va='center', rotation=0, weight='bold', fontsize=title_font_size)

            plt.suptitle(title, fontsize = title_font_size+10, weight='bold')
            plt.tight_layout()
            plt.savefig(f'Plot/{obj_outcome}_{outcome}_B_{B}_fb_{int(p_online*100)}.png',bbox_inches='tight')
            plt.show()

#%%
obj = 'cost_vacc_0'
sa_list = get_test_sa()

condition = [f'$\\rho=0$', f'$\\rho=0.5$',f'$\\rho=1$']
policy = ['(Base)','(Best campaign)']

B_list = [10000]
glob_best_alloc_list = []
glob_ret_list= []
glob_eta_list = []
glob_eta_all_list = []
for B in B_list:
    best_alloc_list = []
    ret_list= []
    eta_list = []
    eta_all_list = []
    for sa in sa_list:
        init_param_list = sa.copy() + [('p_online', p_online)]
        alc = Alloc(fips_num = 53033, obj_type = 'all', alg='reg_age', 
                    B=B, num_alloc = 1, point_index = 0,
                    init_param_list  = init_param_list)
        best_alloc = glob_df[(glob_df['p_emotional'] == sa[0][1]) &
                            (glob_df['vaccine_risk'] == sa[5][1]) &
                            (glob_df['k_R'] == sa[6][1]) &
                            (glob_df['k_E'] == sa[7][1]) &
                            (glob_df['B'] == B) &
                            (glob_df['obj'] == obj)]['org_alloc'].iloc[0]
        alloc_test = [np.zeros(25), np.array(best_alloc)]
        best_alloc_list.append(best_alloc)
        rets,etas,all_etas = alc.run_code(parallel=False, alloc_list = alloc_test, save_result = True)
        for eta in etas: eta_list.append(eta)
        for eta_all in all_etas: eta_all_list.append(eta_all)
        for ret in rets: ret_list.append(ret)
        
    glob_best_alloc_list.append(best_alloc_list)
    glob_ret_list.append(ret_list)
    glob_eta_list.append(eta_list)
    glob_eta_all_list.append(eta_all_list)
if len(alloc_test) == 1:policy = policy[:1]
label = [f'{cond}{pol}' for cond, pol in product(condition, policy)]
# # label = [i+1 for i in range(len(eta_list))]
# plot_eta(eta_list, label= label)
# plot.plot_results_with_calib_one_plot(alc.model, alc.model.t_f, ret_list,to_plot='vacc', 
#                                       error_bar=True, lw=1.5, label = label, includedata=False)

# %%
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',  '#bcbd22', '#17becf', '#1a9850', '#66a61e', '#a6cee3', '#fdbf6f', '#fb9a99', '#e31a1c', '#fb9a99', '#33a02c', '#b2df8a', '#a6cee3']
marker = ['o','s','D','^','v']
age_label = ['under 17','18-34','35-49','50-64','64+']
group_label  = ['1,1','1,2','1,3','1,4','1,5']
nu = [0, '0.0015', '0.0003']
rho = [0, 0.5, 1]

B = 10000 #0 is C=2000, 1 is C=10K, 2 is C=50K
B_ind = B_list.index(B)

# Plot eta
fig, axes = plt.subplots(3,3, figsize=(11,6.5))
plt.rcdefaults()
for k in range(len(sa_list)):
    # print(sa_list[k][4:])
    eta_per_group = []
    best_alloc = glob_best_alloc_list[B_ind][k][15:20]
    for i in range(2):
        x = np.array([data[0] for data in glob_eta_all_list[B_ind][2*k+i]]).T
        y = np.array([data[1] for data in glob_eta_all_list[B_ind][2*k+i]])[:, 15:20]  # Adjusted this line
        eta_per_group.append([x, y])
    ax = axes[k//3, k%3]
    ax.set_title(f'({chr(97 + k)}) $\\nu=${nu[k//3]}, $\\rho={rho[k%3]}$', fontsize=15)

    for i, eta in enumerate(eta_per_group):
        # if i%2==0: continue
        x, y = eta
        for g in range(5):
            ax.plot(x, y.T[g], 
                    linewidth = 2, 
                    color = color[g], 
                    alpha =1,
                    linestyle = 'dashed' if i%2==0 else '-',
                    marker = marker[g] if i%2==1 else '',
                    markersize=5, markevery=4000,
                    label = '1,'+str(g+1)+'' if (i%2==1 and k==0) else None)
                    # label = '(1,'+str(g+1)+') (Seattle, Age '+age_label[g] +')' if (i%2==1 and k==0) else None)
                    # label = group_label[g] if (i%2==1 and k==0) else None)
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
            for k in range(5):
                # ax.plot(x, f0[k])
                # ax.plot(x, f1[k])
                ax.fill_between(x, f0[k], f1[k], color = color[k], alpha=0.3)
                

    # plt.legend(bbox_to_anchor = [1.0,0.85])
    ax.set_xlabel('Month', size=12)
    ax.set_xticks([i*33.8 for i in range(12)])
    ax.set_xticklabels([i+1 for i in range(12)], size=10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], size=10)
    ax.set_ylabel(f'${{\\eta}}_i(t)$', size=12)
    ax.set_ylim([0,1.05])

    title_font_size = 18  # Adjust the title font size
    label_font_size = 14  # Adjust the axis labels' font size

fig.legend(bbox_to_anchor=(1.25, 0.6), ncol=1, title='Group')

fig.text(-0.06, 0.5, f'Perceived Vaccine risk ($\\nu$)', ha='center', va='center', rotation=90, fontsize = title_font_size, weight='bold')
fig.text(-0.02, 0.16, f'High\n($\\nu=${nu[2]})', ha='center', va='center', rotation=90,  fontsize = label_font_size, weight='bold')
fig.text(-0.02, 0.47, f'Medium\n($\\nu=${nu[1]})', ha='center', va='center', rotation=90, fontsize = label_font_size, weight='bold')
fig.text(-0.02, 0.8, f'Low\n($\\nu=${nu[0]})', ha='center', va='center', rotation=90, fontsize = label_font_size, weight='bold')

fig.text(0.5, -0.1, f'Importance of emotional judgment ($\\rho$)', ha='center', va='center', rotation=0, fontsize = title_font_size, weight='bold')
fig.text(0.21, -0.03, f'Rational\n($\\rho=0$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')
fig.text(0.53, -0.03, f'Balanced\n($\\rho=0.5$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')
fig.text(0.86, -0.03, f'Emotional\n($\\rho=1$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')

plt.suptitle('Changes in opinion persuasiveness with optimal campaign', fontsize = title_font_size, weight='bold')

plt.tight_layout()
# plt.savefig(f'Plot/Eta_B_{B[B_ind]}_fb_{int(p_online*100)}.png', transparent=False, dpi=300, bbox_inches='tight')
plt.show()

#%% Plot vacc
fig, axes = plt.subplots(3,3, figsize=(11,6.5))
plt.rcdefaults()
for k in range(len(sa_list)):
    # print(sa_list[k][4:])
    eta_per_group = []
    best_alloc = glob_best_alloc_list[B_ind][k][15:20]
    for i in range(2):
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(np.array(glob_ret_list[B_ind][2*k+i]), (len(glob_ret_list[B_ind][0]), 25, 8)))
        A = SA + IA + RA
        P = SP + IP + RP
        N = A + P
        y = (1-A/N)[15:20,:].T[:365]
        x = np.array([i for i in range(len(y))])
        eta_per_group.append([x, y])
    ax = axes[k//3, k%3]
    ax.set_title(f'({chr(97 + k)}) $\\nu=${nu[k//3]}, $\\rho={rho[k%3]}$', fontsize=15)

    for i, eta in enumerate(eta_per_group):
        # if i%2==0: continue
        x, y = eta
        for g in range(5):
            ax.plot(x, y.T[g], 
                    linewidth = 2, 
                    color = color[g], 
                    alpha =1,
                    linestyle = 'dashed' if i%2==0 else '-',
                    marker = marker[g] if i%2==1 else '',
                    markersize=5, markevery=4000,
                    label = '1,'+str(g+1)+'' if (i%2==1 and k==0) else None)
                    # label = '(1,'+str(g+1)+') (Seattle, Age '+age_label[g] +')' if (i%2==1 and k==0) else None)
                    # label = group_label[g] if (i%2==1 and k==0) else None)
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
            for k in range(5):
                # ax.plot(x, f0[k])
                # ax.plot(x, f1[k])
                ax.fill_between(x, f0[k], f1[k], color = color[k], alpha=0.3)
                

    # plt.legend(bbox_to_anchor = [1.2,0.85])
    ax.set_xlabel('Month', size=12)
    ax.set_xticks([i*33.8 for i in range(12)])
    ax.set_xticklabels([i+1 for i in range(12)], size=10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], size=10)
    ax.set_ylabel(r'$\frac{P(t)}{N(t)}$', size=12)
    ax.set_ylim([0,1.05])

    title_font_size = 18  # Adjust the title font size
    label_font_size = 14  # Adjust the axis labels' font size

fig.legend(bbox_to_anchor=(1.15, 0.6), ncol=1, title='Group')

fig.text(-0.06, 0.5, f'Perceived Vaccine risk ($\\nu$)', ha='center', va='center', rotation=90, fontsize = title_font_size, weight='bold')
fig.text(-0.02, 0.16, f'High\n($\\nu=${nu[2]})', ha='center', va='center', rotation=90,  fontsize = label_font_size, weight='bold')
fig.text(-0.02, 0.47, f'Medium\n($\\nu=${nu[1]})', ha='center', va='center', rotation=90, fontsize = label_font_size, weight='bold')
fig.text(-0.02, 0.8, f'Low\n($\\nu=${nu[0]})', ha='center', va='center', rotation=90, fontsize = label_font_size, weight='bold')

fig.text(0.5, -0.1, f'Importance of emotional judgment ($\\rho$)', ha='center', va='center', rotation=0, fontsize = title_font_size, weight='bold')
fig.text(0.21, -0.03, f'Rational\n($\\rho=0$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')
fig.text(0.53, -0.03, f'Balanced\n($\\rho=0.5$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')
fig.text(0.86, -0.03, f'Emotional\n($\\rho=1$)', ha='center', va='center', rotation=0, fontsize = label_font_size, weight='bold')

plt.suptitle('Changes in pro-vaccination population with optimal campaign', fontsize = title_font_size, weight='bold')

plt.tight_layout()
# plt.savefig(f'Plot/Vacc_B_{B[B_ind]}_fb_{int(p_online*100)}.png', transparent=False, dpi=300, bbox_inches='tight')
plt.show()
# %%
