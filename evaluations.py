import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np 
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.models import Model

from utils import load_config, png2gif
from models import JointModel
from data import data_loader_V2, load_X_only
from clustering.utils import load_data
from losses import binary_crossentropy

"""
Evaluation routines.
"""  

def overall_eval(attn_config_version, v, threshold=[0, 0, 0]):
    """
    This function does all key evaluations and plot all results 
    in one figure for visual exam purpose. This function evals 
    one specific config (could be a hyper or best config).
    
    1. Percentage of zero low-level attn weights
        (two subplots: one errorbar, one CDF; one significance test)
    2. Binary recon loss 
        (three subplots, one per Type)
    3. Decoding accuracy
        (two subplots, and one significance test)
    
    return:
    -------
        - A figure with all the plots.
        - Results of significance tests that will be used for hyper selection.
    """
    problem_types = [1, 2, 6]
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    num_dims = 3
    comparisons = ['zero_attn', 'binary_recon']
    results_path = 'results'
    fig, axes = plt.subplots(3, 3, dpi=100)
    
    for c in range(len(comparisons)):
        comparison = comparisons[c]
        # e.g. { problem_type: {(True, False, False): [metric1, metric2, ... ]} }
        type2strategy2metric = defaultdict(lambda: defaultdict(list))

        alphas_all_types = np.ones((len(problem_types), num_subs, num_dims))
        
        for z in range(len(problem_types)):
            problem_type = problem_types[z]

            # for sub in subs:
            for s in range(num_subs):
                sub = subs[s]
                
                # First grab the final metric
                if comparison == 'zero_attn':
                    # For %attn, we grab the last item
                    metric_fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                                   f'all_percent_zero_attn_type{problem_type}_sub{sub}_cluster.npy'
                    metric = np.load(metric_fpath)[-1]                    
                else:
                    # For binary recon, we grab the last 3 entries (each for a dim)
                    metric_fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                                   f'all_recon_loss_ideal_type{problem_type}_sub{sub}_cluster.npy'
                    metric = np.load(metric_fpath)[-num_dims : ]

                # Second group metric based on attn strategy
                alphas_fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                               f'all_alphas_type{problem_type}_sub{sub}_cluster.npy'
                # get the final 3 alphas
                alphas = np.load(alphas_fpath)[-3:]
                
                if comparison == 'binary_recon':
                    # for recon only, to plot, we need to rearrange the dims
                    # as similarly done when there is counterbalancing.
                    conversion_order = np.argsort(alphas)[::-1]
                    alphas = alphas[conversion_order]
                    metric = metric[conversion_order]
                    
                    # collect the sorted alphas for later plotting average per type.
                    alphas_all_types[z, s, :] = alphas
                
                # 1e-6 is the lower bound of alpha constraint.
                # use tuple instead of list because tuple is not mutable.                    
                alphas = alphas - np.array(threshold)
                strategy = tuple(alphas > 1.0e-6)
                type2strategy2metric[problem_type][strategy].append(metric)
                
        ######## End of extracting results ########
        ######## Begin of plotting ########
                
        # plotting both %attn and binary recon.
        if comparison == 'zero_attn':
            ax = axes[0, 0]
            x_axis = np.linspace(0, 14, 7)
            average_metrics = []
            sem_metrics = []
            
            for z in range(len(problem_types)):
                problem_type = problem_types[z]
                # e.g. {(True, False, False): [metric]}
                strategy2metric = type2strategy2metric[problem_type]
                strategies = list(strategy2metric.keys())
                num_strategies = len(strategies)

                x_left = x_axis[z] + 1
                x_right = x_axis[z+1] - 1
                x_ticks = np.linspace(x_left, x_right, num_strategies)
                
                temp_collector = []
                for i in range(num_strategies):
                    strategy = strategies[i]
                    metrics = strategy2metric[strategy]
                    temp_collector.extend(metrics)

                np.save(
                    f'{results_path}/'\
                    f'{comparison}_type{problem_type}_allStrategies_{attn_config_version}_{v}.npy', 
                    temp_collector
                )
                
                average_metrics.append(np.mean(temp_collector))
                sem_metrics.append(stats.sem(temp_collector))

            # plot bar of averaged over strategies.
            ax.errorbar(
                x=x_axis[:len(problem_types)]+0.5,
                y=average_metrics,
                yerr=sem_metrics,
                fmt='*',
                color='k',
                ls='-',
                label='overall'
            )

            ax.set_xlabel('Problem Types')
            ax.set_xticks(x_axis[:len(problem_types)]+0.5)
            ax.set_xticklabels(problem_types)   
            ax.set_ylim([-0.05, 1.05])
            ax.set_title('Percentage of zero attn')
            
            # plot the histogram on the right.
            ax = axes[0, 1]
            ax, \
                t_type1v2, p_type1v2, t_type2v6, p_type2v6 = histogram_low_attn_weights(
                    ax=ax, 
                    attn_config_version=attn_config_version, 
                    v=v
                )
            
            # write stats on the empty space.
            ax = axes[0, 2]
            ax.axis('off')
            ax.text(-0., 0.8, f'Type 1 vs 2 = \n{t_type1v2:.1f}(p={p_type1v2:.3f})')
            ax.text(-0., 0.3, f'Type 2 vs 6 = \n{t_type2v6:.1f}(p={p_type2v6:.3f})')
    
            
        elif comparison == 'binary_recon':
            num_cols = 3
            x_axis = np.linspace(0, 8, num_dims+1)

            for z in range(len(problem_types)):
                problem_type = problem_types[z]
                col_idx = z % num_cols

                # e.g. {(True, False, False): [ [metric_dim1, dim2. dim3], [dim1, dim2, dim3], .. ]}
                strategy2metric = type2strategy2metric[problem_type]
                strategies = list(strategy2metric.keys())
                num_strategies = len(strategies)

                all_strategies_collector = []
                for i in range(num_strategies):
                    strategy = strategies[i]
                    # metrics -> [ [dim1, dim2, dim3], [dim1, dim2, dim3], .. ]
                    metrics = np.array(
                        strategy2metric[strategy])
                    all_strategies_collector.extend(metrics)

                average_metric = np.mean(np.array(all_strategies_collector), axis=0)
                sem_metric = stats.sem(np.array(all_strategies_collector), axis=0)
                axes[1, col_idx].errorbar(
                    x=x_axis[:num_dims],
                    y=average_metric,
                    yerr=sem_metric,
                    fmt='*',
                    color='k',
                    label='overall'
                )
                axes[1, col_idx].set_xticks([])
                axes[1, col_idx].set_xticks(x_axis[:num_dims]+0.5)
                axes[1, col_idx].set_xticklabels([f'dim{i+1}' for i in range(num_dims)])
                axes[1, col_idx].set_ylim([-0.5, 3])
                axes[1, col_idx].set_title(f'Type {problem_type}')
    
    # finally plot the third row which is the decoding results.
    average_coef_decoding, t_decoding, p_decoding = recon_loss_by_type(
        attn_config_version=attn_config_version, v=v)
    
    relate_recon_loss_to_decoding_error_errorbar(
        axes=axes[2, :], 
        attn_config_version=attn_config_version, 
        num_runs=3, roi='LOC', v=v)
    
    # write stats on the empty space.
    ax = axes[2, 2]
    ax.axis('off')
    ax.text(
        -0., 0.8, 
        f'Avg coef={average_coef_decoding:.3f}\n'\
        f't={t_decoding:.1f}(p={p_decoding:.3f})'
    )
    
    # write behaviour fitting stats
    # 1. Sum of fitting MSE of all subjects.
    per_config_mse_all_subs = 0
    for sub in subs:
        per_config_mse = 0
        for idx in range(len(problem_types)):
            problem_type = problem_types[idx]
            human_lc = np.load(f'clustering/results/human/lc_type{problem_type}_sub{sub}.npy')
            model_lc = np.load(f'results/{attn_config_version}_sub{sub}_{v}/lc_type{problem_type}_sub{sub}_cluster.npy')
            per_config_mse += np.mean( (human_lc - model_lc)**2 )
            per_config_mse_all_subs += per_config_mse
    ax.text(-0., 0.5, f'Sum all subs MSE = {per_config_mse_all_subs:.3f}')
    
    # 2. Final dim-averaged (counterbalanced) high-attn weights.
    ax.text(-0., 0.2, f'Type1 alphas={np.round(np.mean(alphas_all_types[0], axis=0), 1)}')
    ax.text(-0., 0.0, f'Type2 alphas={np.round(np.mean(alphas_all_types[1], axis=0), 1)}')
    ax.text(-0., -0.2, f'Type6 alphas={np.round(np.mean(alphas_all_types[2], axis=0), 1)}')
    
    # 3. Write hyper of this config.
    # NOTE(ken): can hard code sub because for the same hyper, values are the same.
    config = load_config(component=None, config_version=f'{attn_config_version}_sub02_{v}')
    lr_attn = config['lr_attn']
    loop = config['inner_loop_epochs']
    weighting = config['recon_clusters_weighting']
    noise = config['noise_level']
    
    plt.suptitle(f'{attn_config_version}_{v},\nlr_attn={lr_attn}, loop={loop}, weighting={weighting}, noise={noise}')
    plt.tight_layout()
    plt.savefig(f'{results_path}/overall_eval_{attn_config_version}_{v}.png')
    plt.close()
    
    return t_type1v2, p_type1v2, t_type2v6, p_type2v6, t_decoding, p_decoding, per_config_mse_all_subs
            
            
def stats_significance_of_zero_attn(attn_config_version, v):
    """
    Evaluate statistic significance across types 
    of the difference in percentage of zero low attn
    weights over runs & strategies.
    """
    results_path = 'results'
    type1 = np.load(f'{results_path}/zero_attn_type1_allStrategies_{attn_config_version}_{v}.npy')
    type2 = np.load(f'{results_path}/zero_attn_type2_allStrategies_{attn_config_version}_{v}.npy')
    type6 = np.load(f'{results_path}/zero_attn_type6_allStrategies_{attn_config_version}_{v}.npy')

    t_type1v2, p_type1v2 = stats.ttest_ind(type1, type2, equal_var=False)
    t_type2v6, p_type2v6 = stats.ttest_ind(type2, type6, equal_var=False)

    return t_type1v2, p_type1v2/2, t_type2v6, p_type2v6/2


def histogram_low_attn_weights(ax, attn_config_version, v):
    """
    Plot the histogram of learned low-level attn weights
    across types.
    """
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    problem_types = [1, 2, 6]
    num_dims = 3
    
    type1 = []
    type2 = []
    type6 = []
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        for sub in subs:
            results_path = f'results/{attn_config_version}_sub{sub}_{v}'
            
            attn_config = load_config(
                component=None, 
                config_version=f'{attn_config_version}_sub{sub}_{v}')
            
            attn_position = attn_config['attn_positions'].split(',')[0]   
                     
            attn_weights = np.load(
                f'{results_path}/attn_weights_type{problem_type}_sub{sub}_cluster.npy',
                allow_pickle=True).ravel()[0][attn_position]

            if problem_type == 1:
                type1.extend(attn_weights)
            elif problem_type == 2:
                type2.extend(attn_weights)
            elif problem_type == 6:
                type6.extend(attn_weights)

    colors = {'type1': 'y', 'type2': 'g', 'type6': 'r'}
    sns.kdeplot(type1, label='Type 1', ax=ax, color=colors['type1'])
    sns.kdeplot(type2, label='Type 2', ax=ax, color=colors['type2'])
    sns.kdeplot(type6, label='Type 6', ax=ax, color=colors['type6'])
    ax.set_xlabel(f'attn weights')

    # plot t-test results
    t_type1v2, p_type1v2, \
        t_type2v6, p_type2v6 = \
            stats_significance_of_zero_attn(attn_config_version, v) 
            
    return ax, t_type1v2, p_type1v2, t_type2v6, p_type2v6
                            

def visualize_attn_overtime(config_version, sub, ax):
    """
    Visualize the change of attn weights through learning.
    Through time we need to consider even subjects did 
    6-1-2 and odd subjects did 6-2-1.
    """
    config = load_config(
        component=None, config_version=config_version)
    num_dims = 3
    num_repetitions = config['num_repetitions']
    if int(sub) % 2 == 0:
        problem_types = [6, 1, 2]
    else:
        problem_types = [6, 2, 1]
    task_change_timepoint = [15, 31]
    
    attn_weights_overtime = []
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        for rp in range(num_repetitions):
            all_alphas = np.load(
                f'results/{config_version}/' \
                f'all_alphas_type{problem_type}_sub{sub}_cluster_rp{rp}.npy'
            )
            # all_alphas is a collector that extends every repeptition
            # but we only need the last 3 values from the latest rp.
            per_repetition_alphas = all_alphas[-num_dims:]
            attn_weights_overtime.append(per_repetition_alphas)

    # (16*3, 3)
    attn_weights_overtime = np.array(attn_weights_overtime)
    
    # plot 
    for dim in range(num_dims):
        dim_overtime = attn_weights_overtime[:, dim]
        ax.plot(dim_overtime, label=f'dim{dim+1}')
    
    # partition regions into Type 6, 1, 2
    ax.axvline(
        x=task_change_timepoint[0], 
        color='purple', 
        label=f'Type 6->{problem_types[1]}', 
        ls='dashed'
    )
    ax.axvline(
        x=task_change_timepoint[1], 
        color='pink', 
        label=f'Type {problem_types[1]}->{problem_types[2]}',
        ls='dashed'
    )
                        
                        
def examine_subject_lc_and_attn_overtime(attn_config_version, v):
    """
    Plotting per subject (either human or model) lc using
    the best config and plot the attn weights overtime.
    """
    problem_types=[1,2,6]
    num_subs = 23
    num_repetitions = 16
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    
    best_diff_recorder = {}
    for sub in subs:
        fig = plt.figure(dpi=200)
        gs = fig.add_gridspec(2,2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        colors = ['blue', 'orange', 'cyan']
        
        config_version = f'{attn_config_version}_sub{sub}_{v}'
        config = load_config(
            component=None, 
            config_version=config_version)
        print(f'sub{sub}, config={config_version}')
        
        # plot lc - human vs model
        per_config_mse = 0
        all_types_zero_attn = []
        all_types_alphas = []
        all_types_binary_recon = []
        for idx in range(len(problem_types)):
            problem_type = problem_types[idx]
            human_lc = np.load(f'clustering/results/human/lc_type{problem_type}_sub{sub}.npy')
            model_lc = np.load(f'results/{config_version}/lc_type{problem_type}_sub{sub}_cluster.npy')
            per_config_mse += np.mean( (human_lc - model_lc)**2 )

            ax1.set_title('human')
            ax1.errorbar(
                range(human_lc.shape[0]), 
                1-human_lc,
                color=colors[idx],
                label=f'Type {problem_type}',
            )
            ax1.set_xticks(range(0, num_repetitions+4, 4))
            ax1.set_xticklabels(range(0, num_repetitions+4, 4))
            ax1.set_ylim([-0.05, 1.05])
            ax1.set_xlabel('repetitions')
            ax1.set_ylabel('average probability of error')
            
            ax2.set_title('model')
            ax2.errorbar(
                range(model_lc.shape[0]), 
                1-model_lc,
                color=colors[idx],
                label=f'Type {problem_type}',
            )
            ax2.set_xticks(range(0, num_repetitions+4, 4))
            ax2.set_xticklabels(range(0, num_repetitions+4, 4))
            ax2.set_xlabel('repetitions')
            ax2.set_ylim([-0.05, 1.05])
            
            zero_attn = np.round(
                np.load(
                f'results/{attn_config_version}_sub{sub}_{v}/' \
                f'all_percent_zero_attn_type{problem_type}_sub{sub}_cluster.npy')[-1], 3)  
            all_types_zero_attn.append(zero_attn)            
        
            alphas = np.round(
                np.load(
                f'results/{attn_config_version}_sub{sub}_{v}/' \
                f'all_alphas_type{problem_type}_sub{sub}_cluster.npy')[-3:], 3)
            all_types_alphas.append(alphas)
                    
            binary_recon = np.round(
                np.load(
                f'results/{attn_config_version}_sub{sub}_{v}/' \
                f'all_recon_loss_ideal_type{problem_type}_sub{sub}_cluster.npy')[-3:], 3)
            all_types_binary_recon.append(binary_recon)

        best_diff_recorder[sub] = per_config_mse
        
        # plot attn weights overtime
        visualize_attn_overtime(
            config_version=config_version,
            sub=sub,
            ax=ax3
        )
        
        # plot hyper-params of this config on figure
        x_coord = 3
        y_coord = 0.45
        margin = 0.08        
        lr_attn = config['lr_attn']
        inner_loop_epochs = config['inner_loop_epochs']
        recon_clusters_weighting = config['recon_clusters_weighting']
        noise_level = config['noise_level']
        reg_strength = config['reg_strength']
        high_attn_reg_strength = config['high_attn_reg_strength']
        ax2.text(x_coord, y_coord, f'lr_attn={lr_attn}')
        ax2.text(x_coord, y_coord-margin*1, f'inner_loop_epochs={inner_loop_epochs}')
        ax2.text(x_coord, y_coord-margin*2, f'recon_clusters_weighting={recon_clusters_weighting}')
        ax2.text(x_coord, y_coord-margin*3, f'noise_level={noise_level}')
        ax2.text(x_coord, y_coord-margin*4, f'reg_strength={reg_strength}')
        ax2.text(x_coord, y_coord-margin*5, f'high_attn_reg_strength={high_attn_reg_strength}')
        plt.legend()
        
        if int(sub) % 2 == 0:
            i = 2  # 6
            j = 0  # 1
            k = 1  # 2
            type_i = 6
            type_j = 1
            type_k = 2
            
        elif int(sub) % 2 != 0:
            i = 2  # 6
            j = 1  # 2
            j = 0  # 1
            type_i = 6
            type_j = 2
            type_k = 1
            
        plt.suptitle(
            f'sub{sub}, diff={per_config_mse:.3f}\n') \
            # f'Type{type_i}, alpha={all_types_alphas[i]}, rec={all_types_binary_recon[i]}({np.mean(all_types_binary_recon[i]):.3f}), %={all_types_zero_attn[i]}\n' \
            # f'Type{type_j}, alpha={all_types_alphas[j]}, rec={all_types_binary_recon[j]}({np.mean(all_types_binary_recon[j]):.3f}), %={all_types_zero_attn[j]}\n' \
            # f'Type{type_k}, alpha={all_types_alphas[k]}, rec={all_types_binary_recon[k]}({np.mean(all_types_binary_recon[k]):.3f}), %={all_types_zero_attn[k]}\n' \
        # , horizontalalignment='center')
        plt.tight_layout()
        plt.savefig(f'results/lc_sub{sub}_{v}.png')
        plt.close()
                

def recon_loss_by_type(attn_config_version, v):
    """
    Group recon loss by type, following conventions of 
    `brain_data/decoding.py`
    """
    problem_types = [1, 2, 6]
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    num_dims = 3
    results_path = 'results'
    
    # e.g. {problem_type: [sub02_loss, sub03_loss, ... ]}
    recon_loss_collector = defaultdict(list)
    for problem_type in problem_types:
        for sub in subs:
            # For binary recon, we grab the last 3 entries (each for a dim)
            fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                    f'all_recon_loss_ideal_type{problem_type}_sub{sub}_cluster.npy'
            per_type_results = np.load(fpath)[-num_dims : ]
            recon_loss_collector[problem_type].append(np.mean(per_type_results))
            
    np.save(f'{results_path}/recon_loss_{attn_config_version}_{v}.npy', recon_loss_collector)
        
    average_coef, t, p = recon_loss_by_type_regression(
        recon_loss_collector, 
        num_subs=num_subs, 
        problem_types=problem_types)
    
    return average_coef, t, p


def recon_loss_by_type_regression(recon_loss_collector, num_subs, problem_types):
    """Fitting linear regression models to per subject decoding 
    accuracies over problem_types. This way, we can read off the 
    regression coefficient on whether there is a down trend of 
    recon loss as task difficulty increases in order to 
    test statistic significance of our finding that the harder 
    the problem, the better the recon.
    
    Impl:
    -----
        `recon_loss_collector` are saved in format:
            {
             'Type1': [sub02_loss, sub03_loss, ..],
             'Type2}: ...
            }
        
        To fit linear regression per subject across types, we 
        convert the format to a matrix where each row is a subject,
        and each column is a problem_type.
    """
    import pingouin as pg
    from scipy import stats
        
    group_results_by_subject = np.ones((num_subs, len(problem_types)))
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        # [sub02_acc, sub03_acc, ...]
        per_type_all_subjects = recon_loss_collector[problem_type]
        for s in range(num_subs):
            group_results_by_subject[s, z] = per_type_all_subjects[s]
    
    all_coefs = []
    for s in range(num_subs):
        X_sub = problem_types
        # [sub02_type1_acc, sub02_type2_acc, ...]
        y_sub = group_results_by_subject[s, :]
        coef = pg.linear_regression(X=X_sub, y=y_sub, coef_only=True)
        all_coefs.append(coef[-1])
        # print(f'{s}', y_sub, coef[-1])

    average_coef = np.mean(all_coefs)
    t, p = stats.ttest_1samp(all_coefs, popmean=0)
    # print(average_coef)
    # print(t, p)
    
    return average_coef, t, p/2
        

def relate_recon_loss_to_decoding_error_errorbar(axes, attn_config_version, num_runs, roi, v):
    """Relate binary reconstruction loss at the final layer
    of DCNN to the decoding error of the problem types 
    in the brain given ROI. For brain decoding, see `brain_data/`
    
    Impl:
    -----
        For brain decoding, we have produced results which are 
        in `brain_data/decoding.py` and `brain_data/decoding_results/`.

        For model recon, we obtain results in `recon_loss_by_type`
        following conventions of how we save decoding results.
    """
    problem_types = [1, 2, 6]
    
    recon_loss_collector = np.load(
        f'results/recon_loss_{attn_config_version}_{v}.npy', 
        allow_pickle=True).ravel()[0]
    
    decoding_error_collector = np.load(
        f'brain_data/decoding_results/decoding_error_{num_runs}runs_{roi}.npy', 
        allow_pickle=True).ravel()[0]
    
    palette = {'Type 1': 'pink', 'Type 2': 'green', 'Type 6': 'blue'}
    results_collectors = [recon_loss_collector, decoding_error_collector]
    for i in range(len(results_collectors)):
        results_collector = results_collectors[i]
        
        for j in range(len(problem_types)):
            problem_type = problem_types[j]
            data_perType = results_collector[problem_type]
            
            axes[i].errorbar(
                x=j,
                y=np.mean(data_perType),
                yerr=stats.sem(data_perType),
                label=f'Type {problem_type}',
                fmt='o',
                capsize=3,
            )

        axes[i].set_xlabel('Problem Types')
        axes[i].set_xticks(range(len(problem_types)))
        axes[i].set_xticklabels(problem_types)
        if i == 0:
            axes[i].set_title('Model Recon')
        else:
            axes[i].set_title(f'{roi} Neural Recon')
    
    return axes


def subject_dimension_rt_acc():
    """
    Look at subject response time when different 
    dimensions are being diagnostic. This is to see
    if subjects are particularly slow when one of the 
    dimensions is being task relevant.
    Impl:
    -----
        Look at Type 1 only when we know the first dim
        is always the diag dimension regardless of its
        physical meaning. 
        
        We group subjects based on the physical dimension
        they use to solve Type 1. So we can compare RT 
        across different diag dimensions. 
        
    Conclusion:
    -----------
        When the 3rd physical dimension is used as the
        diagnostic dimension, the overall subjects RT is 
        the slowest. 
        dim1:    1060715.5806451612
        dim2:    1181767.4193548388
        dim3:    1366676.3631442343
    """
    # study1: type 6
    # study2: type 1
    # study3: type 2
    # 1. trial number (1-32)
    # 2. task (1-3)
    # 3. run (1-4)
    # 4. dimension 1
    # 5. dimension 2
    # 6. dimension 3
    # 7. answer (1 or 2)
    # 8. response (1 or 2)
    # 9. RT (ms)
    # 10. accuracy (0 or 1)

    runs = [1, 2, 3, 4]
    # load subject's dimension assignment,
    # i.e. each subject's dimension 1 maps to different physical dims.
    subject2dimension = pd.read_csv('brain_data/Mack-Data/featureinfo.txt', header=None).to_numpy()
    subject_ids_temp = subject2dimension[:, 0]

    # convert `1` to `01` etc.
    subject_ids = []
    for subject_id in subject_ids_temp:
        if len(f'{subject_id}') == 1:
            subject_ids.append(f'0{subject_id}')
        else:
            subject_ids.append(f'{subject_id}')

    # For Type 1, diagnostic dimension is always dimenson 1
    # only its physical meaning changes.
    diagnostic_dimension = subject2dimension[:, 1]

    # int here is idx of physical dim not dim1,2,3
    dim2meaning = {
        '1': 'leg',
        '2': 'mouth',
        '3': 'manible'
    }

    # group subjects that use the same physical dim as 
    # the diagnostic dimension.
    # dict = {1: ['02', '03'], 2: [], 3: []}
    dim2subject = defaultdict(list)
    for i in range(subject2dimension.shape[0]):
        dim = diagnostic_dimension[i]
        subject_id = subject_ids[i]
        dim2subject[dim].append(subject_id)

    # for each diag dim, subjects who use that dim 
    # will be used to sum up their RT and accuracy through trials and runs,
    dim2rt = defaultdict(list)
    dim2acc = defaultdict(list)
    for dim in [1, 2, 3]:
        subject_ids = dim2subject[dim]
        
        for subject_id in subject_ids:
            data_dir = f'brain_data/Mack-Data/behaviour/subject_{subject_id}'
            
            # even subject Type 1 - study_2
            # odd subject Type 1 - study_3
            if int(subject_id) % 2 == 0:
                study_id = 2
            else:
                study_id = 3

            subject_rt = []
            for run in runs:
            
                fpath = f'{data_dir}/{subject_id}_study{study_id}_run{run}.txt'
                # data is 2D array, each row is a trial
                data = pd.read_csv(fpath, sep = "\t", header=None).to_numpy()
                    
                # locate nan RT within a run
                nan_i = []
                has_nan = False 
                data_exc_nan = []

                # iterate one trial at a time & collect RT    
                for i in range(data.shape[0]):
                    rt = data[i, 8]
                    if np.isnan(rt):
                        nan_i.append(i)
                        has_nan = True
                    else:
                        data_exc_nan.append(rt)
                        
                # replace nan to the average RT of the rest of the trials.
                if has_nan:
                    for j in nan_i:
                        data[j, 8] = np.mean(data_exc_nan)
                
                # collect trials' RT of this run.
                subject_rt.extend(data[:, 8])
            
                # we compute accuracy over one repetition 
                # which was how human acc was computed. 
                # That is, we compute accuracy over 8 trials of a run.
                num_reps_per_run = data.shape[0] // 8  # 4
                for k in range(num_reps_per_run):
                    subject_wrong = 0
                    subject_correct = 0
                    for acc in data[k*8:(k+1)*8, 9]:
                        if acc == 0:
                            subject_wrong += 1
                        else:
                            subject_correct += 1
                
                    # colect accuracy of every rep.
                    dim2acc[dim].append(subject_correct / (subject_correct + subject_wrong))
            dim2rt[dim].extend(subject_rt)

    # =========================================================================
    # plot dim2rt
    fig, ax = plt.subplots(2)
    data_RT = []
    data_ACC = []
    RT_medians = []
    RT_means = []
    ACC_medians = []
    ACC_means = []
    for dim in [1, 2, 3]:
        rt = dim2rt[dim]  # rt = [32 * 4 * 8] or [32 * 4 * 7]
        acc = dim2acc[dim]
        data_RT.append(rt)
        data_ACC.append(acc)
        RT_medians.append(np.median(rt))
        RT_means.append(np.mean(rt))
        ACC_medians.append(np.median(acc))
        ACC_means.append(np.mean(acc)) 
    assert len(data_RT) == 3
    
    # store results as a dataframe
    dims_RTs = ['Dimension']
    dims_ACCs = ['Dimension']
    RTs = ['RT']
    ACCs = ['Accuracy']
    for dim in [1, 2, 3]:
        rt = dim2rt[dim]
        acc = dim2acc[dim]
        dims_RTs.extend(f'{dim}' * len(rt))
        dims_ACCs.extend(f'{dim}' * len(acc))
        RTs.extend(rt)
        ACCs.extend(acc)
    
    # save results as a csv file
    dims_RTs = np.array(dims_RTs)
    dims_ACCs = np.array(dims_ACCs)
    RTs = np.array(RTs)
    ACCs = np.array(ACCs)

    df_RTs = np.vstack((dims_RTs, RTs)).T
    pd.DataFrame(df_RTs).to_csv(
        f"brain_data/Mack-Data/RTs.csv", 
        index=False, header=False
    )
    df_RTs = pd.read_csv('brain_data/Mack-Data/RTs.csv', usecols=['Dimension', 'RT'])

    df_ACCs = np.vstack((dims_ACCs, ACCs)).T
    pd.DataFrame(df_ACCs).to_csv(
        f"brain_data/Mack-Data/ACCs.csv", 
        index=False, header=False
    )
    df_ACCs = pd.read_csv('brain_data/Mack-Data/ACCs.csv', usecols=['Dimension', 'Accuracy'])

    # ------ statistic testing ------
    from scipy.stats import median_test, ttest_ind
    print(f'------ RT statistic testing ------')
    # stats, p = median_test(data_RT[0], data_RT[1])[:2]
    # print('dim1 vs dim2: ', f'stats={stats}, p={p}')
    
    # stats, p = median_test(data_RT[0], data_RT[2])[:2]
    # print('dim1 vs dim3: ', f'stats={stats}, p={p}')
    
    # stats, p = median_test(data_RT[1], data_RT[2])[:2]
    # print('dim2 vs dim3: ', f'stats={stats}, p={p}')

    stats, p = ttest_ind(data_RT[0], data_RT[1])[:2]
    print('dim1 vs dim2: ', f'stats={stats}, p={p}')
    
    stats, p = ttest_ind(data_RT[0], data_RT[2])[:2]
    print('dim1 vs dim3: ', f'stats={stats}, p={p}')
    
    stats, p = ttest_ind(data_RT[1], data_RT[2])[:2]
    print('dim2 vs dim3: ', f'stats={stats}, p={p}')
    
    print(f'median RT = {RT_medians}')
    print(f'mean RT = {RT_means}')
    
    print(f'\n ------ Acc statistic testing ------')
    stats, p = ttest_ind(data_ACC[0], data_ACC[1])[:2]
    print('dim1 vs dim2: ', f'stats={stats}, p={p}')
    
    stats, p = ttest_ind(data_ACC[0], data_ACC[2])[:2]
    print('dim1 vs dim3: ', f'stats={stats}, p={p}')
    
    stats, p = ttest_ind(data_ACC[1], data_ACC[2])[:2]
    print('dim2 vs dim3: ', f'stats={stats}, p={p}')

    print(f'median Acc = {ACC_medians}')
    print(f'mean Acc = {ACC_means}')

    print(data_ACC[0], '\n')
    print(data_ACC[1], '\n')
    print(data_ACC[2], '\n')

    # ------ plotting ------
    # ### subplot 1: RT ###
    sns.barplot(x='Dimension', y='RT', data=df_RTs, ax=ax[0])
    ax[0].set_xticks(range(len(data_RT)))
    ax[0].set_xticklabels(['Leg', 'Antenna', 'Mandible'])
    ax[0].set_ylabel('Subject RT (ms)')
    ax[0].set_ylim(800, 1550)
    ax[0].set(xlabel=None)

    # sig for dim1 and dim3
    x1, x2 = 0, 2
    y, h = 1450 + 10, 10
    ax[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax[0].text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color='k')
    # sig for dim1 and dim2
    x1, x2 = 0, 1
    y, h = 1250 + 20, 20
    ax[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax[0].text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color='k')
    # sig for dim2 and dim3
    x1, x2 = 1, 2
    y, h = 1390 + 4, 4
    ax[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax[0].text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color='k')
    ax[0].spines.right.set_visible(False)
    ax[0].spines.top.set_visible(False)

    
    # ### subplot 2: ACC ###
    sns.barplot(x='Dimension', y='Accuracy', data=df_ACCs, ax=ax[1])
    ax[1].set_xticks(range(len(data_RT)))
    ax[1].set_xticklabels(['Leg', 'Antenna', 'Mandible'])
    ax[1].set_ylabel('Subject Accuracy')
    ax[1].set_yticks([0.8, 0.9, 1, 1.08])
    ax[1].set_yticklabels(['0.8', '0.9', '1.0', ''])
    ax[1].set_ylim(0.8, 1.08)
    ax[1].set(xlabel=None)

    # sig for dim1 and dim3
    x1, x2 = 1, 2
    y, h = 0.97 + 0.01, 0.01
    ax[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax[1].text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color='k')
    # sig for dim1 and dim2
    x1, x2 = 0, 1
    y, h = 0.99 + 0.01, 0.01
    ax[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax[1].text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color='k')
    # sig for dim1 and dim2
    x1, x2 = 0, 2
    y, h = 1.03 + 0.01, 0.01
    ax[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax[1].text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color='k')
    ax[1].spines.right.set_visible(False)
    ax[1].spines.top.set_visible(False)

    plt.tight_layout()
    plt.savefig('figs/subject_dimension_rt_acc.pdf')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    v = 'fit-human-entropy-fast-nocarryover'
    # attn_config_version = 'best_config'
    attn_config_version = 'hyper4100'
    # overall_eval(attn_config_version, v)
    # examine_subject_lc_and_attn_overtime(attn_config_version, v)
    # consistency_alphas_vs_recon('best_config', v=v)

    subject_dimension_rt_acc()