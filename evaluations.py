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
            f'sub{sub}, diff={per_config_mse:.3f}\n' \
            f'Type{type_i}, alpha={all_types_alphas[i]}, rec={all_types_binary_recon[i]}({np.mean(all_types_binary_recon[i]):.3f}), %={all_types_zero_attn[i]}\n' \
            f'Type{type_j}, alpha={all_types_alphas[j]}, rec={all_types_binary_recon[j]}({np.mean(all_types_binary_recon[j]):.3f}), %={all_types_zero_attn[j]}\n' \
            f'Type{type_k}, alpha={all_types_alphas[k]}, rec={all_types_binary_recon[k]}({np.mean(all_types_binary_recon[k]):.3f}), %={all_types_zero_attn[k]}\n' \
        , horizontalalignment='center')
        plt.tight_layout()
        plt.savefig(f'results/lc_sub{sub}_{v}.png')
        plt.close()
    
    # save current best configs' best diff to human lc.
    # this will be used as benchmark for further search and eval.
    np.save('best_diff_recorder.npy', best_diff_recorder)
    

def consistency_alphas_vs_recon(attn_config_version, v):
    """
    Look at overall how consistent are alphas corresponding to recon loss.
    Ideally, the reverse rank of alphas should be the same as the rank of recon
    because the higher the alppha, the lower the recon for this dimension.
    """
    problem_types=[1, 2, 6]
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    
    fig, ax = plt.subplots()
    all_rhos = []
    all_alphas = []
    all_recon = []
    for sub in subs:
        all_types_alphas = []
        all_types_recon = []
        for idx in range(len(problem_types)):
            problem_type = problem_types[idx]

            alphas = np.round(
                np.load(
                f'results/{attn_config_version}_sub{sub}_{v}/' \
                f'all_alphas_type{problem_type}_sub{sub}_cluster.npy')[-3:], 3)
            
            binary_recon = np.round(
                np.load(
                f'results/{attn_config_version}_sub{sub}_{v}/' \
                f'all_recon_loss_ideal_type{problem_type}_sub{sub}_cluster.npy')[-3:], 3)
            
            all_types_alphas.extend(alphas)
            all_types_recon.extend(binary_recon)
            all_alphas.extend(alphas)
            all_recon.extend(binary_recon)

        rho, _ = stats.spearmanr(all_types_alphas, all_types_recon)
        # print(np.round(all_types_alphas, 3), np.round(all_types_recon, 3), f'rho={rho:.3f}')
        if str(rho) == 'nan':
            pass
        else:
            all_rhos.append(rho)
    
    ax.set_xlabel('Attention Strength')
    ax.set_ylabel('Reconstruction Loss')
    ax.scatter(all_alphas, all_recon)
    plt.savefig(f'results/correlation_highAttn_vs_reconLoss_{v}.png')
    
    print(np.round(all_rhos, 3))
    t, p = stats.ttest_1samp(all_rhos, popmean=0)
    print(f't={t:.3f}, p={p/2:.3f}')
            

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
    
    average_coef = np.mean(all_coefs)
    t, p = stats.ttest_1samp(all_coefs, popmean=0)
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


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    v = 'fit-human-entropy-fast-nocarryover'
    # attn_config_version = 'best_config'
    attn_config_version = 'hyper89'
    overall_eval(attn_config_version, v)
    # examine_subject_lc_and_attn_overtime(
    #     attn_config_version=attn_config_version, v=v)
    # consistency_alphas_vs_recon('best_config', v=v)