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

def compare_across_types_V3(attn_config_version, threshold=[0, 0, 0]):
    """Key differences to the generic `compare_across_types_V3`:
    1. We only consider Type 1, 2 and 6. 
    2. Canonical run or not is no longer valid due to 
        the carryover effect results in recruitment of all clusters.
    3. No counterbalancing since everything is based on human data. 
    """
    problem_types = [1, 2, 6]
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    num_dims = 3
    comparisons = ['zero_attn', 'binary_recon']
    results_path = 'results'
    
    
    for c in range(len(comparisons)):
        comparison = comparisons[c]
        # e.g. { problem_type: {(True, False, False): [metric1, metric2, ... ]} }
        type2strategy2metric = defaultdict(lambda: defaultdict(list))

        for z in range(len(problem_types)):
            problem_type = problem_types[z]
            print(f'------------ problem_type = {problem_type} ------------')

            for sub in subs:
                # First grab the final metric
                if comparison == 'zero_attn':
                    # For %attn, we grab the last item
                    metric_fpath = f'{results_path}/{attn_config_version}_sub{sub}_fit-human/' \
                                   f'all_percent_zero_attn_type{problem_type}_sub{sub}_cluster.npy'
                    metric = np.load(metric_fpath)[-1]                    
                else:
                    # For binary recon, we grab the last 3 entries (each for a dim)
                    metric_fpath = f'{results_path}/{attn_config_version}_sub{sub}_fit-human/' \
                                   f'all_recon_loss_ideal_type{problem_type}_sub{sub}_cluster.npy'
                    metric = np.load(metric_fpath)[-num_dims : ]

                # Second group metric based on attn strategy
                alphas_fpath = f'{results_path}/{attn_config_version}_sub{sub}_fit-human/' \
                               f'all_alphas_type{problem_type}_sub{sub}_cluster.npy'
                # get the final 3 alphas
                alphas = np.load(alphas_fpath)[-3:]
                alphas = alphas - np.array(threshold)

                # 1e-6 is the lower bound of alpha constraint.
                # use tuple instead of list because tuple is not mutable.
                strategy = tuple(alphas > 1.0e-6)
                type2strategy2metric[problem_type][strategy].append(metric)
                
                # if problem_type in [1]:
                #     if np.sum(strategy) >= 2:
                #         pass
                #     else:
                #         type2strategy2metric[problem_type][strategy].append(metric)
                # elif problem_type in [2]:
                #     if np.sum(strategy) == 3:
                #         pass
                #     else:
                #         type2strategy2metric[problem_type][strategy].append(metric)
                # else:
                #     type2strategy2metric[problem_type][strategy].append(metric)

        # plotting both %attn and binary recon.
        if comparison == 'zero_attn':
            fig, ax = plt.subplots()
            x_axis = np.linspace(0, 14, 7)
            colors = ['green', 'red', 'brown', 'orange', 'cyan', 'blue']
            average_metrics = []
            std_metrics = []
            
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
                    # plot bar of a single strategy
                    ax.errorbar(
                        x=x_ticks[i],
                        y=np.mean(metrics),
                        yerr=np.std(metrics),
                        fmt='o',
                        color=colors[z])
                    
                # NOTE: Hacky way of getting legend correct.
                ax.errorbar(
                    x=x_ticks[i],
                    y=np.mean(metrics),
                    yerr=np.std(metrics),
                    fmt='o',
                    color=colors[z],
                    label=f'single strategy, type {problem_type}')

                np.save(f'{results_path}/{comparison}_type{problem_type}_allStrategies_{attn_config_version}.npy', temp_collector)
                average_metrics.append(np.mean(temp_collector))
                std_metrics.append(np.std(temp_collector))

            # plot bar of averaged over strategies.
            ax.errorbar(
                x=x_axis[:len(problem_types)]+0.5,
                y=average_metrics,
                yerr=std_metrics,
                fmt='*',
                color='k',
                ls='-',
                label='overall'
            )

            ax.set_xticks(x_axis[:len(problem_types)]+0.5)
            ax.set_xticklabels([f'Type {problem_type}' for problem_type in problem_types])
            ax.set_ylim([-0.05, 1.05])
            ax.set_ylabel('percentage of zero attention weights')
            plt.tight_layout()
            plt.legend()            
            plt.savefig(f'{results_path}/compare_types_percent_zero_{attn_config_version}.png')
            plt.close()

        elif comparison == 'binary_recon':
            num_cols = 3
            num_rows = int(len(problem_types) / num_cols)
            x_axis = np.linspace(0, 8, num_dims+1)
            fig, ax = plt.subplots(num_rows, num_cols)

            for z in range(len(problem_types)):
                problem_type = problem_types[z]
                row_idx = z // num_cols
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

                    colors = ['green', 'red', 'brown']
                    for dim in range(num_dims):
                        x_left = x_axis[dim] + 1
                        x_right = x_axis[dim+1] - 1
                        x_ticks = np.linspace(x_left, x_right, num_strategies)

                        ax[col_idx].errorbar(
                            x=x_ticks[i],
                            y=np.mean(metrics[:, dim]),
                            yerr=np.std(metrics[:, dim]),
                            fmt='o',
                            color=colors[dim],
                            label=f'single strategy, dim{dim}')

                average_metric = np.mean(np.array(all_strategies_collector), axis=0)
                std_metric = np.std(np.array(all_strategies_collector), axis=0)
                ax[col_idx].errorbar(
                    x=x_axis[:num_dims],
                    y=average_metric,
                    yerr=std_metric,
                    fmt='*',
                    color='k',
                    label='overall'
                )
                ax[col_idx].set_xticks([])
                ax[col_idx].set_xticks(x_axis[:num_dims]+0.5)
                ax[col_idx].set_xticklabels([f'dim{i+1}' for i in range(num_dims)])
                ax[col_idx].set_ylim([-0.5, 9.5])
                ax[0].set_ylabel('binary recon loss')
                ax[col_idx].set_title(f'Type {problem_type}')
            
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(f'{results_path}/compare_types_dimension_binary_recon_{attn_config_version}.png')


def stats_significance_of_zero_attn(attn_config_version):
    """
    Evaluate statistic significance across types 
    of the difference in percentage of zero low attn
    weights over runs & strategies.
    """
    results_path = 'results'
    type1 = np.load(f'{results_path}/zero_attn_type1_allStrategies_{attn_config_version}.npy')
    type2 = np.load(f'{results_path}/zero_attn_type2_allStrategies_{attn_config_version}.npy')
    type6 = np.load(f'{results_path}/zero_attn_type6_allStrategies_{attn_config_version}.npy')
    print(len(type1), len(type2), len(type6))

    print('Type 1 vs 2: ', stats.ttest_ind(type1, type2, equal_var=False))
    t_type1v2, p_type1v2 = stats.ttest_ind(type1, type2, equal_var=False)

    print('Type 2 vs 6: ', stats.ttest_ind(type2, type6, equal_var=False))
    t_type2v6, p_type2v6 = stats.ttest_ind(type2, type6, equal_var=False)

    # cohen_d = (np.mean(type1) - np.mean(type2)) / np.mean((np.std(type1) + np.std(type2)))
    # print(cohen_d)
    # cohen_d = (np.mean(type2) - np.mean(type6)) / np.mean((np.std(type2) + np.std(type6)))
    # print(cohen_d)

    return t_type1v2, p_type1v2, t_type2v6, p_type2v6


def histogram_low_attn_weights(attn_config_version, threshold=[0., 0., 0.]):
    """
    Plot the histogram of learned low-level attn weights
    across types.
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    num_runs = attn_config['num_runs']
    num_types = 6
    num_dims = 3
    attn_position = attn_config['attn_positions'].split(',')[0]
    results_path = f'results/{attn_config_version}'
    type2runs = find_canonical_runs(
        attn_config_version, canonical_runs_only=True)
    
    type1 = []
    type2 = []
    type6 = []
    for z in range(num_types):
        if z in [0, 1, 5]:
            problem_type = z + 1
            for run in type2runs[z]:

                # TODO: do this filtering or not?
                alphas_fpath = f'{results_path}/all_alphas_type{problem_type}_run{run}_cluster.npy'
                alphas = np.load(alphas_fpath)[-3:]
                alphas = alphas - np.array(threshold)
                strategy = tuple(alphas > 1.0e-6)
                if problem_type in [1]:
                    if np.sum(strategy) >= 2:
                        continue
                elif problem_type in [2]:
                    if np.sum(strategy) == 3:
                        continue

                attn_weights = np.load(
                    f'{results_path}/attn_weights_type{problem_type}_run{run}_cluster.npy',
                    allow_pickle=True
                ).ravel()[0][attn_position]

                if problem_type == 1:
                    type1.extend(attn_weights)
                elif problem_type == 2:
                    type2.extend(attn_weights)
                elif problem_type == 6:
                    type6.extend(attn_weights)

    fig, ax = plt.subplots()
    colors = {'type1': 'y', 'type2': 'g', 'type6': 'r'}
    print('max type1 = ', np.max(type1))
    print('max type2 = ', np.max(type2))
    print('max type6 = ', np.max(type6))

    # ax.axvline(np.max(type1), color=colors['type1'], linestyle='dashed')
    # ax.axvline(np.max(type2), color=colors['type2'], linestyle='dashed')
    # ax.axvline(np.max(type6), color=colors['type6'], linestyle='dashed')
    sns.kdeplot(type1, label='Type 1', ax=ax, color=colors['type1'])
    sns.kdeplot(type2, label='Type 2', ax=ax, color=colors['type2'])
    sns.kdeplot(type6, label='Type 6', ax=ax, color=colors['type6'])
    ax.set_xlabel(f'attn weights')

    # plot t-test results
    t_type1v2, p_type1v2, \
    t_type2v6, p_type2v6 = stats_significance_of_zero_attn(attn_config_version)

    plt.text(0.2, 10, f'Type 1 v Type 2: t={t_type1v2:.3f}, p-value < 1e-4')
    plt.text(0.2, 8, f'Type 2 v Type 6: t={t_type2v6:.3f}, p-value < 1e-4')


    plt.legend()
    plt.savefig(f'{results_path}/attn_weights_histogram.png')
                            

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
                        
                        
def examine_subject_lc_and_attn_overtime(attn_config_version):
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
        fig = plt.figure()
        gs = fig.add_gridspec(2,2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        colors = ['blue', 'orange', 'cyan']
        
        config_version = f'{attn_config_version}_sub{sub}_fit-human'
        config = load_config(
            component=None, 
            config_version=config_version)
        print(f'sub{sub}, config={config_version}')
        
        # plot lc - human vs model
        per_config_mse = 0
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
        
        best_diff_recorder[sub] = per_config_mse
        
        # plot attn weights overtime
        visualize_attn_overtime(
            config_version=config_version,
            sub=sub,
            ax=ax3
        )
        
        # plot hyper-params of this config on figure
        x_coord = 3
        y_coord = 0.25
        margin = 0.08        
        lr_attn = config['lr_attn']
        inner_loop_epochs = config['inner_loop_epochs']
        recon_clusters_weighting = config['recon_clusters_weighting']
        noise_level = config['noise_level']
        ax2.text(x_coord, y_coord, f'lr_attn={lr_attn}')
        ax2.text(x_coord, y_coord-margin*1, f'inner_loop_epochs={inner_loop_epochs}')
        ax2.text(x_coord, y_coord-margin*2, f'recon_clusters_weighting={recon_clusters_weighting}')
        ax2.text(x_coord, y_coord-margin*3, f'noise_level={noise_level}')
        
        plt.legend()
        plt.suptitle(f'sub{sub}, diff={per_config_mse:.3f}')
        plt.tight_layout()
        plt.savefig(f'results/lc_sub{sub}.png')
        plt.close()
    
    # save current best configs' best diff to human lc.
    # this will be used as benchmark for further search and eval.
    np.save('best_diff_recorder.npy', best_diff_recorder)
    
                
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    examine_subject_lc_and_attn_overtime('best_config')
    compare_across_types_V3('best_config')