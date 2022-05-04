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

def how_low_can_att_weights(
        attn_weight_constant,
        attn_config_version,
        dcnn_config_version,
        problem_type=1,
        noise_level=0,
        seed=999):
    """
    Check how low can attn weights 
    go (uniformly) without affecting
    binary codings.

    Supply a constant that will be broadcast and used
    as attention weights.

    if noise_const:
        we add non-uniform noise to init attn weights, hoping to 
        get some zero weights sooner than having uniform reduction
        for a very long time.
    """
    joint_model = JointModel(
        attn_config_version=attn_config_version,
        dcnn_config_version=dcnn_config_version, 
    )
    preprocess_func = joint_model.preprocess_func
    
    attn_config = load_config(
        component=None, 
        config_version=attn_config_version
    )
    attn_positions = attn_config['attn_positions'].split(',')
    for attn_position in attn_positions:
        layer_attn_weights = \
            joint_model.get_layer(
                'dcnn_model').get_layer(
                    f'attn_factory_{attn_position}').get_weights()[0]
        
        # lower the attn weights uniformly.                       
        layer_attn_weights = np.ones((layer_attn_weights.shape)) * attn_weight_constant
        if noise_level:
            np.random.seed(seed)
            noise = np.random.uniform(low=-noise_level, high=noise_level, size=layer_attn_weights.shape)
            print(f'noise = {noise}')
            layer_attn_weights += noise
            
        print(f'layer_attn_weights = {layer_attn_weights}')
        print(f'max = {np.max(layer_attn_weights)}, min = {np.min(layer_attn_weights)}')
        joint_model.get_layer(
            'dcnn_model').get_layer(
                f'attn_factory_{attn_position}').set_weights([layer_attn_weights])

    # load data
    dataset, _ = data_loader_V2(
        attn_config_version=attn_config_version,
        dcnn_config_version=dcnn_config_version, 
        preprocess_func=preprocess_func,
        problem_type=problem_type
    )
    
    # each stimulus, train indendently
    y_pred = []
    for i in range(len(dataset)):
        dp = dataset[i]
        x = dp[0]
        y_pred.append(joint_model(x)[0])  # only need the 0th value.
    y_pred = np.array(y_pred)
    print(f'y_pred = \n {y_pred}')


def viz_losses(
        attn_config_version,
        problem_type,
        recon_level,
        run):
    """
    Visualize progress of losses
        - recon loss 
        - theother recon loss
        - reg loss
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    inner_loop_epochs = attn_config['inner_loop_epochs']
    recon_loss = np.load(
        f'results/{attn_config_version}/all_recon_loss_type{problem_type}_run{run}_{recon_level}.npy'
    )
    recon_loss_ideal = np.load(
        f'results/{attn_config_version}/all_recon_loss_ideal_type{problem_type}_run{run}_{recon_level}.npy'
    )
    reg_loss = np.load(
        f'results/{attn_config_version}/all_reg_loss_type{problem_type}_run{run}_{recon_level}.npy'
    )
    percent_zero = np.load(
        f'results/{attn_config_version}/all_percent_zero_attn_type{problem_type}_run{run}_{recon_level}.npy'
    )

    final_recon_loss_ideal = recon_loss_ideal[-1]
    final_percent_zero = percent_zero[-1]


    # ===== plotting =====
    fig, ax = plt.subplots(4, dpi=500)
    default_s = plt.rcParams['lines.markersize'] ** 2
    ax[0].plot(recon_loss)
    ax[0].set_title(f'recon loss wrt current: {recon_level} (DCNN)')

    ax[1].plot(recon_loss_ideal)
    ax[1].set_title(f'recon loss wrt ideal: binary (DCNN) [{final_recon_loss_ideal:.3f}]')

    # reg_loss = np.log(reg_loss)
    ax[2].plot(reg_loss)
    ax[2].set_title(f'reg loss (DCNN)')
    plt.suptitle(f'Target: {recon_level}, inner-loop: {inner_loop_epochs}')

    ax[3].plot(percent_zero)
    ax[3].set_title(f'percentage of zeroed attn weights (DCNN) [{final_percent_zero:.3f}]')

    plt.tight_layout()
    plt.savefig(f'results/{attn_config_version}/losses_type{problem_type}_run{run}_{recon_level}.png')
    plt.close()


def viz_cluster_params(
        attn_config_version,
        problem_type,
        recon_level,
        run
        ):
    """
    Visulize cluster params change over time.
    1. How lambdas change.
    2. How centers change.
    """
    # 1. lambdas ----
    # (744,)
    alphas = np.load(
        f'results/{attn_config_version}/all_alphas_type{problem_type}_run{run}_{recon_level}.npy'
    )
    print(f'alphas.shape = {alphas.shape}')
    
    num_steps = int(len(alphas) / 3)
    # (3, num_steps)
    alphas = alphas.reshape((num_steps, 3)).T

    fig, ax = plt.subplots(dpi=300)
    for i in range(alphas.shape[0]):
        alphas_row_i = alphas[i, :]
        ax.plot(
            range(num_steps), alphas_row_i, 
            # s=plt.rcParams['lines.markersize'] / 8,
            label=f'dim {i}')

    ax.set_title(f'alphas tuning')
    ax.set_xlabel(f'num of epochs (exc first epoch)')
    ax.set_xticks(range(0, num_steps, 8 * 5))
    ax.set_xticklabels(range(0, 31, 5))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/{attn_config_version}/alphas_type{problem_type}_run{run}_{recon_level}.png')
    plt.close()
    print('[Check] saved alphas figure.')

    # 2. centers ----
    # centers = np.load(
    #     f'results/{attn_config_version}/all_centers_type{problem_type}_run{run}_{recon_level}.npy'
    # )
    # # print(centers.shape) # (24 * 8 * 31, )
    # # every step has 24 (3 * 8 clusters) values
    # num_clusters = 8
    # num_dims = 3
    # num_steps = int(len(centers) / (num_clusters * num_dims))
    # # (steps, 24)
    # centers = centers.reshape((num_steps, num_clusters * num_dims)).T

    # num_cols = 4
    # num_rows = int(num_clusters / num_cols)
    # fig, ax = plt.subplots(num_rows, num_cols)
    # for i in range(num_clusters):
    #     # extract
    #     # (3, steps)
    #     center_i_overtime = centers[i * num_dims : (i+1) * num_dims, :]
    #     # plot
    #     row_idx = i // num_cols
    #     col_idx = i % num_cols
    #     # plot each dim of a cluster at a time
    #     for dim in range(num_dims):
    #         center_i_dim_overtime = center_i_overtime[dim, :]
    #         ax[row_idx, col_idx].plot(center_i_dim_overtime, label=f'dim{dim}')
    #     center_i_final_loc = np.around(center_i_overtime[:, -1], 1)
    #     ax[row_idx, col_idx].set_title(f'cluster #{i}\n{center_i_final_loc}')
    #     ax[row_idx, col_idx].get_xaxis().set_visible(False)
    #     ax[row_idx, col_idx].set_ylim([-0.05, 1.05])
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(f'results/{attn_config_version}/centers_type{problem_type}_run{run}_{recon_level}.png')
    # plt.close()
    # print(f'[Check] saved centers figure.')


def examine_clustering_learning_curves(
        attn_config_version, recon_level='cluster'):
    """
    Just sanity check that the lc 
    of types are expected.
    """
    num_types = 6
    colors = ['blue', 'orange', 'black', 'green', 'red', 'cyan']
    fig, ax = plt.subplots()
    trapz_areas = np.empty(num_types)
    for i in range(num_types):
        problem_type = i + 1
        lc = np.load(
            f'results/{attn_config_version}/lc_type{problem_type}_{recon_level}.npy'
        )
        trapz_areas[i] = np.round(np.trapz(lc), 3)
        ax.errorbar(
            range(lc.shape[0]), 
            lc, 
            color=colors[i],
            label=f'Type {problem_type}',
        )
    print(f'[Results] trapzoidal areas = ', trapz_areas)
    plt.legend()
    plt.title(f'{trapz_areas}')
    plt.tight_layout()
    plt.savefig(f'results/{attn_config_version}/lc.png')


def find_canonical_runs(
        attn_config_version,
        canonical_runs_only=True):
    """
    Record the runs that produce canonical solutions
    for each problem type. 
    Specificially, we check the saved `mask_non_recruit`

    if canonical_solutions:
        only return runs that yield canonical solutions.
    """
    num_types = 6
    results_path = f'results/{attn_config_version}'
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    num_runs = attn_config['num_runs']
    type2cluster = {
        1: 2, 2: 4,
        3: 6, 4: 6, 5: 6,
        6: 8}

    from collections import defaultdict
    canonical_runs = defaultdict(list)

    for i in range(num_types):
        problem_type = i + 1

        for run in range(num_runs):

            mask_non_recruit = np.load(
                f'results/{attn_config_version}/mask_non_recruit_type{problem_type}_run{run}_cluster.npy')
            num_nonzero = len(np.nonzero(mask_non_recruit)[0])
            if canonical_runs_only:
                if num_nonzero == type2cluster[problem_type]:
                    canonical_runs[i].append(run)
            else:
                canonical_runs[i].append(run)
        print(f'Type {problem_type}, has {len(canonical_runs[i])}/{num_runs} canonical solutions')
    return canonical_runs


def canonical_runs_correspondence_to_attn_n_binary(
        attn_config_version):
    """
    Within canonical solutions, look at each run's 
    corresponding alphas, zero% and binary recon level.

    This is to better understand the source of variation 
    in the final zero% and binary recon to the relationship
    of cluster solutions and alphas.
    """
    num_types = 6
    results_path = f'results/{attn_config_version}'
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    num_runs = attn_config['num_runs']
    type2cluster = {
        1: 2, 2: 4,
        3: 6, 4: 6, 5: 6,
        6: 8}

    from collections import defaultdict
    canonical_runs = defaultdict(list)

    for i in range(num_types):
        problem_type = i + 1

        problem_type = 3

        print(f'********* Type {problem_type} *********')
        for run in range(num_runs):
            mask_non_recruit = np.load(
                f'results/{attn_config_version}/mask_non_recruit_type{problem_type}_run{run}_cluster.npy')
            num_nonzero = len(np.nonzero(mask_non_recruit)[0])
            if num_nonzero == type2cluster[problem_type]:
                alphas = f'{results_path}/all_alphas_type{problem_type}_run{run}_cluster.npy'
                zero_attn = f'{results_path}/all_percent_zero_attn_type{problem_type}_run{run}_cluster.npy'
                binary_recon = f'{results_path}/all_recon_loss_ideal_type{problem_type}_run{run}_cluster.npy'
                cluster_targets = f'clustering/results/clustering_v1/cluster_actv_targets_{problem_type}_248_{run}.npy'
                centers = f'{results_path}/all_centers_type{problem_type}_run{run}_cluster.npy'
                centers_flat = np.load(centers)[-8*3:]
                centers_notflat = []
                for idx in range(0, len(centers_flat), 3):
                    centers = centers_flat[idx :  idx+3]
                    centers_notflat.append(list(np.round(centers, 2)))
                print(f'centers: {centers_notflat}')
                print(f'cluster targets: \n{np.load(cluster_targets)}')
                print(f'{mask_non_recruit}')
                print(f'alphas: {list(np.round(np.load(alphas)[-3:], 2))}')
                print(f'%attn=0: {np.load(zero_attn)[-1]:.2f}')
                print(f'binary recon: {np.mean(np.load(binary_recon)[-3:]):.2f}')
            print('----------------------------------------------')
        print('----------------------------------------------')
        exit()
  

def compare_across_types_V3(
        attn_config_version, 
        canonical_runs_only=False, 
        threshold=[0, 0, 0], 
        counterbalancing=True):
    """
    For each type's results, here we split them 
    based on the attention solutions.

    E.g. Type1 has two attn solutions by focusing on
    dim1 and dim2 or just focusing on dim1. We will plot
    the results separately for each case.

    NOTE: we do not consider non-canonical vs canonical 
    for now. One reason being that the difference causing 
    by different attention solutions seem bigger.

    NOTE: there are cases where some dimensions are not 
    fully attended, we use a `threshold` parameter to guard 
    these dimensions. 
    E.g. if threshold [0, 0, 0], alphas = [0.33, 0.1, 0] is considered 
            strategy type [True, True, False]
         if threshold [0.1, 0.1, 0.1], the same alphas is considered 
            strategy type [True, False, False]
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    num_runs = attn_config['num_runs']
    num_types = 6
    num_dims = 3
    comparisons = ['zero_attn', 'binary_recon']
    results_path = f'results/{attn_config_version}'
    type2runs = find_canonical_runs(
        attn_config_version, canonical_runs_only=canonical_runs_only)

    for c in range(len(comparisons)):
        comparison = comparisons[c]

        # e.g. { problem_type: {(True, False, False): [metric1, metric2, ... ]} }
        type2strategy2metric = defaultdict(lambda: defaultdict(list))

        for z in range(num_types):
            problem_type = z + 1
            print(f'------------ problem_type = {problem_type} ------------')

            for run in type2runs[z]:
                
                # First grab the final metric
                if comparison == 'zero_attn':
                    metric_fpath = f'{results_path}/all_percent_zero_attn_type{problem_type}_run{run}_cluster.npy'
                    metric = np.load(metric_fpath)[-1]                    
                else:
                    metric_fpath = f'{results_path}/all_recon_loss_ideal_type{problem_type}_run{run}_cluster.npy'
                    metric = np.load(metric_fpath)[-num_dims : ]
                    if counterbalancing:
                        # rotate dimensions based on `rot_dims`
                        # notice here rotation changes the `meaning` of dimensions
                        # because we want to move the task diagnostic dimension to
                        # always be dim1 for plotting only.
                        counter_balancing_fpath = f'{results_path}/counter_balancing_type{problem_type}_run{run}_cluster.npy'
                        counter_balancing = np.load(counter_balancing_fpath, allow_pickle=True)
                        rot_dims = counter_balancing.item()['rot_dims']
                        k = counter_balancing.item()['k'][0]
                        # print(f'run={run}, rot_dims = {rot_dims}, k={k}')
                        if k != 2:
                            # no need to rotate if k == 2 as it is same as k == 0
                            # otherwise just switch based on index.
                            metric[rot_dims[1]], metric[rot_dims[0]] = metric[rot_dims[0]], metric[rot_dims[1]]

                # Second group metric based on attn strategy
                alphas_fpath = f'{results_path}/all_alphas_type{problem_type}_run{run}_cluster.npy'
                # get the final 3 alphas
                alphas = np.load(alphas_fpath)[-3:]
                alphas = alphas - np.array(threshold)

                # 1e-6 is the lower bound of alpha constraint.
                # use tuple instead of list because tuple is not mutable.
                strategy = tuple(alphas > 1.0e-6)
                # type2strategy2metric[problem_type][strategy].append(metric)
                # TODO: do this filtering or not?
                # filter out attn strategies that aren't typical for the 
                # associated problem type.
                if problem_type in [1]:
                    if np.sum(strategy) >= 2:
                        pass
                    else:
                        type2strategy2metric[problem_type][strategy].append(metric)
                elif problem_type in [2]:
                    if np.sum(strategy) == 3:
                        pass
                    else:
                        type2strategy2metric[problem_type][strategy].append(metric)
                else:
                    type2strategy2metric[problem_type][strategy].append(metric)

        # plot
        if comparison == 'zero_attn':
            fig, ax = plt.subplots()
            x_axis = np.linspace(0, 14, 7)

            colors = ['green', 'red', 'brown', 'orange', 'cyan', 'blue']
            average_metrics = []
            std_metrics = []
            for z in range(num_types):
                problem_type = z + 1
                print(f'--------- Type {problem_type} ---------')
                # e.g. {(True, False, False): [metric]}
                strategy2metric = type2strategy2metric[problem_type]
                strategies = list(strategy2metric.keys())
                print(f'strategies = {strategies}')
                num_strategies = len(strategies)
                print(f'num_strategies = {num_strategies}')

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
                        color=colors[z],
                    )
                # NOTE: Hacky way of getting legend correct.
                ax.errorbar(
                    x=x_ticks[i],
                    y=np.mean(metrics),
                    yerr=np.std(metrics),
                    fmt='o',
                    color=colors[z],
                    label=f'single strategy, type {problem_type}'
                )
                print(len(temp_collector))
                np.save(f'{results_path}/{comparison}_type{problem_type}_allStrategies.npy', temp_collector)
                average_metrics.append(np.mean(temp_collector))
                std_metrics.append(np.std(temp_collector))

            # plot bar of averaged over strategies.
            print(f'average_metrics = {average_metrics}')
            print(f'std_metrics = {std_metrics}')
            ax.errorbar(
                x=x_axis[:num_types]+0.5,
                y=average_metrics,
                yerr=std_metrics,
                fmt='*',
                color='k',
                ls='-',
                label='overall'
            )

            ax.set_xticks(x_axis[:num_types]+0.5)
            ax.set_xticklabels(
                [f'Type {problem_type}' for problem_type in range(1, num_types+1)])
            ax.set_ylim([-0.05, 1.05])
            ax.set_ylabel('percentage of zero attention weights')
            plt.tight_layout()
            plt.legend()
            plt.savefig(f'{results_path}/compare_types_percent_zero_canonical_runs{canonical_runs_only}.png')
            plt.close()

        elif comparison == 'binary_recon':
            num_cols = 2
            num_rows = int(num_types / num_cols)
            x_axis = np.linspace(0, 8, num_dims+1)
            fig, ax = plt.subplots(num_rows, num_cols)

            for z in range(num_types):
                problem_type = z + 1
                row_idx = z // num_cols
                col_idx = z % num_cols

                print(f'--------- Type {problem_type} ---------')
                # e.g. {(True, False, False): [ [metric_dim1, dim2. dim3], [dim1, dim2, dim3], .. ]}
                strategy2metric = type2strategy2metric[problem_type]
                strategies = list(strategy2metric.keys())
                print(f'strategies = {strategies}')
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

                        ax[row_idx, col_idx].errorbar(
                            x=x_ticks[i],
                            y=np.mean(metrics[:, dim]),
                            yerr=np.std(metrics[:, dim]),
                            fmt='o',
                            color=colors[dim],
                            label=f'single strategy, dim{dim}')

                average_metric = np.mean(
                    np.array(all_strategies_collector), axis=0)
                std_metric = np.std(
                    np.array(all_strategies_collector), axis=0
                )
                ax[row_idx, col_idx].errorbar(
                    x=x_axis[:num_dims],
                    y=average_metric,
                    yerr=std_metric,
                    fmt='*',
                    color='k',
                    label='overall'
                )
                ax[row_idx, col_idx].set_xticks([])
                ax[num_rows-1, col_idx].set_xticks(x_axis[:num_dims]+0.5)
                ax[num_rows-1, col_idx].set_xticklabels([f'dim{i+1}' for i in range(num_dims)])
                ax[row_idx, col_idx].set_ylim([-0.5, 9.5])
                ax[row_idx, 0].set_ylabel('binary recon loss')
                ax[row_idx, col_idx].set_title(f'Type {problem_type}')
            
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(
                f'{results_path}/compare_types_dimension_binary_recon_canonical_runs_only{canonical_runs_only}.png')


def stats_significance_of_zero_attn(attn_config_version):
    """
    Evaluate statistic significance across types 
    of the difference in percentage of zero low attn
    weights over runs & strategies.
    """
    results_path = f'results/{attn_config_version}'
    type1 = np.load(f'{results_path}/zero_attn_type1_allStrategies.npy')
    type2 = np.load(f'{results_path}/zero_attn_type2_allStrategies.npy')
    type6 = np.load(f'{results_path}/zero_attn_type6_allStrategies.npy')
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
    subject2dimension = pd.read_csv('featureinfo.txt', header=None).to_numpy()
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
    # will be used to sum up their RT through trials and runs,
    
    dim2rt = defaultdict(list)
    dim2acc = defaultdict(list)
    for dim in [1, 2, 3]:
        subject_ids = dim2subject[dim]
        
        for subject_id in subject_ids:
            data_dir = f'behaviour/subject_{subject_id}'
            
            # even subject Type 1 - study_2
            # odd subject Type 1 - study_3
            if int(subject_id) % 2 == 0:
                study_id = 2
            else:
                study_id = 3

            subject_rt = []
            # subject_correct = 0
            # subject_wrong = 0
            for run in runs:
                # print(f'------ dim={dim}, subject={subject_id}, run={run} ------')
                
                subject_correct = 0
                subject_wrong = 0
            
                fpath = f'{data_dir}/{subject_id}_study{study_id}_run{run}.txt'
                data = pd.read_csv(fpath, sep = "\t", header=None).to_numpy()
                    
                # locate nan RT within a run
                nan_i = []
                has_nan = False 
                data_exc_nan = []        
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
                
                # count num of correct and wrong this run.
                for acc in data[:, 9]:
                    if acc == 0:
                        subject_wrong += 1
                    else:
                        subject_correct += 1
                
                dim2acc[dim].append(subject_correct / (subject_correct + subject_wrong))
                
            dim2rt[dim].extend(subject_rt)
            
            
    # plot dim2rt
    fig, ax = plt.subplots(2)
    data_RT = []
    data_ACC = []
    RT_medians = []
    RT_means = []
    for dim in [1, 2, 3]:
        rt = dim2rt[dim]  # rt = [32 * 4 * 8] or [32 * 4 * 7]
        acc = dim2acc[dim]
        data_RT.append(rt)
        data_ACC.append(acc)
        RT_medians.append(np.median(rt))
        RT_means.append(np.mean(rt))
        
    assert len(data_RT) == 3

    # ------ statistic testing ------
    from scipy.stats import median_test, ttest_ind
    print(f'------ RT statistic testing ------')
    stats, p = median_test(data_RT[0], data_RT[1])[:2]
    print('dim1 vs dim2: ', f'stats={stats}, p={p}')
    
    stats, p = median_test(data_RT[0], data_RT[2])[:2]
    print('dim1 vs dim3: ', f'stats={stats}, p={p}')
    
    stats, p = median_test(data_RT[1], data_RT[2])[:2]
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
        
    
    # ------ plotting ------
    sns.violinplot(data=data_RT, ax=ax[0], inner=None)
    ax[0].set_xticks(range(len(data_RT)))
    ax[0].set_xticklabels(['dim1: leg', 'dim2: antenna', 'dim3: mandible'])
    ax[0].set_ylabel('Subject RT')
    ax[0].scatter(range(len(data_RT)), RT_medians, color='k', marker='*')
    
    sns.violinplot(data=data_ACC, ax=ax[1])
    ax[1].set_xticks(range(len(data_ACC)))
    ax[1].set_xticklabels(['dim1: leg', 'dim2: antenna', 'dim3: mandible'])
    ax[1].set_ylabel('Subject accuracy')
    
    plt.tight_layout()
    plt.savefig('subject_dimension_rt_acc.png')
        

def post_attn_actv_thru_time(attn_config_version):
    """
    Capture and analyze each Type's post-attn activations
    over the course of learning for each position of attn.

    Results from this eval will be compared to brain data.
    Notice only Type 1 & 2 are needed due to brain data availability.

    Impl:
    -----
        Given a `Type`, given a `attn_position`, we plug in 
        (epoch,i) level `attn_weights` to get post-attn actv 
        for all stimuli.
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version)
    dcnn_config_version = attn_config['dcnn_config_version']
    attn_positions = attn_config['attn_positions'].split(',')
    num_runs = attn_config['num_runs']
    num_blocks = attn_config['num_blocks']
    num_positions = len(attn_positions)
    num_types = 2
    num_stimuli = 8
    results_path = f'results/{attn_config_version}'
    
    # entire DCNN model with all attn layers
    model, preprocess_func = load_dcnn_model_V2(
        attn_config_version=attn_config_version, 
        dcnn_config_version=dcnn_config_version, 
        intermediate_input=False)

    for z in range(num_types):
        problem_type = z + 1

        for run in range(num_runs):
            
            dataset, _ = data_loader_V2(
                attn_config_version=attn_config_version,
                dcnn_config_version=dcnn_config_version,
                preprocess_func=preprocess_func,
                problem_type=problem_type,
                random_seed=run
            )
            
            batch_x = load_X_only(
                dataset=dataset, 
                attn_config_version=attn_config_version
            )

            # remember we do not have attn saved for epoch=0
            # tho easy to get as its just init.
            for epoch in range(1, num_blocks):
                
                for i in range(num_stimuli):
                                            
                    # set (epoch,i) level attn_weights 
                    fpath = f'{results_path}/' \
                            f'attn_weights_type{problem_type}_' \
                            f'run{run}_epoch{epoch}_i{i}_cluster.npy'
                    # attn_weights = {'layer': [weights], }
                    attn_weights = np.load(fpath, allow_pickle=True).item()
                    
                    for position_idx in range(num_positions):
                        attn_position = attn_positions[position_idx]
                        
                        print(f'\nType={problem_type}, run={run}, {attn_position}, epoch={epoch}, i={i}')
                        
                        # intercept DCNN at an attn position.
                        # and use this partial model to get 
                        # post-attn actv across learning course.
                        outputs = model.get_layer(f'post_attn_actv_{attn_position}').output
                        partial_model = Model(inputs=model.inputs, outputs=outputs)
                    
                        # NOTE: when eval involve multiple attn layers,
                        # we must also load all previous attn layer weights.
                        positions_require_loading = attn_positions[ : position_idx+1]
                        print(f'positions_require_loading = {positions_require_loading}')
                        
                        for position in positions_require_loading:
                            
                            layer_attn_weights = attn_weights[position]
                            partial_model.get_layer(
                                f'attn_factory_{position}').set_weights(
                                [layer_attn_weights]
                            )
                            
                        # get post-attn actv at this (epoch, i)
                        # shape -> (8, H, W, C)
                        post_attn_actv = partial_model(batch_x, training=False)
                        
                        
                        
                        print(
                            post_attn_actv.numpy().flatten()[
                                np.argsort(
                                    post_attn_actv.numpy().flatten()
                                    )[::-1][:5]])
                        
                        
def examine_subject_lc_and_attn_overtime(problem_types):
    """
    Plotting per subject (either human or model) lc using
    the best config and plot the attn weights overtime.
    """
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
        config_version = f'best_config_sub{sub}'
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
        
        # # plot attn weights overtime
        # visualize_attn_overtime(
        #     config_version=config_version,
        #     sub=sub,
        #     ax=ax3
        # )
        
        # plot hyper-params of this config on figure
        x_coord = 9
        y_coor = 0.6
        margin = 0.07
        lr = config['lr']
        ax2.text(x_coord, y_coor, f'lr={lr:.3f}')
        center_lr_multiplier = config['center_lr_multiplier']
        ax2.text(x_coord, y_coor-margin*1, f'center_lr={center_lr_multiplier * lr:.3f}')
        attn_lr_multiplier = config['attn_lr_multiplier']
        ax2.text(x_coord, y_coor-margin*2, f'attn_lr={attn_lr_multiplier * lr:.3f}')
        asso_lr_multiplier = config['asso_lr_multiplier']
        ax2.text(x_coord, y_coor-margin*3, f'asso_lr={asso_lr_multiplier * lr:.3f}')
        specificity = config['specificity']
        ax2.text(x_coord, y_coor-margin*4, f'specificity={specificity:.3f}')
        Phi = config['Phi']
        ax2.text(x_coord, y_coor-margin*5, f'Phi={Phi:.3f}')
        beta = config['beta']
        ax2.text(x_coord, y_coor-margin*6, f'beta={beta:.3f}')
        thr = config['thr']
        ax2.text(x_coord, y_coor-margin*8, f'thr={thr:.3f}')
        temp2 = config['temp2']
        ax2.text(x_coord, y_coor-margin*7, f'temp2={temp2:.3f}')
        
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
        
    examine_subject_lc_and_attn_overtime(problem_types=[1,2,6])
    
    # compare_across_types_V3(
    #     attn_config_version,
    #     canonical_runs_only=True,
    # )

    # stats_significance_of_zero_attn(attn_config_version)
    # histogram_low_attn_weights(attn_config_version)