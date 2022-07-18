import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np 
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import load_config

color_palette = sns.color_palette("bright")
colors = [
    color_palette[1],   # 1
    color_palette[6],   # 2
    color_palette[3],   # 3
    color_palette[4],   # 4
    color_palette[8],   # 5
    color_palette[9],   # 6
]
plt.rcParams.update({'font.size': 12})

"""
Evaluation routines.
"""

def examine_clustering_learning_curves(
        attn_config_version, recon_level='cluster'):
    """
    Just sanity check that the lc 
    of types are expected.
    """
    num_types = 6
    TypeConverter = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI'}
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    shj = [
        [0.211, 0.025, 0.003, 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0.],
        [0.378, 0.156, 0.083, 0.056, 0.031, 0.027, 0.028, 0.016,
        0.016, 0.008, 0., 0.002, 0.005, 0.003, 0.002, 0.],
        [0.459, 0.286, 0.223, 0.145, 0.081, 0.078, 0.063, 0.033,
        0.023, 0.016, 0.019, 0.009, 0.008, 0.013, 0.009, 0.013],
        [0.422, 0.295, 0.222, 0.172, 0.148, 0.109, 0.089, 0.062,
        0.025, 0.031, 0.019, 0.025, 0.005, 0., 0., 0.],
        [0.472, 0.331, 0.23, 0.139, 0.106, 0.081, 0.067,
        0.078, 0.048, 0.045, 0.05, 0.036, 0.031, 0.027, 0.016, 0.014],
        [0.498, 0.341, 0.284, 0.245, 0.217, 0.192, 0.192, 0.177,
        0.172, 0.128, 0.139, 0.117, 0.103, 0.098, 0.106, 0.106]]
    
    for i in range(num_types):
        problem_type = i + 1
        ax[0].plot(shj[i], label=TypeConverter[problem_type], color=colors[i])

    for i in range(num_types):
        problem_type = i + 1
        lc = np.load(
            f'results/{attn_config_version}/lc_type{problem_type}_{recon_level}.npy'
        )
        ax[1].errorbar(
            range(lc.shape[0]), 
            lc, 
            color=colors[i],
            label=f'Type {TypeConverter[problem_type]}',
        )

    ax[0].set_title('Human')
    ax[0].set_xticks(range(0, len(shj[0]), 4))
    ax[0].set_xticklabels([1, 2, 3, 4])
    ax[0].set_xlabel('Learning Block')
    ax[0].set_ylabel('Probability of Error')

    ax[1].set_title('Model')
    ax[1].set_xticks(range(0, lc.shape[0], 8))
    ax[1].set_xticklabels([1, 2, 3, 4])
    ax[1].set_xlabel('Learning Block')    
    ax[1].get_yaxis().set_visible(False)

    plt.legend()
    plt.suptitle('(A)')
    plt.tight_layout()
    plt.savefig(f'results/{attn_config_version}/lc.png')
    plt.savefig(f'figs/lc_{attn_config_version}.png')


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
    type_proportions = {}

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
        
        proportion = np.round(len(canonical_runs[i]) / num_runs, 3)
        type_proportions[i] = proportion
        print(f'Type {problem_type}, has {len(canonical_runs[i])}/{num_runs} canonical solutions')

    return canonical_runs, type_proportions
  

def compare_across_types_V3(
        attn_config_version, 
        canonical_runs_only=True, 
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
    type2runs, type_proportions = find_canonical_runs(
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
            sem_metrics = []
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
                    # ax.errorbar(
                    #     x=x_ticks[i],
                    #     y=np.mean(metrics),
                    #     # yerr=np.std(metrics),
                    #     yerr=stats.sem(metrics),
                    #     fmt='o',
                    #     color=colors[z],
                    # )

                # NOTE: Hacky way of getting legend correct.
                # ax.errorbar(
                #     x=x_ticks[i],
                #     y=np.mean(metrics),
                #     # yerr=np.std(metrics),
                #     yerr=stats.sem(metrics),
                #     fmt='o',
                #     color=colors[z],
                #     label=f'single strategy, type {problem_type}'
                # )

                print(len(temp_collector))
                np.save(f'{results_path}/{comparison}_type{problem_type}_allStrategies.npy', temp_collector)
                average_metrics.append(np.mean(temp_collector))
                std_metrics.append(np.std(temp_collector))
                sem_metrics.append(stats.sem(temp_collector))

            # plot bar of averaged over strategies.
            print(f'average_metrics = {average_metrics}')
            print(f'std_metrics = {std_metrics}')
            ax.errorbar(
                x=x_axis[:num_types]+0.5,
                y=average_metrics,
                # yerr=std_metrics,
                yerr=sem_metrics,
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

                        # ax[row_idx, col_idx].errorbar(
                        #     x=x_ticks[i],
                        #     y=np.mean(metrics[:, dim]),
                        #     # yerr=np.std(metrics[:, dim]),
                        #     yerr=stats.sem(metrics[:, dim]),
                        #     fmt='o',
                        #     color=colors[dim],
                        #     label=f'single strategy, dim{dim}')

                all_strategies_collector = np.array(all_strategies_collector)
                average_metric = np.mean(
                    all_strategies_collector, axis=0)
                std_metric = np.std(
                    all_strategies_collector, axis=0
                )
                sem_metric = stats.sem(
                    all_strategies_collector, axis=0
                )

                # ax[row_idx, col_idx].errorbar(
                #     x=x_axis[:num_dims],
                #     y=average_metric,
                #     # yerr=std_metric,
                #     yerr=sem_metric,
                #     fmt='*',
                #     color='k',
                #     label='overall'
                # )

                sns.barplot(
                    data=all_strategies_collector, 
                    ax=ax[row_idx, col_idx], 
                    palette=colors
                )

                ax[row_idx, col_idx].set_xticks([])
                ax[num_rows-1, col_idx].set_xticks(range(num_dims))
                ax[num_rows-1, col_idx].set_xticklabels([f'dim{i+1}' for i in range(num_dims)])
                ax[row_idx, col_idx].set_ylim([-0.5, 5])
                ax[row_idx, 0].set_ylabel('binary recon loss')
                ax[row_idx, col_idx].set_title(f'Type {problem_type}')
            
            # plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(f'{results_path}/compare_types_dimension_binary_recon_canonical_runs_only{canonical_runs_only}.png')
                                            

def examine_high_attn_and_modal_solutions(attn_config_version, canonical_runs_only=True):
    """
    Final high-attn in clustering module across types and runs.
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    num_runs = attn_config['num_runs']
    problem_types = [1, 2, 3, 4, 5, 6]
    TypeConverter = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI'}
    num_dims = 3
    results_path = f'results/{attn_config_version}'
    type2runs, type_proportions = find_canonical_runs(
        attn_config_version, canonical_runs_only=canonical_runs_only)

    num_cols = 2
    num_rows = 3
    fig, ax = plt.subplots(num_rows, num_cols)
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        print(f'------------ problem_type = {problem_type} ------------')

        # collect type-level all alphas
        alphas_per_type = np.empty((len(type2runs[z]), num_dims))

        for i in range(len(type2runs[z])):
            run = type2runs[z][i]
            # Second group metric based on attn strategy
            alphas_fpath = f'{results_path}/all_alphas_type{problem_type}_run{run}_cluster.npy'
            # get the final 3 alphas
            alphas = np.round(np.load(alphas_fpath)[-3:], 3)

            counter_balancing_fpath = f'{results_path}/counter_balancing_type{problem_type}_run{run}_cluster.npy'
            counter_balancing = np.load(counter_balancing_fpath, allow_pickle=True)
            rot_dims = counter_balancing.item()['rot_dims']
            k = counter_balancing.item()['k'][0]
            if k != 2:
                # no need to rotate if k == 2 as it is same as k == 0
                # otherwise just switch based on index.
                alphas[rot_dims[1]], alphas[rot_dims[0]] = alphas[rot_dims[0]], alphas[rot_dims[1]]

            # NOTE, we do this so that the last 2 dims are the relevant dims
            # this is to be consistent with later Mack et al. dataset.
            # in simulation results, we had the first 2 dims as the relevant dims
            # for Type 2.
            if problem_type == 2:
                alphas = alphas[::-1]

            alphas_per_type[i, :] = alphas
                
        # get mean and sem across runs
        mean_alphas = np.mean(alphas_per_type, axis=0)
        sem_alphas = stats.sem(alphas_per_type, axis=0)
        std_alphas = np.std(alphas_per_type, axis=0)
        print(f'mean_alphas = {mean_alphas}')
        print(f'sem_alphas = {sem_alphas}')

        # plot
        row_idx = z // num_cols
        col_idx = z % num_cols

        # ax[row_idx, col_idx].errorbar(
        #     range(num_dims), 
        #     mean_alphas, 
        #     yerr=std_alphas, 
        #     color=colors[z],
        #     capsize=3,
        #     fmt='o',
        #     ls='none')

        sns.barplot(
            data=alphas_per_type,
            ax=ax[row_idx, col_idx], 
            palette=colors,
        )
        ax[row_idx, col_idx].set_xticks([])
        ax[-1, col_idx].set_xlabel('Abstract Dimensions')
        ax[row_idx, col_idx].set_ylim([-0.1, 1.2])
        ax[row_idx, col_idx].set_yticks([0, 0.5, 1])
        ax[row_idx, col_idx].set_yticklabels([0, 0.5, 1])
        ax[1, 0].set_ylabel(f'Attention Strength')
        ax[row_idx, col_idx].set_title(f'Type {TypeConverter[problem_type]}')
        ax[row_idx, col_idx].axhline(0.333, color='grey', ls='dashed')

    plt.tight_layout()
    plt.suptitle('(C)')
    plt.savefig(f'figs/alphas_{attn_config_version}.png')
    plt.close()

    # plot modal solution proportion as pie chart
    num_cols = 2
    num_rows = 3
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(4, 6))
    for z in range(len(problem_types)):
        problem_type = problem_types[z]

        row_idx = z // num_cols
        col_idx = z % num_cols

        sizes = [type_proportions[z], 1-type_proportions[z]]
        explode = (0.15, 0)

        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax[row_idx, col_idx].axis('equal')
        ax[row_idx, col_idx].set_title(f'Type {TypeConverter[problem_type]}')
    
        wedges, _, _ = \
            ax[row_idx, col_idx].pie(
                sizes, autopct='%1.1f%%', 
                shadow=True, 
                startangle=0, 
                explode=explode,
                colors=['orange', 'grey'],
                textprops=dict(color="k")
            )

    labels = ['Modal solutions', 'Other solutions']
    plt.legend(wedges, labels, bbox_to_anchor=(0.6, 0.1))
    plt.suptitle('(B)')
    plt.savefig(f'figs/modal_solution_proportion_{attn_config_version}.png')


def consistency_alphas_vs_recon(attn_config_version):
    """
    Look at overall how consistent are alphas corresponding to recon loss.
    Ideally, the reverse rank of alphas should be the same as the rank of recon
    because the higher the alppha, the lower the recon for this dimension.
    """
    problem_types=[1, 2, 6]
    num_types = len(problem_types)
    TypeConverter = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI'}
    num_dims = 3
    results_path = f'results/{attn_config_version}'
    type2runs, type_proportions = find_canonical_runs(
        attn_config_version, canonical_runs_only=True)

    all_rhos = []
    all_alphas = []
    all_recon = []
    for z in range(num_types):
        problem_type = z + 1

        for run in type2runs[z]:
            all_types_alphas = []
            all_types_recon = []
            for idx in range(len(problem_types)):
                problem_type = problem_types[idx]

                alphas = np.round(
                    np.load(
                    f'{results_path}/' \
                    f'all_alphas_type{problem_type}_run{run}_cluster.npy')[-3:], 3)
                
                binary_recon = np.round(
                    np.load(
                    f'{results_path}/' \
                    f'all_recon_loss_ideal_type{problem_type}_run{run}_cluster.npy')[-3:], 3)
                
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
    
    print(np.round(all_rhos, 3))
    print(np.mean(all_rhos))
    print(stats.mode(all_rhos))

    t, p = stats.ttest_1samp(all_rhos, popmean=0)
    print(f't={t:.3f}, p={p/2:.3f}')

                        
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    
    attn_config_version = 'v4_naive-withNoise'
    dcnn_config_version = 't1.vgg16.block4_pool.None.run1'
    
    # examine_clustering_learning_curves(attn_config_version)
    examine_high_attn_and_modal_solutions(attn_config_version)

    # consistency_alphas_vs_recon(attn_config_version)
    # compare_across_types_V3(attn_config_version)