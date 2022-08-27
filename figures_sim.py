import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np 
import pingouin as pg
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import rc
from utils import load_config

# rc('text', usetex=True)
# plt.rcParams['text.usetex']=True
color_palette = sns.color_palette("bright")
colors = [
    color_palette[1],   # 1
    color_palette[6],   # 2
    color_palette[3],   # 3
    color_palette[4],   # 4
    color_palette[8],   # 5
    color_palette[9],   # 6
]
plt.rcParams.update({'font.size': 12, 'font.weight': "bold"})
plt.rcParams["font.family"] = "Helvetica"

"""
Plot equivalent figures but using simulated model (not behav fitted).
"""

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
                f'{results_path}/mask_non_recruit_type{problem_type}_run{run}_cluster.npy')
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


def Fig_zero_attn(attn_config_version, threshold=[0, 0, 0]):

    def regression(collector, num_runs, problem_types):
        """Fitting linear regression models to per run %zero attn
        over problem_types. This way, we can read off the 
        regression coefficient on whether there is a down trend of 
        as task difficulty increases in order to 
        test statistic significance of our finding that the harder 
        the problem, the lower %zero attn.
        
        Impl:
        -----
            `collector` are saved in format:
                {
                'Type1': [run1_%, run2_%, ..],
                'Type2}: ...
                }
            
            To fit linear regression per run across types, we 
            convert the format to a matrix where each row is a run,
            and each column is a problem_type.
        """        
        group_results_by_runs = np.ones((num_runs, len(problem_types)))
        for z in range(len(problem_types)):
            problem_type = problem_types[z]
            # [run1_acc, run2_acc, ...]
            per_type_all_runs = collector[problem_type]
            for s in range(num_runs):
                group_results_by_runs[s, z] = per_type_all_runs[s]
        
        all_coefs = []
        for s in range(num_runs):
            X_run = problem_types
            # [run1_type1_%, run2_type2_%, ...]
            y_run = group_results_by_runs[s, :]
            coef = pg.linear_regression(X=X_run, y=y_run, coef_only=True)
            all_coefs.append(coef[-1])

        average_coef = np.mean(all_coefs)
        t, p = stats.ttest_1samp(all_coefs, popmean=0)
        print(f'average_coef={average_coef:.3f}', f't={t:.3f}', f'p={p}')
        return average_coef, t, p/2

    type2runs, _ = find_canonical_runs(
        attn_config_version, canonical_runs_only=True)

    problem_types = [1, 2, 3, 4, 5, 6]
    results_path = f'results/{attn_config_version}'
    fig, ax = plt.subplots(figsize=(6, 4))
    
    collector = defaultdict(list)
    # e.g. { problem_type: {(True, False, False): [metric1, metric2, ... ]} }
    type2strategy2metric = defaultdict(lambda: defaultdict(list))
    
    for z in range(len(problem_types)):
        problem_type = problem_types[z]

        for s in range(len(type2runs[z])):
            run = type2runs[z][s]
            
            # For %attn, we grab the last item
            metric_fpath = f'{results_path}/' \
                            f'all_percent_zero_attn_type{problem_type}_run{run}_cluster.npy'
            metric = np.load(metric_fpath)[-1]

            # Second group metric based on attn strategy
            alphas_fpath = f'{results_path}/' \
                            f'all_alphas_type{problem_type}_run{run}_cluster.npy'
            # get the final 3 alphas
            alphas = np.load(alphas_fpath)[-3:]
                            
            # 1e-6 is the lower bound of alpha constraint.
            # use tuple instead of list because tuple is not mutable.                    
            alphas = alphas - np.array(threshold)
            strategy = tuple(alphas > 1.0e-6)
            # type2strategy2metric[problem_type][strategy].append(metric)
            # collector[problem_type].append(metric)

            if problem_type in [1]:
                if np.sum(strategy) >= 2:
                    pass
                else:
                    type2strategy2metric[problem_type][strategy].append(metric)
                    collector[problem_type].append(metric)
            elif problem_type in [2]:
                if np.sum(strategy) == 3:
                    pass
                else:
                    type2strategy2metric[problem_type][strategy].append(metric)
                    collector[problem_type].append(metric)
            else:
                type2strategy2metric[problem_type][strategy].append(metric)
                collector[problem_type].append(metric)
    
    means = []
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        # e.g. {(True, False, False): [metric]}
        strategy2metric = type2strategy2metric[problem_type]
        strategies = list(strategy2metric.keys())
        num_strategies = len(strategies)

        temp_collector = []
        for i in range(num_strategies):
            strategy = strategies[i]
            metrics = strategy2metric[strategy]
            temp_collector.extend(metrics)

        mean = np.mean(temp_collector)
        means.append(mean)
        sem = stats.sem(temp_collector)
        ax.errorbar(
            x=z,
            y=mean,
            yerr=sem,
            fmt='o',
            capsize=3,
            color=colors[z],
            label=f'Type {problem_type}')

    # plot curve of means
    ax.plot(range(len(problem_types)), means, color='grey', ls='dashed')       
    ax.set_ylabel('Percentage of \nZero Attention Weights')
    ax.set_xticks([])
    ax.set_ylim([0, 0.6])
    
    # WARNING: problematic because modal run nums are different for each type.
    # regression(collector, , problem_types)

    plt.tight_layout()
    plt.legend()
    plt.savefig(f'figs/zero_attn_{attn_config_version}.png')
    plt.close()
    

def Fig_recon_n_decoding(attn_config_version):

    def recon_loss_by_type(attn_config_version):
        """
        Group recon loss by type, following conventions of 
        `brain_data/decoding.py`
        """
        problem_types = [1, 2, 6]
        type2runs, type_proportions = find_canonical_runs(
            attn_config_version, canonical_runs_only=True)
        num_dims = 3
        results_path = f'results/{attn_config_version}'
        
        # e.g. {problem_type: [sub02_loss, sub03_loss, ... ]}
        recon_loss_collector = defaultdict(list)
        for z in range(len(problem_types)):
            problem_type = problem_types[z]
            for run in type2runs[z]:
                # For binary recon, we grab the last 3 entries (each for a dim)
                fpath = f'{results_path}/' \
                        f'all_recon_loss_ideal_type{problem_type}_run{run}_cluster.npy'
                per_type_results = np.load(fpath)[-num_dims : ]
                recon_loss_collector[problem_type].append(np.mean(per_type_results))
                
        np.save(f'{results_path}/recon_loss_{attn_config_version}.npy', recon_loss_collector)
        
        # WARNING: same problem as above due to different num of canonical runs.
        # average_coef, t, p = recon_loss_by_type_regression(
        #     recon_loss_collector, 
        #     num_runs=, 
        #     problem_types=problem_types)
        
        # print(average_coef, t, p)
        # return average_coef, t, p

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

    def relate_recon_loss_to_decoding_error_errorbar(attn_config_version, num_runs, roi):
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
        
        results_collectors = [decoding_error_collector, recon_loss_collector]
        fig, axes = plt.subplots(1, 2)

        for i in range(len(results_collectors)):
            results_collector = results_collectors[i]
            xs = []
            ys = []
            yerrs = []
            for j in range(len(problem_types)):
                problem_type = problem_types[j]
                data_perType = results_collector[problem_type]
                xs.append(j)
                ys.append(np.mean(data_perType))
                yerrs.append(stats.sem(data_perType))
                
            axes[i].errorbar(
                x=xs,
                y=ys,
                yerr=yerrs,
                fmt='o',
                capsize=3,
                linestyle='dashed'
            )

            axes[i].set_xlabel('Problem Types')
            axes[i].set_xticks(range(len(problem_types)))
            axes[i].set_xticklabels(problem_types)
            if i == 0:
                axes[i].set_ylabel(f'{roi} Neural Stimulus Reconstruction Loss\n(1 - decoding accuracy)')
                axes[i].set_title(f'(A)')
            else:
                axes[i].set_ylabel('Model Stimulus Reconstruction Loss')
                axes[i].set_title(f'(B)')

        plt.tight_layout()
        plt.savefig(f'figs/recon_loss_decoding_error.png')

    def relate_recon_loss_to_decoding_error_errorbar_V2(attn_config_version, num_runs, roi):
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
            f'results/{attn_config_version}/recon_loss_{attn_config_version}.npy', 
            allow_pickle=True).ravel()[0]
        
        decoding_error_collector = np.load(
            f'brain_data/decoding_results/decoding_error_{num_runs}runs_{roi}.npy', 
            allow_pickle=True).ravel()[0]
        
        results_collectors = [decoding_error_collector, recon_loss_collector]
        fig, axes = plt.subplots(1, 2)

        for i in range(len(results_collectors)):
            results_collector = results_collectors[i]

            means = []
            for j in range(len(problem_types)):
                problem_type = problem_types[j]
                data_perType = results_collector[problem_type]

                mean = np.mean(data_perType)
                means.append(mean)
                sem =  stats.sem(data_perType)

                if i == 1:
                    label = f'Type {problem_type}'
                else:
                    label = None
                axes[i].errorbar(
                    x=j,
                    y=mean,
                    yerr=sem,
                    fmt='o',
                    capsize=3,
                    color=colors[j],
                    label=label
                )
            
            # plot curve of means
            axes[i].plot(range(len(problem_types)), means, color='grey', linestyle='dashed')
            axes[i].set_xticks([])
            if i == 0:
                axes[i].set_ylabel(f'{roi} Neural Stimulus Reconstruction Loss\n(1 - decoding accuracy)')
                axes[i].set_title(f'(A) Brain')
            else:
                axes[i].set_ylabel('Model Stimulus Reconstruction Loss')
                axes[i].set_title(f'(B) Model')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/recon_loss_decoding_error_{attn_config_version}.png')

    recon_loss_by_type(
        attn_config_version=attn_config_version
    )
    relate_recon_loss_to_decoding_error_errorbar_V2(
        attn_config_version=attn_config_version, num_runs=3, roi='LOC'
    )


def Fig_binary_recon(
        attn_config_version, 
        canonical_runs_only=True, 
        threshold=[0, 0, 0], 
        counterbalancing=True):
    num_types = 6
    num_dims = 3
    results_path = f'results/{attn_config_version}'
    type2runs, _ = find_canonical_runs(attn_config_version, canonical_runs_only=canonical_runs_only)

    # e.g. { problem_type: {(True, False, False): [metric1, metric2, ... ]} }
    type2strategy2metric = defaultdict(lambda: defaultdict(list))
    for z in range(num_types):
        problem_type = z + 1
        print(f'------------ problem_type = {problem_type} ------------')

        for run in type2runs[z]:
            
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
                if k != 2:
                    # no need to rotate if k == 2 as it is same as k == 0
                    # otherwise just switch based on index.
                    metric[rot_dims[1]], metric[rot_dims[0]] = metric[rot_dims[0]], metric[rot_dims[1]]
            
            # if problem_type == 1:
            #     conversion_order = [0, 1, 2]
            #     conversion_order[1:] = np.random.choice(
            #         conversion_order[1:], size=num_dims-1, replace=False
            #     )
            #     alphas = alphas[conversion_order]
            #     metric = metric[conversion_order]

            # NOTE, we do this so that the last 2 dims are the relevant dims
            # this is to be consistent with later Mack et al. dataset.
            # in simulation results, we had the first 2 dims as the relevant dims
            # for Type 2.
            if problem_type == 2:
                alphas = alphas[::-1]
                metric = metric[::-1]

            # Second group metric based on attn strategy
            alphas_fpath = f'{results_path}/all_alphas_type{problem_type}_run{run}_cluster.npy'
            # get the final 3 alphas
            alphas = np.load(alphas_fpath)[-3:]
            alphas = alphas - np.array(threshold)

            # 1e-6 is the lower bound of alpha constraint.
            # use tuple instead of list because tuple is not mutable.
            strategy = tuple(alphas > 1.0e-6)
            type2strategy2metric[problem_type][strategy].append(metric)
            # TODO: do this filtering or not?
            # filter out attn strategies that aren't typical for the 
            # associated problem type.
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

    # plot
    num_cols = 2
    num_rows = int(num_types / num_cols)
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

        average_metric = np.mean(np.array(all_strategies_collector), axis=0)
        sem_metric = stats.sem(np.array(all_strategies_collector), axis=0)
        std_metric = np.std(np.array(all_strategies_collector), axis=0)
        ax[row_idx, col_idx].errorbar(
            x=range(num_dims),
            y=average_metric,
            yerr=std_metric,
            fmt='o',
            capsize=3,
            color=colors[z])

        ax[row_idx, col_idx].set_xticks([])
        ax[row_idx, col_idx].set_ylim([-0.1, 3])
        ax[row_idx, col_idx].set_title(f'Type {problem_type}')
        ax[-1, col_idx].set_xlabel('Abstract Dimension')

    ax[1, 0].set_ylabel('Reconstruction Loss')
    plt.tight_layout()
    plt.savefig(f'figs/binary_recon_{attn_config_version}.png')


def Fig_alphas_against_recon_V2(attn_config_version):
    """
    Look at overall how consistent are alphas corresponding to recon loss.
    Here, we focus on Type 1 and the relevant dimension. 
    It is less interesting just looking at the final because at the end, 
    we have relevant dim alpha=1 and recon=0 (subject to a few exceptions).
    To better visualize the results, we could have look at how the relationship
    between alpha and recon changes over time.
    Implementation-wise, what is tricky is that the saved alphas and recon_loss 
    are not in the same dimensionality because alpha is saved once every trial (or rp),
    but recon is saved once every inner-loop iteration and not saved at the first rp. But we 
    could use recon=0 for rp=0 because at first we know there is no recon loss.
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    problem_types = [6]
    num_dims = 3
    num_reps = 16
    results_path = f'results/{attn_config_version}'
    type2runs, type_proportions = find_canonical_runs(
        attn_config_version, canonical_runs_only=True)

    fig, ax1 = plt.subplots(1, 3, figsize=(10, 3))
    alpha_color = '#E98D6B'
    recon_color = '#AD1759'

    for idx in range(len(problem_types)):

        problem_type = problem_types[idx]
        num_runs = len(type2runs[idx])
        runs = type2runs[idx]

        for relevant_dim_index in [0, 1, 2]:
            relevant_dim_alphas = np.ones((num_reps, num_runs))
            relevant_dim_recons = np.ones((num_reps, num_runs))
            for rp in range(0, 16):
                for r in range(num_runs):
                    run = runs[r]

                    # (768,)
                    alphas = np.load(
                        f'results/{attn_config_version}/' \
                        f'all_alphas_type{problem_type}_run{run}_cluster.npy')
                    alphas = alphas.reshape(-1, 3)
                    per_rp_alphas = alphas[rp*8*2 : (rp+1)*8*2, :]  # (8*2, 3)
                    per_rp_alphas_average = np.mean(per_rp_alphas, axis=0)  # (3)
                    
                    binary_recon = np.load(
                        f'results/{attn_config_version}/' \
                        f'all_recon_loss_ideal_type{problem_type}_run{run}_cluster.npy')
                    binary_recon = binary_recon.reshape(-1, 3)
                    if rp == 0:  # because rp=0 recon not saved but we know it's zero.
                        per_rp_binary_recon_average = np.array([0, 0, 0])
                    else:
                        per_rp_binary_recons = binary_recon[(rp-1)*8*2*5 : (rp)*8*2*5, :]  # (8*2*30, 3)
                        per_rp_binary_recon_average = np.mean(per_rp_binary_recons, axis=0)  # (3)
                    
                    # *** reverse counterbalancing ***
                    # rotate dimensions based on `rot_dims`
                    # notice here rotation changes the `meaning` of dimensions
                    # because we want to move the task diagnostic dimension to
                    # always be dim1 for plotting only.
                    counter_balancing_fpath = f'{results_path}/counter_balancing_type{problem_type}_run{run}_cluster.npy'
                    counter_balancing = np.load(counter_balancing_fpath, allow_pickle=True)
                    rot_dims = counter_balancing.item()['rot_dims']
                    k = counter_balancing.item()['k'][0]
                    if k != 2:
                        # no need to rotate if k == 2 as it is same as k == 0
                        # otherwise just switch based on index.
                        per_rp_alphas_average[rot_dims[1]], \
                            per_rp_alphas_average[rot_dims[0]] = \
                                per_rp_alphas_average[rot_dims[0]], \
                                    per_rp_alphas_average[rot_dims[1]]

                        per_rp_binary_recon_average[rot_dims[1]], \
                            per_rp_binary_recon_average[rot_dims[0]] = \
                                per_rp_binary_recon_average[rot_dims[0]], \
                                    per_rp_binary_recon_average[rot_dims[1]]

                    # *** For type1, relevant is always `dim1`, ie [0] ***
                    relevant_dim_alphas[rp, r] = per_rp_alphas_average[relevant_dim_index]
                    relevant_dim_recons[rp, r] = per_rp_binary_recon_average[relevant_dim_index]
            
            mean_alpha_over_subs = np.mean(relevant_dim_alphas, axis=1)
            mean_recon_over_subs = np.mean(relevant_dim_recons, axis=1)
            sem_alpha_over_subs = stats.sem(relevant_dim_alphas, axis=1)
            sem_recon_over_subs = stats.sem(relevant_dim_recons, axis=1)

            ax1[relevant_dim_index].errorbar(
                np.arange(num_reps),
                mean_alpha_over_subs,
                yerr=sem_alpha_over_subs,
                color=alpha_color,
                marker='*',
                markersize=5,
                capsize=5,
            )
            ax1[relevant_dim_index].set_xlabel('Repetition')
            ax1[relevant_dim_index].set_xticks([0, 15])
            ax1[relevant_dim_index].set_xticklabels([1, 16])
            if relevant_dim_index in [1, 2]:
                ax1[relevant_dim_index].set_yticks([])
            ax1[relevant_dim_index].set_ylim([-0.05, 1.05])

            # 2nd y-axis
            ax2 = ax1[relevant_dim_index].twinx()
            if relevant_dim_index in [0, 1]:
                ax2.set_yticks([])
            ax2.set_ylim([-0.05, 3.5])
            ax2.errorbar(
                np.arange(num_reps),
                mean_recon_over_subs,
                yerr=sem_recon_over_subs,
                color=recon_color,
                marker='o',
                markersize=5,
                capsize=5,
            )
    
    ax1[0].set_ylabel('Attention Strength', color=alpha_color)
    ax1[0].tick_params(axis='y', labelcolor=alpha_color)
    ax2.set_ylabel('Information Loss', color=recon_color)
    ax2.tick_params(axis='y', labelcolor=recon_color)
    plt.tight_layout()
    plt.savefig(f'figs/errorbar_type{problem_type}_highAttn_vs_reconLoss.pdf')


if __name__ == '__main__':
    attn_config_version='v4a_naive-withNoise-entropy'
    
    # Fig_zero_attn(attn_config_version)

    # Fig_recon_n_decoding(attn_config_version)

    # Fig_binary_recon(attn_config_version)

    Fig_alphas_against_recon_V2(attn_config_version)