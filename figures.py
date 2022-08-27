import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np 
import pandas as pd
import pingouin as pg
import seaborn as sns
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import rc
from clustering import human
from utils import load_config

color_palette = sns.color_palette("flare")
colors = [color_palette[0], color_palette[2], color_palette[5]]
print(color_palette.as_hex())
plt.rcParams.update({'font.size': 12, 'font.weight': "bold"})
plt.rcParams["font.family"] = "Helvetica"
TypeConverter = {1: 'I', 2: 'II', 6: 'VI'}

"""
Plot figures that go into the paper.
"""

def Fig_zero_attn(attn_config_version, v, threshold=[0, 0, 0]):

    def regression(collector, num_subs, problem_types):
        """Fitting linear regression models to per subject %zero attn
        over problem_types. This way, we can read off the 
        regression coefficient on whether there is a down trend of 
        as task difficulty increases in order to 
        test statistic significance of our finding that the harder 
        the problem, the lower %zero attn.
        
        Impl:
        -----
            `collector` are saved in format:
                {
                'Type1': [sub02_%, sub03_%, ..],
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
            per_type_all_subjects = collector[problem_type]
            for s in range(num_subs):
                group_results_by_subject[s, z] = per_type_all_subjects[s]
        
        all_coefs = []
        for s in range(num_subs):
            X_sub = problem_types
            # [sub02_type1_%, sub02_type2_%, ...]
            y_sub = group_results_by_subject[s, :]
            coef = pg.linear_regression(X=X_sub, y=y_sub, coef_only=True)
            all_coefs.append(coef[-1])

        average_coef = np.mean(all_coefs)
        t, p = stats.ttest_1samp(all_coefs, popmean=0)
        print(f'all_coef={all_coefs}')
        print(f'average_coef={average_coef:.3f}', f't={t:.3f}', f'p={p}')
        return average_coef, t, p/2
    
    problem_types = [1, 2, 6]
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    results_path = 'results'
    fig, ax = plt.subplots(1, 2, figsize=(5, 2))
    
    collector = defaultdict(list)
    # e.g. { problem_type: {(True, False, False): [metric1, metric2, ... ]} }
    type2strategy2metric = defaultdict(lambda: defaultdict(list))
    
    type1_attn_weights = []
    type2_attn_weights = []
    type6_attn_weights = []
    for z in range(len(problem_types)):
        problem_type = problem_types[z]

        # for sub in subs:
        for s in range(num_subs):
            sub = subs[s]
            
            # For %attn, we grab the last item
            metric_fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                            f'all_percent_zero_attn_type{problem_type}_sub{sub}_cluster.npy'
            metric = np.load(metric_fpath)[-1]

            # Second group metric based on attn strategy
            alphas_fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                            f'all_alphas_type{problem_type}_sub{sub}_cluster.npy'
            # get the final 3 alphas
            alphas = np.load(alphas_fpath)[-3:]
                            
            # 1e-6 is the lower bound of alpha constraint.
            # use tuple instead of list because tuple is not mutable.                    
            alphas = alphas - np.array(threshold)
            strategy = tuple(alphas > 1.0e-6)
            type2strategy2metric[problem_type][strategy].append(metric)
            collector[problem_type].append(metric)
            
            # collect attn_weights themselves
            attn_config = load_config(
                component=None, 
                config_version=f'{attn_config_version}_sub{sub}_{v}'
            )
            attn_position = attn_config['attn_positions'].split(',')[0]   
            attn_weights = np.load(
                f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                f'attn_weights_type{problem_type}_sub{sub}_cluster.npy',
                allow_pickle=True
                ).ravel()[0][attn_position]

            if problem_type == 1:
                type1_attn_weights.extend(attn_weights)
            elif problem_type == 2:
                type2_attn_weights.extend(attn_weights)
            elif problem_type == 6:
                type6_attn_weights.extend(attn_weights)
    
    # plot hist and kde
    attn_weights_all = [type1_attn_weights, type2_attn_weights, type6_attn_weights]
    for z in range(len(problem_types)):
        sns.histplot(
            attn_weights_all[z], 
            ax=ax[0], 
            color=colors[z], 
            edgecolor=None, 
            bins=3,
            alpha=0.24,
            stat='density',
        )

        sns.kdeplot(
            attn_weights_all[z], 
            ax=ax[0], 
            color=colors[z], 
            label=f'Type{problem_types[z]}'
        )

    # plot %zero attn errorbars
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
        
        print(f'\n\nType {problem_type}')
        print('zero% = ', temp_collector)
        print(f'mean={mean}, sem={sem}')

        ax[1].errorbar(
            x=z,
            y=mean,
            yerr=sem,
            fmt='o',
            capsize=3,
            color=colors[z],
            label=f'Type {TypeConverter[problem_type]}')

    # info
    ax[0].set_xlabel('Attention Value')
    ax[0].set_ylabel('Density')
    ax[0].spines.right.set_visible(False)
    ax[0].spines.top.set_visible(False)
    ax[0].legend()

    ax[1].plot(range(len(problem_types)), means, color='grey', ls='dashed')       
    ax[1].set_ylabel('Zero Attention \nPercentage')
    ax[1].set_xticks([])
    ax[1].set_ylim([0, 0.6])
    ax[1].spines.right.set_visible(False)
    ax[1].spines.top.set_visible(False)
    ax[1].legend()

    regression(collector, num_subs, problem_types)
    plt.tight_layout()
    plt.savefig(f'figs/zero_attn.pdf')
    plt.close()
    print('plotted zero attn')
    

def Fig_recon_n_decoding(attn_config_version, v):

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
        
        print(average_coef, t, p)

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
        return average_coef, t, p/2

    def relate_recon_loss_to_decoding_error_errorbar_V2(attn_config_version, num_runs, roi, v):
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
        fig, axes = plt.subplots(1, 2, figsize=(5, 2))

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
                    label = f'Type {TypeConverter[problem_type]}'
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
                axes[i].set_title(f'Brain', fontweight='bold')
            else:
                axes[i].set_title(f'Model', fontweight='bold')
            axes[i].spines.right.set_visible(False)
            axes[i].spines.top.set_visible(False)
        
        axes[0].set_ylabel('Decoding Error', fontweight='bold')
        axes[1].set_ylabel('Information Loss', fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/recon_loss_decoding_error.pdf')

    recon_loss_by_type(
        attn_config_version=attn_config_version, v=v
    )
    relate_recon_loss_to_decoding_error_errorbar_V2(
        attn_config_version=attn_config_version, num_runs=3, roi='LOC', v=v
    )
    print('plotted recon and decoding')


def Fig_binary_recon(attn_config_version, v, threshold=[0, 0, 0]):
    """
    Plot the dimensional recon based on the abstract coding's 
        ground true relevant dimension. V1 was using sorted high-level
        attn weights, which should be almost the same subject to diff
        in the irrelevant dims.

    Impl:
    -----
        The high-attn dims correspond to DCNN coding which could be different
        from abstract coding. To re-sort the dimensions into abstract coding
        order (which has the first dim as relevant for Type 1 for example), we
        must figure out in abstract coding sense, what is the physical meaning
        of the relevant dims, and then we need to sort those dims based on 
        the DCNN fixed coding.

        E.g.,
        -------
        Sub04 for solving type 1 uses first abstract dimension, which means
        mouth. So when to plot recon loss, we need to sort the original recon
        loss which is in order [leg, antenna, mouth] to have mouth as the 
        first dim, and the rest two could be random.
    """
    problem_types = [1, 2, 6]
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    num_dims = 3
    results_path = 'results'
    fig, axes = plt.subplots(1, 3, figsize=(5, 2))
    color_palette = sns.color_palette("crest")
    colors = [
        color_palette[1],   # dim1
        color_palette[3],   # dim2
        color_palette[5],   # dim3
    ]

    # {'02': [2, 1, 3, 12, 12, 12], '03': ...}
    sub2assignment_n_scheme = human.Mappings().sub2assignment_n_scheme
    
    # e.g. { problem_type: {(True, False, False): [metric1, metric2, ... ]} }
    type2strategy2metric = defaultdict(lambda: defaultdict(list))      

    for z in range(len(problem_types)):
        problem_type = problem_types[z]

        # for sub in subs:
        for s in range(num_subs):
            sub = subs[s]
            
            # For binary recon, we grab the last 3 entries (each for a dim)
            metric_fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                            f'all_recon_loss_ideal_type{problem_type}_sub{sub}_cluster.npy'
            metric = np.load(metric_fpath)[-num_dims : ]

            # Second group metric based on attn strategy
            alphas_fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                            f'all_alphas_type{problem_type}_sub{sub}_cluster.npy'
            # get the final 3 alphas
            alphas = np.load(alphas_fpath)[-3:]
            
            # get order of physical meaning
            # e.g. [2, 1, 3] means for this subject, the first dim of the abstract 
            # coding is antenna. And since the original recon vector is fixed in 
            # the order of DCNN coding, which are [leg, antenna, mouth], we need
            # to sort the antenna dim into the first dim.
            sub_physical_order = np.array(sub2assignment_n_scheme[sub][:3])-1
            conversion_order = sub_physical_order
            if problem_type == 1:
                conversion_order[1:] = np.random.choice(
                    conversion_order[1:], size=num_dims-1, replace=False
                )

            alphas = alphas[conversion_order]
            metric = metric[conversion_order]
            
            # 1e-6 is the lower bound of alpha constraint.
            # use tuple instead of list because tuple is not mutable.                    
            alphas = alphas - np.array(threshold)
            strategy = tuple(alphas > 1.0e-6)
            type2strategy2metric[problem_type][strategy].append(metric)

    num_cols = 3
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

        if problem_type == 2:
            average_metric = average_metric[::-1]
            sem_metric = sem_metric[::-1]

        for c in range(num_dims):
            axes[col_idx].errorbar(
                x=c+1, 
                y=average_metric[c], 
                yerr=sem_metric[c],
                color=colors[c], 
                marker='o',
                capsize=3
            )

        axes[col_idx].set_xticks([0, 1, 2, 3, 4])
        axes[col_idx].set_xticklabels([])
        axes[col_idx].set_ylim([-0.1, 2])
        axes[col_idx].set_title(f'Type {TypeConverter[problem_type]}', fontweight='bold')
        axes[col_idx].spines.right.set_visible(False)
        axes[col_idx].spines.top.set_visible(False)
    
    axes[0].set_ylabel('Information Loss', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figs/binary_recon.pdf')
    print('plotted binary recon')


def Fig_high_attn(attn_config_version, v):
    """
    Final high-attn in clustering module across types and runs.
    """
    problem_types = [1, 2, 6]
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    TypeConverter = {1: 'I', 2: 'II', 6: 'VI'}
    num_dims = 3
    results_path = f'results'
    sub2assignment_n_scheme = human.Mappings().sub2assignment_n_scheme

    num_cols = 1
    num_rows = 3
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(10, 5))
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        print(f'------------ problem_type = {problem_type} ------------')

        # collect type-level all alphas
        alphas_per_type = np.empty((len(subs), num_dims))

        for s in range(len(subs)):
            sub = subs[s]
            alphas_fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                            f'all_alphas_type{problem_type}_sub{sub}_cluster.npy'
            # get the final 3 alphas
            alphas = np.load(alphas_fpath)[-3:]
            
            # get order of physical meaning
            # e.g. [2, 1, 3] means for this subject, the first dim of the abstract 
            # coding is antenna. And since the original recon vector is fixed in 
            # the order of DCNN coding, which are [leg, antenna, mouth], we need
            # to sort the antenna dim into the first dim.
            sub_physical_order = np.array(sub2assignment_n_scheme[sub][:3])-1
            conversion_order = sub_physical_order
            if problem_type == 1:
                conversion_order[1:] = np.random.choice(
                    conversion_order[1:], size=num_dims-1, replace=False
                )
            alphas = alphas[conversion_order]

            if problem_type == 2:
                alphas = alphas[::-1]

            alphas_per_type[s, :] = alphas
                
        # get mean and sem across runs
        mean_alphas = np.mean(alphas_per_type, axis=0)
        sem_alphas = stats.sem(alphas_per_type, axis=0)
        std_alphas = np.std(alphas_per_type, axis=0)
        print(f'mean_alphas = {mean_alphas}')
        print(f'sem_alphas = {sem_alphas}')

        # plot
        row_idx = z // num_cols
        color_palette = sns.color_palette("crest")
        colors = [
            color_palette[1],   # dim1
            color_palette[3],   # dim2
            color_palette[5],   # dim3
        ]
        sns.barplot(
            data=alphas_per_type,
            ax=ax[row_idx], 
            palette=colors
        )
        ax[row_idx].set_xticks([])
        ax[row_idx].set_ylim([-0.1, 1.2])
        ax[row_idx].set_yticks([0, 0.5, 1])
        ax[row_idx].set_yticklabels([0, 0.5, 1])
        ax[row_idx].spines.right.set_visible(False)
        ax[row_idx].spines.top.set_visible(False)
        ax[row_idx].set_title(f'Attention Strength', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'figs/alphas_{attn_config_version}.pdf')
    plt.close()


def Fig_high_attn_against_low_attn_V1(attn_config_version, v):
    """
    Plot percentage zero low-attn against compression of high-attn
    """
    problem_types = [1, 2, 6]
    num_types = len(problem_types)
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    results_path = 'results'
    fig, ax = plt.subplots(figsize=(5, 5))
    markers = ['o', 's', '^']

    # csv - we only need the very last compression of each type
    # which corresponds to the very last zero% low-attn
    fname = 'compression_results_repetition_level/high_attn.csv'
    with open(fname) as f:
        df = pd.read_csv(f)

    final_compression_scores = df.loc[df['learning_trial'] == 16]

    all_alphas = np.zeros((num_subs, num_types))
    all_zero_percents = np.zeros((num_subs, num_types))
    for z in range(len(problem_types)):
        problem_type = problem_types[z]

        per_type_final_compression_scores = \
            final_compression_scores.loc[
                final_compression_scores['problem_type'] == problem_type
            ]

        per_type_low_attn_percentages = []
        for s in range(num_subs):
            sub = subs[s]
            
            # For %attn, we grab the last item
            metric_fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                            f'all_percent_zero_attn_type{problem_type}_sub{sub}_cluster.npy'
            
            # (3600,) -> (15*8*30,)
            per_subj_low_attn_percent = np.load(metric_fpath)
            # take moving average (window=8*30), which corresponds to the 
            # final compression score which uses average over 8 trials alphas.
            per_subj_low_attn_percent_average = np.mean(per_subj_low_attn_percent[-30*8:])
            per_type_low_attn_percentages.append(per_subj_low_attn_percent_average)

            all_alphas[s, z] = per_type_final_compression_scores['compression_score'].values[s]
            all_zero_percents[s, z] = per_subj_low_attn_percent_average
        
        ax.scatter(
            per_type_low_attn_percentages, 
            per_type_final_compression_scores['compression_score'].values,
            color=colors[z],
            alpha=0.5,
            edgecolors='none',
            marker=markers[z],
            label=f'Type {TypeConverter[problem_type]}'
        )

    ax.set_xlabel('Peripheral Attention \nZero Proportion')
    ax.set_ylabel('Controller Attention \nCompression')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    ax.legend()
    plt.tight_layout()
    plt.savefig('figs/high_attn_against_low_attn.pdf')
    all_correlations = []
    for s in range(num_subs):
        r, p_value = stats.pearsonr(all_alphas[s, :], all_zero_percents[s, :])
        all_correlations.append(r)

    t, p = stats.ttest_1samp(all_correlations, popmean=0)
    print(f'avg corr={np.mean(all_correlations)}, t={t}, one-sided p={p/2}')
    # Thoughts: 
    # High-attn compression makes sense as we have 0, 0.4, 1 three levels to the 3 types.
    # Low-attn seems less ideal because often a high compression corresponds to a low zero attn.
    # However, I think this could be explained by the fact that, even controller doesn't need 
    # that feature, the peripheral can basically do whatever it wants, not nec turning off things.
    # Also, perhaps it is turning off things which have caused recon loss but not enough to bring
    # attn weight to absolute zero.


def Fig_high_attn_against_low_attn_V2(attn_config_version, v):
    """
    Plot percentage zero low-attn against compression of high-attn
    """
    problem_types = [1]
    num_subs = 23
    num_reps = 16
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    results_path = 'results'
    fig, ax1 = plt.subplots(figsize=(5, 5))
    markers = ['o', 's', '^']

    # csv - we only need the very last compression of each type
    # which corresponds to the very last zero% low-attn
    fname = 'compression_results_repetition_level/high_attn.csv'
    with open(fname) as f:
        df = pd.read_csv(f)

    compression_scores_collector = np.ones((num_reps, num_subs))
    zero_percent_collector = np.ones((num_reps, num_subs))
    for rp in range(num_reps):
        compression_scores = df.loc[df['learning_trial'] == rp+1]

        for z in range(len(problem_types)):
            problem_type = problem_types[z]

            per_type_compression_scores = \
                compression_scores.loc[
                    compression_scores['problem_type'] == problem_type
                ]

            for s in range(num_subs):
                sub = subs[s]
                
                # For %attn, we grab the last item
                metric_fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                                f'all_percent_zero_attn_type{problem_type}_sub{sub}_cluster.npy'
                
                # (3600,) -> (15*8*30,)
                per_subj_low_attn_percent = np.load(metric_fpath)
                if rp == 0:
                    per_subj_low_attn_percent_average = 0
                else:
                    # take moving average (window=8*30), which corresponds to the 
                    # final compression score which uses average over 8 trials alphas.
                    per_subj_low_attn_percent_average = np.mean(per_subj_low_attn_percent[(rp-1)*30*8:(rp)*30*8])
                zero_percent_collector[rp, s] = per_subj_low_attn_percent_average
            compression_scores_collector[rp, :] = per_type_compression_scores['compression_score'].values

        mean_compression_scores = np.mean(compression_scores_collector, axis=1)
        sem_compression_scores = stats.sem(compression_scores_collector, axis=1)
        mean_zero_percent = np.mean(zero_percent_collector, axis=1)
        sem_zero_percent = stats.sem(zero_percent_collector, axis=1)

    compression_color = 'k'
    zero_percent_color = 'r'

    ax1.errorbar(
        np.arange(num_reps),
        mean_compression_scores,
        yerr=sem_compression_scores,
        color=compression_color,
        marker='*',
        markersize=5,
        capsize=5,
    )

    ax2 = ax1.twinx()
    ax2.errorbar(
        np.arange(num_reps),
        mean_zero_percent,
        yerr=sem_zero_percent,
        color=zero_percent_color,
        marker='o',
        markersize=5,
        capsize=5,
    )
    
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xticks([0, 15])
    ax1.set_xticklabels(['1', '16'])
    ax1.set_xlabel('Repetition')
    ax1.set_ylabel('Compression Score')
    ax2.set_ylim([-0.05, 0.65])
    ax2.set_ylabel('Peripheral Attention \nZero Proportion', color=zero_percent_color)

    plt.tight_layout()
    plt.savefig('figs/high_attn_against_low_attn.png')


def Fig_alphas_against_recon_V1(attn_config_version, v):
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
    import matplotlib.colors as clr
    problem_types=[1]
    num_subs = 23
    num_reps = 16
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    sub2assignment_n_scheme = human.Mappings().sub2assignment_n_scheme
    
    fig, ax = plt.subplots()
    norm = clr.Normalize(vmin=0, vmax=15)
    colors = matplotlib.cm.get_cmap('Purples')

    all_alphas = np.zeros((num_subs, num_reps))
    all_recons = np.zeros((num_subs, num_reps))
    for rp in range(num_reps):
        for idx in range(len(problem_types)):
            problem_type = problem_types[idx]

            relevant_dim_alphas = []
            relevant_dim_recons = []
            for s in range(num_subs):
                sub = subs[s]
                # (384, 1) -> (16*8, 3)
                alphas = np.load(
                    f'results/{attn_config_version}_sub{sub}_{v}/' \
                    f'all_alphas_type{problem_type}_sub{sub}_cluster.npy')
                alphas = alphas.reshape(-1, 3)
                per_rp_alphas = alphas[rp*8 : (rp+1)*8, :]  # (8, 3)
                per_rp_alphas_average = np.mean(per_rp_alphas, axis=0)  # (3)
                
                # (10800, 1) -> (3600, 3) -> (15*8*30, 3)
                binary_recon = np.load(
                    f'results/{attn_config_version}_sub{sub}_{v}/' \
                    f'all_recon_loss_ideal_type{problem_type}_sub{sub}_cluster.npy')
                binary_recon = binary_recon.reshape(-1, 3)
                if rp == 0:  # because rp=0 recon not saved but we know it's zero.
                    per_rp_binary_recon_average = np.array([0, 0, 0])
                else:
                    per_rp_binary_recons = binary_recon[(rp-1)*8*30 : (rp)*8*30, :]  # (8*30, 3)
                    per_rp_binary_recon_average = np.mean(per_rp_binary_recons, axis=0)  # (3)
                
                # alphas and binary_recon are initially DCNN order.
                # But to plot the relevant dim (which is in the abstract sense), we need to
                # rearrange alphas and binary_recon such that the first dim is the relevant dim.
                # The order of conversion is provided by the assignment_n_scheme.
                # e.g. sub02, has [2, 1, 3] mapping which means [antenna, leg, mouth] was the 
                # order used during learning and antenna is the relevant dim.
                sub_physical_order = np.array(sub2assignment_n_scheme[sub][:3])-1
                conversion_order = sub_physical_order
                per_rp_alphas_average = per_rp_alphas_average[conversion_order]
                per_rp_binary_recon_average = per_rp_binary_recon_average[conversion_order]

                relevant_dim_alphas.append(per_rp_alphas_average[0])
                relevant_dim_recons.append(per_rp_binary_recon_average[0])

                all_alphas[s, rp] = per_rp_alphas_average[0]
                all_recons[s, rp] = per_rp_binary_recon_average[0]
            
            ax.scatter(
                relevant_dim_alphas, 
                np.log(relevant_dim_recons),
                color=colors(norm(rp)),
                alpha=0.3,
                edgecolors='none',
                marker='*',
                s=100,
            )
            
    ax.set_xlabel('Attention Strength')
    ax.set_ylabel('Information Loss (log scale)')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.tight_layout()
    plt.savefig(f'figs/scatter_overtime_highAttn_vs_reconLoss_{v}.pdf')

    # correlation t-test
    all_correlations = []
    for s in range(num_subs):
        r, p_value = stats.pearsonr(all_alphas[s, :], all_recons[s, :])
        all_correlations.append(r)

    t, p = stats.ttest_1samp(all_correlations, popmean=0)
    print(f'avg corr={np.mean(all_correlations)}, t={t}, one-sided p={p/2}')


def Fig_alphas_against_recon_V2(attn_config_version, v):
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
    problem_types = [6]
    num_subs = 23
    num_reps = 16
    num_dims = 3
    relevant_dim_indices = range(num_dims)
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    sub2assignment_n_scheme = human.Mappings().sub2assignment_n_scheme
    
    fig, ax1 = plt.subplots(1, 3, figsize=(10, 3))
    alpha_color = '#E98D6B'
    recon_color = '#AD1759'

    for idx in range(len(problem_types)):
        problem_type = problem_types[idx]
        if problem_type == 2:  # swap to be consistent with tradition.
            relevant_dim_indices = relevant_dim_indices[::-1]
        for i in range(len(relevant_dim_indices)):
            relevant_dim_index = relevant_dim_indices[i]
            relevant_dim_alphas = np.ones((num_reps, num_subs))
            relevant_dim_recons = np.ones((num_reps, num_subs))
            for rp in range(num_reps):
                for s in range(num_subs):
                    sub = subs[s]
                    # (384, 1) -> (16*8, 3)
                    alphas = np.load(
                        f'results/{attn_config_version}_sub{sub}_{v}/' \
                        f'all_alphas_type{problem_type}_sub{sub}_cluster.npy')
                    alphas = alphas.reshape(-1, 3)
                    per_rp_alphas = alphas[rp*8 : (rp+1)*8, :]  # (8, 3)
                    per_rp_alphas_average = np.mean(per_rp_alphas, axis=0)  # (3)
                    
                    # (10800, 1) -> (3600, 3) -> (15*8*30, 3)
                    binary_recon = np.load(
                        f'results/{attn_config_version}_sub{sub}_{v}/' \
                        f'all_recon_loss_ideal_type{problem_type}_sub{sub}_cluster.npy')
                    binary_recon = binary_recon.reshape(-1, 3)
                    if rp == 0:  # because rp=0 recon not saved but we know it's zero.
                        per_rp_binary_recon_average = np.array([0, 0, 0])
                    else:
                        per_rp_binary_recons = binary_recon[(rp-1)*8*30 : (rp)*8*30, :]  # (8*30, 3)
                        per_rp_binary_recon_average = np.mean(per_rp_binary_recons, axis=0)  # (3)
                    
                    # alphas and binary_recon are initially DCNN order.
                    # But to plot the relevant dim (which is in the abstract sense), we need to
                    # rearrange alphas and binary_recon such that the first dim is the relevant dim.
                    # The order of conversion is provided by the assignment_n_scheme.
                    # e.g. sub02, has [2, 1, 3] mapping which means [antenna, leg, mouth] was the 
                    # order used during learning and antenna is the relevant dim.
                    sub_physical_order = np.array(sub2assignment_n_scheme[sub][:3])-1
                    conversion_order = sub_physical_order
                    if problem_type == 1:
                        conversion_order[1:] = np.random.choice(
                            conversion_order[1:], size=num_dims-1, replace=False
                        )

                    per_rp_alphas_average = per_rp_alphas_average[conversion_order]
                    per_rp_binary_recon_average = per_rp_binary_recon_average[conversion_order]

                    relevant_dim_alphas[rp, s] = per_rp_alphas_average[relevant_dim_index]
                    relevant_dim_recons[rp, s] = per_rp_binary_recon_average[relevant_dim_index]
            
            mean_alpha_over_subs = np.mean(relevant_dim_alphas, axis=1)
            mean_recon_over_subs = np.mean(relevant_dim_recons, axis=1)
            sem_alpha_over_subs = stats.sem(relevant_dim_alphas, axis=1)
            sem_recon_over_subs = stats.sem(relevant_dim_recons, axis=1)

            ax1[i].errorbar(
                np.arange(num_reps),
                mean_alpha_over_subs,
                yerr=sem_alpha_over_subs,
                color=alpha_color,
                marker='*',
                markersize=5,
                capsize=5,
            )
            ax1[i].set_xlabel('Repetition')
            ax1[i].set_xticks([0, 15])
            ax1[i].set_xticklabels([1, 16])
            if i in [1, 2]:
                ax1[i].set_yticks([])
            ax1[i].set_ylim([-0.05, 1.05])

            ax2 = ax1[i].twinx()
            if i in [0, 1]:
                ax2.set_yticks([])
            ax2.set_ylim([-0.05, 1.05])
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
    plt.savefig(f'figs/errorbar_overtime_type{problem_type}_highAttn_vs_reconLoss_{v}.pdf')

    
if __name__ == '__main__':
    attn_config_version='hyper4100'
    v='fit-human-entropy-fast-nocarryover'
    
    # Fig_zero_attn(attn_config_version, v)

    # Fig_recon_n_decoding(attn_config_version, v)

    # Fig_binary_recon(attn_config_version, v)

    # Fig_high_attn(attn_config_version, v)

    # Fig_high_attn_against_low_attn_V1(attn_config_version, v)
    # Fig_high_attn_against_low_attn_V2(attn_config_version, v)
    # Fig_alphas_against_recon_V1(attn_config_version, v)
    # Fig_alphas_against_recon_V2(attn_config_version, v)
