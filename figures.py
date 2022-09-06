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
# plt.rcParams["font.family"] = "Helvetica"
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
            print(problem_type, sub, alphas)

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


def Fig_high_attn_against_low_attn_final(attn_config_version, v):
    """
    Plot percentage zero low-attn against compression of high-attn
    
    Final rep only.
    """
    problem_types = [1, 2, 6]
    num_types = len(problem_types)
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i not in [9]]   # 13 had flat human lc-type6
    num_subs = len(subs)
    results_path = 'results'
    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    sub2assignment_n_scheme = human.Mappings().sub2assignment_n_scheme

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
        per_type_compression_scores = []
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
            per_type_compression_scores.append(
                per_type_final_compression_scores.loc[
                per_type_final_compression_scores['subject'] == int(sub)
                ]['compression_score'].values[:]
            )

            # collect each subject across type high vs low.
            all_alphas[s, z] = per_type_final_compression_scores['compression_score'].values[s]
            all_zero_percents[s, z] = per_subj_low_attn_percent_average
        
        ax[z].scatter(
            per_type_low_attn_percentages, 
            per_type_compression_scores,
            color=colors[z],
            alpha=0.5,
            edgecolors='none',
            marker='o',
            s=100,
            label=f'Type {TypeConverter[problem_type]}'
        )
        ax[z].set_xlim([-0.05, 1.05])
        ax[z].set_ylim([-0.05, 1.05])
        ax[z].spines.right.set_visible(False)
        ax[z].spines.top.set_visible(False)
        ax[z].legend(loc="center")

    ax[1].set_xlabel('Peripheral Attention \n(Zero Proportion)')
    ax[0].set_ylabel('Controller Attention \n(Compression)')
    plt.tight_layout()
    plt.savefig(f'figs/scatter_typeALL_highAttn_vs_lowAttn_{v}_final.pdf')

    # corr analysis
    # per subject across types has r = (high v low)
    all_correlations = []
    for s in range(num_subs):
        r, p_value = stats.pearsonr(all_alphas[s, :], all_zero_percents[s, :])
        all_correlations.append(r)
    t, p = stats.ttest_1samp(all_correlations, popmean=0)
    all_correlations = np.round(all_correlations, 3)
    mean_r = np.mean(all_correlations)
    std_r = np.std(all_correlations)
    print(all_correlations)
    print(f'avg corr={mean_r:.2f}({std_r:.2f}), t={t:.2f}, one-sided p={p/2:.2f}')


def Fig_high_attn_against_low_attn_window(attn_config_version, v, corr):
    """
    Plot compression & %zero over time, where how timestep is defined depends on the
    `window`.

    if window == 'entire':
        there is just one big average over the entire learning.
    if window == 'half:
        first half and second half are separated averaged.
    ( window \in ['entire', 'half', 'run', 'rep'] )

    The choice of `corr` determines how correlation analysis is done.

    if corr == 'collapse_type_n_time':
        corr is computed across all types regardless of type or time
    if corr == 'keep_type_n_time':
        corr is computed within each type and seperate every time step
    if corr == 'collapse_type_keep_time':
    ( corr \in ['collapse_type_n_time', 'keep_type_n_time', 'collapse_type_keep_time'])
    """
    problem_types = [1, 2, 6]
    num_types = len(problem_types)
    num_subs = 23
    num_reps = 16
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i not in [9, 13]]
    num_subs = len(subs)
    results_path = 'results'
    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    fname = 'compression_results_repetition_level/high_attn.csv'
    with open(fname) as f:
        df = pd.read_csv(f)

    windows = ['entire', 'half', 'run', 'rep']
    for window in windows:
        print(f'\n\nwindow={window}, corr={corr}')

        if corr == 'collapse_type_n_time':
            all_alphas = []
            all_zero_percents = []

        elif corr == 'keep_type_n_time':
            all_alphas = defaultdict(lambda: defaultdict(list))
            all_zero_percents = defaultdict(lambda: defaultdict(list))

        elif corr == 'collapse_type_keep_time':
            all_alphas = defaultdict(list)
            all_zero_percents = defaultdict(list)

        for z in range(len(problem_types)):
            problem_type = problem_types[z]
            # first collect each sub and each rp
            # later based on window, rps within a window will be averaged.
            all_subs_compression_over_rps = np.ones((num_subs, num_reps))
            for rp in range(num_reps):
                per_rp_compression_scores = df.loc[df['learning_trial'] == rp+1]  # 1-indexed
                per_rp_per_type_compression_scores = \
                    per_rp_compression_scores.loc[
                        per_rp_compression_scores['problem_type'] == problem_type
                    ]
                for s in range(num_subs):
                    sub = subs[s]
                    per_sub_per_rp_per_type_compression_scores = \
                        per_rp_per_type_compression_scores.loc[
                            per_rp_per_type_compression_scores['subject'] == int(sub)
                        ]
                    all_subs_compression_over_rps[s, rp] = \
                        per_sub_per_rp_per_type_compression_scores['compression_score'].values[:]
                    
            # take average over a window of rps.
            if window == 'entire':
                num_rps_per_window = num_reps
            elif window == 'half':
                num_rps_per_window = 8
            elif window == 'run':
                num_rps_per_window = 4
            elif window == 'rep':
                num_rps_per_window = 1
            
            # collect each subject averaged over a window of rps.
            all_subs_average_compression_over_windows = np.zeros(
                (num_subs, int(num_reps/num_rps_per_window))
            )
            for rp in range(0, num_reps, num_rps_per_window):

                # for all subjects, average over a specific window of rps.
                # rp is the beginning of a window.
                all_subs_average_compression_over_windows[:, int(rp/num_rps_per_window)] = \
                    np.mean(
                        all_subs_compression_over_rps[:, rp:rp+num_rps_per_window], axis=1
                    )
                
                per_window_per_type_low_attn_percentages = []
                per_window_per_type_high_attn_compression = []
                for s in range(num_subs):
                    sub = subs[s]
                    # %zero
                    metric_fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                                    f'all_percent_zero_attn_type{problem_type}_sub{sub}_cluster.npy'
                    per_subj_low_attn_percent = np.load(metric_fpath)
                    # HACK: add 0 as the first rp %zero as not saved. But we know at the 
                    # beginning rps, %zero=0, even given noisy init of low-attn.
                    # ideal way is to actually save %zero for the 0th rp but shouldn't
                    # affect results.
                    hacky_array = np.zeros(30*8)
                    per_subj_low_attn_percent = np.concatenate((hacky_array, per_subj_low_attn_percent))
                    per_subj_low_attn_percent_average = np.mean(
                            per_subj_low_attn_percent[(rp)*30*8 : (rp+num_rps_per_window)*30*8]
                        )

                    per_window_per_type_low_attn_percentages.append(per_subj_low_attn_percent_average)
                    per_window_per_type_high_attn_compression.append(
                        all_subs_average_compression_over_windows[s, int(rp/num_rps_per_window)]
                    )

                    # for stats testing
                    # collect all regardless of type or window.
                    if corr == 'collapse_type_n_time':
                        all_alphas.append(
                            all_subs_average_compression_over_windows[s, int(rp/num_rps_per_window)]
                        )
                        all_zero_percents.append(per_subj_low_attn_percent_average)
                    
                    # collect respecting type 
                    elif corr == 'keep_type_n_time':
                        all_alphas[problem_type][rp].append(
                            all_subs_average_compression_over_windows[s, int(rp/num_rps_per_window)]
                        )
                        all_zero_percents[problem_type][rp].append(per_subj_low_attn_percent_average)
                    
                    # collect respecting time but not type
                    elif corr == 'collapse_type_keep_time':
                        all_alphas[rp].append(
                            all_subs_average_compression_over_windows[s, int(rp/num_rps_per_window)]
                        )
                        all_zero_percents[rp].append(per_subj_low_attn_percent_average)

                if window == 'entire':
                    rp = 15  # trick to get size
                ax[z].scatter(
                    per_window_per_type_low_attn_percentages,
                    per_window_per_type_high_attn_compression,
                    color=colors[z],
                    alpha=0.5,
                    edgecolors='none',
                    marker='o',
                    s=(rp+1)*20,
                )

            ax[z].set_xlim([-0.05, 1.05])
            ax[z].set_ylim([-0.05, 1.05])
            ax[z].spines.right.set_visible(False)
            ax[z].spines.top.set_visible(False)

        ax[1].set_xlabel('Peripheral Attention \n(Zero Proportion)')
        ax[0].set_ylabel('Controller Attention \n(Compression)')
        plt.tight_layout()
        plt.savefig(f'figs/scatter_typeALL_highAttn_vs_lowAttn_{v}_{window}.pdf')

        # **** corr analysis ****
        if corr == 'collapse_type_n_time':
            r, p = stats.pearsonr(all_alphas, all_zero_percents)
            print(f'[{window}] r={r:.3f}, p={p:.3f}')
        
        # stats testing within a type.
        elif corr == 'keep_type_n_time':
            if window == 'entire':
                for problem_type in problem_types:
                    for rp in range(0, num_reps, num_rps_per_window):
                        x = all_zero_percents[problem_type][rp]
                        y = all_alphas[problem_type][rp]
                        r, p = stats.spearmanr(x, y)
                        print(f'type={problem_type}, corr={r:.3f}, p={p:.3f}')
                
            elif window == 'half':
                for problem_type in problem_types:
                    for rp in range(0, num_reps, num_rps_per_window):
                        x = all_zero_percents[problem_type][rp]
                        y = all_alphas[problem_type][rp]
                        r, p = stats.spearmanr(x, y)
                        if rp == 0:
                            print(f'[early], type={problem_type}, corr={r:.3f}, p={p:.3f}')
                        else:
                            print(f'[late],  type={problem_type}, corr={r:.3f}, p={p:.3f}')
                            print('------------------------------------------------------')
            
            elif window == 'run':
                for problem_type in problem_types:
                    for rp in range(0, num_reps, num_rps_per_window):
                        x = all_zero_percents[problem_type][rp]
                        y = all_alphas[problem_type][rp]
                        r, p = stats.spearmanr(x, y)
                        if rp == 0:
                            print(f'[run1], type={problem_type}, corr={r:.3f}, p={p:.3f}')
                        elif rp == 4:
                            print(f'[run2], type={problem_type}, corr={r:.3f}, p={p:.3f}')
                        elif rp == 8:
                            print(f'[run3], type={problem_type}, corr={r:.3f}, p={p:.3f}')
                        elif rp == 12:
                            print(f'[run4], type={problem_type}, corr={r:.3f}, p={p:.3f}')
                            print('------------------------------------------------------')
            
            elif window == 'rep':
                for problem_type in problem_types:
                    for rp in range(0, num_reps, num_rps_per_window):
                        x = all_zero_percents[problem_type][rp]
                        y = all_alphas[problem_type][rp]
                        r, p = stats.spearmanr(x, y)
                        print(f'[rep{rp}], type={problem_type}, corr={r:.3f}, p={p:.3f}')
        
        elif corr == 'collapse_type_keep_time':
            if window == 'entire':
                for rp in range(0, num_reps, num_rps_per_window):
                    x = all_zero_percents[rp]
                    y = all_alphas[rp]
                    r, p = stats.spearmanr(x, y)
                    print(f'corr={r:.3f}, p={p:.3f}')
            
            elif window == 'half':
                for rp in range(0, num_reps, num_rps_per_window):
                    x = all_zero_percents[rp]
                    y = all_alphas[rp]
                    r, p = stats.spearmanr(x, y)
                    if rp == 0:
                        print(f'[early], corr={r:.3f}, p={p:.3f}')
                    else:
                        print(f'[late],  corr={r:.3f}, p={p:.3f}')
                        print('------------------------------------------------------')
            
            elif window == 'run':
                for rp in range(0, num_reps, num_rps_per_window):
                    x = all_zero_percents[rp]
                    y = all_alphas[rp]
                    r, p = stats.spearmanr(x, y)
                    if rp == 0:
                        print(f'[run1], corr={r:.3f}, p={p:.3f}')
                    elif rp == 4:
                        print(f'[run2], corr={r:.3f}, p={p:.3f}')
                    elif rp == 8:
                        print(f'[run3], corr={r:.3f}, p={p:.3f}')
                    elif rp == 12:
                        print(f'[run4], corr={r:.3f}, p={p:.3f}')
                        print('------------------------------------------------------')
            
            elif window == 'rep':
                for rp in range(0, num_reps, num_rps_per_window):
                    x = all_zero_percents[rp]
                    y = all_alphas[rp]
                    r, p = stats.spearmanr(x, y)
                    print(f'[rep{rp}], corr={r:.3f}, p={p:.3f}')


def Fig_high_attn_against_low_attn_window_oneplot(attn_config_version, v, corr):
    """
    Plot compression & %zero over time, where how timestep is defined depends on the
    `window`.

    if window == 'entire':
        there is just one big average over the entire learning.
    if window == 'half:
        first half and second half are separated averaged.
    ( window \in ['entire', 'half', 'run', 'rep'] )

    The choice of `corr` determines how correlation analysis is done.

    if corr == 'collapse_type_n_time':
        corr is computed across all types regardless of type or time
    if corr == 'keep_type_n_time':
        corr is computed within each type and seperate every time step
    if corr == 'collapse_type_keep_time':
    ( corr \in ['collapse_type_n_time', 'keep_type_n_time', 'collapse_type_keep_time'])
    """
    problem_types = [1, 2, 6]
    num_types = len(problem_types)
    num_subs = 23
    num_reps = 16
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i not in [9, 13]]
    num_subs = len(subs)
    results_path = 'results'
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    fname = 'compression_results_repetition_level/high_attn.csv'
    with open(fname) as f:
        df = pd.read_csv(f)

    windows = ['entire']
    for window in windows:
        print(f'\n\nwindow={window}, corr={corr}')

        if corr == 'collapse_type_n_time':
            all_alphas = []
            all_zero_percents = []

        elif corr == 'keep_type_n_time':
            all_alphas = defaultdict(lambda: defaultdict(list))
            all_zero_percents = defaultdict(lambda: defaultdict(list))

        elif corr == 'collapse_type_keep_time':
            all_alphas = defaultdict(list)
            all_zero_percents = defaultdict(list)

        for z in range(len(problem_types)):
            problem_type = problem_types[z]
            # first collect each sub and each rp
            # later based on window, rps within a window will be averaged.
            all_subs_compression_over_rps = np.ones((num_subs, num_reps))
            for rp in range(num_reps):
                per_rp_compression_scores = df.loc[df['learning_trial'] == rp+1]  # 1-indexed
                per_rp_per_type_compression_scores = \
                    per_rp_compression_scores.loc[
                        per_rp_compression_scores['problem_type'] == problem_type
                    ]
                for s in range(num_subs):
                    sub = subs[s]
                    per_sub_per_rp_per_type_compression_scores = \
                        per_rp_per_type_compression_scores.loc[
                            per_rp_per_type_compression_scores['subject'] == int(sub)
                        ]
                    all_subs_compression_over_rps[s, rp] = \
                        per_sub_per_rp_per_type_compression_scores['compression_score'].values[:]
                    
            # take average over a window of rps.
            if window == 'entire':
                num_rps_per_window = num_reps
            elif window == 'half':
                num_rps_per_window = 8
            elif window == 'run':
                num_rps_per_window = 4
            elif window == 'rep':
                num_rps_per_window = 1
            
            # collect each subject averaged over a window of rps.
            all_subs_average_compression_over_windows = np.zeros(
                (num_subs, int(num_reps/num_rps_per_window))
            )
            for rp in range(0, num_reps, num_rps_per_window):

                # for all subjects, average over a specific window of rps.
                # rp is the beginning of a window.
                all_subs_average_compression_over_windows[:, int(rp/num_rps_per_window)] = \
                    np.mean(
                        all_subs_compression_over_rps[:, rp:rp+num_rps_per_window], axis=1
                    )
                
                per_window_per_type_low_attn_percentages = []
                per_window_per_type_high_attn_compression = []
                for s in range(num_subs):
                    sub = subs[s]
                    # %zero
                    metric_fpath = f'{results_path}/{attn_config_version}_sub{sub}_{v}/' \
                                    f'all_percent_zero_attn_type{problem_type}_sub{sub}_cluster.npy'
                    per_subj_low_attn_percent = np.load(metric_fpath)
                    # HACK: add 0 as the first rp %zero as not saved. But we know at the 
                    # beginning rps, %zero=0, even given noisy init of low-attn.
                    # ideal way is to actually save %zero for the 0th rp but shouldn't
                    # affect results.
                    hacky_array = np.zeros(30*8)
                    per_subj_low_attn_percent = np.concatenate((hacky_array, per_subj_low_attn_percent))
                    per_subj_low_attn_percent_average = np.mean(
                            per_subj_low_attn_percent[(rp)*30*8 : (rp+num_rps_per_window)*30*8]
                        )

                    per_window_per_type_low_attn_percentages.append(per_subj_low_attn_percent_average)
                    per_window_per_type_high_attn_compression.append(
                        all_subs_average_compression_over_windows[s, int(rp/num_rps_per_window)]
                    )

                    # for stats testing
                    # collect all regardless of type or window.
                    if corr == 'collapse_type_n_time':
                        all_alphas.append(
                            all_subs_average_compression_over_windows[s, int(rp/num_rps_per_window)]
                        )
                        all_zero_percents.append(per_subj_low_attn_percent_average)
                    
                    # collect respecting type 
                    elif corr == 'keep_type_n_time':
                        all_alphas[problem_type][rp].append(
                            all_subs_average_compression_over_windows[s, int(rp/num_rps_per_window)]
                        )
                        all_zero_percents[problem_type][rp].append(per_subj_low_attn_percent_average)
                    
                    # collect respecting time but not type
                    elif corr == 'collapse_type_keep_time':
                        all_alphas[rp].append(
                            all_subs_average_compression_over_windows[s, int(rp/num_rps_per_window)]
                        )
                        all_zero_percents[rp].append(per_subj_low_attn_percent_average)

                if window == 'entire':
                    rp = 15  # trick to get size
                ax.scatter(
                    per_window_per_type_low_attn_percentages,
                    per_window_per_type_high_attn_compression,
                    color=colors[z],
                    # color='black',
                    alpha=0.5,
                    edgecolors='none',
                    marker='o',
                    label=f'Type {TypeConverter[problem_type]}',
                )

        ax.set_xlabel('Peripheral Attention \n(Zero Proportion)')
        ax.set_ylabel('Controller Attention \n(Compression)')
        ax.set_xlim([-0.05, np.round(np.max(all_zero_percents))])
        ax.set_ylim([-0.05, np.round(np.max(all_alphas))])
        ax.set_xticks([0, np.round(np.max(all_zero_percents))])
        ax.set_xticklabels([0, np.round(np.max(all_zero_percents))])
        ax.set_yticks([0, np.round(np.max(all_alphas))])
        ax.set_yticklabels([0, np.round(np.max(all_alphas))])
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.legend(loc='upper right')

        # plot best line fit
        bias, beta = pg.linear_regression(X=all_zero_percents, y=all_alphas, coef_only=True)
        
        x_coords = np.linspace(0, np.max(all_zero_percents)+0.05, 100)
        y_coords = bias + beta * x_coords
        plt.plot(x_coords, y_coords, color='grey', linewidth=2, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'figs/scatter_typeALL_highAttn_vs_lowAttn_{v}_{window}_oneplot.pdf')

        # **** corr analysis ****
        if corr == 'collapse_type_n_time':
            r, p = stats.pearsonr(all_alphas, all_zero_percents)
            print(f'[{window}] r={r:.3f}, p={p:.3f}')
        
        # stats testing within a type.
        elif corr == 'keep_type_n_time':
            if window == 'entire':
                for problem_type in problem_types:
                    for rp in range(0, num_reps, num_rps_per_window):
                        x = all_zero_percents[problem_type][rp]
                        y = all_alphas[problem_type][rp]
                        r, p = stats.spearmanr(x, y)
                        print(f'type={problem_type}, corr={r:.3f}, p={p:.3f}')
                
            elif window == 'half':
                for problem_type in problem_types:
                    for rp in range(0, num_reps, num_rps_per_window):
                        x = all_zero_percents[problem_type][rp]
                        y = all_alphas[problem_type][rp]
                        r, p = stats.spearmanr(x, y)
                        if rp == 0:
                            print(f'[early], type={problem_type}, corr={r:.3f}, p={p:.3f}')
                        else:
                            print(f'[late],  type={problem_type}, corr={r:.3f}, p={p:.3f}')
                            print('------------------------------------------------------')
            
            elif window == 'run':
                for problem_type in problem_types:
                    for rp in range(0, num_reps, num_rps_per_window):
                        x = all_zero_percents[problem_type][rp]
                        y = all_alphas[problem_type][rp]
                        r, p = stats.spearmanr(x, y)
                        if rp == 0:
                            print(f'[run1], type={problem_type}, corr={r:.3f}, p={p:.3f}')
                        elif rp == 4:
                            print(f'[run2], type={problem_type}, corr={r:.3f}, p={p:.3f}')
                        elif rp == 8:
                            print(f'[run3], type={problem_type}, corr={r:.3f}, p={p:.3f}')
                        elif rp == 12:
                            print(f'[run4], type={problem_type}, corr={r:.3f}, p={p:.3f}')
                            print('------------------------------------------------------')
            
            elif window == 'rep':
                for problem_type in problem_types:
                    for rp in range(0, num_reps, num_rps_per_window):
                        x = all_zero_percents[problem_type][rp]
                        y = all_alphas[problem_type][rp]
                        r, p = stats.spearmanr(x, y)
                        print(f'[rep{rp}], type={problem_type}, corr={r:.3f}, p={p:.3f}')
        
        elif corr == 'collapse_type_keep_time':
            if window == 'entire':
                for rp in range(0, num_reps, num_rps_per_window):
                    x = all_zero_percents[rp]
                    y = all_alphas[rp]
                    r, p = stats.spearmanr(x, y)
                    print(f'corr={r:.3f}, p={p:.3f}')
            
            elif window == 'half':
                for rp in range(0, num_reps, num_rps_per_window):
                    x = all_zero_percents[rp]
                    y = all_alphas[rp]
                    r, p = stats.spearmanr(x, y)
                    if rp == 0:
                        print(f'[early], corr={r:.3f}, p={p:.3f}')
                    else:
                        print(f'[late],  corr={r:.3f}, p={p:.3f}')
                        print('------------------------------------------------------')
            
            elif window == 'run':
                for rp in range(0, num_reps, num_rps_per_window):
                    x = all_zero_percents[rp]
                    y = all_alphas[rp]
                    r, p = stats.spearmanr(x, y)
                    if rp == 0:
                        print(f'[run1], corr={r:.3f}, p={p:.3f}')
                    elif rp == 4:
                        print(f'[run2], corr={r:.3f}, p={p:.3f}')
                    elif rp == 8:
                        print(f'[run3], corr={r:.3f}, p={p:.3f}')
                    elif rp == 12:
                        print(f'[run4], corr={r:.3f}, p={p:.3f}')
                        print('------------------------------------------------------')
            
            elif window == 'rep':
                for rp in range(0, num_reps, num_rps_per_window):
                    x = all_zero_percents[rp]
                    y = all_alphas[rp]
                    r, p = stats.spearmanr(x, y)
                    print(f'[rep{rp}], corr={r:.3f}, p={p:.3f}')

    

def Fig_high_attn_against_low_attn_V2(attn_config_version, v):
    """
    Over time, averaged subjects, errorbar.
    """
    problem_types = [1, 2, 6]
    num_subs = 23
    num_reps = 16
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    results_path = 'results'
    fig, ax1 = plt.subplots(1, 3, figsize=(10, 5))
    compression_color = '#E98D6B'
    zero_percent_color = '#AD1759'

    # csv - we only need the very last compression of each type
    # which corresponds to the very last zero% low-attn
    fname = 'compression_results_repetition_level/high_attn.csv'
    with open(fname) as f:
        df = pd.read_csv(f)

    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        compression_scores_collector = np.ones((num_reps, num_subs))
        zero_percent_collector = np.ones((num_reps, num_subs))

        for rp in range(num_reps):
            compression_scores = df.loc[df['learning_trial'] == rp+1]
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

        ax1[z].errorbar(
            np.arange(num_reps),
            mean_compression_scores,
            yerr=sem_compression_scores,
            color=compression_color,
            marker='o',
            markersize=5,
            capsize=5,
            alpha=0.75
        )
        if z in [1, 2]:
            ax1[z].set_yticks([])

        ax2 = ax1[z].twinx()
        ax2.set_ylim([-0.05, 0.65])
        if z < 2:
            ax2.set_yticks([])
        if z == 2:
            ax2.set_ylabel('Peripheral Attention \n(Zero Proportion)', color=zero_percent_color)
        ax2.errorbar(
            np.arange(num_reps),
            mean_zero_percent,
            yerr=sem_zero_percent,
            color=zero_percent_color,
            marker='o',
            markersize=5,
            capsize=5,
            alpha=0.75
        )
        ax1[z].set_ylim([-0.05, 1.05])
        ax1[z].set_xticks([0, 15])
        ax1[z].set_xticklabels(['1', '16'])
        ax1[1].set_xlabel('Repetition')
        ax1[0].set_ylabel('Controller Attention \n(Compression)', color=compression_color)
        ax1[z].tick_params(axis='y', labelcolor=compression_color)
        ax2.tick_params(axis='y', labelcolor=zero_percent_color)
        ax1[z].set_title(f'Type {TypeConverter[problem_type]}')

    plt.tight_layout()
    plt.savefig(f'figs/errorbar_high_attn_against_low_attn_typeALL_{v}.pdf')
    

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
    problem_types = [1, 2, 6]
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
                relevant_dim_recons,
                color=colors(norm(rp)),
                alpha=0.3,
                edgecolors='none',
                marker='*',
                s=100,
            )
            
    ax.set_xlabel('Attention Strength')
    ax.set_ylabel('Information Loss')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_title(f'Type {TypeConverter[problem_type]}')
    plt.tight_layout()
    plt.savefig(f'figs/scatter_overtime_type{problem_type}_highAttn_vs_reconLoss_{v}.pdf')

    # correlation t-test (each subject yields a corr)
    all_correlations = []
    for s in range(num_subs):
        r, _ = stats.pearsonr(all_alphas[s, :], all_recons[s, :])
        all_correlations.append(r)

    t, p = stats.ttest_1samp(all_correlations, popmean=0)
    mean_coef = np.mean(all_correlations)
    print(f'avg corr={mean_coef:.2f}, t={t:.2f}, one-sided p={p/2:.2f}')

    # correlation t-test (all subjects yield a corr)
    # here, we collapse the subject dimension so all subjects is as if it's a single subject
    # overtime:
    all_alphas_collapsed = np.mean(all_alphas, axis=0)
    all_recons_collapsed = np.mean(all_recons, axis=0)
    r, _ = stats.pearsonr(all_alphas_collapsed, all_recons_collapsed)
    print(f'corr={r:.2f}')


def Fig_alphas_against_recon_V1a(attn_config_version, v):
    """
    V1 -> V1a: Plot 3-by-3, excluding irrelevant dims. Optional overtime.
    """
    problem_types = [1, 2, 6]
    num_subs = 23
    num_reps = 16
    num_dims = 3
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    sub2assignment_n_scheme = human.Mappings().sub2assignment_n_scheme
    
    fig, ax1 = plt.subplots(3, 3, figsize=(5, 5))
    # alpha_color = '#E98D6B'

    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        if problem_type == 2:  # swap to be consistent with tradition.
            relevant_dim_indices = range(num_dims)[::-1]
        else:
            relevant_dim_indices = range(num_dims)

        for i in range(len(relevant_dim_indices)):
            relevant_dim_index = relevant_dim_indices[i]
            relevant_dim_alphas = np.ones((num_reps, num_subs))
            relevant_dim_recons = np.ones((num_reps, num_subs))
            for rp in range(num_reps):
                if rp < 15:
                    continue
                for s in range(num_subs):
                    sub = subs[s]
                    # (384, 1) -> (16*8, 3)
                    alphas = np.load(
                        f'results/{attn_config_version}_sub{sub}_{v}/' \
                        f'all_alphas_type{problem_type}_sub{sub}_cluster.npy')
                    alphas = alphas.reshape(-1, 3)

                    # **** experimental... ****
                    per_rp_alphas = alphas[rp*8 : (rp+1)*8, :]  # (8, 3)
                    # per_rp_alphas = alphas
                    per_rp_alphas_average = np.mean(per_rp_alphas, axis=0)  # (3)
                    ######
                    
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
                    print(
                        sub, problem_type, per_rp_alphas_average
                    )

                if z == 0 and z >= i:
                    ax1[z, i].axvline(1, 0.1, 0.8, c='grey', ls='--', alpha=0.5)
                elif z == 1 and z >= i:
                    ax1[z, i].axvline(0.5, 0.1, 0.8, c='grey', ls='--', alpha=0.5)
                elif z == 2 and z >= i:
                    ax1[z, i].axvline(0.333, 0.1, 0.8, c='grey', ls='--', alpha=0.5)
                
                if i in [1, 2]:
                    ax1[z, i].set_yticks([])
                ax1[z, i].set_xticks([0, 0.5, 1])
                ax1[z, i].set_xticklabels([0, 0.5, 1])
                ax1[z, i].set_xlim([-0.1, 1.1])
                ax1[z, i].set_ylim([-0.005, 0.01])
                ax1[z, 1].set_title(f'Type {TypeConverter[problem_type]}')
                ax1[z, i].spines.right.set_visible(False)
                ax1[z, i].spines.top.set_visible(False)

                if z < i:
                    ax1[z, i].set_axis_off()
                else:
                    ax1[z, i].scatter(
                        relevant_dim_alphas[rp, :],
                        relevant_dim_recons[rp, :],
                        color=colors[z],
                        marker='o', 
                        alpha=0.5,
                        edgecolor='none'
                    )
    
    ax1[1, 0].set_ylabel('Information Loss')
    ax1[-1, 1].set_xlabel('Attention Strength\n(relevant dimension)')
    plt.tight_layout()
    plt.savefig(f'figs/scatter_final_typeALL_highAttn_vs_reconLoss_{v}.pdf')


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
    problem_types = [1, 2, 6]
    num_subs = 23
    num_reps = 16
    num_dims = 3
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    sub2assignment_n_scheme = human.Mappings().sub2assignment_n_scheme
    
    fig, ax1 = plt.subplots(3, 3, figsize=(5, 5))
    alpha_color = '#E98D6B'
    recon_color = '#AD1759'

    for idx in range(len(problem_types)):
        problem_type = problem_types[idx]
        if problem_type == 2:  # swap to be consistent with tradition.
            relevant_dim_indices = range(num_dims)[::-1]
        else:
            relevant_dim_indices = range(num_dims)

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

            # ax1[idx, i].errorbar(
            #     np.arange(num_reps),
            #     mean_alpha_over_subs,
            #     yerr=sem_alpha_over_subs,
            #     color=alpha_color,
            #     marker='*',
            #     markersize=5,
            #     capsize=5,
            # )
            ax1[idx, i].plot(
                np.arange(num_reps), 
                mean_alpha_over_subs, 
                color=alpha_color, 
                marker='o', 
                markersize=5,
                alpha=0.75
            )
            ax1[idx, i].set_xticks([0, 15])
            ax1[idx, i].set_xticklabels([1, 16])
            ax1[idx, i].tick_params(axis='y', labelcolor=alpha_color)
            if i in [1, 2]:
                ax1[idx, i].set_yticks([])
            ax1[idx, i].set_ylim([-0.05, 1.05])

            ax2 = ax1[idx, i].twinx()
            ax2.tick_params(axis='y', labelcolor=recon_color)
            if i in [0, 1]:
                ax2.set_yticks([])
            if idx == 1 and i == 2:
                ax2.set_ylabel('Information Loss', color=recon_color)
            if i == 1:
                ax2.set_title(f'Type {TypeConverter[problem_type]}')
            ax2.set_ylim([-0.05, 1.05])
            # ax2.errorbar(
            #     np.arange(num_reps),
            #     mean_recon_over_subs,
            #     yerr=sem_recon_over_subs,
            #     color=recon_color,
            #     marker='o',
            #     markersize=5,
            #     capsize=5,
            # )
            ax2.plot(
                np.arange(num_reps), 
                mean_recon_over_subs, 
                color=recon_color, 
                marker='o', 
                markersize=5, 
                alpha=0.75
            )
    
    ax1[1, 0].set_ylabel('Attention Strength', color=alpha_color)
    ax1[-1, 1].set_xlabel('Repetition')
    plt.tight_layout()
    plt.savefig(f'figs/errorbar_overtime_typeALL_highAttn_vs_reconLoss_{v}.pdf')    


def Type1_relevant_dim_and_zero_percent(attn_config_version, v):
    """
    Test how %zero corresponds to which dim is relevant in Type1
    (similar to the RT analysis of humans)
    """
    problem_types = [1]
    num_types = len(problem_types)
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i not in [9]]   # 13 had flat human lc-type6
    num_subs = len(subs)
    results_path = 'results'
    sub2assignment_n_scheme = human.Mappings().sub2assignment_n_scheme
    dim2name = {0: 'leg', 1: 'antenna', 2: 'mouth'}

    relevant_dim2zero_percent = defaultdict(list)
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
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

            sub_physical_order = np.array(sub2assignment_n_scheme[sub][:3])-1
            relevant_dim_name = dim2name[sub_physical_order[0]]
            relevant_dim2zero_percent[relevant_dim_name].append(per_subj_low_attn_percent_average)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    dims_zero_percent = ['Dimension']
    zero_percent = ['zero_percent']

    for relevant_dim_index in range(3):
        relevant_dim_name = dim2name[relevant_dim_index]
        dims_zero_percent.extend([relevant_dim_name] * len(relevant_dim2zero_percent[relevant_dim_name]))
        zero_percent.extend(relevant_dim2zero_percent[relevant_dim_name])

    dims_zero_percent = np.array(dims_zero_percent)
    zero_percent = np.array(zero_percent)
    df_zero_percent = np.vstack((dims_zero_percent, zero_percent)).T
    pd.DataFrame(df_zero_percent).to_csv(
        f"df_zero_percent.csv", 
        index=False, header=False
    )
    df_zero_percent = pd.read_csv('df_zero_percent.csv', usecols=['Dimension', 'zero_percent'])

    print(df_zero_percent)

    sns.barplot(x='Dimension', y='zero_percent', data=df_zero_percent, ax=ax)
    plt.tight_layout()
    plt.savefig(f'figs/relevant_dim_v_zero_percent_type1_{v}.pdf')

    # stats testing
    from scipy.stats import ttest_ind
    t, p = ttest_ind(relevant_dim2zero_percent['leg'], relevant_dim2zero_percent['antenna'])[:2]
    print('leg vs antenna: ', f't={t:.3f}, p={p:.3f}')
    
    t, p = ttest_ind(relevant_dim2zero_percent['leg'], relevant_dim2zero_percent['mouth'])[:2]
    print('leg vs mouth: ', f't={t:.3f}, p={p:.3f}')
    
    t, p = ttest_ind(relevant_dim2zero_percent['mouth'], relevant_dim2zero_percent['antenna'])[:2]
    print('mouth vs antenna: ', f't={t:.3f}, p={p:.3f}')
    

def Type1_relevant_dim_and_info_loss(attn_config_version, v):
    """
    Test how LOC info loss corresponds to which dim is relevant in Type1
    (similar to the RT analysis of humans)
    """
    runs = [2, 3, 4]
    roi = 'LOC'
    num_runs = len(runs)
    problem_types = [1]
    num_types = len(problem_types)
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i not in [9]]   # 13 had flat human lc-type6
    num_subs = len(subs)
    results_path = 'results'
    sub2assignment_n_scheme = human.Mappings().sub2assignment_n_scheme
    dim2name = {0: 'leg', 1: 'antenna', 2: 'mouth'}

    relevant_dim2info_loss = defaultdict(list)
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        for s in range(num_subs):
            sub = subs[s]
            info_loss_collector = np.load(
                f'brain_data/decoding_results/decoding_error_{num_runs}runs_{roi}.npy', 
                allow_pickle=True).ravel()[0]

            # info_loss_collector = np.load(
            #     f'results/recon_loss_{attn_config_version}_{v}.npy', 
            #     allow_pickle=True).ravel()[0]

            sub_physical_order = np.array(sub2assignment_n_scheme[sub][:3])-1
            relevant_dim_name = dim2name[sub_physical_order[0]]
            relevant_dim2info_loss[relevant_dim_name].append(
                info_loss_collector[problem_type][s]
            )
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    dims_info_loss = ['Dimension']
    info_loss = ['info_loss']

    for relevant_dim_index in range(3):
        relevant_dim_name = dim2name[relevant_dim_index]
        dims_info_loss.extend([relevant_dim_name] * len(relevant_dim2info_loss[relevant_dim_name]))
        info_loss.extend(relevant_dim2info_loss[relevant_dim_name])

    dims_info_loss = np.array(dims_info_loss)
    info_loss = np.array(info_loss)
    df_info_loss = np.vstack((dims_info_loss, info_loss)).T
    pd.DataFrame(df_info_loss).to_csv(
        f"df_info_loss.csv", 
        index=False, header=False
    )
    df_info_loss = pd.read_csv('df_info_loss.csv', usecols=['Dimension', 'info_loss'])

    print(df_info_loss)

    sns.barplot(x='Dimension', y='info_loss', data=df_info_loss, ax=ax)
    plt.tight_layout()
    plt.savefig(f'figs/relevant_dim_v_info_loss_type1_{v}.pdf')

    # stats testing
    from scipy.stats import ttest_ind
    t, p = ttest_ind(relevant_dim2info_loss['leg'], relevant_dim2info_loss['antenna'])[:2]
    print('leg vs antenna: ', f't={t:.3f}, p={p:.3f}')
    
    t, p = ttest_ind(relevant_dim2info_loss['leg'], relevant_dim2info_loss['mouth'])[:2]
    print('leg vs mouth: ', f't={t:.3f}, p={p:.3f}')
    
    t, p = ttest_ind(relevant_dim2info_loss['mouth'], relevant_dim2info_loss['antenna'])[:2]
    print('mouth vs antenna: ', f't={t:.3f}, p={p:.3f}')


if __name__ == '__main__':
    attn_config_version='hyper4100'
    v='fit-human-entropy-fast-nocarryover'
    
    # Fig_zero_attn(attn_config_version, v)

    # Fig_recon_n_decoding(attn_config_version, v)

    # Fig_binary_recon(attn_config_version, v)

    # Fig_high_attn(attn_config_version, v)

    # Fig_high_attn_against_low_attn_final(attn_config_version, v)
    # Fig_high_attn_against_low_attn_window(attn_config_version, v, corr='keep_type_n_time')
    Fig_high_attn_against_low_attn_window_oneplot(attn_config_version, v, corr='collapse_type_n_time')
    # Fig_high_attn_against_low_attn_V2(attn_config_version, v)

    # Fig_alphas_against_recon_V1(attn_config_version, v)
    # Fig_alphas_against_recon_V1a(attn_config_version, v)
    # Fig_alphas_against_recon_V2(attn_config_version, v)
    
