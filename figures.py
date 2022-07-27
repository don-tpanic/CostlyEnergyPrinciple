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
from clustering import human

# rc('text', usetex=True)
# plt.rcParams['text.usetex']=True
color_palette = sns.color_palette("bright")
colors = [color_palette[1], color_palette[6], color_palette[9]]
plt.rcParams.update({'font.size': 11})
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
        print(f'average_coef={average_coef:.3f}', f't={t:.3f}', f'p={p}')
        return average_coef, t, p/2
    
    problem_types = [1, 2, 6]
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    results_path = 'results'
    fig, ax = plt.subplots(figsize=(6, 4))
    
    collector = defaultdict(list)
    # e.g. { problem_type: {(True, False, False): [metric1, metric2, ... ]} }
    type2strategy2metric = defaultdict(lambda: defaultdict(list))
    
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
            label=f'Type {TypeConverter[problem_type]}')

    # plot curve of means
    ax.plot(range(len(problem_types)), means, color='grey', ls='dashed')       
    ax.set_ylabel('Percentage of \nZero Attention Weights')
    ax.set_xticks([])
    ax.set_ylim([0, 0.6])

    regression(collector, num_subs, problem_types)

    # plt.tight_layout()
    plt.legend()
    plt.suptitle('(C)')
    plt.savefig(f'figs/zero_attn.png')
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

    def relate_recon_loss_to_decoding_error_errorbar(attn_config_version, num_runs, roi, v):
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

        # for i in range(len(results_collectors)):
        #     results_collector = results_collectors[i]
        #     data = []
        #     for problem_type in problem_types:
        #         per_type_data = np.array(results_collector[problem_type])
        #         data.append(per_type_data)
            
        #     sns.violinplot(data=data, ax=axes[i], inner='point')
        #     axes[i].set_xlabel('Problem Types')
        #     axes[i].set_xticks(range(len(problem_types)))
        #     axes[i].set_xticklabels(problem_types)
        #     if i == 0:
        #         axes[i].set_ylabel(f'{roi} Neural Stimulus Reconstruction Loss\n(1 - decoding accuracy)')
        #         axes[i].set_title(f'(A)')
        #     else:
        #         axes[i].set_ylabel('Model Stimulus Reconstruction Loss')
        #         axes[i].set_title(f'(B)')
        
        plt.tight_layout()
        plt.savefig(f'figs/recon_loss_decoding_error.png')

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
        fig, axes = plt.subplots(1, 2, figsize=(9, 5))

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
                axes[i].set_ylabel(f'{roi} Neural Stimulus Information Loss\n(1 - decoding accuracy)')
                axes[i].set_title(f'Brain')
            else:
                axes[i].set_ylabel('Model Stimulus Information Loss')
                axes[i].set_title(f'Model')
        
        plt.legend()
        # plt.tight_layout()
        plt.suptitle('(B)')
        plt.savefig(f'figs/recon_loss_decoding_error.png')

    recon_loss_by_type(
        attn_config_version=attn_config_version, v=v
    )
    # relate_recon_loss_to_decoding_error_errorbar(
    #     attn_config_version=attn_config_version, num_runs=3, roi='LOC', v=v
    # )
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
    fig, axes = plt.subplots(3, figsize=(5, 7))
    
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

    num_rows = 3
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        row_idx = z % num_rows

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

        axes[row_idx].errorbar(
            x=range(num_dims),
            y=average_metric,
            yerr=sem_metric,
            fmt='o',
            capsize=3,
            color=colors[z],
        )

        axes[row_idx].set_xticks([])
        axes[row_idx].set_ylim([-0.1, 2])
        axes[row_idx].set_title(f'Type {TypeConverter[problem_type]}')
        axes[row_idx].spines.right.set_visible(False)
        axes[row_idx].spines.top.set_visible(False)
    
    axes[1].set_ylabel('Information Loss')
    axes[-1].set_xlabel('Abstract Dimension')
    plt.tight_layout()
    plt.savefig(f'figs/binary_recon.pdf')
    print('plotted binary recon')


if __name__ == '__main__':
    attn_config_version='hyper4100'
    v='fit-human-entropy-fast-nocarryover'
    
    # Fig_zero_attn(attn_config_version, v)

    # Fig_recon_n_decoding(attn_config_version, v)

    Fig_binary_recon(attn_config_version, v)