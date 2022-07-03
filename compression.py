import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import multiprocessing
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

from utils import load_config
plt.rcParams.update({'font.size': 12})

"""This script does two main things.
1. Replicate attn compression results in Mack 2020;
2. Explore attn in DCNN whether follows the same pattern
    of compression over task difficulties.
"""

def attn_compression(attn_weights):
    """
    Compute attn compression rate according to
    Mack 2020; attn_weights can be either 
    attn in DCNN or attn in clustering module.
    
    The more selective the higher compression.
    """
    ent = - np.sum( attn_weights * np.log2(attn_weights) )
    compression = 1 + ent / ( np.log2( 1/len(attn_weights) ) )
    return compression


def attn_sparsity(attn_weights):
    zero_percent = (
        len(attn_weights) - len(np.nonzero(attn_weights)[0])
    ) / len(attn_weights)
    return zero_percent


def per_subj_compression(repr_level, sub, problem_type, run, config_version):
    """
    Compute compression score of a given model (per sub, task, run).
    While model's attn is evaluated at trial-level, according to 
    Mack 2020, subject's per run compression is the average over trials
    (i.e. repetitions).
    """
    config_version = f'{config_version}_sub{sub}_fit-human-entropy-fast-nocarryover'
    results_path = f'results/{config_version}'
    compression_scores = []
    for rp in range(1, num_repetitions_per_run+1):

        # convert per run rp to global repetition 
        repetition = (run-1) * num_repetitions_per_run + rp - 1
        
        if repr_level == 'low_attn':
            config = load_config(component=None, config_version=config_version)
            attn_position = config['attn_positions'].split(',')[0]
            attn_weights = np.load(
                f'{results_path}/' \
                f'attn_weights_type{problem_type}_sub{sub}_cluster_rp{repetition}.npy',
                allow_pickle=True).ravel()[0][attn_position]
            # score = attn_compression(attn_weights)
            score = attn_sparsity(attn_weights)
        
        elif repr_level == 'high_attn':
            attn_weights = np.load(
                f'{results_path}/' \
                f'all_alphas_type{problem_type}_sub{sub}_cluster_rp{repetition}.npy')[-3:]
        
            score = attn_compression(attn_weights)
        compression_scores.append(score)
    
    return np.mean(compression_scores)


def compression_execute(config_version, repr_level, subs, runs, tasks, num_processes):
    """
    Top-level execute that compute compression score of model at `repr_level`
    and plot for all subs, runs, tasks.
    """
    if not os.path.exists(f'compression_results'):
        os.mkdir(f'compression_results')
    
    if not os.path.exists(f'compression_results/{repr_level}.npy'):
        with multiprocessing.Pool(num_processes) as pool:
                        
            # compute & collect compression
            run2type2metric = defaultdict(lambda: defaultdict(list))
            for run in runs:
                for task in tasks:
                    for sub in subs:
                                                         
                        if int(sub) % 2 == 0:
                            if task == 2:
                                problem_type = 1
                            elif task == 3:
                                problem_type = 2
                            else:
                                problem_type = 6
                                
                        # odd sub: Type1 is task3
                        else:
                            if task == 2:
                                problem_type = 2
                            elif task == 3:
                                problem_type = 1
                            else:
                                problem_type = 6
                                
                        # per (sub, task, run) compression
                        res_obj = pool.apply_async(
                            per_subj_compression, 
                            args=[
                                repr_level, 
                                sub, problem_type, 
                                run, config_version
                            ]
                        )
                        # Notice res_obj.get() = compression
                        # To enable multiproc, we extract the actual
                        # compression score when plotting later.
                        run2type2metric[run][problem_type].append(res_obj)
            
            pool.close()
            pool.join()
        
        # save & plot compression results
        # ref: https://stackoverflow.com/questions/68629457/seaborn-grouped-violin-plot-without-pandas
        x = []    # each sub's run
        y = []    # each sub problem_type's compression
        hue = []  # each sub problem_type
        means = []
        for run in runs:
            print(f'--------- run {run} ---------')
            type2metric = run2type2metric[run]
            num_types = len(type2metric.keys())
            problem_types = sorted(list(type2metric.keys()))
            for z in range(num_types):
                problem_type = problem_types[z]
                # here we extract a list of res_obj and 
                # extract the actual compression scores.
                list_of_res_obj = type2metric[problem_type]
                # `metrics` is all scores over subs for one (problem_type, run)
                metrics = [res_obj.get() for res_obj in list_of_res_obj]
                # metrics = list(metrics - np.mean(metrics))
                means.append(np.mean(metrics))
                x.extend([f'{run}'] * num_subs)
                y.extend(metrics)
                hue.extend([f'Type {problem_type}'] * num_subs)
                
        compression_results = {}
        compression_results['x'] = x
        compression_results['y'] = y
        compression_results['hue'] = hue
        compression_results['means'] = means
        np.save(f'compression_results/{repr_level}.npy', compression_results)
    
    else:
        print('[NOTE] Loading saved results, make sure it does not need update.')
        # load presaved results dictionary.
        compression_results = np.load(f'compression_results/{repr_level}.npy', allow_pickle=True).ravel()[0]

    # plot compression results
    # compression_plotter(compression_results)
    compression_plotter_V2(compression_results)


def mixed_effects_analysis(repr_level):
    """
    Perform a two-way ANOVA analysis as an alternative of 
    the bayesian mixed effect analysis in Mack et al., 2020.
    
    Independent variable: 
        problem_type, learning_block, interaction
    Dependent variable:
        compression score
    """
    import pingouin as pg
    if not os.path.exists(f"compression_results/{repr_level}.csv"):
        subjects = ['subject']
        types = ['problem_type']
        learning_blocks = ['learning_block']
        compression_scores = ['compression_score']

        compression_results = np.load(
            f'compression_results/{repr_level}.npy', 
            allow_pickle=True).ravel()[0]
        y = compression_results['y']
        num_bars = int(len(y) / (num_subs))
        problem_types = [1, 2, 6]
        
        # global_index: 0-11
        for global_index in range(num_bars):
            # run: 1-4 i.e. learning block
            run = global_index // len(problem_types) + 1
            # within_run_index: 0-2
            within_run_index = global_index % len(problem_types)        
            problem_type = problem_types[within_run_index]
            print(f'run={run}, type={problem_type}')
            
            # data
            per_type_data = y[ global_index * num_subs : (global_index+1) * num_subs ]
            
            for s in range(num_subs):
                sub = subs[s]
                subjects.append(sub)
                types.append(problem_type)
                learning_blocks.append(run)
                compression_scores.append(per_type_data[s])
            
        subjects = np.array(subjects)
        types = np.array(types)
        learning_blocks = np.array(learning_blocks)
        compression_scores = np.array(compression_scores)
        
        df = np.vstack((
            subjects, 
            types, 
            learning_blocks, 
            compression_scores
        )).T
        pd.DataFrame(df).to_csv(
            f"compression_results/{repr_level}.csv", 
            index=False, 
            header=False
        )
        
    df = pd.read_csv(f"compression_results/{repr_level}.csv")
        
    # two-way ANOVA:
    res = pg.rm_anova(
        dv='compression_score',
        within=['problem_type', 'learning_block'],
        subject='subject',
        data=df, 
    )
    print(res)


def compression_plotter(compression_results):
    fig, ax = plt.subplots()
    x = compression_results['x']
    y = compression_results['y']
    hue = compression_results['hue']
    means = compression_results['means']
    palette = {'Type 1': 'pink', 'Type 2': 'green', 'Type 6': 'blue'}
    ax = sns.stripplot(x=x, y=y, hue=hue, palette=palette, dodge=True, alpha=0.8, jitter=0.3, size=4)
    
    # plot mean/median
    num_bars = int(len(y) / (num_subs))
    positions = []
    margin = 0.24
    problem_types = [1, 2, 6]
    for per_run_center in ax.get_xticks():
        positions.append(per_run_center-margin)
        positions.append(per_run_center)
        positions.append(per_run_center+margin)
    
    labels = []
    final_run_data = []  # for t-test
    # global_index: 0-11
    for global_index in range(num_bars):
        # run: 1-4
        run = global_index // len(problem_types) + 1
        # within_run_index: 0-2
        within_run_index = global_index % len(problem_types)        
        problem_type = problem_types[within_run_index]
        
        # data
        per_type_data = y[ global_index * num_subs : (global_index+1) * num_subs ]
        position = [positions[global_index]]
        
        mean = np.mean(per_type_data)
        sem = stats.sem(per_type_data)
        std = np.std(per_type_data)
        mean_obj = ax.scatter(position, mean, marker='^', color='k', s=33, zorder=3)
                
        # print out stats
        print(f'Type=[{problem_type}], run=[{run}], mean=[{mean:.3f}], sem=[{sem:.3f}]')
        if within_run_index == 2:
            print('-'*60)
        
        if global_index in range(num_bars)[-3:]:
            # print(global_index)
            final_run_data.append(per_type_data)
        
    # hacky way getting legend
    ax.scatter(position, mean, marker='^', color='k', s=33, zorder=3, label='mean')
    plt.legend()
    ax.set_xlabel('Learning Blocks')
    ax.set_ylabel(f'Compression')
    if repr_level == 'low_attn':
        title = 'Low-level attn (DCNN)'
    elif repr_level == 'high_attn':
        title = 'High-level attn (Clustering)'
    plt.title(f'{title}')
    plt.savefig(f'compression_results/{repr_level}.png')


def compression_plotter_V2(compression_results):
    fig, ax = plt.subplots(figsize=(6, 4))
    runs = compression_results['x']       
    scores = compression_results['y']       
    types = compression_results['hue']
    color_palette = sns.color_palette("bright")
    palette = {'Type 1': color_palette[1], 'Type 2': color_palette[6], 'Type 6': color_palette[9]}
    problem_types = [1, 2, 6]
    num_bars = int(len(scores) / (num_subs))
    positions = [1,2,3, 5,6,7, 9,10,11, 13,14,15]
    
    means = []
    for i in range(len(positions)):
        position = positions[i]
        run_i = i // 3 + 1
        problem_type = problem_types[i % 3]
        per_run_n_type_data = scores[i * num_subs : (i+1) * num_subs]

        if position >= 13:
            label = f'Type {problem_type}'
        else:
            label = None
        
        mean = np.mean(per_run_n_type_data)
        means.append(mean)
        sem = stats.sem(per_run_n_type_data)
        ax.errorbar(
            position,
            mean,
            yerr=sem,
            fmt='o',
            capsize=3,
            color=palette[f'Type {problem_type}'],
            label=label
        )    
    
    # plot curve of means for each run
    for run_i in range(int(len(positions)/len(problem_types))):
        per_run_positions = positions[run_i * len(problem_types) : (run_i+1) * len(problem_types)]
        per_run_means = means[run_i * len(problem_types) : (run_i+1) * len(problem_types)]
        ax.plot(per_run_positions, per_run_means, color='grey', ls='dashed')

    ax.set_xticks([2, 6, 10, 14])
    ax.set_xticklabels([1, 2, 3, 4])
    # ax.set_yticks([0, 0.05, 0.1, 0.15])
    # ax.set_yticklabels([0, 0.05, 0.1, 0.15])
    # ax.set_ylim([-0.005, 0.15])
    ax.set_xlabel('Learning Blocks')
    ax.set_ylabel(f'Attention Compression')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'compression_results/{repr_level}.png')
    plt.savefig(f'figs/compression_{repr_level}.png')


##### repetition level #####
def per_subj_compression_repetition_level(
        repr_level,
        sub, 
        problem_type, 
        repetition, 
        config_version):
    """
    Compute compression score of a given model (per sub, task, rep).
    """
    config_version = f'{config_version}_sub{sub}_fit-human-entropy-fast-nocarryover'
    results_path = f'results/{config_version}'
    if repr_level == 'low_attn':
        config = load_config(component=None, config_version=config_version)
        attn_position = config['attn_positions'].split(',')[0]
        attn_weights = np.load(
            f'{results_path}/' \
            f'attn_weights_type{problem_type}_sub{sub}_cluster_rp{repetition}.npy',
            allow_pickle=True).ravel()[0][attn_position]
        # score = attn_compression(attn_weights)
        score = attn_sparsity(attn_weights)
    
    elif repr_level == 'high_attn':
        attn_weights = np.load(
            f'{results_path}/' \
            f'all_alphas_type{problem_type}_sub{sub}_cluster_rp{repetition}.npy')[-3:]
        score = attn_compression(attn_weights)
        
    return score


def compression_execute_repetition_level(
        config_version, 
        repr_level, 
        subs,
        num_repetitions,
        tasks, 
        num_processes
    ):
    """
    Top-level execute that compute compression score of model at `repr_level`
    and plot for all subs, runs, tasks.
    """
    if not os.path.exists(f'compression_results_repetition_level'):
        os.mkdir(f'compression_results_repetition_level')
    
    if not os.path.exists(f'compression_results_repetition_level/{repr_level}.npy'):
        with multiprocessing.Pool(num_processes) as pool:
                        
            # compute & collect compression
            repetition2type2metric = defaultdict(lambda: defaultdict(list))
            for repetition in range(num_repetitions):
                for task in tasks:
                    for sub in subs:
                                                         
                        if int(sub) % 2 == 0:
                            if task == 2:
                                problem_type = 1
                            elif task == 3:
                                problem_type = 2
                            else:
                                problem_type = 6
                                
                        # odd sub: Type1 is task3
                        else:
                            if task == 2:
                                problem_type = 2
                            elif task == 3:
                                problem_type = 1
                            else:
                                problem_type = 6
                                
                        # per (sub, task, run) compression
                        res_obj = pool.apply_async(
                            per_subj_compression_repetition_level, 
                            args=[
                                repr_level, 
                                sub, 
                                problem_type, 
                                repetition,
                                config_version
                            ]
                        )
                        # Notice res_obj.get() = compression
                        # To enable multiproc, we extract the actual
                        # compression score when plotting later.
                        repetition2type2metric[repetition][problem_type].append(res_obj)
            
            pool.close()
            pool.join()
        
        # save & plot compression results
        # ref: https://stackoverflow.com/questions/68629457/seaborn-grouped-violin-plot-without-pandas
        x = []    # each sub's rep
        y = []    # each sub problem_type's compression
        hue = []  # each sub problem_type
        means = []
        for repetition in range(num_repetitions):
            print(f'--------- repetition {repetition} ---------')
            type2metric = repetition2type2metric[repetition]
            num_types = len(type2metric.keys())
            problem_types = sorted(list(type2metric.keys()))
            for z in range(num_types):
                problem_type = problem_types[z]
                # here we extract a list of res_obj and 
                # extract the actual compression scores.
                list_of_res_obj = type2metric[problem_type]
                # `metrics` is all scores over subs for one (problem_type, run)
                metrics = [res_obj.get() for res_obj in list_of_res_obj]
                # metrics = list(metrics - np.mean(metrics))
                means.append(np.mean(metrics))
                x.extend([f'{repetition}'] * num_subs)
                y.extend(metrics)
                hue.extend([f'Type {problem_type}'] * num_subs)
                
        compression_results = {}
        compression_results['x'] = x
        compression_results['y'] = y
        compression_results['hue'] = hue
        compression_results['means'] = means
        np.save(f'compression_results_repetition_level/{repr_level}.npy', compression_results)
    
    else:
        print('[NOTE] Loading saved results, make sure it does not need update.')
        # load presaved results dictionary.
        compression_results = np.load(
            f'compression_results_repetition_level/{repr_level}.npy', 
            allow_pickle=True
        ).ravel()[0]

    # plot violinplots / stripplots
    fig, ax = plt.subplots()
    x = compression_results['x']
    y = compression_results['y']
    hue = compression_results['hue']
    means = compression_results['means']
    palette = {'Type 1': 'pink', 'Type 2': 'green', 'Type 6': 'blue'}
    colors = {1: 'pink', 2: 'green', 6: 'blue'}
    ax = sns.stripplot(x=x, y=y, hue=hue, palette=palette, dodge=True, alpha=0.2, jitter=0.3, size=4)
    
    # plot mean/median
    num_bars = int(len(y) / (num_subs))
    positions = []
    margin = 0.24
    problem_types = [1, 2, 6]
    for per_repetition_center in ax.get_xticks():
        positions.append(per_repetition_center-margin)
        positions.append(per_repetition_center)
        positions.append(per_repetition_center+margin)
    
    labels = []
    for global_index in range(num_bars):
        
        # 0-15
        repetition = global_index // len(problem_types)
        within_repetition_index = global_index % len(problem_types)    
        problem_type = problem_types[within_repetition_index]
        
        # data
        per_type_data = y[ global_index * num_subs : (global_index+1) * num_subs ]
        position = [positions[global_index]]
        
        q1, md, q3 = np.percentile(per_type_data, [25,50,75])
        mean = np.mean(per_type_data)
        std = np.std(per_type_data)
        mean_obj = ax.scatter(position, mean, marker='^', color=colors[problem_type], s=33, zorder=3, alpha=1.)
        
        # print out stats
        print(f'Type=[{problem_type}], repetition=[{repetition}], mean=[{mean:.3f}], std=[{std:.3f}]')
        if within_repetition_index == 2:
            print('-'*60)
                
    # hacky way getting legend
    # ax.scatter(position, mean, marker='^', color='k', s=33, zorder=3, label='mean')
    plt.legend()
    ax.set_xlabel('Repetitions')
    ax.set_ylabel(f'Compression')
    if repr_level == 'low_attn':
        title = 'Low-level attn (DCNN)'
    elif repr_level == 'high_attn':
        title = 'High-level attn (Clustering)'
    plt.title(f'{title}')
    plt.savefig(f'compression_results_repetition_level/{repr_level}.png')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    config_version = 'hyper4100'
    repr_level = 'high_attn'
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    runs = [1, 2, 3, 4]
    tasks = [1, 2, 3]
    num_processes = 72
    num_repetitions_per_run = 4
    num_repetitions = 16
    
    compression_execute(
        config_version=config_version, 
        repr_level=repr_level, 
        subs=subs, 
        runs=runs, 
        tasks=tasks, 
        num_processes=num_processes
    )
    
    mixed_effects_analysis(repr_level)
    
    # compression_execute_repetition_level(
    #     config_version=config_version, 
    #     repr_level=repr_level, 
    #     subs=subs, 
    #     num_repetitions=num_repetitions,
    #     tasks=tasks, 
    #     num_processes=num_processes
    # )