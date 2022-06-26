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
    attn_weights = softmax(attn_weights)   # sum to one.
    ent = - np.sum( attn_weights * np.log2(attn_weights) )
    compression = 1 + ent / ( np.log2( 1/len(attn_weights) ) )
    return compression


def attn_sparsity(attn_weights):
    zero_percent = (
        len(attn_weights) - len(np.nonzero(attn_weights)[0])
    ) / len(attn_weights)
    return zero_percent


##### repetition level #####
def per_subj_compression_repetition_level(
        repr_level,
        run, 
        problem_type, 
        repetition, 
        config_version):
    """
    Compute compression score of a given model (per sub, task, rep).
    """
    results_path = f'results/{config_version}'
    if repr_level == 'low_attn':
        config = load_config(component=None, config_version=config_version)
        attn_position = config['attn_positions'].split(',')[0]
        attn_weights = np.load(
            f'{results_path}/' \
            f'attn_weights_type{problem_type}_run{run}_cluster_rp{repetition}.npy',
            allow_pickle=True).ravel()[0][attn_position]
        # score = attn_compression(attn_weights)
        score = attn_sparsity(attn_weights)
    
    elif repr_level == 'high_attn':
        attn_weights = np.load(
            f'{results_path}/' \
            f'all_alphas_type{problem_type}_run{run}_cluster_rp{repetition}.npy')[-3:]
        score = attn_compression(attn_weights)
        
    return score


def compression_execute_repetition_level(
        config_version, 
        repr_level, 
        runs,
        num_repetitions,
        problem_types, 
        num_processes
    ):
    """
    Top-level execute that compute compression score of model at `repr_level`
    and plot for all runs, types.
    """
    if not os.path.exists(f'compression_results_repetition_level'):
        os.mkdir(f'compression_results_repetition_level')
    
    if not os.path.exists(f'compression_results_repetition_level/{repr_level}.npy'):
        with multiprocessing.Pool(num_processes) as pool:
                        
            # compute & collect compression
            repetition2type2metric = defaultdict(lambda: defaultdict(list))
            for repetition in range(num_repetitions):
                for problem_type in problem_types:
                    for run in runs:
                                                         
                        # per (sub, task, run) compression
                        res_obj = pool.apply_async(
                            per_subj_compression_repetition_level, 
                            args=[
                                repr_level, 
                                run, 
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
                x.extend([f'{repetition}'] * num_runs)
                y.extend(metrics)
                hue.extend([f'Type {problem_type}'] * num_runs)
                
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
    num_bars = int(len(y) / (num_runs))
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
        per_type_data = y[ global_index * num_runs : (global_index+1) * num_runs ]
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
    config_version = 'v4a_naive-withNoise-entropy'
    repr_level = 'high_attn'
    num_runs = 50
    runs = range(num_runs)
    problem_types = [1, 2, 6]
    num_processes = 72
    num_repetitions = 32        # i.e. `num_blocks` in simulation
        
    compression_execute_repetition_level(
        config_version=config_version, 
        repr_level=repr_level, 
        runs=runs, 
        num_repetitions=num_repetitions,
        problem_types=problem_types, 
        num_processes=num_processes)