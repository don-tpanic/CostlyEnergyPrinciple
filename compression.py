import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import multiprocessing
from collections import defaultdict

import numpy as np
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

from utils import load_config

"""This script does two main things.
1. Replicate attn compression results in Mack 2020;
2. Explore attn in DCNN whether follows the same pattern
    of compression over task difficulties.
"""

def attn_compression(attn_weights):
    """
    Compute attn compressio rate according to
    Mack 2020; attn_weights can be either 
    attn in DCNN or attn in clustering module.
    """
    attn_weights = softmax(attn_weights)   # sum to one.
    print(attn_weights)
    ent = - np.sum( attn_weights * np.log2(attn_weights) )
    compression = 1 + ent / ( np.log2( 1/len(attn_weights) ) )
    print(compression)
    return compression


def per_subj_compression(repr_level, sub, problem_type, run, config_version):
    """
    Compute compression score of a given model (per sub, task, run).
    While model's attn is evaluated at trial-level, according to 
    Mack 2020, subject's per run compression is the average over trials
    (i.e. repetitions).
    """
    config_version = f'{config_version}_sub{sub}_fit-human'
    results_path = f'results/{config_version}'
    compression_scores = []
    for repetition in range(num_repetitions):
        # load trained joint_model at this repetition
        # TODO: replace after tuning finishes.
        model_path = os.path.join(results_path, f'model_type{problem_type}_sub{sub}')
        # model_path = os.path.join(results_path, f'model_type{problem_type}_sub{sub}_rp{repetition}')
        trained_model = tf.keras.models.load_model(model_path, compile=False)
        
        if repr_level == 'LOC':
            config = load_config(component=None, config_version=config_version)
            attn_position = config['attn_positions'].split(',')[0]
            attn_weights = trained_model.get_layer(
                'dcnn_model').get_layer(
                    f'attn_factory_{attn_position}').get_weights()[0]
        
        elif repr_level == 'cluster':
            attn_weights = trained_model.get_layer(
                'dimensionwise_attn_layer').get_weights()[0]
        
        K.clear_session()
        del trained_model
        
        compression_scores.append(
            attn_compression(attn_weights)
        )
            
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
                        print(res_obj.get())
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
            print(f'num_types={num_types}')
            
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

    # plot violinplots / stripplots
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
        
        q1, md, q3 = np.percentile(per_type_data, [25,50,75])
        mean = np.mean(per_type_data)
        std = np.std(per_type_data)
        median_obj = ax.scatter(position, md, marker='s', color='red', s=33, zorder=3)
        mean_obj = ax.scatter(position, mean, marker='^', color='k', s=33, zorder=3)
        
        # print out stats
        print(f'Type=[{problem_type}], run=[{run}], mean=[{mean:.3f}], std=[{std:.3f}], centerBy=[{centering_by}]')
        if within_run_index == 2:
            print('-'*60)
        
        if global_index in range(num_bars)[-3:]:
            # print(global_index)
            final_run_data.append(per_type_data)
        
    # hacky way getting legend
    ax.scatter(position, md, marker='s', color='red', s=33, zorder=3, label='median')
    ax.scatter(position, mean, marker='^', color='k', s=33, zorder=3, label='mean')
    plt.legend()
    ax.set_xlabel('Learning Blocks')
    ax.set_ylabel(f'Compression')
    plt.title(f'{repr_level}')
    plt.savefig(f'compression_results/{repr_level}.png')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    config_version = 'hyper0'
    repr_level = 'LOC'
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    runs = [1, 2, 3, 4]
    tasks = [1, 2, 3]
    num_processes = 1
    num_repetitions = 16 
    
    compression_execute(
        config_version=config_version, 
        repr_level=repr_level, 
        subs=subs, 
        runs=runs, 
        tasks=tasks, 
        num_processes=num_processes
    )