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
        # One rp has 8 trials,
        # here we compute the average alphas over 8 trials,
        # then compute the compression score of that average alphas.
        # Note, attn_weights is a list growing in length per rp,
        # that is, after each rp, it grows by 8*3
        attn_weights = np.load(
            f'{results_path}/' \
            f'all_alphas_type{problem_type}_run{run}_cluster_rp{repetition}.npy')
        # So to get alphas of a rp, we extract the 8*3
        # of that rp
        attn_weights = \
            np.mean(
                attn_weights[repetition*8*3 : (repetition+1)*8*3].reshape((8, 3)),
                axis=0
            )

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
    
    compression_plotter_repetition_level_V2(compression_results)


def compression_plotter_repetition_level_V2(compression_results):
    fig, ax = plt.subplots(figsize=(5, 4))
    reps = compression_results['x']       
    scores = compression_results['y']       
    types = compression_results['hue']
    color_palette = sns.color_palette("flare")
    print(color_palette.as_hex())
    palette = {'Type 1': color_palette[0], 'Type 2': color_palette[2], 'Type 6': color_palette[5]}
    problem_types = [1, 2, 6]
    num_bars = int(len(scores) / (num_runs))  # 48 = 16 * 3
    TypeConverter = {1: 'I', 2: 'II', 6: 'VI'}
    # create positions for errorbars, a repetition has 3 bars; each
    # repetition is separated by a margin of 2; each bar within a rep
    # is separated by 1.
    position = 1
    positions = []
    counter = 0
    for i in range(num_bars):
        positions.append(position)
        if counter == 2:
            counter = 0
            position += 2
        else:
            counter += 1
            position += 1
            
    means = []
    for i in range(len(positions)):
        position = positions[i]

        problem_type = problem_types[i % 3]

        per_rep_n_type_data = scores[i * num_runs : (i+1) * num_runs]

        if position >= positions[-3]:
            label = f'Type {TypeConverter[problem_type]}'
        else:
            label = None
        
        mean = np.mean(per_rep_n_type_data)
        means.append(mean)
        sem = stats.sem(per_rep_n_type_data)
        ax.errorbar(
            position,
            mean,
            yerr=sem,
            fmt='o',
            capsize=3,
            color=palette[f'Type {problem_type}'],
            label=label
        )
    
    # ax.set_xticks([2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62])
    # ax.set_xticklabels(range(int(num_bars/len(problem_types))))
    ax.set_xlabel('Learning Block', fontweight='bold')
    ax.set_ylabel(f'Attention Compression', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'compression_results_repetition_level/{repr_level}.pdf')


def mixed_effects_analysis_repetition_level(repr_level):
    """
    Perform a two-way ANOVA analysis as an alternative of 
    the bayesian mixed effect analysis in Mack et al., 2020.
    
    Independent variable: 
        problem_type, learning_block, interaction
    Dependent variable:
        compression score
    """
    import pingouin as pg
    if not os.path.exists(f"compression_results_repetition_level/{repr_level}.csv"):
        runs = ['run']
        types = ['problem_type']
        learning_trials = ['learning_trial']
        compression_scores = ['compression_score']

        compression_results = np.load(
            f'compression_results_repetition_level/{repr_level}.npy', 
            allow_pickle=True).ravel()[0]
        y = compression_results['y']
        num_bars = int(len(y) / (num_runs))
        problem_types = [1, 2, 6]
        
        for global_index in range(num_bars):
            rep = global_index // len(problem_types) + 1
            within_rep_index = global_index % len(problem_types)        
            problem_type = problem_types[within_rep_index]
            print(f'rep={rep}, type={problem_type}')
            
            # data
            per_type_data = y[ global_index * num_runs : (global_index+1) * num_runs ]
            
            for r in range(num_runs):
                run = rns[r]
                runs.append(run)
                types.append(problem_type)
                learning_trials.append(rep)
                compression_scores.append(per_type_data[r])
            
        runs = np.array(runs)
        types = np.array(types)
        learning_trials = np.array(learning_trials)
        compression_scores = np.array(compression_scores)
        
        df = np.vstack((
            runs, 
            types, 
            learning_trials, 
            compression_scores
        )).T
        pd.DataFrame(df).to_csv(
            f"compression_results_repetition_level/{repr_level}.csv", 
            index=False, 
            header=False
        )
        
    df = pd.read_csv(f"compression_results_repetition_level/{repr_level}.csv")
        
    # two-way ANOVA:
    res = pg.rm_anova(
        dv='compression_score',
        within=['problem_type', 'learning_trial'],
        subject='run',
        data=df, 
    )
    print(res)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    config_version = 'v4a_naive-withNoise-entropy'
    repr_level = 'high_attn'
    num_runs = 500
    rns = range(num_runs)
    problem_types = [1,2,6]
    num_processes = 72
    num_repetitions = 32        # i.e. `num_blocks` in simulation
        
    compression_execute_repetition_level(
        config_version=config_version, 
        repr_level=repr_level, 
        runs=rns, 
        num_repetitions=num_repetitions,
        problem_types=problem_types, 
        num_processes=num_processes)
    
    mixed_effects_analysis_repetition_level(repr_level)