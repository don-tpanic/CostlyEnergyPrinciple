import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras import backend as K

from utils import load_config
# rc('text', usetex=True)
plt.rcParams.update({'font.size': 8})


def examine_lc(config_version, 
                problem_types=[1, 2, 3, 4, 5, 6], 
                plot_learn_curves=True):
    """
    Follow sustain impl, we examine learning curves (y-axis is proberror)

    return:
    -------
        trapz_areas: An array of scalars which are the areas under the learning curves computed
                     using the trapzoidal rule.
        figure (optional): If `plot_learn_curves=True` will plot the learning curves.
    """
    config = load_config(config_version=config_version, component=None)
    num_runs = config['num_runs']
    num_blocks = config['num_blocks']
    recon_level = config['recon_level']

    colors = ['blue', 'orange', 'black', 'green', 'red', 'cyan']
    if plot_learn_curves:
        fig, ax = plt.subplots()

    trapz_areas = np.empty(len(problem_types))
    for idx in range(len(problem_types)):
        problem_type = problem_types[idx]

        lc_file = f'results/{config_version}/lc_type{problem_type}_{recon_level}.npy'
        lc = np.load(lc_file)[:num_blocks]

        trapz_areas[idx] = np.round(np.trapz(lc), 3)
        if plot_learn_curves:
            ax.errorbar(
                range(lc.shape[0]), 
                lc, 
                color=colors[idx],
                label=f'Type {problem_type}',
            )
    
    # print(f'[Results] {config_version} trapzoidal areas = ', trapz_areas)
    if plot_learn_curves:
        plt.legend()
        plt.title(f'{trapz_areas}')
        plt.xlabel('epochs')
        plt.ylabel('average probability of error')
        plt.tight_layout()
        plt.savefig(f'results/{config_version}/lc.png')
        plt.close()
    return trapz_areas
            
            
def examine_recruited_clusters_n_attn(config_version, canonical_runs_only=True):
    """
    Record the runs that produce canonical solutions
    for each problem type. 
    Specificially, we check the saved `mask_non_recruit`
    """
    config = load_config(config_version=config_version, component=None)
    num_types = 6
    results_path = f'results/{config_version}'
    num_runs = config['num_runs']
    num_dims = 3
    type2cluster = {
        1: 2, 2: 4,
        3: 6, 4: 6, 5: 6,
        6: 8}

    from collections import defaultdict
    canonical_runs = defaultdict(list)

    attn_weights = []
    for z in range(num_types):
        problem_type = z + 1

        # problem_type = 4
        print(f'\n\nType = {problem_type}')

        all_runs_attn = np.empty((num_runs, num_dims))
        for run in range(num_runs):            
            model_path = os.path.join(results_path, f'model_type{problem_type}_run{run}')
            model = tf.keras.models.load_model(model_path, compile=False)
            mask_non_recruit = model.get_layer('mask_non_recruit').get_weights()[0]

            dim_wise_attn_weights = model.get_layer('dimensionwise_attn_layer').get_weights()[0]
            all_runs_attn[run, :] = dim_wise_attn_weights
            del model
            
            num_nonzero = len(np.nonzero(mask_non_recruit)[0])

            print(f'no. clusters = {num_nonzero}, attn={dim_wise_attn_weights}')

            if canonical_runs_only:
                if num_nonzero == type2cluster[problem_type]:
                    canonical_runs[z].append(run)
            else:
                canonical_runs[z].append(run)

        # exit()
        per_type_attn_weights = np.round(
            np.mean(all_runs_attn, axis=0), 3
        )
        attn_weights.append(per_type_attn_weights)

    proportions = []
    for z in range(num_types):
        print(f'Type {z+1}, has {len(canonical_runs[z])}/{num_runs} canonical solutions')
        proportions.append(
            np.round(
                len(canonical_runs[z]) / num_runs 
            )
        )
    
    print(attn_weights)
    return proportions, attn_weights


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config_version = 'v0'
    examine_lc(config_version, problem_types=[1,2,3,4,5,6])
    examine_recruited_clusters_n_attn(config_version)

