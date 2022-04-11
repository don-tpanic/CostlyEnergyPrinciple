import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras import backend as K

from utils import load_config, load_data
# rc('text', usetex=True)
plt.rcParams.update({'font.size': 8})


def process_lc(config_version, problem_types):
    """
    Process saved problem_type & sub lc to
    get average, std etc.
    """
    config = load_config(config_version)
    num_subs = config['num_subs']
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    results_path = f'results/{config_version}'
    for problem_type in problem_types:
        lc = []
        
        for s in range(num_subs):
            sub = subs[s]
            lc.append(np.load(f'{results_path}/lc_type{problem_type}_sub{sub}.npy'))
        
        avg_lc = np.mean(lc, axis=0)
        std_lc = np.std(lc, axis=1)
        np.save(f'{results_path}/lc_type{problem_type}_avg.npy', avg_lc)
        np.save(f'{results_path}/lc_type{problem_type}_std.npy', std_lc)


def examine_lc(config_version, problem_types):
    """
    Follow sustain impl, we examine learning curves (y-axis is proberror)

    return:
    -------
        Plot the learning curves.
    """
    config = load_config(config_version)
    num_subs = config['num_subs']
    num_repetitions = config['num_repetitions']

    colors = ['blue', 'orange', 'cyan']
    fig, ax = plt.subplots()
    trapz_areas = np.empty(len(problem_types))
    for idx in range(len(problem_types)):
        problem_type = problem_types[idx]
        # lc_file = f'results/{config_version}/lc_type{problem_type}.npy'
        avg_lc_file = f'results/{config_version}/lc_type{problem_type}_avg.npy'
        std_lc_file = f'results/{config_version}/lc_type{problem_type}_std.npy'
        
        lc = np.load(avg_lc_file)[:num_repetitions]
        std = np.load(std_lc_file)[:num_repetitions]
        
        trapz_areas[idx] = np.round(np.trapz(lc), 3)
        ax.errorbar(
            range(lc.shape[0]), 
            1-lc,
            # yerr=std, 
            color=colors[idx],
            label=f'Type {problem_type}',
        )
        ax.set_xticks(range(0, num_repetitions+4, 4))
        ax.set_xticklabels(range(0, num_repetitions+4, 4))
        ax.set_ylim([-0.05, 1.05])
    
    config = load_config(config_version)
    
    # load hyper-params and write them on the figure
    if config_version != 'human':
        lr = config['lr']
        plt.text(12, 0.6, f'lr={lr}')
        center_lr_multiplier = config['center_lr_multiplier']
        plt.text(12, 0.57, f'center_lr={center_lr_multiplier * lr}')
        attn_lr_multiplier = config['attn_lr_multiplier']
        plt.text(12, 0.54, f'attn_lr={attn_lr_multiplier * lr}')
        asso_lr_multiplier = config['asso_lr_multiplier']
        plt.text(12, 0.51, f'asso_lr={asso_lr_multiplier * lr}')
        specificity = config['specificity']
        plt.text(12, 0.48, f'specificity={specificity}')
        Phi = config['Phi']
        plt.text(12, 0.45, f'Phi={Phi}')
        beta = config['beta']
        plt.text(12, 0.42, f'beta={beta}')
        temp2 = config['temp2']
        plt.text(12, 0.39, f'temp2={temp2}')
        thr = config['thr']
        plt.text(12, 0.36, f'thr={thr}')
        
    
    plt.legend()
    plt.title(f'{trapz_areas}')
    plt.xlabel('repetitions')
    plt.ylabel('average probability of error')
    plt.tight_layout()
    plt.savefig(f'results/{config_version}/lc.png')
    plt.close()
    return trapz_areas
            
   
def examine_subject_lc_and_attn_overtime(problem_types):
    """
    Plotting per subject (either human or model) lc using
    the best config and plot the attn weights overtime.
    """
    num_subs = 23
    num_repetitions = 16
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    
    for sub in subs:
        fig = plt.figure()
        gs = fig.add_gridspec(2,2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        colors = ['blue', 'orange', 'cyan']
        config_version = str(np.load(f'results/sub{sub}_best_config.npy'))
        config = load_config(config_version)
        print(f'sub{sub}, config={config_version}')
        
        # plot lc - human vs model
        per_config_sum_of_abs_diff = 0
        for idx in range(len(problem_types)):
            problem_type = problem_types[idx]
            human_lc = np.load(f'results/human/lc_type{problem_type}_sub{sub}.npy')
            model_lc = np.load(f'results/{config_version}/lc_type{problem_type}_sub{sub}.npy')
            per_config_sum_of_abs_diff += np.sum(np.abs(human_lc - model_lc))

            ax1.set_title('human')
            ax1.errorbar(
                range(human_lc.shape[0]), 
                1-human_lc,
                color=colors[idx],
                label=f'Type {problem_type}',
            )
            ax1.set_xticks(range(0, num_repetitions+4, 4))
            ax1.set_xticklabels(range(0, num_repetitions+4, 4))
            ax1.set_ylim([-0.05, 1.05])
            ax1.set_xlabel('repetitions')
            ax1.set_ylabel('average probability of error')
            
            ax2.set_title('model')
            ax2.errorbar(
                range(model_lc.shape[0]), 
                1-model_lc,
                color=colors[idx],
                label=f'Type {problem_type}',
            )
            ax2.set_xticks(range(0, num_repetitions+4, 4))
            ax2.set_xticklabels(range(0, num_repetitions+4, 4))
            ax2.set_xlabel('repetitions')
            ax2.set_ylim([-0.05, 1.05])
        
        # plot attn weights overtime
        visualize_attn_overtime(
            config_version=config_version,
            sub=sub,
            ax=ax3
        )
        
        # plot hyper-params of this config on figure
        x_coord = 9
        y_coor = 0.6
        margin = 0.07
        lr = config['lr']
        ax2.text(x_coord, y_coor, f'lr={lr:.3f}')
        center_lr_multiplier = config['center_lr_multiplier']
        ax2.text(x_coord, y_coor-margin*1, f'center_lr={center_lr_multiplier * lr:.3f}')
        attn_lr_multiplier = config['attn_lr_multiplier']
        ax2.text(x_coord, y_coor-margin*2, f'attn_lr={attn_lr_multiplier * lr:.3f}')
        asso_lr_multiplier = config['asso_lr_multiplier']
        ax2.text(x_coord, y_coor-margin*3, f'asso_lr={asso_lr_multiplier * lr:.3f}')
        specificity = config['specificity']
        ax2.text(x_coord, y_coor-margin*4, f'specificity={specificity:.3f}')
        Phi = config['Phi']
        ax2.text(x_coord, y_coor-margin*5, f'Phi={Phi:.3f}')
        beta = config['beta']
        ax2.text(x_coord, y_coor-margin*6, f'beta={beta:.3f}')
        thr = config['thr']
        ax2.text(x_coord, y_coor-margin*8, f'thr={thr:.3f}')
        temp2 = config['temp2']
        ax2.text(x_coord, y_coor-margin*7, f'temp2={temp2:.3f}')
        
        plt.legend()
        plt.suptitle(f'sub{sub}, diff={per_config_sum_of_abs_diff:.3f}')
        plt.tight_layout()
        plt.savefig(f'results/lc_sub{sub}.png')
        plt.close()
        
            
def examine_recruited_clusters_n_attn(config_version, canonical_runs_only=True):
    """
    Record the runs that produce canonical solutions
    for each problem type. 
    Specificially, we check the saved `mask_non_recruit`
    """
    config = load_config(config_version)
    problem_types = [1, 2, 6]
    results_path = f'results/{config_version}'
    num_subs = config['num_subs']
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    num_dims = 3
    type2cluster = {
        1: 2, 2: 4,
        3: 6, 4: 6, 5: 6,
        6: 8}

    from collections import defaultdict
    canonical_runs = defaultdict(list)

    attn_weights = []
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        
        print(f'------- problem_type = {problem_type} -------')
        
        all_runs_attn = np.empty((num_subs, num_dims))
        for i in range(num_subs):
            sub = subs[i]
            model_path = os.path.join(results_path, f'model_type{problem_type}_sub{sub}')
            model = tf.keras.models.load_model(model_path, compile=False)
            mask_non_recruit = model.get_layer('mask_non_recruit').get_weights()[0]

            dim_wise_attn_weights = model.get_layer('dimensionwise_attn_layer').get_weights()[0]
            all_runs_attn[i, :] = dim_wise_attn_weights
            del model
            
            num_nonzero = len(np.nonzero(mask_non_recruit)[0])
            print(f'sub{sub}, no. clusters = {num_nonzero}, attn={dim_wise_attn_weights}')
            if canonical_runs_only:
                if num_nonzero == type2cluster[problem_type]:
                    canonical_runs[z].append(i)
            else:
                canonical_runs[z].append(i)

        per_type_attn_weights = np.round(
            np.mean(all_runs_attn, axis=0), 3
        )
        attn_weights.append(per_type_attn_weights)

    proportions = []
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        print(f'Type {problem_type}, has {len(canonical_runs[z])}/{num_subs} canonical solutions')
        proportions.append(
            np.round(
                len(canonical_runs[z]) / num_subs 
            )
        )
    
    print(attn_weights)
    return proportions, attn_weights

     
def visualize_attn_overtime(config_version, sub, ax):
    """
    Visualize the change of attn weights through learning.
    Through time we need to consider even subjects did 
    6-1-2 and odd subjects did 6-2-1.
    """
    config = load_config(config_version)
    num_dims = 3
    num_repetitions = config['num_repetitions']
    if int(sub) % 2 == 0:
        problem_types = [6, 1, 2]
    else:
        problem_types = [6, 2, 1]
    task_change_timepoint = [15, 31]
    
    attn_weights_overtime = []
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        for rp in range(num_repetitions):
            attn_weights = np.load(
                f'results/{config_version}/' \
                f'attn_weights_type{problem_type}_sub{sub}_rp{rp}.npy'
            )
            attn_weights_overtime.append(attn_weights)
    # (16*3, 3)
    attn_weights_overtime = np.array(attn_weights_overtime)
    
    # plot 
    for dim in range(num_dims):
        dim_overtime = attn_weights_overtime[:, dim]
        ax.plot(dim_overtime, label=f'dim{dim+1}')
    
    # partition regions into Type 6, 1, 2
    ax.axvline(
        x=task_change_timepoint[0], 
        color='purple', 
        label=f'Type 6->{problem_types[1]}', 
        ls='dashed'
    )
    ax.axvline(
        x=task_change_timepoint[1], 
        color='pink', 
        label=f'Type {problem_types[1]}->{problem_types[2]}',
        ls='dashed'
    )
        

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    problem_types = [1, 2, 6]
    examine_subject_lc_and_attn_overtime(problem_types)
    # if config_version != 'human':
    #     examine_recruited_clusters_n_attn(config_version)
        
    # visualize_attn_overtime(config_version, sub='04')
