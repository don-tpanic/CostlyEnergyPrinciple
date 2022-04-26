import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

import tensorflow as tf
from tensorflow.keras import backend as K

from models import ClusterModel
from utils import load_config, load_data, load_X_only
from human import load_data_human_order, reorder_RDM_entries_into_chunks

from matplotlib import rc
# rc('text', usetex=True)
plt.rcParams.update({'font.size': 4})

"""This script does two main things:
1. Prepare RSA: (Compute, save, visualize) RDM per (problem_type, subject, repetition)
    using model cluster activations.
    
    - create_mdoel_RDMs: top-level execute
    - return_RDM: produce 1 RDM per (problem_type, sub rp)
    - load_trained_model: a trained model per (problem_type, sub rp)
    - visualize_RDM: heatmaps a given RDM

2. Perform RSA: compare correlation beween model- and subject-RDMs
   which is done in Mack et al. 2016. 
   
   Two levels of RSA can be performed:
    For repetition level:
    - repetition_level_RSA: model- and subject-RDM one-to-one correlation 
                            per repetition.
    - repetition_level_allCombo_RSA: correlation of all possible repetition pairs 
                                     between model and subject-RDM are computed.

    For run level:
    - run_level_RSA: model-RDMs are averaged over repetitions within a run.
    - repetition_level_allCombo_RSA: ..
"""

def load_trained_model(config_version, problem_type, sub, repetition):
    if 'best' in config_version:
        config_version = f'{config_version}{sub}'
    config = load_config(config_version)
    num_subs = config['num_subs']
    num_repetitions = config['num_repetitions']
    random_seed = config['random_seed']
    from_logits = config['from_logits']
    lr = config['lr']
    center_lr_multiplier = config['center_lr_multiplier']
    attn_lr_multiplier = config['attn_lr_multiplier']
    asso_lr_multiplier = config['asso_lr_multiplier']
    lr_multipliers = [center_lr_multiplier, attn_lr_multiplier, asso_lr_multiplier]
    num_clusters = config['num_clusters']
    r = config['r']
    q = config['q']
    specificity = config['specificity']
    trainable_specificity = config['trainable_specificity']
    attn_constraint = config['attn_constraint']
    Phi = config['Phi']
    actv_func = config['actv_func']
    beta = config['beta']
    temp1 = config['temp1']
    temp2 = config['temp2']
    results_path = f'results/{config_version}'
    
    # a new initialised model (all weights frozen)
    model = ClusterModel(
        num_clusters=num_clusters, r=r, q=q, 
        specificity=specificity, 
        trainable_specificity=trainable_specificity, 
        attn_constraint=attn_constraint,
        Phi=Phi, 
        actv_func=actv_func,
        beta=beta,
        temp1=temp1,
        temp2=temp2
    )
    model.build(input_shape=(1, 3))
    
    model_path = os.path.join(results_path, f'model_type{problem_type}_sub{sub}_rp{repetition}')
    trained_model = tf.keras.models.load_model(model_path, compile=False)
    # carryover cluster centers
    for i in range(num_clusters):
        model.get_layer(f'd{i}').set_weights(
            trained_model.get_layer(f'd{i}').get_weights()
        )

    # carryover attn weights
    model.get_layer(f'dimensionwise_attn_layer').set_weights(
        trained_model.get_layer(f'dimensionwise_attn_layer').get_weights()
    )
    
    # carryover cluster recruitment
    model.get_layer(f'mask_non_recruit').set_weights(
        trained_model.get_layer(f'mask_non_recruit').get_weights()
    )
    
    # carryover association weights
    model.get_layer(f'classification').set_weights(
        trained_model.get_layer(f'classification').get_weights()
    )
    
    return model


def return_RDM(problem_type, sub, distance, repetition):
    """
    Produce RDM of a (sub, problem_type)
    """        
    if int(sub) % 2 == 0:
        if problem_type == 1:
            task = 2
        elif problem_type == 2:
            task = 3
        else:
            task = 1
    # odd sub: Type1 is task3, Type2 is task2
    else:
        if problem_type == 1:
            task = 3
        elif problem_type == 2:
            task = 2
        else:
            task = 1
        
    RDM_fpath = f'model_RDMs/sub-{sub}_task-{task}_rp-{repetition}_{distance}.npy'
        
    # For now, let's just use the final repetition
    model = load_trained_model(config_version, problem_type, sub, repetition)
    
    # The order of the stimuli does not matter here
    # For convenience, we load using the original order
    # i.e. 000, 001, ..., 111
    dataset = load_data(problem_type)
    
    # Compile into batch
    batch_x = load_X_only(dataset)
    cluster_actv, _, _ = model(batch_x)
    
    # `cluster_actv` is basically an embedding matrix
    # which we use to produce RDM
    if distance == 'euclidean':
        RDM = pairwise_distances(cluster_actv, metric=distance)
    elif distance == 'pearson':
        RDM = pairwise_distances(cluster_actv, metric='correlation')
    
    # We then rearrange based on category structure
    reorder_mapper = reorder_RDM_entries_into_chunks()
    conversion_ordering = reorder_mapper[sub][task]
    RDM = RDM[conversion_ordering, :][:, conversion_ordering]
    np.save(RDM_fpath, RDM)
    print(f'[Check] Saved: {RDM_fpath}')
    
    
def visualize_RDM(sub, problem_type, distance, repetition):
    """
    Visualize subject's RDM given a problem_type
    """
    if int(sub) % 2 == 0:
        if problem_type == 1:
            task = 2
        elif problem_type == 2:
            task = 3
            
    # odd sub: Type1 is task3, Type2 is task2
    else:
        if problem_type == 1:
            task = 3
        elif problem_type == 2:
            task = 2
            
    RDM = np.load(
        f'model_RDMs/sub-{sub}_task-{task}_rp-{repetition}_{distance}.npy'
    )
    
    fig, ax = plt.subplots()
    for i in range(RDM.shape[0]):
        for j in range(RDM.shape[0]):
            text = ax.text(
                j, i, np.round(RDM[i, j], 1),
                ha="center", va="center", color="w"
            )
    
    ax.set_title(f'sub: {sub}, distance: {distance}, Type {problem_type}')
    plt.imshow(RDM)
    plt.savefig(f'model_RDMs/sub-{sub}_task-{task}_rp-{repetition}_{distance}.png')
    plt.close()
    print(f'[Check] plotted.')

    
def create_mdoel_RDMs():
    """
    Create (save and visualize) per (problem_type, sub)
    RDMs.

    `return_RDM`: loads a trained model for a type, and produces
    cluster representations and then RDMs using the stimulus set.
    `visualize_RDM`: heatmaps the saved RDMs.
    """
    for problem_type in problem_types:
        for sub in subs:
            for repetition in range(num_repetitions):
                print(f'repetition = {repetition}')
                return_RDM(problem_type, sub, distance, repetition)
                visualize_RDM(sub, problem_type, distance, repetition)


############################################################################


def compute_RSA(RDM_1, RDM_2, method):
    """
    Compute spearman correlation between 
    two RDMs' upper trigular entries
    """
    RDM_1_triu = RDM_1[np.triu_indices(RDM_1.shape[0])]
    RDM_2_triu = RDM_2[np.triu_indices(RDM_2.shape[0])]
    
    if method == 'spearman':
        rho, _ = stats.spearmanr(RDM_1_triu, RDM_2_triu)
    elif method == 'kendall_a':
        rho = kendall_a(RDM_1_triu, RDM_2_triu)
    
    return rho
        

def repetition_level_RSA(rois, distance, problem_type, num_shuffles, method='spearman', dataType='beta', seed=999):
    """
    Doing subject-model RSA at the repetition level. Subject RDMs are computed using 
    repetition- (i.e. trial) level GLM estimates and model RDMs are computed using 
    repetition-level model cluster representations. 
    
    Impl:
    -----
        Notice, for model, repetitions is 0-15.
                for subject, repetitions is 1-4 for 4 runs.
        Subject to model conversion:
                model_rp = (run-1) * 4 + subject_rp - 1
    """
    for roi in rois:
        for run in runs:
            for rp in range(1, num_repetitions_per_run+1):
                # NOTE: for model, rp starts from 0
                
                np.random.seed(seed)
                all_rho = []  # one per subject-run-repetition of a task
                for shuffle in range(num_shuffles):          
                    for sub in subs:
                        
                        # even sub: Type1 is task2, Type2 is task3
                        if int(sub) % 2 == 0:
                            if problem_type == 1:
                                task = 2
                            elif problem_type == 2:
                                task = 3
                                
                        # odd sub: Type1 is task3, Type2 is task2
                        else:
                            if problem_type == 1:
                                task = 3
                            elif problem_type == 2:
                                task = 2
                                
                        # get one repetition's RDM
                        sub_RDM_fpath = f'subject_RDMs/sub-{sub}_task-{task}_run-{run}_rp-{rp}_roi-{roi}_{distance}_{dataType}.npy'
                        sub_RDM = np.load(sub_RDM_fpath)
                        # convert run and subject rp to model rp:
                        # for model rp -> [0, 16) across 4 runs.
                        model_rp = (run-1) * num_repetitions_per_run + rp - 1
                        model_RDM_fpath = f'model_RDMs/sub-{sub}_task-{task}_rp-{model_rp}_{distance}.npy'
                        model_RDM = np.load(model_RDM_fpath)
                                                
                        if num_shuffles > 1:
                            shuffle_indices = np.random.choice(
                                range(sub_RDM.shape[0]), 
                                size=sub_RDM.shape[0],
                                replace=False
                            )
                            sub_RDM = sub_RDM[shuffle_indices, :]
                            
                        # compute one repetition's correlation to the ideal RDM
                        rho = compute_RSA(sub_RDM, model_RDM, method=method)
                        # collects all repetitions of a run and of all subjects
                        all_rho.append(rho)
                print(
                    f'Dist=[{distance}], Type=[{problem_type}], roi=[{roi}], run=[{run}], rp=[{rp}], ' \
                    f'avg_rho=[{np.mean(all_rho):.2f}], ' \
                    f'std=[{np.std(all_rho):.2f}], ' \
                    f't-stats=[{stats.ttest_1samp(a=all_rho, popmean=0)[0]:.2f}], ' \
                    f'pvalue=[{stats.ttest_1samp(a=all_rho, popmean=0)[1]:.2f}]' \
                )    
            print('------------------------------------------------------------------------')


def run_level_RSA(rois, distance, problem_type, num_shuffles, method='spearman', dataType='beta', seed=999):
    """
    Doing subject-model RSA at the run level. Subject RDMs are computed using 
    run-level GLM estimates and model RDMs are computed using 
    repetition-level model cluster representations but averaged over repetitions within 
    that run. Notice, for model, there isn't such thing as run but we could convert based 
    on the subject run and average over repetitions.
    """
    for roi in rois:
        for run in runs:
                
            np.random.seed(seed)
            all_rho = []
            for shuffle in range(num_shuffles):          
                for sub in subs:
                    
                    # even sub: Type1 is task2, Type2 is task3
                    if int(sub) % 2 == 0:
                        if problem_type == 1:
                            task = 2
                        elif problem_type == 2:
                            task = 3
                            
                    # odd sub: Type1 is task3, Type2 is task2
                    else:
                        if problem_type == 1:
                            task = 3
                        elif problem_type == 2:
                            task = 2
                            
                    # get one run's RDM
                    sub_RDM_fpath = f'subject_RDMs/sub-{sub}_task-{task}_run-{run}_roi-{roi}_{distance}_{dataType}.npy'
                    sub_RDM = np.load(sub_RDM_fpath)
                    
                    # get one run's model RDM (averaged over 4 reps)
                    avg_model_RDM = np.zeros((sub_RDM.shape[0], sub_RDM.shape[0]))
                    for rp in range(1, num_repetitions_per_run+1):
                        # convert run and subject rp to model rp:
                        # for model rp -> [0, 16) across 4 runs.
                        model_rp = (run-1) * num_repetitions_per_run + rp - 1
                        model_RDM_fpath = f'model_RDMs/sub-{sub}_task-{task}_rp-{model_rp}_{distance}.npy'
                        # print(model_RDM_fpath)
                        model_RDM = np.load(model_RDM_fpath)
                        avg_model_RDM += model_RDM
                    avg_model_RDM /= num_repetitions_per_run
                                       
                    if num_shuffles > 1:
                        shuffle_indices = np.random.choice(
                            range(sub_RDM.shape[0]), 
                            size=sub_RDM.shape[0],
                            replace=False
                        )
                        sub_RDM = sub_RDM[shuffle_indices, :]
                        
                    # compute one repetition's correlation to the ideal RDM
                    rho = compute_RSA(sub_RDM, avg_model_RDM, method=method)
                    # collects all repetitions of a run and of all subjects
                    all_rho.append(rho)
            print(
                f'Dist=[{distance}], Type=[{problem_type}], roi=[{roi}], run=[{run}], ' \
                f'avg_rho=[{np.mean(all_rho):.2f}], ' \
                f'std=[{np.std(all_rho):.2f}], ' \
                f't-stats=[{stats.ttest_1samp(a=all_rho, popmean=0)[0]:.2f}], ' \
                f'pvalue=[{stats.ttest_1samp(a=all_rho, popmean=0)[1]:.2f}]' \
            )    
        print('------------------------------------------------------------------------')


def run_level_allCombo_RSA(rois, distance, problem_type, method='spearman', dataType='beta', seed=999):
    """
    Compute corr between subject and model RDM of a run of all possible run pairs.
    Should return a higher-order RDM matrix for each subject whose entries are subject-model RDM corr
    of a given run pair. Only the diagonal has corr that of the same run for subject and model. 
    
    This way we get to see how RDMs from different runs correlate with one another 
    which will help us gain insight into whether there is any significant difference 
    across runs between subjects and models.
    """
    for roi in rois:
        average_higher_order_RDM = np.zeros((len(runs), len(runs)))
        
        for sub in subs:
            # even sub: Type1 is task2, Type2 is task3
            if int(sub) % 2 == 0:
                if problem_type == 1:
                    task = 2
                elif problem_type == 2:
                    task = 3
                    
            # odd sub: Type1 is task3, Type2 is task2
            else:
                if problem_type == 1:
                    task = 3
                elif problem_type == 2:
                    task = 2
                            
            per_sub_higher_order_RDM = np.zeros((len(runs), len(runs)))
            for i in range(len(runs)):
                for j in range(len(runs)):
                    # compute rho for (run_i, run_j)
                    sub_run = runs[i]
                    model_run = runs[j]
                    print(f'sub{sub}, sub_run{sub_run}, model_run{model_run}')
                    
                    # get one run's RDM
                    sub_RDM_fpath = f'subject_RDMs/sub-{sub}_task-{task}_run-{sub_run}_roi-{roi}_{distance}_{dataType}.npy'
                    sub_RDM = np.load(sub_RDM_fpath)
                    
                    # get one run's model RDM (averaged over 4 reps)
                    avg_model_RDM = np.zeros((sub_RDM.shape[0], sub_RDM.shape[0]))
                    for rp in range(1, num_repetitions_per_run+1):
                        # convert run and subject rp to model rp:
                        # for model rp -> [0, 16) across 4 runs.
                        model_rp = (model_run-1) * num_repetitions_per_run + rp - 1
                        model_RDM_fpath = f'model_RDMs/sub-{sub}_task-{task}_rp-{model_rp}_{distance}.npy'
                        model_RDM = np.load(model_RDM_fpath)
                        avg_model_RDM += model_RDM
                    avg_model_RDM /= num_repetitions_per_run
                    
                    # compute rho for (sub:run_i, model:run_j)
                    rho = compute_RSA(sub_RDM, avg_model_RDM, method=method)
                    per_sub_higher_order_RDM[i, j] = rho
            
            # after all run pairs, return per subject higher-order RDM
            # np.save(f'higher_order_RDM_sub-{sub}_task-{task}_roi-{roi}.npy', per_sub_higher_order_RDM)
            # and add to the average too.
            average_higher_order_RDM += per_sub_higher_order_RDM
        
        # after all subjects, average the higher_order_RDM
        average_higher_order_RDM /= num_subs
        np.save(f'higher_order_RDM_problem_type{problem_type}_roi-{roi}_{distance}_run_level.npy', average_higher_order_RDM)
            
      
def repetition_level_allCombo_RSA(rois, distance, problem_type, method='spearman', dataType='beta', seed=999):
    """
    Similar to `run_level_allCombo_RSA` but now we compute corr with repetition pairs.
    """
    for roi in rois:
        average_higher_order_RDM = np.zeros((num_repetitions, num_repetitions))
        all_higher_order_RDM = np.zeros((num_subs, num_repetitions, num_repetitions))
        
        # for sub in subs:
        for s in range(num_subs):
            sub = subs[s]
            
            # even sub: Type1 is task2, Type2 is task3
            if int(sub) % 2 == 0:
                if problem_type == 1:
                    task = 2
                elif problem_type == 2:
                    task = 3
                    
            # odd sub: Type1 is task3, Type2 is task2
            else:
                if problem_type == 1:
                    task = 3
                elif problem_type == 2:
                    task = 2
                            
            per_sub_higher_order_RDM = np.zeros((num_repetitions, num_repetitions))
            for i in range(num_repetitions):
                for j in range(num_repetitions):
                    # compute rho for (rp_i, rp_j)
                    model_rp_i = repetitions[i]   # NOTE: subject axis, need to convert
                    model_rp_j = repetitions[j]
                    
                    # get subject run and rp (subject rp 1-4 not 0-15)
                    sub_run_i = model_rp_i // 4 + 1
                    sub_rp_i = model_rp_i + 1 - (sub_run_i-1) * num_repetitions_per_run

                    # get one subject one (rp, run)'s RDM
                    sub_RDM_fpath = f'subject_RDMs/sub-{sub}_task-{task}_run-{sub_run_i}_rp-{sub_rp_i}_roi-{roi}_{distance}_{dataType}.npy'
                    sub_RDM = np.load(sub_RDM_fpath)
                    
                    # get one model one rp's RDM
                    model_RDM_fpath = f'model_RDMs/sub-{sub}_task-{task}_rp-{model_rp_j}_{distance}.npy'
                    model_RDM = np.load(model_RDM_fpath)
                                        
                    # compute rho for (sub:rp_i, model:rp_j)
                    rho = compute_RSA(sub_RDM, model_RDM, method=method)
                    per_sub_higher_order_RDM[i, j] = rho
            
            average_higher_order_RDM += per_sub_higher_order_RDM
            all_higher_order_RDM[s, :, :] = per_sub_higher_order_RDM
            
        # after all subjects, average the higher_order_RDM
        average_higher_order_RDM /= num_subs
        np.save(f'higher_order_RDM_problem_type{problem_type}_roi-{roi}_{distance}_rp_level.npy', average_higher_order_RDM)
        
        # analyse statistical difference between on and off diagonals of each subject's higher order RDM.
        # (sub, diag) -> (diag, sub)
        diag = np.diagonal(all_higher_order_RDM, axis1=1, axis2=2).T
        diag_mean = np.mean(diag, axis=0)
        diag_std = np.std(diag, axis=0)
        print(f'diag_mean={diag_mean}, \n\ndiag_std={diag_std}')
        print(f'overall diag_mean={np.mean(diag_mean):.3f}, overall diag_std={np.std(diag_std):.3f}\n\n')
        assert diag_mean.shape == (num_subs,)
        
        off_diag = []
        for s in range(num_subs):
            one_sub_mtx = all_higher_order_RDM[s, :, :]
            off_diag.append(one_sub_mtx[np.where(~np.eye(one_sub_mtx.shape[0], dtype=bool))])
        # (sub, off_diag) -> (off_diag, sub)
        off_diag = np.array(off_diag).T
        off_diag_mean = np.mean(off_diag, axis=0)
        off_diag_std = np.std(off_diag, axis=0)
        print(f'off_diag_mean={off_diag_mean}, \n\noff_diag_std={off_diag_std}')
        print(f'overall off_diag_mean={np.mean(off_diag_mean):.3f}, overall_off_diag={np.std(off_diag_std):.3f}')
        assert off_diag_mean.shape == (num_subs,)
        
        print(stats.ttest_ind(diag_mean, off_diag_mean))
            
        
def visualize_any_RDM(problem_type, roi, distance, level):
    RDM = np.load(f'higher_order_RDM_problem_type{problem_type}_roi-{roi}_{distance}_{level}_level.npy')    
    fig, ax = plt.subplots(dpi=200)
    
    for i in range(RDM.shape[0]):
        for j in range(RDM.shape[0]):
            text = ax.text(
                j, i, np.round(RDM[i, j], 3),
                ha="center", va="center", color="w"
            )
    
    ax.set_title(f'ROI: {roi}, distance: {distance}, level: {level}, Type {problem_type}')
    ax.set_xlabel('model')
    ax.set_ylabel('subject')
    ax.set_xticks(range(RDM.shape[0]))
    ax.set_yticks(range(RDM.shape[0]))
    ax.set_xticklabels([f'{level}{i+1}' for i in range(RDM.shape[0])])
    ax.set_yticklabels([f'{level}{i+1}' for i in range(RDM.shape[0])])
    plt.imshow(RDM)
    plt.savefig(f'higher_order_RDM_problem_type{problem_type}_roi-{roi}_{distance}_{level}_level.png')
    plt.close()
    print(f'[Check] plotted.')

    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    num_subs = 23
    num_repetitions = 16
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    problem_types = [1, 2]
    distance = 'pearson'
    config_version = 'best_config_sub'
    # create_model_RDMs()
    
    ############################################################################

    rois = ['V1-3', 'V4', 'LOC', 'RHHPC', 'LHHPC']
    runs = [1, 2, 3, 4]
    num_repetitions = 16
    repetitions = range(num_repetitions)
    num_repetitions_per_run = 4
    
    # repetition_level_RSA(
    #     rois=rois, 
    #     distance=distance, 
    #     problem_type=1, 
    #     num_shuffles=1
    # )
    
    # run_level_RSA(
    #     rois=rois, 
    #     distance=distance, 
    #     problem_type=2, 
    #     num_shuffles=1
    # )
    
    level = 'rp'
    rois = ['LHHPC']
    problem_type = 1
    distance = 'pearson'
    if level == 'run':
        run_level_allCombo_RSA(
            rois=rois, 
            distance=distance, 
            problem_type=problem_type
        )
    
    else:
        repetition_level_allCombo_RSA(
            rois=rois,
            distance=distance, 
            problem_type=problem_type
        )
    
    visualize_any_RDM(
        problem_type=problem_type, 
        roi=rois[0], 
        distance=distance,
        level=level
    )