import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import multiprocessing
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from sklearn.metrics import pairwise_distances

from models import DCNN, JointModel
from utils import load_config
from data import data_loader_human_order, load_X_only, dict_layer2attn_size
from clustering.human import reorder_RDM_entries_into_chunks

from matplotlib import rc
# rc('text', usetex=True)
plt.rcParams.update({'font.size': 4})

"""Producing model RDMs from specified layer and
correlate with brain RDMs from ROIs.
"""

def load_trained_model(
        attn_config_version, 
        problem_type, 
        sub, 
        repetition, 
        repr_level,
        image_shape=(14, 14, 512)):
    """
    Load a trained model, intercepted at some layer,
    specified by `repr_level`.
    """
    # load configs
    attn_config = load_config(component=None, config_version=attn_config_version)
    dcnn_config_version = attn_config['dcnn_config_version']
    dcnn_config = load_config(component='finetune', config_version=dcnn_config_version)
    print(f'[Check] {attn_config_version}')
    attn_position = attn_config['attn_positions'].split(',')[0]
    num_clusters = attn_config['num_clusters']
    results_path = f'results/{attn_config_version}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # load trained joint_model whose weights will be sub into 
    # a new joint_model that will be intercepted.
    model_path = os.path.join(results_path, f'model_type{problem_type}_sub{sub}_rp{repetition}')
    trained_model = tf.keras.models.load_model(model_path, compile=False)
    
    # load empty joint_model
    model = JointModel(attn_config_version=attn_config_version, 
                       dcnn_config_version=dcnn_config_version)
    preprocess_func = model.preprocess_func
    layer2attn_size = dict_layer2attn_size(model_name=dcnn_config['model_name'])[attn_position]
    # due to subclassing, we need to specify input. 
    # currently we only support a single attn layer.
    model.build(input_shape=[(1,) + image_shape, (1, layer2attn_size)])
    
    # as long as its not the baseline model without low-attn
    # we will sub in trained attn weights regardless of `repr_level`
    if 'no_attn' not in repr_level:
        print('[Check] sub in trained DCNN attn weights')
        model.get_layer(
            'dcnn_model').get_layer(
                f'attn_factory_{attn_position}').set_weights(
                    trained_model.get_layer(
                        'dcnn_model').get_layer(
                            f'attn_factory_{attn_position}').get_weights())
    
    # return joint_model until post-attn activation
    if 'LOC' in repr_level:
        inputs = model.get_layer('dcnn_model').input
        layer_reprs = model.get_layer(
            'dcnn_model').get_layer(
                f'post_attn_actv_{attn_position}').output
        model = Model(inputs=inputs, outputs=layer_reprs)
    
    # sub in trained weights from clustering module.
    # no need to intercept at cluster becuz the joint
    # model has output for it.
    elif 'cluster' in repr_level:                
        # carryover cluster centers
        for i in range(num_clusters):
            model.get_layer(
                f'd{i}').set_weights(
                    trained_model.get_layer(f'd{i}').get_weights())

        # carryover high-attn weights
        model.get_layer(
            'dimensionwise_attn_layer').set_weights(
                trained_model.get_layer(
                    'dimensionwise_attn_layer').get_weights())
        
        # carryover cluster recruitment
        model.get_layer(
            'mask_non_recruit').set_weights(
                trained_model.get_layer(
                    'mask_non_recruit').get_weights())
        
        # carryover association weights
        model.get_layer(
            'classification').set_weights(
                trained_model.get_layer(
                    'classification').get_weights())
    
    # TODO: temp - test early layers of DCNN
    elif '_no_attn' in repr_level:
        inputs = model.get_layer('dcnn_model').input
        layer_reprs = model.get_layer(
            'dcnn_model').get_layer(f'{repr_level[:-8]}').output
        model = Model(inputs=inputs, outputs=layer_reprs)
    
    del trained_model
    return model, preprocess_func


def return_n_visualize_RDM(
        config_version, 
        problem_type, 
        sub, 
        distance, 
        repetition, 
        repr_level, 
    ):
    """Produce RDM of a (sub, problem_type) & Visualize it.
    Notice, a tricky thing in implementation is there are two order 
    conversion. The first time is to rearrange raw image batch into
    subject-specific sorted order; the second time is to rearrange 
    RDM entries (both row and col) into category structure based on
    binary labels.
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
        
    RDM_fpath = f'model_RDMs/sub-{sub}_task-{task}_rp-{repetition}_{distance}_{repr_level}.npy'
    
    # load a trained model at a repetition
    model, preprocess_func = \
        load_trained_model(
            attn_config_version=config_version, 
            problem_type=problem_type, 
            sub=sub, repetition=repetition, 
            repr_level=repr_level,
        )
    
    dataset, subj_signatures, _ = data_loader_human_order(
        attn_config_version=config_version, 
        problem_type=problem_type, 
        sub=sub, 
        repetition=repetition)
    
    # We convert the images into order 000, 001, ...
    # in terms of SUBJECT coding. We do this because 
    # later when we rearrange RDM based on categorical 
    # structure, the ordering was based on subject coding
    # rather than dcnn coding (video note available)
    batch_x = load_X_only(
        dataset=dataset, 
        attn_config_version=config_version)
    conversion_ordering = np.argsort(subj_signatures)
    batch_x[0] = batch_x[0][conversion_ordering]
    
    # compute target layer output reprs
    if 'LOC' in repr_level:
        layer_reprs = model(batch_x)
        layer_reprs = tf.reshape(layer_reprs, [layer_reprs.shape[0], -1])
        assert layer_reprs.shape == (8, 100352)
        
    elif 'cluster' in repr_level:
        _, layer_reprs, _, _ = model(batch_x)
        assert layer_reprs.shape == (8, 8)
    
    # TODO: temp - testing early layers of DCNN
    elif 'block' in repr_level:
        layer_reprs = model(batch_x)
        layer_reprs = tf.reshape(layer_reprs, [layer_reprs.shape[0], -1])
        
    # produce RDM given distance metric
    if distance == 'euclidean':
        RDM = pairwise_distances(layer_reprs, metric=distance)
    elif distance == 'pearson':
        RDM = pairwise_distances(layer_reprs, metric='correlation')
    
    # rearrange based on category structure
    reorder_mapper = reorder_RDM_entries_into_chunks()
    conversion_ordering = reorder_mapper[sub][task]
    RDM = RDM[conversion_ordering, :][:, conversion_ordering]
    np.save(RDM_fpath, RDM)
    print(f'[Check] Saved: {RDM_fpath}')
    
    # visualize as heatmap
    visualize_RDM(sub, problem_type, distance, repetition, repr_level)


def visualize_RDM(sub, problem_type, distance, repetition, repr_level):
    """
    Visualize subject's RDM given a problem_type
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

    RDM = np.load(
        f'model_RDMs/sub-{sub}_task-{task}_rp-{repetition}_{distance}_{repr_level}.npy'
    )
    
    fig, ax = plt.subplots()
    for i in range(RDM.shape[0]):
        for j in range(RDM.shape[0]):
            text = ax.text(
                j, i, np.round(RDM[i, j], 1),
                ha="center", va="center", color="w"
            )
    
    ax.set_title(f'sub: {sub}, distance: {distance}, Type {problem_type}, repr: {repr_level}')
    plt.imshow(RDM)
    plt.savefig(f'model_RDMs/sub-{sub}_task-{task}_rp-{repetition}_{distance}_{repr_level}.pdf')
    plt.close()
    print(f'[Check] plotted.')


def create_model_RDMs(
        config_version, 
        problem_types,
        subs, distance,
        num_repetitions,
        repr_levels, 
        num_processes):
    """Create (save and visualize) per (problem_type, sub) RDMs.
    """            
    with multiprocessing.Pool(num_processes) as pool:
        for repr_level in repr_levels:
            for problem_type in problem_types:
                for sub in subs:
                    for repetition in range(num_repetitions):
                        print(f'repetition = {repetition}')
                        results = pool.apply_async(
                            return_n_visualize_RDM, 
                            args=[
                                f'{config_version}_sub{sub}_fit-human-entropy-fast-nocarryover', 
                                problem_type, 
                                sub, 
                                distance, 
                                repetition, 
                                repr_level, 
                            ]
                        )
        print(results.get())                
        pool.close()
        pool.join()


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


def run_level_RSA(
        repr_level,
        rois, 
        distance, 
        problem_type, 
        num_shuffles=1, 
        method='spearman', 
        dataType='beta', 
        seed=999):
    """Subject-model RSA at the run level. 
    Subject RDMs are computed using run-level GLM estimates 
    and model RDMs are computed using repetition-level representations 
    (at specified layer) but averaged over repetitions within that run. 
    Notice, for model, there isn't such thing as run but we could convert based 
    on the subject run and average over repetitions.
    """
    for roi in rois:
        for run in runs:  
            np.random.seed(seed)
            per_run_all_rhos = []
            for shuffle in range(num_shuffles):
                for sub in subs:
                    # even sub: Type1 is task2, Type2 is task3
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
                            
                    # get one run's RDM
                    sub_RDM_fpath = \
                        f'clustering/subject_RDMs/'\
                        f'sub-{sub}_task-{task}_run-{run}_roi-{roi}_{distance}_{dataType}.npy'
                    sub_RDM = np.load(sub_RDM_fpath)
                    
                    # get one run's model RDM (averaged over 4 reps)
                    avg_model_RDM = np.zeros((sub_RDM.shape[0], sub_RDM.shape[0]))
                    for rp in range(1, num_repetitions_per_run+1):
                        # convert run and subject rp to model rp:
                        # for model rp -> [0, 16) across 4 runs.
                        model_rp = (run-1) * num_repetitions_per_run + rp - 1
                        model_RDM_fpath = \
                            f'model_RDMs/' \
                            f'sub-{sub}_task-{task}_rp-{model_rp}_{distance}_{repr_level}.npy'
                        # print(f'[Check] run={run}, rp={rp}, sub={sub}, model_RDM_fpath={model_RDM_fpath}')
                        model_RDM = np.load(model_RDM_fpath)
                        avg_model_RDM += model_RDM
                    avg_model_RDM /= num_repetitions_per_run
                                       
                    if num_shuffles > 1:
                        shuffle_indices = np.random.choice(
                            range(sub_RDM.shape[0]), 
                            size=sub_RDM.shape[0],
                            replace=False)
                        sub_RDM = sub_RDM[shuffle_indices, :]
                        
                    # compute one repetition's correlation to the ideal RDM
                    rho = compute_RSA(sub_RDM, avg_model_RDM, method=method)
                    # collects all repetitions of a run and of all subjects
                    per_run_all_rhos.append(rho)
                    
            # print per run results
            print(
                f'Dist=[{distance}], Type=[{problem_type}], roi=[{roi}], run=[{run}], ' \
                f'avg_rho=[{np.mean(per_run_all_rhos):.2f}], ' \
                f'std=[{np.std(per_run_all_rhos):.2f}], ' \
                f't-stats=[{stats.ttest_1samp(a=per_run_all_rhos, popmean=0)[0]:.2f}], ' \
                f'pvalue=[{stats.ttest_1samp(a=per_run_all_rhos, popmean=0)[1]:.2f}]' \
            )    
        print('------------------------------------------------------------------------')
        




def compare_pre_and_post_attn_actv_RSA(problem_type, distance='pearson'):
    """
    Check how correlated pre and post-attn
    activations are.
    """
    for problem_type in problem_types:
        all_rhos = []
        for sub in subs:
            for repetition in range(num_repetitions):
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
                            
                RDM_pre_fpath = \
                    f'model_RDMs/' \
                    f'sub-{sub}_task-{task}_rp-{repetition}_{distance}_LOC_no_attn.npy'
                RDM_pre = np.load(RDM_pre_fpath)
                
                RDM_post_fpath = \
                    f'model_RDMs/' \
                    f'sub-{sub}_task-{task}_rp-{repetition}_{distance}_LOC.npy'
                RDM_post = np.load(RDM_post_fpath)
                
                rho = compute_RSA(RDM_pre, RDM_post, method='spearman')
                print(f'sub{sub}, repetition={repetition}, rho={rho}')
                all_rhos.append(rho)

    print(f'avg_rho={np.mean(all_rhos):.3f}, std={np.std(all_rhos):.3f}')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    
    config_version = 'hyper89'
    repr_levels = ['LOC', 'cluster', 'LOC_no_attn']
    problem_types = [1, 2, 6]
    runs = [1, 2, 3, 4]
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_repetitions_per_run = 4
    num_repetitions = 16
    distance = 'pearson'
    num_processes = 72
    
    # create_model_RDMs(
    #     config_version=config_version, 
    #     problem_types=problem_types,
    #     subs=subs, 
    #     distance=distance,
    #     num_repetitions=num_repetitions,
    #     repr_levels=repr_levels, 
    #     num_processes=num_processes)

    # repr_level = 'LOC_no_attn'
    # rois = ['LOC']
    # problem_type = 1
    # run_level_RSA(
    #     repr_level=repr_level,
    #     rois=rois, 
    #     distance=distance, 
    #     problem_type=problem_type,
    #     num_shuffles=1)
    
    compare_pre_and_post_attn_actv_RSA(problem_type=1)