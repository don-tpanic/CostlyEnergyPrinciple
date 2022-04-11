import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

import tensorflow as tf
from tensorflow.keras import backend as K

from models import ClusterModel
from utils import load_config, load_data, load_X_only
from human import load_data_human_order, reorder_RDM_entries_into_chunks

"""
Prepare for RSA: compute RDM of each subject & problem_type
out of cluster-level activations.

Note, with carryover, we only need to load the final model.
But for testing the code without carryover, we can load model
by problem_type.

Also notice, we only analyze second run of learning same as Mack et al.
"""

def load_trained_model(config_version, problem_type, sub):
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
    
    model_path = os.path.join(results_path, f'model_type{problem_type}_sub{sub}')
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


def return_RDM(problem_type, sub, distance):
    """
    Produce RDM of a (sub, problem_type)
    
    Impl:
    -----

    """
    if not os.path.exists(rdm_path):
        os.mkdir(rdm_path)
        
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
        
    RDM_fpath = f'{rdm_path}/sub-{sub}_task-{task}_{distance}.npy'
        
    # For now, let's just use the final repetition
    model = load_trained_model(config_version, problem_type, sub)
    
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
    
    
def visualize_RDM(sub, problem_type, distance):
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
        f'{rdm_path}/sub-{sub}_task-{task}_{distance}.npy'
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
    plt.savefig(
        f'RDMs/sub-{sub}_task-{task}_{distance}.png'
    )
    print(f'[Check] plotted.')

    
def execute():
    for problem_type in problem_types:
        for sub in subs:
            return_RDM(problem_type, sub, distance)
            visualize_RDM(sub, problem_type, distance)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    rdm_path = 'RDMs'
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    problem_types = [1, 2]
    distance = 'euclidean'
    config_version = 'best_config_sub'
    execute()