import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

from models import ClusterModel
from train import fit
from evaluations import *
from utils import load_config #, load_data, cuda_manager

try:
    from clustering.human import load_data_human_order
except ModuleNotFoundError:
    from human import load_data_human_order



def carryover(trained_model_path, new_model, num_clusters):
    """
    Transfer trained model's attn weights & clusters
    to a new initialised model. This follows Mack et al.
    """
    trained_model = tf.keras.models.load_model(trained_model_path, compile=False)
    
    # carryover cluster centers
    for i in range(num_clusters):
        new_model.get_layer(f'd{i}').set_weights(
            trained_model.get_layer(f'd{i}').get_weights()
        )

    # carryover attn weights
    new_model.get_layer(f'dimensionwise_attn_layer').set_weights(
        trained_model.get_layer(f'dimensionwise_attn_layer').get_weights()
    )
    
    # carryover cluster recruitment
    new_model.get_layer(f'mask_non_recruit').set_weights(
        trained_model.get_layer(f'mask_non_recruit').get_weights()
    )
    return new_model
    

def train_model(sub, config_version):
    """
    Train clustering model for a few times.
    
    Save:
    -----
        - save the trained model
        - save probability of error like in sustain.
    """
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
    thr = config['thr']
    
    print(f'[Check] {config_version}')
    results_path = f'results/{config_version}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    np.random.seed(random_seed)
    # --------------------------------------------------------------------------
    print(f'[Check] Beginning sub {sub}')
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
            
    if int(sub) % 2 == 0:
        problem_types = [6, 1, 2]
    else:
        problem_types = [6, 2, 1]
    for problem_type in problem_types:
        lc = np.zeros(num_repetitions)
        
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
        
        # build the graph so carryover can work (enable weights substitute)
        model.build(input_shape=(1, 3))
        
        if 'nocarryover' not in config_version:
            # carryover from type6
            if (int(sub) % 2 == 0 and problem_type == 1) \
                or (int(sub) % 2 !=0 and problem_type == 2):
                model_path = os.path.join(results_path, f'model_type6_sub{sub}')
                model = carryover(
                    trained_model_path=model_path, 
                    new_model=model, 
                    num_clusters=num_clusters
                )
        
            # carryover from 1 if 2 (for even sub)
            elif (int(sub) % 2 == 0 and problem_type == 2):
                model_path = os.path.join(results_path, f'model_type1_sub{sub}')
                model = carryover(
                    trained_model_path=model_path, 
                    new_model=model, 
                    num_clusters=num_clusters
                )
            
            # carryover from 2 if 1 (for odd sub)
            elif (int(sub) % 2 != 0 and problem_type == 1):
                model_path = os.path.join(results_path, f'model_type2_sub{sub}')
                model = carryover(
                    trained_model_path=model_path, 
                    new_model=model, 
                    num_clusters=num_clusters
                )
        else:
            print(f'[Check] No carryover is applied.')
        
        # --------------------------------------------------------------------------
        # train multiple repetitions
        # model keeps improving at this level.
        global_steps = 0
        # repetitions (0 - 15): 8trials * 16 = 128 
        # which is the same as 32trials * 4 runs in Mack file setup.
        for repetition in range(num_repetitions):
                        
            # load data of per repetition (determined order)
            dataset = load_data_human_order(
                problem_type=problem_type, 
                sub=sub, 
                repetition=repetition
            )
            
            # each repetition trains on all items once
            for i in range(len(dataset)):
                dp = dataset[i]
                x = tf.cast(dp[0], dtype=tf.float32)
                y_true = tf.cast(dp[1], dtype=tf.float32)
                signature = dp[2]
                model, item_proberror, global_steps = fit(
                    model=model,
                    x=x, 
                    y_true=y_true, 
                    signature=signature, 
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    lr=lr,
                    lr_multipliers=lr_multipliers,
                    thr=thr,
                    repetition=repetition, 
                    i=i,
                    problem_type=problem_type,
                    config_version=config_version,
                    global_steps=global_steps,
                    sub=sub,
                )
                
                print(f'[Check] item_proberror = {item_proberror}')
                lc[repetition] += item_proberror

            # save one sub's per repetition model weights.
            model.save(os.path.join(results_path, f'model_type{problem_type}_sub{sub}_rp{repetition}')) 
            
        # save one sub's lc
        lc = lc / len(dataset)
        np.save(os.path.join(results_path, f'lc_type{problem_type}_sub{sub}.npy'), lc)
        # save one sub's model weights.
        model.save(os.path.join(results_path, f'model_type{problem_type}_sub{sub}')) 
        K.clear_session()
        del model
        
        
if __name__ == '__main__':
    num_subs = 23
    num_processes = 70
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    
    import multiprocessing
    with multiprocessing.Pool(num_processes) as pool:
        for sub in subs:
            config_version = f'best_config_sub{sub}'
            results = pool.apply_async(
                    train_model, 
                    args=[sub, config_version]
                )
        pool.close()
        pool.join()