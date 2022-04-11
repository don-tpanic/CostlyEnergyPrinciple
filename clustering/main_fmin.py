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
from main import carryover
from evaluations import *
from utils import load_config, load_data, cuda_manager

from human import load_data_human_order
    

def train_model(x0, sub, config_version):
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
    lr = x0[0]
    center_lr_multiplier = x0[1]
    attn_lr_multiplier = x0[2]
    asso_lr_multiplier = x0[3]
    lr_multipliers = [center_lr_multiplier, attn_lr_multiplier, asso_lr_multiplier]
    num_clusters = config['num_clusters']
    r = config['r']
    q = config['q']
    specificity = x0[4]
    trainable_specificity = config['trainable_specificity']
    attn_constraint = config['attn_constraint']
    Phi = x0[5]
    actv_func = config['actv_func']
    beta = x0[6]
    temp1 = config['temp1']
    temp2 = x0[7]
    thr = x0[8]
    
    print(f'[Check] trying x0 = {x0}')
    
    print(f'[Check] {config_version}')
    results_path = f'results/best_config_sub{sub}'
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
        
    per_config_sum_of_abs_diff = 0
    for problem_type in problem_types:
        print(f'[Check] problem_type = {problem_type}')
        lc = np.empty(num_repetitions)
        
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
            pass
            # print(f'[Check] No carryover is applied.')
        
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
                
                # print(f'[Check] item_proberror = {item_proberror}')
                lc[repetition] = item_proberror
        
        human_lc = np.load(f'results/human/lc_type{problem_type}_sub{sub}.npy')
        # save one sub's model weights.
        model.save(os.path.join(results_path, f'model_type{problem_type}_sub{sub}')) 
        K.clear_session()
        del model
        
        model_lc = lc
        per_config_sum_of_abs_diff += np.sum(np.abs(human_lc - model_lc))
    
    # the one value we want to minimize.
    print(f'total diff = {per_config_sum_of_abs_diff}')
    return per_config_sum_of_abs_diff

        
        
if __name__ == '__main__':
    config_version = 'nocarryover'
    
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    for sub in subs:
        train_model(sub=sub, config_version=config_version)