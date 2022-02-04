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
from utils import load_config, load_data, cuda_manager


def train_model(
        problem_type, 
        config_version, 
        save_model=True, 
        save_proberror=True
    ):
    """
    Train clustering model for a few times.
    
    Save:
    -----
        if save_model:
            save the trained model
        if save_proberror:
            save probability of error like in sustain.
    """
    config = load_config(config_version)
    num_runs = config['num_runs']
    num_blocks = config['num_blocks']
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
    temp = config['temp']
    print(f'[Check] Type={problem_type}, {config_version}')
    results_path = f'results/{config_version}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    np.random.seed(random_seed)
    # --------------------------------------------------------------------------
    lc = np.empty(num_blocks)
    ct = 0
    for run in range(num_runs):
        print(f'[Check] Beginning run {run}')
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

        # a new initialised model (all weights frozen)
        model = ClusterModel(
            num_clusters=num_clusters, r=r, q=q, 
            specificity=specificity, 
            trainable_specificity=trainable_specificity, 
            attn_constraint=attn_constraint,
            Phi=Phi, 
            actv_func=actv_func,
            beta=beta,
            temp=temp,
        )
        assoc_weights = np.random.uniform(
            low=0, high=0, size=(num_clusters, 2)
        )
        # model.get_layer('classification').set_weights([assoc_weights])
        # --------------------------------------------------------------------------

        # train multiple epochs
        # model keeps improving at this level.
        global_steps = 0
        for epoch in range(num_blocks):

            print(f'[Check] epoch = {epoch}')
            # load and shuffle data
            dataset = load_data(problem_type)
            run2indices = np.load(f'run2indices_num_runs={num_runs}.npy')
            shuffled_indices = run2indices[run][epoch]
            shuffled_dataset = dataset[shuffled_indices]
            print('[Check] shuffled_indices', shuffled_indices)

            # each epoch trains on all items
            for i in range(len(shuffled_dataset)):
                dp = shuffled_dataset[i]
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
                    epoch=epoch, 
                    i=i,
                    problem_type=problem_type,
                    run=run,
                    config_version=config_version,
                    global_steps=global_steps
                )
                
                print(f'[Check] item_proberror = {item_proberror}')
                lc[epoch] += item_proberror
                ct += 1
            
            print(f'[Check] type = [{problem_type}], run=[{run}], epoch=[{epoch}]')
            print('---------\n')

        # save one run's model weights.
        if save_model:
            model.save(os.path.join(results_path, f'model_type{problem_type}_run{run}')) 
        K.clear_session()
        del model
    
    assert num_runs * num_blocks * len(dataset) == ct, f'got incorrect ct = {ct}'
    lc = lc / (num_runs * len(dataset))
    if save_proberror:
        np.save(os.path.join(results_path, f'lc_type{problem_type}.npy'), lc)


def multicuda_execute(target_func, config_list):
    """
    Train a bunch of models at once
    by launching them to all available GPUs.
    """
    num_types = 6
    cuda_id_list = [0, 1, 2, 3, 4, 6]

    args_list = []
    single_entry = {}
    for config_version in config_list:
        for problem_type in range(1, num_types+1):
            single_entry['problem_type'] = problem_type
            single_entry['config_version'] = config_version
            args_list.append(single_entry)
            single_entry = {}

    print(args_list)
    print(len(args_list))

    # Execute multiprocess code.
    cuda_manager(
        target_func, args_list, cuda_id_list
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config_version')
    parser.add_argument('-p', '--problem_type', default=None, type=int, dest='problem_type')
    parser.add_argument('-g', '--gpu_index', default='0', dest='gpu_index')
    parser.add_argument('-m', '--mode', dest='mode')
    args = parser.parse_args()
    config_version = args.config_version
    problem_type = args.problem_type
    gpu_index = args.gpu_index
    mode = args.mode

    start_time = time.time()
    if mode == 'train':
        # Execute a single problem type.
        if problem_type:
            print(f'*** Run problem type = {problem_type} ***')
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_index}"
            train_model(problem_type=problem_type, config_version=config_version)
        # Execute all problem types.
        else:
            multicuda_execute(
                target_func=train_model, 
                config_list=['sustain_v1']
            )

    duration = time.time() - start_time
    print(f'duration = {duration}s')