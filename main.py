import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import yaml
import argparse
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

from utils import load_config, cuda_manager
from models import JointModel
from train import fit
from data import data_loader_human_order
from losses import binary_crossentropy
from clustering import main as clustering_main


def train_model(sub, attn_config_version):
    # ------------------ load configs ------------------
    # load top-level config
    attn_config = load_config(
        component=None, 
        config_version=attn_config_version
    )

    # Low attn training things
    num_subs = attn_config['num_subs']
    num_repetitions = attn_config['num_repetitions']
    random_seed = attn_config['random_seed']
    lr_attn = attn_config['lr_attn']
    recon_level = attn_config['recon_level']
    inner_loop_epochs = attn_config['inner_loop_epochs']
    attn_positions = attn_config['attn_positions'].split(',')
    lr = attn_config['lr']
    from_logits = attn_config['from_logits']
    lr_multipliers = [
        attn_config['center_lr_multiplier'], 
        attn_config['attn_lr_multiplier'], 
        attn_config['asso_lr_multiplier']]
    recon_clusters_weighting = attn_config['recon_clusters_weighting']
    # ClusterModel things
    num_clusters = attn_config['num_clusters']
    
    # stimulus_set is in dcnn_config
    dcnn_config_version = attn_config['dcnn_config_version']
    dcnn_config = load_config(
        component='finetune',
        config_version=dcnn_config_version)
    stimulus_set = dcnn_config['stimulus_set']
    print(f'[Check] {attn_config_version}')
    results_path = f'results/{attn_config_version}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    np.random.seed(random_seed)
    # --------------------------------------------------------------------------
    # --- initialize models, optimizers, losses, training loops ----
    print(f'[Check] Beginning sub {sub}')
    if int(sub) % 2 == 0:
        problem_types = [6, 1, 2]
    else:
        problem_types = [6, 2, 1]
            
    for problem_type in problem_types:
        lc = np.zeros(num_repetitions)
        
        joint_model = JointModel(
            attn_config_version=attn_config_version, 
            dcnn_config_version=dcnn_config_version)
        preprocess_func = joint_model.preprocess_func
        
        # TODO: avoid hard coding shape
        joint_model.build(input_shape=[(1,224,224,3), (1, 512)])
        
        # TODO: should Adam also get carryover?
        # TODO: should DCNN(low-attn) get carryover?
        optimizer_clus = tf.keras.optimizers.SGD(learning_rate=lr)
        optimizer_attn = tf.keras.optimizers.Adam(learning_rate=lr_attn)
        loss_fn_clus = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
        
        # different level of recon uses different loss func
        if recon_level == 'cluster':
            loss_fn_attn = tf.keras.losses.MeanSquaredError()
        
        # whether there is carryover effect 
        # where attn and clusters in clustering module gets
        # carried over from one problem type to the next
        if 'nocarryover' not in attn_config_version:
            # carryover from type6
            if (int(sub) % 2 == 0 and problem_type == 1) \
                or (int(sub) % 2 !=0 and problem_type == 2):
                model_path = os.path.join(results_path, f'model_type6_sub{sub}')
                joint_model = clustering_main.carryover(
                    trained_model_path=model_path, 
                    new_model=joint_model, 
                    num_clusters=num_clusters
                )
        
            # carryover from 1 if 2 (for even sub)
            elif (int(sub) % 2 == 0 and problem_type == 2):
                model_path = os.path.join(results_path, f'model_type1_sub{sub}')
                joint_model = clustering_main.carryover(
                    trained_model_path=model_path, 
                    new_model=joint_model, 
                    num_clusters=num_clusters
                )
            
            # carryover from 2 if 1 (for odd sub)
            elif (int(sub) % 2 != 0 and problem_type == 1):
                model_path = os.path.join(results_path, f'model_type2_sub{sub}')
                joint_model = clustering_main.carryover(
                    trained_model_path=model_path, 
                    new_model=joint_model, 
                    num_clusters=num_clusters
                )
        else:
            print(f'[Check] No carryover is applied.')
            
        # --------------------------------------------------------------------------
        # train multiple epochs
        # model keeps improving at this level.
        all_recon_loss = []
        all_recon_loss_ideal = []
        all_reg_loss = []
        all_percent_zero_attn = []
        all_alphas = []
        all_centers = []
        global_steps = 0  # single step counter
        
        for repetition in range(num_repetitions):
            
            # load data of per repetition (determined order)
            dataset = data_loader_human_order(
                attn_config_version=attn_config_version, 
                problem_type=problem_type, 
                sub=sub, repetition=repetition,
                preprocess_func=preprocess_func,
            ) 
                
            for i in range(len(dataset)):
                dp = dataset[i]
                x = dp[0]
                y_true = dp[1]
                signature = dp[2]
                joint_model, attn_weights, item_proberror, \
                recon_loss_collector, recon_loss_ideal_collector, \
                reg_loss_collector, percent_zero_attn_collector, \
                alpha_collector, center_collector, global_steps, \
                optimizer_clus, optimizer_attn = fit(
                    joint_model=joint_model,
                    attn_positions=attn_positions,
                    num_clusters=num_clusters,
                    dataset=dataset,
                    x=x, 
                    y_true=y_true, 
                    signature=signature, 
                    loss_fn_clus=loss_fn_clus,
                    loss_fn_attn=loss_fn_attn,
                    optimizer_clus=optimizer_clus,
                    optimizer_attn=optimizer_attn,
                    lr_multipliers=lr_multipliers,
                    repetition=repetition, 
                    i=i,
                    sub=sub,
                    attn_config_version=attn_config_version,
                    dcnn_config_version=dcnn_config_version,
                    inner_loop_epochs=inner_loop_epochs,
                    global_steps=global_steps,
                    problem_type=problem_type,
                    recon_clusters_weighting=recon_clusters_weighting,
                )

                # record losses related to attn.
                if repetition > 0:
                    all_recon_loss.extend(recon_loss_collector)
                    all_recon_loss_ideal.extend(recon_loss_ideal_collector)
                    all_reg_loss.extend(reg_loss_collector)
                    all_percent_zero_attn.extend(percent_zero_attn_collector)
                    all_alphas.extend(alpha_collector)
                    all_centers.extend(center_collector)

                # record item-level prob error
                print(f'[Check] item_proberror = {item_proberror}')
                lc[repetition] += item_proberror
                
            # save one sub's per repetition model weights.
            model.save(os.path.join(results_path, f'model_type{problem_type}_sub{sub}_rp{repetition}'))
                        
        # ===== Saving stuff at the end of a problem type =====
        # save one sub's trained joint model.
        joint_model.save(os.path.join(results_path, f'model_type{problem_type}_sub{sub}')) 
        
        # sub in model_double's final attn weights.
        mask_non_recruit = joint_model.get_layer('mask_non_recruit').get_weights()[0]

        # save final params (attn weights, mask_non_recruit)
        np.save(
            os.path.join(
                results_path, f'attn_weights_type{problem_type}_sub{sub}_{recon_level}.npy'),
                attn_weights  # NOTE: [[aw_position1], [aw_position2], [aw_position3], ...]
        )
        np.save(
            os.path.join(
                results_path, f'mask_non_recruit_type{problem_type}_sub{sub}_{recon_level}.npy'),
                mask_non_recruit
        )
        K.clear_session()
        del joint_model

        # Save one sub's, all steps' losses, % zero attn weights, alphas, centers..
        np.save(
            os.path.join(
                results_path, f'all_recon_loss_type{problem_type}_sub{sub}_{recon_level}.npy'),
                all_recon_loss
        )
        np.save(
            os.path.join(
                results_path, f'all_recon_loss_ideal_type{problem_type}_sub{sub}_{recon_level}.npy'),
                all_recon_loss_ideal
        )
        np.save(
            os.path.join(
                results_path, f'all_reg_loss_type{problem_type}_sub{sub}_{recon_level}.npy'), 
                all_reg_loss
        )
        np.save(
            os.path.join(
                results_path, f'all_percent_zero_attn_type{problem_type}_sub{sub}_{recon_level}.npy'),
                all_percent_zero_attn
        )
        np.save(
            os.path.join(
                results_path, f'all_alphas_type{problem_type}_sub{sub}_{recon_level}.npy'),
                all_alphas
        )
        np.save(
            os.path.join(
                results_path, f'all_centers_type{problem_type}_sub{sub}_{recon_level}.npy'),
                all_centers
        )
    
        # Save per (sub, problem_type) lc.
        # per repetition need averaging over unique stimuli.
        lc = lc / len(dataset)
        np.save(os.path.join(results_path, f'lc_type{problem_type}_sub{sub}_{recon_level}.npy'), lc)


def multicuda_execute(configs, target_func):
    """
    Train a bunch of models at once
    by launching them to all available GPUs.
    """
    cuda_id_list = [0, 1, 2, 3, 4, 6]
    args_list = []
    single_entry = {}
    
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    
    for attn_config_version in configs:
        for sub in subs:
            single_entry['sub'] = sub
            single_entry['attn_config_version'] = f'{attn_config_version}_sub{sub}_fit-human'
            args_list.append(single_entry)
            single_entry = {}

    print(args_list)
    print(len(args_list))
    cuda_manager(
        target_func, args_list, cuda_id_list
    )
    

def multiprocess_search(begin, end, target_func, num_processes):
    """
    Train a bunch of models at once 
    by launching them to all avaialble CPU cores.
    
    One process will run (one config, one sub)'s all problems
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    with multiprocessing.Pool(num_processes) as pool:
        for i in range(begin, end):
            for sub in subs:
                # the same hyper_i with different sub 
                # is a different set of params.                
                config_version = f'hyper{i}_sub{sub}_fit-human'
                results = pool.apply_async(
                    target_func, 
                    args=[sub, config_version]
                )
        pool.close()
        pool.join()


if __name__ == '__main__':
    start_time = time.time()
    
    configs = []
    for i in range(0, 48):
        configs.append(f'hyper{i}')
    
    multicuda_execute(configs=configs, target_func=train_model)
    # multiprocess_search(begin=0, end=1, target_func=train_model, num_processes=70)
        
    duration = time.time() - start_time
    print(f'duration = {duration}s')