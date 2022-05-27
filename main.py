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
from data import data_loader_human_order, dict_layer2attn_size
from losses import binary_crossentropy


def carryover(trained_model_path, new_model, num_clusters, attn_position):
    """
    Transfer trained joint_model's low-attn weights into
    an initialised joint_model.
    """
    trained_model = tf.keras.models.load_model(trained_model_path, compile=False)
    
    # carryover low_attn weights
    new_model.get_layer(
        'dcnn_model').get_layer(
            f'attn_factory_{attn_position}'
    ).set_weights(
        trained_model.get_layer(
            'dcnn_model').get_layer(
                f'attn_factory_{attn_position}').get_weights()
    )
    
    # carryover cluster centers
    for i in range(num_clusters):
        new_model.get_layer(
            f'd{i}'
        ).set_weights(
            trained_model.get_layer(
                f'd{i}').get_weights()
        )

    # carryover attn weights
    new_model.get_layer(
        f'dimensionwise_attn_layer'
    ).set_weights(
        trained_model.get_layer(
            f'dimensionwise_attn_layer').get_weights()
    )
    
    # carryover cluster recruitment
    new_model.get_layer(
        'mask_non_recruit').set_weights(
            trained_model.get_layer(
                'mask_non_recruit').get_weights()
    )
    
    del trained_model
    return new_model


def train_model(sub, attn_config_version):
    # ------------------ load configs ------------------
    # load top-level config
    attn_config = load_config(
        component=None, 
        config_version=attn_config_version)
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
    image_shape = (14, 14, 512)
    
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
    # np.random.seed(random_seed)
    # --------------------------------------------------------------------------
    # --- initialize models, optimizers, losses, training loops ----
    print(f'[Check] Beginning sub {sub}')
    if int(sub) % 2 == 0:
        problem_types = [6, 1, 2]
    else:
        problem_types = [6, 2, 1]
        
    # carryover Adam.
    optimizer_attn = tf.keras.optimizers.Adam(learning_rate=lr_attn)
    for problem_type in problem_types:
        lc = np.zeros(num_repetitions)
        
        # init new model
        joint_model = JointModel(
            attn_config_version=attn_config_version, 
            dcnn_config_version=dcnn_config_version)
        
        attn_position = attn_positions[0]
        layer2attn_size = \
            dict_layer2attn_size(model_name=dcnn_config['model_name'])[attn_position]    
        joint_model.build(input_shape=[(1,)+image_shape, (1, layer2attn_size)])
        
        # NOTE(ken): Adam carryover
        optimizer_clus = tf.keras.optimizers.SGD(learning_rate=lr)
        loss_fn_clus = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
        
        # different level of recon uses different loss func
        if recon_level == 'cluster':
            loss_fn_attn = tf.keras.losses.MeanSquaredError()
        
        # when carryover
        # attn and clusters in clustering module are
        # carried over from one problem type to the next
        if 'nocarryover' not in attn_config_version:
            # carryover from type6
            if (int(sub) % 2 == 0 and problem_type == 1) \
                or (int(sub) % 2 !=0 and problem_type == 2):
                model_path = os.path.join(results_path, f'model_type6_sub{sub}')        
                joint_model = carryover(
                    trained_model_path=model_path, 
                    new_model=joint_model, 
                    num_clusters=num_clusters,
                    attn_position=attn_position
                )
        
            # carryover from 1 if 2 (for even sub)
            elif (int(sub) % 2 == 0 and problem_type == 2):
                model_path = os.path.join(results_path, f'model_type1_sub{sub}')
                joint_model = carryover(
                    trained_model_path=model_path, 
                    new_model=joint_model, 
                    num_clusters=num_clusters,
                    attn_position=attn_position
                )
            
            # carryover from 2 if 1 (for odd sub)
            elif (int(sub) % 2 != 0 and problem_type == 1):
                model_path = os.path.join(results_path, f'model_type2_sub{sub}')
                joint_model = carryover(
                    trained_model_path=model_path, 
                    new_model=joint_model, 
                    num_clusters=num_clusters,
                    attn_position=attn_position
                )
        else:
            print(f'[Check] No carryover is applied.')
            pass
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
            dataset, _, dcnn_signatures = data_loader_human_order(
                attn_config_version=attn_config_version, 
                problem_type=problem_type, 
                sub=sub, repetition=repetition,
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
                    dcnn_signatures=dcnn_signatures
                )

                # record losses related to attn.
                all_recon_loss.extend(recon_loss_collector)
                all_recon_loss_ideal.extend(recon_loss_ideal_collector)
                all_reg_loss.extend(reg_loss_collector)
                all_percent_zero_attn.extend(percent_zero_attn_collector)
                all_alphas.extend(alpha_collector)
                all_centers.extend(center_collector)

                # record item-level prob error
                print(f'[Check] item_proberror = {item_proberror}')
                lc[repetition] += item_proberror
                
            # save one sub's per repetition low_attn weights.
            # np.save(
            #     os.path.join(
            #         results_path, f'attn_weights_type{problem_type}_sub{sub}_{recon_level}_rp{repetition}.npy'),
            #         attn_weights  # NOTE: [[aw_position1], [aw_position2], [aw_position3], ...]
            # )
            
            # save one sub's per repetition high_attn weights
            np.save(
                os.path.join(
                    results_path, f'all_alphas_type{problem_type}_sub{sub}_{recon_level}_rp{repetition}.npy'),
                    all_alphas
            )
            
            # # save one sub's per repetition model
            # joint_model.save(os.path.join(results_path, f'model_type{problem_type}_sub{sub}_rp{repetition}')) 
                        
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
        # np.save(
        #     os.path.join(
        #         results_path, f'mask_non_recruit_type{problem_type}_sub{sub}_{recon_level}.npy'),
        #         mask_non_recruit
        # )
        K.clear_session()
        del joint_model

        # Save one sub's, all steps' losses, % zero attn weights, alphas, centers..
        # np.save(
        #     os.path.join(
        #         results_path, f'all_recon_loss_type{problem_type}_sub{sub}_{recon_level}.npy'),
        #         all_recon_loss
        # )
        np.save(
            os.path.join(
                results_path, f'all_recon_loss_ideal_type{problem_type}_sub{sub}_{recon_level}.npy'),
                all_recon_loss_ideal
        )
        # np.save(
        #     os.path.join(
        #         results_path, f'all_reg_loss_type{problem_type}_sub{sub}_{recon_level}.npy'), 
        #         all_reg_loss
        # )
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
        # np.save(
        #     os.path.join(
        #         results_path, f'all_centers_type{problem_type}_sub{sub}_{recon_level}.npy'),
        #         all_centers
        # )
    
        # Save per (sub, problem_type) lc.
        # per repetition need averaging over unique stimuli.
        lc = lc / len(dataset)
        np.save(os.path.join(results_path, f'lc_type{problem_type}_sub{sub}_{recon_level}.npy'), lc)


if __name__ == '__main__':
    start_time = time.time()
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_processes = 70
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with multiprocessing.Pool(num_processes) as pool:
        for s in range(len(subs)):
            sub = subs[s]
            attn_config_version = \
                f'best_config_sub{sub}_fit-human-entropy-fast'
            results = pool.apply_async(
                train_model, 
                args=[sub, attn_config_version]
            )
        print(results.get())
        pool.close()
        pool.join()
    
    duration = time.time() - start_time
    print(f'duration = {duration}s')