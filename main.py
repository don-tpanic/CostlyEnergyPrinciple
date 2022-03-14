import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import yaml
import argparse
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

from utils import load_config, cuda_manager
from models import JointModel
from train import fit
from data import data_loader_V2
from losses import binary_crossentropy


def train_model(
        problem_type, 
        config_version,
    ):

    # ------------------ load configs ------------------
    # load top-level config
    config = load_config(config_version=config_version)

    # Low attn training things
    num_runs = config['num_runs']
    num_blocks = config['num_blocks']
    random_seed = config['random_seed']
    lr_low_attn = config['lr_low_attn']
    recon_level = config['recon_level']
    inner_loop_epochs = config['inner_loop_epochs']
    low_attn_positions = config['low_attn_positions'].split(',')
    lr_clus = config['lr_clus']
    from_logits = config['from_logits']
    lr_multipliers = [
        config['center_lr_multiplier'], 
        config['attn_lr_multiplier'], 
        config['asso_lr_multiplier']
    ]
    recon_clusters_weighting = config['recon_clusters_weighting']
    # ClusterModel things
    num_clusters = config['num_clusters']
    stimulus_set = config['stimulus_set']
    print(f'[Check] Type={problem_type}, {config_version}')
    results_path = f'results/{config_version}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    np.random.seed(random_seed)

    # --- initialize models, optimizers, losses, training loops ----
    lc = np.empty(num_blocks)
    ct = 0
    for run in range(num_runs):
        print(f'[Check] Beginning run {run}')

        optimizer_clus = tf.keras.optimizers.SGD(learning_rate=lr_clus)
        optimizer_attn = tf.keras.optimizers.Adam(learning_rate=lr_low_attn)
        loss_fn_clus = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
        
        # different level of recon uses different loss func
        if recon_level == 'cluster':
            loss_fn_attn = tf.keras.losses.MeanSquaredError()
        
        joint_model = JointModel(config_version=config_version)
        preprocess_func = joint_model.preprocess_func
        assoc_weights = np.random.uniform(
            low=0, high=0, size=(num_clusters, 2)
        )
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

        # load dataset in same order.
        dataset, counter_balancing = data_loader_V2(
            config_version=config_version,
            preprocess_func=preprocess_func,
            problem_type=problem_type,
            random_seed=run
        )
        # save each run's counter balancing used for later 
        # in `evaluations.py`
        np.save(
            os.path.join(
                results_path, f'counter_balancing_type{problem_type}_run{run}_{recon_level}.npy'),
                counter_balancing
        )
        
        for epoch in range(num_blocks):
            # shuffle for every epoch
            run2indices = np.load(f'run2indices_num_runs={num_runs}.npy')
            shuffled_indices = run2indices[run][epoch]
            shuffled_dataset = dataset[shuffled_indices]
            print('[Check] shuffled_indices', shuffled_indices)

            for i in range(len(shuffled_dataset)):
                print(f'\n\n **** [Check] epoch={epoch}, item={i} ****')
                dp = shuffled_dataset[i]
                x = dp[0]
                y_true = dp[1]
                signature = dp[2]
                
                joint_model, attn_weights, item_proberror, \
                recon_loss_collector, recon_loss_ideal_collector, \
                reg_loss_collector, percent_zero_attn_collector, \
                alpha_collector, center_collector, global_steps, \
                optimizer_clus, optimizer_attn = fit(
                    joint_model=joint_model,
                    low_attn_positions=low_attn_positions,
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
                    epoch=epoch, 
                    i=i,
                    run=run,
                    config_version=config_version,
                    inner_loop_epochs=inner_loop_epochs,
                    global_steps=global_steps,
                    problem_type=problem_type,
                    recon_clusters_weighting=recon_clusters_weighting,
                )

                # record losses related to attn.
                if epoch > 0:
                    all_recon_loss.extend(recon_loss_collector)
                    all_recon_loss_ideal.extend(recon_loss_ideal_collector)
                    all_reg_loss.extend(reg_loss_collector)
                    all_percent_zero_attn.extend(percent_zero_attn_collector)
                    all_alphas.extend(alpha_collector)
                    all_centers.extend(center_collector)

                    # save trial-level (epoch,i) DCNN attn_weights (all positions)
                    # np.save(
                    #     os.path.join(
                    #         results_path, 
                    #         f'attn_weights_type{problem_type}_run{run}_epoch{epoch}_i{i}_{recon_level}.npy'
                    #     ),
                    #     attn_weights
                    # )

                # record item-level prob error
                print(f'[Check] item_proberror = {item_proberror}')
                lc[epoch] += item_proberror
                ct += 1
            print(f'>> run=[{run}], epoch=[{epoch}]')
            print('---------\n')
            
            
        # ===== Saving stuff at the end of each run =====
        # save one run's trained joint model.
        joint_model.save(os.path.join(results_path, f'model_type{problem_type}_run{run}')) 
        
        # sub in model_double's final attn weights.
        mask_non_recruit = joint_model.get_layer('mask_non_recruit').get_weights()[0]

        # save final params (attn weights, mask_non_recruit)
        np.save(
            os.path.join(
                results_path, f'attn_weights_type{problem_type}_run{run}_{recon_level}.npy'),
                attn_weights  # NOTE: [[aw_position1], [aw_position2], [aw_position3], ...]
        )
        np.save(
            os.path.join(
                results_path, f'mask_non_recruit_type{problem_type}_run{run}_{recon_level}.npy'),
                mask_non_recruit
        )
        K.clear_session()
        del joint_model

        # Save one run's, all steps' losses, % zero attn weights, alphas, centers..
        np.save(
            os.path.join(
                results_path, f'all_recon_loss_type{problem_type}_run{run}_{recon_level}.npy'),
                all_recon_loss
        )
        np.save(
            os.path.join(
                results_path, f'all_recon_loss_ideal_type{problem_type}_run{run}_{recon_level}.npy'),
                all_recon_loss_ideal
        )
        np.save(
            os.path.join(
                results_path, f'all_reg_loss_type{problem_type}_run{run}_{recon_level}.npy'), 
                all_reg_loss
        )
        np.save(
            os.path.join(
                results_path, f'all_percent_zero_attn_type{problem_type}_run{run}_{recon_level}.npy'),
                all_percent_zero_attn
        )
        np.save(
            os.path.join(
                results_path, f'all_alphas_type{problem_type}_run{run}_{recon_level}.npy'),
                all_alphas
        )
        np.save(
            os.path.join(
                results_path, f'all_centers_type{problem_type}_run{run}_{recon_level}.npy'),
                all_centers
        )
    
    # Save average lc across all runs
    assert num_runs * num_blocks * len(dataset) == ct, f'got incorrect ct = {ct}'
    lc = lc / (num_runs * len(dataset))
    # save lc across all runs like sustain.
    np.save(os.path.join(results_path, f'lc_type{problem_type}_{recon_level}.npy'), lc)


def multicuda_execute(
        target_func, 
        version_begin, version_end, 
        cuda_id_list, 
        run_begin, run_end
    ):
    """
    Train a bunch of models at once
    by launching them to all available GPUs.
    """    
    num_types = 6
    args_list = []
    single_entry = {}

    for v in range(version_begin, version_end+1):
        # for run in range(run_begin, run_end+1):
        for run in [36, 38, 41, 42, 43, 44]:
            config_version = f'v{v}_naive-withNoise-t1.vgg16.block4_pool.None.run{run}-with-lowAttn'

            for problem_type in range(1, num_types+1):
                single_entry['problem_type'] = problem_type
                single_entry['config_version'] = config_version
                args_list.append(single_entry)
                single_entry = {}

    print(args_list)
    print(len(args_list))
    cuda_manager(
        target_func, args_list, cuda_id_list
    )


def multiprocess_execute(
        target_func, 
        version_begin,
        version_end,
        num_processes,
        run_begin,
        run_end
    ):
    """
    Train a bunch of models at once 
    by launching them to all avaialble CPU cores.
    
    One process will run one (config & problem_type)
    """
    num_types = 6

    with multiprocessing.Pool(num_processes) as pool:

        for v in range(version_begin, version_end+1):
            for run in range(begin, end+1):
                config_version = f'v{v}_naive-withNoise-t1.vgg16.block4_pool.None.run{run}-with-lowAttn'

                for problem_type in range(1, num_types+1):
                    results = pool.apply_async(
                        target_func, args=[problem_type, config_version]
                    )
            
        pool.close()
        pool.join()
        print(results.get())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', dest='mode')
    parser.add_argument('-c', '--config', dest='config_version', default=None)
    parser.add_argument('-p', '--problem_type', type=int, dest='problem_type', default=None)
    parser.add_argument('-gpu', '--gpu_index', dest='gpu_index', default=None)
    parser.add_argument('-cpu', '--num_cpus', type=int, dest='num_cpus', default=None)
    parser.add_argument('-vb', '--version_begin', type=int, dest='version_begin', default=None)
    parser.add_argument('-ve', '--version_end', type=int, dest='version_end', default=None)
    parser.add_argument('-rb', '--run_begin', type=int, dest='run_begin', default=None)
    parser.add_argument('-re', '--run_end', type=int, dest='run_end', default=None)

    args = parser.parse_args()
    mode = args.mode
    config_version = args.config_version
    problem_type = args.problem_type
    gpu_index = args.gpu_index
    num_cpus = args.num_cpus
    version_begin = args.version_begin
    version_end = args.version_end
    run_begin = args.run_begin
    run_end = args.run_end

    start_time = time.time()
    if mode == 'train':
        # Train one problem_type at a time on single GPU
        if problem_type and gpu_index:
            print(f'*** Run Type: {problem_type} ***')
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_index}"
            train_model(
                problem_type=problem_type, 
                config_version=config_version,
            )

        # Do multi-GPU or multi-CPU training 
        # for all when there is no problem_type specified.
        else:
            # run on cpus when provided
            if num_cpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
                multiprocess_execute(
                    target_func=train_model,
                    version_begin=version_begin,
                    version_end=version_end,
                    num_processes=num_cpus,
                    run_begin=run_begin,
                    run_end=run_end
                )

            # otherwise run on gpus
            else:
                multicuda_execute(
                    target_func=train_model,
                    version_begin=version_begin,
                    version_end=version_end,
                    cuda_id_list=[0, 1, 2, 3, 4, 6],
                    run_begin=run_begin,
                    run_end=run_end
                )

    duration = time.time() - start_time
    print(f'duration = {duration}s')