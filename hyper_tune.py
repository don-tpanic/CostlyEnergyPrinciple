import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# https://stackoverflow.com/questions/63336300/tensorflow-2-0-utilize-all-cpu-cores-100
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import time
import yaml
import argparse
import subprocess
import numpy as np
import pandas as pd
import multiprocessing

from main import train_model
from utils import cuda_manager, load_config
from evaluations import examine_subject_lc_and_attn_overtime, \
    compare_across_types_V3


def per_subject_compare_human_to_model(sub, v, hyper_begin, hyper_end):
    """
    For a given subject, find the best config over
    a range of hyper-params combo.
    """
    problem_types = [1, 2, 6]
    best_config = f'best_config_sub{sub}_{v}'
    best_diff_recorder = np.load('best_diff_recorder.npy', allow_pickle=True)
    best_diff = best_diff_recorder.ravel()[0][sub]
    # best_diff = 999
    for i in range(hyper_begin, hyper_end):
        config_version = f'hyper{i}_sub{sub}_{v}'

        per_config_mse = 0
        try:
            # one subject's one config's score
            for problem_type in problem_types:
                human_lc = np.load(f'clustering/results/human/lc_type{problem_type}_sub{sub}.npy')
                model_lc = np.load(f'results/{config_version}/lc_type{problem_type}_sub{sub}_cluster.npy')
                per_config_mse += np.mean( (human_lc - model_lc)**2 )
        except FileNotFoundError:
            # catch config with missing files
            continue
        
        # print(f'[sub{sub}], current config {config_version}, diff = {per_config_mse}')
        if per_config_mse < best_diff:
            best_config = config_version
            best_diff = per_config_mse
    
    print(f'[sub{sub}], best config {best_config}, diff = {best_diff}')
    
    # override the current best config 
    # by the best config we just found
    subprocess.run(
        ['cp', 
         f'configs/config_{best_config}.yaml', 
         f'configs/config_best_config_sub{sub}_{v}.yaml'
        ]
    )


def multiprocess_train(target_func, subs, v, hyper_begin, hyper_end, num_processes):
    """
    Train a bunch of models at once 
    by launching them to all avaialble CPU cores.
    """
    with multiprocessing.Pool(num_processes) as pool:
        for i in range(hyper_begin, hyper_end):
            for sub in subs:
                # the same hyper_i with different sub 
                # is a different set of params.                
                config_version = f'hyper{i}_sub{sub}_{v}'
                results = pool.apply_async(
                    target_func, 
                    args=[sub, config_version]
                )
        print(results.get())
        pool.close()
        pool.join()


def multiprocess_eval(subs, v, hyper_begin, hyper_end, num_processes):
    """
    Eval all choices of hyper-param combo.
    After finding the best configs so far, retrain best configs.
    
    return:
    -------
        For each subject, the best config will be overriden.
    """
    # eval lc and obtain best configs.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with multiprocessing.Pool(num_processes) as pool:
        for s in range(len(subs)):
            sub = subs[s]
            results = pool.apply_async(
                per_subject_compare_human_to_model, 
                args=[sub, v, hyper_begin, hyper_end]
            )
        print(results.get())
        pool.close()
        pool.join()
    
    # retrain to get best config named files.
    with multiprocessing.Pool(num_processes) as pool:
        for s in range(len(subs)):
            sub = subs[s]
            attn_config_version = \
                f'best_config_sub{sub}_{v}'
            results = pool.apply_async(
                train_model, 
                args=[sub, attn_config_version]
            )
        print(results.get())
        pool.close()
        pool.join()
    
    # evaluate the lc of each sub's best config.
    print(f'[Check] evaluating best config lc...')
    examine_subject_lc_and_attn_overtime('best_config', v=v)
    compare_across_types_V3('best_config', v=v)
    
    
if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    # search params or eval trained candidates
    parser.add_argument('-m', '--mode', dest='mode')
    # beginning index of the hyper param choice
    parser.add_argument('-b', '--begin', dest='begin', type=int)
    # ending index of the hyper param choice (exc.)
    parser.add_argument('-e', '--end', dest='end', type=int)
    
    """
    e.g. python hyper_tune.py -m search -b 0 -e 1
    """

    args = parser.parse_args()
    mode = args.mode
    hyper_begin = args.begin
    hyper_end = args.end
    
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    v = 'fit-human-entropy-fast'
    num_processes = 72
    
    if mode == 'search':
        multiprocess_train(
            target_func=train_model, 
            subs=subs,
            v=v, 
            hyper_begin=hyper_begin, 
            hyper_end=hyper_end, 
            num_processes=num_processes
        )
    
    elif mode == 'eval':
        multiprocess_eval(
            subs=subs,
            v=v,
            hyper_begin=hyper_begin, 
            hyper_end=hyper_end, 
            num_processes=num_processes
        )
    
    duration = time.time() - start_time
    print(f'duration = {duration}s')
