import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import yaml
import argparse
import numpy as np
import pandas as pd
import multiprocessing

from main import train_model
from utils import cuda_manager, load_config

"""
Functions used for hyper-params search and evaluation.
"""

def per_subject_compare_human_to_model(sub, begin, end):
    """
    For a given subject, find the best config over
    a range of hyper-params combo.
    """
    problem_types = [1, 2, 6]
    best_config = ''
    best_diff = 99999
    for i in range(begin, end):
        # config_version = f'hyper{i}_{v}'       # for 0-22100 when sub are not separated.
        config_version = f'hyper{i}_sub{sub}_{v}'

        per_config_sum_of_abs_diff = 0
        try:
            # one subject's one config's score
            for problem_type in problem_types:
                human_lc = np.load(f'results/human/lc_type{problem_type}_sub{sub}.npy')
                model_lc = np.load(f'results/{config_version}/lc_type{problem_type}_sub{sub}.npy')
                per_config_sum_of_abs_diff += np.sum(np.abs(human_lc - model_lc))
        except FileNotFoundError:
            # catch config with missing files
            continue
        
        print(f'[sub{sub}], current config {config_version}, diff = {per_config_sum_of_abs_diff}')
        if per_config_sum_of_abs_diff < best_diff:
            best_config = config_version
            best_diff = per_config_sum_of_abs_diff
        print(f'[sub{sub}], best config {best_config}, diff = {best_diff}')
    
    # save best config's name and the best diff
    np.save(f'results/sub{sub}_best_config.npy', best_config)


def multiprocess_eval(target_func, v, num_processes, retrain=False):
    """
    Eval all choices of hyper-param combo.
    
    Optional:
        if retrain: we retrain the best config for each subject, 
        this is a workaround to get attn_weights overtime that 
        were not previously saved (only applies to hyper0-22100)

    return:
    -------
        For each subject, the best config will be saved 
        as .npy
    """
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    with multiprocessing.Pool(num_processes) as pool:
        
        for s in range(num_subs):
            sub = subs[s]
            
            results = pool.apply_async(
                target_func, args=[sub, begin, end]
            )
        
        pool.close()
        pool.join()
    
    if retrain:
        with multiprocessing.Pool(num_processes) as pool:
            for sub in subs:
                config_version = str(
                    np.load(
                        f'results/sub{sub}_best_config.npy'
                    )
                )
                results = pool.apply_async(
                    train_model,
                    args=[sub, config_version]
                )
        
            print(results.get())
            pool.close()
            pool.join()


def multiprocess_search(target_func, v, num_processes):
    """
    Train a bunch of models at once 
    by launching them to all avaialble CPU cores.
    
    One process will run one (config's all problem_types and subs)
    """
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    with multiprocessing.Pool(num_processes) as pool:
        for i in range(begin, end):
            for sub in subs:
                # the same hyper_i with different sub 
                # is a different set of params.                
                config_version = f'hyper{i}_sub{sub}_{v}'
                results = pool.apply_async(
                    target_func, 
                    args=[sub, config_version]
                )
        pool.close()
        pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # latest v=='general'
    parser.add_argument('-v', '--version', dest='v')
    # search params or eval trained candidates
    parser.add_argument('-m', '--mode', dest='mode')
    # beginning index of the hyper param choice
    parser.add_argument('-b', '--begin', dest='begin', type=int)
    # ending index of the hyper param choice (exc.)
    parser.add_argument('-e', '--end', dest='end', type=int)
    
    """
    e.g. python hyper_tune.py -v fit-human -m search -b 0 -e 1
    """

    args = parser.parse_args()
    mode = args.mode
    begin = args.begin
    end = args.end
    v = args.v
    start_time = time.time()

    if mode == 'search':
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
        multiprocess_search(
            target_func=train_model,
            v=v,
            num_processes=multiprocessing.cpu_count()-2
        )

    elif mode == 'eval':
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
        multiprocess_eval(
            target_func=per_subject_compare_human_to_model,
            v=v,
            num_processes=multiprocessing.cpu_count()-2,
        )

    end_time = time.time()
    print(f'dur = {end_time - start_time}s')