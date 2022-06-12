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
from evaluations import overall_eval


def select_best_config(i, v):
    """
    Go over all hyper-params combo, and find the best config.
    The selection is based on multiple criteria. 
    1. Diff to lc of human behaviour.
    2. Statistical significance of %zero attn between Types.
    3. Statistical significance of the direction of the decoding.
    """
    print(f'hyper{i}')
    t_type1v2, _, \
        t_type2v6, _, \
            t_decoding, _, \
                per_config_mse_all_subs = \
                    overall_eval(attn_config_version=f'hyper{i}', v=v)
    
    return per_config_mse_all_subs


def multiprocess_train(target_func, subs, v, hyper_begin, hyper_end, num_processes):
    """
    Train a bunch of models at once 
    by launching them to all avaialble CPU cores.
    """
    with multiprocessing.Pool(num_processes, maxtasksperchild=1) as pool:
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


def multiprocess_eval(target_func, v, hyper_begin, hyper_end, num_processes):
    """
    Go over all hyper-params combo, and find the best config.
    The selection is based on multiple criteria. 
    1. Diff to lc of human behaviour.
    2. Statistical significance of %zero attn between Types.
    3. Statistical significance of the direction of the decoding.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with multiprocessing.Pool(num_processes) as pool:
        
        results_obj_collector = []
        for i in range(hyper_begin, hyper_end):
            results = pool.apply_async(
                target_func, 
                args=[i, v]
            )
            
            results_obj_collector.append(results)
            
        # print(results.get())
        pool.close()
        pool.join()
    
    results_collector = [res.get() for res in results_obj_collector]
    print(range(hyper_begin, hyper_end)[np.argmin(results_collector)])
        
        
    
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
    v = 'fit-human-entropy-fast-nocarryover'
    num_processes = 70
    
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
            target_func=select_best_config,
            v=v,
            hyper_begin=hyper_begin, 
            hyper_end=hyper_end, 
            num_processes=num_processes
        )
    
    duration = time.time() - start_time
    print(f'duration = {duration}s')
