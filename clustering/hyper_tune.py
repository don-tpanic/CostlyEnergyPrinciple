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
from evaluations import examine_lc, examine_recruited_clusters_n_attn
from utils import cuda_manager, load_config
    

def search(problem_type, v):    
    for i in range(begin, end):
        config_version = f'hyper{i}_{v}'        
        train_model(
            problem_type=problem_type, 
            config_version=config_version
        )


def eval_candidates(v):
    """
    Eval all choices of hyper-param combo.

    return:
    -------
        A list of hyper-param indices that satisfy 
        the criteria.
    """
    qualified = []
    for i in range(begin, end):
        config_version = f'hyper{i}_{v}'
        config = load_config(config_version)

        print(f'----------------------------------------')
        try:
            areas = examine_lc(
                config_version, 
                plot_learn_curves=True
            )
        except FileNotFoundError:
            print(f'{config_version} incomplete!')
            continue
        
        print(f'{config_version}')
        if (np.max(areas) != areas[-1] 
            or areas[0] > 3 
            or areas[2] - areas[1] < 0.5
            or areas[-1] > 8
            or (areas[2] > areas[3] and areas[2] > areas[4])
        ):
            continue
        
        else:
            qualified.append(i)

    np.save(f'qualified_{begin}-{end}.npy', qualified)

                
def multicuda_execute(target_func, v, cuda_id_list):
    num_types = 6
    args_list = []
    single_entry = {}
    for problem_type in range(1, num_types+1):
        single_entry['problem_type'] = problem_type
        single_entry['v'] = v
        args_list.append(single_entry)
        single_entry = {}

    # Execute
    cuda_manager(
        target_func, args_list, cuda_id_list
    )


def multiprocess_execute(target_func, v, num_processes):
    """
    Train a bunch of models at once 
    by launching them to all avaialble CPU cores.
    
    One process will run one (config & problem_type)
    """
    num_types = 6

    with multiprocessing.Pool(num_processes) as pool:

        for i in range(begin, end):
            config_version = f'hyper{i}_{v}'

            for problem_type in range(1, num_types+1):
                results = pool.apply_async(
                    target_func, args=[problem_type, config_version]
                )
        
        pool.close()
        pool.join()
        print(results.get())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # latest v=='general'
    parser.add_argument('-v', '--version', dest='v')
    # run code on CPU or GPU
    parser.add_argument('-d', '--device', dest='device')
    # search params or eval trained candidates
    parser.add_argument('-m', '--mode', dest='mode')
    # beginning index of the hyper param choice
    parser.add_argument('-b', '--begin', dest='begin', type=int)
    # ending index of the hyper param choice (exc.)
    parser.add_argument('-e', '--end', dest='end', type=int)

    args = parser.parse_args()
    mode = args.mode
    device = args.device
    begin = args.begin
    end = args.end
    v = args.v
    start_time = time.time()

    if mode == 'search':
        if device == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
            multiprocess_execute(
                target_func=train_model,
                v=v,
                num_processes=multiprocessing.cpu_count()-2
            )
        elif device == 'gpu':
            multicuda_execute(
                target_func=search,
                v=v,
                cuda_id_list=[0, 1, 2, 3, 4, 6]
            )

    elif mode == 'eval':
        eval_candidates(v=v)

    end_time = time.time()
    print(f'dur = {end_time - start_time}s')