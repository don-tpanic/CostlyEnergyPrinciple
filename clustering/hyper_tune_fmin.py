import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import yaml
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from scipy import optimize

import main_fmin
import main
from utils import cuda_manager, load_config

"""
Functions used for hyper-params search and evaluation.
"""

template = {
    'config_version': 'config_3',
    'num_subs': 23,
    'num_repetitions': 16,
    'random_seed': 999,
    'from_logits': True,
    'num_clusters': 8,
    'r': 2,
    'q': 1,
    'attn_constraint': 'sumtoone',
    'actv_func': 'softmax',
    'lr': 0.3,
    'center_lr_multiplier': 1,
    'attn_lr_multiplier': 1,
    'asso_lr_multiplier': 1,
    'Phi': 1.5,
    'specificity': 1,
    'trainable_specificity': False,
    'unsup_rule': 'threshold',
    'thr': 999,
    'beta': 1.25,
    'temp1': 'equivalent'
}

def multiprocess_search(v, num_processes):
    """
    For each sub (a process):
        1. Load the params of the best config from prev search, 
            which is based on `sub{sub}_best_config.npy`
            
        2. Run fmin over the params (collect the best params),
        
        3. Get sub's best params and save as normal config file,
            named `config_best_config_sub{sub}.yaml` and the prev
            best config will be overriden `sub{sub}_best_config.npy`
            
        4. Retrain a model for this sub using the best config,
            to get lc and attn overtime for eval.
    """
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    results_collector = []
    with multiprocessing.Pool(num_processes) as pool:
        
        for sub in subs:
            config_version = str(np.load(f'results/sub{sub}_best_config.npy'))
            config = load_config(config_version)
            lr = config['lr']
            center_lr_multiplier = config['center_lr_multiplier']
            attn_lr_multiplier = config['attn_lr_multiplier']
            asso_lr_multiplier = config['asso_lr_multiplier']
            specificity = config['specificity']
            Phi = config['Phi'] 
            beta = config['beta']
            temp2 = config['temp2']
            thr = config['thr']
            
            # initial guess
            x0 = [
                lr, 
                center_lr_multiplier,
                attn_lr_multiplier, 
                asso_lr_multiplier,
                specificity, 
                Phi, 
                beta, 
                temp2,
                thr
            ]
            results = pool.apply_async(
                optimize.fmin,
                args=(main_fmin.train_model, x0, (sub, config_version)),
                kwds={'maxiter': 18000, 'full_output': 1}
            )
            results_collector.append(results)
            
        pool.close()
        pool.join()
        
        # collect each subject's best params 
        # save to config
        for s in range(num_subs):
            sub = subs[s]
            
            xopt = results_collector[s].get()[0]
            lr = xopt[0]
            center_lr_multiplier = xopt[1]
            attn_lr_multiplier = xopt[2]
            asso_lr_multiplier = xopt[3]
            specificity = xopt[4]
            Phi = xopt[5]
            beta = xopt[6]
            temp2 = xopt[7]
            thr = xopt[8]
            
            config_version = f'best_config_sub{sub}'
            template['config_version'] = config_version
            template['lr'] = float(lr)
            template['center_lr_multiplier'] = float(center_lr_multiplier)
            template['attn_lr_multiplier'] = float(attn_lr_multiplier)
            template['asso_lr_multiplier'] = float(asso_lr_multiplier)
            template['Phi'] = float(Phi)
            template['specificity'] = float(specificity)
            template['beta'] = float(beta)
            template['temp2'] = float(temp2)
            template['thr'] = float(thr)
            
            filepath = os.path.join('configs', f'config_{config_version}.yaml')
            with open(filepath, 'w') as yaml_file:
                yaml.dump(template, yaml_file, default_flow_style=False)
            
            # save best_config is a bit redundant now because 
            # the best config is best config but not hyper[xxx],
            # we do this to keep the rest of the code functioning for now.
            np.save(f'results/sub{sub}_best_config.npy', config_version)
            print(f'[Check] saved sub{sub}_best_config.npy')
    
    # once we have the best configs, we retrain these models (23 of them so fast)
    # to save lc and attn weights for evaluations. 
    # A cleaner way should not have to do this but using some sort of callback.
    with multiprocessing.Pool(num_processes) as pool:
        for sub in subs:
            config_version = str(np.load(f'results/sub{sub}_best_config.npy'))
            results = pool.apply_async(
                    main.train_model, 
                    args=[sub, config_version]
                )
        pool.close()
        pool.join()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--maxiter', dest='maxiter')
    # """
    # e.g. python hyper_tune.py -i 4200
    # """
    # args = parser.parse_args()
    # maxiter = args.maxiter
    start_time = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    multiprocess_search(
        v='fit-human',
        num_processes=multiprocessing.cpu_count()-2,
    )

    end_time = time.time()
    print(f'dur = {end_time - start_time}s')