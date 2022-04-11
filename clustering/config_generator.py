import os
import yaml
import numpy as np

from utils import load_config

"""
Automatically generate a bunch of config files 
by iterating through a range of params.
"""

def per_subject_hyperparams_ranges(sub):
    """
    We set the range of each param around 
    the best combo from previous search.
    """
    # -----------------------------
    lr_margin = 0.1
    Phi_margin = 2
    specificity_margin = 0.1
    thr_margin = 0.05
    beta_margin = 0.5
    temp2_margin = 0.1
    # -----------------------------
    
    # best so far 
    config_version = str(np.load(f'results/sub{sub}_best_config.npy'))
    config = load_config(config_version)
    lr = config['lr']
    Phi = config['Phi']
    specificity = config['specificity']
    thr = config['thr']
    beta = config['beta']
    temp2 = config['temp2']
    
    lr_ = [
        lr-lr_margin*2, 
        lr-lr_margin*1, 
        lr, 
        lr+lr_margin*1, 
        lr+lr_margin*2
    ]
    center_lr_multiplier_ = [1]
    attn_lr_multiplier_ = [
        0.5, 
        1, 
        5, 
    ]
    asso_lr_multiplier_ = [1]
    Phi_ = [
        # Phi-Phi_margin*2, 
        Phi-Phi_margin*1, 
        Phi, 
        Phi+Phi_margin*1, 
        # Phi+Phi_margin*2
    ]
    specificity_ = [
        # specificity-specificity_margin*2, 
        specificity-specificity_margin*1, 
        specificity, 
        specificity+specificity_margin*1, 
        # specificity+specificity_margin*2
    ]
    thr_ = [
        # thr-thr_margin*2, 
        thr-thr_margin*1, 
        thr, 
        thr+thr_margin*1, 
        # thr+thr_margin*2
    ]
    beta_ = [
        # beta-beta_margin*2,
        beta-beta_margin*1,
        beta,
        beta+beta_margin*1,
        # beta+beta_margin*2
    ]
    temp2_ = [
        # temp2-temp2_margin*2,
        temp2-temp2_margin*1,
        temp2,
        temp2+temp2_margin*1,
        # temp2+temp2_margin*2
    ]
        
    return lr_, center_lr_multiplier_, \
        attn_lr_multiplier_, asso_lr_multiplier_, \
            Phi_, specificity_, thr_, beta_, temp2_


def per_subject_generate_candidate_configs(ct, v, sub):
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
    
    lr_, center_lr_multiplier_, \
    attn_lr_multiplier_, asso_lr_multiplier_, \
    Phi_, specificity_, thr_, beta_, temp2_ = per_subject_hyperparams_ranges(sub)
    
    for lr in lr_:        
        for center_lr_multiplier in center_lr_multiplier_:
            for attn_lr_multiplier in attn_lr_multiplier_:
                for asso_lr_multiplier in asso_lr_multiplier_:
                    for Phi in Phi_:
                        for specificity in specificity_:
                            for thr in thr_:    
                                for beta in beta_:
                                    for temp2 in temp2_:
                                        config_version = f'hyper{ct}_sub{sub}_{v}' 
                                        template['config_version'] = config_version
                                        template['lr'] = float(lr)
                                        template['center_lr_multiplier'] = float(center_lr_multiplier)
                                        template['attn_lr_multiplier'] = float(attn_lr_multiplier)
                                        template['asso_lr_multiplier'] = float(asso_lr_multiplier)
                                        template['Phi'] = float(Phi)
                                        template['specificity'] = float(specificity)
                                        template['thr'] = float(thr)
                                        template['beta'] = float(beta)
                                        template['temp2'] = float(temp2)
                                            
                                        print(template)

                                        filepath = os.path.join('configs', f'config_{config_version}.yaml')
                                        with open(filepath, 'w') as yaml_file:
                                            yaml.dump(template, yaml_file, default_flow_style=False)
                                        ct += 1
                                
    print(f'Total number of candidates = {ct}')


def hyperparams_ranges():
    """
    Return searching ranges for each hyperparameter
    in the cluster model.
    """
    lr_ = [0.3, 0.4, 0.5, 0.6, 0.7]      # 0.5
    center_lr_multiplier_ = [1]                 
    attn_lr_multiplier_ = [1]
    asso_lr_multiplier_ = [1]
    Phi_ = [12, 15, 18, 21]                   # 15
    specificity_ = [0.3, 0.4, 0.5, 0.6, 0.7]       # 0.5
    thr_ = [-0.55, -0.5, -0.45, -0.4, -0.35]  # -0.45
    beta_ = [4, 4.5, 5, 5.5, 6]               # 5
    temp2_ = [0.7, 0.6, 0.5, 0.4, 0.3]        # 0.5
    
    
    return lr_, center_lr_multiplier_, \
        attn_lr_multiplier_, asso_lr_multiplier_, \
            Phi_, specificity_, thr_, beta_, temp2_


def generate_candidate_configs(ct, v):
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
    
    lr_, center_lr_multiplier_, \
    attn_lr_multiplier_, asso_lr_multiplier_, \
    Phi_, specificity_, thr_, beta_, temp2_ = hyperparams_ranges()
    
    for lr in lr_:        
        for center_lr_multiplier in center_lr_multiplier_:
            for attn_lr_multiplier in attn_lr_multiplier_:
                for asso_lr_multiplier in asso_lr_multiplier_:
                    for Phi in Phi_:
                        for specificity in specificity_:
                            for thr in thr_:    
                                for beta in beta_:
                                    for temp2 in temp2_:
                                        config_version = f'hyper{ct}_{v}' 
                                        template['config_version'] = config_version
                                        template['lr'] = float(lr)
                                        template['center_lr_multiplier'] = float(center_lr_multiplier)
                                        template['attn_lr_multiplier'] = float(attn_lr_multiplier)
                                        template['asso_lr_multiplier'] = float(asso_lr_multiplier)
                                        template['Phi'] = float(Phi)
                                        template['specificity'] = float(specificity)
                                        template['thr'] = float(thr)
                                        template['beta'] = float(beta)
                                        template['temp2'] = float(temp2)
                                            
                                        print(template)

                                        filepath = os.path.join('configs', f'config_{config_version}.yaml')
                                        with open(filepath, 'w') as yaml_file:
                                            yaml.dump(template, yaml_file, default_flow_style=False)
                                        ct += 1
                                
    print(f'Total number of candidates = {ct}')
    

if __name__ == '__main__':
    # generate_candidate_configs(
    #     ct=22101, v='fit-human'
    # )
    
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    for sub in subs:
        per_subject_generate_candidate_configs(
            ct=0, v='fit-human', sub=sub
        )