import os
import yaml
import numpy as np

"""
Automatically generate a bunch of config files 
by iterating through a range of params.
"""

def hyperparams_ranges():
    """
    Return searching ranges for each hyperparameter
    in the cluster model.
    """
    lr_ = [0.1, 0.15]
    center_lr_multiplier_ = [1]
    attn_lr_multiplier_ = [1]
    asso_lr_multiplier_ = [1]
    Phi_ = [10, 10.5, 11, 11.5, 12, 12.5]
    specificity_ = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    attn_constraint_ = ['sumtoone']
    thr_ = [-0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1]
    beta_ = [1, 1.5, 2, 2.5, 3, 3.5]
    temp_ = [0.012]
    
    
    return lr_, center_lr_multiplier_, \
        attn_lr_multiplier_, asso_lr_multiplier_, \
            Phi_, specificity_, attn_constraint_, thr_, beta_, temp_
            

def generate_candidate_configs(ct, v):
    template = {
        'config_version': 'config_3',
        'num_runs': 10,
        'num_blocks': 32,
        'random_seed': 999,
        'from_logits': True,
        'num_clusters': 8,
        'r': 2,
        'q': 1,
        'attn_constraint': 'nonneg',
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
        'beta': 1.25
    }
    
    lr_, center_lr_multiplier_, \
    attn_lr_multiplier_, asso_lr_multiplier_, \
    Phi_, specificity_, attn_constraint_, thr_, beta_, temp_ = hyperparams_ranges()
    
    for lr in lr_:        
        for center_lr_multiplier in center_lr_multiplier_:
            for attn_lr_multiplier in attn_lr_multiplier_:
                for asso_lr_multiplier in asso_lr_multiplier_:
                    for Phi in Phi_:
                        for specificity in specificity_:
                            for attn_constraint in attn_constraint_:
                                for thr in thr_:    
                                    for beta in beta_:
                                        for temp in temp_:                          
                                            config_version = f'hyper{ct}_{v}' 
                                            template['config_version'] = config_version
                                            template['lr'] = float(lr)
                                            template['center_lr_multiplier'] = float(center_lr_multiplier)
                                            template['attn_lr_multiplier'] = float(attn_lr_multiplier)
                                            template['asso_lr_multiplier'] = float(asso_lr_multiplier)
                                            template['Phi'] = float(Phi)
                                            template['specificity'] = float(specificity)
                                            template['attn_constraint'] = attn_constraint
                                            template['thr'] = float(thr)
                                            template['beta'] = float(beta)
                                            template['temp'] = float(temp)

                                            print(template)

                                            filepath = os.path.join('configs', f'config_{config_version}.yaml')
                                            with open(filepath, 'w') as yaml_file:
                                                yaml.dump(template, yaml_file, default_flow_style=False)
                                            ct += 1
                                
    print(f'Total number of candidates = {ct}')
    

if __name__ == '__main__':
    generate_candidate_configs(
        ct=10080, v='general'
    )