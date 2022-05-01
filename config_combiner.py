import os
import yaml 
from utils import load_config

"""
Given the best config from `clustering/configs/`
combine it with config for the joint_model from `configs/`
to produce top-level best_configs that has all variables 
for both clustering and sustain module in the joint model.

Impl:
-----
"""

joint_model_config_version = 'v0_naive-withNoise'

num_subs = 23 
subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
num_subs = len(subs)

joint_config = load_config(
    component=None, 
    config_version=joint_model_config_version
)
joint_config['clustering_config_version'] = ''
joint_keys = joint_config.keys()

for sub in subs:
    config_version = f'best_config_sub{sub}'
    config = load_config(
        component='clustering',
        config_version=config_version)
    config_keys = config.keys()
    
    for key in config_keys:
        if key == 'config_version':
            joint_config[f'clustering_{key}'] = config[key]
        else:
            joint_config[key] = config[key]
    
    
    filepath = os.path.join(f'configs', f'config_{config_version}.yaml')
    with open(filepath, 'w') as yaml_file:
        yaml.dump(joint_config, yaml_file, default_flow_style=False)
        
    exit()