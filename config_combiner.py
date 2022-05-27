import os
import yaml 
from utils import load_config

"""
We search hyper-params of the joint_model in two separate stages.
First, we search hyper-params for clustering module for each subject,
independent to DCNN. 

Second, we freeze clustering hyper-params in clustering module, and 
search for hyper-params in DCNN.

Therefore, after we search for clustering module, we need to carry
over the best configs into configs of the joint_model so that we could
start the second stage search. 

This script does the `carryover` by first loading the current best config
of the joint_model (with out dated hyper-params for clustering), then we 
load the best configs of clustering and we substitute the hyper-params 
of the clustering configs into the joint_model configs. This way, 
the joint_model configs will have best configs from clustering and could 
continue to search hyper-params for DCNN.
"""

def merge(v='fit-human-entropy-fast'):
    num_subs = 23 
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)

    for sub in subs:
        clustering_config_version = f'best_config_sub{sub}_fit-human-entropy'            # best config from fit-human-entropy for clustering
        joint_config_version = f'best_config_sub{sub}_fit-human-entropy'   # best config from fit-human-entropy for joint model
        
        # get the current joint_model best config
        # subject-specific
        joint_config = load_config(
            component=None, 
            config_version=joint_config_version)
        joint_config['clustering_config_version'] = ''
        
        # load the current best clustering model config
        # (clustering params not update-to-date)
        config = load_config(
            component='clustering',
            config_version=clustering_config_version)
        config_keys = config.keys()
        
        # sub clustering model params into
        # the joint_model config
        for key in config_keys:
            if key == 'config_version':
                joint_config[f'clustering_{key}'] = config[key]
            else:
                joint_config[key] = config[key]
        
        # save the best config overall.
        filepath = os.path.join(f'configs', f'config_best_config_sub{sub}_{v}.yaml')
        with open(filepath, 'w') as yaml_file:
            yaml.dump(joint_config, yaml_file, default_flow_style=False)


if __name__ == '__main__':
    merge()