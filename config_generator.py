import os
import yaml
from utils import load_config


def hyperparams_ranges():
    """
    Return searching ranges for each hyperparameter
    in joint_model
    """
    lr_attn_ = [0.00092, 0.0092, 0.092]
    inner_loop_epochs_ = [5, 10, 15, 20]
    recon_clusters_weighting_ = [1000, 10000, 100000, 1000000]
    noise_level_ = [0.3, 0.4, 0.5]
    
    return lr_attn_, inner_loop_epochs_, recon_clusters_weighting_, noise_level_


def per_subject_generate_candidate_configs(ct, v, sub, template_version='v4_naive-withNoise'):
    """
    Given a joint_model config as template, we replace params regarding clustering model with 
    subject-specific best config. Then we iterate through hypers for training low_level attn
    in the DCNN and save the configs.
    """
    clustering_config = load_config(
        component='clustering',
        config_version=f'best_config_sub{sub}')
    clustering_config_keys = clustering_config.keys()
    
    # use the joint_model config as template
    template = load_config(
        component=None, 
        config_version=template_version)
    template['clustering_config_version'] = ''
    template_keys = template.keys()
    
    # update all clustering entries in template
    for key in clustering_config_keys:
        if key == 'config_version':
            template[f'clustering_{key}'] = clustering_config[key]
        else:
            template[key] = clustering_config[key]
                
    # TODO: this could be later adjusted to per sub
    lr_attn_, \
        inner_loop_epochs_, \
            recon_clusters_weighting_, \
                noise_level_ = hyperparams_ranges()

    # update all low_attn entries in template
    for lr_attn in lr_attn_:
        for inner_loop_epochs in inner_loop_epochs_:
            for recon_clusters_weighting in recon_clusters_weighting_:
                for noise_level in noise_level_:
                    config_version = f'hyper{ct}_sub{sub}_{v}'
                    template['lr_attn'] = lr_attn
                    template['inner_loop_epochs'] = inner_loop_epochs
                    template['recon_clusters_weighting'] = recon_clusters_weighting
                    template['noise_level'] = noise_level

                    filepath = os.path.join(f'configs', f'config_{config_version}.yaml')
                    with open(filepath, 'w') as yaml_file:
                        yaml.dump(template, yaml_file, default_flow_style=False)
                    ct += 1
                                
    print(f'Total number of candidates = {ct}')


if __name__ == '__main__':
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    for sub in subs:
        per_subject_generate_candidate_configs(
            ct=0, v='fit-human', sub=sub
        )
        