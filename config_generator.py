import os
import yaml
from utils import load_config


# TODO: if use this, we do not need the merge below.
def per_subject_hyperparams_ranges(sub, v):
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
    reg_strength_margin = 0.1
    # -----------------------------
    
    # best so far 
    config_version = f'best_config_sub{sub}_{v}'
    config = load_config(config_version)
    lr = config['lr']
    attn_lr_multiplier = config['attn_lr_multiplier']
    Phi = config['Phi']
    specificity = config['specificity']
    thr = config['thr']
    beta = config['beta']
    temp2 = config['temp2']
    
    lr_ = np.abs([
        # lr-lr_margin*2,
        lr-lr_margin*1, 
        lr, 
        lr+lr_margin*1, 
        # lr+lr_margin*2
    ])
    center_lr_multiplier_ = [1]
    attn_lr_multiplier_ = [
        0.05,
        0.1,
        0.5, 
        1, 
        2,
        attn_lr_multiplier
    ]
    asso_lr_multiplier_ = [1]
    Phi_ = np.abs([
        # Phi-Phi_margin*2, 
        Phi-Phi_margin*1, 
        Phi, 
        Phi+Phi_margin*1, 
        # Phi+Phi_margin*2
    ])
    specificity_ = np.abs([
        # specificity-specificity_margin*2, 
        specificity-specificity_margin*1, 
        specificity, 
        specificity+specificity_margin*1, 
        # specificity+specificity_margin*2
    ])
    
    # thr_ = np.abs([
    #     # thr-thr_margin*2, 
    #     thr-thr_margin*1, 
    #     thr, 
    #     thr+thr_margin*1, 
    #     # thr+thr_margin*2
    # ])

    thr_ = [thr]
    
    beta_ = np.abs([
        # beta-beta_margin*2,
        beta-beta_margin*1,
        beta,
        beta+beta_margin*1,
        # beta+beta_margin*2
    ])
    temp2_ = np.abs([
        # temp2-temp2_margin*2,
        temp2-temp2_margin*1,
        temp2,
        temp2+temp2_margin*1,
        # temp2+temp2_margin*2
    ])
    
    reg_strength_ = [0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
        
    return lr_, center_lr_multiplier_, \
        attn_lr_multiplier_, asso_lr_multiplier_, \
            Phi_, specificity_, thr_, beta_, temp2_, reg_strength_


def hyperparams_ranges():
    """
    Return searching ranges for each hyperparameter
    in joint_model
    """
    lr_attn_ = [0.00092, 0.0092, 0.092]
    inner_loop_epochs_ = [2, 5, 10, 15, 20, 25, 30]
    recon_clusters_weighting_ = [1000, 10000, 100000, 1000000, 10000000]
    noise_level_ = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    return lr_attn_, inner_loop_epochs_, recon_clusters_weighting_, noise_level_


def per_subject_generate_candidate_configs(ct, v, sub):
    """
    Given a joint_model best config as template, we replace params regarding clustering model with 
    subject-specific best config. Then we iterate through hypers for training low_level attn
    in the DCNN and save the configs.
    """
    clustering_config = load_config(
        component='clustering',
        config_version=f'best_config_sub{sub}_fit-human-entropy')
    clustering_config_keys = clustering_config.keys()
    
    # use the joint_model config as template
    template = load_config(
        component=None, 
        config_version=f'best_config_sub{sub}_fit-human-entropy')
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
            ct=0, v='fit-human-entropy-fast', sub=sub
        )
        
    # [0, 525)
            # lr_attn_ = [0.00092, 0.0092, 0.092]
            # inner_loop_epochs_ = [2, 5, 10, 15, 20, 25, 30]
            # recon_clusters_weighting_ = [1000, 10000, 100000, 1000000, 10000000]
            # noise_level_ = [0.2, 0.3, 0.4, 0.5, 0.6]
        
    
        