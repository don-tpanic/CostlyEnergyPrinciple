import os
import yaml
from utils import load_config


def per_subject_hyperparams_ranges(sub, v):
    """
    We set the range of each param around 
    the best combo from previous search.
    """
    # best so far 
    config_version = f'best_config_sub{sub}_{v}'
    config = load_config(
        component=None, config_version=config_version)
    lr = config['lr']
    attn_lr_multiplier = config['attn_lr_multiplier']
    Phi = config['Phi']
    specificity = config['specificity']
    thr = config['thr']
    beta = config['beta']
    temp2 = config['temp2']
    reg_strength = config['reg_strength']
    lr_attn = config['lr_attn']
    inner_loop_epochs = config['inner_loop_epochs']
    recon_clusters_weighting = config['recon_clusters_weighting']
    noise_level = config['noise_level']
    
    lr_ = [
        lr*0.5,
        lr, 
        lr*1.5,
    ]
    
    attn_lr_multiplier_ = [
        attn_lr_multiplier*0.75,
        attn_lr_multiplier,
        attn_lr_multiplier*1.25,
    ]
    
    Phi_ = [
        Phi, 
    ]
    
    specificity_ = [
        specificity, 
    ]
    
    thr_ = [
        thr,
    ]
    
    beta_ = [
        beta,
    ]
    
    temp2_ = [
        temp2,
    ]
        
    reg_strength_ = [
        reg_strength*0.75,
        reg_strength,
        reg_strength*1.5, 
    ]
    
    lr_attn_ = [
        lr_attn*0.75,
        lr_attn, 
        lr_attn*1.25, 
    ]
    
    inner_loop_epochs_ = [
        int(inner_loop_epochs*0.5),
        inner_loop_epochs,
        int(inner_loop_epochs*1.5)
    ]
    
    recon_clusters_weighting_ = [
        recon_clusters_weighting*0.5,
        recon_clusters_weighting, 
        recon_clusters_weighting*5,
    ]
    
    noise_level_ = [
        noise_level*0.75,
        noise_level, 
        noise_level*1.25,
    ]
        
    return lr_, attn_lr_multiplier_, \
                Phi_, specificity_, thr_, beta_, temp2_, reg_strength_, \
                    lr_attn_, inner_loop_epochs_, recon_clusters_weighting_, noise_level_
                        

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
    # load clustering best configs (indepedently optimised in `sustain_plus`)
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
    
    # update all clustering entries in the template
    for key in clustering_config_keys:
        if key == 'config_version':
            template[f'clustering_{key}'] = clustering_config[key]
        else:
            template[key] = clustering_config[key]
        
    # Not subject specific    
    # lr_attn_, \
    #     inner_loop_epochs_, \
    #         recon_clusters_weighting_, \
    #             noise_level_ = hyperparams_ranges()
    
    lr_, attn_lr_multiplier_, \
        Phi_, specificity_, thr_, beta_, temp2_, reg_strength_, \
            lr_attn_, inner_loop_epochs_, recon_clusters_weighting_, noise_level_ = \
                per_subject_hyperparams_ranges(sub=sub, v=v)
    
    # update all low_attn entries in template
    for lr in lr_:
        for attn_lr_multiplier in attn_lr_multiplier_:
            for reg_strength in reg_strength_:
                for lr_attn in lr_attn_:
                    for inner_loop_epochs in inner_loop_epochs_:
                        for recon_clusters_weighting in recon_clusters_weighting_:
                            for noise_level in noise_level_:
                                config_version = f'hyper{ct}_sub{sub}_{v}'
                                template['lr'] = lr
                                template['attn_lr_multiplier'] = attn_lr_multiplier
                                template['reg_strength'] = reg_strength
                                template['lr_attn'] = lr_attn
                                template['inner_loop_epochs'] = inner_loop_epochs
                                template['recon_clusters_weighting'] = recon_clusters_weighting
                                template['noise_level'] = noise_level

                                filepath = os.path.join(f'configs', f'config_{config_version}.yaml')
                                with open(filepath, 'w') as yaml_file:
                                    yaml.dump(template, yaml_file, default_flow_style=False)
                                ct += 1
                                
    print(f'sub[{sub}] Total number of candidates = {ct}')


if __name__ == '__main__':
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    for sub in subs:
        per_subject_generate_candidate_configs(
            ct=525, v='fit-human-entropy-fast', sub=sub
        )
        
    # [0, 525)
            # lr_attn_ = [0.00092, 0.0092, 0.092]
            # inner_loop_epochs_ = [2, 5, 10, 15, 20, 25, 30]
            # recon_clusters_weighting_ = [1000, 10000, 100000, 1000000, 10000000]
            # noise_level_ = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    # [525, 2712)