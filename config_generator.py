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
    high_attn_reg_strength = config['high_attn_reg_strength']
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
        
    high_attn_reg_strength_ = [
        high_attn_reg_strength*0.75,
        high_attn_reg_strength,
        high_attn_reg_strength*1.5
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
                Phi_, specificity_, thr_, beta_, temp2_, high_attn_reg_strength_, \
                    lr_attn_, inner_loop_epochs_, recon_clusters_weighting_, noise_level_
                        

def hyperparams_ranges(sub, clustering_config_version):
    """
    Return searching ranges for each hyperparameter
    in joint_model
    """
    # best so far 
    config = load_config(
        component='clustering', 
        config_version=clustering_config_version
    )
    lr = config['lr']
    attn_lr_multiplier = config['attn_lr_multiplier']
    Phi = config['Phi']
    specificity = config['specificity']
    thr = config['thr']
    beta = config['beta']
    temp2 = config['temp2']
    high_attn_reg_strength = config['high_attn_reg_strength']
    
    lr_ = [lr]
    attn_lr_multiplier_ = [attn_lr_multiplier]
    Phi_ = [Phi]
    specificity_ = [specificity]
    thr_ = [thr]
    beta_ = [beta]
    temp2_ = [temp2]
    high_attn_reg_strength_ = [high_attn_reg_strength]
    
    lr_attn_ = [0.00092, 0.0092, 0.092]
    inner_loop_epochs_ = [5, 10, 15, 20, 25, 30]
    recon_clusters_weighting_ = [1000, 10000, 100000, 1000000, 10000000]
    noise_level_ = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    return lr_, attn_lr_multiplier_, \
                Phi_, specificity_, thr_, beta_, temp2_, high_attn_reg_strength_, \
                    lr_attn_, inner_loop_epochs_, recon_clusters_weighting_, noise_level_


def per_subject_generate_candidate_configs(ct, v, sub, subj_general=False):
    """
    Given a joint_model best config as template, we replace params regarding clustering model with 
    subject-specific best config. Then we iterate through hypers for training low_level attn
    in the DCNN and save the configs.
    """
    # load clustering best configs (indepedently optimised in `sustain_plus` before move onto joint).
    clustering_config_version = f'best_config_sub{sub}_fit-human-entropy-nocarryover'
    clustering_config = load_config(
        component='clustering', config_version=clustering_config_version)
    clustering_config_keys = clustering_config.keys()
    
    # use the joint_model config as template
    # NOTE(ken), the template is used so we have all the params names,
    # it does not matter which version is the template as long as 
    # it is not the same as the latest. 
    # e.g., if we plan on searching for joint_model of `fit-human-entropy-fast-nocarryover`
    # the template config could be anything before it.
    template_config_version = f'best_config_sub{sub}_fit-human-entropy'
    template = load_config(component=None, config_version=template_config_version)   
    template['clustering_config_version'] = ''
    template_keys = template.keys()
    
    # update all clustering entries in the template
    for key in clustering_config_keys:
        if key == 'config_version':
            template[f'clustering_{key}'] = clustering_config[key]
        else:
            template[key] = clustering_config[key]

    # NOTE(ken), one complication is that subj_general should always be run 
    # before subject specific searches. This is due to after we intergrate the 
    # best configs from the `sustain_plus` we do not have best configs for joint_model
    # hence we cannot go on search in a subject-specific way. This should really 
    # be done in an automatic manner in the future. i.e.
    # subj_general -> subj_specific under a single execute.
    if not subj_general:
        lr_, attn_lr_multiplier_, \
            Phi_, specificity_, thr_, beta_, temp2_, high_attn_reg_strength_, \
                lr_attn_, inner_loop_epochs_, recon_clusters_weighting_, noise_level_ = \
                    per_subject_hyperparams_ranges(
                        sub=sub, 
                        clustering_config_version=clustering_config_version
                    )
    else:
        lr_, attn_lr_multiplier_, \
            Phi_, specificity_, thr_, beta_, temp2_, high_attn_reg_strength_, \
                lr_attn_, inner_loop_epochs_, recon_clusters_weighting_, noise_level_ = \
                    hyperparams_ranges(
                        sub=sub, 
                        clustering_config_version=clustering_config_version
                    )
    
    # update all searchable entries in template
    for lr in lr_:
        for attn_lr_multiplier in attn_lr_multiplier_:
            for high_attn_reg_strength in high_attn_reg_strength_:
                for lr_attn in lr_attn_:
                    for inner_loop_epochs in inner_loop_epochs_:
                        for recon_clusters_weighting in recon_clusters_weighting_:
                            for noise_level in noise_level_:
                                config_version = f'hyper{ct}_sub{sub}_{v}'
                                template['lr'] = lr
                                template['attn_lr_multiplier'] = attn_lr_multiplier
                                template['high_attn_reg_strength'] = high_attn_reg_strength
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
            ct=0, 
            v='fit-human-entropy-fast-nocarryover', 
            sub=sub,
            subj_general=True
        )
        
    # [0, 450): subj_general=True, clustering not searched.
            # lr_attn_ = [0.00092, 0.0092, 0.092]
            # inner_loop_epochs_ = [5, 10, 15, 20, 25, 30]
            # recon_clusters_weighting_ = [1000, 10000, 100000, 1000000, 10000000]
            # noise_level_ = [0.2, 0.3, 0.4, 0.5, 0.6]
            