import os
import yaml
from utils import load_config


def per_subject_hyperparams_ranges(sub, v, DCNN_config_version):
    """
    We set the range of each param around 
    the best combo from previous search.
    """
    # best so far 
    config_version = f'{DCNN_config_version}_sub{sub}_{v}'
    config = load_config(
        component=None, config_version=config_version)
    
    # Clustering
    lr = config['lr']
    center_lr_multiplier = config['center_lr_multiplier']
    attn_lr_multiplier = config['attn_lr_multiplier']
    asso_lr_multiplier = config['asso_lr_multiplier']
    Phi = config['Phi']
    specificity = config['specificity']
    thr = config['thr']
    beta = config['beta']
    temp2 = config['temp2']
    high_attn_reg_strength = config['high_attn_reg_strength']

    # DCNN
    lr_attn = config['lr_attn']
    inner_loop_epochs = config['inner_loop_epochs']
    recon_clusters_weighting = config['recon_clusters_weighting']
    noise_level = config['noise_level']

    print(
        f'lr_attn={lr_attn}, ' \
        f'inner_loop_epochs={inner_loop_epochs}, ' \
        f'recon_clusters_weighting={recon_clusters_weighting}, ' \
        f'noise_level={noise_level}'
    )
    
    lr_ = [lr]
    center_lr_multiplier_ = [center_lr_multiplier]
    # attn_lr_multiplier_ = [attn_lr_multiplier]
    asso_lr_multiplier_ = [asso_lr_multiplier]
    Phi_ = [Phi]
    specificity_ = [specificity]
    thr_ = [thr]
    beta_ = [beta]
    temp2_ = [temp2]
    high_attn_reg_strength_ = [high_attn_reg_strength]


    # TODO: temp - search lower high-attn lr hoping for better compression results.
    attn_lr_multiplier_ = [
        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ]

    
    lr_attn_ = [
        # lr_attn*0.1,
        # lr_attn*0.5,
        # lr_attn*0.75,
        lr_attn, 
        # lr_attn*1.25,
        # lr_attn*1.5,
    ]
    
    inner_loop_epochs_ = [
        # int(inner_loop_epochs*0.5),
        # int(inner_loop_epochs*0.75),
        inner_loop_epochs,
        # int(inner_loop_epochs*1.25),
        # int(inner_loop_epochs*1.5),

    ]
    
    recon_clusters_weighting_ = [
        # recon_clusters_weighting*0.75,
        recon_clusters_weighting,
        # recon_clusters_weighting*1.25,
        # recon_clusters_weighting*1.5,
        # recon_clusters_weighting*5,
    ]
    
    noise_level_ = [
        # noise_level*0.75,
        noise_level, 
        # noise_level*1.25,
        # noise_level*1.5,
    ]
        
    return lr_, center_lr_multiplier_, \
                attn_lr_multiplier_, asso_lr_multiplier_, \
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

    # clustering
    lr = config['lr']
    center_lr_multiplier = config['center_lr_multiplier']
    attn_lr_multiplier = config['attn_lr_multiplier']
    asso_lr_multiplier = config['asso_lr_multiplier']
    Phi = config['Phi']
    specificity = config['specificity']
    thr = config['thr']
    beta = config['beta']
    temp2 = config['temp2']
    high_attn_reg_strength = config['high_attn_reg_strength']
    
    lr_ = [lr]
    center_lr_multiplier_ = [center_lr_multiplier]
    attn_lr_multiplier_ = [attn_lr_multiplier]
    asso_lr_multiplier_ = [asso_lr_multiplier]
    Phi_ = [Phi]
    specificity_ = [specificity]
    thr_ = [thr]
    beta_ = [beta]
    temp2_ = [temp2]
    high_attn_reg_strength_ = [high_attn_reg_strength]
    
    # DCNN
    lr_attn_ = [0.00092, 0.0092]
    inner_loop_epochs_ = [10, 15, 20, 25, 30]
    recon_clusters_weighting_ = [1000, 10000, 100000, 1000000, 10000000]
    noise_level_ = [0.4, 0.5, 0.6]
    
    return lr_, center_lr_multiplier_, \
                attn_lr_multiplier_, asso_lr_multiplier_, \
                    Phi_, specificity_, thr_, beta_, temp2_, high_attn_reg_strength_, \
                        lr_attn_, inner_loop_epochs_, recon_clusters_weighting_, noise_level_


def per_subject_generate_candidate_configs(DCNN_config_version, ct, v, sub):
    """
    There are two scenarios where the configs for joint are created.
    1. We just finished searching for the best configs for the clustering module
        which is subject-specific. Once that search is done, we do not touch 
        it again. Then, we carry over the clustering params to the joint config,
        and we search for the best configs for the joint module (clustering params fixed).
        
        Notice, for this scenario, there was no joint config created at this point (actually
        there is the template for the joint config from the previous version but the values 
        do not matter).
    
    2. We finished 1. and want to further search for params. Here we have joint configs. We search 
        for both clustering and joint params on top of the best configs from 1.
        
    These 2 steps must run in this order.
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

    lr_, center_lr_multiplier_, \
        attn_lr_multiplier_, asso_lr_multiplier_, \
            Phi_, specificity_, thr_, beta_, temp2_, high_attn_reg_strength_, \
                lr_attn_, inner_loop_epochs_, recon_clusters_weighting_, noise_level_ = \
                    per_subject_hyperparams_ranges(
                        sub=sub, 
                        v=v,
                        DCNN_config_version=DCNN_config_version)
    
    # NOTE, only used for the first search.
    # lr_, attn_lr_multiplier_, \
    #     Phi_, specificity_, thr_, beta_, temp2_, high_attn_reg_strength_, \
    #         lr_attn_, inner_loop_epochs_, recon_clusters_weighting_, noise_level_ = \
    #             hyperparams_ranges(
    #                 sub=sub, 
    #                 clustering_config_version=clustering_config_version
    #             )
    
    # update all searchable entries in template
    for lr in lr_:        
        for center_lr_multiplier in center_lr_multiplier_:
            for attn_lr_multiplier in attn_lr_multiplier_:
                for asso_lr_multiplier in asso_lr_multiplier_:
                    for Phi in Phi_:
                        for specificity in specificity_:
                            for thr in thr_:    
                                for beta in beta_:
                                    for temp2 in temp2_:
                                        for high_attn_reg_strength in high_attn_reg_strength_:
                                            for lr_attn in lr_attn_:
                                                for inner_loop_epochs in inner_loop_epochs_:
                                                    for recon_clusters_weighting in recon_clusters_weighting_:
                                                        for noise_level in noise_level_:
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
                                                            template['high_attn_reg_strength'] = float(high_attn_reg_strength)
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
            DCNN_config_version='hyper89',
            ct=4016, 
            v='fit-human-entropy-fast-nocarryover', 
            sub=sub,
        )
        
    # [0, 450): Building on best subject-specific hypers from searching clustering independently
            # in hyper[0, 27024), 
            # we fix those hypers and search DCNN hypers in a subject-general manner.
            # lr_attn_ = [0.00092, 0.0092, 0.092]
            # inner_loop_epochs_ = [5, 10, 15, 20, 25, 30]
            # recon_clusters_weighting_ = [1000, 10000, 100000, 1000000, 10000000]
            # noise_level_ = [0.2, 0.3, 0.4, 0.5, 0.6]
        # best overall: hyper89
    
    # [450, 2637): Building on the best joint config from hyper[0, 450), 
            # we re-search hypers of clustering module in a subject-specific manner. 
            # At the same time, we keep search hypers of the DCNN in a subject-general manner 
            # because we care about overall results over individual fits to human lc.
        # no better.
            
    # [2637, 2787): Building on the best subject-specific config from searching clustering independently
            # in hyper[27024, 50352)
            # we fix those hypers and search DCNN hypers in a subject-general manner.
            # lr_attn_ = [0.00092, 0.0092]
            # inner_loop_epochs_ = [10, 15, 20, 25, 30]
            # recon_clusters_weighting_ = [1000, 10000, 100000, 1000000, 10000000]
            # noise_level_ = [0.4, 0.5, 0.6]
        # best overall: hyper2639

    # [2787, 3412): Building on the best joint config from hyper[2637, 2787),
            # we re-search hypers of DCNN in a subject-general manner,
            # because we want better overall results.
        # best overall: hyper2969    

    # [3412, 4012): Building on the best joint config from hyper[0, 450) which is hyper89,
            # we re-search hypers of DCNN in a subject-specific manner.
        # best overall: hyper3808
    
    # [4012, 4024): To get better compression results, we try to use lower lr for 
            # high-attn update.
            # attn_lr_multiplier_ = [
            #     0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
            # ]