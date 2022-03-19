import os
import yaml


default_dict = {
    'config_version': 'config_v4_naive-withNoise',

    # --- training ---
    'num_runs': 50,
    'num_blocks': 32,
    'random_seed': 999,

    # --- DCNN attn layer ---
    'attn_initializer': 'ones-withNoise',
    'noise_distribution': 'uniform',
    'noise_level': 0.5,
    'low_attn_constraint': 'nonneg',
    'attn_regularizer': 'l1',
    'reg_strength': 0.001,
    'attn_positions': 'block4_pool',
    'lr_attn': 0.00092,
    'recon_level': 'cluster',
    'inner_loop_epochs': 5,
    'recon_clusters_weighting': 1000,

    # --- clustering model ---
    'Phi': 10.0,
    'actv_func': 'softmax',
    'asso_lr_multiplier': 1.0,
    'high_attn_constraint': 'sumtoone',
    'attn_lr_multiplier': 1.0,
    'beta': 3.0,
    'center_lr_multiplier': 1.0,
    'from_logits': True,
    'lr': 0.1,
    'num_clusters': 8,
    'q': 1,
    'r': 2,
    'specificity': 0.25,
    'temp1': 'equivalent',
    'temp2': 0.012,
    'thr': -0.4,
    'trainable_specificity': False,
    'unsup_rule': 'threshold',

    # stimulus set and finetuned DCNN.
    'dcnn_config_version': 't1.vgg16.block4_pool.None.run1'
}

##############################################
lr_attn_ = [0.00092, 0.0092, 0.092]
recon_clusters_weighting_ = [500, 1500, 2000, 2500, 3000, 4000, 5000, 8000, 12000]
# recon_clusters_weighting_ = [10, 100, 1000, 10000, 100000, 1000000]
##############################################
v = 20   # v to resume
for recon_clusters_weighting in recon_clusters_weighting_:
    for lr_attn in lr_attn_:
        config_version = f'config_v{v}_naive-withNoise'
        default_dict['config_version'] = config_version
        default_dict['recon_clusters_weighting'] = recon_clusters_weighting
        # default_dict['inner_loop_epochs'] = inner_loop_epochs
        default_dict['lr_attn'] = lr_attn

        filepath = os.path.join(f'configs', f'{config_version}.yaml')
        with open(filepath, 'w') as yaml_file:
            yaml.dump(default_dict, yaml_file, default_flow_style=False)
        v += 1
