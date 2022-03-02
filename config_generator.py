import os
import yaml


default_dict = {
    'config_version': None,
    'dcnn_config_version': None,
    'dcnn_actv_func': 'sigmoid',
    'low_attn_initializer': 'ones',
    'low_attn_positions': None,
    'low_attn_regularizer': 'l1',
    'low_attn_constraint': 'nonneg',
    'layer': None,
    'lr_finetune': 0.003,
    'dcnn_base': None,
    'noise_distribution': None,
    'noise_level': None,
    'finetune_run': None,
    'stimulus_set': 1,
    'train': 'finetune-with-lowAttn',
    'reg_strength': None,
    'num_runs': 50,
    'lr_low_attn': None,
    'recon_level': 'cluster',
    'inner_loop_epochs': 5,
    'recon_clusters_weighting': None,
    'num_blocks': 32,
    'random_seed': 999,
    'Phi': 10.0,
    'clus_actv_func': 'softmax',
    'asso_lr_multiplier': 1.0,
    'high_attn_constraint': 'sumtoone',
    'attn_lr_multiplier': 1.0,
    'beta': 3.0,
    'center_lr_multiplier': 1.0,
    'from_logits': True,
    'lr_clus': 0.1,
    'num_clusters': 8,
    'q': 1,
    'r': 2,
    'specificity': 0.25,
    'temp1': 'equivalent',
    'temp2': 0.012,
    'thr': -0.4,
    'trainable_specificity': False,
    'unsup_rule': 'threshold',
}

##############################################
begin_run = 8
end_run = 8
finetune_runs = range(begin_run, end_run+1)
# finetune_runs = [1, 5, 12, 17, 19, 20, 26]
dcnn_base = 'vgg16'
low_attention_positions = 'block4_pool'
layer = 'block4_pool'
reg_strength_ = [0.001]
lr_low_attn_ = [0.00092]
inner_loop_epochs_ = [5]
recon_clusters_weighting_ = [1, 10, 100, 1000]
##############################################

default_dict['dcnn_base'] = dcnn_base
default_dict['low_attn_positions'] = low_attention_positions
default_dict['layer'] = layer

for run in finetune_runs:

    dcnn_config_version = f't1.{dcnn_base}.{layer}.None.run{run}-with-lowAttn'
    default_dict['dcnn_config_version'] = f'config_{dcnn_config_version}'

    v = 1   # v to resume
    for recon_clusters_weighting in recon_clusters_weighting_:

        for inner_loop_epochs in inner_loop_epochs_:

            for lr_low_attn in lr_low_attn_:

                config_version = f'config_v{v}_naive-withNoise-{dcnn_config_version}'
                default_dict['finetune_run'] = run
                default_dict['config_version'] = config_version
                default_dict['recon_clusters_weighting'] = recon_clusters_weighting
                default_dict['inner_loop_epochs'] = inner_loop_epochs
                default_dict['lr_low_attn'] = lr_low_attn

                filepath = os.path.join(f'configs', f'{config_version}.yaml')
                with open(filepath, 'w') as yaml_file:
                    yaml.dump(default_dict, yaml_file, default_flow_style=False)
                
                v += 1


    




