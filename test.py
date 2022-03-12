import numpy as np 
import matplotlib.pyplot as plt


def attn_analyze(mode):
    if mode == 'finetune':
        # finetune attn
        finetune_path = f'finetune/results/{dcnn_base}/config_{dcnn_config_version}/trained_weights'
        layer_attn_weights = np.load(f'{finetune_path}/attn_weights.npy', allow_pickle=True)
        layer_attn_weights = layer_attn_weights.ravel()[0][attn_position]
    
    elif mode == 'joint-lowAttn':
        # joint attn with lowAttn
        config_version = f'v4_naive-withNoise-{dcnn_config_version}'
        joint_path = f'results/{config_version}/attn_weights_type1_run0_cluster.npy'
        layer_attn_weights = np.load(f'{joint_path}', allow_pickle=True)
        layer_attn_weights = layer_attn_weights.ravel()[0][attn_position]
    
    elif mode == 'joint':
        # joint attn 
        config_version = f'v4_naive-withNoise'
        joint_path = f'results/{config_version}/attn_weights_type1_run0_cluster.npy'
        layer_attn_weights = np.load(f'{joint_path}', allow_pickle=True)
        layer_attn_weights = layer_attn_weights.ravel()[0][attn_position]

    nonzero_percentage = len(np.nonzero(layer_attn_weights)[0]) / len(layer_attn_weights)

    print(f'---------------------------------------------------------------------')
    print(f'nonzero_percentage = {nonzero_percentage}')
    print(f'mean = {np.mean(layer_attn_weights)}')
    print(f'std = {np.std(layer_attn_weights)}')
    print(f'max = {np.max(layer_attn_weights)}, min = {np.min(layer_attn_weights)}')
    print(f'sum = {np.sum(layer_attn_weights)}')

    print('small than zero = ', len(layer_attn_weights[layer_attn_weights < 0]))

    fig, ax = plt.subplots()
    plt.hist(layer_attn_weights)
    plt.savefig(f'attn_hist_{mode}.png')


if __name__ == '__main__':
    dcnn_base = 'vgg16'
    attn_position = 'block4_pool'
    dcnn_config_version = 't1.vgg16.block4_pool.None.run8-with-lowAttn'

    attn_analyze(mode='finetune')
    attn_analyze(mode='joint-lowAttn')
    attn_analyze(mode='joint')

