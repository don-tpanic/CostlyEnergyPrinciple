import numpy as np 
import scipy.stats as stats
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances


def percent_of_nonzero(x):
    return len(np.nonzero(x.flatten())[0]) / len(x.flatten())


def per_channel_percent_of_nonzero(x):
    
    x = x.reshape((8, 14, 14, 512))
    
    num_channels = x.shape[-1]
    all_channels = []
    for i in range(num_channels):
        all_channels.append(
            percent_of_nonzero(x[:, :, :, i])
        )
        
    all_channels = np.round(all_channels, 3)
    return all_channels, np.sum(all_channels==0)/len(all_channels)


def compute_RSA(layer1_reprs, layer2_reprs):
    RDM_1 = pairwise_distances(layer1_reprs, metric='correlation')
    print(f'RDM_1, \n{RDM_1}')
    
    RDM_2= pairwise_distances(layer2_reprs, metric='correlation')
    print(f'\nRDM_2, \n{RDM_2}')
    
    RDM_1_triu = RDM_1[np.triu_indices(RDM_1.shape[0])]
    RDM_2_triu = RDM_2[np.triu_indices(RDM_2.shape[0])]
    rho, _ = stats.spearmanr(RDM_1_triu, RDM_2_triu)
    print(f'rho={rho}')
    return rho


def manual_attn_weights(LOC_no_attn_actv):
    np.random.seed(999)
    # attn_weights = np.random.random((1, 1, 1, 512))
    attn_weights = np.zeros((1, 1, 1, 512))
    attn_weights[:, :, :, 0] = 1
    
    LOC_no_attn_actv = LOC_no_attn_actv.reshape((8, 14, 14, 512))
    post_attn_actv = (attn_weights * LOC_no_attn_actv).reshape((8, -1))
    return post_attn_actv
    

def how_similar_are_random_pre_n_post(percent_zero_actv, percent_zero_attn):
    """
    Randomly create pre-attn activation (with percent_zero)
    to immitate DCNN activation.
    
    Attn weights are randomly created too.
    """
    np.random.seed(999)
    
    # randomly masking pre-attn actv
    pre_attn_actv = np.random.random((8, 14, 14, 512))
    mask = np.ones(pre_attn_actv.size)
    random_indices = np.random.choice(
        np.arange(len(mask)),
        size=int(len(mask)*percent_zero_actv),
        replace=False)
    mask[random_indices] = 0
    mask = mask.reshape((8, 14, 14, 512))
    pre_attn_actv = pre_attn_actv * mask
    
    # randomly masking attn weights
    attn_weights = np.random.random((1, 1, 1, 512))
    mask = np.ones(attn_weights.size)
    random_indices = np.random.choice(
        np.arange(len(mask)),
        size=int(len(mask)*percent_zero_attn),
        replace=False)
    mask[random_indices] = 0
    mask.reshape((1, 1, 1, 512))
    attn_weights = attn_weights * mask
    
    # element-wise multiply and compute RSA
    post_attn_actv = pre_attn_actv * attn_weights
    compute_RSA(
        pre_attn_actv.reshape((8, -1)), 
        post_attn_actv.reshape((8, -1)))


if __name__ == '__main__':
    problem_type = 1
    rp = 0
    sub = '08'
    LOC_actv = np.load(f'LOC_sub{sub}_type{problem_type}_rp{rp}.npy')
    LOC_no_attn_actv = np.load(f'LOC_no_attn_sub{sub}_type{problem_type}_rp{rp}.npy')
    b1p_no_attn_actv = np.load(f'block1_pool_no_attn_sub{sub}_type{problem_type}_rp{rp}.npy')

    print(f'sub={sub}')
    # print('-'*37)
    # print('*** percent of nonzero units ***')
    # print(f'post-attn: {percent_of_nonzero(LOC_actv):.5f}')
    # print(f'pre-attn: {percent_of_nonzero(LOC_no_attn_actv):.5f}')
    # print(f'block1_pool: {percent_of_nonzero(b1p_no_attn_actv):.5f}')
    # print('-'*37)
    
    # print('\n\n*** percent of nonzero units per channel')
    # print(f'post-attn: {per_channel_percent_of_nonzero(LOC_actv)[0]}')
    # print(f'pre-attn: {per_channel_percent_of_nonzero(LOC_no_attn_actv)[0]}')
    
    # print('*** percent of complete zero channel ***')
    # print(f'post-attn: {per_channel_percent_of_nonzero(LOC_actv)[1]}')
    # print(f'pre-attn: {per_channel_percent_of_nonzero(LOC_no_attn_actv)[1]}')
    # print('-'*37)
    
    # print('*** between layers RSA ***')
    # print(f'post-attn vs pre-attn: {compute_RSA(LOC_actv, LOC_no_attn_actv):.5f}')
    # print(f'pre-attn vs block1_pool: {compute_RSA(LOC_no_attn_actv, b1p_no_attn_actv):.5f}')
    # print('-'*37)
    
    # print('*** RSA using random attn-weights  ***')
    # post_attn_actv = manual_attn_weights(LOC_no_attn_actv)
    # print(f'post-attn vs pre-attn: {compute_RSA(post_attn_actv, LOC_no_attn_actv):.5f}')
        
    how_similar_are_random_pre_n_post(
        percent_zero_actv=0.8, 
        percent_zero_attn=0
    )