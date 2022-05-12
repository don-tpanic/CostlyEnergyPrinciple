import numpy as np


def within_type_all_subs(problem_type, repetition):
    attn_position = 'block4_pool'
    
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    
    percent_zero_all = []
    for sub in subs:
        attn_weights = np.load(
            f'results/best_config_sub{sub}_fit-human/' \
            f'attn_weights_type{problem_type}_sub{sub}_cluster_rp{repetition}.npy',
            allow_pickle=True
        ).ravel()[0][attn_position]
        
        percent_zero = (
            attn_weights.size - len(np.nonzero(attn_weights)[0])
        ) / attn_weights.size
        print(f'sub{sub}, percent zero={percent_zero}')
        percent_zero_all.append(percent_zero)
    
    print(f'mean={np.mean(percent_zero_all)}')
    
    
if __name__ == '__main__':
    print('---------- Type 6 ----------')
    within_type_all_subs(problem_type=6, repetition=15)
    print('---------- Type 1 ----------')
    within_type_all_subs(problem_type=1, repetition=15)
    print('---------- Type 2 ----------')
    within_type_all_subs(problem_type=2, repetition=15)
    