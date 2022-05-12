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
        print(f'sub{sub}, %={percent_zero}')
        percent_zero_all.append(percent_zero)
    
    print(f'mean={np.mean(percent_zero_all)}')


def within_sub_all_types(sub, repetition):
    attn_position = 'block4_pool'
    
    if int(sub) % 2 == 0:
        problem_types = [6, 1, 2]
    else:
        problem_types = [6, 2, 1]
        
    for problem_type in problem_types:
        attn_weights = np.load(
            f'results/best_config_sub{sub}_fit-human/' \
            f'attn_weights_type{problem_type}_sub{sub}_cluster_rp{repetition}.npy',
            allow_pickle=True
        ).ravel()[0][attn_position]
        
        percent_zero = (
            attn_weights.size - len(np.nonzero(attn_weights)[0])
        ) / attn_weights.size
        print(f'sub{sub}, type={problem_type}, %={percent_zero}')

    
    
if __name__ == '__main__':
    # print('---------- Type 6 ----------')
    # within_type_all_subs(problem_type=6, repetition=15)
    # print('---------- Type 1 ----------')
    # within_type_all_subs(problem_type=1, repetition=15)
    # print('---------- Type 2 ----------')
    # within_type_all_subs(problem_type=2, repetition=15)
    
    num_subs = 23
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    for sub in subs:
        print('-------------------------')
        within_sub_all_types(sub=sub, repetition=15)
    
    
    