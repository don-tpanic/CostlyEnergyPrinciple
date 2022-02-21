import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from models import JointModel

"""
We move the seeded shuffling operation out of 
`main.py` so that we no longer need the seeding 
instead we will save all shuffled indices to a 
file and read them when needed.

This way, 
1. We can rerun any single shuffling scenario independently;
2. We no longer need a seed.
"""

def generate_shuffled_indices(
        num_runs, 
        num_blocks=32, 
        dataset_size=8, 
        random_seed=999,
        attn_config_version='v1_naive-withNoise',
        dcnn_config_version='t1.vgg16.block4_pool.None.run1'):
    
    np.random.seed(random_seed)
    run2indices = np.empty((num_runs, num_blocks, dataset_size), dtype=np.int64)
    
    for run in range(num_runs):
        # model = JointModel(
        #     attn_config_version=attn_config_version,
        #     dcnn_config_version=dcnn_config_version, 
        # )
        # del model
        for epoch in range(num_blocks):
            shuffled_indices = np.random.choice(
                np.arange(dataset_size, dtype=int), 
                size=dataset_size, 
                replace=False)
            run2indices[run, epoch, :] = shuffled_indices
            print(f'[Check] shuffled_indices={shuffled_indices}')

    np.save(f'run2indices_num_runs={num_runs}.npy', run2indices)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # generate_shuffled_indices(num_runs=1)
    # generate_shuffled_indices(num_runs=10)
    generate_shuffled_indices(num_runs=500)
    
