import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from utils import load_config


def dict_int2binary():
    int2binary = {
        0: [0, 0, 0],
        1: [0, 0, 1],
        2: [0, 1, 0],
        3: [0, 1, 1],
        4: [1, 0, 0],
        5: [1, 0, 1],
        6: [1, 1, 0],
        7: [1, 1, 1]
    }
    return int2binary


def dict_layer2attn_size(model_name='vgg16'):
    if model_name == 'vgg16':
        layer2attn_size = {
            'block1_pool': 64,
            'block2_pool': 128,
            'block3_pool': 256,
            'block4_pool': 512
        }
        
    return layer2attn_size
    

def type2labels(problem_type):
    """
    This is a mapping from
    1 out of 6 problems to its corresponding one-hot labelling.
    E.g., If problem_type=1, the first 4 data-points have label [1,0],
        and the last 4 data-points have label [0,1]

    Note, signature is unique to each stimulus and does not change.
    It is the label of each stimulus changes based on problem type.

    return:
    -------
        A problem_type dictated dict mapping
        whose element looks like {signature: [label]}
    """
    if problem_type == 1:
        mapping = {
            0: [1, 0],
            1: [1, 0],
            2: [1, 0],
            3: [1, 0],
            4: [0, 1],
            5: [0, 1],
            6: [0, 1],
            7: [0, 1]
        }
    elif problem_type == 2:
        mapping = {
            0: [1, 0],
            1: [1, 0],
            2: [0, 1],
            3: [0, 1],
            4: [0, 1],
            5: [0, 1],
            6: [1, 0],
            7: [1, 0]
        }
    elif problem_type == 3:
        mapping = {
            0: [0, 1],
            1: [0, 1],
            2: [0, 1],
            3: [1, 0],
            4: [1, 0],
            5: [0, 1],
            6: [1, 0],
            7: [1, 0]
        }
    elif problem_type == 4:
        mapping = {
            0: [0, 1],
            1: [0, 1],
            2: [0, 1],
            3: [1, 0],
            4: [0, 1],
            5: [1, 0],
            6: [1, 0],
            7: [1, 0]
        }
    elif problem_type == 5:
        mapping = {
            0: [0, 1],
            1: [0, 1],
            2: [0, 1],
            3: [1, 0],
            4: [1, 0],
            5: [1, 0],
            6: [1, 0],
            7: [0, 1]
        }
    elif problem_type == 6:
        mapping = {
            0: [0, 1],
            1: [1, 0],
            2: [1, 0],
            3: [0, 1],
            4: [1, 0],
            5: [0, 1],
            6: [0, 1],
            7: [1, 0]
        }
    return mapping


def data_loader_V2(
        attn_config_version,
        dcnn_config_version,
        preprocess_func,
        problem_type,
        color_mode='rgb',
        interpolation='nearest',
        target_size=(224, 224),
        data_format='channels_last',
        random_seed=999):
    """
    V2: supports multi_attn, i.e.
    it combines multiple fake inputs.

    Load the original 8/16 stimuli 
    for evaluating the trained model.

    Note, 
    1. since we evaluate the trained 
    attention models, we need to add
    second or more fake inputs (ones)
    when there are multiple attention 
    positions.

    2. As we train the joint model end-to-end,
    data_loader takes extra argument `problem_type`
    which dictates the Y targets.

    return:
    -------
        dataset [X, Y, signature], in which, 
        X[i] = [(1, 224, 224, 3), (1, 512)] if single attn.

        Or if multi attn:
        X[i] = [(1, 224, 224, 3), (1, 64), (1, 256), (1, 512)]
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version)
    dcnn_config = load_config(
        component='finetune', 
        config_version=dcnn_config_version)
    attn_positions = attn_config['attn_positions'].split(',')
    layer2attn_size = dict_layer2attn_size(
        model_name=dcnn_config['model_name'])
    stimulus_set = dcnn_config['stimulus_set']
    data_dir = f'dataset/task{stimulus_set}'

    # load n preprocess images
    all_image_fnames = sorted(os.listdir(data_dir))
    all_image_fnames = [i for i in all_image_fnames \
        if not i.startswith('.')]
    num_images = len(all_image_fnames)
    if data_format == 'channels_last':
        image_shape = target_size + (3,)
        
    dataset = []
    dict_int2binary_counterbalanced = {}
    for i in range(num_images):
        # for each image, 
        # load -> preprocess -> add fake inputs 
        # -> get label (w counterbalancing)
        fname = all_image_fnames[i]
        img_idx = int(fname.split('.')[0])
        fpath = f'{data_dir}/{fname}'
        img = image.load_img(
            fpath,
            color_mode=color_mode,
            target_size=target_size,
            interpolation=interpolation)
        x = image.img_to_array(
            img, 
            data_format=data_format)
        # Pillow images should be closed after `load_img`,
        # but not PIL images.
        if hasattr(img, 'close'):
            img.close()
        if preprocess_func:
            x = preprocess_func(x)

        # ------ /counterbalancing ------
        num_dims = np.int(np.log2(num_images))
        dim_shapes = (2, ) * num_dims
        # initialize signature cube which is the same as `img_cube`
        signature_cube = np.arange(num_images).reshape(dim_shapes)
        
        # sample ways of rotating
        np.random.seed(random_seed)
        k = np.random.choice([1, 2, 3], size=1)
        rot_dims = np.random.choice(
            np.arange(num_dims), size=2, replace=False
        )

        signature_cube = np.rot90(
            signature_cube, k=k, axes=rot_dims)
        print(f'k={k}, rot_dims={rot_dims}')
        # print(f'signature_cube = {signature_cube}')

        # get updated signature using img_idx's unrotated coordinates
        img_idx2binary = dict_int2binary()
        signature = signature_cube[
            img_idx2binary[img_idx][0],
            img_idx2binary[img_idx][1],
            img_idx2binary[img_idx][2]
        ]
        print(f'img_idx={img_idx}, signature={signature}')

        # collect to save in `main.py`
        counter_balancing = {}
        counter_balancing['k'] = k
        counter_balancing['rot_dims'] = rot_dims
        # ------ counterbalancing/ ------

        # preserve dim=0 (i.e. batch_size dim)
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(
            type2labels(problem_type)[signature],
            axis=0)

        # add fake inputs.
        inputs = [x]
        for attn_position in attn_positions:
            attn_size = layer2attn_size[attn_position]
            fake_input = np.ones((1, attn_size))
            inputs.extend([fake_input])

        dataset.append([inputs, y, signature])
        dict_int2binary_counterbalanced[signature] = img_idx2binary[img_idx]
        
    return np.array(dataset, dtype=object), counter_balancing, \
        dict_int2binary_counterbalanced


def load_X_only(dataset, 
                attn_config_version,
                dcnn_config_version):
    """
    Given a dataset, extract and return 
    only the X part. This is for evaluating 
    model activations when Y labels are not needed.
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    dcnn_config = load_config(
        component='finetune',
        config_version=dcnn_config_version
    )
    dcnn_model_name = dcnn_config['model_name']
    layer2attn_size = dict_layer2attn_size(
        model_name=dcnn_model_name
    )
    
    # image -> [(N, 224, 224, 3)]
    image_batch = np.empty( (len(dataset), ) + dataset[0][0][0].shape[1:])
    for i in range(len(dataset)):
        dp = dataset[i]
        image = dp[0][0]
        ones = dp[0][1]
        y = dp[1]
        image_batch[i] = image

    batch_x = [image_batch]

    # fake inputs -> [(1, 64), (1, 256), (1, 512)]
    attn_positions = attn_config['attn_positions'].split(',')
    for attn_position in attn_positions:
        attn_size = layer2attn_size[attn_position]
        fake_input = np.ones((1, attn_size))

        # batch_x finally is [(image batch), (fake input 1), (fake input 2), ...]
        batch_x.extend([fake_input])
    
    return batch_x
        

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    # data_loader_V2(
    #     dcnn_config_version='t1.block4_pool.None.run1', 
    #     problem_type=1,
    #     preprocess_func=None
    # )

    data_loader_V2(
        attn_config_version='attn_v3b_cb_multi_test',
        dcnn_config_version='t1.block4_pool.None.run1', 
        problem_type=1,
        preprocess_func=None
    )

    