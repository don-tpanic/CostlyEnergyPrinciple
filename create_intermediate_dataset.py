import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

from utils import load_config
from models import presave_dcnn


def load_dcnn_intermediate_model(
        attn_config_version,
        dcnn_config_version
    ):
    path_model = f'dcnn_models/{dcnn_config_version}'
    model_dcnn = tf.keras.models.load_model(path_model)
    
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    
    dcnn_config = load_config(
        component='finetune', 
        config_version=dcnn_config_version)
    
    attn_position_begin = attn_config['attn_positions'].split(',')[0]
    print(f'attn_position_begin = {attn_position_begin}')
    layer_reprs = model_dcnn.get_layer(attn_position_begin).output

    intermediate_model = Model(inputs=model_dcnn.input, outputs=layer_reprs)
    intermediate_model.summary()
    return intermediate_model
    

def compute_intermediate_dataset(
        attn_config_version,
        dcnn_config_version,
        color_mode='rgb',
        interpolation='nearest',
        target_size=(224, 224),
        data_format='channels_last'
    ):
    dcnn_config = load_config(
        component='finetune', 
        config_version=dcnn_config_version)
    model = load_dcnn_intermediate_model(
        attn_config_version=attn_config_version,
        dcnn_config_version=dcnn_config_version,
    )
    dcnn_base = dcnn_config['model_name']
    stimulus_set = dcnn_config['stimulus_set']
    data_dir = f'dataset/task{stimulus_set}'
    
    if dcnn_base == 'vgg16':
        preprocess_func = tf.keras.applications.vgg16.preprocess_input
        
    for fname in range(8):
        img_fpath = f'{data_dir}/{fname}.jpg'
        img = image.load_img(
            img_fpath,
            color_mode=color_mode,
            target_size=target_size,
            interpolation=interpolation
        )
        x = image.img_to_array(
            img,
            data_format=data_format
        )
        # Pillow images should be closed after `load_img`,
        # but not PIL images.
        if hasattr(img, 'close'):
            img.close()
        if preprocess_func:
            x = preprocess_func(x)
        
        x = np.expand_dims(x, axis=0)
        intermediate_input = model.predict(x)
        np.save(f'{data_dir}/{fname}.npy', intermediate_input)


if __name__ == '__main__':
    compute_intermediate_dataset(
        attn_config_version='best_config_sub02_fit-human-entropy',
        dcnn_config_version='t1.vgg16.block4_pool.None.run1'
    )