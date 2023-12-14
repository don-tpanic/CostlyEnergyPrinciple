import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from utils import load_config


def recruit_cluster(center, signature, model):
    """
    Recruit a new cluster by centering on that item.
    
    The mask for cluster recruitment has the corresponding
    unit permanently set to 1.
    
    inputs:
    -------
        center: current trial item.
        signature: current trial item unique ID.
        model: current trial sustain model.
    """
    # Center on that item
    model.get_layer(f'd{signature}').set_weights(center)
    
    # A cluster is permanently recruited -- set mask to 1.
    mask_non_recruit_weights = model.get_layer('mask_non_recruit').get_weights()
    mask_non_recruit_weights[0][signature] = 1.
    model.get_layer('mask_non_recruit').set_weights(mask_non_recruit_weights)
    print(f'[Check] Recruited clusters mask = {mask_non_recruit_weights}')
    

def fit(model, x, y_true, signature, 
        loss_fn, optimizer, lr, lr_multipliers, 
        epoch, i,
        problem_type,
        run,
        config_version,
        global_steps):
    """
    A complete learning trial of the clustering model.
    """
    """
    A complete learning trial of the clustering model.
    """
    print(f'========================================================')
    print(f'[Check] epoch = {epoch}, trial {i}')
    print(f'[Check] x={x}, y_true={y_true}, sig={signature}')

    if epoch == 0 and i == 0:
        print(f'[Check] Load the first item.')
        print(f'--------------------------------------------------------')
    
        model(x, build_model=True)
        print(f'[Check] *** finished building ***')
        
        # recruit the first item by centering on it.
        recruit_cluster(center=x, signature=signature, model=model)

        # Evaluation to get classification loss.
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_value = loss_fn(y_true, y_pred)
            print(f'[Check] y_pred = {y_pred}')
        
        # Convert loss to proberror used in SUSTAIN.
        item_proberror = 1. - tf.reduce_max(y_pred * y_true)
        print(f'[Check] item_proberror = {item_proberror}')
        
    else:
        print(f'[Check] Load non-first item.')
        print(f'--------------------------------------------------------')
        
        # First eval loss using existing.
        with tf.GradientTape() as tape:
            y_pred, totalSupport = model(x, y_true=y_true, training=True)
            loss_value = loss_fn(y_true, y_pred)
        
        item_proberror = 1. - tf.reduce_max(y_pred * y_true)
        print(f'[Check] item_proberror = {item_proberror}')

        # Apply the unsupervised thresholding rule.
        config = load_config(config_version)
        unsup_rule = config['unsup_rule']
        thr = config['thr']
        print(f'[Check] totalSupport={totalSupport} vs thr={thr}')
        
        # Successful recruit if lower than thr
        if totalSupport < thr:
            print(f'[Check] lower than thr, recruit.')
            recruit_cluster(center=x, signature=signature, model=model)
            
            # Evaluate loss given new recruit.
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss_value = loss_fn(y_true, y_pred)
            
        # Unsuccessful recruit if higher than thr
        else:
            print(f'[Check] exceeding thr, do not recruit.')
        
    # Update trainable parameters.
    model = update_params(
        model, 
        tape, 
        loss_value, 
        optimizer,
        lr_multipliers
    )
    return model, item_proberror, global_steps


def update_params(
        model, 
        tape, 
        loss_value, 
        optimizer,
        lr_multipliers
        ):
    """
    Update trainable params in the model.
    """
    print(f'[Check] update ALL parameters')

    # len(grads) == 10 (8 + 1 + 1)
    grads = tape.gradient(loss_value, model.trainable_weights)
    assert len(grads) == 10, f'[Check] len(grads) = {len(grads)}'
    
    # adjust lr for each component.
    for i in range(len(grads)):
        if i < 8:
            # centers
            grads[i] *= lr_multipliers[0]
        elif i == 8:
            # attn
            grads[i] *= lr_multipliers[1]
        else:
            # assoc
            grads[i] *= lr_multipliers[2]
    
    print(f'---------------------- all grads -----------------------')
    for i in range(len(grads[:8])):
        print(f'[Check] center {i} grads {grads[i]}')
    print(f'[Check] attn grads {grads[8]}')
    print(f'[Check] assco grads \n{grads[9]}')
    print(f'--------------------------------------------------------')


    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    
    print('--------------------- updated params -------------------')
    for i in range(8):
        print(f'[Check] center {i}', model.get_layer(f'd{i}').get_weights()[0])
    print('[Check] attn', model.get_layer('dimensionwise_attn_layer').get_weights()[0])
    print('[Check] assoc', model.get_layer(f'classification').get_weights()[0])
    print(f'--------------------------------------------------------')
    return model