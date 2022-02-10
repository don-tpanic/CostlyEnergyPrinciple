import os
import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.ops import math_ops
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from utils import load_config
from losses import binary_crossentropy
from data import load_X_only, dict_int2binary
from clustering.train import recruit_cluster, update_params


def set_trainable(objective, attn_positions, num_clusters, model):
    """
    Set some layers to be trainable/not trainable based 
    on objective.
    
    If objective == 'low': only low attn trainable
    If objective == 'high': only high cluster trainable
    """
    if objective == 'low':
        low_trainable = True
        high_trainable = False
        
    elif objective == 'high':
        low_trainable = False
        high_trainable = True
    
    for attn_position in attn_positions:
        model.get_layer(
            f'dcnn_model').get_layer(
                f'attn_factory_{attn_position}').trainable = low_trainable
                
    for i in range(num_clusters):
        model.get_layer(f'd{i}').trainable = high_trainable
    model.get_layer('dimensionwise_attn_layer').trainable = high_trainable
    model.get_layer('classification').trainable = high_trainable


def fit(joint_model, 
        attn_positions,
        num_clusters,
        dataset, 
        x, 
        y_true, 
        signature, 
        loss_fn_clus, 
        loss_fn_attn, 
        optimizer_clus, 
        optimizer_attn, 
        lr_multipliers, 
        epoch, 
        i, 
        run,
        attn_config_version, 
        dcnn_config_version,
        inner_loop_epochs,
        global_steps,
        problem_type):
    """
    A single train step given a stimulus.
    """
    signature2coding = dict_int2binary()
    print(f'[Check] Incoming stimulus: ', signature2coding[signature])
        
    # Go thru the `latest` joint_model to get binary output,
    x_binary, _, _, _ = joint_model(x)
        
    set_trainable(
        objective='high', 
        attn_positions=attn_positions,
        num_clusters=num_clusters,
        model=joint_model
    )
                
    if epoch == 0 and i == 0:
        print(f'[Check] Load the first item.')
        print(f'--------------------------------------------------------')
        
        # recruit the first item by centering on it.
        recruit_cluster(
            center=x_binary, 
            signature=signature, 
            model=joint_model
        )

        # Evaluation to get classification loss.
        with tf.GradientTape() as tape:
            _, _, y_pred, _ = joint_model(x, training=True)
            loss_value = loss_fn_clus(y_true, y_pred)
            print(f'[Check] y_pred = {y_pred}')
        
        # Convert loss to proberror used in SUSTAIN.
        item_proberror = 1. - tf.reduce_max(y_pred * y_true)
        print(f'[Check] item_proberror = {item_proberror}')
    
    else:
        print(f'[Check] Load non-first item.')
        print(f'--------------------------------------------------------')
        
        # First eval loss using existing.
        with tf.GradientTape() as tape:
            _, _, y_pred, totalSupport = joint_model(x, y_true=y_true, training=True)
            loss_value = loss_fn_clus(y_true, y_pred)
        
        item_proberror = 1. - tf.reduce_max(y_pred * y_true)
        print(f'[Check] item_proberror = {item_proberror}')

        # Apply the unsupervised thresholding rule.
        attn_config = load_config(
            component=None, 
            config_version=attn_config_version
        )
        unsup_rule = attn_config['unsup_rule']
        thr = attn_config['thr']
        print(f'[Check] totalSupport={totalSupport} vs thr={thr}')
        
        # Successful recruit if lower than thr
        if totalSupport < thr:
            print(f'[Check] lower than thr, recruit.')
            recruit_cluster(
                center=x_binary, 
                signature=signature, 
                model=joint_model
            )
            
            # Evaluate loss given new recruit.
            with tf.GradientTape() as tape:
                _, _, y_pred, _ = joint_model(x, training=True)
                loss_value = loss_fn_clus(y_true, y_pred)
            
        # Unsuccessful recruit if higher than thr
        else:
            print(f'[Check] exceeding thr, do not recruit.')
                
    # Update trainable parameters.
    joint_model = update_params(
        model=joint_model, 
        tape=tape, 
        loss_value=loss_value, 
        optimizer=optimizer_clus,
        lr_multipliers=lr_multipliers
    )

    # track cluster_model's attn & centers 
    # after every trial update.
    alpha_collector = joint_model.get_layer('dimensionwise_attn_layer').get_weights()[0]
    num_centers = len(joint_model.get_layer('mask_non_recruit').get_weights()[0])
    center_collector = []
    for d in range(num_centers):
        centers = joint_model.get_layer(f'd{d}').get_weights()[0]
        center_collector.extend(centers)
        print(f'[Check] center {d} after trial update {centers}')

    ################################
    # inner-loop low_attn learning.
    # 1. call after every trial (exc epoch 0)
    # 2. batch update
    ################################    
    print('[Check] *** Beginning inner-loop ***')
    if epoch > 0:
        joint_model, attn_weights, \
        recon_loss_collector, recon_loss_ideal_collector, \
        reg_loss_collector, percent_zero_attn_collector, global_steps, \
        optimizer_attn = learn_low_attn(
            joint_model=joint_model,
            attn_positions=attn_positions,
            num_clusters=num_clusters,
            dataset=dataset,
            attn_config_version=attn_config_version,
            dcnn_config_version=dcnn_config_version,
            loss_fn_attn=loss_fn_attn,
            optimizer_attn=optimizer_attn,
            inner_loop_epochs=inner_loop_epochs,
            global_steps=global_steps,
            run=run,
            item_proberror=item_proberror,
            problem_type=problem_type)
        return joint_model, attn_weights, item_proberror, \
            recon_loss_collector, recon_loss_ideal_collector, \
            reg_loss_collector, percent_zero_attn_collector, \
            alpha_collector, center_collector, global_steps, \
            optimizer_clus, optimizer_attn
    
    # Only when epoch=0
    return joint_model, [], item_proberror, \
        [], [], [], [], [], [], global_steps, optimizer_clus, optimizer_attn


def learn_low_attn(
        joint_model,
        attn_positions,
        num_clusters,
        dataset,
        attn_config_version, 
        dcnn_config_version,
        loss_fn_attn,
        optimizer_attn, 
        inner_loop_epochs,
        global_steps,
        run,
        item_proberror,
        problem_type):
    """
    Learning routine for low-level attn.
    This learning happens after the 
    clustering model has been updated (one trial)
    
    The training objective is to minimise
    attn weights L1 while keep the cluster actv 
    the same.
    """
    set_trainable(
        objective='low',
        attn_positions=attn_positions,
        num_clusters=num_clusters,
        model=joint_model
    )
        
    # a batch of raw images(+ fake ones)
    batch_x = load_X_only(
        dataset=dataset, 
        attn_config_version=attn_config_version,
        dcnn_config_version=dcnn_config_version
    )
    
    # true cluster actv
    _, batch_y_true, _, _ = joint_model(batch_x)
    print(f'[Check] batch_y_true.shape={batch_y_true.shape}')
    print(f'[Check] batch_y_true={batch_y_true}')
    
    # Save trial-level cluster targets
    fname = f'results/{attn_config_version}/cluster_targets_{problem_type}_{global_steps}_{run}.npy'
    np.save(fname, batch_y_true)
    
    recon_loss_collector = []           # recon loss at cluster level (learning)
    recon_loss_ideal_collector = []     # recon loss at binary level  (tracking)
    reg_loss_collector = []
    percent_zero_attn_collector = []
    
    for i in range(inner_loop_epochs):
        print(f' \n------- inner loop epoch = {i} -------')
        print(f'global step = {global_steps}')
        
        with tf.GradientTape() as tape:
            # use the separate trainable dcnn 
            # in order to track loss.
            batch_x_binary_pred, batch_y_pred, _, _ = joint_model(batch_x, training=True)
            recon_loss = loss_fn_attn(batch_y_true, batch_y_pred)
            reg_loss = joint_model.losses
            loss_value = recon_loss + reg_loss

            # current attn weights for all positions.
            zero_counts = 0.
            tol_counts = 0.
            for attn_position in attn_positions:
                layer_attn_weights = \
                    joint_model.get_layer(
                        'dcnn_model').get_layer(
                            f'attn_factory_{attn_position}').get_weights()[0]
                        
                zero_counts += (len(layer_attn_weights) - len(np.nonzero(layer_attn_weights)[0]))
                tol_counts += len(layer_attn_weights)

            percent_zero = zero_counts / tol_counts
        
        # convert binary_true to legal batch shape.
        batch_x_binary_true = [
            [0,0,0], [0,0,1], [0,1,0], [0,1,1],
            [1,0,0], [1,0,1], [1,1,0], [1,1,1]
        ]
        batch_x_binary_true = tf.reshape(
            tf.convert_to_tensor(batch_x_binary_true, dtype=tf.float32),
            batch_x_binary_pred.shape
        )
        recon_loss_ideal = binary_crossentropy(
            reduction_method='SUM_OVER_DIMENSION')(batch_x_binary_true, batch_x_binary_pred)

        # update attn weights in `joint_model`
        grads = tape.gradient(loss_value, joint_model.trainable_weights)
        optimizer_attn.apply_gradients(zip(grads, joint_model.trainable_weights))

        # tracking losses (notice, all metrics are BEFORE updates)
        recon_loss_collector.append(recon_loss)
        recon_loss_ideal_collector.extend(recon_loss_ideal)
        reg_loss_collector.append(reg_loss)
        percent_zero_attn_collector.append(percent_zero)
        
        print(f'------------------------')
        print(f'# Before Update Metrics:')
        print(f'[Check] batch_x_binary_pred = {batch_x_binary_pred}')
        print(f'[Check] batch_y_pred = {batch_y_pred}')
        print(f'[Check] batch_y_true = {batch_y_true}')
        print(f'[Check] recon_loss = {recon_loss}')
        print(f'[Check] recon_loss_binary = {recon_loss_ideal}, avg = {np.mean(recon_loss_ideal)}')
        print(f'[Check] reg_loss = {reg_loss[0]}')
        print(f'[Check] percent_zero = {percent_zero}')
        print(f'[Check] item_proberror = {item_proberror}')
        global_steps += 1

    # log the latest attn weights at the end of this inner-loop.
    attn_weights = {}
    for attn_position in attn_positions:
        layer_attn_weights = \
            joint_model.get_layer(
                'dcnn_model').get_layer(
                    f'attn_factory_{attn_position}').get_weights()[0]
        attn_weights[attn_position] = layer_attn_weights
        
    return joint_model, attn_weights, \
        recon_loss_collector, recon_loss_ideal_collector, \
        reg_loss_collector, percent_zero_attn_collector, global_steps, \
        optimizer_attn