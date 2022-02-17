import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from utils import load_config
from finetune.models import model_base
from clustering.layers import *
from clustering.models import ClusterModel
from layers import AttnFactory


def presave_dcnn(dcnn_config_version, path_model):
    """
    Load & save a finetuned dcnn.
    """
    # load two configs
    dcnn_config = load_config(
        component='finetune', 
        config_version=dcnn_config_version)

    # load dcnn model.
    model_dcnn, input_shape, preprocess_func = model_base(
        model_name=dcnn_config['model_name'], 
        actv_func=dcnn_config['actv_func'],
        kernel_constraint=dcnn_config['kernel_constraint'],
        kernel_regularizer=dcnn_config['kernel_regularizer'],
        hyperbolic_strength=dcnn_config['hyperbolic_strength'],
        lr=dcnn_config['lr'],
        train=dcnn_config['train'],
        stimulus_set=dcnn_config['stimulus_set'],
        layer=dcnn_config['layer'],
        intermediate_input=False)
    model_name = dcnn_config['model_name']
    dcnn_save_path = f'finetune/results/{model_name}/config_{dcnn_config_version}/trained_weights'
    with open(os.path.join(dcnn_save_path, 'pred_weights.pkl'), 'rb') as f:
        pred_weights = pickle.load(f)
    model_dcnn.get_layer('pred').set_weights(pred_weights)
    # save model for loading later.
    model_dcnn.save(path_model)
    print(f'dcnn model saved as {path_model}.')


def DCNN(attn_config_version, 
         dcnn_config_version, 
         intermediate_input):
    """
    Load trained dcnn model with option to add 
    a single or multiple layers of attn.

    impl: 
    -----
        Try loading the presaved finetuned dcnn, 
        If not available, call `presave_dcnn`.
        Then, attn layers are added to the dcnn.
    
    return:
    -------
        model_dcnn: finetuned dcnn with attn
        preprocess_func: preprocessing for this dcnn
    """
    attn_config = load_config(
            component=None, 
            config_version=attn_config_version)
    dcnn_config = load_config(
        component='finetune', 
        config_version=dcnn_config_version)
    dcnn_base = dcnn_config['model_name']
    
    if dcnn_base == 'vgg16':
        preprocess_func = tf.keras.applications.vgg16.preprocess_input

    # load dcnn model.
    path_model = f'dcnn_models/{dcnn_config_version}'
    if not os.path.exists(path_model):
        presave_dcnn(dcnn_config_version, path_model)
    model_dcnn = tf.keras.models.load_model(path_model)

    # ------ attn stuff ------ #
    # attn layer positions
    attn_positions = attn_config['attn_positions'].split(',')
    # attn layer settings
    if attn_config['attn_initializer'] == 'ones':
        attn_initializer = tf.keras.initializers.Ones()
    if attn_config['low_attn_constraint'] == 'nonneg':
        low_attn_constraint = tf.keras.constraints.NonNeg()
    if attn_config['attn_regularizer'] == 'l1':
        attn_regularizer = tf.keras.regularizers.l1(
            attn_config['reg_strength'])
    else:
        attn_regularizer = None

    # if no attn used, return the finetuned dcnn itself.
    if attn_positions is None:
        return model_dcnn, preprocess_func

    # loop thru all layers and apply attn at multiple positions.
    else:
        dcnn_layers = model_dcnn.layers[1:]
        x = model_dcnn.input
        fake_inputs = []
        for layer in dcnn_layers:

            # regardless of attn
            # apply one layer at a time from DCNN.
            layer.trainable = False
            x = layer(x)

            # apply attn at the output of the above layer output
            if layer.name in attn_positions:
                attn_size = x.shape[-1]

                fake_input = layers.Input(
                    shape=(attn_size,),
                    name=f'fake_input_{layer.name}'
                )
                fake_inputs.append(fake_input)
                
                attn_weights = AttnFactory(
                    output_dim=attn_size, 
                    input_shape=fake_input.shape,
                    name=f'attn_factory_{layer.name}',
                    initializer=attn_initializer,
                    constraint=low_attn_constraint,
                    regularizer=attn_regularizer
                )(fake_input)

                # reshape attn to be compatible.
                attn_weights = layers.Reshape(
                    target_shape=(1, 1, attn_weights.shape[-1]),
                    name=f'reshape_attn_{layer.name}')(attn_weights)

                # apply attn to prev layer output
                x = layers.Multiply(name=f'post_attn_actv_{layer.name}')([x, attn_weights])

        inputs = [model_dcnn.inputs]
        inputs.extend(fake_inputs)
        model_dcnn = Model(inputs=inputs, outputs=x, name='dcnn_model')
        return model_dcnn, preprocess_func


class JointModel(Model):
    
    def __init__(
            self,
            attn_config_version,
            dcnn_config_version, 
            intermediate_input=False,
            name="joint_model",
            **kwargs
        ):
        super(JointModel, self).__init__(name=name, **kwargs)
        
        attn_config = load_config(
            component=None,
            config_version=attn_config_version,
        )
        
        # --- finetuned DCNN with attn --- #
        self.DCNN, self.preprocess_func = DCNN(
            attn_config_version=attn_config_version,
            dcnn_config_version=dcnn_config_version,
            intermediate_input=intermediate_input
        )
        
        # --- freshly defined cluster model --- #
        num_clusters = attn_config['num_clusters']
        r = attn_config['r']
        q = attn_config['q']
        specificity = attn_config['specificity']
        trainable_specificity = attn_config['trainable_specificity']
        high_attn_constraint = attn_config['high_attn_constraint']
        Phi = attn_config['Phi']
        actv_func = attn_config['actv_func']
        beta = attn_config['beta']
        temp1 = attn_config['temp1']
        temp2 = attn_config['temp2']
        
        output_dim = self.DCNN.output.shape[-1]
        self.Distance0 = Distance(output_dim=output_dim, name=f'd0')
        self.Distance1 = Distance(output_dim=output_dim, name=f'd1')
        self.Distance2 = Distance(output_dim=output_dim, name=f'd2')
        self.Distance3 = Distance(output_dim=output_dim, name=f'd3')
        self.Distance4 = Distance(output_dim=output_dim, name=f'd4')
        self.Distance5 = Distance(output_dim=output_dim, name=f'd5')
        self.Distance6 = Distance(output_dim=output_dim, name=f'd6')
        self.Distance7 = Distance(output_dim=output_dim, name=f'd7')

        self.DimensionWiseAttnLayer = DimensionWiseAttn(
            output_dim=output_dim,
            r=r, 
            attn_constraint=high_attn_constraint,
            name='dimensionwise_attn_layer'
        )

        self.ClusterActvLayer = ClusterActivation(
            output_dim=1, r=r, q=q, specificity=specificity, 
            trainable_specificity=trainable_specificity,
            name=f'cluster_actv_layer'
        )

        self.Concat = tf.keras.layers.Concatenate(
            axis=1, 
            name='concat'
        )

        self.MaskNonRecruit = Mask(
            num_clusters, 
            name='mask_non_recruit'
        )

        self.Inhibition = ClusterInhibition(
            num_clusters, 
            beta=beta, 
            name='inhibition'
        )
        
        self.ClsLayer = Classification(
            2, Phi=Phi, activation=actv_func, 
            name='classification'
        )

        self.beta = beta
        self.temp1 = temp1
        self.temp2 = temp2
    
    def cluster_support(self, cluster_index, assoc_weights, y_true):
        """
        Compute support for a single cluster
        support_i 
            = (w_i_correct - w_i_incorrect) / (|w_i_correct| + |w_i_incorrect|)
        """
        # get assoc weights of a cluster
        cluster_assoc_weights = assoc_weights[cluster_index, :]
        # print(f'[Check] cluster_assoc_weights', cluster_assoc_weights)

        # based on y_true
        if y_true[0][0] == 0:
            w_correct = cluster_assoc_weights[0, 1]
            w_incorrect = cluster_assoc_weights[0, 0]
        else:
            w_correct = cluster_assoc_weights[0, 0]
            w_incorrect = cluster_assoc_weights[0, 1]

        support = (w_correct - w_incorrect) / (
            np.abs(w_correct) + np.abs(w_incorrect)
        )
        # print(f'[Check] w_correct={w_correct}, w_incorrect={w_incorrect}')
        # print(f'[Check] cluster{cluster_index} support = {support}')
        return support
    
    def cluster_softmax(self, clusters_actv, nonzero_clusters, which_temp):
        """
        Compute new clusters_actv by 
            weighting clusters_actv using softmax(clusters_actv) probabilities.
        """
        clusters_actv_nonzero = tf.gather(
            clusters_actv[0], indices=nonzero_clusters)
        
        if which_temp == 1:
            temp = self.temp1
            if temp == 'equivalent':
                temp = clusters_actv_nonzero / (
                    self.beta * tf.math.log(clusters_actv_nonzero)
                )
        elif which_temp == 2:
            temp = self.temp2
            
        # softmax probabilities and flatten as required
        nom = tf.exp(clusters_actv_nonzero / temp)
        denom = tf.reduce_sum(nom)
        softmax_proba = nom / denom
        softmax_proba = tf.reshape(softmax_proba, [-1])
        
        # To expand the proba into the same size 
        # as the cluster activities because proba is 
        # only a subset.
        # To do that, we initialise a zero-tensor with
        # the size of the cluster activations and perform
        # value update.
        softmax_weights = tf.constant(
            [0], 
            shape=clusters_actv[0].shape,
            dtype=tf.float32
        )
        
        # print(f'tensor=softmax_weights', softmax_weights)
        # print(f'indices=nonzero_clusters', nonzero_clusters)
        # print(f'updates=softmax_proba', softmax_proba)
        softmax_weights = tf.tensor_scatter_nd_update(
            tensor=softmax_weights,
            indices=nonzero_clusters,
            updates=softmax_proba,
        )

        clusters_actv_softmax = tf.multiply(clusters_actv, softmax_weights)
        print('[Check] clusters_actv_softmax', clusters_actv_softmax)
        return clusters_actv_softmax
        
    def call(self, inputs, y_true=None):
        """
        Only when y_true is provided, 
        totalSupport will be computed.
        """
        # DCNN to produce binary outputs.
        inputs_binary = self.DCNN(inputs)
                
        # Continue with cluster model to 
        # produce cluster outputs.
        dist0 = self.Distance0(inputs_binary)
        dist1 = self.Distance1(inputs_binary)
        dist2 = self.Distance2(inputs_binary)
        dist3 = self.Distance3(inputs_binary)
        dist4 = self.Distance4(inputs_binary)
        dist5 = self.Distance5(inputs_binary)
        dist6 = self.Distance6(inputs_binary)
        dist7 = self.Distance7(inputs_binary)

        H_list = []
        for dist in [dist0, dist1, dist2, dist3, dist4, dist5, dist6, dist7]:
            attn_dist = self.DimensionWiseAttnLayer(dist)
            H_j_act = self.ClusterActvLayer(attn_dist)
            H_list.append(H_j_act)

        H_concat = self.Concat(H_list)
        # print(f'H_concat', H_concat)

        clusters_actv = self.MaskNonRecruit(H_concat)
        # print(f'H_out', H_out)

        # do softmax only on the recruited clusters.
        nonzero_clusters = tf.cast(
            tf.where(clusters_actv[0] > 0),
            dtype=tf.int32
        )

        # weight cluster activations by softmax.
        clusters_actv_softmax = self.cluster_softmax(
            clusters_actv, 
            nonzero_clusters,
            which_temp=1
        )
        
        clusters_actv_softmax = self.cluster_softmax(
            clusters_actv_softmax, 
            nonzero_clusters,
            which_temp=2
        )
        
        # final model probability reponse.
        y_pred = self.ClsLayer(clusters_actv_softmax)
        
        # whether to compute totalSupport depends.
        totalSupport = 0
        if y_true is None:
            print(f'[Check] Not evaluating totalSupport.')
        else:
            print(f'[Check] Evaluating totalSupport')
            assoc_weights = self.ClsLayer.get_weights()[0]
            totalSupport = 0
            for cluster_index in nonzero_clusters:
                support = self.cluster_support(
                    cluster_index, assoc_weights, y_true
                )

                single_cluster_actv = tf.gather(
                    clusters_actv_softmax[0], indices=cluster_index
                )

                totalSupport += support * single_cluster_actv

            totalSupport = totalSupport / tf.reduce_sum(clusters_actv_softmax)
            print(f'[Check] totalSupport = {totalSupport}')
            
        # return inputs_binary, clusters_actv_softmax, y_pred, totalSupport
        return inputs_binary, clusters_actv, y_pred, totalSupport


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'