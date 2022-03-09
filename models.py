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
from keras_custom import initializers


def presave_dcnn(config_version, path_model):
    """
    Load & save a finetuned dcnn with lowAttn.
    """
    
    config = load_config(
        config_version=config_version
    )

    # load dcnn model.
    model_dcnn, _, _ = model_base(
        config=config
    )
    
    dcnn_base = config['dcnn_base']
    dcnn_config_version = config['dcnn_config_version']
    dcnn_save_path = f'finetune/results/{dcnn_base}/{dcnn_config_version}/trained_weights'
    
    # load pred layer weights.
    with open(os.path.join(dcnn_save_path, 'pred_weights.pkl'), 'rb') as f:
        pred_weights = pickle.load(f)
    model_dcnn.get_layer('pred').set_weights(pred_weights)
    
    # load attn layer weights.
    attn_positions = config['low_attn_positions'].split(',')
    attn_weights = np.load(
        f'{dcnn_save_path}/attn_weights.npy', allow_pickle=True
    )
    attn_weights = attn_weights.ravel()[0]
    for attn_position in attn_positions:
        layer_attn_weights = attn_weights[attn_position]
        model_dcnn.get_layer(
            f'attn_factory_{attn_position}').set_weights([layer_attn_weights])
    
    # # FIXME:
    # # save model for loading later.
    # model_dcnn.save(path_model)
    # print(f'dcnn_model-with-lowAttn saved as {path_model}.')
    return model_dcnn


def DCNN(config_version):
    """
    Load trained dcnn model.

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
    config = load_config(config_version=config_version)
    dcnn_config_version = config['dcnn_config_version']
    dcnn_base = config['dcnn_base']
    if dcnn_base == 'vgg16':
        preprocess_func = tf.keras.applications.vgg16.preprocess_input

    # load dcnn model.
    path_model = f'dcnn_models-with-lowAttn/{dcnn_config_version}'
    # if not os.path.exists(path_model):
    #     presave_dcnn(config_version, path_model)
    # model_dcnn = tf.keras.models.load_model(path_model)
    
    # FIXME:
    model_dcnn = presave_dcnn(config_version, path_model)

    return model_dcnn, preprocess_func


class JointModel(Model):
    
    def __init__(
            self,
            config_version,
            name="joint_model",
            **kwargs
        ):
        super(JointModel, self).__init__(name=name, **kwargs)
        
        config = load_config(config_version=config_version)
        
        # --- finetuned DCNN with attn --- #
        self.DCNN, self.preprocess_func = DCNN(
            config_version=config_version,
        )
        
        # freeze DCNN except attn layers.
        for layer in self.DCNN.layers:
            if 'attn' in layer.name:
                continue
            else:
                layer.trainable = False

        # --- freshly defined cluster model --- #
        num_clusters = config['num_clusters']
        r = config['r']
        q = config['q']
        specificity = config['specificity']
        trainable_specificity = config['trainable_specificity']
        high_attn_constraint = config['high_attn_constraint']
        Phi = config['Phi']
        actv_func = config['clus_actv_func']
        beta = config['beta']
        temp1 = config['temp1']
        temp2 = config['temp2']
        
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

        print(f'[Check] un-inhibited cluster outputs')
        return inputs_binary, clusters_actv, y_pred, totalSupport
    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    
    JointModel(config_version='top')