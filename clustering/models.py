import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input

try:
    from clustering.layers import *
except ModuleNotFoundError:
    from layers import *


class ClusterModel(Model):

    def __init__(
            self,
            num_clusters,
            r, q, specificity, trainable_specificity, attn_constraint,
            Phi, actv_func, beta, temp, output_dim=3,
            name="clustering_model",
            **kwargs
        ):
        super(ClusterModel, self).__init__(name=name, **kwargs)

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
            attn_constraint=attn_constraint,
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

        self.temp = temp

    def cluster_support(self, cluster_index, assoc_weights, y_true):
        """
        Compute support for a single cluster

        support_i 
            = (w_i_correct - w_i_incorrect) / (|w_i_correct| + |w_i_incorrect|)
        """
        # get assoc weights of a cluster
        cluster_assoc_weights = assoc_weights[cluster_index, :]
        print(f'[Check] cluster_assoc_weights', cluster_assoc_weights)

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
        print(f'[Check] w_correct={w_correct}, w_incorrect={w_incorrect}')
        print(f'[Check] cluster{cluster_index} support = {support}')
        return support

    def cluster_softmax(self, clusters_actv_inhibition, nonzero_clusters):
        # nominator of softmax
        nom = tf.exp(
            tf.gather(
                clusters_actv_inhibition[0], indices=nonzero_clusters) / self.temp
        )
        # denominator of softmax 
        denom = tf.reduce_sum(nom)

        # softmax probabilities
        softmax_proba = nom / tf.reduce_sum(nom)

        # flatten as required
        softmax_proba = tf.reshape(softmax_proba, [-1])
        
        # To expand the proba into the same size 
        # as the cluster activities because proba is 
        # only a subset.
        # To do that, we initialise a zero-tensor with
        # the size of the cluster activations and perform
        # value update.
        softmax_weights = tf.constant(
            [0], 
            shape=clusters_actv_inhibition[0].shape,
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

        clusters_actv_softmax = tf.multiply(clusters_actv_inhibition, softmax_weights)
        print('[Check] clusters_actv_softmax', clusters_actv_softmax)
        return clusters_actv_softmax

    def call(self, inputs, build_model=False, y_true=None):
        """
        inputs:
        -------
            inputs: current trial's item
            build_model: 
                if True, just for building the graph, called at the beginning.
            y_true: 
                if None, the forward pass is about loss eval 
                if given, the forward pass is about recruitment (checking support)

        """
        dist0 = self.Distance0(inputs)
        dist1 = self.Distance1(inputs)
        dist2 = self.Distance2(inputs)
        dist3 = self.Distance3(inputs)
        dist4 = self.Distance4(inputs)
        dist5 = self.Distance5(inputs)
        dist6 = self.Distance6(inputs)
        dist7 = self.Distance7(inputs)

        H_list = []
        for dist in [dist0, dist1, dist2, dist3, dist4, dist5, dist6, dist7]:
            attn_dist = self.DimensionWiseAttnLayer(dist)
            H_j_act = self.ClusterActvLayer(attn_dist)
            H_list.append(H_j_act)

        H_concat = self.Concat(H_list)
        print(f'H_concat', H_concat)

        H_out = self.MaskNonRecruit(H_concat)
        print(f'H_out', H_out)

        clusters_actv_inhibition = self.Inhibition(H_out)
        print(f'clusters_actv_inhibition', clusters_actv_inhibition)

        if build_model:
            y_pred = self.ClsLayer(clusters_actv_inhibition)
            return y_pred
        else:
            # do softmax only on the recruited clusters.
            nonzero_clusters = tf.cast(
                tf.where(clusters_actv_inhibition[0] > 0),
                dtype=tf.int32
            )

            print(f'[Check] nonzero_clusters = ', nonzero_clusters)

            # weight cluster activations by softmax.
            clusters_actv_softmax = self.cluster_softmax(
                clusters_actv_inhibition, 
                nonzero_clusters
            )

            y_pred = self.ClsLayer(clusters_actv_softmax)
            
            # loss eval
            if y_true is None:
                return y_pred
            
            # recruitment rule: checking `support`
            else:
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
                return y_pred, totalSupport


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
