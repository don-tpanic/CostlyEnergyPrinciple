import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np 
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.models import Model

from utils import load_config, png2gif
from models import JointModel
from data import data_loader_V2, load_X_only
from clustering.utils import load_data
from losses import binary_crossentropy

"""
Evaluation routines.
"""

def dcnn_final_connections(
        attn_config_version='attn_v1',
        dcnn_config_version='t1.block4_pool.None.run1'):
    """
    Examine how the final layer connections 
    of the dcnn look like. Since a lot of 
    the block4_pool outputs are zero, and 
    it seems reducing attn weights to 0.05 
    wont affect binary codings, maybe there is
    something in the final connections that 
    can explain this robustness.
    """
    # dcnn model
    model, preprocess_func = load_dcnn_model(
        attn_config_version=attn_config_version,
        dcnn_config_version=dcnn_config_version,
        intermediate_input=False
    )
    weights = model.get_layer('pred').get_weights()[0]
    print(weights.shape)

    # plot dim-wise
    fig, ax = plt.subplots(weights.shape[1])
    for dim in range(weights.shape[1]):
        dim_weights = weights[:, dim]
        ax[dim].hist(dim_weights)
        ax[dim].set_title(f'to output unit {dim}')
        ax[dim].set_xlim([-0.01, 0.01])
    plt.tight_layout()
    plt.savefig('pred_weights_hist.png')


def how_many_preattn_actv_already_zero(
        layer='block4_pool',
        attn_config_version='attn_v1',
        dcnn_config_version='t1.block4_pool.None.run1',
        threshold=0.8):
    """
    Activations at later layers of dcnn
    are already very sparse. Here we want to find out 
    given the finetuned DCNN, how a given layer's output
    look like given the stimulus inputs.
    """
    # dcnn model
    model, preprocess_func = load_dcnn_model(
        attn_config_version=attn_config_version,
        dcnn_config_version=dcnn_config_version,
        intermediate_input=False
    )

    # model kept at a layer
    partial_model = Model(
        inputs=model.input,
        outputs=model.get_layer(layer).output
    )

    # load inputs
    fig, ax = plt.subplots(1, 2)
    dataset = data_loader(
        dcnn_config_version=dcnn_config_version, 
        preprocess_func=preprocess_func,
        problem_type=1  # does not matter in this analysis
    )

    num_filter = partial_model.layers[-1].output.shape[-1]
    num_stimuli = len(dataset)
    filter_zero_percentage = np.empty((num_filter, num_stimuli))
    print(f'filter_zero_percentage.shape = {filter_zero_percentage.shape}')

    for i in range(num_stimuli):
        dp = dataset[i]
        x = dp[0]
        signature = dp[2]
        print(f'[Check] Signature = {signature}')
        # layer's output of 1 input (H, W, C)
        actv = partial_model(x, training=False)[0].numpy()

        already_all_zero_filter_idx = []

        for c in range(actv.shape[-1]):
            filter_actv = actv[:, :, c].flatten()
            zero_percent = (
                len(filter_actv) - len(np.nonzero(filter_actv)[0])
            ) / (len(filter_actv))
            # print(f'zero_percent = {zero_percent}')
            filter_zero_percentage[c, i] = zero_percent

            if zero_percent >= threshold:
                already_all_zero_filter_idx.append(c)
            
        np.save(
            f'already_all_zero_filter_idx_signature{signature}.npy',
            already_all_zero_filter_idx)
        print(f'% already zero filters = {len(already_all_zero_filter_idx) / actv.shape[-1]}')
    
    
    data = []
    for i in range(num_stimuli):
        data.append(
            filter_zero_percentage[:, i]
        )
    fig, ax = plt.subplots()
    sns.violinplot(
        data=data, 
        linewidth=.8,
        gridsize=300,
        ax=ax
    )
    ax.set_title('average percentage zero pre-attn actv')

    plt.tight_layout()
    plt.savefig('test.png')


def how_low_can_att_weights(
        attn_weight_constant,
        attn_config_version='attn_v1',
        dcnn_config_version='t1.block4_pool.None.run1',
        problem_type=1,
        noise_const=0):
    """
    Check how low can attn weights 
    go (uniformly) without affecting
    binary codings.

    Supply a constant that will be broadcast and used
    as attention weights.

    if noise_const:
        we add non-uniform noise to init attn weights, hoping to 
        get some zero weights sooner than having uniform reduction
        for a very long time.
    """
    # dcnn model
    model, preprocess_func = load_dcnn_model(
        attn_config_version=attn_config_version,
        dcnn_config_version=dcnn_config_version,
        intermediate_input=False
    )
    # sub in a low set of attn weights
    attn_weights = model.get_layer('attention_factory').get_weights()[0]
    low_attn_weights = np.ones((attn_weights.shape)) * attn_weight_constant
    if noise_const:
        noise = np.random.uniform(low=-noise_const, high=noise_const, size=attn_weights.shape)
        print(noise)
        low_attn_weights += noise
    print(low_attn_weights)
    print(f'max = {np.max(low_attn_weights)}, min = {np.min(low_attn_weights)}')
    model.get_layer('attention_factory').set_weights([low_attn_weights])

    # load data
    dataset = data_loader(
        dcnn_config_version=dcnn_config_version, 
        preprocess_func=preprocess_func,
        problem_type=problem_type
    )

    # each stimulus, train indendently
    y_pred = []
    for i in range(len(dataset)):
        dp = dataset[i]
        x = dp[0]
        y_pred.append(model.predict(x))
    y_pred = np.array(y_pred)
    print(y_pred)


def if_remove_already_zero_filters(
        attn_config_version='attn_v1',
        dcnn_config_version='t1.block4_pool.None.run1'):
    """
    `how_many_preattn_actv_already_zero` tells us 
    some filters already have zero activations. 
    Here we set the filter's attention weights directly
    to zero and
    """
    # dcnn model
    model, preprocess_func = load_dcnn_model(
        attn_config_version=attn_config_version,
        dcnn_config_version=dcnn_config_version,
        intermediate_input=False
    )
    # sub in a low set of attn weights
    attn_weights = model.get_layer('attention_factory').get_weights()[0]

    # load data
    dataset = data_loader(
        dcnn_config_version=dcnn_config_version, 
        preprocess_func=preprocess_func,
        problem_type=1
    )

    # each stimulus, train indendently
    y_pred = []
    for i in range(len(dataset)):
        dp = dataset[i]
        x = dp[0]
        signature = dp[2]
        already_zero_filters = np.load(
            f'already_all_zero_filter_idx_signature{signature}.npy')

        attn_weights[already_zero_filters] = 0
        # print(f'attn_weights = {attn_weights}')
        model.get_layer('attention_factory').set_weights([attn_weights])
        y_pred.append(model.predict(x))

    y_pred = np.array(y_pred)
    print(y_pred)


def examine_optimal_attn_weights_stimuluslevel(
        stimulus_set,
        attn_config_version,
        dcnn_config_version,
        num_epochs=100,
        lr=0.01,
        problem_type=1,
        attn_weight_constant=1,
        recon_level='dcnn',
        mode='train'):
    """
    Find out the optimal (upper bound) of the 
    attention weights can be when being independently
    optimised for each stimulus.

    Optimising targets can be either the cluster activations
    of a trained clustering model or the targets can be the 
    binary codins of the stimuli. The difference is that if 
    cluster activations, attention weights are problem_type
    specfic. Also when optimising cluster activations, 
    there is no guarantee that binary codings will be respected.
    But the opposite is true.

    If recon_level is 'binary':
        this code is self-contained, meaning 
        we do attn training and plotting in one.
    If recon_level is 'cluster': 
        this code can only plot the trained 
        and saved attn weights using the normal 
        `train.py` routine as it involves clustering
        learning which cannot be contained here.

    Outputs of this function:
    -------------------------
    1. trained attention weights
    2. Progress of recon & reg losses
    3. Heatmaps of nonzero intersections between pairs of attn weights
    4. Heatmaps of pairwise correlations of attn weights.
    5. Histograms of stimulus-level attn weights.
    """
    signature2coding = {
        0: [0, 0, 0],
        1: [0, 0, 1],
        2: [0, 1, 0],
        3: [0, 1, 1],
        4: [1, 0, 0],
        5: [1, 0, 1],
        6: [1, 1, 0],
        7: [1, 1, 1]
    }

    if recon_level == 'binary' and mode == 'train':
        # model till 3 binary units.
        model, preprocess_func = load_dcnn_model(
            attn_config_version,
            dcnn_config_version,
            intermediate_input=False
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        # dataset 
        dataset = data_loader(
            dcnn_config_version=dcnn_config_version, 
            preprocess_func=preprocess_func,
            problem_type=problem_type
        )

        # each stimulus, train indendently
        for i in range(len(dataset)):
            dp = dataset[i]
            x = dp[0]
            signature = dp[2]
            y_true = np.array(signature2coding[signature]).reshape((1, 3))

            # sub in a low set of attn weights
            attn_weights = model.get_layer('attention_factory').get_weights()[0]
            low_attn_weights = np.ones((attn_weights.shape)) * attn_weight_constant
            model.get_layer('attention_factory').set_weights([low_attn_weights])

            # for each stimulus, train a few epochs

            recon_loss_overtime = []
            reg_loss_overtime = []
            for e in range(num_epochs):
                with tf.GradientTape() as tape:
                    y_pred = model(x)
                    recon_loss = loss_fn(y_true, y_pred)
                    reg_loss = model.losses
                    loss_value = recon_loss + reg_loss

                # update attn weights
                model.get_layer(f'attention_factory').trainable = True
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                print(f'* epoch=[{e}], y_pred={y_pred}\n, recon_loss={recon_loss}, reg_loss={reg_loss}')
                # print('grads', grads)

                # track loss overtime
                recon_loss_overtime.append(recon_loss)
                reg_loss_overtime.append(reg_loss)

            # plot for per stimulus
            fig, ax = plt.subplots(2)
            ax[0].plot(range(len(recon_loss_overtime)), recon_loss_overtime)
            ax[0].set_title('recon loss')
            ax[1].plot(range(len(reg_loss_overtime)), reg_loss_overtime)
            ax[1].set_title('reg loss')
            plt.tight_layout()
            plt.savefig(f'temp_analysis/loss_overtime-{signature}-{stimulus_set}-{problem_type}-{lr}.png')

            # at the end of all epochs,
            attn_weights = model.get_layer('attention_factory').get_weights()[0]
            np.save(f'temp_analysis/attn_weights-{signature}-{stimulus_set}-{problem_type}-{lr}.npy', attn_weights)

    signatures = range(8)
    intersect_mtx = np.zeros((8, 8))
    correlation_mtx_full = np.zeros((8, 8))
    correlation_mtx_intersect = np.zeros((8, 8))
    for i in signatures:
        for j in signatures:
            if i >= j:
                continue
            else:
                fpath_i = f'temp_analysis/attn_weights-{i}-{stimulus_set}-{problem_type}-{lr}.npy'
                attn_weights_i = np.load(fpath_i)
                nonzero_filter_i = np.nonzero(attn_weights_i)[0]
                # print(f'sig {i}, nonzero {len(nonzero_filter_i)}')

                fpath_j = f'temp_analysis/attn_weights-{j}-{stimulus_set}-{problem_type}-{lr}.npy'
                attn_weights_j = np.load(fpath_j)
                nonzero_filter_j = np.nonzero(attn_weights_j)[0]
                # print(f'sig {j}, nonzero {len(nonzero_filter_j)}')

                intersect = np.intersect1d(nonzero_filter_i, nonzero_filter_j)
                # NOTE, nonzero differs between two data-points, take average.
                intersect_percent = 2 * len(intersect) / ( len(nonzero_filter_i) + len(nonzero_filter_j))
                intersect_mtx[i, j] = intersect_percent

                # compute pair-wise correlation 
                # one is corr between 512 weights; 
                # the other is corr between intersected weights
                r_full, _ = stats.spearmanr(attn_weights_i, attn_weights_j)
                correlation_mtx_full[i, j] = r_full

                r_intersect, _ = stats.spearmanr(
                    attn_weights_i[intersect],
                    attn_weights_j[intersect]
                )
                correlation_mtx_intersect[i, j] = r_intersect
    print('[Check] intersect indices', intersect)
    # mtx_corr, _ = stats.spearmanr(
    #     intersect_mtx[np.triu_indices(intersect_mtx.shape[0], k=1)], 
    #     correlation_mtx_intersect[np.triu_indices(correlation_mtx_intersect.shape[0], k=1)]
    # )
    # print(f'intersect_mtx vs correlation_mtx_intersect, r={mtx_corr}')

    # ============ plotting ===============
    labels = ['000', '001', '010', '011', '100', '101', '110', '111']
    fig, ax = plt.subplots(2)
    sns.heatmap(
        intersect_mtx, 
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax[0])
    ax[0].set_title(f'percentage overlap\nstimulus_set={stimulus_set}, problem={problem_type}')
    # sns.heatmap(
    #     correlation_mtx_full, 
    #     annot=True,
    #     xticklabels=labels,
    #     yticklabels=labels,
    #     ax=ax[1])
    # ax[1].set_title(f'spearman correlation (full)')
    sns.heatmap(
        correlation_mtx_intersect, 
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax[1])
    ax[1].set_title(f'spearman correlation (intersect)')
    plt.tight_layout()
    plt.savefig(f'temp_analysis/attn_intersect+corr-{stimulus_set}-{problem_type}-{lr}.png')



    # plot attn weights histograms
    fig, ax = plt.subplots(2, 4)
    for i in signatures:
        fpath = f'temp_analysis/attn_weights-{i}-{stimulus_set}-{problem_type}-{lr}.npy'
        attn_weights = np.load(fpath)
        if i // 4 == 0:
            ax[0, i].hist(attn_weights)
        else:
            ax[1, i % 4].hist(attn_weights)
    plt.suptitle('optimal attn weights')
    plt.savefig(f'temp_analysis/attn_hist-{stimulus_set}-{problem_type}-{lr}.png')


def examine_optimal_attn_weights_batchlevel(
        stimulus_set,
        attn_config_version,
        dcnn_config_version,
        num_epochs=100,
        lr=0.01,
        problem_type=1,
        attn_weight_constant=1,
        recon_level='binary',
        mode='train',
        batch_size=None):
    """
    Find out the optimal (upper bound) of the 
    attention weights can be when being independently
    optimised for a batch (all or some stimuli)

    NOTE: different from stimulus level, we now have only
    one set of attn weights for all stimuli. Hence we cannot 
    compare attn weights intersection like we did above. If we 
    really want to, we could compare post-attn actv.

    There are two settings due to `batch_size`
    1. batch_size=None, all stimuli involved in one update;
    2. batch_size=1, 1 stimulus per update. Note this is different
       from indepedently optimise for each stimulus.
    """
    # model till 3 binary units.
    model, preprocess_func = load_dcnn_model(
        attn_config_version,
        dcnn_config_version,
        intermediate_input=False
    )

    # dataset
    dataset = data_loader(
        dcnn_config_version=dcnn_config_version, 
        preprocess_func=preprocess_func,
        problem_type=problem_type
    )
    
    y_true = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]],
        dtype=np.float32
    )

    if batch_size is None:
        num_steps = 1
        batch_size = len(dataset)
        signature = f'batch{batch_size}'
    else:
        num_steps = len(dataset)
        signature = f'batch{batch_size}'
    fname = f'{signature}-{stimulus_set}-{problem_type}-{lr}'

    if recon_level == 'binary' and mode == 'train':
        model.get_layer(f'attention_factory').trainable = True
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

        # model.get_layer('pred').activation = None
        # loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss_fn = binary_crossentropy()

        # convert dataset to a TF data mtx.
        # (8, 224, 224, 3)
        images = np.empty( (len(dataset), ) + dataset[0][0][0].shape[1:])
        for i in range(len(dataset)):
            dp = dataset[i]
            image = dp[0][0]
            images[i] = image
        
        # sub in a low set of attn weights
        attn_weights = model.get_layer('attention_factory').get_weights()[0]
        low_attn_weights = np.ones((attn_weights.shape)) * attn_weight_constant
        model.get_layer('attention_factory').set_weights([low_attn_weights])

        # for every batch, train a few epochs
        recon_loss_overtime = []
        reg_loss_overtime = []
        percent_zero_attn = []
        np.random.seed(999)

        assoc_weights = np.random.uniform(
            low=-1e-6, high=1e-6, size=(8,2)
        )

        for e in range(num_epochs):
            print('-------------------------------------------------------------------')
            # at every epoch, shuffle dataset
            shuffled_indices = np.random.choice(
                np.arange(len(dataset), 
                dtype=int), 
                size=len(dataset), 
                replace=False
            )
            print('shuffled_indices', shuffled_indices)
            shuffled_images = images[shuffled_indices]
            shuffled_y_true = y_true[shuffled_indices]
            print(f'shuffled_images.shape = {shuffled_images.shape}')
            print(f'shuffled_y_true.shape = {shuffled_y_true.shape}')
            for step in range(num_steps):
                with tf.GradientTape() as tape:
                    # add ones
                    ones_batch = np.ones((batch_size, dataset[0][0][1].shape[1]))
                    # get batch of x and y
                    batch_x = [shuffled_images[ step * batch_size : (step+1) * batch_size ], ones_batch]
                    batch_y = shuffled_y_true[ step * batch_size : (step+1) * batch_size ]
                    # predict and compute loss
                    y_pred = model(batch_x, training=False)
                    recon_loss = loss_fn(batch_y, y_pred)
                    reg_loss = model.losses
                    loss_value = recon_loss + reg_loss
                    print(f'>>> y_true', batch_y)
                    print(f'>>> y_pred', y_pred)
                    print(f'>>> loss_value', loss_value)

                # update attn weights
                grads = tape.gradient(loss_value, model.trainable_weights)
                print('grads', grads, '\n\n')
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                
                attn_weights = model.get_layer('attention_factory').get_weights()[0]
                percent_zero = (len(attn_weights) - len(np.nonzero(attn_weights)[0])) / len(attn_weights)
                print(f'* epoch=[{e}], step=[{step}], recon_loss={recon_loss}, reg_loss={reg_loss}, percent_zero={percent_zero}')

                # exit()

                # track loss n change of attn overtime
                recon_loss_overtime.append(recon_loss)
                reg_loss_overtime.append(reg_loss)
                percent_zero_attn.append(percent_zero)

        # at the end of all epochs,
        np.save(f'temp_analysis/attn_weights-{fname}.npy', attn_weights)
        np.save(f'temp_analysis/recon_loss-{fname}.npy', recon_loss_overtime)
        np.save(f'temp_analysis/reg_loss-{fname}.npy', reg_loss_overtime)
        np.save(f'temp_analysis/percent_zero_attn-{fname}.npy', percent_zero_attn)

    attn_weights = np.load(f'temp_analysis/attn_weights-{fname}.npy')
    recon_loss_overtime = np.load(f'temp_analysis/recon_loss-{fname}.npy')
    reg_loss_overtime = np.load(f'temp_analysis/reg_loss-{fname}.npy')
    percent_zero_attn = np.load(f'temp_analysis/percent_zero_attn-{fname}.npy')

    # plot for per stimulus
    fig, ax = plt.subplots(3)
    ax[0].plot(range(len(recon_loss_overtime)), recon_loss_overtime)
    ax[0].set_title('recon loss')
    ax[1].plot(range(len(reg_loss_overtime)), reg_loss_overtime)
    ax[1].set_title('reg loss')
    # ax[2].hist(attn_weights)
    # ax[2].set_title('attention weights')
    ax[2].plot(range(len(percent_zero_attn)), percent_zero_attn)
    ax[2].set_title('percentage of zero attn weights')
    plt.tight_layout()
    plt.savefig(f'temp_analysis/loss_overtime_n_attn_weights-{fname}.png')


def examine_optimal_attn_weights_binary_output(
        stimulus_set=1,
        attn_config_version='attn_v1',
        dcnn_config_version='t1.block4_pool.None.run1',
        num_epochs=100000,
        lr=0.1,
        problem_type=1,
        signature='batch'):
    """
    After obtain optimal attn weights from above,
    plug them back into dcnn and check binary outputs.
    """
    model, preprocess_func = load_dcnn_model(
        attn_config_version,
        dcnn_config_version,
        intermediate_input=False
    )

    attn_weights_fpath = f'temp_analysis/attn_weights-{signature}-{stimulus_set}-{problem_type}-{lr}.npy'
    attn_weights = np.load(attn_weights_fpath)
    model.get_layer('attention_factory').set_weights([attn_weights])

    dataset = data_loader(
        dcnn_config_version,
        preprocess_func,
        problem_type
    )

    y_pred = []
    for i in range(len(dataset)):
        dp = dataset[i]
        x = dp[0]
        y_pred.append(model.predict(x))
    print(np.array(y_pred))
    

def viz_losses(
        attn_config_version,
        problem_type,
        recon_level,
        run):
    """
    Visualize progress of losses
        - recon loss 
        - theother recon loss
        - reg loss
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    inner_loop_epochs = attn_config['inner_loop_epochs']
    recon_loss = np.load(
        f'results/{attn_config_version}/all_recon_loss_type{problem_type}_run{run}_{recon_level}.npy'
    )
    recon_loss_ideal = np.load(
        f'results/{attn_config_version}/all_recon_loss_ideal_type{problem_type}_run{run}_{recon_level}.npy'
    )
    reg_loss = np.load(
        f'results/{attn_config_version}/all_reg_loss_type{problem_type}_run{run}_{recon_level}.npy'
    )
    percent_zero = np.load(
        f'results/{attn_config_version}/all_percent_zero_attn_type{problem_type}_run{run}_{recon_level}.npy'
    )

    final_recon_loss_ideal = recon_loss_ideal[-1]
    final_percent_zero = percent_zero[-1]


    # ===== plotting =====
    fig, ax = plt.subplots(4, dpi=500)
    default_s = plt.rcParams['lines.markersize'] ** 2
    ax[0].plot(recon_loss)
    ax[0].set_title(f'recon loss wrt current: {recon_level} (DCNN)')

    ax[1].plot(recon_loss_ideal)
    ax[1].set_title(f'recon loss wrt ideal: binary (DCNN) [{final_recon_loss_ideal:.3f}]')

    # reg_loss = np.log(reg_loss)
    ax[2].plot(reg_loss)
    ax[2].set_title(f'reg loss (DCNN)')
    plt.suptitle(f'Target: {recon_level}, inner-loop: {inner_loop_epochs}')

    ax[3].plot(percent_zero)
    ax[3].set_title(f'percentage of zeroed attn weights (DCNN) [{final_percent_zero:.3f}]')

    plt.tight_layout()
    plt.savefig(f'results/{attn_config_version}/losses_type{problem_type}_run{run}_{recon_level}.png')
    plt.close()


def viz_cluster_params(
        attn_config_version,
        problem_type,
        recon_level,
        run
        ):
    """
    Visulize cluster params change over time.
    1. How lambdas change.
    2. How centers change.
    """
    # 1. lambdas ----
    # (744,)
    alphas = np.load(
        f'results/{attn_config_version}/all_alphas_type{problem_type}_run{run}_{recon_level}.npy'
    )
    print(f'alphas.shape = {alphas.shape}')
    
    num_steps = int(len(alphas) / 3)
    # (3, num_steps)
    alphas = alphas.reshape((num_steps, 3)).T

    fig, ax = plt.subplots(dpi=300)
    for i in range(alphas.shape[0]):
        alphas_row_i = alphas[i, :]
        ax.plot(
            range(num_steps), alphas_row_i, 
            # s=plt.rcParams['lines.markersize'] / 8,
            label=f'dim {i}')

    ax.set_title(f'alphas tuning')
    ax.set_xlabel(f'num of epochs (exc first epoch)')
    ax.set_xticks(range(0, num_steps, 8 * 5))
    ax.set_xticklabels(range(0, 31, 5))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/{attn_config_version}/alphas_type{problem_type}_run{run}_{recon_level}.png')
    plt.close()
    print('[Check] saved alphas figure.')

    # 2. centers ----
    # centers = np.load(
    #     f'results/{attn_config_version}/all_centers_type{problem_type}_run{run}_{recon_level}.npy'
    # )
    # # print(centers.shape) # (24 * 8 * 31, )
    # # every step has 24 (3 * 8 clusters) values
    # num_clusters = 8
    # num_dims = 3
    # num_steps = int(len(centers) / (num_clusters * num_dims))
    # # (steps, 24)
    # centers = centers.reshape((num_steps, num_clusters * num_dims)).T

    # num_cols = 4
    # num_rows = int(num_clusters / num_cols)
    # fig, ax = plt.subplots(num_rows, num_cols)
    # for i in range(num_clusters):
    #     # extract
    #     # (3, steps)
    #     center_i_overtime = centers[i * num_dims : (i+1) * num_dims, :]
    #     # plot
    #     row_idx = i // num_cols
    #     col_idx = i % num_cols
    #     # plot each dim of a cluster at a time
    #     for dim in range(num_dims):
    #         center_i_dim_overtime = center_i_overtime[dim, :]
    #         ax[row_idx, col_idx].plot(center_i_dim_overtime, label=f'dim{dim}')
    #     center_i_final_loc = np.around(center_i_overtime[:, -1], 1)
    #     ax[row_idx, col_idx].set_title(f'cluster #{i}\n{center_i_final_loc}')
    #     ax[row_idx, col_idx].get_xaxis().set_visible(False)
    #     ax[row_idx, col_idx].set_ylim([-0.05, 1.05])
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(f'results/{attn_config_version}/centers_type{problem_type}_run{run}_{recon_level}.png')
    # plt.close()
    # print(f'[Check] saved centers figure.')


def examine_clustering_learning_curves(
        attn_config_version, recon_level='cluster'):
    """
    Just sanity check that the lc 
    of types are expected.
    """
    num_types = 6
    colors = ['blue', 'orange', 'black', 'green', 'red', 'cyan']
    fig, ax = plt.subplots()
    trapz_areas = np.empty(num_types)
    for i in range(num_types):
        problem_type = i + 1
        lc = np.load(
            f'results/{attn_config_version}/lc_type{problem_type}_{recon_level}.npy'
        )
        trapz_areas[i] = np.round(np.trapz(lc), 3)
        ax.errorbar(
            range(lc.shape[0]), 
            lc, 
            color=colors[i],
            label=f'Type {problem_type}',
        )
    print(f'[Results] trapzoidal areas = ', trapz_areas)
    plt.legend()
    plt.title(f'{trapz_areas}')
    plt.tight_layout()
    plt.savefig(f'results/{attn_config_version}/lc.png')


def attn_weights_stability():
    versions = [
        'attn_v3b_rerun0', 'attn_v3b_rerun1', 
        'attn_v3b_rerun2', 'attn_v3b_rerun3',
        'attn_v3b_rerun4'
    ]

    num_types = 6
    num_runs = 10

    for i in range(num_types):
        problem_type = i + 1

        print(f'========= [Check] Type {problem_type} =========')

        for run in range(num_runs):
            all_attn_weights = np.empty((len(versions), 512))
            for i in range(len(versions)):
                v = versions[i]
                fpath = f'results/{v}/attn_weights_type{problem_type}_run{run}_cluster.npy'
                attn_weights = np.load(fpath)
                all_attn_weights[i, :] = attn_weights
            
            for i in range(len(versions)):
                for j in range(len(versions)):
                    if i >= j:
                        continue
                    else:
                        attn1 = all_attn_weights[i]
                        attn2 = all_attn_weights[j]
                        r, _ = stats.spearmanr(attn1, attn2)
                        print(f'run {run}, {versions[i]} v {versions[j]}, r = {r}')


def find_canonical_runs(
        attn_config_version,
        canonical_runs_only=True):
    """
    Record the runs that produce canonical solutions
    for each problem type. 
    Specificially, we check the saved `mask_non_recruit`

    if canonical_solutions:
        only return runs that yield canonical solutions.
    """
    num_types = 6
    results_path = f'results/{attn_config_version}'
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    num_runs = attn_config['num_runs']
    type2cluster = {
        1: 2, 2: 4,
        3: 6, 4: 6, 5: 6,
        6: 8}

    from collections import defaultdict
    canonical_runs = defaultdict(list)

    for i in range(num_types):
        problem_type = i + 1

        for run in range(num_runs):

            mask_non_recruit = np.load(
                f'results/{attn_config_version}/mask_non_recruit_type{problem_type}_run{run}_cluster.npy')
            num_nonzero = len(np.nonzero(mask_non_recruit)[0])
            if canonical_runs_only:
                if num_nonzero == type2cluster[problem_type]:
                    canonical_runs[i].append(run)
            else:
                canonical_runs[i].append(run)
        print(f'Type {problem_type}, has {len(canonical_runs[i])}/{num_runs} canonical solutions')
    return canonical_runs


def canonical_runs_correspondence_to_attn_n_binary(
        attn_config_version):
    """
    Within canonical solutions, look at each run's 
    corresponding alphas, zero% and binary recon level.

    This is to better understand the source of variation 
    in the final zero% and binary recon to the relationship
    of cluster solutions and alphas.
    """
    num_types = 6
    results_path = f'results/{attn_config_version}'
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    num_runs = attn_config['num_runs']
    type2cluster = {
        1: 2, 2: 4,
        3: 6, 4: 6, 5: 6,
        6: 8}

    from collections import defaultdict
    canonical_runs = defaultdict(list)

    for i in range(num_types):
        problem_type = i + 1

        problem_type = 3

        print(f'********* Type {problem_type} *********')
        for run in range(num_runs):
            mask_non_recruit = np.load(
                f'results/{attn_config_version}/mask_non_recruit_type{problem_type}_run{run}_cluster.npy')
            num_nonzero = len(np.nonzero(mask_non_recruit)[0])
            if num_nonzero == type2cluster[problem_type]:
                alphas = f'{results_path}/all_alphas_type{problem_type}_run{run}_cluster.npy'
                zero_attn = f'{results_path}/all_percent_zero_attn_type{problem_type}_run{run}_cluster.npy'
                binary_recon = f'{results_path}/all_recon_loss_ideal_type{problem_type}_run{run}_cluster.npy'
                cluster_targets = f'clustering/results/clustering_v1/cluster_actv_targets_{problem_type}_248_{run}.npy'
                centers = f'{results_path}/all_centers_type{problem_type}_run{run}_cluster.npy'
                centers_flat = np.load(centers)[-8*3:]
                centers_notflat = []
                for idx in range(0, len(centers_flat), 3):
                    centers = centers_flat[idx :  idx+3]
                    centers_notflat.append(list(np.round(centers, 2)))
                print(f'centers: {centers_notflat}')
                print(f'cluster targets: \n{np.load(cluster_targets)}')
                print(f'{mask_non_recruit}')
                print(f'alphas: {list(np.round(np.load(alphas)[-3:], 2))}')
                print(f'%attn=0: {np.load(zero_attn)[-1]:.2f}')
                print(f'binary recon: {np.mean(np.load(binary_recon)[-3:]):.2f}')
            print('----------------------------------------------')
        print('----------------------------------------------')
        exit()
  

def compare_across_types_V3(
        attn_config_version, 
        canonical_runs_only=False, 
        threshold=[0, 0, 0], 
        counterbalancing=True):
    """
    For each type's results, here we split them 
    based on the attention solutions.

    E.g. Type1 has two attn solutions by focusing on
    dim1 and dim2 or just focusing on dim1. We will plot
    the results separately for each case.

    NOTE: we do not consider non-canonical vs canonical 
    for now. One reason being that the difference causing 
    by different attention solutions seem bigger.

    NOTE: there are cases where some dimensions are not 
    fully attended, we use a `threshold` parameter to guard 
    these dimensions. 
    E.g. if threshold [0, 0, 0], alphas = [0.33, 0.1, 0] is considered 
            strategy type [True, True, False]
         if threshold [0.1, 0.1, 0.1], the same alphas is considered 
            strategy type [True, False, False]
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    num_runs = attn_config['num_runs']
    num_types = 6
    num_dims = 3
    comparisons = ['zero_attn', 'binary_recon']
    results_path = f'results/{attn_config_version}'
    type2runs = find_canonical_runs(
        attn_config_version, canonical_runs_only=canonical_runs_only)

    for c in range(len(comparisons)):
        comparison = comparisons[c]

        # e.g. { problem_type: {(True, False, False): [metric1, metric2, ... ]} }
        type2strategy2metric = defaultdict(lambda: defaultdict(list))

        for z in range(num_types):

            for run in type2runs[z]:
                problem_type = z + 1
                
                print(f'problem_type = {problem_type}')
                
                # First grab the final metric
                if comparison == 'zero_attn':
                    metric_fpath = f'{results_path}/all_percent_zero_attn_type{problem_type}_run{run}_cluster.npy'
                    metric = np.load(metric_fpath)[-1]                    
                else:
                    metric_fpath = f'{results_path}/all_recon_loss_ideal_type{problem_type}_run{run}_cluster.npy'
                    metric = np.load(metric_fpath)[-num_dims : ]
                    if counterbalancing:
                        # rotate dimensions based on `rot_dims`
                        # notice here rotation changes the `meaning` of dimensions
                        # because we want to move the task diagnostic dimension to
                        # always be dim1 for plotting only.
                        counter_balancing_fpath = f'{results_path}/counter_balancing_type{problem_type}_run{run}_cluster.npy'
                        counter_balancing = np.load(counter_balancing_fpath, allow_pickle=True)
                        print(f'counter_balancing = {counter_balancing}')
                        rot_dims = counter_balancing.item()['rot_dims']
                        k = counter_balancing.item()['k'][0]
                        # print(f'run={run}, rot_dims = {rot_dims}, k={k}')
                        if k != 2:
                            # no need to rotate if k == 2 as it is same as k == 0
                            # otherwise just switch based on index.
                            metric[rot_dims[1]], metric[rot_dims[0]] = metric[rot_dims[0]], metric[rot_dims[1]]

                # Second group metric based on attn strategy
                alphas_fpath = f'{results_path}/all_alphas_type{problem_type}_run{run}_cluster.npy'
                # get the final 3 alphas
                alphas = np.load(alphas_fpath)[-3:]
                print(f'alphas = {alphas}')
                alphas = alphas - np.array(threshold)

                # 1e-6 is the lower bound of alpha constraint.
                # use tuple instead of list because tuple is not mutable.
                strategy = tuple(alphas > 1.0e-6)
                type2strategy2metric[problem_type][strategy].append(metric)
                
        # plot
        if comparison == 'zero_attn':
            fig, ax = plt.subplots()
            x_axis = np.linspace(0, 14, 7)

            colors = ['green', 'red', 'brown', 'orange', 'cyan', 'blue']
            average_metrics = []
            std_metrics = []
            for z in range(num_types):
                problem_type = z + 1
                print(f'--------- Type {problem_type} ---------')
                # e.g. {(True, False, False): [metric]}
                strategy2metric = type2strategy2metric[problem_type]
                strategies = list(strategy2metric.keys())
                print(f'strategies = {strategies}')
                num_strategies = len(strategies)
                print(f'num_strategies = {num_strategies}')

                x_left = x_axis[z] + 1
                x_right = x_axis[z+1] - 1
                x_ticks = np.linspace(x_left, x_right, num_strategies)
                
                temp_collector = []
                for i in range(num_strategies):
                    strategy = strategies[i]
                    metrics = strategy2metric[strategy]
                    temp_collector.extend(metrics)
                    # plot bar of a single strategy
                    ax.errorbar(
                        x=x_ticks[i],
                        y=np.mean(metrics),
                        yerr=np.std(metrics),
                        fmt='o',
                        color=colors[z],
                    )
                # NOTE: Hacky way of getting legend correct.
                ax.errorbar(
                    x=x_ticks[i],
                    y=np.mean(metrics),
                    yerr=np.std(metrics),
                    fmt='o',
                    color=colors[z],
                    label=f'single strategy, type {problem_type}'
                )
                average_metrics.append(np.mean(temp_collector))
                std_metrics.append(np.std(temp_collector))

            # plot bar of averaged over strategies.
            ax.errorbar(
                x=x_axis[:num_types]+0.5,
                y=average_metrics,
                yerr=std_metrics,
                fmt='*',
                color='k',
                ls='-',
                label='overall'
            )

            ax.set_xticks(x_axis[:num_types]+0.5)
            ax.set_xticklabels(
                [f'Type {problem_type}' for problem_type in range(1, num_types+1)])
            # ax.set_ylim([0.6, 0.9])
            ax.set_ylabel('percentage of zero attention weights')
            plt.tight_layout()
            plt.legend()
            plt.savefig(f'{results_path}/compare_types_percent_zero_canonical_runs{canonical_runs_only}.png')
            plt.close()

        elif comparison == 'binary_recon':
            num_cols = 2
            num_rows = int(num_types / num_cols)
            x_axis = np.linspace(0, 8, num_dims+1)
            fig, ax = plt.subplots(num_rows, num_cols)

            for z in range(num_types):
                problem_type = z + 1
                row_idx = z // num_cols
                col_idx = z % num_cols

                print(f'--------- Type {problem_type} ---------')
                # e.g. {(True, False, False): [ [metric_dim1, dim2. dim3], [dim1, dim2, dim3], .. ]}
                strategy2metric = type2strategy2metric[problem_type]
                strategies = list(strategy2metric.keys())
                print(f'strategies = {strategies}')
                num_strategies = len(strategies)

                all_strategies_collector = []
                for i in range(num_strategies):
                    strategy = strategies[i]
                    # metrics -> [ [dim1, dim2, dim3], [dim1, dim2, dim3], .. ]
                    metrics = np.array(
                        strategy2metric[strategy])
                    all_strategies_collector.extend(metrics)

                    colors = ['green', 'red', 'brown']
                    for dim in range(num_dims):
                        x_left = x_axis[dim] + 1
                        x_right = x_axis[dim+1] - 1
                        x_ticks = np.linspace(x_left, x_right, num_strategies)

                        ax[row_idx, col_idx].errorbar(
                            x=x_ticks[i],
                            y=np.mean(metrics[:, dim]),
                            yerr=np.std(metrics[:, dim]),
                            fmt='o',
                            color=colors[dim],
                            label=f'single strategy, dim{dim}')

                average_metric = np.mean(
                    np.array(all_strategies_collector), axis=0)
                std_metric = np.std(
                    np.array(all_strategies_collector), axis=0
                )
                ax[row_idx, col_idx].errorbar(
                    x=x_axis[:num_dims],
                    y=average_metric,
                    yerr=std_metric,
                    fmt='*',
                    color='k',
                    label='overall'
                )
                ax[row_idx, col_idx].set_xticks([])
                ax[num_rows-1, col_idx].set_xticks(x_axis[:num_dims]+0.5)
                ax[num_rows-1, col_idx].set_xticklabels([f'dim{i+1}' for i in range(num_dims)])
                ax[row_idx, col_idx].set_ylim([-0.5, 2.5])
                ax[row_idx, 0].set_ylabel('binary recon loss')
                ax[row_idx, col_idx].set_title(f'Type {problem_type}')
            
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(
                f'{results_path}/compare_types_dimension_binary_recon_canonical_runs_only{canonical_runs_only}.png')


def compare_across_types_thru_time_V3(
        attn_config_version, canonical_runs_only=True, counterbalancing=True):
    """
    No longer just plotting the last time step 
    statistic but plotting statistics (avg runs) 
    over time steps and turn into gif to visualize
    the evolution of the two experiments we care about.
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version
    )
    num_runs = attn_config['num_runs']
    num_types = 6
    num_dims = 3
    comparisons = ['zero_attn', 'binary_recon']
    results_path = f'results/{attn_config_version}'
    type2runs = find_canonical_runs(
        attn_config_version, canonical_runs_only=canonical_runs_only)

    # Just to get total steps:
    fpath = f'{results_path}/all_percent_zero_attn_type1_run0_cluster.npy'
    num_steps = len(np.load(fpath))
    print(f'[Check] num_steps = {num_steps}')
    steps = range(1080, num_steps)

    for c in range(len(comparisons)):
        comparison = comparisons[c]
            
        for step in steps:
            print(f'step = [{step}]')            
            # {type: [run1_metric, run2_metric..]}
            types_runs_metrics = defaultdict(list)
                        
            for z in range(num_types):
                problem_type = z + 1
            
                for run in type2runs[z]:
                
                    if comparison == 'zero_attn':
                        fpath = f'{results_path}/all_percent_zero_attn_type{problem_type}_run{run}_cluster.npy'
                        ylabel = 'percentage of zero attn weights'
                        metric = np.load(fpath)[step]
                        
                    elif comparison == 'binary_recon':
                        fpath = f'{results_path}/all_recon_loss_ideal_type{problem_type}_run{run}_cluster.npy'
                        ylabel = 'binary recon loss'
                        metric = np.load(fpath)[step * num_dims : (step+1) * num_dims]
                        
                        if counterbalancing:
                            counter_balancing_fpath = f'{results_path}/counter_balancing_type{problem_type}_run{run}_cluster.npy'
                            counter_balancing = np.load(counter_balancing_fpath, allow_pickle=True)
                            rot_dims = counter_balancing.item()['rot_dims']
                            k = counter_balancing.item()['k'][0]
                            if k != 2:
                                metric[rot_dims[1]], metric[rot_dims[0]] = metric[rot_dims[0]], metric[rot_dims[1]]
                    
                    types_runs_metrics[z].append(metric)                        
            
            if comparison == 'zero_attn':
                fig, ax = plt.subplots()          
        
                type_averages = []
                type_stds = []
                
                for z in range(num_types):
                    type_averages.append(
                        np.mean(types_runs_metrics[z])
                    )
                    type_stds.append(
                        np.std(types_runs_metrics[z])
                    )
                
                # subplot
                print(type_averages)
                ax.errorbar(
                    x=range(len(type_averages)), 
                    y=type_averages,
                    yerr=type_stds)
                ax.set_xlabel(f'Problem Type')
                ax.set_xticks(np.arange(num_types))
                ax.set_xticklabels(range(1, num_types+1))
                ax.set_ylim([0.5, 1])

                plt.suptitle(f'step = {step}')
                plt.savefig(f'{results_path}/gif/comparison_across_types_{comparison}_step{step}.png')
                plt.clf()
    
        # convert png to gif
        png2gif(attn_config_version, comparison)


def compare_cluster_targets_acrossruns_acrosstypes(
        attn_config_version):
    """
    For the same problem type, compare MSE between every
    pair of cluster targets at the same step across runs.
    This can show us how much variation there is in the 
    cluster targets themselves across runs. It can also
    be informative for us to understand attention weights
    correlations across runs (see the following eval.)

    For each step, there is a (mean, std) of MSE across runs.
    For each type, there are all the steps.
    So, the figure is going to have 6 rows, each a type,
    within a row, we have xaxis as steps, yaxis as MSE variations.
    """
    num_types = 6
    attn_config = load_config(
        component=None,
        config_version=attn_config_version)
    num_runs = attn_config['num_runs']
    inner_loop_epochs = attn_config['inner_loop_epochs']
    # 32-1 because epoch=0 does not learn attention.
    global_steps = 8 * 31 * inner_loop_epochs
    print(f'global_steps = {global_steps}')
    loss_fn = tf.keras.losses.MeanSquaredError()
    steps = range(0, global_steps, inner_loop_epochs)

    num_row = 2
    num_col = int(num_types/2)
    fig, ax = plt.subplots(num_row, num_col)
    for z in range(num_types):
        problem_type = z + 1
        row_idx = z // num_col
        col_idx = z % num_col

        # at each step, we compute across run mean MSE by taking 
        # the upper tri of the pairwise mtx (num_runs * num_runs)
        diff_across_steps = np.empty(len(steps))
        for step_idx in range(len(steps)):
            step = steps[step_idx]

            # each run's target is a (batch, num_cluster) mtx
            step_cluster_targets = np.empty((num_runs, 8, 8))
            for run in range(num_runs):
                
                # (batch, num_cluster)
                cluster_targets = np.load(
                    f'results/{attn_config_version}/' \
                    f'cluster_targets_{problem_type}_{step}_{run}.npy'
                )
                step_cluster_targets[run, :, :] = cluster_targets
            
            # for each step, compute mean MSE from upper tri
            step_target_MSE = np.empty((num_runs, num_runs))
            for i in range(num_runs):
                for j in range(num_runs):
                    if i >= j:
                        continue
                    else:
                        cluster_target_i = step_cluster_targets[i, :, :]
                        cluster_target_j = step_cluster_targets[j, :, :]
                        diff = loss_fn(cluster_target_i, cluster_target_j)
                        step_target_MSE[i, j] = diff

            # compute mean for this step
            diff_mean = np.mean(
                step_target_MSE[np.triu_indices(step_target_MSE.shape[0])]
            )
            diff_std = np.std(
                step_target_MSE[np.triu_indices(step_target_MSE.shape[0])]
            )
            # print(f'step={step}, mean={diff_mean}[{diff_std}]')
            # print(f'max={np.max(step_target_MSE[np.triu_indices(step_target_MSE.shape[0])])}')
            # print(f'min={np.min(step_target_MSE[np.triu_indices(step_target_MSE.shape[0])])}')
            diff_across_steps[step_idx] = diff_mean
    
        ax[row_idx, col_idx].plot(diff_across_steps)
        ax[row_idx, col_idx].set_title(f'Type {problem_type}')
        ax[num_row-1, col_idx].set_xlabel(f'Steps')
        ax[row_idx, num_col-1].set_ylabel(f'Average difference across runs (MSE)')
    plt.savefig(f'results/{attn_config_version}/cluster_targets_samestep_acrossruns.png')


def compare_attn_weights_acrossruns_acrosstypes(
        attn_config_version,
        mode='type'):
    """
    Compare attn weights based on `mode`.

    if mode == 'run':
        will eval and plot for each type, the violin plot
        of intersect/r_full/r_intersect across runs.

    if mode == 'type':
        will eval and plot a confusion mtx (type*type) for each metric
        from (intersect/r_full/r_intersect) average across all runs.
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version)
    num_runs = attn_config['num_runs']
    num_types = 6
    all_attn_weights = np.empty((num_types, num_runs, 512))
    
    for i in range(num_types):
        problem_type = i + 1

        for run in range(num_runs):
            attn_weights = np.load(
                f'results/{attn_config_version}/attn_weights_type{problem_type}_run{run}_cluster.npy'
            )
            all_attn_weights[i, run, :] = attn_weights

    # for each type, compute between run similarity.
    # one type points at a run*run matrix.
    if mode == 'run':
        intersect_mtx = np.zeros((num_types, num_runs, num_runs))
        correlation_mtx_full = np.zeros((num_types, num_runs, num_runs))
        correlation_mtx_intersect = np.zeros((num_types, num_runs, num_runs))

        for z in range(num_types):

            for i in range(num_runs):

                for j in range(num_runs):

                    if i >= j:
                        continue
                    else:
                        attn1 = all_attn_weights[z, i, :]
                        attn2 = all_attn_weights[z, j, :]

                        nonzero_filter_i = np.nonzero(attn1)[0]
                        nonzero_filter_j = np.nonzero(attn2)[0]
                        intersect = np.intersect1d(nonzero_filter_i, nonzero_filter_j)
                        intersect_percent = 2 * len(intersect) / ( len(nonzero_filter_i) + len(nonzero_filter_j))
                        intersect_mtx[z, i, j] = intersect_percent

                        r_full, _ = stats.spearmanr(
                            attn1, 
                            attn2
                        )
                        r_intersect, _ = stats.spearmanr(
                            attn1[intersect],
                            attn2[intersect]
                        )
                        correlation_mtx_full[z, i, j] = r_full
                        correlation_mtx_intersect[z, i, j] = r_intersect

        fig, ax = plt.subplots(3)
        metric_names = ['intersect_percentage', 'r_full', 'r_intersect']
        metric_mats = [intersect_mtx, correlation_mtx_full, correlation_mtx_intersect]
        
        for i in range(len(metric_names)):
            data = []
            print(f'-------------------------')
            print(f'[Check] {metric_names[i]}')
            print(f'-------------------------')
            for j in range(num_types):
                # one metric mtx (6, 50, 50)
                mtx = metric_mats[i]
                mtx_slice = mtx[j, :, :]
                mtx_slice = mtx_slice[np.triu_indices(mtx_slice.shape[0])]
                print(f'mean={np.mean(mtx_slice):.3f}[{np.std(mtx_slice):.3f}]')
                data.append(mtx_slice)

            print(f'[Check] len(data) = {len(data)}')
            ax[i].set_title(f'{metric_names[i]}')
            ax[i].set_xticks(range(num_types))
            ax[i].set_xticklabels([1,2,3,4,5,6])
            sns.violinplot(
                data=data, 
                linewidth=.8,
                gridsize=300,
                ax=ax[i]
            )
        plt.tight_layout()
        plt.savefig(f'results/{attn_config_version}/attn_similarity_{mode}.png')

    # for each run, compute between type similarity.
    # one run points at a type*type matrix.
    elif mode == 'type':
        intersect_mtx = np.zeros((num_runs, num_types, num_types))
        correlation_mtx_full = np.zeros((num_runs, num_types, num_types))
        correlation_mtx_intersect = np.zeros((num_runs, num_types, num_types))

        for z in range(num_runs):

            for i in range(num_types):

                for j in range(num_types):

                    if i >= j:
                        continue
                    else:
                        attn1 = all_attn_weights[i, z, :]
                        attn2 = all_attn_weights[j, z, :]

                        nonzero_filter_i = np.nonzero(attn1)[0]
                        nonzero_filter_j = np.nonzero(attn2)[0]
                        intersect = np.intersect1d(nonzero_filter_i, nonzero_filter_j)
                        intersect_percent = 2 * len(intersect) / ( len(nonzero_filter_i) + len(nonzero_filter_j))
                        intersect_mtx[z, i, j] = intersect_percent

                        r_full, _ = stats.spearmanr(
                            attn1, 
                            attn2
                        )
                        r_intersect, _ = stats.spearmanr(
                            attn1[intersect],
                            attn2[intersect]
                        )
                        correlation_mtx_full[z, i, j] = r_full
                        correlation_mtx_intersect[z, i, j] = r_intersect

        # TODO: how to show std in confusion?
        intersect_mtx_average = np.mean(intersect_mtx, axis=0)
        correlation_mtx_full_average = np.mean(correlation_mtx_full, axis=0)
        correlation_mtx_intersect_average = np.mean(correlation_mtx_intersect, axis=0)
        
        metric_names = ['intersect_percentage', 'r_full', 'r_intersect']
        metric_mats = [
            intersect_mtx_average, 
            correlation_mtx_full_average, 
            correlation_mtx_intersect_average
        ]
        fig, ax = plt.subplots(3)
        labels = ['type1', 'type2', 'type3', 'type4', 'type5', 'type6']
        for i in range(len(metric_names)):
            mtx = metric_mats[i]
            ax[i].set_title(f'{metric_names[i]}')
            sns.heatmap(
                mtx, 
                annot=True,
                xticklabels=labels,
                yticklabels=labels,
                ax=ax[i]
            )
        plt.tight_layout()
        plt.savefig(f'results/{attn_config_version}/attn_similarity_{mode}_mean.png')
        plt.close()

        # TODO: clean, temp for std.
        intersect_mtx_average = np.std(intersect_mtx, axis=0)
        correlation_mtx_full_average = np.std(correlation_mtx_full, axis=0)
        correlation_mtx_intersect_average = np.std(correlation_mtx_intersect, axis=0)
        
        metric_names = ['intersect_percentage', 'r_full', 'r_intersect']
        metric_mats = [
            intersect_mtx_average, 
            correlation_mtx_full_average, 
            correlation_mtx_intersect_average
        ]
        fig, ax = plt.subplots(3)
        labels = ['type1', 'type2', 'type3', 'type4', 'type5', 'type6']
        for i in range(len(metric_names)):
            mtx = metric_mats[i]
            ax[i].set_title(f'{metric_names[i]}')
            sns.heatmap(
                mtx, 
                annot=True,
                xticklabels=labels,
                yticklabels=labels,
                ax=ax[i]
            )
        plt.tight_layout()
        plt.savefig(f'results/{attn_config_version}/attn_similarity_{mode}_std.png')


def compare_alt_cluster_actv_targets(
        original, 
        alt):
    """
    We compare, overtime, how cluster targets 
    change between 
        1. original version where sustain is updated
            using init attn 
        2. alt version where sustain is updated 
            using latest attn.
    
    # Fig: Compare targets using MSE loss wrt overall.
    """
    attn_config = load_config(
        component=None,
        config_version=original)
    num_runs = attn_config['num_runs']
    num_problem_types = 6
    inner_loop_epochs = attn_config['inner_loop_epochs']
    global_steps = 8 * 31 * inner_loop_epochs
    steps = range(0, global_steps, inner_loop_epochs)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # plotting a 2 * 3 figure
    num_rows = 2
    num_cols = int(num_problem_types / num_rows)
    fig, ax = plt.subplots(num_rows, num_cols, dpi=500)

    for i in range(num_problem_types):
        problem_type = i + 1
        print(f'subplotting problem_type={problem_type}...')
        row_idx = i // num_cols
        col_idx = i % num_cols

        # one subplot
        loss_overtime_overruns = np.empty((num_runs, len(steps)))
        for run in range(num_runs):
            for step_idx in range(len(steps)):
                step = steps[step_idx]
                stats_ori = np.load(f'results/{original}/cluster_targets_{problem_type}_{step}_{run}.npy')
                stats_alt = np.load(f'results/{alt}/cluster_targets_{problem_type}_{step}_{run}.npy')
                loss = loss_fn(stats_ori, stats_alt)
                loss_overtime_overruns[run, step_idx] = loss

        loss_overtime_avg = np.mean(loss_overtime_overruns, axis=0)
        loss_overtime_std = np.std(loss_overtime_overruns, axis=0)
        ax[row_idx, col_idx].plot(loss_overtime_avg)
        ax[row_idx, col_idx].set_title(f'Type {problem_type}')
        ax[row_idx, col_idx].set_xticks(range(0, len(steps), 5 * 8))
        ax[row_idx, col_idx].set_xticklabels(range(0, 31, 5))

    fig.supxlabel('num of epochs')
    fig.supylabel('cluster target difference (MSE)')
    plt.suptitle(f'{original} vs {alt}')
    plt.tight_layout()
    plt.savefig(f'results/{alt}/{original}-{alt}.png')
    plt.close()


def how_cluster_targets_change_overtime(
        attn_config_version):
    """
    Under `ideal`, we visualize how cluster activation targets
    vary between trials in terms of MSE loss across all problem
    types across runs.

    Notice, this is different from `compare_alt_cluster_actv_targets`
    above whose purpose is to compare cluster targets BETWEEN 
    two configs (ideal vs latest) at the SAME step.
    """
    num_problem_types = 6
    attn_config = load_config(
        component=None,
        config_version=attn_config_version)
    num_runs = attn_config['num_runs']
    inner_loop_epochs = attn_config['inner_loop_epochs']
    # 32-1 because epoch=0 does not learn attention.
    global_steps = 8 * 31 * inner_loop_epochs
    print(f'global_steps = {global_steps}')
    loss_fn = tf.keras.losses.MeanSquaredError()
    steps = range(0, global_steps-inner_loop_epochs, inner_loop_epochs)

    # plotting a 2 * 3 figure
    num_rows = 2
    num_cols = int(num_problem_types / num_rows)
    fig, ax = plt.subplots(num_rows, num_cols, dpi=500)

    for i in range(num_problem_types):
        problem_type = i + 1
        print(f'subplotting problem_type={problem_type}...')
        row_idx = i // num_cols
        col_idx = i % num_cols

        # one subplot
        loss_overtime_overruns = np.empty((num_runs, len(steps)))
        for run in range(num_runs):
            for step_idx in range(len(steps)):
                step = steps[step_idx]
                step_ = step + inner_loop_epochs
                trg1 = np.load(f'results/{attn_config_version}/cluster_targets_{problem_type}_{step}_{run}.npy')
                trg2 = np.load(f'results/{attn_config_version}/cluster_targets_{problem_type}_{step_}_{run}.npy')

                loss = loss_fn(trg1, trg2)
                loss_overtime_overruns[run, step_idx] = loss

        loss_overtime_avg = np.mean(loss_overtime_overruns, axis=0)
        loss_overtime_std = np.std(loss_overtime_overruns, axis=0)
        ax[row_idx, col_idx].plot(loss_overtime_avg)
        # ax[row_idx, col_idx].errorbar(
        #     range(len(loss_overtime_avg)), 
        #     loss_overtime_avg, 
        #     yerr=loss_overtime_std, 
        #     marker='.')
        ax[row_idx, col_idx].set_title(f'Type {problem_type}')
    fig.supxlabel('num of trials')
    fig.supylabel('change of cluster targets (MSE)')
    plt.tight_layout()
    plt.savefig(f'results/{attn_config_version}/change_of_cluster_targets.png')


def turned_off_attn_across_positions(
        attn_config_version, format_='V1', canonical_runs_only=True):
    """
    Examine percentage of zeroed out attn weights
    across different `attn_positions`, across different
    `Type`
    
    format_1: xaxis represent types
    format_2: xaxis represent positions
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version)
    attn_positions = attn_config['attn_positions'].split(',')
    num_positions = len(attn_positions)
    num_types = 6
    type2runs = find_canonical_runs(
        attn_config_version, canonical_runs_only=canonical_runs_only)
    results_path = f'results/{attn_config_version}'

    fig, ax = plt.subplots()
    colors = ['green', 'red', 'brown', 'orange', 'cyan', 'blue']
    
    if format_ == 'V1':
        x_axis = np.linspace(0, 14, num_types+1)
        
        type_run_percent = defaultdict(list)
        for z in range(num_types):
            problem_type = z + 1
            runs = type2runs[z]        
            num_runs = len(runs)
            type_run_position_percent = np.empty((num_runs, num_positions))
            markers = ['o', 's', 'P', '>']
            
            for run_idx in range(num_runs):
                run = runs[run_idx]
                
                zero_counts = 0
                tot_counts = 0
                for position_idx in range(num_positions):
                    attn_position = attn_positions[position_idx]
                    fpath = f'{results_path}/' \
                            f'attn_weights_type{problem_type}_' \
                            f'run{run}_cluster.npy'
                    attn_weights = np.load(fpath, allow_pickle=True).item()
                    layer_attn_weights = attn_weights[attn_position]
                    
                    # percent for this layer this run.                    
                    percent_zero = (
                        len(layer_attn_weights) - len(np.nonzero(layer_attn_weights)[0])
                        ) / len(layer_attn_weights)
                    type_run_position_percent[run_idx, position_idx] = percent_zero
                    
                    # collect percent for all layers
                    zero_counts += (
                        len(layer_attn_weights) - len(np.nonzero(layer_attn_weights)[0])
                    )
                    tot_counts += len(layer_attn_weights)
                
                fpath = f'{results_path}/all_percent_zero_attn_type{problem_type}_run{run}_cluster.npy'
                type_run_percent[z].append(np.load(fpath)[-1])
                # type_run_percent[z].append(zero_counts / tot_counts)
                
            x_left = x_axis[z] + 1
            x_right = x_axis[z+1] - 1
            x_ticks = np.linspace(x_left, x_right, num_positions)

            for position_idx in range(num_positions):
                
                # HACK: to get legend
                if z == 0:
                    label = f'layer: {attn_positions[position_idx]}'
                else:
                    label = None
                
                ax.errorbar(
                    x=x_ticks[position_idx],
                    y=np.mean(type_run_position_percent[:, position_idx], axis=0),
                    yerr=np.std(type_run_position_percent[:, position_idx], axis=0),
                    fmt=markers[position_idx],
                    color=colors[z],
                    label=label
                )
        
        # in order to plot the overall trend, needs to 
        # collect and plot all Types at once    
        averages = []
        stds = []
        for z in range(num_types):
            averages.append(
                np.mean(type_run_percent[z])
            )
            stds.append(
                np.std(type_run_percent[z])
            )
            
        ax.errorbar(
            x=x_axis[:num_types]+0.5,
            y=averages,
            yerr=stds,
            fmt='*',
            label='overall',
            color='k',
            ls='-'
        )
                
        ax.set_xticks(x_axis[:num_types]+0.5)
        ax.set_xticklabels(
            [f'Type {problem_type}' for problem_type in range(1, num_types+1)])
    
    
    elif format_ == 'V2':
        pass
            
    ax.set_ylabel('percentage of zero attention weights')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'{results_path}/compare_types_layerwise_percent_zero_canonical_runs{canonical_runs_only}_format{format_}.png')
    plt.close()
    
    
def subject_dimension_rt_acc():
    """
    Look at subject response time when different 
    dimensions are being diagnostic. This is to see
    if subjects are particularly slow when one of the 
    dimensions is being task relevant.

    Impl:
    -----
        Look at Type 1 only when we know the first dim
        is always the diag dimension regardless of its
        physical meaning. 
        
        We group subjects based on the physical dimension
        they use to solve Type 1. So we can compare RT 
        across different diag dimensions. 
        
    Conclusion:
    -----------
        When the 3rd physical dimension is used as the
        diagnostic dimension, the overall subjects RT is 
        the slowest. 
        dim1:    1060715.5806451612
        dim2:    1181767.4193548388
        dim3:    1366676.3631442343
    """

    # study1: type 6
    # study2: type 1
    # study3: type 2

    # 1. trial number (1-32)
    # 2. task (1-3)
    # 3. run (1-4)
    # 4. dimension 1
    # 5. dimension 2
    # 6. dimension 3
    # 7. answer (1 or 2)
    # 8. response (1 or 2)
    # 9. RT (ms)
    # 10. accuracy (0 or 1)

    runs = [1, 2, 3, 4]
    # load subject's dimension assignment,
    # i.e. each subject's dimension 1 maps to different physical dims.
    subject2dimension = pd.read_csv('featureinfo.txt', header=None).to_numpy()
    subject_ids_temp = subject2dimension[:, 0]

    # convert `1` to `01` etc.
    subject_ids = []
    for subject_id in subject_ids_temp:
        if len(f'{subject_id}') == 1:
            subject_ids.append(f'0{subject_id}')
        else:
            subject_ids.append(f'{subject_id}')

    # For Type 1, diagnostic dimension is always dimenson 1
    # only its physical meaning changes.
    diagnostic_dimension = subject2dimension[:, 1]

    # int here is idx of physical dim not dim1,2,3
    dim2meaning = {
        '1': 'leg',
        '2': 'mouth',
        '3': 'manible'
    }

    # group subjects that use the same physical dim as 
    # the diagnostic dimension.
    # dict = {1: ['02', '03'], 2: [], 3: []}
    dim2subject = defaultdict(list)
    for i in range(subject2dimension.shape[0]):
        dim = diagnostic_dimension[i]
        subject_id = subject_ids[i]
        dim2subject[dim].append(subject_id)


    # for each diag dim, subjects who use that dim 
    # will be used to sum up their RT through trials and runs,
    
    dim2rt = defaultdict(list)
    dim2acc = defaultdict(list)
    for dim in [1, 2, 3]:
        subject_ids = dim2subject[dim]
        
        for subject_id in subject_ids:
            data_dir = f'behaviour/subject_{subject_id}'
            
            # even subject Type 1 - study_2
            # odd subject Type 1 - study_3
            if int(subject_id) % 2 == 0:
                study_id = 2
            else:
                study_id = 3

            subject_rt = []
            # subject_correct = 0
            # subject_wrong = 0
            for run in runs:
                # print(f'------ dim={dim}, subject={subject_id}, run={run} ------')
                
                subject_correct = 0
                subject_wrong = 0
            
                fpath = f'{data_dir}/{subject_id}_study{study_id}_run{run}.txt'
                data = pd.read_csv(fpath, sep = "\t", header=None).to_numpy()
                    
                # locate nan RT within a run
                nan_i = []
                has_nan = False 
                data_exc_nan = []        
                for i in range(data.shape[0]):
                    rt = data[i, 8]
                    if np.isnan(rt):
                        nan_i.append(i)
                        has_nan = True
                    else:
                        data_exc_nan.append(rt)
                        
                # replace nan to the average RT of the rest of the trials.
                if has_nan:
                    for j in nan_i:
                        data[j, 8] = np.mean(data_exc_nan)
                
                # collect trials' RT of this run.
                subject_rt.extend(data[:, 8])
                
                # count num of correct and wrong this run.
                for acc in data[:, 9]:
                    if acc == 0:
                        subject_wrong += 1
                    else:
                        subject_correct += 1
                
                dim2acc[dim].append(subject_correct / (subject_correct + subject_wrong))
                
            dim2rt[dim].extend(subject_rt)
            
            
    # plot dim2rt
    fig, ax = plt.subplots(2)
    data_RT = []
    data_ACC = []
    RT_medians = []
    RT_means = []
    for dim in [1, 2, 3]:
        rt = dim2rt[dim]  # rt = [32 * 4 * 8] or [32 * 4 * 7]
        acc = dim2acc[dim]
        data_RT.append(rt)
        data_ACC.append(acc)
        RT_medians.append(np.median(rt))
        RT_means.append(np.mean(rt))
        
    assert len(data_RT) == 3

    # ------ statistic testing ------
    from scipy.stats import median_test, ttest_ind
    print(f'------ RT statistic testing ------')
    stats, p = median_test(data_RT[0], data_RT[1])[:2]
    print('dim1 vs dim2: ', f'stats={stats}, p={p}')
    
    stats, p = median_test(data_RT[0], data_RT[2])[:2]
    print('dim1 vs dim3: ', f'stats={stats}, p={p}')
    
    stats, p = median_test(data_RT[1], data_RT[2])[:2]
    print('dim2 vs dim3: ', f'stats={stats}, p={p}')
    
    print(f'median RT = {RT_medians}')
    print(f'mean RT = {RT_means}')
    
    print(f'\n ------ Acc statistic testing ------')
    stats, p = ttest_ind(data_ACC[0], data_ACC[1])[:2]
    print('dim1 vs dim2: ', f'stats={stats}, p={p}')
    
    stats, p = ttest_ind(data_ACC[0], data_ACC[2])[:2]
    print('dim1 vs dim3: ', f'stats={stats}, p={p}')
    
    stats, p = ttest_ind(data_ACC[1], data_ACC[2])[:2]
    print('dim2 vs dim3: ', f'stats={stats}, p={p}')
        
    
    # ------ plotting ------
    sns.violinplot(data=data_RT, ax=ax[0], inner=None)
    ax[0].set_xticks(range(len(data_RT)))
    ax[0].set_xticklabels(['dim1: leg', 'dim2: antenna', 'dim3: mandible'])
    ax[0].set_ylabel('Subject RT')
    ax[0].scatter(range(len(data_RT)), RT_medians, color='k', marker='*')
    
    sns.violinplot(data=data_ACC, ax=ax[1])
    ax[1].set_xticks(range(len(data_ACC)))
    ax[1].set_xticklabels(['dim1: leg', 'dim2: antenna', 'dim3: mandible'])
    ax[1].set_ylabel('Subject accuracy')
    
    plt.tight_layout()
    plt.savefig('subject_dimension_rt_acc.png')
        

def post_attn_actv_thru_time(attn_config_version):
    """
    Capture and analyze each Type's post-attn activations
    over the course of learning for each position of attn.

    Results from this eval will be compared to brain data.
    Notice only Type 1 & 2 are needed due to brain data availability.

    Impl:
    -----
        Given a `Type`, given a `attn_position`, we plug in 
        (epoch,i) level `attn_weights` to get post-attn actv 
        for all stimuli.
    """
    attn_config = load_config(
        component=None,
        config_version=attn_config_version)
    dcnn_config_version = attn_config['dcnn_config_version']
    attn_positions = attn_config['attn_positions'].split(',')
    num_runs = attn_config['num_runs']
    num_blocks = attn_config['num_blocks']
    num_positions = len(attn_positions)
    num_types = 2
    num_stimuli = 8
    results_path = f'results/{attn_config_version}'
    
    # entire DCNN model with all attn layers
    model, preprocess_func = load_dcnn_model_V2(
        attn_config_version=attn_config_version, 
        dcnn_config_version=dcnn_config_version, 
        intermediate_input=False)

    for z in range(num_types):
        problem_type = z + 1

        for run in range(num_runs):
            
            dataset, _ = data_loader_V2(
                attn_config_version=attn_config_version,
                dcnn_config_version=dcnn_config_version,
                preprocess_func=preprocess_func,
                problem_type=problem_type,
                random_seed=run
            )
            
            batch_x = load_X_only(
                dataset=dataset, 
                attn_config_version=attn_config_version
            )

            # remember we do not have attn saved for epoch=0
            # tho easy to get as its just init.
            for epoch in range(1, num_blocks):
                
                for i in range(num_stimuli):
                                            
                    # set (epoch,i) level attn_weights 
                    fpath = f'{results_path}/' \
                            f'attn_weights_type{problem_type}_' \
                            f'run{run}_epoch{epoch}_i{i}_cluster.npy'
                    # attn_weights = {'layer': [weights], }
                    attn_weights = np.load(fpath, allow_pickle=True).item()
                    
                    for position_idx in range(num_positions):
                        attn_position = attn_positions[position_idx]
                        
                        print(f'\nType={problem_type}, run={run}, {attn_position}, epoch={epoch}, i={i}')
                        
                        # intercept DCNN at an attn position.
                        # and use this partial model to get 
                        # post-attn actv across learning course.
                        outputs = model.get_layer(f'post_attn_actv_{attn_position}').output
                        partial_model = Model(inputs=model.inputs, outputs=outputs)
                    
                        # NOTE: when eval involve multiple attn layers,
                        # we must also load all previous attn layer weights.
                        positions_require_loading = attn_positions[ : position_idx+1]
                        print(f'positions_require_loading = {positions_require_loading}')
                        
                        for position in positions_require_loading:
                            
                            layer_attn_weights = attn_weights[position]
                            partial_model.get_layer(
                                f'attn_factory_{position}').set_weights(
                                [layer_attn_weights]
                            )
                            
                        # get post-attn actv at this (epoch, i)
                        # shape -> (8, H, W, C)
                        post_attn_actv = partial_model(batch_x, training=False)
                        
                        
                        
                        print(
                            post_attn_actv.numpy().flatten()[
                                np.argsort(
                                    post_attn_actv.numpy().flatten()
                                    )[::-1][:5]])
                        
                    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    attn_config_version = 'v2_independent'
    # for problem_type in [1]:
    #     for run in [0]:
    #         viz_losses(
    #             attn_config_version=attn_config_version,
    #             problem_type=problem_type,
    #             recon_level='cluster',
    #             run=run
    #         )

    compare_across_types_V3(
        attn_config_version,
        canonical_runs_only=True,
        threshold=[0.1, 0.1, 0.1]   # NOTE: non-diagostic dims not abs zero.
    )

    examine_clustering_learning_curves(attn_config_version)

    # compare_alt_cluster_actv_targets(
    #     original='v1_independent', 
    #     alt='v1_independent-partial'
    # )
    
    # how_cluster_targets_change_overtime(attn_config_version)
    # compare_cluster_targets_acrossruns_acrosstypes(attn_config_version)
    