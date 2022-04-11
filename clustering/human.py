import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from utils import load_config


# signatures are needed for cluster recruitment 
# due to static graph TF implementation.
binary_to_signature = {
    '000': 0,
    '001': 1,
    '010': 2,
    '011': 3,
    '100': 4,
    '101': 5,
    '110': 6,
    '111': 7
}


# human experiment ground true labels are 1,2
# we need to convert to one-hot for model learning.
label_to_oneHot = {
    '1': [1, 0], 
    '2': [0, 1]
}


class Mappings(object):
    def __init__(self, num_subs=23):
        # fixed mapping between binary coding and physical dims
        # it's fixed because DCNN finetuned is unique.
        dcnn_mapping = {
            0: {'0': 'thin leg', '1': 'thick leg'},
            1: {'0': 'thick antenna', '1': 'thin antenna'},
            2: {'0': 'pincer mandible', '1': 'shovel mandible'}
        }

        # '12' and '21' are coding scheme
        # 12: is the original coding scheme that is 
        # the same as the provided stimulus set coding.
        # 21: is the flipped coding scheme that is 
        # the opposite as the provided stimulus set coding.
        coding_scheme = {
            12: {'0': '0','1': '1'},
            21: {'0': '1','1': '0'}
        }

        behaviour_columns = {
            0: 'trial',
            1: 'task',
            2: 'run',
            3: 'dim1',
            4: 'dim2',
            5: 'dim3',
            6: 'answer',
            7: 'response',
            8: 'RT',
            9: 'accuracy'
        }

        trial_columns = {
            0: 'subject',
            1: 'trial',
            2: 'task',
            3: 'run',
            4: 'stimulus onset',
            5: 'feedback onset'
        }
        
        sub2assignment_n_scheme = {
            '02': [2,1,3,12,12,12],
            '03': [3,1,2,12,12,12],
            '04': [1,2,3,21,21,12],
            '05': [3,2,1,12,12,21],
            '06': [3,1,2,21,12,21],
            '07': [3,1,2,12,21,12],
            '08': [3,2,1,21,12,12],
            '09': [1,2,3,12,21,21],
            '10': [2,3,1,12,12,12],
            '11': [1,3,2,21,12,21],
            '12': [3,2,1,12,12,21],
            '13': [1,2,3,21,21,21],
            '14': [2,3,1,12,12,21],
            '15': [1,2,3,12,12,21],
            '16': [2,3,1,12,21,21],
            '17': [3,1,2,12,12,21],
            '18': [2,1,3,21,21,12],
            '19': [2,1,3,21,12,12],
            '20': [3,1,2,12,12,12],
            '21': [1,3,2,21,21,12],
            '22': [1,3,2,21,12,21],
            '23': [2,3,1,12,21,12],
            '24': [2,1,3,12,21,21],
        }
        
        self.dcnn_mapping = dcnn_mapping
        self.coding_scheme = coding_scheme
        self.behaviour_columns = behaviour_columns
        self.trial_columns = trial_columns
        self.sub2assignment_n_scheme = sub2assignment_n_scheme


def convert_dcnnCoding_to_subjectCoding(sub):
    """
    In order to extract stimulus-specific activations (for RSA later), 
    we need to first establish the mapping between stimulus coding of 
    DCNN (which is fixed due to finetuning) and stimulus coding of 
    subjects (which differs for each subject due to random scheme).
    
    Since the ultimate purpose is to find brain activation of each stimulus
    (in terms of DCNN coding), we need to find the coding used by each subject
    of every stimulus.
    
    e.g. In terms of DCNN coding, stimulus 000 (thin leg, thick antenna, pincer mandible)
    so what is the coding for this subject? i.e. thin leg=? thick antenna=? pincer mandible=?
    For different subs, this coding is different, depending on '12' or '21' random scheme: 
            
    e.g. if DCNN=101, 
            with assignment 213 and scheme 12, 12, 12
            101 -> 011 -> 011
            
            with assignment 312 and scheme 12, 21, 12
            101 -> 110 -> 100
            
    return:
    -------
        Given all DCNN stimuli, return the conversion ordering for a given subject.
        E.g. for subject a, the orders of the stimuli should be [6, 1, 7, 5, 4, 2, 3, 0]
        where 6 corresponds to 000 in DCNN coding but 110 in subject coding.
    """
    sub2assignment_n_scheme = Mappings().sub2assignment_n_scheme
    coding_scheme = Mappings().coding_scheme
    
    conversion_ordering = []
    stimulus2order_mapping = {
        '000': 0, '001': 1, '010': 2, '011': 3,
        '100': 4, '101': 5, '110': 6, '111': 7,
    }
    for dcnn_stimulus in ['000', '001', '010', '011', '100', '101', '110', '111']:
        sub_stimulus = [i for i in dcnn_stimulus]
        # print(f'\n\n--------------------------------')
        # print(f'[Check] DCNN stimulus {sub_stimulus}')
        
        # assignment (flip three dims)
        assignment_n_scheme = sub2assignment_n_scheme[sub]
        new_stimulus_0 = sub_stimulus[assignment_n_scheme[0]-1]
        new_stimulus_1 = sub_stimulus[assignment_n_scheme[1]-1]
        new_stimulus_2 = sub_stimulus[assignment_n_scheme[2]-1]
        sub_stimulus[0] = new_stimulus_0
        sub_stimulus[1] = new_stimulus_1
        sub_stimulus[2] = new_stimulus_2
        # print(f'[Check] sub{sub}, assignment stimulus {sub_stimulus}')
        
        # scheme (flip binary codings)
        dim1_scheme = assignment_n_scheme[3]
        dim2_scheme = assignment_n_scheme[4]
        dim3_scheme = assignment_n_scheme[5]
        sub_stimulus[0] = coding_scheme[dim1_scheme][sub_stimulus[0]]
        sub_stimulus[1] = coding_scheme[dim2_scheme][sub_stimulus[1]]
        sub_stimulus[2] = coding_scheme[dim3_scheme][sub_stimulus[2]]
        # print(f'[Check] sub{sub}, scheme stimulus {sub_stimulus}')
        
        conversion_ordering.append(
            stimulus2order_mapping[
                ''.join(sub_stimulus)
            ]
        )
        
    return np.array(conversion_ordering)


def reorder_RDM_entries_into_chunks():
    """
    For each subject, the groud true label for each stimulus coding is different.
    When visualising RDMs of conditions, we want to make sure that
    rows and columns are grouped (in chunk) by their labels. This asks
    for a mapping from each subject's stimulus coding to their labels in each
    task.
    
    Impl:
    -----
        We create a dictionary like:
        
        mapping = {sub1: 
                    {task1: [orderOf(000), orderOf(001), ...], 
                     task2: [orderOf(000), orderOf(001), ...], 
                    ...
        
        Notice, within each task (across runs), the labels are the same so we can
        simply use run1.
        
        Notice, there is one extra conversion inside to get the orderOf(..). 
        That is, after we get a list of labels correspond to 000, 001, ..., 111, which look 
        like [1, 2, 2, 1, 2, ...], we argsort them to get indices which we will return as 
        `conversion_ordering` that is going to reorder the RDM entries into desired chunks.
    
    return:
    -------
        `mapping` explained above
    """
    mapping = defaultdict(lambda: defaultdict(list))
    num_subs = 23
    tasks = [1, 2, 3]
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    stimuli = ['000', '001', '010', '011', '100', '101', '110', '111']
    behaviour_path = f'behaviour'
    for sub in subs:
        for task in tasks:
            
            behaviour = pd.read_csv(
                f'{behaviour_path}/subject_{sub}/{sub}_study{task}_run1.txt', 
                header=None
            ).to_numpy()
            
            i = 0
            temp_mapping = dict()
            # search exactly all stimuli once and stop.
            while len(temp_mapping.keys()) != len(stimuli):
                behaviour_i = behaviour[i][0].split('\t')
                stimulus = ''.join(behaviour_i[3:6])
                label = behaviour_i[6]  # 6 - ground true answer
                # print(f'stimulus = {stimulus}, label = {label}')
                temp_mapping[stimulus] = int(label)
                i += 1
                        
            # this is to reorder the stimuli as 000, 001, ..., 111
            # so the corresponding list of labels match the order.
            labels = []
            for stimulus in stimuli:
                labels.append(temp_mapping[stimulus])
                
            # sort the labels and get indices in asc order
            grouped_labels_indices = np.argsort(labels)
            # print(f'sub{sub}, task{task}, labels={labels}, order={grouped_labels_indices}')
            # print('----------------------------------------------------------------------')
            mapping[sub][task].extend(grouped_labels_indices)

    # mapping[sub][task] = a list of indices that will be used to sort the RDM entries.
    return mapping


def load_behaviour(problem_type, sub, repetition):
    """
    Given (problem_type, sub, repetition), return 
    all the behaviours of a particular repetition as ndarray.
    
    Reminder: 1 run has 4 repetitions. We need to convert repetition
    first to run in order to access the behaviour data. We then need
    repetition to extract 8 trials of a particular repetition.
    """
    # Human experiment run starts from 1 not 0.
    run = repetition // 4 + 1
    
    # even sub: Type1 is task2, Type2 is task3
    if int(sub) % 2 == 0:
        if problem_type == 1:
            task = 2
        elif problem_type == 2:
            task = 3
        else:
            task = 1
            
    # odd sub: Type1 is task3, Type2 is task2
    else:
        if problem_type == 1:
            task = 3
        elif problem_type == 2:
            task = 2
        else:
            task = 1
            
    # a list of lists, where each list is a trial in a run (32 trials)
    behaviour = pd.read_csv(
        f'behaviour/subject_{sub}/{sub}_study{task}_run{run}.txt', 
        header=None
    ).to_numpy()
    
    # get behaviour of one repetition (8 trials)
    trial_begin = repetition % 4 * 8
    trial_end = repetition % 4 * 8 + 8
    behaviour = behaviour[trial_begin : trial_end]
    return behaviour
    

def load_data_human_order(problem_type, sub, repetition):
    """
    Given a problem type, a subject and a repetition (16 total),
    return a dataset that contains data-points of that repetition,
    which are the exact 8 stimuli in a random order. The data-points 
    include stimuli representations, labels and signatures.
    
    What makes things a bit complicated is that within each fMRI run,
    there are 32 trials which need to be further divided into 4 repetitions.
    
    return:
    -------
        An array of data-points of all trials within a repetition.
    """
    behaviour = load_behaviour(
        problem_type=problem_type, 
        sub=sub,
        repetition=repetition
    )
    
    dataset = []
    for i in range(len(behaviour)):
        behaviour_i = behaviour[i][0].split('\t')
        stimulus = behaviour_i[3:6]                         # ['0', '0', '1']
        signature = binary_to_signature[''.join(stimulus)]            
        stimulus = [int(d) for d in stimulus]               # [0, 0, 1]
        label = behaviour_i[6]                              # 6 - ground true answer
        label = label_to_oneHot[label]
        # compile a data-point into a dataset
        dp_i = [ [stimulus], [label], signature ]
        dataset.append(dp_i)
    
    return np.array(dataset, dtype=object)


def per_repetition_performance(problem_type, sub, repetition):
    """
    Return the average of `item_proberror` of a given repetition.
    I.e. the accuracy of a given repetition
    This is to be consistent with model performance evaluation.
    """
    behaviour = load_behaviour(
        problem_type=problem_type, 
        sub=sub, 
        repetition=repetition
    )
    
    repetition_sum = 0
    for i in range(len(behaviour)):
        behaviour_i = behaviour[i][0].split('\t')
        item_proberror = 1 - int(behaviour_i[9])   # use item_proberror to be consistent with model
        repetition_sum += item_proberror
    
    per_repetition_accuracy = repetition_sum / 8.
    return per_repetition_accuracy


def overall_performance(config_version='human'):
    """
    Compute lc out of human data.
    This is meant to be in the same logic procedure
    as `train_model` in `main.py`
    """
    results_path = f'results/{config_version}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    config = load_config(config_version)
    num_subs = config['num_subs']
    num_repetitions = config['num_repetitions']
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    
    for sub in subs:
        if int(sub) % 2 == 0:
            problem_types = [6, 1, 2]
        else:
            problem_types = [6, 2, 1]
            
        for problem_type in problem_types:
            lc = np.empty(num_repetitions)
            for repetition in range(num_repetitions):      
                per_repetition_accuracy = per_repetition_performance(
                    problem_type=problem_type, 
                    sub=sub, 
                    repetition=repetition
                )
                lc[repetition] = per_repetition_accuracy
        
            np.save(f'{results_path}/lc_type{problem_type}_sub{sub}.npy', lc)
    
    # average over subjects per problem_type 
    # so we can plot the average lc.
    for problem_type in problem_types:
        lc = []
        
        for s in range(num_subs):
            sub = subs[s]
            lc.append(np.load(f'{results_path}/lc_type{problem_type}_sub{sub}.npy'))
        
        avg_lc = np.mean(lc, axis=0)
        std_lc = np.std(lc, axis=1)
        np.save(f'{results_path}/lc_type{problem_type}_avg.npy', avg_lc)
        np.save(f'{results_path}/lc_type{problem_type}_std.npy', std_lc)
        

def smooth_lc(config_version, problem_type):
    # TODO: figure out how was Mike plotting the smooth lc.
    # although maybe less important as we still fine tune based 
    # per subject. It was only for presentation fig1c
    results_path = f'results/{config_version}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    X = []
    Y = []
    config = load_config(config_version)
    num_subs = config['num_subs']
    num_repetitions = config['num_repetitions']
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    
    for s in range(num_subs):
        sub = subs[s]
        X.extend(range(1, num_repetitions+1))
        Y.extend(np.load(f'{results_path}/lc_type{problem_type}_sub{sub}.npy'))

    clf = LogisticRegression(random_state=999).fit(np.array(X).reshape(-1, 1), Y)
    coef_ = clf.coef_
    intercept_ = clf.intercept_
    print(intercept_)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # load_data_human_order(problem_type=6, sub='02', repetition=15)
    # overall_performance()
    
    smooth_lc(config_version='human', problem_type='6')