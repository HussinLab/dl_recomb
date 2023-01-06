import sys
#sys.path.append("../../code")

import argparse
from google.cloud import storage
import hypertune

from .configs import *
import glob
import os
import sys
import pandas as pd
from io import StringIO
from functools import partial
import tensorflow as tf
from tensorflow.keras.layers import *

import h5py
import json
import pickle
import numpy as np

#trainer.code.dataloading
from .code.dataloading import train_test_splits
from .code.dataloading.dataloading import load_and_create_dictionary

import multiprocessing
import time

from .kfold_utils import multiprocess_fit
############

#output_name = "AA1"
#seq_len = 3500
#batch_size = 128
#random_seed = 123
chroms_to_use = list(range(1,23))
############

parser = argparse.ArgumentParser()

parser.add_argument('--job-dir',  # Handled automatically by AI Platform
                    help='GCS location to write checkpoints and export models',
                    required=True)
#------------- Experiment args
parser.add_argument('--output_name',  # Specified in the config file
                    help='output_name is the person, so AA1, AA2 ..etc',
                    nargs='+',
                    required=True)
parser.add_argument('--seq_len',  # Specified in the config file
                    help='There are multiple seq_len datasets, e.g. 800, 2000, 3500',
                    default=3500,
                    type=int)
parser.add_argument('--fold_fn_name',  # Specified in the config file
                    help='Which split to use, e.g. whole_chroms, partial_chrom_contig, partial_chrom_random, by_random',
                    required=True)
parser.add_argument('--random_seed',  # Specified in the config file
                    help='random_seed',
                    default=123,
                    type=int)
parser.add_argument('--batch_size',  # Specified in the config file
                    help='batch_size',
                    default=128,
                    type=int)
parser.add_argument('--epochs',  # Specified in the config file
                    help='epochs',
                    default=100,
                    type=int)
#-- input args
parser.add_argument('--use_rev_compl',  # Specified in the config file
                    help='Pass to the model the sequence reverse complement',
                    default='True',
                    type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument('--chrom_idx',  # Specified in the config file
                    help='pass to the model the chromosome ID',
                    default='True',
                    type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument('--midpoint',  # Specified in the config file
                    help='Pass to the model the location of the sequence on the chromosome',
                    default='True',
                    type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument('--use_x_y',  # Specified in the config file
                    help='whether to use chromosomes x and y or not in the training set',
                    default='True',
                    type=lambda x: (str(x).lower() in ['true','1', 'yes']))

#-- kfold params
parser.add_argument('--n_folds',  # Specified in the config file
                    help='n_folds',
                    default=3,
                    type=int)
parser.add_argument('--train_perc',  # Specified in the config file
                    help='Train percentage, the test will be = (1 - this_value)',
                    default=0.9,
                    type=float)
parser.add_argument('--test2_interval',  # Specified in the config file
                    help='Every test2_interval\'th example will be a test2 example for evaluation',
                    default=0,
                    type=int)
parser.add_argument('--auto_balance_bce',  # Specified in the config file
                    help='Increases the loss weight of minority class, useful in unbalanced problems.',
                    default='True',
                    type=lambda x: (str(x).lower() in ['true','1', 'yes']))

#-- Model analysis params
parser.add_argument('--save_dataset',  # Specified in the config file
                    help='If we should save the dataset .h5 file to the artifacts (defaults to false)',
                    default='False',
                    type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument('--save_model',  # Specified in the config file
                    help='If we should save the model weights .h5 file to the artifacts (defaults to false)',
                    default='False',
                    type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument('--eval_model',  # Specified in the config file
                    help='If we should evaluate (predict) the model on all folds (defaults to false)',
                    default='False',
                    type=lambda x: (str(x).lower() in ['true','1', 'yes']))

#-- hyperparameters
parser.add_argument('--model_type',  # Handled automatically by AI Platform
                    help='CNN_ONLY or CNN_GRU',
                    required=True)

parser.add_argument('--first_cnn_filter_size',  # first_cnn_filter_size
                help='network hyperparameter first_cnn_filter_size',
                required=True,
                type=int)

parser.add_argument('--first_cnn_n_filters',  # first_cnn_n_filters
                help='network hyperparameter first_cnn_n_filters',
                required=True,
                type=int)

parser.add_argument('--first_cnn_pool_size_strides',  # first_cnn_pool_size_strides
                help='network hyperparameter first_cnn_pool_size_strides',
                required=True,
                type=int)

parser.add_argument('--n_convs',  # n_convs
                help='network hyperparameter n_convs',
                required=True,
                type=int)

parser.add_argument('--n_convs_filter_size',  # n_convs_filter_size
                help='network hyperparameter n_convs_filter_size',
                required=True,
                type=int)

parser.add_argument('--n_convs_n_filters',  # n_convs_n_filters
                help='network hyperparameter n_convs_n_filters',
                required=True,
                type=int)

parser.add_argument('--n_convs_pool_size_stride',  # n_convs_pool_size_stride
                help='network hyperparameter n_convs_pool_size_stride',
                required=True,
                type=int)


parser.add_argument('--use_GRU',  # Specified in the config file
                    help='If we should evaluate (predict) the model on all folds (defaults to false)',
                    default='False',
                    type=lambda x: (str(x).lower() in ['true','1', 'yes']))

parser.add_argument('--n_gru',  # first_fcc_size
                help='network hyperparameter first_fcc_size',
                type=int)
parser.add_argument('--gru_hidden_size',  # first_fcc_size
                help='network hyperparameter first_fcc_size',
                type=int)

parser.add_argument('--first_fcc_size',  # first_fcc_size
                help='network hyperparameter first_fcc_size',
                required=True,
                type=int)

parser.add_argument('--n_fccs',  # n_fccs
                help='network hyperparameter n_fccs',
                required=True,
                type=int)

parser.add_argument('--fccs_size',  # fccs_size
                help='network hyperparameter fccs_size',
                required=True,
                type=int)

parser.add_argument('--outputs_separate_fc',  # outputs_separate_fc
                help='network hyperparameter outputs_separate_fc',
                required=True,
                type=int)

parser.add_argument('--lr',  # lr
                help='network hyperparameter lr',
                required=True,
                type=float)
parser.add_argument('--dropout_rate',  # lr
                help='network hyperparameter lr',
                required=True,
                type=float)




args = parser.parse_args()


################################################
#FILL ME FIRST!
## DEFINITIONS
output_name = args.output_name#"AA1"
print(output_name)
if len(output_name) > 0:
    fname_prefix = "".join(output_name)
else:
    fname_prefix = output_name[0]


seq_len = args.seq_len#3500
chrom_idx = args.chrom_idx
midpoint = args.midpoint

dataset_local_name = f"{fname_prefix}_hotspots_seqlen={seq_len}_multnegs.h5"

training_jobs_bucket = 'recombination-genomics-1'
train_file = f'datasets/{dataset_local_name}'
bucket = storage.Client().bucket(training_jobs_bucket)
## Define the source blob name (aka file name) for the training data
blob = bucket.blob(train_file)
## Download the data into a file name
#blob.download_to_filename(os.path.join('trainer', 'dataset.h5'))
blob.download_to_filename('dataset.h5')
ds_path = 'dataset.h5'

fold_fn_name = args.fold_fn_name#'partial_chrom_shuffled'
test2_interval = args.test2_interval

random_seed = args.random_seed
batch_size = args.batch_size

use_x_y = args.use_x_y
if use_x_y:
    chroms_to_use += ['X', 'Y']

input_keys = ['seq']
if args.midpoint:
    input_keys.append('midpoint')
if args.chrom_idx:
    input_keys.append('chrom_idx')

use_rev_compl = args.use_rev_compl

n_folds = args.n_folds
train_perc = args.train_perc
epochs = args.epochs

############
# Parse hps
hps = {}
hps['first_cnn_filter_size'] = args.first_cnn_filter_size
hps['first_cnn_n_filters'] = args.first_cnn_n_filters
hps['first_cnn_pool_size_strides'] = args.first_cnn_pool_size_strides
hps['n_convs'] = args.n_convs
hps['n_convs_filter_size'] = args.n_convs_filter_size
hps['n_convs_n_filters'] = args.n_convs_n_filters
hps['n_convs_pool_size_stride'] = args.n_convs_pool_size_stride

hps['use_GRU'] = args.use_GRU
if hps['use_GRU']:
    hps['gru_hidden_size'] = args.gru_hidden_size
    hps['n_gru'] = args.n_gru

hps['first_fcc_size'] = args.first_fcc_size
hps['n_fccs'] = args.n_fccs
hps['fccs_size'] = args.fccs_size
hps['outputs_separate_fc'] = [args.outputs_separate_fc] # this one is actually a list, but we optimize one layer only
hps['lr'] = args.lr
hps['dropout_rate'] = args.dropout_rate
hps['dropout_type'] = "normal" #make it mc to become mc dropout
hps['seq_len'] = seq_len

hps['outputs_names'] = output_name
############
dataset_config = get_dataset_config(dataset_full_name='dataset.h5',
                                    chroms_to_use=chroms_to_use)

train_test_split_config = get_train_test_split_config(strategy=fold_fn_name,
                                                      random_seed=random_seed)

data_io_config = get_data_io_config(input_keys=input_keys,
                                    chroms_to_use=chroms_to_use,
                                    use_rev_compl=use_rev_compl,
                                    output_name=output_name,
                                    batch_size=batch_size)

model_config = get_model_config(output_name=output_name,
                                chrom_idx=chrom_idx, 
                                midpoint=midpoint,
                                seq_len=seq_len,
                                data_io_config=data_io_config)
model_config['model_type'] = args.model_type

model_config['hp_dict'] = hps

k_fold_params = get_k_fold_params(n_folds=n_folds,
                                  train_perc=train_perc,
                                  random_seed=random_seed)

negexamples_config = {(1000,5000):0.1,
                      (5000,10000):0.1,
                      'shuffle_pos':0.1}

############
chroms_list = dataset_config['chroms_to_use']
input_keys = data_io_config['input_keys']
output_keys = data_io_config['output_keys']
index_keys = data_io_config['index_keys']
n_folds = k_fold_params['n_folds']
train_perc = k_fold_params['train_perc']
test_perc = k_fold_params['test_perc']
random_seed = k_fold_params['random_seed']
############



data_dict = load_and_create_dictionary(dataset_config, data_io_config, negexamples_config, 'train')

# Extract the test2 dataset
data_dict, test2_dict = train_test_splits.train_val_split_dicts_by_interval(data_dict,
                                                                            input_keys,
                                                                            output_keys,
                                                                            index_keys,
                                                                            test2_interval)


for c, d in data_dict.items():
    print(c, d['seq'].shape, [d[o].shape for o in output_name], d['chrom_idx'].shape,
          d['ds_index'].shape, d['index'].shape, d['ex_type'].shape)

total_examples = 0
pos_examples = 0
neg_examples = 0
for chrom in data_dict.keys():
    o = data_dict[chrom][output_name[0]]
    l = o.shape
    pos_examples += (o == 1).sum()
    neg_examples += (o == 0).sum()
    total_examples += l[0]
print(f"There are {total_examples} examples in the dataset")
print(f"Pos examples: {pos_examples}, ratio: {pos_examples/total_examples:.2f}")
print(f"Neg examples: {neg_examples}, ratio: {neg_examples/total_examples:.2f}")

if fold_fn_name == 'partial_chrom_shuffled':
    fold_fn = train_test_splits.partial_chrom_shuffled_k_fold
elif fold_fn_name == 'partial_chrom_contig':
    fold_fn = train_test_splits.partial_chrom_contig_k_fold
elif fold_fn_name == 'partial_chrom_contig_alternate':
    fold_fn = train_test_splits.partial_chrom_contig_alternate_k_fold
elif fold_fn_name == 'whole_genome_shuffled_k_fold':
    fold_fn = train_test_splits.whole_genome_shuffled_k_fold
else:
    raise ValueError("Wrong fold name")
    
    
print(data_dict[1].keys())
print(data_dict[1]['index'])

folds = fold_fn(data_dict,
                n_folds,
                chroms_list,
                train_perc,
                test_perc,
                input_keys,
                output_keys,
                index_keys,
                random_seed)

h5_path = dataset_config['dataset_name']
f = h5py.File(h5_path, 'r')
metadata = json.loads(f['/'].attrs["metadata"])
f.close()

multiprocess_fit_init = partial(multiprocess_fit,
                                folds=folds,
                                test2_dict=test2_dict,
                                metadata=metadata,
                                dataset_config=dataset_config,
                                data_io_config=data_io_config,
                                model_config=model_config,
                                dataset_local_name=dataset_local_name,
                                fold_fn_name=fold_fn_name,
                                epochs=epochs,
                                balance_bce=args.auto_balance_bce,
                                save_model=args.save_model,
                                evaluate_model=False) # We do hyperparameters search, no need to eval every single example


folds_val_res_dict = {}
folds_test2_res_dict = {}

for i in range(k_fold_params['n_folds']):
    print(f"Training fold {i}, launching the process")

    p = multiprocessing.Process(target=multiprocess_fit_init, args=(i,))

    p.start()
    print(f"\tProcess launched")
    p.join()

    print(f"End of fold {i+1}/{k_fold_params['n_folds']}")
    print("*"*78)
    
    #Load pickle file to log metrics in a graphic form
    print("Loading result from the training processes and computing the folds average")
    
    # The following file contains history.history, which according to the default configs
    # should look like this:
    # {'loss': [0.6652615070343018, 0.47628292441368103],
    # 'accuracy': [0.6251851916313171, 0.7789629697799683],
    # 'auc_1': [0.6670173406600952, 0.8540434241294861],
    # 'precision_1': [0.6307108402252197, 0.7932062149047852],
    # 'recall_1': [0.6023198962211609, 0.7541027069091797],
    # 'val_loss': [0.5642127990722656, 1.1570193767547607],
    # 'val_accuracy': [0.7163559794425964, 0.6229497194290161],
    # 'val_auc_1': [0.7875217199325562, 0.8865352272987366],
    # 'val_precision_1': [0.6939992904663086, 0.9783375263214111],
    # 'val_recall_1': [0.7783024907112122, 0.2547553479671478]}
    fname = f'{dataset_local_name}_{fold_fn_name}_fold_{i}_histories.pkl'
    try:
        with open(fname, 'rb') as f:
            history = pickle.load(f)
            folds_val_res_dict[i] = history
    except FileNotFoundError:
        # Training have failed
        break

    
    # Register the test2 result
    #{'loss': 0.3623631000518799,
    #'accuracy': 0.8561009168624878,
    #'auc': 0.9307486414909363,
    #'precision': 0.8951964974403381,
    #'recall': 0.8070865869522095}
    fname = f'{dataset_local_name}_{fold_fn_name}_fold_{i}_test2_metrics.pkl'
    with open(fname, 'rb') as f:
        test2_metrics = pickle.load(f)
        folds_test2_res_dict[i] = test2_metrics



use_test = True
metric_to_hps_optimize = 'f1_score'
direction = 'maximize'
use_mc = True
if use_mc and use_test:
    metric_to_hps_optimize += '_mc'
#     val_str = 'val_mc'
# else:
#     val_str = 'val'

# If training has failed, report a very bad performance
if len(folds_val_res_dict.keys()) == 0:
    if direction == 'maximize':
        score = -1
    else:
        score = 10

else:
    total_performance = []
    if use_test:
        dict_to_use = folds_test2_res_dict
        keys_to_get = [f'{o}_{metric_to_hps_optimize}' for o in output_name]
    else:
        dict_to_use = folds_val_res_dict
        keys_to_get = [f'val_{o}_{metric_to_hps_optimize}' for o in output_name]

    print("\n\n\n")
    print(folds_test2_res_dict)
    print("\n\n\n")
    print(folds_val_res_dict)
    print("\n\n\n")
    if direction == 'maximize':
        func = max
    else:
        func = min

    for k, v in dict_to_use.items():
        for m in keys_to_get:
            if isinstance(v[m], list):
                val = func(v[m])
            else:
                val = v[m]
            total_performance.append(val)

    print(total_performance)
    print(np.mean(total_performance))

    # compute score to be returned to the hps optimizer
    score = np.mean(total_performance)


    print(metric_to_hps_optimize, score, epochs)
    # Hypertune stuff
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag=metric_to_hps_optimize,
                                       metric_value=score,
                                       global_step=epochs
                                       )

# save the model to bucket
# save_model_to_bucket(model_filename=model_name+".h5", 
#                      bucket_id='recombination-genomics-1', 
#                      inside_bucket_path='datasets')