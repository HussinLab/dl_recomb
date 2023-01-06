import sys
#sys.path.append("./code") 

from io import StringIO
import os
from functools import partial
import tensorflow as tf
from .code.dataloading.feature_engineering import dummy_func, separate_elemets
import pandas as pd


chroms_lens = """
chrom,size
1,249250621
2,243199373
3,198022430
4,191154276
5,180915260
6,171115067
7,159138663
X,155270560
8,146364022
9,141213431
10,135534747
11,135006516
12,133851895
13,115169878
14,107349540
15,102531392
16,90354753
17,81195210
18,78077248
20,63025520
Y,59373566
19,59128983
22,51304566
21,48129895
"""

chrom_len_dfs = pd.read_csv(StringIO(chroms_lens))
chroms_lens_dict = chrom_len_dfs.set_index('chrom').to_dict()['size']


def get_dataset_config(dataset_full_name, chroms_to_use):
    dataset_config = {'dataset_name':dataset_full_name, 
                      'chroms_lens_dict':chroms_lens_dict,
                      'chroms_to_use':chroms_to_use
                      }
    return dataset_config


def get_train_test_split_config(strategy, random_seed):
    train_test_split_config = {
                                #whole_chroms, partial_chrom_contig, partial_chrom_random, by_random
                                   'strategy':strategy, #'partial_chrom_contig',
                                 #if strategy is by_whole_chrom, it's a list. Else, a float between 0 and 1 and must add up to 1
                                   'train':0.6, 
                                   'val':0.2,
                                   'test':0.2,
                                   'random_seed':random_seed
                              }
    return train_test_split_config

def get_data_io_config(input_keys, chroms_to_use, use_rev_compl, output_name, batch_size):
    #seq is mandatory
    X_transform_dict = {'seq': partial(tf.one_hot, depth=4)}
    
    #Add the optional inputs
    if 'chrom_idx' in input_keys:
        X_transform_dict['chrom_idx'] = partial(tf.one_hot, depth=23 )#len(chroms_to_use))
    if 'midpoint' in input_keys:
        X_transform_dict['midpoint'] = dummy_func
    
    data_io_config = {
                          #'use_seq_as_input':True, #This one is assumed by default
                          'use_rev_compl_as_input':use_rev_compl,
                          'input_keys':input_keys,
                          'output_keys':['out'],
                          'separate_outputs':{'out':output_name}, #MAKE THIS 
                          'index_keys':['chrom_idx', 'index', 'ex_type', 'ds_index'],
                          'X_transform_dict':X_transform_dict,
                          'y_transform_dict':{o: dummy_func for o in output_name},
                          'batch_size':batch_size,
                          'shuffle_train':True,
                          'shuffle_val':False,
                          'shuffle_test':False,
                     }
    return data_io_config




def get_model_config(output_name, seq_len, chrom_idx, midpoint, data_io_config):
    hparams_dict = {}

    hparams_dict['first_cnn_filter_size'] = 30
    hparams_dict['first_cnn_n_filters']   = 120
    hparams_dict['first_cnn_pool_size_strides']   = 15

    hparams_dict['n_convs'] = 1
    hparams_dict['n_convs_dilation'] = 1
    hparams_dict['n_convs_filter_size']  = 3
    hparams_dict['n_convs_n_filters']    = 15
    hparams_dict['n_convs_pool_size_stride']    = 10
    #hparams_dict['n_convs_pool_strides'] = 10

    hparams_dict['fc_activations'] = tf.keras.activations.selu#'elu'
    
    hparams_dict['use_GRU'] = True
    hparams_dict['n_gru'] = 1
    hparams_dict['gru_hidden_size'] = 50

    hparams_dict['first_fcc_size'] = 250
    hparams_dict['n_fccs'] = 1
    hparams_dict['fccs_size'] = 400
    hparams_dict['seq_len'] = seq_len


    hparams_dict['dropout_type'] = "normal" #make it mc to become mc dropout
    hparams_dict['dropout_rate'] = 0.15

    hparams_dict['lr'] = 0.002#0.0004

    hparams_dict['outputs_separate_fc'] = [30, 10]

    hparams_dict['n_outputs'] = 1
    hparams_dict['outputs_names'] = output_name if isinstance(output_name, list) else [output_name]
    
    inputs_dict = {
                    'seq': {'type':'normal', 
                            'dim':(seq_len, 4) },
                    'seq_rc':{'type':'normal', 
                              'dim':(seq_len, 4) }
                  }
    if chrom_idx == True:
        #dims here is 1 hot size, embedding size
        inputs_dict['chrom_idx'] = {'type':'embedding', 
                                    'dim':(23, 2) }
    if midpoint == True:
        inputs_dict['midpoint'] = {'type':'normal', 
                                   'dim':(1,) }
    
    model_config = {
                        'bayesian_epistemic':False,
                        'bayesian_aleatoric':False,
                        'inputs':inputs_dict,
                        'outputs':{o:{'loss':'binary_crossentropy', 'weight':1} for o in output_name},
                        'data_config':data_io_config, #Used to know the model's input
                        'hp_dict':hparams_dict
    #                     'model_hparams_dict':hparams_dict
                   }
    
    if (chrom_idx == True) and (midpoint == True):
        model_config['embed_seq_loc'] = {'dim':2}
    
    return model_config


def get_k_fold_params(n_folds, train_perc, random_seed):
    k_fold_params = {
                        'n_folds':n_folds,
                        'train_perc':train_perc,
                        'test_perc':1-train_perc,
                        'random_seed':random_seed
                    }
    return k_fold_params


def get_negexamples_config(strtgy_0_prop,
                           strtgy_1_prop,
                           strtgy_2_prop,
                           strtgy_3_prop,
                           shuffle_pos_prop,
                           ):
    negexamples_config = {
                            0:strtgy_0_prop,
                            1:strtgy_1_prop,
                            2:strtgy_2_prop,
                            3:strtgy_3_prop,
                            'shuffle_pos':shuffle_pos_prop
                         }
    return negexamples_config

