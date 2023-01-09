from trainer.code.dataloading.feature_engineering import dummy_func, separate_elemets
from trainer.code.dataloading import train_test_splits
from trainer.code.dataloading.dataloading import create_tf_ds, load_and_create_dictionary
from trainer.code.dataloading.tf_dataloaders import  DNA_tf_dl
from trainer.code.dataloading.feature_engineering import dummy_func, separate_elemets
from trainer.code.dataloading import train_test_splits
from trainer.code.dataloading.tf_dataloaders import  DNA_tf_dl

from trainer.code.modeling.tf.models import create_multioutput_model
from trainer.code.modeling.tf.models import create_multioutput_model


from io import StringIO
from functools import partial
import tensorflow as tf
import h5py
import json
from keras import backend as K

from tensorflow.keras.layers import *

from functools import partial, update_wrapper


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
    hparams_dict['first_cnn_n_filters']   = 20   
    hparams_dict['first_cnn_pool_size_strides']   = 15

    hparams_dict['n_convs'] = 1
    hparams_dict['n_convs_dilation'] = 1
    hparams_dict['n_convs_filter_size']  = 3
    hparams_dict['n_convs_n_filters']    = 15
    hparams_dict['n_convs_pool_size_stride']    = 10
    #hparams_dict['n_convs_pool_strides'] = 10

    hparams_dict['fc_activations'] = tf.keras.activations.selu#'elu'
    
    hparams_dict['use_GRU'] = False
    hparams_dict['n_gru'] = 1
    hparams_dict['gru_hidden_size'] = 50

    hparams_dict['first_fcc_size'] = 250
    hparams_dict['n_fccs'] = 1
    hparams_dict['fccs_size'] = 400
    hparams_dict['seq_len'] = seq_len


    hparams_dict['dropout_type'] = "normal" #make it mc to become mc dropout
    hparams_dict['dropout_rate'] = 0.15

    hparams_dict['lr'] = 0.002#0.0004

    hparams_dict['outputs_separate_fc'] = [250]

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



## KERAS 

def recall_m(y_true, y_pred):
    """
    https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
    There is a slight deviation between this calculation and the builtin one (3rd decimal point)
    For now I am keeping it as the F1 score is very near the true and is useful enough for HPS search, however
    I should implement a more correct version
    TODO: Do it with a more straghtforward one, where you compare y_pred to 0.5 and then check equality
    to y_true
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def get_chosen_dropout(mc=False):
    if mc:
        return MCDropout
    else:
        return tf.keras.layers.Dropout


def create_multioutput_model(model_config):
    """
    """
    hparams = model_config['hp_dict']
    dropout_layer = get_chosen_dropout(hparams['dropout_type'] == "mc")

    all_inputs_list = []
    # -------------------------------------------------------------------------
    # Direct Path (Mandatory in all models)
    inp_direct = Input(shape=(hparams['seq_len'], 4), name="seq")
    all_inputs_list.append(inp_direct)

    shared_conv = Convolution1D(filters=hparams['first_cnn_n_filters'],
                                kernel_size=hparams['first_cnn_filter_size'],
                                padding="valid",
                                activation="relu",
                                strides=1,
                                input_shape=(hparams['seq_len'], 4),
                                name="shared_first_conv")

    direct_path = shared_conv(inp_direct)

    direct_path = MaxPooling1D(pool_size=hparams['first_cnn_pool_size_strides'], 
                               strides=hparams['first_cnn_pool_size_strides'],
                               name="direct_path_first_max_pool")(direct_path)

    direct_path = dropout_layer(hparams['dropout_rate'],
                                name="direct_path_first_drop_out")(direct_path)
    for i in range(hparams['n_convs']):
        # Create common layer
        new_conv = Convolution1D(filters=hparams['n_convs_n_filters'],
                                 kernel_size=hparams['n_convs_filter_size'],
                                 padding="valid",
                                 activation="relu",
                                 strides=1,
                                 # dilation_rate=hparams['n_convs_dilation'],
                                 name=f"n_convs_{i+1}_conv")

        # Direct path
        direct_path = new_conv(direct_path)

        direct_path = MaxPooling1D(pool_size=hparams['n_convs_pool_size_stride'],
                                   strides=hparams['n_convs_pool_size_stride'],
                                   name=f"direct_path_n_convs_{i+1}_max_pool")(direct_path)

        direct_path = dropout_layer(hparams['dropout_rate'],
                                    name=f"direct_path_n_convs_{i+1}_drop_out")(direct_path)

        # TODO: Compare flattening before and after addition
        # direct_path = Flatten()(direct_path)

    # ----------------------------------------------------------------------------
    # Reverse Complement Path (Optional)
    if 'seq_rc' in model_config['inputs'].keys():
        inp_reverse = Input(shape=(hparams['seq_len'], 4), name="seq_rc")
        all_inputs_list.append(inp_reverse)

        reverse_path = shared_conv(inp_reverse)

        reverse_path = MaxPooling1D(pool_size=hparams['first_cnn_pool_size_strides'],
                                    strides=hparams['first_cnn_pool_size_strides'],
                                    name="reverse_path_first_max_pool")(reverse_path)

        reverse_path = dropout_layer(hparams['dropout_rate'],
                                     name="reverse_path_first_drop_out")(reverse_path)

        for i in range(hparams['n_convs']):
            # REVERSE PATH
            reverse_path = new_conv(reverse_path)

            reverse_path = MaxPooling1D(pool_size=hparams['n_convs_pool_size_stride'],
                                        strides=hparams['n_convs_pool_size_stride'],
                                        name=f"reverse_path_n_convs_{i+1}_max_pool")(reverse_path)

            reverse_path = dropout_layer(hparams['dropout_rate'],
                                         name=f"reverse_path_n_convs_{i+1}_drop_out")(reverse_path)

        # TODO: Compare flattening before and after addition
        # reverse_path = Flatten()(reverse_path)
        seq_repr = tf.keras.layers.add([direct_path, reverse_path])
    else:
        seq_repr = direct_path

    if hparams['use_GRU']:
        for i in range(hparams['n_gru']):
            # In order to stack GRUs, we need to return the sequences for everyone except the last one
            if i == ():
                return_sequences = True
            else:
                return_sequences = False
            seq_repr = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(hparams['gru_hidden_size'],
                                                                         return_sequences=True,
                                                                         name=f"gru_layer_{i}"))(seq_repr)
            seq_repr = dropout_layer(hparams['dropout_rate'],
                                     name=f"gru_droupout_{i}")(seq_repr)
    
    seq_repr = Flatten()(seq_repr)

    processed_inputs_list = [seq_repr]

    # ----------------------------------------------------------------------------
    # Chrom Index (Optional)
    if 'chrom_idx' in model_config['inputs'].keys():
        inp_size = model_config['inputs']['chrom_idx']['dim'][0]
        embed_size = model_config['inputs']['chrom_idx']['dim'][1]

        inp_chrom_idx = Input(shape=(inp_size, ), name="chrom_idx")
        all_inputs_list.append(inp_chrom_idx)

        chrom_emb = Dense(embed_size,
                          name="chrom_emb",
                          use_bias=False,
                          kernel_constraint=tf.keras.constraints.max_norm(1.))(inp_chrom_idx)

        # if we will not embed seq loc first, then add it to the concat list
        if not ('embed_seq_loc' in model_config.keys()):
            processed_inputs_list.append(inp_chrom_idx)

    if 'midpoint' in model_config['inputs'].keys():
        inp_size = model_config['inputs']['midpoint']['dim'][0]
        # NOTE: I decided just to force size to be equal to 1. Reason is that
        # this is the only correct value anyway, and it looks very verbose to
        # read it form the dict.
        inp_midpoint = Input(shape=(1, ), name="midpoint")
        all_inputs_list.append(inp_midpoint)

        # if we will not embed seq loc first, then add it to the concat list
        if not 'embed_seq_loc' in model_config.keys():
            processed_inputs_list.append(inp_midpoint)

    # This is a second embedding of the chrom_ID and the location, the idea
    # is to create an embedding that may show similarities between locs
    # at different chromosomes. So despite it may seem redundunt, I want to
    # take a look at it using T-SNE for example
    if 'embed_seq_loc' in model_config.keys():
        seq_loc_embd = tf.keras.layers.Concatenate(axis=1)([chrom_emb,
                                                            inp_midpoint])
        seq_loc_embd = Dense(model_config['embed_seq_loc']['dim'],
                             name="seq_loc_embd",
                             use_bias=False,
                             kernel_constraint=tf.keras.constraints.max_norm(1.))(seq_loc_embd)

        processed_inputs_list.append(seq_loc_embd)

    # Since Concatenate layer will throw an error if the list contains only
    # one item, we handle this in here
    # Error is: "ValueError: A `Concatenate` layer should be called on a list
    #           of at least 2 inputs"
    if len(processed_inputs_list) > 1:
        unified = tf.keras.layers.Concatenate(axis=1)(processed_inputs_list)
    else:
        unified = processed_inputs_list[0]

    unified = tf.keras.layers.BatchNormalization()(unified)

    unified = Dense(hparams['first_fcc_size'],
                    name="first_fcc",
                    kernel_constraint=tf.keras.constraints.max_norm(1.))(unified)
    unified = Activation('relu')(unified)
    unified = tf.keras.layers.BatchNormalization()(unified)

    unified = dropout_layer(hparams['dropout_rate'],
                            name="first_fcc_drop_out")(unified)

    for i in range(hparams['n_fccs']):
        unified = Dense(hparams['fccs_size'],
                        name=f"fcc_{i+1}",
                        kernel_constraint=tf.keras.constraints.max_norm(1.))(unified)
        unified = Activation('relu')(unified)
        unified = tf.keras.layers.BatchNormalization()(unified)
        unified = dropout_layer(hparams['dropout_rate'],
                                name=f"fcc_{i+1}_drop_out")(unified)

    # unified = Dense(hparams['n_outputs'],
    #                name="prediction_layer")(unified)
    # output = Activation('sigmoid', dtype='float32')(unified)

    outputs = []
    for o in hparams['outputs_names']:
        print(o, hparams['outputs_names'])
        ind_output_n_fcs = len(hparams['outputs_separate_fc'])
        curr_output = unified
        for i in range(ind_output_n_fcs):
            layers_size = hparams['outputs_separate_fc'][i]
            curr_output = Dense(layers_size,
                                name=f"{o}_raw_{i}")(curr_output)
            curr_output = Activation('relu')(curr_output)
            curr_output = dropout_layer(hparams['dropout_rate'],
                                        name=f"{o}_separate_fc_{i+1}_drop_out")(curr_output)

        curr_output = Dense(1,
                            name=f"{o}_logit")(curr_output)
        curr_output = Activation('sigmoid',
                                 dtype='float32',
                                 name=f"{o}")(curr_output)
        outputs.append(curr_output)

    model = tf.keras.Model(inputs=all_inputs_list,
                           outputs=outputs)

    return model


def wbce( tf_y_true, tf_y_pred, pos_class_wgt ) :
    y_true = tf.cast(tf_y_true, dtype=tf_y_pred.dtype)
    y_pred = tf.cast(tf_y_pred, dtype=tf_y_pred.dtype)
    
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    
    #logloss = -((y_true * K.log(y_pred) * class_wgts_dict[1]) + ((1 - y_true) * K.log(1 - y_pred) * class_wgts_dict[0]) )
    #tf.math.scalar_mul(
    #logloss = -( tf.math.scalar_mul(class_wgts_dict[1], (y_true* K.log(y_pred)) ) + tf.math.scalar_mul(class_wgts_dict[0], ((1 - y_true) * K.log(1 - y_pred)) ))
    
    pos_class = y_true * K.log(y_pred)
    neg_class = (1 - y_true) * K.log(1 - y_pred)
    
    logloss = -((pos_class*pos_class_wgt) + neg_class)
    
    return K.mean( logloss, axis=-1)
def wrapped_partial(func, *args, **kwargs):
    """
    http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func



# Update the weights in model_config
# for o in output_keys:
#    model_config['outputs'][o]['weight'] /= train_pos_ratio_dict[o]


