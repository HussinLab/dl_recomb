from trainer.code.dataloading.feature_engineering import dummy_func, separate_elemets
from trainer.code.dataloading import train_test_splits
from trainer.code.dataloading.dataloading import load_and_create_dictionary
from trainer.code.dataloading.tf_dataloaders import  DNA_tf_dl
from trainer.code.dataloading.feature_engineering import dummy_func, separate_elemets
from trainer.code.dataloading import train_test_splits
from trainer.code.dataloading.tf_dataloaders import  DNA_tf_dl

from trainer.code.modeling.tf.models import create_multioutput_model
from trainer.code.modeling.tf.models import create_multioutput_model
from trainer.code.utils import *

from io import StringIO
from functools import partial
import tensorflow as tf
import h5py
import json
from keras import backend as K
import os

import pandas as pd
import wandb
from wandb.keras import WandbCallback
import argparse
 
 
# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-d", "--dataset_path", help = "Show Output")
parser.add_argument("-c", "--chrom_use",nargs="*", default=[20, 21], help = "Show Output")
parser.add_argument("-o", "--outdir", help = "Show Output")

infos = args.dataset_path.split('/')
prefix = infos[len(infos)-1].split('_seqlen=')[0]
seqlen = infos[len(infos)-1].split('_seqlen=')[1].split('_')[0]

def run_experiment(chrom, seq_len, prefix, fold_fn_name, chrom_idx, outdir, ds_path, niter=0,name_run='new_run'):

    # prefix = prefix[0]
    # chrom = all_chrom[0]
    # seq_len = seq_len[0]
    # fold_fn_name=fold_fn_name[0]
    # chrom_idx=True
    # name_run = 'new_run'
    
    wandb.init(project="test-chip",mode='dryrun',name=name_run)
    
    batch_size = 128
    random_seed = 123
    n_folds = 3
    train_perc = 0.7
    epochs = 10
    use_rev_compl = True
    midpoint = True
    
    output_name = [f"{prefix}"] 
    fname_prefix = "".join(output_name)
    
    chroms_to_use = chrom
    input_keys = ['seq']
    if midpoint:
        input_keys.append('midpoint')
    if chrom_idx:
        input_keys.append('chrom_idx')
    
    dataset_config = get_dataset_config(dataset_full_name=ds_path,
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
    
    k_fold_params = get_k_fold_params(n_folds=n_folds,
                                      train_perc=train_perc,
                                      random_seed=random_seed)
    
    chroms_list = dataset_config['chroms_to_use']
    input_keys = data_io_config['input_keys']
    output_keys = data_io_config['output_keys']
    index_keys = data_io_config['index_keys']
    n_folds = k_fold_params['n_folds']
    train_perc = k_fold_params['train_perc']
    test_perc = k_fold_params['test_perc']
    random_seed = k_fold_params['random_seed']
        
    data_dict = load_and_create_dictionary(dataset_config, 
                                           data_io_config, 
                                           negexamples_config, 
                                           'train',
                                           chroms_to_use=chroms_to_use)               
    data_dict, test2_dict = train_test_splits.train_val_split_dicts_by_interval(data_dict,
                                                                                input_keys,
                                                                                output_keys,
                                                                                index_keys,
                                                                                test2_interval)
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
    #print(f"Pos examples: {pos_examples}, ratio: {pos_examples/total_examples:.2f}")
    #print(f"Neg examples: {neg_examples}, ratio: {neg_examples/total_examples:.2f}")
    
    
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
    
    folds = fold_fn(data_dict,
                    n_folds,
                    chroms_list,
                    train_perc,
                    test_perc,
                    input_keys,
                    output_keys,
                    index_keys,
                    random_seed)
    
    f = h5py.File(dataset_config['dataset_name'], 'r')
    metadata = json.loads(f['/'].attrs["metadata"])
    f.close()
    
    rev_com_dict_str = metadata['num_reverse_compliment_dict']
    rev_com_dict = {int(k): v for k, v in rev_com_dict_str.items()}
    
    use_rc = data_io_config['use_rev_compl_as_input']
    
    if use_rc:
        add_rc = rev_com_dict
    else:
        add_rc = None
    
    X_transform_dict = data_io_config['X_transform_dict']
    y_transform_dict = data_io_config['y_transform_dict']
    batch_size = data_io_config['batch_size']
    shuffle_train = data_io_config['shuffle_train']
    shuffle_val = data_io_config['shuffle_val']
    shuffle_test = data_io_config['shuffle_test']
    hparams_dict = model_config['hp_dict']
    
    fold_train_data, fold_val_data = train_test_splits.merge_folds(folds, 0)
    
    train_ds = DNA_tf_dl(X_dict=fold_train_data['input'],
                             X_transform_dict=X_transform_dict,
                             y_dict=fold_train_data['output'],
                             y_transform_dict=y_transform_dict,
                             rev_comp_dict=add_rc,
                             batch_size=batch_size,
                             shuffle=shuffle_train,
                             reshuffle_on_epoch_end=False)
    
    val_ds = DNA_tf_dl(X_dict=fold_val_data['input'],
                           X_transform_dict=X_transform_dict,
                           y_dict=fold_val_data['output'],
                           y_transform_dict=y_transform_dict,
                           rev_comp_dict=add_rc,
                           batch_size=batch_size,
                           shuffle=shuffle_val,
                           reshuffle_on_epoch_end=False)
    
    test2_ds = DNA_tf_dl(X_dict=test2_dict['input'],
                            X_transform_dict=X_transform_dict,
                            y_dict=test2_dict['output'],
                            y_transform_dict=y_transform_dict,
                            rev_comp_dict=add_rc,
                            batch_size=batch_size,
                            shuffle=False,
                            reshuffle_on_epoch_end=False)
    
    
    #earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=77)
    
    #callbacks = [[earlystop_callback]]
    
    metrics=['accuracy', 
                 tf.keras.metrics.AUC(), 
                 tf.keras.metrics.Precision(), 
                 precision_m,
                 tf.keras.metrics.Recall(),
                 recall_m,
                 f1_m]
    
    tf.keras.backend.clear_session()
    model = create_multioutput_model(model_config)
    opt = tf.keras.optimizers.RMSprop(learning_rate=hparams_dict['lr'])
    balance_bce = False
    
    if balance_bce:
        train_pos_ratio_dict = {k:v.sum()/v.shape[0] for k, v in output_keys}
        
        for o in output_keys:
            model_config['outputs'][o]['weight'] /= train_pos_ratio_dict[o]
            loss_dict = {o:wrapped_partial(wbce,
                                 pos_class_wgt=train_pos_ratio_dict[o]) for o in output_keys}
    else:
        loss_dict = {o:'binary_crossentropy' for o in hparams_dict['outputs_names']}
    
    model.compile(loss=loss_dict, 
                  optimizer=opt,
                  metrics=metrics)
    
    history = model.fit(train_ds,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=val_ds,
                        callbacks=[WandbCallback()],
                        verbose=1
                        ) 
    model.save(f'{outdir}/chr_all.seqLen{seq_len}.group{prefix}.{fold_fn_name}.chrom_idx_{str(chrom_idx)}.iter_{niter}')
    pd.DataFrame(history.history).to_csv(f'{outdir}/chr_all.seqLen{seq_len}.group{prefix}.{fold_fn_name}.chrom_idx_{str(chrom_idx)}.iter_{niter}.csv',index=False)
    

chrom_idx = False
midpoint = True
test2_interval = 10
random_seed = 123
batch_size = 128
use_rev_compl = True

all_chrom = args.chrom_use
seq_len = seqlen
outdir = args.outdir
ds_path = args.dataset_path

n_folds = 3
chrom_idx = [True,False]
run_nb = 0

ds_root = ''
negexamples_config = {(1000,5000):1.0}
fold_fn_names = ['partial_chrom_shuffled','partial_chrom_contig','partial_chrom_contig_alternate','whole_genome_shuffled_k_fold']
fold_fn_name = fold_fn_names[0]

os.environ["WANDB_API_KEY"] = "181fe15119f612e9e270418216720c40b876b43e"
os.environ["WANDB_MODE"] = "dryrun"

wandb.config = {
  "learning_rate": 0.02,
  "epochs": 10,
  "batch_size": 128
}

run_experiment(all_chrom, 
               seq_len, 
               prefix, 
               fold_fn_name, 
               True, 
               1, 
               '.'.join(['all_chroms.','sl'+str(seq_len),prefix,fold_fn_name,'chr_idx_'+str(True)]))