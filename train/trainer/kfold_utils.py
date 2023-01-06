import sys
#sys.path.append("../../code") 

from .code.dataloading import train_test_splits
from .code.dataloading.tf_dataloaders import  DNA_tf_dl
from .code.modeling.tf.models import create_multioutput_model
from .code.modeling.tf.custom_losses import wbce, wrapped_partial
from .code.model_evaluation.tf.cnn_model_evaluation import mc_dropout_analyze_v2, analyze_mc_result

import json
import pickle
import pandas as pd

from tensorflow.keras import backend as K

def recall_score(y_true, y_pred):
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

def precision_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def multiprocess_fit(i, 
                     folds,
                     test2_dict,
                     metadata,
                     dataset_config,
                     data_io_config ,
                     model_config,
                     dataset_local_name, 
                     fold_fn_name,
                     epochs,
                     balance_bce=False,
                     save_model=False,
                     evaluate_model=False):
    import tensorflow as tf
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                          patience=20)
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)

    callbacks = [earlystop_callback, ckpt_callback]
    
    if save_model or evaluate_model:
        model_fname = f'{dataset_local_name}_{fold_fn_name}_fold_{i}_model_weight.h5'
        ckpnt_callback = tf.keras.callbacks.ModelCheckpoint(model_fname)
        callbacks.append(ckpnt_callback)

    metrics=['accuracy', 
             tf.keras.metrics.AUC(), 
             tf.keras.metrics.Precision(), 
             tf.keras.metrics.Recall(),
             f1_score]

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
    
    fold_train_data, fold_val_data = train_test_splits.merge_folds(folds, i)
    
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
    
    tf.keras.backend.clear_session()

    # For now, we control the model inclusion of GRU through HPS, not different functions
    #if model_config['model_type'] == 'CNN_ONLY':
    model = create_multioutput_model(model_config)

    opt = tf.keras.optimizers.RMSprop(learning_rate=hparams_dict['lr'])

    output_keys = train_ds.y_dict.keys()
    if balance_bce:
        train_pos_ratio_dict = {k:v.sum()/v.shape[0] for k, v in train_ds.y_dict.items()}
        for o in output_keys:
            model_config['outputs'][o]['weight'] /= train_pos_ratio_dict[o]
            loss_dict = {o:wrapped_partial(wbce,
                                 pos_class_wgt=train_pos_ratio_dict[o]) for o in output_keys}
    else:
        loss_dict = {o:'binary_crossentropy' for o in hparams_dict['outputs_names']}

    model.compile(loss=loss_dict, #'binary_crossentropy',  #loss_dict
                  #loss_weights=class_wgts,
                  optimizer=opt,
                  metrics=metrics)
    #tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    
    history = model.fit(train_ds,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=val_ds,
                        callbacks=callbacks,
                        verbose=2
                        )
    fname = f'{dataset_local_name}_{fold_fn_name}_fold_{i}_histories.pkl'
    print(f"Saving fold data to file {fname}")
    with open(fname, 'wb') as f:
        pickle.dump(history.history, f)

    # Load best model for evaluation
    model.load_weights('model.h5')
    #if save_model:
    #    model.save(f'{dataset_local_name}_{fold_fn_name}_fold_{i}_model_weight.h5')
    test2_ds = DNA_tf_dl(X_dict=test2_dict['input'],
                        X_transform_dict=X_transform_dict,
                        y_dict=test2_dict['output'],
                        y_transform_dict=y_transform_dict,
                        rev_comp_dict=add_rc,
                        batch_size=batch_size,
                        shuffle=False,
                        reshuffle_on_epoch_end=False)

    # TODO: Add all the metrics in the prediction of test2
    test2_res = model.evaluate(test2_ds,
                               batch_size=batch_size,
                               return_dict=True)

    ret = mc_dropout_analyze_v2(model,
                                test2_ds, output_keys, 
                                n_samples=20,
                                return_stats_only=True,
                                return_as_df=True)
    mc_res = analyze_mc_result(ret, output_keys)
    eval_funcs_names = mc_res.index
    # Registering MC Results
    for f in eval_funcs_names:
        #test2_res[f'{f}_mc'] = []
        for o in output_keys:
            #test2_res[f'{f}_mc'].append(mc_res.loc[f, f'{o}_mc'])
            test2_res[f'{o}_{f}_mc'] = mc_res.loc[f, f'{o}_mc']
            print(f'{o}_{f}_mc', mc_res.loc[f, f'{o}_mc'])

    print("\n\nTest 2 result:", test2_res)
    fname = f'{dataset_local_name}_{fold_fn_name}_fold_{i}_test2_metrics.pkl'
    print(f"Saving fold test2 result to file {fname}")
    with open(f"{fname}", 'wb') as f:
        pickle.dump(test2_res, f)
    
    if evaluate_model:
        print(f"Evaluating Model of Fold {i}")
        fold_res_df = pd.DataFrame()
        for k in range(len(folds)):
            temp_df = pd.DataFrame()
            curr_fold = folds[k]
            
            for o in curr_fold['output'].keys():
                temp_df[f'truth_{o}'] = curr_fold['output'][o]
            
            fold_ds = DNA_tf_dl(X_dict=curr_fold['input'],
                                X_transform_dict=X_transform_dict,
                                y_dict=curr_fold['output'],
                                y_transform_dict=y_transform_dict,
                                rev_comp_dict=add_rc,
                                batch_size=batch_size,
                                shuffle=False,
                                reshuffle_on_epoch_end=False)
            preds = model.predict(fold_ds)
            for o_index, o in enumerate(curr_fold['output'].keys()):
                temp_df[f'pred_{o}'] = preds[:, o_index]
            
            temp_df['chrom'] = curr_fold['index']['chrom_idx']
            temp_df['example_idx'] = curr_fold['index']['index']
            temp_df['fold_ID'] = k
            print(f"FOLD {k} RESULT:")
            print(temp_df.head())
            fold_res_df = fold_res_df.append(temp_df, ignore_index=True)

        test2_preds = model.predict(test2_ds)
        test2_df = pd.DataFrame()
        for o in test2_dict['output'].keys():
                test2_df[f'truth_{o}'] = test2_dict['output'][o]
        for o_index, o in enumerate(test2_dict['output'].keys()):
            test2_df[f'pred_{o}'] = test2_preds[:, o_index]
        test2_df['chrom'] = test2_dict['index']['chrom_idx']
        test2_df['example_idx'] = test2_dict['index']['index']
        test2_df['fold_ID'] = 'test2'
        
        print(f"Saving result to: {dataset_local_name}_{fold_fn_name}_fold_{i}_eval.csv")
        fold_res_df.to_csv(f'{dataset_local_name}_{fold_fn_name}_fold_{i}_eval.csv',
                           index=False)
        
        print(f"Saving test2 result to: {dataset_local_name}_{fold_fn_name}_fold_{i}_test2.csv")
        test2_df.to_csv(f'{dataset_local_name}_{fold_fn_name}_fold_{i}_test2.csv',
                           index=False)



