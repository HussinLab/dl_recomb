import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, fbeta_score
from functools import partial

def mc_pred_normal_model(in_model, dataset, outputs_names, n_samples, return_stats_only=True, return_as_df=True):
    """
    """
    all_res = {o:[] for o in outputs_names}
    
    for sample_idx in range(n_samples):
        print(f"\t MC Predicting sample {sample_idx+1} / {n_samples}")
        sample_accumulator = {o:[] for o in outputs_names}
        for x, y in dataset:
            mc_pred = in_model(x, training=True)
            for oi, o in enumerate(outputs_names):
                sample_accumulator[o].append(mc_pred[oi])
        for oi, o in enumerate(outputs_names):
            flattened_o = np.concatenate(sample_accumulator[o]).flatten()
            all_res[o].append(flattened_o)
        
    mc_preds_all = {k:np.array(v).T for k, v in all_res.items()}
    
    if return_stats_only:
        stats_dict = {}
        pred_mean = {f"{o}_mc_mean":mc_preds_all[o].mean(axis=1) for o in outputs_names}
        pred_std = {f"{o}_mc_std":mc_preds_all[o].mean(axis=1) for o in outputs_names}
        stats_dict.update(pred_mean)
        stats_dict.update(pred_std)
        
        if return_as_df:
            return pd.DataFrame(stats_dict)
        else:
            return stats_dict
    else:
        if return_as_df:
            cols = np.concatenate([mc_preds_all[o] for o in outputs_names], axis=1)
            col_names = [f"{o}_{i}" for o in outputs_names for i in range(n_samples)]
            return pd.DataFrame(cols, columns=col_names)
        else:
            return mc_preds_all


def mc_dropout_analyze_v2(normal_model, dataset, outputs_names, 
                          n_samples=100, return_stats_only=True, return_as_df=True):
    """
    in_model, dataset, outputs_names, n_samples, return_stats_only=True, return_as_df=True
    """
    preds = normal_model.predict(dataset)
    
    final_df_dict = {}
    for i, o in enumerate(outputs_names):
        final_df_dict[f'{o}_normal_pred'] = preds[i].flatten() #preds are a list
        final_df_dict[f'{o}_truth'] = dataset.y_dict[o].flatten()
    
    print(f"Regular model predictions done, starting MC prediction")
    mc_res = mc_pred_normal_model(normal_model, dataset, outputs_names, 
                         n_samples, return_stats_only, return_as_df)
    final_df_dict.update(mc_res)
    if return_as_df:
        return pd.DataFrame(final_df_dict)
    else:
        return final_df_dict

def analyze_mc_result(res_stats_df, outputs_names, thresh=0.5):
    # for metric in accuracy, precision, recall, f1:
    eval_funcs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, 
                  partial(fbeta_score, beta=1.2)]
    eval_funcs_names = ['accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score', 'fbeta_score']
    res_idx = []
    res_vals = []
    for i, f in enumerate(eval_funcs):
        curr_res = {}
        fname = eval_funcs_names[i]
        res_idx.append(fname)
        for o in outputs_names:
            o_truth = res_stats_df[f'{o}_truth'] > thresh
            o_pred_normal = res_stats_df[f'{o}_normal_pred'] > thresh
            o_pred_mc = res_stats_df[f'{o}_mc_mean'] > thresh
            
            curr_res[f'{o}_normal'] = f(o_truth, o_pred_normal)
            curr_res[f'{o}_mc'] = f(o_truth, o_pred_mc)
        res_vals.append(curr_res)
    
    metrics_res_df = pd.DataFrame(res_vals, index=res_idx)
    return metrics_res_df
