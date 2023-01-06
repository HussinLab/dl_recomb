import numpy as np
import h5py
import random 
import datetime
from Dataset import SubIntervalDataset
import json 

def get_test_indices(data_len, test_set_size, test_set_sampling):
    """
    This function takes a data length and creates test indices with the
    required proprtions and the required sampling strategy. It returns the test
    indices
    """
    # Get data every nth example as test set
    if test_set_sampling == 'uniform':
        test_set_idx = np.array([False]*data_len)
        sample_interval = int(1/test_set_size)
        # idx = test_set_idx[::sample_interval]
        # idx = True
        test_set_idx[::sample_interval] = True

    elif test_set_sampling == 'random':
        test_set_idx = np.random.choice(a=[True, False],
                                        size=(data_len, ),
                                        p=[test_set_size, 1-test_set_size])
    else:
        raise NotImplemented(f"Passed test_set_sampling {test_set_sampling} is not supported")

    return test_set_idx


def split_and_write_data_to_h5(fh5,
                               data_dict,
                               test_set_size,
                               test_set_sampling):
    """
    This function takes in the data dict and then splits it into train\test and
    saves each one in its respective folder.

    Since not necessarily that all data fields have the same length (Because we
    may not find enough negative examples at a certain interval, so in that
    case we will have less examples), we compute the data_len inside the
    for loop
    """

    for data_name, data_value in data_dict.items():
        data_len = len(data_dict[data_name])
        print(f"\tSplitting train\\test, data length = {data_len}")
        if test_set_size > 0:
            test_idx = get_test_indices(data_len,
                                        test_set_size,
                                        test_set_sampling)
            train_idx = ~test_idx
            print(f" test first 20 indices: {test_idx[:20]}, total_positive = {test_idx.sum()}")
        else:
            print(f"Passed dataset size is {test_set_size}, so skipping the split and all data will be saved to training set")
            train_idx = np.array([True]*data_len)
            test_idx = ~train_idx

        print(f"\t\tdata field: {data_name}, shape={data_value.shape}")
        train_data = data_value[train_idx]
        fh5.create_dataset(f'/train/{data_name}',
                           data=train_data,
                           compression="gzip",
                           compression_opts=9)

        test_data = data_value[test_idx]
        fh5.create_dataset(f'/test/{data_name}',
                           data=test_data,
                           compression="gzip",
                           compression_opts=9)


def create_h5_dataset_by_chromosome(df,
                                    output_fname,
                                    grch_obj,
                                    chroms_lens_dict,
                                    seq_len,
                                    negative_sampling_intervals,
                                    chr_col,
                                    start_col,
                                    end_col,
                                    seq_midpoint_col,
                                    output_cols_list,

                                    binarize_output,

                                    random_seed,

                                    pos_suffix='pos',
                                    neg_suffix='neg',

                                    seq_suffix='seq',
                                    metadata_suffix='meta',
                                    output_suffix='out',

                                    forbidden_negative_sampling_regions=None,

                                    test_set_size=0.1,
                                    test_set_sampling='uniform'):
    """
    test_set_sampling can be either uniform (i.e. every nth row) or random

    TODO: ADD CHROM LEN METADATA
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    dataset_file_name = f"{output_fname}.h5"
    f = h5py.File(dataset_file_name, "a")

    chr_groups = df.groupby(chr_col)

    metadata_dict = {}
    metadata_dict['chroms'] = []

    metadata_dict['dataset_params'] = {
                                        'type': 'by_chrom', 
                                        'random_seed': None,
                                        'date': str(datetime.datetime.now())
                                       }

    metadata_dict['names_suffix'] = {
                                      'positive_dataset': f"_{pos_suffix}_",
                                      'negative_dataset': f"_{neg_suffix}_",
                                      'model_input': f"_{seq_suffix}",
                                      'model_output': f"_{output_suffix}",
                                      'chrom_extra_info': f"_{metadata_suffix}"
                                    }

    metadata_dict['num_2_base_dict'] = grch_obj.metadata['num_2_base_dict']
    metadata_dict['encoding_dict']   = grch_obj.metadata['encoding_dict']
    metadata_dict['num_reverse_compliment_dict'] = grch_obj.metadata['num_reverse_compliment_dict']

    metadata_dict['positive_metadata_cols'] = ['start', 'end', 'seq_midpoint',
                                               'original_A', 'original_C',
                                               'original_T', 'original_G',
                                               'example_A', 'example_C',
                                               'example_T',
                                               'example_G'] + output_cols_list

    metadata_dict['negative_metadata_cols'] = ['start', 'end', 'seq_midpoint',
                                               'original_A', 'original_C',
                                               'original_T', 'original_G',
                                               'example_A', 'example_C',
                                               'example_T', 'example_G']

    metadata_dict['binarize_output'] = binarize_output

    metadata_dict['negative_sampling_intervals'] = negative_sampling_intervals

    metadata_dict['chroms_lens'] = chroms_lens_dict

    metadata_dict['test_set_size'] = test_set_size

    metadata_dict['random_seed'] = random_seed

    for chr_name, chr_df in chr_groups:
        print(f"Creating dataset for chromosome {chr_name}:")
        chrom_len = chroms_lens_dict[chr_name]
        subdataset_obj = SubIntervalDataset(pos_df=chr_df,
                                            min_inter_positive_examples=-1, # Negative value means don't care
                                            seq_len=seq_len,
                                            grch_obj=grch_obj,
                                            negative_sampling_intervals=negative_sampling_intervals,
                                            chr_col=chr_col,
                                            start_col=start_col,
                                            end_col=end_col,
                                            seq_midpoint_col=seq_midpoint_col,
                                            output_cols_list=output_cols_list,
                                            chrom_len=chrom_len,
                                            binarize_output=binarize_output,
                                            forbidden_negative_sampling_regions=forbidden_negative_sampling_regions)
        
        
        chr_pstv_model_in = subdataset_obj.positive_dataset.seq
        chr_pstv_model_out = subdataset_obj.positive_dataset.output
        chr_pstv_model_meta = subdataset_obj.positive_dataset.metadata
        
        data_dict = {}
        data_dict[f'{chr_name}_{pos_suffix}_{seq_suffix}'] = chr_pstv_model_in
        data_dict[f'{chr_name}_{pos_suffix}_{output_suffix}'] = chr_pstv_model_out
        data_dict[f'{chr_name}_{pos_suffix}_{metadata_suffix}'] = chr_pstv_model_meta
        
        for neg_name, neg_data in subdataset_obj.negative_datasets_dict.items():
            chr_ngtv_model_in = neg_data.seq
            chr_ngtv_model_meta = neg_data.metadata
        
            data_dict[f'{chr_name}_{neg_suffix}_{neg_name}_{seq_suffix}'] = chr_ngtv_model_in
            data_dict[f'{chr_name}_{neg_suffix}_{neg_name}_{metadata_suffix}'] = chr_ngtv_model_meta
        
        
        split_and_write_data_to_h5(f, data_dict, test_set_size, test_set_sampling)
        
        metadata_dict['chroms'].append(chr_name)
        
    f["/"].attrs["metadata"] = json.dumps(metadata_dict)
    f.close()
    print("Done, file closed successfully.")
