import tensorflow as tf
import h5py
import os
from .train_test_splits import *
import json
from .tf_dataloaders import DNA_tf_dl
import numpy as np


class IllegalSplitError(ValueError):
    pass


def get_midpoint(h5_file,
                 ds_name,
                 c,
                 metadata_suffix,
                 seq_midpoint_index,
                 normalize_midpoint,
                 chroms_lens_dict):
    assert isinstance(seq_midpoint_index, int)
    print(f"GETTING META {ds_name}{metadata_suffix}")
    midpoint = h5_file[f'{ds_name}{metadata_suffix}']
    midpoint = midpoint[:, seq_midpoint_index]

    if normalize_midpoint:
        chrom_len = chroms_lens_dict[str(c)]
        midpoint = midpoint/chrom_len

    return midpoint


def get_chroms_data_dict(h5_file,
                         negexamples_config,
                         input_suffix,
                         output_suffix,
                         positive_suffix,
                         negative_suffix,
                         metadata_suffix,
                         chroms_lens_dict,
                         seq_midpoint_index=None,
                         normalize_midpoint=True,
                         binarize_output=True,
                         separate_outputs=None,
                         chroms_to_get='all'):
    """[summary]

    Note: Although the design philosophy of the API wants to provide a multi
    output h5 datasets, for now this function handles only one output h5
    dataset name 'out'. This behavious is hardcoded, however it will be changed
    in the future.

    TODO: Change using a built-in out in the h5 dataset into a more flexible
          code that takes a list of ds names.

    Args:
        h5_file ([type]): [description]
        input_suffix ([type]): [description]
        output_suffix ([type]): [description]
        metadata_suffix ([type]): [description]
        chroms_lens_dict ([type]): [description]
        seq_midpoint_index ([integer or None], optional): an integer or None.
                    If None, we won't get the sequence midpoint as a feature,
                    else, the the integer is the index of the sequence midpoint
                    in the metadata table. Defaults to None.
        normalize_midpoint (bool, optional): will make the midpoint a number
                    between 0 and 1 by  dividing by the chromosome length.
                    Defaults to True.
        binarize_output (bool, optional): [description]. Defaults to True.
        separate_outputs (List or None, optional): If a list is provided, it is
                    assumed that the output is an MxN array. The function will
                    separate it into a dictionary of Nx1 arrays with names
                    passed in this list. Defaults to None.
        chroms_to_get (str, optional): [description]. Defaults to 'all'.

    Returns:
        [type]: [description]
    """

    chrs_id_dict = {i: (i-1) for i in range(1, 23)}
    chrs_id_dict['X'] = 22
    chrs_id_dict['Y'] = 23

    if chroms_to_get == 'all':
        chroms_to_get = list(range(1, 23)) + ['X', 'Y']
    else:
        assert isinstance(chroms_to_get, list)

    result_dict = {}
    for c in chroms_to_get:
        # data_dict will hold the return of 1 chromosome. It has keys of:
        # seq, a key for each output_name, midpoint, chrom_idx, index
        # Another dict, result_dict holds all the data.
        data_dict = {}

        # example_set_index is used to keep track where each example came from.
        # it contains the name of the set. The index within the set is saved
        # in the 'index' entry
        example_set_index = []
        exmpl_original_index = []
        midpoints = []

        pos_seq = h5_file[f'chr_{c}{positive_suffix}{input_suffix}'][:]

        pos_out = h5_file[f'chr_{c}{positive_suffix}{output_suffix}'][:]
        n_pos = pos_out.shape[0]
        if seq_midpoint_index:
            pos_midpoint = get_midpoint(h5_file,
                                        f'chr_{c}{positive_suffix}',
                                        c,
                                        metadata_suffix,
                                        seq_midpoint_index,
                                        normalize_midpoint,
                                        chroms_lens_dict)
            midpoints.append(pos_midpoint)

        example_set_index += [positive_suffix]*n_pos
        exmpl_original_index += list(range(n_pos))

        # TODO: Construct negative seq
        # For each negative strategy config, get the required number of
        # examples
        neg_seq_list = []
        neg_indices_setname = []
        total_n_neg = 0
        for k, v in negexamples_config.items():
            if v > 0.0:
                if k == 'shuffle_pos':
                    # We cannot have more positive than negative for now, to
                    # simplify the code. TODO: Continue shuffling to generate
                    # an even larger dataset if required.
                    n_neg = np.min([int(n_pos*v), n_pos])

                    # Sample from pos. The -1 is because
                    # n_pos is coming from shape, so the int is the full size
                    # and not a 0 based index count.
                    random_indices = np.random.choice(n_pos - 1,
                                                      size=n_neg,
                                                      replace=False)

                    # Get a copy of the sampled positive examples
                    neg_examples = np.copy(pos_seq[random_indices, :])
                    np.random.shuffle(neg_examples)
                    if seq_midpoint_index:
                        neg_midpoint = pos_midpoint[random_indices]

                else:
                    neg_ds_name = f'chr_{c}{negative_suffix}{k}{input_suffix}'
                    neg_seq = h5_file[neg_ds_name][:]
                    neg_ds_size = neg_seq.shape[0]
                    print(neg_ds_size, n_pos*v)
                    n_neg = np.min([int(n_pos*v), neg_ds_size])
                    random_indices = np.random.choice(neg_ds_size,
                                                      size=n_neg,
                                                      replace=False)
                    # Get a copy of the sampled positive examples
                    neg_seq = neg_seq[random_indices]
                    if seq_midpoint_index:
                        dssuffix = f"chr_{c}{negative_suffix}{k}"
                        neg_midpoint = get_midpoint(h5_file,
                                                    dssuffix,
                                                    c,
                                                    metadata_suffix,
                                                    seq_midpoint_index,
                                                    normalize_midpoint,
                                                    chroms_lens_dict)

                midpoints.append(neg_midpoint)
                total_n_neg += n_neg
                example_set_index += [k]*n_neg
                exmpl_original_index += list(random_indices)
                neg_indices_setname.append([k]*n_neg)
                neg_seq_list.append(neg_seq)

        # TODO: Concat pos to seq and order by position
        all_seqs = [pos_seq] + neg_seq_list

        data_dict['seq'] = np.concatenate(all_seqs)
        n_ex = data_dict['seq'].shape[0]
        print(f"\n\n{n_ex} Training examples generated for chrom {c}")
        print(f"\t{n_pos} positive, {total_n_neg} negative")
        # Although now positive and negative are separated, which means that
        # for binary classification an out dataset isn't needed (can be
        # inferred from _pos_ or _neg_ datasets), but it was kept to keep the
        # code keeping track of the original dataset read values.
        out = h5_file[f'chr_{c}{positive_suffix}{output_suffix}'][:]
        if binarize_output:
            out[out > 0] = 1.

        # Add the output to the negative examples, which is basically 0's
        n_outs = out.shape[1]

        neg_outs = np.zeros((total_n_neg, n_outs))
        out = np.concatenate([out, neg_outs])

        if separate_outputs:
            for idx, name in enumerate(separate_outputs['out']):
                data_dict[name] = out[:, idx]
        else:
            data_dict['out'] = out

        if seq_midpoint_index:
            data_dict['midpoint'] = np.concatenate(midpoints)
            # this idx is an array or chr_{num} with the same length as the
            # DNA sequence. Used to track where the example came from after
            # shuffling the data. Of course a better one index coule be built
            # TODO: Make the index one number made of two parts, chr ID and idx
            idx = np.array([chrs_id_dict[c]]*data_dict['seq'].shape[0])
            data_dict['chrom_idx'] = idx

        # ds_index keeps track of the index of the example from within the
        # dataset. However, we need another index to be used for shuffling.
        # Therefore, we create another entry 'index', which will contain the
        # usual monotonically increasing integer position, and this will be
        # the one used for shuffling
        data_dict['ds_index'] = np.array(exmpl_original_index)
        data_dict['index'] = np.arange(data_dict['seq'].shape[0])
        data_dict['ex_type'] = np.array(example_set_index)

        result_dict[c] = data_dict

    return result_dict


def load_and_create_dictionary(dataset_config,
                               data_io_config,
                               negexamples_config,
                               fold_name,
                               chroms_to_use='all'):
    # Still considering if we should enforce balanced problem, or consider it
    # the problem of the configuration files
    # neg_sum = np.sum([v for v in negexamples_config.values])
    # assert np.isclose(neg_sum, 1)

    h5_path = dataset_config['dataset_name']

    # Load the file and get the necessary information
    # # H5 will create a file if it doesn't exist, so we handle this
    if os.path.isfile(h5_path):
        froot = h5py.File(h5_path, "r")
        print(f"\n\nFile {h5_path} was successfully opened")
    else:
        raise FileNotFoundError(f"File {h5_path} does not exist")

    folds_available = list(froot['/'].keys())
    print("\n\nFile opened. Found the following Folds:")
    print(folds_available)

    if fold_name not in froot['/'].keys():
        raise ValueError(f"{fold_name} was not found in the hdf5 file")
    else:
        f = froot[f'./{fold_name}']

    datasets_available = list(f.keys())
    print("\n\nFold folder opened. Found the following datasets:")
    print(datasets_available)

    metadata = json.loads(froot['/'].attrs["metadata"])
    print("\n\nDataset Metadata:")
    print(metadata)

    # Workaround:
    # usually positive_dataset _pos_ and model_input is _seq. Problem is that
    # there are two underscore, and concatenating will create _pos__seq, which
    # is wrong, so we read the positive_dataset without the last underscore.
    # This problem is not there in the negative dataset, we need the 2nd
    # underscore
    positive_suffix = metadata['names_suffix']['positive_dataset'][:-1]
    negative_suffix = metadata['names_suffix']['negative_dataset']

    input_suffix = metadata['names_suffix']['model_input']

    output_suffix = metadata['names_suffix']['model_output']
    metadata_suffix = metadata['names_suffix']['chrom_extra_info']
    chroms_lens_dict = dataset_config['chroms_lens_dict']

    input_keys = data_io_config['input_keys']
    output_keys = data_io_config['output_keys']
    index_keys = data_io_config['index_keys']

    # Neg examples config has a key int for index of the strategy and value
    # to be the proportion of the positive examples. replace the integer with
    # the proper names. This is done because the names can be complicated and
    # prone to typos, while just typing an integer has less chance of errors
    neg_intrvls = metadata['negative_sampling_intervals']

    negs_by_names = {}
    for k, v in negexamples_config.items():
        if isinstance(k, int):
            intrvl_start = neg_intrvls[k][0]
            intrvl_end = neg_intrvls[k][1]

            # neg_name = f"({intrvl_start}, {intrvl_end})"
            neg_name = (intrvl_start, intrvl_end)
            negs_by_names[neg_name] = v
        else:
            negs_by_names[k] = v

    print("\n\nRenamed Negative Examples:")
    print(negs_by_names)

    # If separate_outputs is true, this means that instead of having the last
    # layer have n neurons, each output will have an independent before-last
    # layer, then a single neuron as output. In this case, the output_keys
    # dictionary need to be changed from the default name 'out' to a list of
    # each individual output.
    if 'separate_outputs' in data_io_config.keys():
        separate_outputs = data_io_config['separate_outputs']
        if isinstance(separate_outputs, dict):
            for k, v in separate_outputs.items():
                print("\n\n")
                print(f"Expanding output {k} into its individual columns {v}")

                # Remove the original name
                output_keys.remove(k)
                # Add the names passed
                output_keys += v
    else:
        separate_outputs = None

    data_dict = get_chroms_data_dict(h5_file=f,
                                     negexamples_config=negs_by_names,
                                     input_suffix=input_suffix,
                                     output_suffix=output_suffix,
                                     positive_suffix=positive_suffix,
                                     negative_suffix=negative_suffix,
                                     metadata_suffix=metadata_suffix,
                                     chroms_lens_dict=chroms_lens_dict,
                                     seq_midpoint_index=2,
                                     normalize_midpoint=True,
                                     binarize_output=True,
                                     separate_outputs=separate_outputs,
                                     chroms_to_get=chroms_to_use)

    froot.close()

    return data_dict


def create_tf_ds(dataset_config,
                 train_test_split_config,
                 data_io_config):
    """[summary]

    Args:
        dataset_config ([type]): [description]
        train_test_split_config ([type]): [description]
        data_io_config ([type]): [description]

    Returns:
        [type]: [description]
    """
    h5_path = dataset_config['dataset_name']
    f = h5py.File(h5_path)
    metadata = json.loads(f['/'].attrs["metadata"])
    f.close()

    split_strategy = train_test_split_config['strategy']

    use_rc = data_io_config['use_rev_compl_as_input']

    rev_com_dict_str = metadata['num_reverse_compliment_dict']
    rev_com_dict = {int(k): v for k, v in rev_com_dict_str.items()}

    input_keys = data_io_config['input_keys']
    output_keys = data_io_config['output_keys']
    index_keys = data_io_config['index_keys']

    data_dict = load_and_create_dictionary(dataset_config, data_io_config)

    print("=========================================")
    print(data_dict[1].keys())
    # Train-test split
    # whole_chroms, partial_chrom_contig, partial_chrom_random, by_random
    # # Construct X and y

    if split_strategy == "whole_chroms":
        train_chroms = train_test_split_config['train']
        val_chroms = train_test_split_config['val']
        test_chroms = train_test_split_config['test']

        train_tuple, val_tuple, test_tuple = whole_chrom_ds_split(
                                                                  data_dict,
                                                                  train_chroms,
                                                                  val_chroms,
                                                                  test_chroms,
                                                                  input_keys,
                                                                  output_keys,
                                                                  index_keys)

    elif split_strategy == "partial_chrom_contig":
        chroms_list = dataset_config['chroms_to_use']
        train_perc = train_test_split_config['train']
        val_perc = train_test_split_config['val']
        test_perc = train_test_split_config['test']

        train_tuple, val_tuple, test_tuple = partial_chrom_contig_ds_split(
                                                                  data_dict,
                                                                  chroms_list,
                                                                  train_perc,
                                                                  val_perc,
                                                                  test_perc,
                                                                  input_keys,
                                                                  output_keys,
                                                                  index_keys)

    elif split_strategy == "partial_chrom_random":
        chroms_list = dataset_config['chroms_to_use']
        train_perc = train_test_split_config['train']
        val_perc = train_test_split_config['val']
        test_perc = train_test_split_config['test']
        random_seed = train_test_split_config['random_seed']

        train_val_test_tuples = partial_chrom_shuffled_ds_split(chroms_list,
                                                                train_perc,
                                                                val_perc,
                                                                test_perc,
                                                                input_keys,
                                                                output_keys,
                                                                index_keys,
                                                                random_seed)

        train_tuple = train_val_test_tuples[0]
        val_tuple = train_val_test_tuples[1]
        test_tuple = train_val_test_tuples[2]

    elif split_strategy == "by_random":
        raise NotImplementedError
    else:
        raise IllegalSplitError(f"Unrecognized train-test split strategy \
                                    {split_strategy}.")

    # convert each fold to tf_ds

    # seq_len = f[f'chr_{c}{input_suffix}'][0].shape[0]
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

    print("Creating training TF dataset:")
    train_ds = DNA_tf_dl(X_dict=train_tuple[0],
                         X_transform_dict=X_transform_dict,
                         y_dict=train_tuple[1],
                         y_transform_dict=y_transform_dict,
                         rev_comp_dict=add_rc,
                         batch_size=batch_size,
                         shuffle=shuffle_train,
                         reshuffle_on_epoch_end=False)

    val_ds = DNA_tf_dl(X_dict=val_tuple[0],
                       X_transform_dict=X_transform_dict,
                       y_dict=val_tuple[1],
                       y_transform_dict=y_transform_dict,
                       rev_comp_dict=add_rc,
                       batch_size=batch_size,
                       shuffle=shuffle_val,
                       reshuffle_on_epoch_end=False)

    test_ds = DNA_tf_dl(X_dict=test_tuple[0],
                        X_transform_dict=X_transform_dict,
                        y_dict=test_tuple[1],
                        y_transform_dict=y_transform_dict,
                        rev_comp_dict=add_rc,
                        batch_size=batch_size,
                        shuffle=shuffle_test,
                        reshuffle_on_epoch_end=False)

    indices = (train_tuple[2], val_tuple[2], test_tuple[2])
    # train_ds = create_dataset_from_names(f,
    #                                      chroms_train,
    #                                      input_suffix,
    #                                      output_suffix,
    #                                      batch_size,
    #                                      add_rc,
    #                                      shuffle_train)

    # print("Creating validation TF dataset:")
    # val_ds   = create_dataset_from_names(f,
    #                                      chroms_val,
    #                                      input_suffix,
    #                                      output_suffix,
    #                                      batch_size,
    #                                      add_rc,
    #                                      shuffle_val)

    # print("Creating test TF dataset:")
    # test_ds  = create_dataset_from_names(f,
    #                                      chroms_test,
    #                                      input_suffix,
    #                                      output_suffix,
    #                                      batch_size,
    #                                      add_rc,
    #                                      shuffle_test)

    # return train_ds, val_ds, test_ds#, indices_dict
    # return train_tuple, val_tuple, test_tuple
    return train_ds, val_ds, test_ds, indices
