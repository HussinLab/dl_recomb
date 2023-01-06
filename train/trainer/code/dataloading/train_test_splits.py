import numpy as np
from numpy.lib.index_tricks import ix_


def get_chroms_data(data_dict,
                    chroms_list,
                    input_keys,
                    output_keys,
                    index_keys):
    """[summary]

    Args:
        data_dict ([type]): [description]
        chroms_list ([type]): [description]
        input_keys ([type]): [description]
        output_keys ([type]): [description]
        index_keys ([type]): [description]

    Returns:
        [type]: [description]
    """
    X_dict = {i: [] for i in input_keys}
    y_dict = {i: [] for i in output_keys}
    idx_dict = {i: [] for i in index_keys}

    for c in chroms_list:
        for k in input_keys:
            X_dict[k].append(data_dict[c][k])

        for k in output_keys:
            y_dict[k].append(data_dict[c][k])

        for k in index_keys:
            idx_dict[k].append(data_dict[c][k])

    return X_dict, y_dict, idx_dict


def whole_chrom_ds_split(data_dict,
                         train_chroms,
                         val_chroms,
                         test_chroms,
                         input_keys,
                         output_keys,
                         index_keys):
    """[summary]

    Args:
        data_dict ([type]): [description]
        train_chroms ([type]): [description]
        val_chroms ([type]): [description]
        test_chroms ([type]): [description]
        input_keys ([type]): [description]
        output_keys ([type]): [description]
        index_keys ([type]): [description]

    Returns:
        [type]: [description]
    """
    X_train_dict, y_train_dict, idx_train_dict = get_chroms_data(data_dict,
                                                                 train_chroms,
                                                                 input_keys,
                                                                 output_keys,
                                                                 index_keys)

    X_val_dict, y_val_dict, idx_val_dict = get_chroms_data(data_dict,
                                                           val_chroms,
                                                           input_keys,
                                                           output_keys,
                                                           index_keys)

    X_test_dict, y_test_dict, idx_test_dict = get_chroms_data(data_dict,
                                                              test_chroms,
                                                              input_keys,
                                                              output_keys,
                                                              index_keys)

    train_tuple = (X_train_dict, y_train_dict, idx_train_dict)
    val_tuple = (X_val_dict, y_val_dict, idx_val_dict)
    test_tuple = (X_test_dict, y_test_dict, idx_test_dict)

    return train_tuple, val_tuple, test_tuple


def flatten_dict_of_lists(dict_in):
    """
    This function take a dictionary where each value is a list of arrays,
    and makes is only one array. The reason we need this is that we build the
    data by appending the arrays to a list, so this is the last step in the
    preprocessing pipeline

    Args:
        dict_in ([type]): [description]
    """
    dict_out = {}
    for k, v in dict_in.items():
        dict_out[k] = np.concatenate(dict_in[k])
    return dict_out


def partial_chrom_contig_ds_split(data_dict,
                                  chroms_list,
                                  train_perc,
                                  val_perc,
                                  test_perc,
                                  input_keys,
                                  output_keys,
                                  index_keys):
    """[summary]

    Args:
        data_dict ([type]): [description]
        chroms_list ([type]): [description]
        train_perc ([type]): [description]
        val_perc ([type]): [description]
        test_perc ([type]): [description]
        input_keys ([type]): [description]
        output_keys ([type]): [description]
        index_keys ([type]): [description]

    Returns:
        [type]: [description]
    """

    assert (train_perc + val_perc + test_perc) == 1
    assert isinstance(input_keys, list)
    assert isinstance(output_keys, list)
    assert isinstance(index_keys, list)

    X_train_dict = {i: [] for i in input_keys}
    y_train_dict = {i: [] for i in output_keys}
    idx_train_dict = {i: [] for i in index_keys}

    X_val_dict = {i: [] for i in input_keys}
    y_val_dict = {i: [] for i in output_keys}
    idx_val_dict = {i: [] for i in index_keys}

    X_test_dict = {i: [] for i in input_keys}
    y_test_dict = {i: [] for i in output_keys}
    idx_test_dict = {i: [] for i in index_keys}

    for c in chroms_list:
        chrom_data = data_dict[c]
        # Measure the size of any of the inputs
        data_len = len(chrom_data[input_keys[0]])

        train_end_idx = int(data_len*train_perc)
        val_end_idx = train_end_idx + int(data_len*val_perc)

        for k in input_keys:
            X_train_dict[k].append(chrom_data[k][:train_end_idx])
            X_val_dict[k].append(chrom_data[k][train_end_idx:val_end_idx])
            X_test_dict[k].append(chrom_data[k][val_end_idx:])

        for k in output_keys:
            y_train_dict[k].append(chrom_data[k][:train_end_idx])
            y_val_dict[k].append(chrom_data[k][train_end_idx:val_end_idx])
            y_test_dict[k].append(chrom_data[k][val_end_idx:])

        for k in index_keys:
            idx_train_dict[k].append(chrom_data[k][:train_end_idx])
            idx_val_dict[k].append(chrom_data[k][train_end_idx:val_end_idx])
            idx_test_dict[k].append(chrom_data[k][val_end_idx:])

    # Convert from a list of arrays to one array
    X_train_dict = flatten_dict_of_lists(X_train_dict)
    X_val_dict = flatten_dict_of_lists(X_val_dict)
    X_test_dict = flatten_dict_of_lists(X_test_dict)

    y_train_dict = flatten_dict_of_lists(y_train_dict)
    y_val_dict = flatten_dict_of_lists(y_val_dict)
    y_test_dict = flatten_dict_of_lists(y_test_dict)

    idx_train_dict = flatten_dict_of_lists(idx_train_dict)
    idx_val_dict = flatten_dict_of_lists(idx_val_dict)
    idx_test_dict = flatten_dict_of_lists(idx_test_dict)

    train_tuple = (X_train_dict, y_train_dict, idx_train_dict)
    val_tuple = (X_val_dict, y_val_dict, idx_val_dict)
    test_tuple = (X_test_dict, y_test_dict, idx_test_dict)

    return train_tuple, val_tuple, test_tuple


def partial_chrom_shuffled_ds_split(data_dict,
                                    chroms_list,
                                    train_perc,
                                    val_perc,
                                    test_perc,
                                    input_keys,
                                    output_keys,
                                    index_keys,
                                    random_seed):
    """[summary]

    Args:
        data_dict (dict): [description]
        chroms_list (list): [description]
        train_perc (float): [description]
        val_perc (float): [description]
        test_perc (float): [description]
        input_keys (list): [description]
        output_keys (list): [description]
        index_keys (list): [description]
        random_seed (int): [description]

    Returns:
        tuple: [description]
    """

    assert (train_perc + val_perc + test_perc) == 1
    assert isinstance(input_keys, list)
    assert isinstance(output_keys, list)
    assert isinstance(index_keys, list)

    np.random.seed(random_seed)

    X_train_dict = {i: [] for i in input_keys}
    y_train_dict = {i: [] for i in output_keys}
    idx_train_dict = {i: [] for i in index_keys}

    X_val_dict = {i: [] for i in input_keys}
    y_val_dict = {i: [] for i in output_keys}
    idx_val_dict = {i: [] for i in index_keys}

    X_test_dict = {i: [] for i in input_keys}
    y_test_dict = {i: [] for i in output_keys}
    idx_test_dict = {i: [] for i in index_keys}

    for c in chroms_list:
        chrom_data = data_dict[c]
        # Measure the size of any of the inputs
        data_len = len(chrom_data[input_keys[0]])
        shuffled_idx = np.random.shuffle(np.arange(data_len))

        train_end_idx = int(data_len*train_perc)
        val_end_idx = int(data_len*val_perc)

        for k in input_keys:
            in_data = chrom_data[k][shuffled_idx]
            X_train_dict[k].append(in_data[:train_end_idx])
            X_val_dict[k].append(in_data[train_end_idx:val_end_idx])
            X_test_dict[k].append(in_data[val_end_idx:])

        for k in output_keys:
            out_data = chrom_data[k][shuffled_idx]
            y_train_dict[k].append(out_data[:train_end_idx])
            y_val_dict[k].append(out_data[train_end_idx:val_end_idx])
            y_test_dict[k].append(out_data[val_end_idx:])

        for k in index_keys:
            idx_data = chrom_data[k][shuffled_idx]
            idx_train_dict[k].append(idx_data[:train_end_idx])
            idx_val_dict[k].append(idx_data[train_end_idx:val_end_idx])
            idx_test_dict[k].append(idx_data[val_end_idx:])

    # Convert from a list of arrays to one array
    X_train_dict = flatten_dict_of_lists(X_train_dict)
    X_val_dict = flatten_dict_of_lists(X_val_dict)
    X_test_dict = flatten_dict_of_lists(X_test_dict)

    y_train_dict = flatten_dict_of_lists(y_train_dict)
    y_val_dict = flatten_dict_of_lists(y_val_dict)
    y_test_dict = flatten_dict_of_lists(y_test_dict)

    idx_train_dict = flatten_dict_of_lists(idx_train_dict)
    idx_val_dict = flatten_dict_of_lists(idx_val_dict)
    idx_test_dict = flatten_dict_of_lists(idx_test_dict)

    train_tuple = (X_train_dict, y_train_dict, idx_train_dict)
    val_tuple = (X_val_dict, y_val_dict, idx_val_dict)
    test_tuple = (X_test_dict, y_test_dict, idx_test_dict)

    return train_tuple, val_tuple, test_tuple


def whole_chrom_k_fold(data_dict,
                       n_folds,
                       chroms_list,
                       train_perc,
                       test_perc,
                       input_keys,
                       output_keys,
                       index_keys,
                       random_seed,
                       test2_interval):
    """[summary]

    Args:
        data_dict ([type]): [description]
        n_folds ([type]): [description]
        chroms_list ([type]): [description]
        train_perc ([type]): [description]
        test_perc ([type]): [description]
        input_keys ([type]): [description]
        output_keys ([type]): [description]
        index_keys ([type]): [description]
        random_seed ([type]): [description]
        test2_interval ([type]): Every nth chromosome becomes a test2. Passing
                                 a 0 or negative number means no test2. Note
                                 that in this function, every nth is on the
                                 level of chromosomes, not individual example
                                 as in other splits
    """
    # NOt sure if I will implement this, splitting is hard to automate
    # TODO: Decide on this
    pass


def create_empty_fold_data_dict(input_keys, output_keys, index_keys):
    """[summary]

    Args:
        input_keys ([type]): [description]
        output_keys ([type]): [description]
        index_keys ([type]): [description]
    """
    test2_dict = {'input': {}, 'output': {}, 'index': {}}
    # Init folds list of dict
    for in_k in input_keys:
        test2_dict['input'][in_k] = []

    for out_k in output_keys:
        test2_dict['output'][out_k] = []

    # These are used to keep track which example came from where
    for ix_k in index_keys:
        test2_dict['index'][ix_k] = []

    return test2_dict


def init_fold_dict(n_folds,
                   input_keys,
                   output_keys,
                   index_keys):
    folds = []
    for i in range(n_folds):
        folds.append(create_empty_fold_data_dict(input_keys,
                                                 output_keys,
                                                 index_keys))

    return folds


def train_val_split_dicts_by_interval(data_dict,
                                      input_keys,
                                      output_keys,
                                      index_keys,
                                      test2_interval):
    """[summary]

    Args:
        data_dict ([type]): [description]
        input_keys ([type]): [description]
        output_keys ([type]): [description]
        index_keys ([type]): [description]
        test2_interval ([type]): [description]
    """
    test2_dict = create_empty_fold_data_dict(input_keys,
                                             output_keys,
                                             index_keys)
    if test2_interval == 0:
        print("test2_interval is 0, nothing to split.")
        return data_dict, test2_dict
    chroms_list = list(data_dict.keys())
    for c in chroms_list:
        # Measure the size of any of the inputs
        chrom_data = data_dict[c].copy()
        data_len = chrom_data['index'].shape[0]

        all_idx = np.arange(data_len)
        test2_idx = np.arange(0, data_len, test2_interval)

        train_idx = np.array([i for i in all_idx if i not in test2_idx])

        for in_k in input_keys:
            test_data = chrom_data[in_k][test2_idx]
            train_data = chrom_data[in_k][train_idx]
            test2_dict['input'][in_k].append(test_data)
            data_dict[c][in_k] = train_data

        for out_k in output_keys:
            test_data = chrom_data[out_k][test2_idx]
            train_data = chrom_data[out_k][train_idx]
            test2_dict['output'][out_k].append(test_data)
            data_dict[c][out_k] = train_data

        for i in index_keys:
            test_data = chrom_data[i][test2_idx]
            train_data = chrom_data[i][train_idx]
            test2_dict['index'][i].append(test_data)
            data_dict[c][i] = train_data

    test2_dict['input'] = flatten_dict_of_lists(test2_dict['input'])
    test2_dict['output'] = flatten_dict_of_lists(test2_dict['output'])
    test2_dict['index'] = flatten_dict_of_lists(test2_dict['index'])

    return data_dict, test2_dict


def extract_test2_data_from_folds(folds,
                                  input_keys,
                                  output_keys,
                                  index_keys,
                                  test2_dict,
                                  test2_interval):
    """[summary]
    TODO: Convert using index_keys instead of hard coded 'chrom_idx' and 'idx
          current problem is that each has a special logic, chrom_idx we
          create by ['chrom_idx'] * data_len while interval works normally.
          A solution could be to just create the chrom_idx outside the
          functions
    Args:
        folds ([type]): [description]
        input_keys ([type]): [description]
        output_keys ([type]): [description]
        index_keys ([type]): [description]
        test2_dict ([type]): [description]
        test2_interval ([type]): [description]

    Returns:
        [type]: [description]
    """
    n_folds = len(folds)

    # Handle the testing dataset
    if test2_interval > 0:
        for k in range(n_folds):
            sample_input = input_keys[0]
            fold_size = len(folds[k]['input'][sample_input])
            sampling_idx = np.array([False]*fold_size)
            sampling_idx[::test2_interval] = True

            for in_k in input_keys:
                test_data = folds[k]['input'][in_k][sampling_idx]
                train_data = folds[k]['input'][in_k][~sampling_idx]
                test2_dict['input'][in_k].append(test_data)
                folds[k]['input'][in_k] = train_data

            for out_k in output_keys:
                test_data = folds[k]['output'][out_k][sampling_idx]
                train_data = folds[k]['output'][out_k][~sampling_idx]
                test2_dict['output'][out_k].append(test_data)
                folds[k]['output'][out_k] = train_data

            for i in ['chrom_idx', 'index']:
                test_data = folds[k]['index'][i][sampling_idx]
                train_data = folds[k]['index'][i][~sampling_idx]
                test2_dict['index'][i].append(test_data)
                folds[k]['index'][i] = train_data

        test2_dict['input'] = flatten_dict_of_lists(test2_dict['input'])
        test2_dict['output'] = flatten_dict_of_lists(test2_dict['output'])
        test2_dict['index'] = flatten_dict_of_lists(test2_dict['index'])

    return folds, test2_dict


def partial_chrom_contig_k_fold(data_dict,
                                n_folds,
                                chroms_list,
                                train_perc,
                                test_perc,
                                input_keys,
                                output_keys,
                                index_keys,
                                random_seed):
    """[summary]

    Args:
        data_dict ([type]): [description]
        n_folds ([type]): [description]
        chroms_list ([type]): [description]
        train_perc ([type]): [description]
        test_perc ([type]): [description]
        input_keys ([type]): [description]
        output_keys ([type]): [description]
        index_keys ([type]): [description]
        random_seed ([type]): [description]
    """
    np.random.seed(random_seed)

    folds = init_fold_dict(n_folds, input_keys, output_keys, index_keys)

    for c in chroms_list:
        # Measure the size of any of the inputs
        chrom_data = data_dict[c]
        data_len = len(chrom_data[input_keys[0]])
        fold_len = data_len/n_folds

        for k in range(n_folds):
            start_idx = int(fold_len*k)
            end_idx = int(fold_len*(k+1))
            folds[k]['index']['chrom_idx'].append([c]*(end_idx - start_idx))
            folds[k]['index']['index'].append(list(range(start_idx, end_idx)))

            for in_k in input_keys:
                if not isinstance(chrom_data[in_k], np.ndarray):
                    chrom_data[in_k] = np.array(chrom_data[in_k])
                in_data = chrom_data[in_k]
                folds[k]['input'][in_k].append(in_data[start_idx:end_idx])

            for out_k in output_keys:
                if not isinstance(chrom_data[out_k], np.ndarray):
                    chrom_data[out_k] = np.array(chrom_data[out_k])
                out_data = chrom_data[out_k]
                folds[k]['output'][out_k].append(out_data[start_idx:end_idx])

    for k in range(n_folds):
        folds[k]['input'] = flatten_dict_of_lists(folds[k]['input'])
        folds[k]['output'] = flatten_dict_of_lists(folds[k]['output'])
        folds[k]['index'] = flatten_dict_of_lists(folds[k]['index'])

    return folds


def partial_chrom_contig_alternate_k_fold(data_dict,
                                          n_folds,
                                          chroms_list,
                                          train_perc,
                                          test_perc,
                                          input_keys,
                                          output_keys,
                                          index_keys,
                                          random_seed):
    """
    Implementation of Raph's idea to alternate fold segments among
    chromosomes, so that models would see a little bit of everywhere
    no matter which fold it is

    Args:
        data_dict ([type]): [description]
        n_folds ([type]): [description]
        chroms_list ([type]): [description]
        train_perc ([type]): [description]
        test_perc ([type]): [description]
        input_keys ([type]): [description]
        output_keys ([type]): [description]
        index_keys ([type]): [description]
        random_seed ([type]): [description]
    """
    np.random.seed(random_seed)

    folds = init_fold_dict(n_folds, input_keys, output_keys, index_keys)

    for chrom_i, c in enumerate(chroms_list):
        # Measure the size of any of the inputs
        chrom_data = data_dict[c]
        data_len = len(chrom_data[input_keys[0]])
        fold_len = data_len/n_folds

        for k in range(n_folds):
            chrom_k_pos = ((chrom_i-1) + k) % n_folds
            start_idx = int(fold_len*chrom_k_pos)
            end_idx = int(fold_len*(chrom_k_pos+1))
            folds[k]['index']['chrom_idx'].append([c]*(end_idx - start_idx))
            folds[k]['index']['index'].append(list(range(start_idx, end_idx)))

            for in_k in input_keys:
                if not isinstance(chrom_data[in_k], np.ndarray):
                    chrom_data[in_k] = np.array(chrom_data[in_k])
                in_data = chrom_data[in_k]
                folds[k]['input'][in_k].append(in_data[start_idx:end_idx])

            for out_k in output_keys:
                if not isinstance(chrom_data[out_k], np.ndarray):
                    chrom_data[out_k] = np.array(chrom_data[out_k])
                out_data = chrom_data[out_k]
                folds[k]['output'][out_k].append(out_data[start_idx:end_idx])

    for k in range(n_folds):
        folds[k]['input'] = flatten_dict_of_lists(folds[k]['input'])
        folds[k]['output'] = flatten_dict_of_lists(folds[k]['output'])
        folds[k]['index'] = flatten_dict_of_lists(folds[k]['index'])

    return folds


def partial_chrom_shuffled_k_fold(data_dict,
                                  n_folds,
                                  chroms_list,
                                  train_perc,
                                  test_perc,
                                  input_keys,
                                  output_keys,
                                  index_keys,
                                  random_seed):
    """[summary]

    Args:
        data_dict ([type]): [description]
        n_folds ([type]): [description]
        chroms_list ([type]): [description]
        train_perc ([type]): [description]
        test_perc ([type]): [description]
        input_keys ([type]): [description]
        output_keys ([type]): [description]
        index_keys ([type]): [description]
        random_seed ([type]): [description]

    Returns:
        [type]: [description]
    """
    np.random.seed(random_seed)

    folds = init_fold_dict(n_folds, input_keys, output_keys, index_keys)
    print(folds)
    for c in chroms_list:
        # Measure the size of any of the inputs
        chrom_data = data_dict[c]
        data_len = len(data_dict[c][input_keys[0]])
        fold_len = data_len/n_folds
        shuffled_idx = np.arange(data_len)
        np.random.shuffle(shuffled_idx)

        for k in range(n_folds):
            start_idx = int(fold_len*k)
            end_idx = int(fold_len*(k+1))
            # chromosome index needs some special treatment to be created,
            # since we create it from scratch. However, the rest of the
            # indexing will be sliced as is
            folds[k]['index']['chrom_idx'].append([c]*(end_idx - start_idx))

            for ix_k in index_keys:
                if ix_k == 'chrom_idx':
                    continue
                ix_k_data = chrom_data[ix_k][shuffled_idx]
                folds[k]['index'][ix_k].append(ix_k_data[start_idx:end_idx])

            for in_k in input_keys:
                if not isinstance(chrom_data[in_k], np.ndarray):
                    chrom_data[in_k] = np.array(chrom_data[in_k])
                in_data = chrom_data[in_k][shuffled_idx]
                folds[k]['input'][in_k].append(in_data[start_idx:end_idx])

            for out_k in output_keys:
                if not isinstance(chrom_data[out_k], np.ndarray):
                    chrom_data[out_k] = np.array(chrom_data[out_k])
                out_data = chrom_data[out_k][shuffled_idx]
                folds[k]['output'][out_k].append(out_data[start_idx:end_idx])

    for k in range(n_folds):
        folds[k]['input'] = flatten_dict_of_lists(folds[k]['input'])
        folds[k]['output'] = flatten_dict_of_lists(folds[k]['output'])
        folds[k]['index'] = flatten_dict_of_lists(folds[k]['index'])

    return folds


def whole_genome_shuffled_k_fold(data_dict,
                                 n_folds,
                                 chroms_list,
                                 train_perc,
                                 test_perc,
                                 input_keys,
                                 output_keys,
                                 index_keys,
                                 random_seed):
    """[summary]

    Args:
        data_dict ([type]): [description]
        n_folds ([type]): [description]
        chroms_list ([type]): [description]
        train_perc ([type]): [description]
        test_perc ([type]): [description]
        input_keys ([type]): [description]
        output_keys ([type]): [description]
        index_keys ([type]): [description]
        random_seed ([type]): [description]

    Returns:
        [type]: [description]
    """
    np.random.seed(random_seed)

    folds = init_fold_dict(n_folds, input_keys, output_keys, index_keys)

    overall_data_len = 0
    # Init the data dict with the needed keys by copying an empty fold struct
    data = folds[0].copy()
    #print("\n\n\n\n\n\n\n")
    #print(data.keys())
    #print(data['index'])
    #print("data_dict", data_dict.keys())
    #print("data_dict [1]", data_dict[1].keys())
    #print(index_keys)
    #print("\n\n\n\n\n\n\n")
    for c in chroms_list:
        # Measure the size of any of the inputs
        chrom_data = data_dict[c]
        chrom_data_len = len(data_dict[c][input_keys[0]])
        overall_data_len += chrom_data_len

        data['index']['chrom_idx'].append([c]*(chrom_data_len))
        data['index']['index'].append(chrom_data['index'])
        remaining_idx_fields = [k for k in index_keys if k not in ['index', 'chrom_idx']]
        for ix_k in remaining_idx_fields:
            ix_data = chrom_data[ix_k]
            data['index'][ix_k].append(ix_data)
        
        for in_k in input_keys:
            if not isinstance(chrom_data[in_k], np.ndarray):
                chrom_data[in_k] = np.array(chrom_data[in_k])
            in_data = chrom_data[in_k]
            data['input'][in_k].append(in_data)

        for out_k in output_keys:
            if not isinstance(chrom_data[out_k], np.ndarray):
                chrom_data[out_k] = np.array(chrom_data[out_k])
            out_data = chrom_data[out_k]
            data['output'][out_k].append(out_data)

    fold_len = overall_data_len/n_folds
    
    print("\n\n\n\n\n\n\n")
    print(data['index'].keys())
    print("\n\n\n\n\n\n\n")
    data['input'] = flatten_dict_of_lists(data['input'])
    data['output'] = flatten_dict_of_lists(data['output'])
    data['index'] = flatten_dict_of_lists(data['index'])

    shuffled_idx = np.arange(overall_data_len)
    np.random.shuffle(shuffled_idx)

    # Shuffle all data
    for in_k in input_keys:
        data['input'][in_k] = data['input'][in_k][shuffled_idx]
    for out_k in output_keys:
        data['output'][out_k] = data['output'][out_k][shuffled_idx]

    data['index']['chrom_idx'] = data['index']['chrom_idx'][shuffled_idx]
    data['index']['index'] = data['index']['index'][shuffled_idx]

    # Split the folds
    for k in range(n_folds):
        start_idx = int(fold_len*k)
        end_idx = int(fold_len*(k+1))

        for in_k in input_keys:
            folds[k]['input'][in_k] = data['input'][in_k][start_idx:end_idx]
        for ot_k in output_keys:
            folds[k]['output'][ot_k] = data['output'][ot_k][start_idx:end_idx]

        chrom_id_data = data['index']['chrom_idx']
        chrom_index_data = data['index']['index']
        folds[k]['index']['chrom_idx'] = chrom_id_data[start_idx:end_idx]
        folds[k]['index']['index'] = chrom_index_data[start_idx:end_idx]

    return folds


def merge_folds(folds_data, val_idx):
    """[summary]

    Args:
        folds_data ([type]): [description]
        val_idx ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    train_data = {'input': {}, 'output': {}}
    input_keys = []
    for k in folds_data[val_idx]['input'].keys():
        train_data['input'][k] = []
        input_keys.append(k)

    output_keys = []
    for k in folds_data[val_idx]['output'].keys():
        train_data['output'][k] = []
        output_keys.append(k)

    n_folds = len(folds_data)

    if (val_idx < 0) or (val_idx > len(folds_data)):
        raise ValueError

    for i in range(n_folds):
        if i == val_idx:
            val_data = folds_data[i].copy()
        else:
            for k in input_keys:
                train_data['input'][k].append(folds_data[i]['input'][k])
            for k in output_keys:
                train_data['output'][k].append(folds_data[i]['output'][k])
    
    train_data['input'] = flatten_dict_of_lists(train_data['input'])
    train_data['output'] = flatten_dict_of_lists(train_data['output'])

    return train_data, val_data
