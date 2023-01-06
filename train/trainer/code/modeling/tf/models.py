from tensorflow.keras.layers import *
import tensorflow as tf


class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def get_chosen_dropout(mc=False):
    if mc:
        return MCDropout
    else:
        return tf.keras.layers.Dropout


from tensorflow.keras.layers import *
import tensorflow as tf


class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def get_chosen_dropout(mc=False):
    if mc:
        return MCDropout
    else:
        return tf.keras.layers.Dropout


from tensorflow.keras.layers import *
import tensorflow as tf


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
