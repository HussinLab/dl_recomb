import tensorflow as tf
import numpy as np
from functools import partial
from .feature_engineering import get_seq_reverse_complement
import math


class DNA_tf_dl(tf.keras.utils.Sequence):
    def __init__(self,
                 X_dict,
                 X_transform_dict,
                 y_dict,
                 y_transform_dict,
                 rev_comp_dict,
                 batch_size=64,
                 shuffle=True,
                 reshuffle_on_epoch_end=False):
        """[summary]
        Args:
            X ([type]): [description]
            y ([type]): [description]
            rev_comp_dict ([type]): [description]
            batch_size ([type]): [description]
            shuffle ([type]): [description]
        """
        # Check if computing the reverse complement on the fly in __get_item__
        # gives a big hit to performace or not
        pfunc_get_rc = partial(get_seq_reverse_complement,
                               num_reverse_compliment_dict=rev_comp_dict)

        self.X_dict = X_dict
        self.X_transform_dict = X_transform_dict
        self.y_dict = y_dict
        self.y_transform_dict = y_transform_dict

        self.rev_comp_dict = rev_comp_dict
        if self.rev_comp_dict is not None:
            # Transform each base to its compliment then reverse the order
            self.X_dict['seq_rc'] = np.apply_along_axis(pfunc_get_rc,
                                                        1,
                                                        self.X_dict['seq'])
            # Preprocess the reverse complement the same way as seq
            self.X_transform_dict['seq_rc'] = self.X_transform_dict['seq']

        self.batch_size = batch_size

        self.dataset_size = len(self.X_dict['seq'])
        self.indexes = np.arange(self.dataset_size)

        self.shuffle = shuffle
        self.reshuffle_on_epoch_end = reshuffle_on_epoch_end

        if self.shuffle is True:
            # self.on_epoch_end()
            np.random.shuffle(self.indexes)
            self.X_dict = {k: v[self.indexes] for k, v in self.X_dict.items()}
            self.y_dict = {k: v[self.indexes] for k, v in self.y_dict.items()}

        self.number_of_batches = math.ceil(self.dataset_size / self.batch_size)

    def __len__(self):
        """Returns the number of batches per epoch

        Returns:
            int: number of items (i.e. batches) in the dataset
        """
        return self.number_of_batches

    def __getitem__(self, batch_index):
        """
        Generate one batch of data

        Returns:
            [type]: [description]
        """
        # Generate indexes of the batch
        idxs = np.arange(batch_index*self.batch_size,
                         min(self.dataset_size,
                             (batch_index+1) * self.batch_size))

        ret_x = {k: self.X_transform_dict[k](v[idxs])
                 for k, v in self.X_dict.items()}

        ret_y = {k: self.y_transform_dict[k](v[idxs])
                 for k, v in self.y_dict.items()}

        return ret_x, ret_y

    def on_epoch_end(self):
        """
        Updates\\Reshuffles indexes after each epoch
        """
        if self.reshuffle_on_epoch_end is True:
            np.random.shuffle(self.indexes)
            self.X_dict = {k: v[self.indexes] for k, v in self.X_dict}
            self.y_dict = {k: v[self.indexes] for k, v in self.y_dict}

    def get_curr_indices(self):
        """[summary]
        Since the indices could get shuffled, we may want to know which is what
        example
        """
        return self.indexes

    def set_indexes(self, new_index, perform_sanity_check=True):
        """[summary]

        Args:
            new_index ([type]): [description]
        """
        if perform_sanity_check is True:
            # I think if min = 0, max = size and all unique, then checking
            # that sizes match is not necessary
            # assert len(new_index) == self.dataset_size
            assert np.min(new_index) == 0
            assert np.max(new_index) == (self.dataset_size - 1)
            assert len(np.unique(new_index)) == self.dataset_size

        self.indexes = new_index
        self.X_dict = {k: v[self.indexes] for k, v in self.X_dict}
        self.y_dict = {k: v[self.indexes] for k, v in self.y_dict}

    def reset_indices(self):
        """
        resets the indices to the original order, this is useful for
        studying the model, to know which instance is what
        """
        ordered_index = np.arange(self.dataset_size)
        self.set_indexes(ordered_index)
