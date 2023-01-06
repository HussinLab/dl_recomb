import h5py
import json 
import numpy as np

class refGenome_h5:
    
    def __init__(self, h5_path):

        # read file
        self.f = h5py.File(h5_path, 'r')

        # Print available datasets
        for key in self.f.keys():
            print(key)

        # print read
        self.metadata = json.loads(self.f.attrs['metadata'])
        enc_dct_str = 'encoding_dict'
        self.metadata[enc_dct_str] = json.loads(self.metadata[enc_dct_str])
        self.metadata['num_2_base_dict'] = {}
        for k, v in self.metadata[enc_dct_str].items():
            self.metadata['num_2_base_dict'][v] = k

        # The following dictionary is used to convert the base to its
        # compliment, but both input and output are in the numerical encoding.
        # The code is revers_dict[num value of base] = num value of its 
        # complement
        num_rc_dct_name = 'num_reverse_compliment_dict'
        self.metadata[num_rc_dct_name] = {}
        self.metadata[num_rc_dct_name][self.metadata[enc_dct_str]['A']] = self.metadata[enc_dct_str]['T']
        self.metadata[num_rc_dct_name][self.metadata[enc_dct_str]['T']] = self.metadata[enc_dct_str]['A']
        self.metadata[num_rc_dct_name][self.metadata[enc_dct_str]['C']] = self.metadata[enc_dct_str]['G']
        self.metadata[num_rc_dct_name][self.metadata[enc_dct_str]['G']] = self.metadata[enc_dct_str]['C']
        self.metadata[num_rc_dct_name][self.metadata[enc_dct_str]['N']] = self.metadata[enc_dct_str]['N']
        self.metadata[num_rc_dct_name][self.metadata[enc_dct_str]['M']] = self.metadata[enc_dct_str]['M']
        self.metadata[num_rc_dct_name][self.metadata[enc_dct_str]['R']] = self.metadata[enc_dct_str]['R']

        print("Object Metadata:")
        print(self.metadata)

    # Returns sequence queried by chr id and a position interval
    def get_seq_by_chr_and_indices(self, chr_num, start, end, strand='+'):
        
        seq = self.f[chr_num][start:end]

        if strand == '-':
            seq = seq[::-1]
            seq = np.array( [self.metadata['num_reverse_compliment_dict'][seq[i]] for i in range(len(seq))] )

        return seq

    # Returns sequence queried by chr id and central pos + length of desired sequence
    def get_seq_by_chr_and_midpoint(self, chr_num, mid_point, seq_len, strand='+'):
        start = mid_point - seq_len//2
        end = mid_point + seq_len//2
        if seq_len % 2 != 0:
            end += 1
        return self.f[chr_num][start:end]


    def convert_num_to_bases(self, numerical_seq):
        return [self.metadata['num_2_base_dict'][v] for v in numerical_seq]

    def one_hot_encode(self, seq, out_seq_len=None):
        """
        out_seq_len: padding, if required
        """
        # because we have -1 for N, add 1 to everything, so that N = no
        # encoding, A = 0001, C=0010 ...etc
        one_hot = np.eye(4)[seq]
        
        # set the N's to be zeros
        one_hot[seq == -1] = 0
        return one_hot

    def convert_bases_to_num(self, seq, out_seq_len=None):
        return [ self.metadata['encoding_dict'][v] for v in seq]

    def __del__(self):
        self.f.close()
