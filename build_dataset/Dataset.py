import h5py
import json 
import numpy as np
import pandas as pd
from collections import namedtuple

PositiveData = namedtuple('PositiveData',['seq','output','metadata'])
NegativeData = namedtuple('NegativeData',['seq', 'metadata', 'interval'])

class SubIntervalDataset:
    def __init__(self,
                 pos_df,
                 min_inter_positive_examples,  # -ve value means don't care
                 seq_len,
                 grch_obj,
                 negative_sampling_intervals,
                 chr_col,
                 start_col,
                 end_col,
                 seq_midpoint_col,
                 output_cols_list,
                 chrom_len,
                 forbidden_negative_sampling_regions=None, # Come up with a better name
                 binarize_output=True,
                 meta_cols_to_keep=None):
        """
        This class creates a sub dataset (Which is a dataframe of positive
        intervals), and creates the sequence and other negative examples as
        well. It is assumed that the subdataset is a whole chromosome, and if
        this will always be the case then we should rename the class to
        ChromosomeDataset when we refactor the code.

        TODO: Add meta_cols_to_keep to add extra metadata columns in the df,
              if needed
        """

        self.chr_col = chr_col
        self.start_col = start_col
        self.end_col = end_col
        self.seq_midpoint_col = seq_midpoint_col
        self.output_cols_list = output_cols_list

        self.data_len = len(pos_df)

        self.grch_obj = grch_obj
        self.negative_sampling_intervals = negative_sampling_intervals

        for (lower_limit, upper_limit) in negative_sampling_intervals:
            assert (upper_limit - lower_limit) > seq_len

        self.seq_len = seq_len
        self.half_seq = self.seq_len // 2
        self.encoding_dict = grch_obj.metadata['num_2_base_dict']

        self.binarize_output = binarize_output
        # Stats
        self.contain_N_bp_count = 0

        # Used to get the last portion of the chromosome
        self.chrom_len = chrom_len
        self.forbidden_negative_sampling_regions = forbidden_negative_sampling_regions

        self.negative_examples_meta_names = ['seq_start',
                                             'seq_end',
                                             'seq_midpoint',
                                             'original_start',
                                             'original_end',
                                             'original_seq_len',
                                             'original_count_dict_A',
                                             'original_count_dict_C',
                                             'original_count_dict_T',
                                             'original_count_dict_G',
                                             'example_count_dict_A',
                                             'example_count_dict_C',
                                             'example_count_dict_T',
                                             'example_count_dict_G']

        self.pos_df = self.compute_global_positive_regions(pos_df,
                                                           min_inter_positive_examples)
        self.neg_df = self.compute_global_negative_regions(self.pos_df,
                                                           forbidden_negative_sampling_regions,
                                                           sort_merged_by='start')

        self.positive_examples_meta_names = self.negative_examples_meta_names + self.output_cols_list

        self.positive_dataset = self.get_positive_dataset()
        self.negative_datasets_dict = {}
        for neg_intrvl in self.negative_sampling_intervals:
            self.negative_datasets_dict[neg_intrvl] = self.get_negative_dataset(lower_limit=neg_intrvl[0],
                                                                                upper_limit=neg_intrvl[1])

    def get_seqs_and_meta(self, df, negative_example=False):
        """
        Setting negative_example True means that this is not a negative
        sampled, and therefore we have to compute both ACGT contents of the
        original interval and the sampled sequence with length seq_len

        Also, we add the values of the output in the metadata, which is assumed
        to be some sort of a read count\\signal strength
        """
        curr_dataset_len = len(df)
        print(f'\t Passed dataset {df.shape} is {100*curr_dataset_len/self.data_len:.2f}% of the original dataset. Is Negative = {negative_example}')
        ten_perc = curr_dataset_len // 10
        # Sometimes, the result is too small, we avoid an error
        if (ten_perc == 0):
            print(f"\n\n\n\n\n\n\n\n\n\n WARNING: ten_perc in get_seqs_and_meta is zero, make sure that there is no problems with the dataset or the code\n\n\n\n\n\n\n\n\n\n")
            ten_perc = 1
        
        seq_list = []
        meta_list = []

        encoding_dict = self.grch_obj.metadata['num_2_base_dict']
        
        used_rows_mask = []

        for i in range(curr_dataset_len):
            chrom_name = df[self.chr_col].iloc[i]
            # if i%ten_perc == 0:
            #     print(f"\t {1 + 100*i//curr_dataset_len}% processed")

            seq_midpoint = df[self.seq_midpoint_col].iloc[i]

            seq_start = int(df['seq_start'].iloc[i])
            seq_end = int(df['seq_end'].iloc[i])

            seq = self.grch_obj.get_seq_by_chr_and_midpoint(chrom_name,
                                                            seq_midpoint,
                                                            self.seq_len)

            if np.any(seq == self.grch_obj.metadata['encoding_dict']['N']):
                print(f"\t\tWarning: interval between {seq_start} and {seq_end} contains an N, it will \
                           be skipped.")
                self.contain_N_bp_count += 1
                used_rows_mask.append(False)
                continue

            if len(seq) < self.seq_len:
                print(f"\t\tWarning: interval between {seq_start} and {seq_end} returned an empty sequence!!!")
                used_rows_mask.append(False)
                continue

            #We passed all the filters, we will use this row in the final dataset
            
            used_rows_mask.append(True)
            # Compute GC content for the metadata
            original_seq = self.grch_obj.get_seq_by_chr_and_indices(chrom_name, 
                                                                    df[self.start_col].iloc[i],
                                                                    df[self.end_col].iloc[i])

            original_count_dict = SubIntervalDataset.compute_gc_content(original_seq,
                                                                        encoding_dict)

            example_count_dict = SubIntervalDataset.compute_gc_content(seq,
                                                                       encoding_dict)
            seq_list.append(seq)

            original_start = df[self.start_col].iloc[i]
            original_end = df[self.end_col].iloc[i]
            original_seq_len = original_end - original_start

            if not negative_example:
                meta = [seq_start,
                        seq_end,
                        seq_midpoint,
                        original_start,
                        original_end,
                        original_seq_len,
                        original_count_dict['A'],
                        original_count_dict['C'],
                        original_count_dict['T'],
                        original_count_dict['G'],
                        example_count_dict['A'],
                        example_count_dict['C'],
                        example_count_dict['T'],
                        example_count_dict['G']]

                curr_outs_strength = []  # strength is
                for out in self.output_cols_list:
                    out_val = df[out].iloc[i]
                    curr_outs_strength.append(out_val)  # out_strength
                meta += curr_outs_strength
            else:
                meta = [seq_start,
                        seq_end,
                        seq_midpoint,
                        original_start,
                        original_end,
                        original_seq_len,
                        -1,
                        -1,
                        -1,
                        -1,
                        example_count_dict['A'],
                        example_count_dict['C'],
                        example_count_dict['T'],
                        example_count_dict['G']]

            meta_list.append(meta)

        return seq_list, meta_list, used_rows_mask

    def get_positive_dataset(self):
        """
        This function calls the get_seqs_and_meta, and saves the result in the
        PositiveDataset named tuple format
        
        """
        print("\tConstructing Positive Dataset")
        seq_list, meta_list, used_rows_mask = self.get_seqs_and_meta(self.pos_df,
                                                                     negative_example=False)
        print(f"Positive constructed, total {len(seq_list)} examples")
        seq_arr = np.array(seq_list)
        meta_df = pd.DataFrame(meta_list, columns=self.positive_examples_meta_names)
        
        out_vals = self.pos_df.loc[used_rows_mask, self.output_cols_list].copy()
        if self.binarize_output:
            out_vals = (out_vals > 0).astype(np.int32)
        print(meta_df)
        return PositiveData(seq_arr, out_vals, meta_df.astype(np.int32))

    def get_negative_dataset(self, lower_limit=None, upper_limit=None):
        print(f"\tConstructing Negative dataset ({lower_limit}, {upper_limit})")
        neg_df_sampling = self.get_negative_intervals(self.neg_df,
                                                      lower_limit=lower_limit,
                                                      upper_limit=upper_limit)
        neg_df = neg_df_sampling.copy()

        neg_df[self.seq_midpoint_col] = neg_df_sampling.apply(lambda x: np.random.randint(x[self.start_col], 
                                                                                          x[self.end_col]),
                                                              axis=1)
        neg_df["seq_start"] = neg_df[self.seq_midpoint_col] - self.half_seq
        neg_df["seq_end"] = neg_df[self.seq_midpoint_col] + self.half_seq

        # Do some sanity check
        wrong_lens = (neg_df['seq_end'] - neg_df['seq_start']) != self.seq_len
        neg_df.loc[wrong_lens, "seq_end"] += self.seq_len - neg_df.loc[wrong_lens, "seq_end"]

        # Sometimes, the index columns become floats and this raises an error
        index_cols = [self.start_col, self.end_col, 'seq_start', 'seq_end']
        neg_df[index_cols] = neg_df[index_cols].astype(int)

        # used_rows_mask is not used here
        seq_list, meta_list, _ = self.get_seqs_and_meta(neg_df, negative_example=True)

        seq_arr = np.array(seq_list)
        meta_df = pd.DataFrame(meta_list,
                               columns=self.negative_examples_meta_names)

        print(f"Negative ({lower_limit}, {upper_limit}) constructed, total {len(seq_list)} examples")

        return NegativeData(seq_arr,
                            meta_df.astype(np.int32),
                            (lower_limit, upper_limit))

    def get_negative_intervals(self,
                               neg_df_in,
                               lower_limit=None,
                               upper_limit=None):
        """
        1) Suppose we have the following negative region
            *******************************************************************

        2) We sample from beginning and end a sequence of length ------

            *******************************************************************
            ------                                                       ------

        3) But in the prvious setting, there is no buffer defined as the lower
        limit, so we add this +++
            *******************************************************************
            +++------                                                 ------+++

        We actually define the region to sample the midpoint from as:
            (start + lower_limit + half seq len), (start + upper_limit - half seq len)

            AND

            (end - upper_limit + half seq len), (end - lower_limit - half seq len)
        """
        assert lower_limit is not None
        assert upper_limit is not None

        # Filter-out negative regions that do not contain enough length
        enough_len = upper_limit - lower_limit
        enough_len_idx = (neg_df_in[self.end_col] - neg_df_in[self.start_col]) > enough_len
        neg_df = neg_df_in[enough_len_idx]

        half_seq = self.seq_len // 2
        # lneg = local negative
        start_lneg_df = pd.DataFrame(columns=[self.start_col, self.end_col])
        end_lneg_df = pd.DataFrame(columns=[self.start_col, self.end_col])
        merged_negs = pd.DataFrame(columns=[self.start_col, self.end_col])

        start_lneg_df[self.start_col] = neg_df[self.start_col] + lower_limit + half_seq
        start_lneg_df[self.end_col] = neg_df[self.start_col] + upper_limit - half_seq

        end_lneg_df[self.start_col] = neg_df[self.end_col] - upper_limit + half_seq
        end_lneg_df[self.end_col] = neg_df[self.end_col] - lower_limit - half_seq

        # If there are overlapping regions, then merge them. This is due to the
        # cases when  the negative space between two positive regions does not
        # fit 2 seq_len, so we cannot sample 2 negative sequences without
        # overlap in that case, so we merge them into one negative region
        overlapps_filter = (start_lneg_df[self.start_col] < end_lneg_df.shift()[self.end_col])

        merged_negs[self.start_col] = start_lneg_df.loc[overlapps_filter, self.start_col]
        merged_negs[self.end_col] = end_lneg_df.shift().loc[overlapps_filter, self.end_col]

        start_lneg_df = start_lneg_df[~overlapps_filter]

        end_overlapps_filter = overlapps_filter.copy()
        end_overlapps_filter[:-1] = end_overlapps_filter[1:]
        end_overlapps_filter[-1] = False
        end_lneg_df = end_lneg_df[~end_overlapps_filter]

        # Now concatenate all the results
        local_neg_df = pd.concat([start_lneg_df, end_lneg_df, merged_negs])

        # TODO: ENABLE THIS: Filter out negative intervals
        filter_start = (local_neg_df > 0).all(axis=1)
        local_neg_df = local_neg_df[filter_start]

        filter_end = (local_neg_df < self.chrom_len).all(axis=1)
        local_neg_df = local_neg_df[filter_end]

        local_neg_df = local_neg_df.sort_values(self.start_col)
        local_neg_df = local_neg_df.reset_index(drop=True)

        # Add the chromosome name
        local_neg_df[self.chr_col] = self.pos_df[self.chr_col].iloc[0]

        return local_neg_df

    def compute_global_negative_regions(self,
                                        pos_df,
                                        forbidden_negative_sampling_regions=None,
                                        sort_merged_by=None ):
        """
        Negative regions are the spaces between the end of a hotspot and the
        start of the next.
        So we shift the start of regions infront of the end of the previous and
        this way we get the intervals. This algorithm is true except for the
        start and the end of the chromosome, so we handle those explicitly
        
        @param sort_merged_by: we will merge pos_df and forbidden_negative_sampling_regions
                               in order to compute the negative sampling regions. In order
                               for the algo to work correctly, the final dataset must be 
                               correctly sorted. So we sort them, usually by 'start' if it's
                               a single chromosomes or ['chrom', 'start'] if it's a whole 
                               genome (But current implementation will fail for whole genome)
                               TODO: Generalize the function
        """
        pos_df_in = pos_df.copy()
        if forbidden_negative_sampling_regions is not None:
            pos_df_in = pd.concat([pos_df_in, forbidden_negative_sampling_regions]).sort_values(sort_merged_by).reset_index()
        
        neg_df = pd.DataFrame(columns=['chrom','start', 'end'])
        
        chrs = pos_df_in['chrom']
        if(len(neg_df)!=len(pos_df_in)):
            chrs.append(pd.Series(pos_df_in['chrom'][len(pos_df_in)-1]))
        
        neg_df[self.chr_col] = chrs
        neg_df[self.start_col] = pos_df_in[self.end_col].shift()
        neg_df[self.end_col] = pos_df_in[self.start_col]

        # Handle the start of the chromosome. If the positive data does not
        # start at the very beginning of the chromosome, then add that the
        # interval of 0 and start of the first hotspot is a negative region
        if pos_df_in.iloc[0][self.start_col] != 0:
            neg_df.iloc[0, neg_df.columns.get_loc('start')] = 0
        else:
            neg_df.drop(neg_df.index[0], inplace=True)

        # Handle the end of the chromosome
        if pos_df_in.iloc[-1, pos_df_in.columns.get_loc(self.end_col)] < self.chrom_len:
            neg_df = neg_df.append({self.start_col: pos_df_in.iloc[-1, pos_df_in.columns.get_loc(self.end_col)], 
                                    self.end_col: self.chrom_len},
                                   ignore_index=True)

        # If the resulting negative region does not have enough space to sample
        # a negative example,  delete it from the table.
        neg_data_len = neg_df['end'] - neg_df['start']
        suitable_regions = (neg_data_len - self.seq_len) > 0
        neg_data_len = neg_data_len[suitable_regions]
        
        neg_df['start'] = neg_df['start'].astype(int)
        neg_df['end'] = neg_df['end'].astype(int)
        
        return neg_df

    def compute_global_positive_regions(self, pos_df_in, min_inter_positive_examples):
        """
        This function disqualifies positive intervals that are too close to each other

        DO NOT USE IT WITH min_inter_positive_examples > 0
        TODO: Merge these or make a black list regions that cannot be sampled from in the
              negative examples creation
        """
        pos_df = pos_df_in.copy()
        half_seq = self.seq_len // 2

        # Just make sure that we sort by start of sequence
        pos_df = pos_df.sort_values(self.start_col).reset_index(drop=True)

        # If there are examples whose midpoint is too close to the start of the
        # chromosome, push it downstream a little bit (half seq) so that we can
        # have enough space to take the example
        pos_df.loc[pos_df[self.seq_midpoint_col] < self.seq_len, self.seq_midpoint_col] = 1 + half_seq

        # Conversly, if the positive interval midpoint is at the end of the
        # chromosome, give it enough room for sampling the example by pushing
        # down the midpoint
        pos_df.loc[pos_df[self.seq_midpoint_col] > self.chrom_len, self.seq_midpoint_col] = self.chrom_len - 1 - half_seq

        pos_df['seq_start'] = pos_df[self.seq_midpoint_col] - half_seq
        pos_df['seq_end'] = pos_df[self.seq_midpoint_col] + half_seq

        if min_inter_positive_examples > 0:
            inter_seq = pos_df.start.shift(-1) - pos_df.end
            return pos_df[inter_seq > inter_seq ]
        else:
            return pos_df

    def compute_gc_content(seq, bp_dict):
        unique, counts = np.unique(seq, return_counts=True)
        d = dict(zip(unique, counts))
        ret_dict = {}

        for k, v in d.items():
            ret_dict[bp_dict[k]] = v

        # Handle the case where a sequence does not contain a certain basepair
        for missing_bp in list('ACTG'):
            if missing_bp not in ret_dict.keys():
                ret_dict[missing_bp] = 0

        return ret_dict

