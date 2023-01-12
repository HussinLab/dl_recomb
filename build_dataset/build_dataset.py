import getopt, sys

import h5py
import gzip
from Bio import SeqIO
import os
from mimetypes import guess_type
from functools import partial
import time
import json
import numpy as np
from refGenome import refGenome_h5
import seaborn as sb

import matplotlib.backends.backend_tkagg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from Dataset import SubIntervalDataset
import pandas as pd
from io import StringIO
import cv2

from utils import *


import argparse
 
 
# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-i", "--input", help = "Show Output")
parser.add_argument("-l", "--seqlen", help = "Show Output")
parser.add_argument("-c", "--chrom_use",nargs="*", default=[20, 21], help = "Show Output")
parser.add_argument("-g", "--h5_ref_genome", help = "Show Output")
parser.add_argument("-p", "--prefix", help = "Show Output")
parser.add_argument("-o", "--outdir", help = "Show Output")

# Read arguments from command line
args = parser.parse_args()
 
df = pd.read_csv(args.input, sep='\t')
df['seq_len'] =  df['end'] - df['start']
df['seq_midpoint'] = df['start'] + ((df['end'] - df['start'])//2)

outputs = [args.prefix]
seq_lens  = [int(args.seqlen)]
chroms_to_use = args.chrom_use
negative_sampling_intervals=[(1000,5000), (5000,10000), (10000, 20000), (20000,100000)]
random_seed=123
binarize_output = True

grch37_ds = refGenome_h5(args.h5_ref_genome)
print('done')
chroms_lens_dict = grch37_ds.metadata['chr_lengths']


df = df.loc[df['chrom'].isin(list(chroms_lens_dict.keys()))]

for output in outputs:
    for seq_len in seq_lens:
        print(f"OUTPUT::: {output}")
        print(f"SEQLEN::: {seq_len}")
        # If it's a multioutput dataset
        if isinstance(output, list):
            output_name = output
            
            cols = ['chrom', 'start', 'end']
            cols += output_name
            cols += ['seq_len', 'seq_midpoint']
            
            masks = [df[c] == 1 for c in output]
            masks = np.array(masks).T
            masks = masks.any(axis=1)
            
            df_output = df.loc[masks, cols].copy().reset_index(drop=True)
            
            meta_cols = ['chrom', 'start', 'end', 'seq_len', 'seq_midpoint'] + output_name
            outs_name = ".".join(output)
            output_fname = f"{args.outdir}/{outs_name}_seqlen={seq_len}_multnegs"
            
        else:
            
            output_name = [output]
            cols = ['chrom', 'start', 'end', f"{output}", 'seq_len', 'seq_midpoint']  
            
            df_output = df.loc[df[output] == 1, cols].copy().reset_index(drop=True)
            
            meta_cols = ['chrom', 'start', 'end', 'seq_len', 'seq_midpoint', f"{output}"]

            output_fname = f"{args.outdir}/{output}_seqlen={seq_len}_multnegs"
        
        print(f"OUTPUT DATAFRAME AFTER FILTERING {output}")
        print(df_output)
        print(output_name, seq_len)
        print("\n\n\n")
        
        create_h5_dataset_by_chromosome(df=df_output,
                                        output_fname=output_fname,
                                        grch_obj=grch37_ds,
                                        chroms_lens_dict=chroms_lens_dict,
                                        seq_len=seq_len,
                                        negative_sampling_intervals=negative_sampling_intervals,
                                        chr_col='chrom',
                                        start_col='start',
                                        end_col='end',
                                        seq_midpoint_col='seq_midpoint',
                                        output_cols_list=output_name, #always pass a list, even for a single out

                                        binarize_output=binarize_output,

                                        pos_suffix='pos',
                                        neg_suffix='neg',

                                        seq_suffix='seq',
                                        metadata_suffix='meta',
                                        output_suffix='out',
                                        
                                        random_seed=random_seed,

                                        test_set_size=0.1,
                                        test_set_sampling='uniform')
        
        print("*"*80)
        print("*"*80)
        print("*"*80)