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
import matplotlib.pyplot as plt
from Dataset import SubIntervalDataset
import pandas as pd

import argparse
 
 
# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-r", "--ref_genome_path", help = "Show Output")
parser.add_argument("-o", "--out_genome_h5", help = "Show Output")

parser.add_argument("-c", "--chrom_use",nargs="*", default=[20, 21], help = "Show Output")
args = parser.parse_args()

#@numba.jit(parallel=True)
def build_h5(fastas_path, 
             output_file_name, 
             encoding_dict={'A':0, 'C':1, 'T':2, 'G':3, 'N':4, 'M':4, 'R':4}, 
             files_prefix="Homo_sapiens.GRCh38.dna.chromosome.",
             chromosomes_list=list(range(1,23)) + ['X','Y','MT'],
             file_extension = 'fa.gz',
             chunk_size=(100000,)):
    """
    https://stackoverflow.com/questions/42757283/seqio-parse-on-a-fasta-gz
    
    It will always read the first record only
    Save the encoding dict as metadata in the h5 file
    Each chromosome is a dataset
    
    @param files_prefix: the fasta files start with this string, followed by the number\name of the chromosome
    @param chromosomes: a list of the names that come after the prefix to use. You can use this to filter out 
                  chromosomes if not needed.
                  
    @param chunk_size: the chunk size of the h5 dataset. Data elements will be loaded in blocks of this size, so
                       the larger the number is, the more RAM it takes but less disk IO
    """
    #TODO: MAKE SURE THAT THE ENCODING_DICT TRANSLATES TO INTEGERS
    output_file = h5py.File(output_file_name + ".h5", "a")
    
    output_file["./"].attrs["available_chromosomes"] = []
    output_file["./"].attrs['time created'] = time.time()
    
    #Dictionaries are not natively supported in h5, and need to be serialized. Using json.dumps for that
    metadata_dict = {}
    metadata_dict["encoding_dict"] = json.dumps(encoding_dict)
    metadata_dict["available_chromosomes"] = []
    metadata_dict['chr_lengths'] = {}
    metadata_dict['time created'] = time.time()
    
    for chromosome in chromosomes_list:
        fname = f"{files_prefix}{chromosome}.{file_extension}"
        file = os.path.join(fastas_path, fname)
        
        encoding = guess_type(file)[1]  # uses file extension
        _open = partial(gzip.open, mode='rt') if encoding == 'gzip' else open
        print(f"\tencoding {encoding}")
        
        with _open(file) as handle:
            print(f"\t{file} was opened successfully, parsing")
            print("\tFound FASTA records in file:")
            
            for i, record in enumerate(SeqIO.parse(handle, "fasta")):
                
                if i > 0:
                    print(f"\tFile {file} contains more than one record, skipping them ({record})")
                    continue
                
                print("\tSequence ID:", record.id)
                sequence = list(str(record.seq))
                
                sequence = [encoding_dict[ str.upper(bp) 
                                         ] for bp in sequence]
                
                seq_len = len(sequence)
                print("\tSequence Length:", seq_len)
                
                chromosome_arr  = np.array(sequence, dtype=np.int8)
                
                #Save each as an integer of 1 byte
                cs = chunk_size if chromosome != "MT" else True
                output_file.create_dataset(f"chr{chromosome}", data=chromosome_arr, dtype='i1', chunks=cs)
                metadata_dict["available_chromosomes"].append(record.id)
                metadata_dict['chr_lengths'][f"chr{chromosome}"] = seq_len
                
    print(metadata_dict)
    output_file["./"].attrs["metadata"] = json.dumps(metadata_dict)
    output_file.close()
    


build_h5(args.ref_genome_path,
         files_prefix ="",
         chromosomes_list=list(args.chrom),
         output_file_name=args.out_genome_h5)