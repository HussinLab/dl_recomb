# Introduction

The code provided here allows to:
1. Take a list of ChIP-Seq hot-spots, 
2. Create a dataset of:
   i.  Fixed size segments around the peaks  
   ii. negative (ie not a ChIP-Seq peak) segments, 
3. Use a deep learning approach to train a model to classify DNA fragments into being a peak or not.  

# Folders Structure

## **input_data**
1. genome_test/
2. hotspots/
## **build_dataset**
1. datasets/
2. Dataset.py
3. refGenome.py
4. test_refGenome.ipynb
5. test_DSBuilder.ipynb
## **train**
1. models/
2. trainer/
3. wandb/
## **motif_identifier** 

# Running Code\Experiments
..
1. Build a h5 reference genome:

For easy loading, the reference genome is converted into a h5 format. In the following example, a test reference genome is built for only chromosomes 20 and 21. The folder **./input_data/genome_test/ref_by_chr** contains the gzipped fasta annotation of the each chromosomes (in this case, *20.fa.gz*, *21.fa.gz*). The folder also contains a folder named **h5/** in which the h5 reference genome will be stored. 

```
python build_dataset/make_refgenome.py \ 
  -r ./input_data/genome_test/ref_by_chr \
  -c 20 21 \ 
  -o ./input_data/genome_test/h5/

```
2. Build a dataset given a hotspot file *input_file*
Given a dataframe of ChIP-Seq peaks, this script builds a dataset ready for training. The dataset is comprised of positive (is a ChIP-Seq peak) and negative (is not a ChIP-Seq peak) genomic intervals. The reference is used to extract the nucleotide identity of the desired intervals. Since the peaks are generally a single "hotspot" genomic position, one needs to specify the length of the intervals (centered on the hotspots). In this case, -l 800 signifies intervals of length 800bp were used. The user also needs to specify the chromosomes to use (-c 20 21) and the path to the reference genome created in **1)**. Finally, the last argument (-o ./build_dataset/datasets/) specifies the path to store the output h5 dataset. 
```
python build_dataset/build_dataset.py \
  -i ./input_data/hotspots/input_test.txt \
  -l 800 \
  -c 20 21 \
  -p AA1_hotspots \ 
  -g ./input_data/genome_test/h5/genome_test.h5 \
  -o ./build_dataset/datasets/
```
3. Train model
```
python train/run_experiment.py \
  -l 800 \
  -p AA1_hotspots \
  -d ./models/new_exp/
```

# Output






