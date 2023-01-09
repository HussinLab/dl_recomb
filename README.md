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

> python make_refgenome.py -r ./input_data/genome_test/ref_by_chr -c 20 21 -o ./input_data/genome_test/h5/
2. Build a dataset given a hotspot file *input_file*

  > python build_dataset.py -i ./input_data/hotspots/input_test.txt -l 800 -c 20 21 -g ./input_data/genome_test/h5/genome_test.h5
3. Train model
  > module load cuda cudnn 
  > python run_experiment.py -l 800 -p AA1_hotspots -d ./models/new_exp/

# Output

The output is 





