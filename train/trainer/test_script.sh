MAIN_TRAINER_MODULE="trainer.train"
RUNTIME_VERSION="2.1"
PYTHON_VERSION="3.7"
REGION="us-central1"
SCALE_TIER=custom
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="count=1,type=nvidia-tesla-t4"


# Define HPS to pass
## File
OUTPUT_NAME="A_hotspots_intersect C"
SEQ_LEN=800

## Experiment tracking params
SAVE_DATASET=False
SAVE_MODEL=False
EVAL_MODEL=False


## Dataset params
FOLD_FN_NAME="whole_genome_shuffled_k_fold"
RANDOM_SEED=123
USE_X_Y=True

## Training params
BATCH_SIZE=128
EPOCHS=2
N_FOLDS=3
TRAIN_PERC="0.6"
TEST2_INTERVAL=10
AUTO_BALANCE_BCE=True

## Architecture params
MODEL_TYPE="CNN_ONLY"
USE_REV_COMPL=True
CHROM_IDX=False
MIDPOINT=False

## Hyperparameters (TEST ONLY)
FIRST_CNN_FILTER_SIZE=20
FIRST_CNN_N_FILTERS=12
FIRST_CNN_POOL_SIZE_STRIDES=10
N_CONVS=1
N_CONVS_FILTER_SIZE=10
N_CONVS_N_FILTERS=12
N_CONVS_POOL_SIZE_STRIDE=5
FIRST_FCC_SIZE=50
N_FCCS=2
FCCS_SIZE=60
OUTPUTS_SEPARATE_FC=20

LR="0.01"
use_GRU=False
DROPOUT_RATE=0.2

# ALWAYS FILL
JOB_NAME="$EXPERIMENT_NAME$MODEL_TYPE$(date +"%Y%m%d_%H%M%S")"
JOB_DIR="gs://$BUCKET_NAME/hp_job_dir_$MODEL_TYPE"
SCRIPTS_FOLDER_NAME='./../template'
TRAINER_PACKAGE_PATH="./$SCRIPTS_FOLDER_NAME/trainer"


EXPERIMENT_NAME="HPS_CNN_800_A_C_DEC21_001"
MLFLOW_URI="http://34.122.175.239:5000/"


# Launch
python train.py --job-dir $JOB_DIR\
                --output_name  $OUTPUT_NAME\
                --seq_len  $SEQ_LEN\
                --save_dataset  $SAVE_DATASET\
                --save_model  $SAVE_MODEL\
                --eval_model  $EVAL_MODEL\
                --fold_fn_name  $FOLD_FN_NAME\
                --random_seed  $RANDOM_SEED\
                --use_x_y  $USE_X_Y\
                --batch_size  $BATCH_SIZE\
                --epochs  $EPOCHS\
                --n_folds  $N_FOLDS\
                --train_perc  $TRAIN_PERC\
                --test2_interval  $TEST2_INTERVAL\
                --auto_balance_bce $AUTO_BALANCE_BCE\
                --model_type  $MODEL_TYPE\
                --use_rev_compl  $USE_REV_COMPL\
                --chrom_idx  $CHROM_IDX\
                --midpoint  $MIDPOINT\
                --first_cnn_filter_size  $FIRST_CNN_FILTER_SIZE\
                --first_cnn_n_filters  $FIRST_CNN_N_FILTERS\
                --first_cnn_pool_size_strides  $FIRST_CNN_POOL_SIZE_STRIDES\
                --n_convs  $N_CONVS\
                --n_convs_filter_size  $N_CONVS_FILTER_SIZE\
                --n_convs_n_filters  $N_CONVS_N_FILTERS\
                --n_convs_pool_size_stride  $N_CONVS_POOL_SIZE_STRIDE\
                --use_GRU $use_GRU\
                --first_fcc_size  $FIRST_FCC_SIZE\
                --n_fccs  $N_FCCS\
                --fccs_size  $FCCS_SIZE\
                --outputs_separate_fc $OUTPUTS_SEPARATE_FC\
                --lr  $LR\
                --dropout_rate $DROPOUT_RATE