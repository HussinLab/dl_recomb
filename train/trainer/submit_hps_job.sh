# ALWAYS FILL:
#EXPERIMENT_NAME
#SCRIPTS_FOLDER_NAME
#MLFLOW_URI if used

EXPERIMENT_NAME="HPS_CNN_800_A_C_DEC21_001"
SCRIPTS_FOLDER_NAME='../../new_HPS_Search_Dec21'
#MLFLOW_URI="http://34.122.175.239:5000/"
# add   --mlflow_uri      $MLFLOW_URI \ to the argument

#ONLY FOR HPS
HPS_CONFIG_FILE=hptuning_config.yaml

BUCKET_NAME="recombination-genomics-1"
MODEL_TYPE="TWIN_CNN_"
JOB_NAME="$EXPERIMENT_NAME$MODEL_TYPE$(date +"%Y%m%d_%H%M%S")"
JOB_DIR="gs://$BUCKET_NAME/hp_job_dir_$MODEL_TYPE"


TRAINER_PACKAGE_PATH="./$SCRIPTS_FOLDER_NAME/trainer"
MAIN_TRAINER_MODULE="trainer.train"
RUNTIME_VERSION="2.1"
PYTHON_VERSION="3.7"
REGION="us-central1"
SCALE_TIER=custom
MACHINE_TYPE="n1-standard-8"
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
EPOCHS=100
N_FOLDS=3
TRAIN_PERC="0.7"
TEST2_INTERVAL=10
AUTO_BALANCE_BCE=True

## Architecture params
USE_REV_COMPL=True
CHROM_IDX=False
MIDPOINT=False

# REMOVE !!!!
## Hyperparameters (TEST ONLY)
# FIRST_CNN_FILTER_SIZE=20
# FIRST_CNN_N_FILTERS=12
# FIRST_CNN_POOL_SIZE_STRIDES=10
# N_CONVS=1
# N_CONVS_FILTER_SIZE=10
# N_CONVS_N_FILTERS=12
# N_CONVS_POOL_SIZE_STRIDE=5
# FIRST_FCC_SIZE=50
# N_FCCS=2
# FCCS_SIZE=60
# OUTPUTS_SEPARATE_FC=20
# LR="0.01"
# use_GRU=False
# DROPOUT_RATE=0.2



# Launch ai-platform is the new renamed service, ml-engine is the old one.
 gcloud ai-platform jobs submit training $JOB_NAME \
          --job-dir $JOB_DIR \
          --package-path $TRAINER_PACKAGE_PATH \
          --module-name $MAIN_TRAINER_MODULE \
          --region $REGION \
          --runtime-version=$RUNTIME_VERSION \
          --python-version=$PYTHON_VERSION \
          --scale-tier $SCALE_TIER \
          --master-machine-type $MACHINE_TYPE \
          --master-accelerator $GPU_TYPE \
          --config=$HPS_CONFIG_FILE\
          -- \
          --random_seed     $RANDOM_SEED \
          --seq_len         $SEQ_LEN \
          --batch_size      $BATCH_SIZE \
          --epochs          $EPOCHS \
          --fold_fn_name    $FOLD_FN_NAME \
          --output_name     $OUTPUT_NAME \
          --midpoint        $MIDPOINT \
          --chrom_idx       $CHROM_IDX \
          --save_model      $SAVE_MODEL \
          --eval_model      $EVAL_MODEL \
                --use_rev_compl  $USE_REV_COMPL\
                --auto_balance_bce $AUTO_BALANCE_BCE\
                --test2_interval  $TEST2_INTERVAL\
                --train_perc  $TRAIN_PERC\
                --n_folds  $N_FOLDS\
                --use_x_y  $USE_X_Y\
                --save_dataset  $SAVE_DATASET\
                --model_type  $MODEL_TYPE\

#  gcloud ai-platform jobs submit training $JOB_NAME \
#           --job-dir $JOB_DIR \
#           --package-path $TRAINER_PACKAGE_PATH \
#           --module-name $MAIN_TRAINER_MODULE \
#           --region $REGION \
#           --runtime-version=$RUNTIME_VERSION \
#           --python-version=$PYTHON_VERSION \
#           --scale-tier $SCALE_TIER \
#           --master-machine-type $MACHINE_TYPE \
#           --master-accelerator $GPU_TYPE \
#           -- \
#           --random_seed     $RANDOM_SEED \
#           --seq_len         $SEQ_LEN \
#           --batch_size      $BATCH_SIZE \
#           --epochs          $EPOCHS \
#           --fold_fn_name    $FOLD_FN_NAME \
#           --output_name     $OUTPUT_NAME \
#           --midpoint        $MIDPOINT \
#           --chrom_idx       $CHROM_IDX \
#           --save_model      $SAVE_MODEL \
#           --eval_model      $EVAL_MODEL \
#                 --use_rev_compl  $USE_REV_COMPL\
#                 --auto_balance_bce $AUTO_BALANCE_BCE\
#                 --test2_interval  $TEST2_INTERVAL\
#                 --train_perc  $TRAIN_PERC\
#                 --n_folds  $N_FOLDS\
#                 --use_x_y  $USE_X_Y\
#                 --save_dataset  $SAVE_DATASET\
#                 --model_type  $MODEL_TYPE\