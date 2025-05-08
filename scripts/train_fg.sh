#!/bin/bash
#
# ATPT Training Script
# This script runs the RTPT model on various datasets with specified parameters
#

echo "=== Starting ATPT Training Script ==="
echo "Initializing parameters..."

# Input parameters with defaults
DATA_ROOT=${1:-"F:\Code\datasets\downstream_datasets\downstream_datasets"}
GPU=${2:-0}
NUM_WORKERS=${3:-0}

# Model parameters
MODEL_NAME=${4:-"RN50"} # Options: RN50, ViT-B/16

# Adversarial example parameters, defaults attack in paper as epsilon=1.0 and steps=7 for RN50 and epsilon=4.0 and steps=100 for ViT-B/16
EPSILON=${5:-0.0}
ATTACK_STEPS=${6:-0}

# Test-time augmentation parameters
TTA_STEPS=${7:-1}
FRACTION_CONFIDENT_SAMPLES=${8:-0.1}
TOP_K_NEIGHBOURS_FOR_SIMILARITY_MATRIX=${9:-20}
SOFTMAX_TEMP_FOR_SIMILARITY_WEIGHTING=${10:-0.01}
TPT_LOSS=${11:-"rtpt"}
ENSEMBLE_TYPE=${12:-"weighted_rtpt"}

OUTPUT_DIR=${13:-"output_results"}
COUNTER_ATTACK=${14:-"false"}
COUNTER_ATTACK_TYPE=${15:-"pgd"}
COUNTER_ATTACK_STEPS=${16:-2}
COUNTER_ATTACK_EPSILON=${17:-4.0}
COUNTER_ATTACK_ALPHA=${18:-1.0}
COUNTER_ATTACK_TAU_THRES=${19:-0.2}
COUNTER_ATTACK_BETA=${20:-2.0}
COUNTER_ATTACK_W_PERTURBATION=${21:-"true"}


############################################
JOB_ID=${22:-1}



# bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 1.0 1 1 0.1 20 0.01 rtpt weighted_rtpt "F:\Code\datasets\atpt_results"
# bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 1.0 1 1 0.1 20 0.01 rtpt vanilla "F:\Code\datasets\atpt_results"
# bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 1.0 1 1 0.1 20 0.01 rtpt none "F:\Code\datasets\atpt_results"
# bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 1.0 1 1 0.1 20 0.01 tpt weighted_rtpt "F:\Code\datasets\atpt_results"
# bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 1.0 1 1 0.1 20 0.01 tpt vanilla "F:\Code\datasets\atpt_results"
# bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 1.0 1 1 0.1 20 0.01 tpt none "F:\Code\datasets\atpt_results"



# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 0 4 RN50 1.0 7 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 0 4 RN50 4.0 7 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-B/16 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-B/16 1.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-L/14 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-L/14 1.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 fare2 1.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 fare2 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 tecoa2 1.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 tecoa2 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 fare4 1.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 fare4 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 tecoa4 1.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash train.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 tecoa4 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /mnt/nvme0n1/Dataset/muzammal/atpt_results



# Common parameters for all runs
COMMON_PARAMS="--gpu $GPU --n_ctx 4 --ctx_init a_photo_of_a --tpt_loss $TPT_LOSS"
COMMON_PARAMS+=" --output_dir $OUTPUT_DIR  --eps $EPSILON --steps $ATTACK_STEPS"
COMMON_PARAMS+=" --selection_p $FRACTION_CONFIDENT_SAMPLES --tta_steps $TTA_STEPS"
COMMON_PARAMS+=" --ensemble_type $ENSEMBLE_TYPE --top_k $TOP_K_NEIGHBOURS_FOR_SIMILARITY_MATRIX --softmax_temp $SOFTMAX_TEMP_FOR_SIMILARITY_WEIGHTING"
COMMON_PARAMS+=" --counter_attack $COUNTER_ATTACK --counter_attack_type $COUNTER_ATTACK_TYPE --counter_attack_steps $COUNTER_ATTACK_STEPS --counter_attack_eps $COUNTER_ATTACK_EPSILON"
COMMON_PARAMS+=" --counter_attack_alpha $COUNTER_ATTACK_ALPHA --counter_attack_tau_thres $COUNTER_ATTACK_TAU_THRES --counter_attack_beta $COUNTER_ATTACK_BETA --counter_attack_weighted_perturbations $COUNTER_ATTACK_W_PERTURBATION"

# Model parameters
MODEL="-a $MODEL_NAME -b 64 --workers $NUM_WORKERS --print-freq 20"

# Display configuration
# Display configuration
echo "=== Configuration ==="
echo "GPU: $GPU"
echo "Model: $MODEL_NAME with batch size 64"
echo "Workers: $NUM_WORKERS"
echo "Context Init for TPT: a_photo_of_a"
echo "TPT Loss: $TPT_LOSS"
echo "Output Dir: $OUTPUT_DIR"
echo "Epsilon for Adversarial Examples: $EPSILON"
echo "Attack Steps for Adversarial Examples: $ATTACK_STEPS"
echo "Fraction Confident Samples to select views with low entropy: $FRACTION_CONFIDENT_SAMPLES"
echo "TTA Steps: $TTA_STEPS"
echo "Ensemble Type for inference: $ENSEMBLE_TYPE"
echo "Top K Neighbours for weighted ensemble: $TOP_K_NEIGHBOURS_FOR_SIMILARITY_MATRIX"
echo "Softmax Temperature for weighted ensemble: $SOFTMAX_TEMP_FOR_SIMILARITY_WEIGHTING"
echo "Counter Attack: $COUNTER_ATTACK"
echo "Counter Attack Type: $COUNTER_ATTACK_TYPE"
echo "Counter Attack Steps: $COUNTER_ATTACK_STEPS"
echo "Counter Attack Epsilon: $COUNTER_ATTACK_EPSILON"
echo "Counter Attack Alpha: $COUNTER_ATTACK_ALPHA"
echo "Counter Attack Tau Threshold: $COUNTER_ATTACK_TAU_THRES"
echo "Counter Attack Beta: $COUNTER_ATTACK_BETA"
echo "Counter Attack Weighted Perturbations: $COUNTER_ATTACK_W_PERTURBATION"
echo "========================"


#
# Section 1: Fine-grained Datasets
#

echo "Running tests on Fine-grained datasets..."
echo "  [1/8] Testing DTD dataset..."
CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets DTD $MODEL $COMMON_PARAMS
echo "  ✓ DTD dataset testing complete"

echo "  [2/8] Testing Flower102 dataset..."
CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Flower102 $MODEL $COMMON_PARAMS
echo "  ✓ Flower102 dataset testing complete"

echo "  [3/8] Testing Cars dataset..."
CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Cars $MODEL $COMMON_PARAMS
echo "  ✓ Cars dataset testing complete"

echo "Running tests on Fine-grained datasets..."
echo "  [4/8] Testing Aircraft dataset..."
CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Aircraft $MODEL $COMMON_PARAMS
echo "  ✓ Aircraft dataset testing complete"

echo "  [5/8] Testing Pets dataset..."
CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Pets $MODEL $COMMON_PARAMS
echo "  ✓ Pets dataset testing complete"

echo "  [6/8] Testing Caltech101 dataset..."
CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Caltech101 $MODEL $COMMON_PARAMS
echo "  ✓ Caltech101 dataset testing complete"

echo "Running tests on Fine-grained datasets..."
echo "  [7/8] Testing UCF101 dataset..."
CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets UCF101 $MODEL $COMMON_PARAMS
echo "  ✓ UCF101 dataset testing complete"

echo "  [8/8] Testing eurosat dataset..."
CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets eurosat $MODEL $COMMON_PARAMS
echo "  ✓ eurosat dataset testing complete"

echo "Fine-grained datasets testing complete"

#elif [ "$JOB_ID" -eq 4 ]; then
#  echo "Running tests on ImageNet datasets..."
#  echo "  [1/5] Testing ImageNet-A dataset..."
#  CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets A $MODEL $COMMON_PARAMS
#  echo "  ✓ ImageNet-A dataset testing complete"
#
#  echo "  [2/5] Testing ImageNet-V dataset..."
#  CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets V $MODEL $COMMON_PARAMS
#  echo "  ✓ ImageNet-V dataset testing complete"
#
#elif [ "$JOB_ID" -eq 5 ]; then
#  echo "Running tests on ImageNet datasets..."
#  echo "  [3/5] Testing ImageNet-R dataset..."
#  CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets R $MODEL $COMMON_PARAMS
#  echo "  ✓ ImageNet-R dataset testing complete"
#
#  echo "  [4/5] Testing ImageNet-V dataset..."
#  CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets K $MODEL $COMMON_PARAMS
#  echo "  ✓ ImageNet-V dataset testing complete"
#
#elif [ "$JOB_ID" -eq 6 ]; then
#  echo "Running tests on ImageNet datasets..."
#  echo "  [5/5] Testing ImageNet dataset..."
#  CUDA_VISIBLE_DEVICES=$GPU python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets I $MODEL $COMMON_PARAMS
#  echo "  ✓ ImageNet dataset testing complete"
#
#  echo "ImageNet datasets testing complete"
#fi
