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
OUTPUT_DIR=${11:-"output_results"}
LOG_OUTPUT_DIR=${12:-"output_results"}

IMAGE_PURIFY=${13:-"false"}
PURIFY_TYPE=${14:-"noisy_anchor"}
NOISY_ANCHORS=${15:-10}
ANCHORS_ALPHA=${16:-1.2}
NOISY_SIGMA=${17:-0.18}
DIFF_THRESHOLD=${18:-0.85}
IMAGE_ONLY_ATTACK=${19:-"true"}
IMAGE_ONLY_ATTACK_TYPE=${20:-"prm"}
DATASET_ID=${21:-"all"}

# bash gen_adv.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 1.0 1 1 0.1 20 0.01 "F:\Code\datasets\atpt_results"


# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 0 4 RN50 1.0 7 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 0 4 RN50 4.0 7 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-B/16 4.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-B/16 1.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-L/14 1.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 fare2 1.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 fare2 4.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 tecoa2 1.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 tecoa2 4.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 fare4 1.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 fare4 4.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 tecoa4 1.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
# bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 tecoa4 4.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results

# Common parameters for all runs
COMMON_PARAMS="--gpu 0 --ctx_init a_photo_of_a --output_dir $OUTPUT_DIR --log_output_dir $LOG_OUTPUT_DIR --workers $NUM_WORKERS"
COMMON_PARAMS+=" --eps $EPSILON --steps $ATTACK_STEPS --tta_steps $TTA_STEPS --image_only_attack $IMAGE_ONLY_ATTACK --image_only_attack $IMAGE_ONLY_ATTACK_TYPE"
COMMON_PARAMS+=" --selection_p $FRACTION_CONFIDENT_SAMPLES"
COMMON_PARAMS+=" --top_k $TOP_K_NEIGHBOURS_FOR_SIMILARITY_MATRIX"
COMMON_PARAMS+=" --softmax_temp $SOFTMAX_TEMP_FOR_SIMILARITY_WEIGHTING --print-freq 20 --image_feature_purify $IMAGE_PURIFY --image_feature_purify_type $PURIFY_TYPE --image_feature_purify_noisy_anchors $NOISY_ANCHORS --image_feature_purify_anchors_alpha $ANCHORS_ALPHA --image_feature_purify_noisy_sigma $NOISY_SIGMA --image_feature_purify_diff_threshold $DIFF_THRESHOLD"

# Model parameters
MODEL="-a $MODEL_NAME -b 64"

# Display configuration
echo "=== Configuration ==="
echo "Data Root: $DATA_ROOT"
echo "GPU: $GPU"
echo "Workers: $NUM_WORKERS"
echo "Epsilon: $EPSILON"
echo "Attack Steps: $ATTACK_STEPS"
echo "TTA Steps: $TTA_STEPS"
echo "Fraction Confident Samples: $FRACTION_CONFIDENT_SAMPLES"
echo "Top K Neighbours: $TOP_K_NEIGHBOURS_FOR_SIMILARITY_MATRIX"
echo "Softmax Temperature: $SOFTMAX_TEMP_FOR_SIMILARITY_WEIGHTING"
echo "Model: $MODEL_NAME with batch size 64"
echo "Output Directory: $OUTPUT_DIR"
echo "Log Output Directory: $LOG_OUTPUT_DIR"
echo "Image Feature Purify: $IMAGE_PURIFY"
echo "Image Feature Purify Type: $PURIFY_TYPE"
echo "Image Feature Purify Noisy Anchors: $NOISY_ANCHORS"
echo "Image Feature Purify Anchors Alpha: $ANCHORS_ALPHA"
echo "Image Feature Purify Noisy Sigma: $NOISY_SIGMA"
echo "Image Feature Purify Diff Threshold: $DIFF_THRESHOLD"
echo "Image Only Attack: $IMAGE_ONLY_ATTACK"
echo "Image Only Attack Type: $IMAGE_ONLY_ATTACK_TYPE"
echo "Dataset ID: $DATASET_ID"
echo "========================"

#
# Section 1: Fine-grained Datasets
#
echo "Generating Adv Examples  on Fine-grained datasets..."



if [ "$DATASET_ID" = "all"   ]; then
  echo "Running tests on Fine-grained datasets..."

  echo "  [1/8] Adv Examples  DTD dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets DTD $MODEL $COMMON_PARAMS
  echo "  ✓ DTD dataset Adv Examples  complete"

  echo "  [2/8] Adv Examples  Flower102 dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets Flower102 $MODEL $COMMON_PARAMS
  echo "  ✓ Flower102 dataset Adv Examples  complete"

  echo "  [3/8] Adv Examples  Cars dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets Cars $MODEL $COMMON_PARAMS
  echo "  ✓ Cars dataset Adv Examples  complete"

  echo "  [4/8] Adv Examples  Aircraft dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets Aircraft $MODEL $COMMON_PARAMS
  echo "  ✓ Aircraft dataset Adv Examples  complete"

  echo "  [5/8] Adv Examples  Pets dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets Pets $MODEL $COMMON_PARAMS
  echo "  ✓ Pets dataset Adv Examples  complete"

  echo "  [6/8] Adv Examples  Caltech101 dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets Caltech101 $MODEL $COMMON_PARAMS
  echo "  ✓ Caltech101 dataset Adv Examples  complete"

  echo "  [7/8] Adv Examples  UCF101 dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets UCF101 $MODEL $COMMON_PARAMS
  echo "  ✓ UCF101 dataset Adv Examples  complete"

  echo "  [8/8] Adv Examples  eurosat dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets eurosat $MODEL $COMMON_PARAMS
  echo "  ✓ eurosat dataset Adv Examples  complete"

  echo "Fine-grained datasets Adv Examples  complete"

elif [ "$DATASET_ID" = "dtd"   ]; then
  echo "  [1/8] Adv Examples  DTD dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets DTD $MODEL $COMMON_PARAMS
  echo "  ✓ DTD dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "flower102"   ]; then
  echo "  [2/8] Adv Examples  Flower102 dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets Flower102 $MODEL $COMMON_PARAMS
  echo "  ✓ Flower102 dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "cars"   ]; then
  echo "  [3/8] Adv Examples  Cars dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets Cars $MODEL $COMMON_PARAMS
  echo "  ✓ Cars dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "aircraft"   ]; then
  echo "  [4/8] Adv Examples  Aircraft dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets Aircraft $MODEL $COMMON_PARAMS
  echo "  ✓ Aircraft dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "pets"   ]; then

  echo "  [5/8] Adv Examples  Pets dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets Pets $MODEL $COMMON_PARAMS
  echo "  ✓ Pets dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "caltech101"   ]; then
  echo "  [6/8] Adv Examples  Caltech101 dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets Caltech101 $MODEL $COMMON_PARAMS
  echo "  ✓ Caltech101 dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "ucf101"   ]; then
  echo "  [7/8] Adv Examples  UCF101 dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets UCF101 $MODEL $COMMON_PARAMS
  echo "  ✓ UCF101 dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "eurosat"   ]; then
    echo "  [8/8] Adv Examples  eurosat dataset..."
  CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets eurosat $MODEL $COMMON_PARAMS
  echo "  ✓ eurosat dataset Adv Examples  complete"

fi



#
## Section 2: ImageNet Datasets
####
#echo "Generating Adv Examples  on ImageNet datasets..."
##
#echo "  [1/5] Adv Examples  ImageNet-A dataset..."
#CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets A $MODEL $COMMON_PARAMS
#echo "  ✓ ImageNet-A dataset Adv Examples  complete"
#
#echo "  [2/5] Adv Examples  ImageNet-R dataset..."
#CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets R $MODEL $COMMON_PARAMS
#echo "  ✓ ImageNet-R dataset Adv Examples  complete"
#
#echo "  [3/5] Adv Examples  ImageNet-S dataset..."
#CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets K $MODEL $COMMON_PARAMS
#echo "  ✓ ImageNet-S dataset Adv Examples  complete"
#
#echo "  [4/5] Adv Examples  ImageNet-V dataset..."
#CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets V $MODEL $COMMON_PARAMS
#echo "  ✓ ImageNet-V dataset Adv Examples  complete"
#
#echo "  [5/5] Adv Examples  ImageNet dataset..."
#CUDA_VISIBLE_DEVICES=$GPU python rtpt_adv_generation.py $DATA_ROOT --test_sets I $MODEL $COMMON_PARAMS
#echo "  ✓ ImageNet dataset Adv Examples  complete"
#
#echo "ImageNet datasets Adv Examples  complete"
#
## Add final completion message
#echo "=== All tests completed successfully ==="
