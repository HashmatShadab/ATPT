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

# Common parameters for all runs
COMMON_PARAMS="--gpu $GPU --ctx_init a_photo_of_a --output_dir output_results --workers $NUM_WORKERS"
COMMON_PARAMS+=" --eps $EPSILON --steps $ATTACK_STEPS --tta_steps $TTA_STEPS"
COMMON_PARAMS+=" --selection_p $FRACTION_CONFIDENT_SAMPLES"
COMMON_PARAMS+=" --top_k $TOP_K_NEIGHBOURS_FOR_SIMILARITY_MATRIX"
COMMON_PARAMS+=" --softmax_temp $SOFTMAX_TEMP_FOR_SIMILARITY_WEIGHTING"

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
echo "Model: RN50 with batch size 64"
echo "========================"

#
# Section 1: Fine-grained Datasets
#
echo "Running tests on Fine-grained datasets..."

echo "  [1/8] Testing DTD dataset..."
python rtpt.py $DATA_ROOT --test_sets DTD $MODEL $COMMON_PARAMS
echo "  ✓ DTD dataset testing complete"

echo "  [2/8] Testing Flower102 dataset..."
python rtpt.py $DATA_ROOT --test_sets Flower102 $MODEL $COMMON_PARAMS
echo "  ✓ Flower102 dataset testing complete"

echo "  [3/8] Testing Cars dataset..."
python rtpt.py $DATA_ROOT --test_sets Cars $MODEL $COMMON_PARAMS
echo "  ✓ Cars dataset testing complete"

echo "  [4/8] Testing Aircraft dataset..."
python rtpt.py $DATA_ROOT --test_sets Aircraft $MODEL $COMMON_PARAMS
echo "  ✓ Aircraft dataset testing complete"

echo "  [5/8] Testing Pets dataset..."
python rtpt.py $DATA_ROOT --test_sets Pets $MODEL $COMMON_PARAMS
echo "  ✓ Pets dataset testing complete"

echo "  [6/8] Testing Caltech101 dataset..."
python rtpt.py $DATA_ROOT --test_sets Caltech101 $MODEL $COMMON_PARAMS
echo "  ✓ Caltech101 dataset testing complete"

echo "  [7/8] Testing UCF101 dataset..."
python rtpt.py $DATA_ROOT --test_sets UCF101 $MODEL $COMMON_PARAMS
echo "  ✓ UCF101 dataset testing complete"

echo "  [8/8] Testing eurosat dataset..."
python rtpt.py $DATA_ROOT --test_sets eurosat $MODEL $COMMON_PARAMS
echo "  ✓ eurosat dataset testing complete"

echo "Fine-grained datasets testing complete"

#
# Section 2: ImageNet Datasets
#
echo "Running tests on ImageNet datasets..."

echo "  [1/5] Testing ImageNet-A dataset..."
python rtpt.py $DATA_ROOT --test_sets A $MODEL $COMMON_PARAMS
echo "  ✓ ImageNet-A dataset testing complete"

echo "  [2/5] Testing ImageNet-R dataset..."
python rtpt.py $DATA_ROOT --test_sets R $MODEL $COMMON_PARAMS
echo "  ✓ ImageNet-R dataset testing complete"

echo "  [3/5] Testing ImageNet-S dataset..."
python rtpt.py $DATA_ROOT --test_sets K $MODEL $COMMON_PARAMS
echo "  ✓ ImageNet-S dataset testing complete"

echo "  [4/5] Testing ImageNet-V dataset..."
python rtpt.py $DATA_ROOT --test_sets V $MODEL $COMMON_PARAMS
echo "  ✓ ImageNet-V dataset testing complete"

echo "  [5/5] Testing ImageNet dataset..."
python rtpt.py $DATA_ROOT --test_sets I $MODEL $COMMON_PARAMS
echo "  ✓ ImageNet dataset testing complete"

echo "ImageNet datasets testing complete"

# Add final completion message
echo "=== All tests completed successfully ==="
