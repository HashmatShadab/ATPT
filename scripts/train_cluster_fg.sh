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
LOG_OUTPUT_DIR=${22:-"none"}
DATASET_NAME=${23:-"all"}
TRANSFERABILITY=${24:-"false"}
SOURCE_MODEL_NAME=${25:-"RN50"} # Options: RN50, ViT-B/16

IMAGE_ONLY_ATTACK=${26:-"false"} #
IMAGE_ONLY_ATTACK_TYPE=${27:-"prm"} #


IMAGE_FEATURE_PURIFY=${28:-"false"} #
IMAGE_FEATURE_PURIFY_TYPE=${29:-"noisy_anchor"} #
IMAGE_FEATURE_PURIFY_NOISY_ANCHORS=${30:-10} #
IMAGE_FEATURE_PURIFY_ANCHORS_ALPHA=${31:-1.2} #
IMAGE_FEATURE_PURIFY_NOISY_SIGMA=${32:-0.18} #
IMAGE_FEATURE_PURIFY_DIFF_THRESHOLD=${33:-0.0} #
DIFFPURE=${34:-"false"} #
COUNTER_ATTACK_INIT_NOISE=${35:-"uniform"}
COUNTER_ATTACK_GAUSSIAN_SIGMA=${36:-0.18}
COUNTER_ATTACK_TAU=${37:-"normal"}
COUNTER_ATTACK_NOISY_TAU_NUM_ANCHORS=${38:-10}
AUGMENTATION_POOL_ABLATION=${39:-"false"}
AUGMENTATION_POOL=${40:-"tpt"}


# Common parameters for all runs
COMMON_PARAMS="--gpu $GPU --n_ctx 4 --ctx_init a_photo_of_a --tpt_loss $TPT_LOSS"
COMMON_PARAMS+=" --output_dir $OUTPUT_DIR --log_output_dir $LOG_OUTPUT_DIR  --eps $EPSILON --steps $ATTACK_STEPS --transferability $TRANSFERABILITY --source_model $SOURCE_MODEL_NAME --image_only_attack $IMAGE_ONLY_ATTACK --image_only_attack_type $IMAGE_ONLY_ATTACK_TYPE"
COMMON_PARAMS+=" --selection_p $FRACTION_CONFIDENT_SAMPLES --tta_steps $TTA_STEPS --augmentation_pool_ablation $AUGMENTATION_POOL_ABLATION --augmentation_pool $AUGMENTATION_POOL"
COMMON_PARAMS+=" --ensemble_type $ENSEMBLE_TYPE --top_k $TOP_K_NEIGHBOURS_FOR_SIMILARITY_MATRIX --softmax_temp $SOFTMAX_TEMP_FOR_SIMILARITY_WEIGHTING"
COMMON_PARAMS+=" --diffpure $DIFFPURE --counter_attack $COUNTER_ATTACK --counter_attack_type $COUNTER_ATTACK_TYPE --counter_attack_tau $COUNTER_ATTACK_TAU --counter_attack_noisy_tau_num_anchors $COUNTER_ATTACK_NOISY_TAU_NUM_ANCHORS --counter_attack_init_noise $COUNTER_ATTACK_INIT_NOISE --counter_attack_gaussian_sigma $COUNTER_ATTACK_GAUSSIAN_SIGMA --counter_attack_steps $COUNTER_ATTACK_STEPS --counter_attack_eps $COUNTER_ATTACK_EPSILON"
COMMON_PARAMS+=" --counter_attack_alpha $COUNTER_ATTACK_ALPHA --counter_attack_tau_thres $COUNTER_ATTACK_TAU_THRES --counter_attack_beta $COUNTER_ATTACK_BETA --counter_attack_weighted_perturbations $COUNTER_ATTACK_W_PERTURBATION"
COMMON_PARAMS+=" --image_feature_purify $IMAGE_FEATURE_PURIFY --image_feature_purify_type $IMAGE_FEATURE_PURIFY_TYPE --image_feature_purify_noisy_anchors $IMAGE_FEATURE_PURIFY_NOISY_ANCHORS --image_feature_purify_anchors_alpha $IMAGE_FEATURE_PURIFY_ANCHORS_ALPHA --image_feature_purify_noisy_sigma $IMAGE_FEATURE_PURIFY_NOISY_SIGMA --image_feature_purify_diff_threshold $IMAGE_FEATURE_PURIFY_DIFF_THRESHOLD"

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
echo "Log Output Dir: $LOG_OUTPUT_DIR"
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
echo "TRANSFERABILITY : $TRANSFERABILITY"
echo "SOURCE_MODEL_NAME: $SOURCE_MODEL_NAME"
echo "Image Only Attack: $IMAGE_ONLY_ATTACK"
echo "Image Only Attack Type: $IMAGE_ONLY_ATTACK_TYPE"
echo "Image Feature Purify: $IMAGE_FEATURE_PURIFY"
echo "Image Feature Purify Type: $IMAGE_FEATURE_PURIFY_TYPE"
echo "Image Feature Purify Noisy Anchors: $IMAGE_FEATURE_PURIFY_NOISY_ANCHORS"
echo "Image Feature Purify Anchors Alpha: $IMAGE_FEATURE_PURIFY_ANCHORS_ALPHA"
echo "Image Feature Purify Noisy Sigma: $IMAGE_FEATURE_PURIFY_NOISY_SIGMA"
echo "Image Feature Purify Diff Threshold: $IMAGE_FEATURE_PURIFY_DIFF_THRESHOLD"

echo "DIFFPURE: $DIFFPURE"
echo "COUNTER_ATTACK_INIT_NOISE:  $COUNTER_ATTACK_INIT_NOISE"
echo "COUNTER_ATTACK_GAUSSIAN_SIGMA: $COUNTER_ATTACK_GAUSSIAN_SIGMA"
echo "COUNTER_ATTACK_TAU: $COUNTER_ATTACK_TAU"
echo "COUNTER_ATTACK_NOISY_TAU_NUM_ANCHORS: $COUNTER_ATTACK_NOISY_TAU_NUM_ANCHORS"


echo "Dataset Root: $DATA_ROOT"
echo "Dataset Name: $DATASET_NAME"
echo "========================"


#
# Section 1: Fine-grained Datasets
#


if [ "$DATASET_NAME" = "all"   ]; then


  echo "  [1/8] Testing Caltech101 dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Caltech101 $MODEL $COMMON_PARAMS
  echo "  ✓ Caltech101 dataset testing complete"

  echo "  [2/8] Testing Cars dataset..."
   python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Cars $MODEL $COMMON_PARAMS
  echo "  ✓ Cars dataset testing complete"


  echo "  [3/8] Testing DTD dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets DTD $MODEL $COMMON_PARAMS
  echo "  ✓ DTD dataset testing complete"

  echo "  [4/8] Testing Flower102 dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Flower102 $MODEL $COMMON_PARAMS
  echo "  ✓ Flower102 dataset testing complete"

  echo "  [5/8] Testing Aircraft dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Aircraft $MODEL $COMMON_PARAMS
  echo "  ✓ Aircraft dataset testing complete"

  echo "  [6/8] Testing UCF101 dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets UCF101 $MODEL $COMMON_PARAMS
  echo "  ✓ UCF101 dataset testing complete"


  echo "  [7/8] Testing eurosat dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets eurosat $MODEL $COMMON_PARAMS
  echo "  ✓ eurosat dataset testing complete"

  echo "  [8/8] Testing Pets dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Pets $MODEL $COMMON_PARAMS
  echo "  ✓ Pets dataset testing complete"
  echo "Fine-grained datasets testing complete"


elif [ "$DATASET_NAME" = "caltech101"   ]; then
  echo "Running tests on Fine-grained datasets..."
  echo "Testing Caltech101 dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Caltech101 $MODEL $COMMON_PARAMS
  echo "  ✓ Caltech101 dataset testing complete"

elif [ "$DATASET_NAME" = "cars" ]; then
  echo "Running tests on Fine-grained datasets..."
  echo "Testing Cars dataset..."
   python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Cars $MODEL $COMMON_PARAMS
  echo "  ✓ Cars dataset testing complete"

elif [ "$DATASET_NAME" = "dtd" ]; then
  echo "Running tests on Fine-grained datasets..."
  echo "Testing DTD dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets DTD $MODEL $COMMON_PARAMS
  echo "  ✓ DTD dataset testing complete"

elif [ "$DATASET_NAME" = "flower102" ]; then
  echo "Running tests on Fine-grained datasets..."
  echo "Testing Flower102 dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Flower102 $MODEL $COMMON_PARAMS
  echo "  ✓ Flower102 dataset testing complete"

elif [ "$DATASET_NAME" = "aircraft" ]; then
  echo "Running tests on Fine-grained datasets..."
  echo "Testing Aircraft dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Aircraft $MODEL $COMMON_PARAMS
  echo "  ✓ Aircraft dataset testing complete"

elif [ "$DATASET_NAME" = "ucf101" ]; then
  echo "Running tests on Fine-grained datasets..."
  echo "Testing UCF101 dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets UCF101 $MODEL $COMMON_PARAMS
  echo "  ✓ UCF101 dataset testing complete"

elif [ "$DATASET_NAME" = "eurosat" ]; then
  echo "Running tests on Fine-grained datasets..."
  echo "Testing eurosat dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets eurosat $MODEL $COMMON_PARAMS
  echo "  ✓ eurosat dataset testing complete"

elif [ "$DATASET_NAME" = "pets" ]; then
  echo "Running tests on Fine-grained datasets..."
  echo "Testing Pets dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Pets $MODEL $COMMON_PARAMS
  echo "  ✓ Pets dataset testing complete"

elif [ "$DATASET_NAME" = "pets" ]; then
  echo "Running tests on Fine-grained datasets..."
  echo "Testing Pets dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets Pets $MODEL $COMMON_PARAMS
  echo "  ✓ Pets dataset testing complete"

elif [ "$DATASET_NAME" = "A" ]; then
  echo "  [1/5] Adv Examples  ImageNet-A dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets A $MODEL $COMMON_PARAMS
  echo "  ✓ A dataset testing complete"

elif [ "$DATASET_NAME" = "R" ]; then
  echo "  [2/5] Adv Examples  ImageNet-R dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets R $MODEL $COMMON_PARAMS
  echo "  ✓ R dataset testing complete"

elif [ "$DATASET_NAME" = "K" ]; then
  echo "  [3/5] Adv Examples  ImageNet-S dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets K $MODEL $COMMON_PARAMS
  echo "  ✓ K dataset testing complete"

elif [ "$DATASET_NAME" = "V" ]; then
  echo "  [4/5] Adv Examples  ImageNet-V dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets V $MODEL $COMMON_PARAMS
  echo "  ✓ V dataset testing complete"

elif [ "$DATASET_NAME" = "I" ]; then
  echo "  [5/5] Adv Examples  ImageNet dataset..."
  python rtpt_weighted_ensembling.py $DATA_ROOT --test_sets I $MODEL $COMMON_PARAMS
  echo "  ✓ I dataset testing complete"

fi
