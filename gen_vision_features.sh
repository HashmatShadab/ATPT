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
DIFF_THRESHOLD=${18:-0.0}
IMAGE_ONLY_ATTACK=${19:-"false"}
IMAGE_ONLY_ATTACK_TYPE=${20:-"prm"}
DATASET_ID=${21:-"all"}
COUNTER_ATTACK=${22:-"false"}
COUNTER_ATTACK_STEPS=${23:-0}
COUNTER_ATTACK_INIT_NOISE=${24:-"uniform"}
COUNTER_ATTACK_EPS=${25:-4.0}
COUNTER_ATTACK_GAUSSIAN_SIGMA=${26:-0.03}
COUNTER_ATTACK_TAU=${27:-"normal"}
COUNTER_ATTACK_NUM_ANCHORS=${28:-10}



# Common parameters for all runs
COMMON_PARAMS="--gpu 0 --ctx_init a_photo_of_a --output_dir $OUTPUT_DIR --log_output_dir $LOG_OUTPUT_DIR --workers $NUM_WORKERS"
COMMON_PARAMS+=" --eps $EPSILON --steps $ATTACK_STEPS --tta_steps $TTA_STEPS --image_only_attack $IMAGE_ONLY_ATTACK --image_only_attack_type $IMAGE_ONLY_ATTACK_TYPE --add_noise $COUNTER_ATTACK  --add_noise_init_noise $COUNTER_ATTACK_INIT_NOISE --add_noise_eps $COUNTER_ATTACK_EPS  --add_noise_gaussian_sigma $COUNTER_ATTACK_GAUSSIAN_SIGMA --add_noise_tau $COUNTER_ATTACK_TAU --add_noise_noisy_tau_num_anchors $COUNTER_ATTACK_NUM_ANCHORS"
COMMON_PARAMS+=" --selection_p $FRACTION_CONFIDENT_SAMPLES"
COMMON_PARAMS+=" --top_k $TOP_K_NEIGHBOURS_FOR_SIMILARITY_MATRIX"
COMMON_PARAMS+=" --softmax_temp $SOFTMAX_TEMP_FOR_SIMILARITY_WEIGHTING --print-freq 20"

# Model parameters
MODEL="-a $MODEL_NAME -b 16 --adv_bs 16"

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
echo "Model: $MODEL_NAME with batch size 8"
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

#  echo "  [1/8] Adv Examples  DTD dataset..."
#  python rtpt_adv_generation.py $DATA_ROOT --test_sets DTD $MODEL $COMMON_PARAMS
#  echo "  ✓ DTD dataset Adv Examples  complete"
#
#  echo "  [2/8] Adv Examples  Flower102 dataset..."
#  python rtpt_adv_generation.py $DATA_ROOT --test_sets Flower102 $MODEL $COMMON_PARAMS
#  echo "  ✓ Flower102 dataset Adv Examples  complete"

#  echo "  [3/8] Adv Examples  Cars dataset..."
#  python rtpt_adv_generation.py $DATA_ROOT --test_sets Cars $MODEL $COMMON_PARAMS
#  echo "  ✓ Cars dataset Adv Examples  complete"

#  echo "  [4/8] Adv Examples  Aircraft dataset..."
#  python rtpt_adv_generation.py $DATA_ROOT --test_sets Aircraft $MODEL $COMMON_PARAMS
#  echo "  ✓ Aircraft dataset Adv Examples  complete"

#  echo "  [5/8] Adv Examples  Pets dataset..."
#  python adv_gen_analysis.py $DATA_ROOT --test_sets Pets $MODEL $COMMON_PARAMS
#  echo "  ✓ Pets dataset Adv Examples  complete"
#
  echo "  [6/8] Adv Examples  Caltech101 dataset..."
  python adv_gen_analysis.py $DATA_ROOT --test_sets Caltech101 $MODEL $COMMON_PARAMS
  echo "  ✓ Caltech101 dataset Adv Examples  complete"

#  echo "  [7/8] Adv Examples  UCF101 dataset..."
#  python adv_gen_analysis.py $DATA_ROOT --test_sets UCF101 $MODEL $COMMON_PARAMS
#  echo "  ✓ UCF101 dataset Adv Examples  complete"

#  echo "  [8/8] Adv Examples  eurosat dataset..."
#  python rtpt_adv_generation.py $DATA_ROOT --test_sets eurosat $MODEL $COMMON_PARAMS
#  echo "  ✓ eurosat dataset Adv Examples  complete"

  echo "Fine-grained datasets Adv Examples  complete"

elif [ "$DATASET_ID" = "all_1"   ]; then
  echo "Running tests on Fine-grained datasets..."

  echo "  [1/8] Adv Examples  DTD dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets DTD $MODEL $COMMON_PARAMS
  echo "  ✓ DTD dataset Adv Examples  complete"

  echo "  [2/8] Adv Examples  Flower102 dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets Flower102 $MODEL $COMMON_PARAMS
  echo "  ✓ Flower102 dataset Adv Examples  complete"

  echo "  [3/8] Adv Examples  Cars dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets Cars $MODEL $COMMON_PARAMS
  echo "  ✓ Cars dataset Adv Examples  complete"

  echo "  [4/8] Adv Examples  Aircraft dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets Aircraft $MODEL $COMMON_PARAMS
  echo "  ✓ Aircraft dataset Adv Examples  complete"



  echo "Fine-grained datasets part 1 Adv Examples  complete"

elif [ "$DATASET_ID" = "all_2"   ]; then
  echo "Running tests on Fine-grained datasets..."

  echo "  [5/8] Adv Examples  Pets dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets Pets $MODEL $COMMON_PARAMS
  echo "  ✓ Pets dataset Adv Examples  complete"

  echo "  [6/8] Adv Examples  Caltech101 dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets Caltech101 $MODEL $COMMON_PARAMS
  echo "  ✓ Caltech101 dataset Adv Examples  complete"

  echo "  [7/8] Adv Examples  UCF101 dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets UCF101 $MODEL $COMMON_PARAMS
  echo "  ✓ UCF101 dataset Adv Examples  complete"

  echo "  [8/8] Adv Examples  eurosat dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets eurosat $MODEL $COMMON_PARAMS
  echo "  ✓ eurosat dataset Adv Examples  complete"

  echo "Fine-grained datasets part 2 Adv Examples  complete"


elif [ "$DATASET_ID" = "all_3"   ]; then
  echo "Running tests on ImageNet datasets..."

  echo "  [1/5] Adv Examples  ImageNet-A dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets A $MODEL $COMMON_PARAMS
  echo "  ✓ ImageNet-A dataset Adv Examples  complete"

  echo "  [2/5] Adv Examples  ImageNet-R dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets R $MODEL $COMMON_PARAMS
  echo "  ✓ ImageNet-R dataset Adv Examples  complete"

  echo "  [3/5] Adv Examples  ImageNet-S dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets K $MODEL $COMMON_PARAMS
  echo "  ✓ ImageNet-S dataset Adv Examples  complete"

  echo "  [4/5] Adv Examples  ImageNet-V dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets V $MODEL $COMMON_PARAMS
  echo "  ✓ ImageNet-V dataset Adv Examples  complete"

  echo "  [5/5] Adv Examples  ImageNet dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets I $MODEL $COMMON_PARAMS
  echo "  ✓ ImageNet dataset Adv Examples  complete"


elif [ "$DATASET_ID" = "dtd"   ]; then
  echo "  [1/8] Adv Examples  DTD dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets DTD $MODEL $COMMON_PARAMS
  echo "  ✓ DTD dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "flower102"   ]; then
  echo "  [2/8] Adv Examples  Flower102 dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets Flower102 $MODEL $COMMON_PARAMS
  echo "  ✓ Flower102 dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "cars"   ]; then
  echo "  [3/8] Adv Examples  Cars dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets Cars $MODEL $COMMON_PARAMS
  echo "  ✓ Cars dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "aircraft"   ]; then
  echo "  [4/8] Adv Examples  Aircraft dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets Aircraft $MODEL $COMMON_PARAMS
  echo "  ✓ Aircraft dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "pets"   ]; then

  echo "  [5/8] Adv Examples  Pets dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets Pets $MODEL $COMMON_PARAMS
  echo "  ✓ Pets dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "caltech101"   ]; then
  echo "  [6/8] Adv Examples  Caltech101 dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets Caltech101 $MODEL $COMMON_PARAMS
  echo "  ✓ Caltech101 dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "ucf101"   ]; then
  echo "  [7/8] Adv Examples  UCF101 dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets UCF101 $MODEL $COMMON_PARAMS
  echo "  ✓ UCF101 dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "eurosat"   ]; then
    echo "  [8/8] Adv Examples  eurosat dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets eurosat $MODEL $COMMON_PARAMS
  echo "  ✓ eurosat dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "A"   ]; then
  echo "  [1/5] Adv Examples  ImageNet-A dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets A $MODEL $COMMON_PARAMS
  echo "  ✓ ImageNet-A dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "R"   ]; then
  echo "  [2/5] Adv Examples  ImageNet-R dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets R $MODEL $COMMON_PARAMS
  echo "  ✓ ImageNet-R dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "K"   ]; then
  echo "  [3/5] Adv Examples  ImageNet-S dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets K $MODEL $COMMON_PARAMS
  echo "  ✓ ImageNet-S dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "V"   ]; then
  echo "  [4/5] Adv Examples  ImageNet-V dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets V $MODEL $COMMON_PARAMS
  echo "  ✓ ImageNet-V dataset Adv Examples  complete"

elif [ "$DATASET_ID" = "I"   ]; then
  echo "  [5/5] Adv Examples  ImageNet dataset..."
  python rtpt_adv_generation.py $DATA_ROOT --test_sets I $MODEL $COMMON_PARAMS
  echo "  ✓ ImageNet dataset Adv Examples  complete"


fi



#
## Section 2: ImageNet Datasets
####
#echo "Generating Adv Examples  on ImageNet datasets..."
##
#echo "  [1/5] Adv Examples  ImageNet-A dataset..."
#python rtpt_adv_generation.py $DATA_ROOT --test_sets A $MODEL $COMMON_PARAMS
#echo "  ✓ ImageNet-A dataset Adv Examples  complete"
#
#echo "  [2/5] Adv Examples  ImageNet-R dataset..."
#python rtpt_adv_generation.py $DATA_ROOT --test_sets R $MODEL $COMMON_PARAMS
#echo "  ✓ ImageNet-R dataset Adv Examples  complete"
#
#echo "  [3/5] Adv Examples  ImageNet-S dataset..."
#python rtpt_adv_generation.py $DATA_ROOT --test_sets K $MODEL $COMMON_PARAMS
#echo "  ✓ ImageNet-S dataset Adv Examples  complete"
#
#echo "  [4/5] Adv Examples  ImageNet-V dataset..."
#python rtpt_adv_generation.py $DATA_ROOT --test_sets V $MODEL $COMMON_PARAMS
#echo "  ✓ ImageNet-V dataset Adv Examples  complete"
#
#echo "  [5/5] Adv Examples  ImageNet dataset..."
#python rtpt_adv_generation.py $DATA_ROOT --test_sets I $MODEL $COMMON_PARAMS
#echo "  ✓ ImageNet dataset Adv Examples  complete"
#
#echo "ImageNet datasets Adv Examples  complete"
#
## Add final completion message
#echo "=== All tests completed successfully ==="
