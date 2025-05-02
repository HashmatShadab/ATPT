#!/bin/bash


DATA_ROOT=${1:-"F:\Code\datasets\downstream_datasets\downstream_datasets"}
ADV_DATA_ROOT=${2:-"F:\Code\datasets\atpt_results"}
GPU=${3:-0}

# Record start time
START_TIME=$(date +%s)

# bash tta_rn50.sh "/mnt/nvme0n1/Dataset/muzammal/downstream_datasets" "/mnt/nvme0n1/Dataset/muzammal/atpt_results" 0

# R-TPT Baseline RN50

bash train.sh $DATA_ROOT $GPU 4 RN50 0 0 1 0.1 20 0.01 rtpt weighted_rtpt $ADV_DATA_ROOT
bash train.sh $DATA_ROOT $GPU 4 RN50 1.0 7 0 1 0.1 20 0.01 rtpt weighted_rtpt $ADV_DATA_ROOT
bash train.sh $DATA_ROOT $GPU 4 RN50 4.0 7 1 0.1 20 0.01 rtpt weighted_rtpt $ADV_DATA_ROOT


# R-TPT Baseline - Weighted Ensemble RN50
bash train.sh $DATA_ROOT $GPU 4 RN50 0 0 1 0.1 20 0.01 rtpt vanilla $ADV_DATA_ROOT
bash train.sh $DATA_ROOT $GPU 4 RN50 1.0 7 0 1 0.1 20 0.01 rtpt vanilla $ADV_DATA_ROOT
bash train.sh $DATA_ROOT $GPU 4 RN50 4.0 7 1 0.1 20 0.01 rtpt vanilla $ADV_DATA_ROOT



# TPT + Weighted Inference
bash train.sh $DATA_ROOT $GPU 4 RN50 0 0 1 0.1 20 0.01 tpt weighted_rtpt $ADV_DATA_ROOT
bash train.sh $DATA_ROOT $GPU 4 RN50 1.0 7 1 0.1 20 0.01 tpt weighted_rtpt $ADV_DATA_ROOT
bash train.sh $DATA_ROOT $GPU 4 RN50 4.0 7 1 0.1 20 0.01 tpt weighted_rtpt $ADV_DATA_ROOT


# TPT + Vanilla Inference
bash train.sh $DATA_ROOT $GPU 4 RN50 0 0 1 0.1 20 0.01 tpt vanilla $ADV_DATA_ROOT
bash train.sh $DATA_ROOT $GPU 4 RN50 1.0 7 1 0.1 20 0.01 tpt vanilla $ADV_DATA_ROOT
bash train.sh $DATA_ROOT $GPU 4 RN50 4.0 7 1 0.1 20 0.01 tpt vanilla $ADV_DATA_ROOT



# TPT + No ensemble
bash train.sh $DATA_ROOT $GPU 4 RN50 0 0 1 0.1 20 0.01 tpt none $ADV_DATA_ROOT
bash train.sh $DATA_ROOT $GPU 4 RN50 1.0 7 1 0.1 20 0.01 tpt none $ADV_DATA_ROOT
bash train.sh $DATA_ROOT $GPU 4 RN50 4.0 7 1 0.1 20 0.01 tpt none $ADV_DATA_ROOT


# Record end time
END_TIME=$(date +%s)

# Calculate elapsed time
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

# Display elapsed time
echo "Script completed in ${HOURS}h ${MINUTES}m ${SECONDS}s."


