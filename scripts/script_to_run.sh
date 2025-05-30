#!/bin/bash
# Submit all the TODO batch JOB

LAUNCHER="bash scripts/train_cluster_fg.sh"
ATPT_DATASET="/leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data"
DOWNSTREAM_DATASET="$ATPT_DATASET/downstream_datasets"
MODEL="ViT-L/14"
OUTPUT_DIR="/leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"

sbatch run_1d.sh "$LAUNCHER $DOWNSTREAM_DATASET  0 4 $MODEL 4.0 100 0 0.1 20 0.01 rtpt none $ATPT_DATASET true pgd 2 4.0 1.0 0.2 2.0 true $OUTPUT_DIR"

sbatch run_1d.sh "$LAUNCHER $DOWNSTREAM_DATASET  0 4 $MODEL 4.0 100 0 0.1 20 0.01 rtpt vanilla $ATPT_DATASET true pgd 2 4.0 1.0 0.2 2.0 true $OUTPUT_DIR"

sbatch run_1d.sh "$LAUNCHER $DOWNSTREAM_DATASET  0 4 $MODEL 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt $ATPT_DATASET true pgd 2 4.0 1.0 0.2 2.0 true $OUTPUT_DIR"

sbatch run_1d.sh "$LAUNCHER $DOWNSTREAM_DATASET  0 4 $MODEL 0.0 0 0 0.1 20 0.01 rtpt none $ATPT_DATASET true pgd 2 4.0 1.0 0.2 2.0 true $OUTPUT_DIR"

sbatch run_1d.sh "$LAUNCHER $DOWNSTREAM_DATASET  0 4 $MODEL 0.0 0 0 0.1 20 0.01 rtpt vanilla $ATPT_DATASET true pgd 2 4.0 1.0 0.2 2.0 true $OUTPUT_DIR"

sbatch run_1d.sh "$LAUNCHER $DOWNSTREAM_DATASET  0 4 $MODEL 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt $ATPT_DATASET true pgd 2 4.0 1.0 0.2 2.0 true $OUTPUT_DIR"

sbatch run_1d.sh "$LAUNCHER $DOWNSTREAM_DATASET  0 4 $MODEL 4.0 100 1 0.1 20 0.01 rtpt none $ATPT_DATASET false pgd 0 4.0 1.0 0.2 2.0 true $OUTPUT_DIR"

sbatch run_1d.sh "$LAUNCHER $DOWNSTREAM_DATASET  0 4 $MODEL 4.0 100 1 0.1 20 0.01 rtpt vanilla $ATPT_DATASET false pgd 0 4.0 1.0 0.2 2.0 true $OUTPUT_DIR"

sbatch run_1d.sh "$LAUNCHER $DOWNSTREAM_DATASET  0 4 $MODEL 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt $ATPT_DATASET false pgd 0 4.0 1.0 0.2 2.0 true $OUTPUT_DIR"

sbatch run_1d.sh "$LAUNCHER $DOWNSTREAM_DATASET  0 4 $MODEL 0.0 0 1 0.1 20 0.01 rtpt none $ATPT_DATASET false pgd 0 4.0 1.0 0.2 2.0 true $OUTPUT_DIR"

sbatch run_1d.sh "$LAUNCHER $DOWNSTREAM_DATASET  0 4 $MODEL 0.0 0 1 0.1 20 0.01 rtpt vanilla $ATPT_DATASET false pgd 0 4.0 1.0 0.2 2.0 true $OUTPUT_DIR"

sbatch run_1d.sh "$LAUNCHER $DOWNSTREAM_DATASET  0 4 $MODEL 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt $ATPT_DATASET false pgd 0 4.0 1.0 0.2 2.0 true $OUTPUT_DIR"
