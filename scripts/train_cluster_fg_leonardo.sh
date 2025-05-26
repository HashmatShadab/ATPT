#!/bin/bash


JOB_ID=${1:-1}
EVAL=${2:-"adv"}
MODEL_NAME=${3:-"$MODEL_NAME"}

if [ $JOB_ID -eq 1 ]; then
  # Zero shot Performance
  echo "Running $MODEL_NAME ZS tests Fine-grained datasets..."

  if [ "$EVAL" == "adv" ]; then
    echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
    bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
    bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
    bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
  else
    echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
    bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
    bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
    bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  fi

elif [ "$JOB_ID" -eq 2 ]; then
  #Zero-shot Performance + Gaussian Noise
  echo "Running $MODEL_NAME ZS tests Fine-grained datasets with Gaussian Noise Eps 2/255"

  if [ "$EVAL" == "adv" ]; then
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
        bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
        bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
        bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  else:
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
        bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
        bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
        bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  fi


elif [ "$JOB_ID" -eq 3 ]; then
  #Zero-shot Performance + Gaussian Noise
  echo "Running $MODEL_NAME ZS tests Fine-grained datasets with Gaussian Noise Eps 4/255"

  if [ "$EVAL" == "adv" ]; then
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  else:
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  fi


elif [ "$JOB_ID" -eq 4 ]; then
  #Zero-shot Performance + Counter Attack Eps 2/255
  echo "Running $MODEL_NAME ZS tests Fine-grained datasets with Counter Attack  Eps 2/255"

  if [ "$EVAL" == "adv" ]; then
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  else:
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  fi



elif [ "$JOB_ID" -eq 5 ]; then
  #Zero-shot Performance + Counter Attack Eps 4/255
  echo "Running $MODEL_NAME ZS tests Fine-grained datasets with Counter Attack  Eps 4/255"

  if [ "$EVAL" == "adv" ]; then
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  else:
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  fi


elif [ "$JOB_ID" -eq 6 ]; then
  #RTPT Performance
  echo "Running $MODEL_NAME RTPT tests Fine-grained datasets..."

  if [ "$EVAL" == "adv" ]; then
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  else:
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  fi


elif [ "$JOB_ID" -eq 7 ]; then
  #RTPT Performance + Gaussian Noise Eps 2/255
  echo "Running $MODEL_NAME RTPT tests Fine-grained datasets with Gaussian Noise Eps 2/255"

  if [ "$EVAL" == "adv" ]; then
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  else:
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  fi


elif [ "$JOB_ID" -eq 8 ]; then
  #RTPT Performance + Gaussian Noise Eps 4/255
  echo "Running $MODEL_NAME RTPT tests Fine-grained datasets with Gaussian Noise Eps 4/255"

  if [ "$EVAL" == "adv" ]; then
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  else:
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  fi


elif [ "$JOB_ID" -eq 9 ]; then
  #RTPT Performance + Counter Attack Eps 2/255
  echo "Running $MODEL_NAME RTPT tests Fine-grained datasets with Counter Attack Eps 2/255"

  if [ "$EVAL" == "adv" ]; then
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  else:
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  fi



elif [ "$JOB_ID" -eq 10 ]; then
  #RTPT Performance + Counter Attack Eps 4/255
  echo "Running $MODEL_NAME RTPT tests Fine-grained datasets with Counter Attack Eps 4/255"

  if [ "$EVAL" == "adv" ]; then
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  else:
      echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results
      bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results

  fi


else
  echo "Invalid JOB_ID. Please use 1, 2, or 3."
  exit 1
fi
