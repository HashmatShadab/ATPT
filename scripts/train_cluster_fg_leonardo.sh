#!/bin/bash


MODEL_NAME=${1:-"ViT-L/14"}

# Function to wait for all SLURM jobs to complete
wait_for_jobs() {
    echo "Waiting for all jobs to complete..."
    while true; do
        # Check if there are any jobs running for the current user
        job_count=$(squeue -u $USER -h | wc -l)
        if [ "$job_count" -eq 0 ]; then
            echo "All jobs completed."
            break
        fi
        echo "$job_count jobs still running. Checking again in 5 minutes..."
        sleep 5m
    done
}


## Zero shot Performance
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets..."
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#
#
##Zero-shot Performance + Gaussian Noise
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with Gaussian Noise Eps 2/255"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
## Wait for all jobs to complete before proceeding
#wait_for_jobs
#
##Zero-shot Performance + Gaussian Noise
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with Gaussian Noise Eps 4/255"
#
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#
##Zero-shot Performance + Counter Attack Eps 2/255
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with Counter Attack  Eps 2/255"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
## Wait for all jobs to complete before proceeding
#wait_for_jobs
#
##Zero-shot Performance + Counter Attack Eps 4/255
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with Counter Attack  Eps 4/255"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#
#
##RTPT Performance
#echo "Running $MODEL_NAME RTPT tests Fine-grained datasets..."
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
## Wait for all jobs to complete before proceeding
#wait_for_jobs
#
#
##RTPT Performance + Gaussian Noise Eps 2/255
#echo "Running $MODEL_NAME RTPT tests Fine-grained datasets with Gaussian Noise Eps 2/255"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#
#
#
##RTPT Performance + Gaussian Noise Eps 4/255
#echo "Running $MODEL_NAME RTPT tests Fine-grained datasets with Gaussian Noise Eps 4/255"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
## Wait for all jobs to complete before proceeding
#wait_for_jobs
#
##RTPT Performance + Counter Attack Eps 2/255
#echo "Running $MODEL_NAME RTPT tests Fine-grained datasets with Counter Attack Eps 2/255"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 2.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#
#
#
##RTPT Performance + Counter Attack Eps 4/255
#echo "Running $MODEL_NAME RTPT tests Fine-grained datasets with Counter Attack Eps 4/255"
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
#echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data true pgd 2 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
#
## Wait for all jobs to complete before proceeding
#wait_for_jobs

#TPT Performance
echo "Running $MODEL_NAME TPT tests Fine-grained datasets..."

echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 tpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 tpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 tpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"

echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 ViT-L/14 0.0 0 1 0.1 20 0.01 tpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 ViT-L/14 0.0 0 1 0.1 20 0.01 tpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 ViT-L/14 0.0 0 1 0.1 20 0.01 tpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"


echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 tpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 tpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 tpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"

echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 delta_clip_l14_224 0.0 0 1 0.1 20 0.01 tpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 delta_clip_l14_224 0.0 0 1 0.1 20 0.01 tpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 delta_clip_l14_224 0.0 0 1 0.1 20 0.01 tpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"

# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 fare4 4.0 100 1 0.1 20 0.01 tpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 fare4 4.0 100 1 0.1 20 0.01 tpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 fare4 4.0 100 1 0.1 20 0.01 tpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"

echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 fare4 0.0 0 1 0.1 20 0.01 tpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 fare4 0.0 0 1 0.1 20 0.01 tpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 fare4 0.0 0 1 0.1 20 0.01 tpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"

echo "Running $MODEL_NAME ZS tests Fine-grained datasets with adversarial evaluation"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 tpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 tpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 tpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"

echo "Running $MODEL_NAME ZS tests Fine-grained datasets with standard evaluation"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 vit_l_14_datacomp_1b 0.0 0 1 0.1 20 0.01 tpt none /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 vit_l_14_datacomp_1b 0.0 0 1 0.1 20 0.01 tpt vanilla /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
sbatch run_1d.sh "bash scripts/train_cluster_fg.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets  0 4 vit_l_14_datacomp_1b 0.0 0 1 0.1 20 0.01 tpt weighted_rtpt /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data false pgd 0 4.0 1.0 0.2 2.0 true /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results"
