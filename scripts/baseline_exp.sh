#!/bin/bash

MODEL_NAME=${1:-"fare4"} # Options: fare2, fare4, tecoa2, tecoa4, ViT-B/16, ViT-L/14, RN50

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



# Zero shot Performance
echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: None"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: Vanilla"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: Weighted RTPT"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


echo "Running Clean Zero-shot experiments Ensembling: None"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean Zero-shot experiments Ensembling: Vanilla"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean Zero-shot experiments Ensembling: Weighted RTPT"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


#Zero-shot Performance + Gaussian Noise


#Eps 2/255
echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: None with Gaussian Noise Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: Vanilla with Gaussian Noise Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: Weighted RTPT with Gaussian Noise Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running  Clean Zero-shot experiments Ensembling: None with Gaussian Noise Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running  Clean Zero-shot experiments Ensembling: Vanilla with Gaussian Noise Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running  Clean Zero-shot experiments Ensembling: Weighted RTPT with Gaussian Noise Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: None with Gaussian Noise Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: Vanilla with Gaussian Noise Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: Weighted RTPT with Gaussian Noise Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


echo "Running  Clean Zero-shot experiments Ensembling: None with Gaussian Noise Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running  Clean Zero-shot experiments Ensembling: Vanilla with Gaussian Noise Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running  Clean Zero-shot experiments Ensembling: Weighted RTPT with Gaussian Noise Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


#Zero-shot Performance + CounterAttacks

#Eps 2/255
echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: None with CounterAttacks Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: Vanilla with CounterAttacks Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: Weighted RTPT with CounterAttacks Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running  Clean Zero-shot experiments Ensembling: None with CounterAttacks Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running  Clean Zero-shot experiments Ensembling: Vanilla with CounterAttacks Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running  Clean Zero-shot experiments Ensembling: Weighted RTPT with CounterAttacks Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

#Eps 4/255

echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: None with CounterAttacks Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: Vanilla with CounterAttacks Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) Zero-shot experiments Ensembling: Weighted RTPT with CounterAttacks Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running  Clean Zero-shot experiments Ensembling: None with CounterAttacks Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running  Clean Zero-shot experiments Ensembling: Vanilla with CounterAttacks Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running  Clean Zero-shot experiments Ensembling: Weighted RTPT with CounterAttacks Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


#RTPT
echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: None"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: Vanilla"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: Weighted RTPT"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: None"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: Vanilla"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: Weighted RTPT"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs




#RTPT with Gaussian Noise

#Eps 2/255
echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: None with Gaussian Noise Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: Vanilla with Gaussian Noise Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: Weighted RTPT with Gaussian Noise Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: None with Gaussian Noise Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: Vanilla with Gaussian Noise Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: Weighted RTPT with Gaussian Noise Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


#Eps 4/255
echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: None with Gaussian Noise Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: Vanilla with Gaussian Noise Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: Weighted RTPT with Gaussian Noise Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: None with Gaussian Noise Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: Vanilla with Gaussian Noise Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: Weighted RTPT with Gaussian Noise Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

#RTPT with counter attacks

#Eps 2/255
echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: None with Counter Attacks Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: Vanilla with Counter Attacks Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: Weighted RTPT with Counter Attacks Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: None with Counter Attacks Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: Vanilla with Counter Attacks Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: Weighted RTPT with Counter Attacks Eps 2/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

#Eps 4/255

echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: None with Counter Attacks Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs


echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: Vanilla with Counter Attacks Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Adv (Eps 4.0 Steps 100) RTPT experiments Ensembling: Weighted RTPT with Counter Attacks Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 4.0 100 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: None with Counter Attacks Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: Vanilla with Counter Attacks Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"
# Wait for all jobs to complete before proceeding
wait_for_jobs

echo "Running Clean RTPT experiments Ensembling: Weighted RTPT with Counter Attacks Eps 4/255"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 1"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 2"
sbatch run.sh "bash scripts/train_cluster.sh /l/users/hashmat.malik/downstream_datasets  0 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/Final_Results 3"