#!/bin/bash


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




# TPT + OTPT: Steps 1 ANchor lambda 18.0

sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  dtd"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  caltech101"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  cars"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  flower102"
wait_for_jobs

sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  aircraft"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  ucf101"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  eurosat"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  pets"
wait_for_jobs


# TPT + OTPT: Steps 1 ANchor lambda 36.0

sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  dtd 36.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  caltech101 36.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  cars 36.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  flower102 36.0"
wait_for_jobs

sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  aircraft 36.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  ucf101 36.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  eurosat 36.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  pets 36.0"
wait_for_jobs

# TPT + OTPT: Steps 1 ANchor lambda 54.0

sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  dtd 54.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  caltech101 54.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  cars 54.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  flower102 54.0"
wait_for_jobs

sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  aircraft 54.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  ucf101 54.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  eurosat 54.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  pets 54.0"
wait_for_jobs


# TPT + OTPT: Steps 1 ANchor lambda 72.0

sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  dtd 72.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  caltech101 72.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  cars 72.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  flower102 72.0"
wait_for_jobs

sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  aircraft 72.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  ucf101 72.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  eurosat 72.0"
sbatch run.sh "bash scripts/train_cluster.sh  /l/users/hashmat.malik/downstream_datasets  0 4 ViT-B/16 0.0 0 1 0.1 20 0.01 tpt_anchor all /l/users/hashmat.malik/atpt_results false pgd 0 4.0 1.0 0.2 2.0 true /l/users/hashmat.malik/Projects/ATPT/OTPT_Cscc_Results 1000  pets 72.0"
wait_for_jobs