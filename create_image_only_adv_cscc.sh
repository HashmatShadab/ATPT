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


sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true cars"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true dtd"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true caltech101"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true pets"
wait_for_jobs

sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true ucf101"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true eurosat"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true flower102"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true aircraft"
wait_for_jobs

sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true cars"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true dtd"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true caltech101"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true pets"
wait_for_jobs

sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true ucf101"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true eurosat"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true flower102"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true aircraft"
wait_for_jobs

sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 delta_clip_l14_224  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true cars"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 delta_clip_l14_224  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true dtd"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 delta_clip_l14_224  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true caltech101"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 delta_clip_l14_224  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true pets"
wait_for_jobs

sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 delta_clip_l14_224  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true ucf101"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 delta_clip_l14_224  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true eurosat"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 delta_clip_l14_224  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true flower102"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 delta_clip_l14_224  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true aircraft"
wait_for_jobs

sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 vit_l_14_datacomp_1b  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true cars"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 vit_l_14_datacomp_1b  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true dtd"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 vit_l_14_datacomp_1b  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true caltech101"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 vit_l_14_datacomp_1b  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true pets"
wait_for_jobs

sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 vit_l_14_datacomp_1b  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true ucf101"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 vit_l_14_datacomp_1b  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true eurosat"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 vit_l_14_datacomp_1b  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true flower102"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 vit_l_14_datacomp_1b  4.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results /l/users/hashmat.malik/atpt_results  true  noisy_anchor 1 1.2 0.18 0.0 true aircraft"