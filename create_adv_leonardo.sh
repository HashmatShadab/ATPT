

#bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 0 4 RN50 1.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
#bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 0 4 RN50 4.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
#
#bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 fare4 1.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
#bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-L-14 1.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
#
#
#bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 2 0 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
#bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 2 0 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
#bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 2 0 vit_l_14_datacomp_1b 1.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results

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

#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets GPU WORKERS vit_l_14_datacomp_1b EPS STEPS TTA_STEPS 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results PURIFY TYPE ANCHORS ALPHA SIGMA THRESH"
# Alpha 1.2 Sigma 0.18
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.18 0.0"

# Alpha 1.2 Sigma 0.06
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.06 0.0"

# Alpha 1.2 Sigma 0.12
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.12 0.0"

wait_for_jobs


# Alpha 0.2 Sigma 0.18
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 3 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 3 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 3 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 3 1.2 0.18 0.0"

# Alpha 1.2 Sigma 0.06
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 3 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 3 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 3 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 3 1.2 0.06 0.0"

# Alpha 1.2 Sigma 0.12
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 3 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 3 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 3 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 3 1.2 0.12 0.0"

wait_for_jobs

# Alpha 1.2 Sigma 0.18
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 5 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 5 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 5 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 5 1.2 0.18 0.0"

# Alpha 1.2 Sigma 0.06
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 5 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 5 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 5 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 5 1.2 0.06 0.0"

# Alpha 1.2 Sigma 0.12
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 5 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 5 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 5 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 5 1.2 0.12 0.0"

wait_for_jobs

# Alpha 1.2 Sigma 0.18
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 7 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 7 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 7 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 7 1.2 0.18 0.0"

# Alpha 1.2 Sigma 0.06
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 7 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 7 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 7 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 7 1.2 0.06 0.0"

# Alpha 1.2 Sigma 0.12
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 7 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 7 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 7 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 7 1.2 0.12 0.0"

wait_for_jobs


# Alpha 1.2 Sigma 0.18
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 9 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 9 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 9 1.2 0.18 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 9 1.2 0.18 0.0"

# Alpha 1.2 Sigma 0.06
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 9 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 9 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 9 1.2 0.06 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 9 1.2 0.06 0.0"

# Alpha 1.2 Sigma 0.12
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 9 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 9 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 9 1.2 0.12 0.0"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 9 1.2 0.12 0.0"

wait_for_jobs


## Alpha 1.2 Sigma 0.18
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.2 0.18 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.2 0.18 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.2 0.18 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.2 0.18 0.0"
#
## Alpha -1.2 Sigma 0.06
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.2 0.06 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.2 0.06 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.2 0.06 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.2 0.06 0.0"
#
## Alpha -1.2 Sigma 0.12
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.2 0.12 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.2 0.12 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.2 0.12 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.2 0.12 0.0"
#
#wait_for_jobs