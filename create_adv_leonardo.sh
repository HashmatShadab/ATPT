

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
    fixed_job_name="adv_rob_clip"

    echo "[INFO] Waiting for jobs with name '$fixed_job_name' to complete..."

    while true; do
        job_count=$(squeue -u "$USER" -h -o "%j" | grep -w "$fixed_job_name" | wc -l)

        if [ "$job_count" -eq 0 ]; then
            echo "[INFO] All jobs with name '$fixed_job_name' have completed."
            break
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $job_count job(s) with name '$fixed_job_name' still running. Checking again in 30 minutes..."
        sleep 30m
    done
}

#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets GPU WORKERS vit_l_14_datacomp_1b EPS STEPS TTA_STEPS 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results PURIFY TYPE ANCHORS ALPHA SIGMA THRESH"



#
## Alpha 0.2 Sigma 0.18
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.2 0.18 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.2 0.18 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.18 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.18 0.0"
#
## Alpha 0.2 Sigma 0.06
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.2 0.06 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.2 0.06 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.06 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.06 0.0"
#
## Alpha 0.2 Sigma 0.12
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.2 0.12 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.2 0.12 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.12 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.12 0.0"
#
#wait_for_jobs
#
#
## Alpha 0.4 Sigma 0.18
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.4 0.18 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.4 0.18 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.18 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.18 0.0"
#
## Alpha 0.4 Sigma 0.06
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.4 0.06 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.4 0.06 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.06 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.06 0.0"
#
## Alpha 0.4 Sigma 0.12
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.4 0.12 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.4 0.12 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.12 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.12 0.0"
#
#wait_for_jobs
#
#
## Alpha 0.8 Sigma 0.18
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.8 0.18 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.8 0.18 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.18 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.18 0.0"
#
## Alpha 0.8 Sigma 0.06
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.8 0.06 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.8 0.06 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.06 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.06 0.0"
#
## Alpha 0.8 Sigma 0.12
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.8 0.12 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 0.8 0.12 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.12 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.12 0.0"
#
#wait_for_jobs
#
#
## Alpha 1.0 Sigma 0.18
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.0 0.18 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.0 0.18 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.18 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.18 0.0"
#
## Alpha 1.0 Sigma 0.06
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.0 0.06 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.0 0.06 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.06 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.06 0.0"
#
## Alpha 1.0 Sigma 0.12
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.0 0.12 0.0"
#sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 10 1.0 0.12 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.12 0.0"
##sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Purify_Results true noisy_anchor 1 1.2 0.12 0.0"
#
#wait_for_jobs

sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm dtd"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm caltech101"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm cars"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm flower102"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm aircraft"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm ucf101"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm eurosat"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 vit_l_14_datacomp_1b 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm pets"
wait_for_jobs

sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm dtd"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm caltech101"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm cars"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm flower102"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm aircraft"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm ucf101"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm eurosat"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm pets"
wait_for_jobs

sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm dtd"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm caltech101"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm cars"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm flower102"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm aircraft"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm ucf101"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm eurosat"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 fare4 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm pets"
wait_for_jobs

sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm dtd"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm caltech101"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm cars"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm flower102"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm aircraft"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm ucf101"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm eurosat"
sbatch run_1d.sh "bash gen_adv.sh /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data/downstream_datasets 0 4 delta_clip_l14_224 4.0 100 1 0.1 20 0.01 /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/atpt_data /leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/ATPT/Final_Results false noisy_anchor 10 1.2 0.18 0.0 true prm pets"
wait_for_jobs
