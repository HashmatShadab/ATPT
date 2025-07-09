

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

wait_for_jobs
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 RN50 1.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results true"
sbatch run.sh "bash gen_adv.sh /l/users/hashmat.malik/downstream_datasets 0 4 fare4 1.0 100 1 0.1 20 0.01 /l/users/hashmat.malik/atpt_results true"
