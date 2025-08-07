#!/bin/bash

MODEL_NAME=$1
GPU=$2

#bash get_mean_std.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets $GPU 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results /mnt/nvme0n1/Dataset/muzammal/atpt_results dtd
#bash get_mean_std.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets $GPU 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results /mnt/nvme0n1/Dataset/muzammal/atpt_results cars
#bash get_mean_std.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets $GPU 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results /mnt/nvme0n1/Dataset/muzammal/atpt_results flower102
#bash get_mean_std.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets $GPU 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results /mnt/nvme0n1/Dataset/muzammal/atpt_results aircraft
#bash get_mean_std.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets $GPU 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results /mnt/nvme0n1/Dataset/muzammal/atpt_results pets

bash get_mean_std.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets $GPU 4 $MODEL_NAME 0.0 0 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results /mnt/nvme0n1/Dataset/muzammal/atpt_results I
