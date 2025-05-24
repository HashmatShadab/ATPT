#!/bin/bash


bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-B/16 4.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-B/16 1.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-L/14 4.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
bash gen_adv.sh /mnt/nvme0n1/Dataset/muzammal/downstream_datasets 3 4 ViT-L/14 1.0 100 1 0.1 20 0.01 /mnt/nvme0n1/Dataset/muzammal/atpt_results
