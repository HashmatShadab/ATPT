#!/bin/bash



bash scripts/train_cluster_fg.sh /l/users/hashmat.malik/downstream_datasets  0 4 RN50 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true 1
bash scripts/train_cluster_fg.sh /l/users/hashmat.malik/downstream_datasets  0 4 RN50 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true 1
bash scripts/train_cluster_fg.sh /l/users/hashmat.malik/downstream_datasets  0 4 RN50 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true 1
bash scripts/train_cluster_fg.sh /l/users/hashmat.malik/downstream_datasets  0 4 RN50 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 2.0 1.0 0.2 2.0 true 1
bash scripts/train_cluster_fg.sh /l/users/hashmat.malik/downstream_datasets  0 4 RN50 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 0 4.0 1.0 0.2 2.0 true 1
bash scripts/train_cluster_fg.sh /l/users/hashmat.malik/downstream_datasets  0 4 RN50 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true 1
bash scripts/train_cluster_fg.sh /l/users/hashmat.malik/downstream_datasets  0 4 RN50 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true 1

#
#bash scripts/train_cluster_fg.sh /l/users/hashmat.malik/downstream_datasets  0 4 RN50 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 4.0 1.0 0.2 2.0 true 1
#bash scripts/train_cluster_fg.sh /l/users/hashmat.malik/downstream_datasets  0 4 RN50 0.0 0 0 0.1 20 0.01 rtpt none /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true 1
#bash scripts/train_cluster_fg.sh /l/users/hashmat.malik/downstream_datasets  0 4 RN50 0.0 0 0 0.1 20 0.01 rtpt vanilla /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true 1
#bash scripts/train_cluster_fg.sh  /l/users/hashmat.malik/downstream_datasets  0 4 RN50 0.0 0 0 0.1 20 0.01 rtpt weighted_rtpt /l/users/hashmat.malik/atpt_results true pgd 2 2.0 1.0 0.2 2.0 true 1