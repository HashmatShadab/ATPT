
# R-TPT Baseline RN50
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 0 0 1 0.1 20 0.01 rtpt weighted_rtpt "F:\Code\datasets\atpt_results"
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 1.0 7 0 1 0.1 20 0.01 rtpt weighted_rtpt "F:\Code\datasets\atpt_results"
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 4.0 7 1 0.1 20 0.01 rtpt weighted_rtpt "F:\Code\datasets\atpt_results"


# R-TPT Baseline - Weighted Ensemble RN50
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 0 0 1 0.1 20 0.01 rtpt vanilla "F:\Code\datasets\atpt_results"
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 1.0 7 0 1 0.1 20 0.01 rtpt vanilla "F:\Code\datasets\atpt_results"
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 4.0 7 1 0.1 20 0.01 rtpt vanilla "F:\Code\datasets\atpt_results"





# TPT + Weighted Inference
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 0 0 1 0.1 20 0.01 tpt weighted_rtpt "F:\Code\datasets\atpt_results"
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 1.0 7 1 0.1 20 0.01 tpt weighted_rtpt "F:\Code\datasets\atpt_results"
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 4.0 7 1 0.1 20 0.01 tpt weighted_rtpt "F:\Code\datasets\atpt_results"


# TPT + Vanilla Inference
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 0 0 1 0.1 20 0.01 tpt vanilla "F:\Code\datasets\atpt_results"
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 1.0 7 1 0.1 20 0.01 tpt vanilla "F:\Code\datasets\atpt_results"
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 4.0 7 1 0.1 20 0.01 tpt vanilla "F:\Code\datasets\atpt_results"



# TPT + No ensemble
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 0 0 1 0.1 20 0.01 tpt none "F:\Code\datasets\atpt_results"
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 1.0 7 1 0.1 20 0.01 tpt none "F:\Code\datasets\atpt_results"
bash train.sh "F:\Code\datasets\downstream_datasets\downstream_datasets" 0 4 RN50 4.0 7 1 0.1 20 0.01 tpt none "F:\Code\datasets\atpt_results"




