
#NOISE_TYPE=("gaussian_noise" "uniform_noise" "brightness_dark" "brightness_bright" "contrast_low" "contrast_high" "saturation_low" "saturation_high" "sharpness_low" "sharpness_high" "gamma_bright" "gamma_dark" "hue_negative" "hue_positive" "gaussian_blur" "rotation" "translation" "posterize" "solarize" "downsample" "jpeg")

NOISE_TYPE=("brightness_dark" "brightness_bright" "contrast_low" "contrast_high" "saturation_low" "saturation_high" "sharpness_low" "sharpness_high" "gamma_bright" "gamma_dark" "hue_negative" "hue_positive" "gaussian_blur" "rotation" "translation" "posterize" "solarize" "downsample" "jpeg")

#NOISE_TYPE=("brightness_dark")

for noise in "${NOISE_TYPE[@]}"; do
  echo "Running with noise type: $noise"

  python3 rtpt_adv_generation.py /home/malik/projects/datasets/downstream_datasets \
  --test_sets Caltech101 -a vit_l_14_datacomp_1b -b 64 --adv_bs 64 --gpu 0 --ctx_init a_photo_of_a \
  --output_dir /home/malik/projects/datasets/downstream_datasets/atpt_adversrial_datasets \
  --log_output_dir ./transformation_ablation \
  --workers 0 --eps 4.0 --steps 100 --tta_steps 1 --image_only_attack false --image_only_attack_type prm \
  --counter_attack true --counter_attack_steps 0 --counter_attack_init_noise $noise --counter_attack_eps 24.0 \
  --counter_attack_gaussian_sigma 0.03 --counter_attack_tau normal --counter_attack_noisy_tau_num_anchors 1 --selection_p 0.1 \
  --top_k 20 --softmax_temp 0.01 --print-freq 20 --image_feature_purify false --image_feature_purify_type noisy_anchor --image_feature_purify_noisy_anchors 10 \
  --image_feature_purify_anchors_alpha 1.2 --image_feature_purify_noisy_sigma 0.18 --image_feature_purify_diff_threshold 0.0


  python3 rtpt_adv_generation.py /home/malik/projects/datasets/downstream_datasets \
  --test_sets Caltech101 -a vit_l_14_datacomp_1b -b 64 --adv_bs 64 --gpu 0 --ctx_init a_photo_of_a \
  --output_dir /home/malik/projects/datasets/downstream_datasets/atpt_adversrial_datasets \
  --log_output_dir ./transformation_ablation \
  --workers 0 --eps 0.0 --steps 0 --tta_steps 1 --image_only_attack false --image_only_attack_type prm \
  --counter_attack true --counter_attack_steps 0 --counter_attack_init_noise $noise --counter_attack_eps 24.0 \
  --counter_attack_gaussian_sigma 0.03 --counter_attack_tau normal --counter_attack_noisy_tau_num_anchors 1 --selection_p 0.1 \
  --top_k 20 --softmax_temp 0.01 --print-freq 20 --image_feature_purify false --image_feature_purify_type noisy_anchor --image_feature_purify_noisy_anchors 10 \
  --image_feature_purify_anchors_alpha 1.2 --image_feature_purify_noisy_sigma 0.18 --image_feature_purify_diff_threshold 0.0

done

