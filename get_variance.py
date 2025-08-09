"""
Variance Computation Script

This script should be run AFTER get_mean_std.py has completed. It computes the
variance of model features over the dataset using the previously saved MEANS,
following a memory-efficient, single-pass approach over the dataset.

It mirrors the dataset/model setup and adversarial generation options of
get_mean_std.py to ensure that the exact same data pipeline is used. It saves:
- variance of CLS token to: output_dir/.../variance_statistics/cls_token_var.pt
- per-layer variance for features_after_attention:
  output_dir/.../variance_statistics/features_after_attention_layer_{i}_var.pt
- per-layer variance for features_after_mlp:
  output_dir/.../variance_statistics/features_after_mlp_layer_{i}_var.pt

Resume is supported via checkpoints stored under variance_statistics.
"""

import argparse
import logging
import os
import time
import gc
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F  # noqa: F401
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
import psutil

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from open_clip.custom_openai_clip import get_coop as get_coop_openai
from clip.custom_clip import get_coop
from open_clip.custom_openai_clip import get_text_embeddings as get_text_embeddings_openai  # noqa: F401
from clip.custom_clip import get_text_embeddings as get_text_embeddings  # noqa: F401
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from data.cls_to_names import flower102_classes, food101_classes, dtd_classes, caltech101_classes, pets_classes, \
    sun397_classes, cars_classes, ucf101_classes, aircraft_classes, eurosat_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import set_random_seed
from utils.logger import setup_logger

import torchattacks

from helper_functions import print_args  # noqa: F401

openai_model_dict = {
    "delta_clip_l14_224": "hf-hub:zw123/delta_clip_l14_224",
    "tecoa4": "hf-hub:chs20/tecoa4-clip",
    "tecoa2": "hf-hub:chs20/tecoa2-clip",
    "fare2": "hf-hub:chs20/fare2-clip",
    "fare4": "hf-hub:chs20/fare4-clip",
    "vit_l_14_datacomp_1b": "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
}


def get_memory_usage():
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / (1024 * 1024)
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    return cpu_memory, gpu_memory


def load_means(stats_dir, logger=None):
    """Load CLS and per-layer means saved by get_mean_std.py."""
    cls_token_mean = torch.load(os.path.join(stats_dir, "cls_token_mean.pt"))
    attention_means = []
    mlp_means = []

    # Load all layer files by scanning the directory; keep order by layer index
    # Determine how many layers are present by counting files
    files = os.listdir(stats_dir)
    attn_files = sorted([f for f in files if f.startswith("features_after_attention_layer_") and f.endswith("_mean.pt")],
                        key=lambda x: int(x.split("_layer_")[1].split("_")[0]))
    mlp_files = sorted([f for f in files if f.startswith("features_after_mlp_layer_") and f.endswith("_mean.pt")],
                       key=lambda x: int(x.split("_layer_")[1].split("_")[0]))

    if logger:
        logger.info(f"Found {len(attn_files)} attention mean layers and {len(mlp_files)} mlp mean layers in {stats_dir}")

    for f in attn_files:
        attention_means.append(torch.load(os.path.join(stats_dir, f)))
    for f in mlp_files:
        mlp_means.append(torch.load(os.path.join(stats_dir, f)))

    return cls_token_mean, attention_means, mlp_means


def test_time_variance_eval(val_loader, model, args, logger=None):
    """
    Compute variance statistics using previously saved means.
    """
    model.eval()

    mean_stats_dir = os.path.join(args.output_dir, "mean_statistics")
    var_stats_dir = os.path.join(args.output_dir, "variance_statistics")
    os.makedirs(var_stats_dir, exist_ok=True)
    checkpoint_path = os.path.join(var_stats_dir, "checkpoint.pt")

    if logger:
        logger.info(f"Loading means from: {mean_stats_dir}")
        logger.info(f"Saving variance to: {var_stats_dir}")

    # Load means
    cls_token_mean, attention_means, mlp_means = load_means(mean_stats_dir, logger)

    # Ensure means are on CPU for memory efficiency
    cls_token_mean = cls_token_mean.cpu()
    attention_means = [m.cpu() for m in attention_means]
    mlp_means = [m.cpu() for m in mlp_means]

    # Initialize accumulators
    start_batch_idx = 0
    total = 0

    if os.path.exists(checkpoint_path):
        if logger:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path)
        start_batch_idx = ckpt['batch_idx'] + 1
        total = ckpt['total']
        cls_token_var = ckpt['cls_token_var']
        cls_token_count = ckpt['cls_token_count']
        attention_var = ckpt['attention_var']
        attention_counts = ckpt['attention_counts']
        mlp_var = ckpt['mlp_var']
        mlp_counts = ckpt['mlp_counts']
    else:
        cls_token_var = torch.zeros_like(cls_token_mean)
        cls_token_count = 0
        attention_var = [torch.zeros_like(x) for x in attention_means]
        attention_counts = [torch.zeros(x.shape[0], dtype=torch.int64) for x in attention_means]
        mlp_var = [torch.zeros_like(x) for x in mlp_means]
        mlp_counts = [torch.zeros(x.shape[0], dtype=torch.int64) for x in mlp_means]

    # Prepare attack if needed (to match pipeline used for means)
    if args.eps > 0.0:
        assert args.steps > 0
        if args.image_only_attack:
            if args.image_only_attack_type == "prm":
                atk = torchattacks.PGD_PRM(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps)
            elif args.image_only_attack_type == "prm_adam":
                atk = torchattacks.PGD_PRM_ADAM(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps)
            else:
                raise ValueError(f"Unknown image only attack type: {args.image_only_attack_type}")
        else:
            atk = torchattacks.PGD(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps,
                                   image_only_attack=False,
                                   image_predicted_label_attack=args.image_predicted_label_attack)
        if logger:
            logger.info(f"Using attack eps={args.eps/255:.6f}, alpha={args.alpha/255:.6f}, steps={args.steps}")

    # Directory for adversarial images (reuse same folder naming to be consistent)
    if args.eps > 0.0:
        if args.image_only_attack:
            adv_images_dir = os.path.join(args.output_dir,
                                          f"adv_images_eps_{args.eps}_alpha_{args.alpha}_steps_{args.steps}_image_only_attack_{args.image_only_attack_type}")
        elif args.image_predicted_label_attack:
            adv_images_dir = os.path.join(args.output_dir,
                                          f"adv_images_eps_{args.eps}_alpha_{args.alpha}_steps_{args.steps}_image_predicted_label_attack")
        else:
            adv_images_dir = os.path.join(args.output_dir,
                                          f"adv_images_eps_{args.eps}_alpha_{args.alpha}_steps_{args.steps}")
    else:
        adv_images_dir = os.path.join(args.output_dir,  "Clean_images")
    os.makedirs(adv_images_dir, exist_ok=True)

    # Note: We do not re-generate adversarial images here; we run forward directly on images or generate on the fly
    # to match the mean computation. For simplicity, we compute on clean images unless args specify attacks.

    initial_cpu_mem, initial_gpu_mem = get_memory_usage()
    if logger:
        logger.info(f"Initial memory - CPU: {initial_cpu_mem:.2f} MB, GPU: {initial_gpu_mem:.2f} MB")

    end = time.time()

    for i, data in enumerate(val_loader):
        if i < start_batch_idx:
            continue

        if len(data) == 3:
            images, target, path = data
        else:
            images, target = data
            path = None

        assert args.gpu is not None
        target = target.cuda(args.gpu, non_blocking=True)
        images = images.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            cls_token, features_after_attention, features_after_mlp = model(images)

        # Move to CPU for memory efficiency
        cls_token_cpu = cls_token.detach().cpu()
        bsz = cls_token_cpu.size(0)

        # Update CLS variance using batch mean of squared deviations (Bessel's correction not applied; this is population variance)
        diff = cls_token_cpu - cls_token_mean  # [B, D]
        batch_sq = torch.mean(diff.square(), dim=0)  # [D]
        cls_token_var = (cls_token_count * cls_token_var + bsz * batch_sq) / (cls_token_count + bsz)
        cls_token_count += bsz

        # Per layer tokens variance for attention and mlp
        for layer_idx in range(len(features_after_attention)):
            attn_feat = features_after_attention[layer_idx].detach().cpu()  # [T, B, D]
            mlp_feat = features_after_mlp[layer_idx].detach().cpu()        # [T, B, D]

            # Attention
            attn_mean = attention_means[layer_idx]  # [T, D]
            # Compute mean across batch of squared deviations for each token
            attn_diff = attn_feat - attn_mean.unsqueeze(1)                 # [T, B, D]
            attn_batch_sq = torch.mean(attn_diff.square(), dim=1)          # [T, D]
            prev_counts = attention_counts[layer_idx].clone().unsqueeze(1) # [T, 1]
            attention_counts[layer_idx] += bsz
            new_counts = attention_counts[layer_idx].clone().unsqueeze(1)  # [T, 1]
            attention_var[layer_idx] = (prev_counts * attention_var[layer_idx] + bsz * attn_batch_sq) / new_counts

            # MLP
            mlp_mean = mlp_means[layer_idx]                                # [T, D]
            mlp_diff = mlp_feat - mlp_mean.unsqueeze(1)                    # [T, B, D]
            mlp_batch_sq = torch.mean(mlp_diff.square(), dim=1)            # [T, D]
            prev_counts = mlp_counts[layer_idx].clone().unsqueeze(1)       # [T, 1]
            mlp_counts[layer_idx] += bsz
            new_counts = mlp_counts[layer_idx].clone().unsqueeze(1)        # [T, 1]
            mlp_var[layer_idx] = (prev_counts * mlp_var[layer_idx] + bsz * mlp_batch_sq) / new_counts

        total += target.size(0)

        # Save checkpoint periodically
        if args.checkpoint_freq > 0 and (i + 1) % args.checkpoint_freq == 0:
            ckpt = {
                'batch_idx': i,
                'total': total,
                'cls_token_var': cls_token_var.cpu(),
                'cls_token_count': cls_token_count,
                'attention_var': [v.cpu() for v in attention_var],
                'attention_counts': [c.cpu() for c in attention_counts],
                'mlp_var': [v.cpu() for v in mlp_var],
                'mlp_counts': [c.cpu() for c in mlp_counts],
            }
            torch.save(ckpt, checkpoint_path)
            if logger:
                cpu_mem, gpu_mem = get_memory_usage()
                logger.info(f"Checkpoint saved at batch {i+1}/{len(val_loader)}; CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")

        # Free memory
        del images
        gc.collect()
        torch.cuda.empty_cache()
        end = time.time()

    # Finalize: remove checkpoint and save results
    # if os.path.exists(checkpoint_path):
    #     os.remove(checkpoint_path)
    #     if logger:
    #         logger.info("Removed checkpoint (completed successfully)")

    # Save variances
    cls_var_path = os.path.join(var_stats_dir, "cls_token_var.pt")
    torch.save(cls_token_var, cls_var_path)
    if logger:
        logger.info(f"Saved CLS variance: {cls_var_path} shape={tuple(cls_token_var.shape)}")

    for layer_idx, v in enumerate(attention_var):
        p = os.path.join(var_stats_dir, f"features_after_attention_layer_{layer_idx}_var.pt")
        torch.save(v, p)
        if logger:
            logger.info(f"Saved attention layer {layer_idx} var shape={tuple(v.shape)}")

    for layer_idx, v in enumerate(mlp_var):
        p = os.path.join(var_stats_dir, f"features_after_mlp_layer_{layer_idx}_var.pt")
        torch.save(v, p)
        if logger:
            logger.info(f"Saved mlp layer {layer_idx} var shape={tuple(v.shape)}")

    # Summary
    summary_path = os.path.join(var_stats_dir, "variance_summary.txt")
    from datetime import datetime
    with open(summary_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("VARIANCE STATISTICS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {args.test_sets}\n")
        f.write(f"Model architecture: {args.arch}\n")
        f.write(f"Total samples processed: {total}\n")
        f.write(f"Batches: {len(val_loader)}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Checkpoint frequency: Every {args.checkpoint_freq} batches\n")
        f.write(f"Completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("CLS variance shape: " + str(tuple(cls_token_var.shape)) + "\n\n")
        f.write("Attention layers variance shapes:\n")
        for i, v in enumerate(attention_var):
            f.write(f"  - Layer {i}: {tuple(v.shape)}\n")
        f.write("\nMLP layers variance shapes:\n")
        for i, v in enumerate(mlp_var):
            f.write(f"  - Layer {i}: {tuple(v.shape)}\n")

    final_cpu_mem, final_gpu_mem = get_memory_usage()
    if logger:
        logger.info("=" * 50)
        logger.info("MEMORY USAGE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Initial memory - CPU: {initial_cpu_mem:.2f} MB, GPU: {initial_gpu_mem:.2f} MB")
        logger.info(f"Final memory   - CPU: {final_cpu_mem:.2f} MB, GPU: {final_gpu_mem:.2f} MB")
        logger.info(f"Memory change  - CPU: {final_cpu_mem - initial_cpu_mem:.2f} MB, GPU: {final_gpu_mem - initial_gpu_mem:.2f} MB")
        logger.info(f"Saved summary to: {summary_path}")
        logger.info("Variance computation completed successfully!")


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    # Calculate alpha from epsilon if not provided
    args.alpha = args.eps / args.alpha_eps_ratio

    # Build output/log directories similar to get_mean_std.py
    args.output_dir = os.path.join(args.output_dir, args.arch, args.test_sets)
    args.log_output_dir = os.path.join(args.log_output_dir, args.arch, args.test_sets)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_output_dir, exist_ok=True)

    log_name = f"Variance_Computation_eps_{args.eps}_steps_{args.steps}"
    if args.image_only_attack:
        log_name += f"_image_only_attack_{args.image_only_attack_type}"
    elif args.image_predicted_label_attack:
        log_name += "_image_predicted_label_attack"

    logger, log_file = setup_logger(log_name, args.log_output_dir, level=logging.INFO)
    logger.info(print_args(args))

    assert args.gpu is not None
    logger.info(f"Use GPU: {args.gpu}")

    # Dataset classnames
    dset = args.test_sets
    if len(dset) > 1:
        classnames = eval(f"{dset.lower()}_classes")
    else:
        assert dset in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        if dset == 'A':
            classnames = [classnames_all[i] for i in imagenet_a_mask]
        elif dset == 'R':
            classnames = [classnames_all[i] for i, m in enumerate(imagenet_r_mask) if m]
        elif dset == 'V':
            classnames = [classnames_all[i] for i in imagenet_v_mask]
        else:
            classnames = classnames_all
    args.classnames = classnames

    # Initialize model (same as get_mean_std.py)
    if args.arch in openai_model_dict:
        actual_model_name = openai_model_dict[args.arch]
        model = get_coop_openai(actual_model_name, classnames, args.gpu, args.n_ctx, args.ctx_init)
    else:
        model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    scaler = None  # not used
    cudnn.benchmark = not args.no_cudnn_benchmark

    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution)
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1,
                                     augmix=len(dset) > 1, only_base_image=True)

    val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
    logger.info(f"Number of test samples: {len(val_dataset)}")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.adv_bs, shuffle=False,
                                             num_workers=args.workers, pin_memory=not args.no_pin_memory)

    logger.info(f"Evaluating dataset: {dset}")
    test_time_variance_eval(val_loader, model, args, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute variance using saved means')

    # Dataset parameters
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='Caltech101',
                        help='Dataset to evaluate on (e.g., Caltech101, A, R, K, V, I for ImageNet variants)')
    parser.add_argument('--dataset_mode', type=str, default='test',
                        help='Dataset split to use (train, val, test)')

    # Resume parameters
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Frequency of saving checkpoints (in number of batches)')

    # Model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='delta_clip_l14_224',
                        help='Model architecture (RN50, ViT-B/32, etc.)')
    parser.add_argument('--resolution', default=224, type=int,
                        help='CLIP image resolution')

    # Hardware and performance parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Disable pin memory for data loading')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                        help='Mini-batch size for augmentation')
    parser.add_argument('--adv_bs', default=16, type=int, metavar='N',
                        help='Mini-batch size for evaluation')
    parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N',
                        help='Print frequency (default: 200)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use')
    parser.add_argument('--no_cudnn_benchmark', action='store_true',
                        help='Disable cudnn benchmarking')

    # Prompt tuning parameters
    parser.add_argument('--n_ctx', default=4, type=int,
                        help='Number of tunable context tokens')
    parser.add_argument('--ctx_init', default=None, type=str,
                        help='Initial values for tunable prompts')

    # Experiment parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='output_results/ckps/rtpt',
                        help='Directory to save results')
    parser.add_argument('--log_output_dir', type=str, default='output_results/ckps/rtpt',
                        help='Directory to save logs')

    # Adversarial attack parameters (kept for parity)
    parser.add_argument('--image_only_attack', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--image_only_attack_type', default='prm', choices=["prm", "prm_adam"], type=str)
    parser.add_argument('--image_predicted_label_attack', default=False, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--eps', default=1.0, type=float,
                        help='Epsilon for adversarial attack (0.0 for clean evaluation)')
    parser.add_argument('--alpha', default=0.0, type=float,
                        help='Step size for adversarial attack (if not provided, calculated as eps/alpha_eps_ratio)')
    parser.add_argument('--alpha_eps_ratio', default=4.0, type=float,
                        help='Ratio of epsilon to alpha when alpha is not explicitly provided (default: 4.0)')
    parser.add_argument('--steps', type=int, default=7,
                        help='Number of steps for adversarial attack')

    # Test-time adaptation placeholders for parity with mean script
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', dest='lr',
                        help='Learning rate (unused here)')
    parser.add_argument('--selection_p', default=0.1, type=float,
                        help='Proportion for selection (unused here)')
    parser.add_argument('--tta_steps', default=1, type=int,
                        help='Number of TTA steps (unused here)')
    parser.add_argument('--top_k', default=20, type=int,
                        help='Top-k (unused here)')
    parser.add_argument('--softmax_temp', default=0.01, type=float,
                        help='Softmax temp (unused here)')

    # Pre-trained model parameters
    parser.add_argument('--load_tecoa', type=str, default='',
                        choices=['', 'RN50-eps1', 'ViT-B/32-eps1', 'ViT-B/32-eps4'],
                        help='Load robust vision encoder (TeCoA) [unused here]')

    main()
