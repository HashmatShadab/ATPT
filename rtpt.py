"""
Robust Test-time Prompt Tuning (RTPT) for CLIP models.

This script implements test-time adaptation techniques for CLIP models to improve
their robustness against distribution shifts and adversarial attacks. It uses prompt
tuning to adapt the model at test time without modifying the model weights.
"""

import argparse
import logging
import time
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    # Fallback for older torchvision versions
    BICUBIC = Image.BICUBIC

from open_clip.custom_openai_clip import get_coop as get_coop_openai
from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from data.cls_to_names import flower102_classes, food101_classes, dtd_classes, caltech101_classes, pets_classes, \
    sun397_classes, cars_classes, ucf101_classes, aircraft_classes, eurosat_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
from utils.logger import setup_logger
import os

import torchattacks

import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
from helper_functions import plot_image, get_top_sim, print_args, calculate_entropy, entropy_avg, select_confident_samples, test_time_tuning

openai_model_dict = {
    "delta_clip_l14_224": "hf-hub:zw123/delta_clip_l14_224",
    "tecoa4": "hf-hub:chs20/tecoa4-clip",
    "tecoa2": "hf-hub:chs20/tecoa2-clip",
    "fare2": "hf-hub:chs20/fare2-clip",
    "fare4": "hf-hub:chs20/fare4-clip",
    # "RN50": "RN50",
}



def create_log_name(args):
    """Creates a standardized log name from experiment parameters"""
    # R-TPT hyperparameters for log name
    log_name = f"ADV_eps_{args.eps}_steps_{args.steps}_TPT_lr_{args.lr}_step_{args.tta_steps}_selection_{args.selection_p}_topk_neighbours_{args.top_k}_sftemp_{args.softmax_temp}"
    return log_name




def main():


    # Parse arguments and set random seed
    args = parser.parse_args()
    set_random_seed(args.seed)

    # Calculate alpha from epsilon if not provided
    args.alpha = args.eps / args.alpha_eps_ratio

    # Create output directory path with experiment parameters
    args.output_dir = os.path.join(args.output_dir, args.arch, args.test_sets)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging

    # Create a log name that includes TTA variations
    log_name = create_log_name(args)
    logger, log_file = setup_logger(log_name, args.output_dir, level=logging.INFO)
    logger.info(print_args(args))

    # Ensure GPU is available
    assert args.gpu is not None
    set_random_seed(args.seed)
    logger.info(f"Use GPU: {args.gpu} for training")

    # Determine class names based on dataset
    dset = args.test_sets
    if len(dset) > 1: 
        # For multi-character dataset names (e.g., 'Caltech101')
        # This would require importing the specific classes for each dataset
        # For now, we keep using eval for this case as it's not a common path
        classnames = eval(f"{dset.lower()}_classes")
    else:
        # For single-character dataset codes (ImageNet variants)
        assert dset in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes

        # Select appropriate class names based on dataset code
        if dset == 'A':
            # ImageNet-A
            classnames = [classnames_all[i] for i in imagenet_a_mask]
        elif dset == 'R':
            # ImageNet-R
            classnames = [classnames_all[i] for i, m in enumerate(imagenet_r_mask) if m]
        elif dset == 'V':
            # ImageNet-V
            classnames = [classnames_all[i] for i in imagenet_v_mask]
        else:
            # For ImageNet (I) or ImageNet-K
            classnames = classnames_all
    args.classnames = classnames

    # Initialize model with CoOp (Context Optimization)
    if args.arch in openai_model_dict:
        actual_model_name = openai_model_dict[args.arch]
        model = get_coop_openai(actual_model_name, classnames, args.gpu, args.n_ctx, args.ctx_init)
    else:
        model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)

    model_state = None

    # Load robust vision encoder (TeCoA) if specified
    if len(args.load_tecoa) > 0:
        args.robust_pretrain_path = {
            'RN50-eps1': 'pretrain/tecoa/rn50_eps1.pth.tar',
        }[args.load_tecoa]
        robust_state_dict = torch.load(args.robust_pretrain_path, map_location='cpu')
        model.image_encoder.load_state_dict(robust_state_dict['vision_encoder_state_dict'])
        logger.info('Loaded robust vision encoder')

    # Freeze all parameters except prompt learner
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
                param.requires_grad_(False)

    logger.info(f"=> Model created: visual backbone {args.arch}")

    # Move model to GPU
    if not torch.cuda.is_available():
        logger.warning('Using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # Set up optimizer for prompt parameters only
    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.lr)
    optim_state = deepcopy(optimizer.state_dict())

    # Set up additional training parameters
    scaler = None  # No mixed precision scaling used
    cudnn.benchmark = not args.no_cudnn_benchmark  # Enable cudnn benchmarking for faster training unless disabled, default is True


    # Set up data transformations and evaluation

    # Set up image transformations
    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # normalize is commented out - intentional
        ])

    # # Create data augmentation transformer
    data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1,
                                    augmix=len(dset)>1, only_base_image=False)

    batchsize = 1 # Process images one at a time for test-time adaptation

    # Create dataset and data loader
    val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
    logger.info(f"Number of test samples: {len(val_dataset)}")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                num_workers=args.workers, pin_memory=not args.no_pin_memory)

    logger.info(f"Evaluating dataset: {dset}")

    # Run evaluation with test-time adaptation
    results = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform, logger)

    # Clean up to free memory
    del val_dataset, val_loader

    # Format and save results
    if args.eps <= 0:
        # Clean accuracy (no adversarial attack)
        log_msg = f"=> Acc. on testset [{dset}]: Clean Acc @1 {results[0]}/ TTA Clean Acc @1 {results[1]}"
        save_log = {'clean_acc': results[0], 'tta_clean_acc': results[1]}
    else:
        # Adversarial accuracy
        log_msg = f"=> Acc. on testset [{dset}]: Adv Acc @1 {results[0]}/ TTA Adv Acc @1 {results[1]}"
        save_log = {'adv_acc': results[0], 'tta_adv_acc': results[1]}

    # Log results
    logger.info(log_msg)

    # Save results to file
    torch.save(save_log, os.path.join(args.output_dir, 'results_log.pt'))


def get_adversarial_image(image, target, attack, path, index, output_dir, logger=None):
    """
    Generate or load a cached adversarial image.

    Args:
        image (torch.Tensor): Original image tensor.
        target (torch.Tensor): Target label.
        attack (torchattacks.Attack): Adversarial attack object.
        path (list or None): Path to the original image file.
        index (int): Index of the current sample.
        output_dir (str): Directory to save/load adversarial images.
        logger (logging.Logger, optional): Logger for logging information.

    Returns:
        PIL.Image.Image: Adversarial image.
    """
    # Create a unique filename for the adversarial image
    if path is not None:
        # Extract filename from path and the preceding directory
        img_filename = os.path.basename(path[0])
        # change the extension to .png
        img_filename = os.path.splitext(img_filename)[0] + '.pt'
        parent_folder_name = os.path.basename(os.path.dirname(path[0]))
        adv_img_path = os.path.join(output_dir, f"{parent_folder_name}_{img_filename}")

    else:
        # If path is not available, use index as identifier
        adv_img_path = os.path.join(output_dir, f"{index}.pt")

    # Check if adversarial image already exists
    if os.path.exists(adv_img_path):
        if logger:
            logger.info(f"Loading existing adversarial image from {adv_img_path}")
        # Load existing adversarial image tensor
        adv_tensor = torch.load(adv_img_path)
        # Convert to PIL for return
        img_adv = transforms.ToPILImage()(adv_tensor)


    else:
        # Create adversarial image using attack
        adv_image = attack(image, target)
        if logger:
            logger.debug(f"Generated adversarial image with shape: {adv_image.shape}")


        # Move tensor to CPU before saving
        adv_tensor = adv_image.squeeze(0).detach().cpu()

        # Save the adversarial tensor
        torch.save(adv_tensor, adv_img_path)

        if logger:
            logger.info(f"Saved adversarial image to {adv_img_path}")

        # Convert to PIL for return
        img_adv = transforms.ToPILImage()(adv_tensor)

        # Free memory for large datasets
        del adv_image
        torch.cuda.empty_cache()


    return img_adv


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform, logger=None):
    """
    Evaluate model performance with test-time adaptation.

    This function evaluates the model on a validation dataset, applying test-time adaptation
    to improve performance. It can also evaluate robustness against adversarial attacks
    if specified in the arguments.

    Args:
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        model (torch.nn.Module): The model to evaluate.
        model_state (dict, optional): Model state dictionary for resetting.
        optimizer (torch.optim.Optimizer): Optimizer for test-time tuning.
        optim_state (dict): Optimizer state dictionary for resetting.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision.
        args (argparse.Namespace): Arguments containing evaluation parameters.
        data_transform (callable): Data transformation function.

    Returns:
        list: [original_accuracy, test_time_adapted_accuracy]
    """
    # Initialize metrics tracking
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)  # Original model accuracy
    tpt1 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)  # Test-time adapted accuracy
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    # Progress display
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, tpt1],
        prefix='Test: ')

    # Set model to evaluation mode
    model.eval()

    if logger:
        logger.info(f"Starting evaluation with batch size: {args.batch_size}, selection percentage: {args.selection_p}")
        logger.info(f"Test-time adaptation steps: {args.tta_steps}, learning rate: {args.lr}")

    # Initialize adversarial attack if specified
    if args.eps > 0.0:
        assert args.steps > 0
        # Create PGD attack with specified parameters
        atk = torchattacks.PGD(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps)
        if logger:
            logger.info(f"Using PGD attack with epsilon: {args.eps/255:.6f}, alpha: {args.alpha/255:.6f}, steps: {args.steps}")

    end = time.time()
    # Create directory for saving adversarial images if needed
    adv_images_dir = os.path.join(args.output_dir, f"adv_images_eps_{args.eps}_alpha_{args.alpha}_steps_{args.steps}")
    if args.eps > 0.0:
        os.makedirs(adv_images_dir, exist_ok=True)
        if logger:
            logger.info(f"Using directory for adversarial images: {adv_images_dir}")

    # Iterate through validation data
    for i, data in enumerate(val_loader):
        # Handle different return formats (with or without path)
        if len(data) == 3:
            images, target, path = data
        else:
            images, target = data
            path = None

        assert args.gpu is not None
        target = target.cuda(args.gpu, non_blocking=True)

        # Generate adversarial examples if specified
        if args.eps > 0.0:
            image = images[0].cuda(args.gpu, non_blocking=True)
            if logger and i == 0:
                logger.debug(f"Original image shape: {image.shape}, target: {target.item()}")

            # Get adversarial image (either generate or load from cache)
            img_adv = get_adversarial_image(
                image, target, atk, path, i, adv_images_dir, logger=logger
            )

            # Apply data transformations to adversarial image
            images = data_transform(img_adv)
            images = [_.unsqueeze(0) for _ in images]

            if logger:
                logger.debug(f"Created {len(images)} augmented views of the adversarial image")

        # Process images based on their format
        if isinstance(images, list):
            # Handle list of tensors (augmented views)
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]  # Original image is the first one
        else:
            # Handle single tensor
            if len(images.size()) > 4:
                # When using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images

        # Concatenate all images (original and augmented views)
        images = torch.cat(images, dim=0)

        # Reset model to initial state for each batch
        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)

        # Get original model outputs and features
        with torch.no_grad():
            clip_output = model(image)  # Output for original image
            clip_features, _, _ = model.forward_features(images)  # Features for all images
            # clip_outputs = model(images)  # Outputs for all images

        # Perform test-time adaptation
        assert args.tta_steps > 0
        test_time_tuning(model, images, optimizer, scaler, args, logger)

        # Get outputs after test-time adaptation
        with torch.no_grad():
            tuned_outputs = model(images)

        # Calculate similarity matrix between features
        sim_matrix_images = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
        # Get top similarity scores
        score = get_top_sim(sim_matrix_images, args)
        # Calculate weights based on similarity scores
        weight = torch.nn.functional.softmax(score/args.softmax_temp, dim=-1) # softmax temperature default is 0.01
        # Weighted average of tuned outputs
        tta_output = torch.bmm(weight.unsqueeze(-1).transpose(1, 2), tuned_outputs.unsqueeze(0)).squeeze(1)

        # Measure accuracy
        acc1, acc5 = accuracy(clip_output, target, topk=(1, 5))  # Original model accuracy
        tpt_acc1, _ = accuracy(tta_output, target, topk=(1, 5))  # Test-time adapted accuracy

        # Update accuracy metrics
        top1.update(acc1[0], images.size(0))
        tpt1.update(tpt_acc1[0], images.size(0))

        if logger and (i < 5 or i % 20 == 0):  # Log detailed info for first few samples and periodically
            if args.eps <= 0:
                logger.debug(f"Sample {i+1}: Original accuracy: {acc1[0].item():.2f}, TTA accuracy: {tpt_acc1[0].item():.2f}")
            else:
                logger.debug(f"Sample {i+1}: Adversarial accuracy: {acc1[0].item():.2f}, TTA adversarial accuracy: {tpt_acc1[0].item():.2f}")

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log progress
        if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
            if logger:
                if args.eps <= 0:
                    logger.info(f'iter:{i+1}/{len(val_loader)}, clip_acc1={top1.avg}, tta_acc1={tpt1.avg}')
                else:
                    logger.info(f'iter:{i+1}/{len(val_loader)}, clip_adv1={top1.avg}, tta_adv1={tpt1.avg}')
            progress.display(i)

    # Display final summary
    progress.display_summary()

    if logger:
        if args.eps <= 0:
            logger.info(f"Final results - Original accuracy: {top1.avg:.2f}, TTA accuracy: {tpt1.avg:.2f}")
            logger.info(f"Improvement from TTA: {tpt1.avg - top1.avg:.2f}")
        else:
            logger.info(f"Final results - Adversarial accuracy: {top1.avg:.2f}, TTA adversarial accuracy: {tpt1.avg:.2f}")
            logger.info(f"Improvement from TTA: {tpt1.avg - top1.avg:.2f}")

    # Return original and test-time adapted accuracies
    return [top1.avg, tpt1.avg]


if __name__ == '__main__':
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')

    # Dataset parameters
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='Caltech101', 
                        help='Dataset to evaluate on (e.g., Caltech101, A, R, K, V, I for ImageNet variants)')
    parser.add_argument('--dataset_mode', type=str, default='test',
                        help='Dataset split to use (train, val, test)')

    # Model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50',
                        help='Model architecture (RN50, ViT-B/32, tecoa4, tecoa2, fare2, fare4, delta_clip_l14_224 etc.)')
    parser.add_argument('--resolution', default=224, type=int, 
                        help='CLIP image resolution')

    # Hardware and performance parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', 
                        help='Number of data loading workers (default: 4)')
    # pin memory, default is True
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Pin memory for data loading')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                        help='Mini-batch size for augmentation')
    parser.add_argument('-p', '--print-freq', default=20, type=int, metavar='N',
                        help='Print frequency (default: 200)')
    parser.add_argument('--gpu', default=0, type=int, 
                        help='GPU id to use')
    parser.add_argument('--no_cudnn_benchmark', action='store_true',
                        help='Disable cudnn benchmarking for potentially more deterministic behavior')

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

    # Adversarial attack parameters
    parser.add_argument('--eps', default=0.0, type=float,
                        help='Epsilon for adversarial attack (0.0 for clean evaluation)')
    parser.add_argument('--alpha', default=0.0, type=float,
                        help='Step size for adversarial attack (if not provided, calculated as eps/alpha_eps_ratio)')
    parser.add_argument('--alpha_eps_ratio', default=4.0, type=float,
                        help='Ratio of epsilon to alpha when alpha is not explicitly provided (default: 4.0)')
    parser.add_argument('--steps', type=int, default=0,
                        help='Number of steps for adversarial attack')

    # Test-time adaptation parameters
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', 
                        help='Learning rate for test-time adaptation', dest='lr')
    parser.add_argument('--selection_p', default=0.1, type=float, 
                        help='Proportion of confident samples to select for adaptation (0.0-1.0)')
    parser.add_argument('--tta_steps', default=1, type=int, 
                        help='Number of test-time adaptation steps')
    parser.add_argument('--top_k', default=20, type=int,
                        help='Number of neighbors for similarity calculation')
    parser.add_argument('--softmax_temp', default=0.01, type=float,
                        help='Temperature parameter for softmax in similarity weighting')

    # Pre-trained model parameters
    parser.add_argument('--load_tecoa', type=str, default='', 
                        choices=['', 'RN50-eps1', 'ViT-B/32-eps1', 'ViT-B/32-eps4'],
                        help='Load robust vision encoder (TeCoA)')

    # Run the main function
    main()
