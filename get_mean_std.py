"""
Robust Test-time Prompt Tuning (RTPT) for CLIP models.

This script implements test-time adaptation techniques for CLIP models to improve
their robustness against distribution shifts and adversarial attacks. It uses prompt
tuning to adapt the model at test time without modifying the model weights.

Memory-Efficient Statistics Computation:
This implementation uses Welford's online algorithm to compute mean and standard deviation
statistics for model features (cls_token, features_after_attention, features_after_mlp)
in a memory-efficient way. Instead of storing all feature tensors and then computing
statistics at the end, it incrementally updates running statistics with each batch,
which significantly reduces memory usage for large datasets. The script also includes
memory usage monitoring to track CPU and GPU memory consumption during processing.
"""

import argparse
import logging
import time
import gc
import psutil
from copy import deepcopy

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    # Fallback for older torchvision versions
    BICUBIC = Image.BICUBIC

from open_clip.custom_openai_clip import get_coop as get_coop_openai
from clip.custom_clip import get_coop
from open_clip.custom_openai_clip import get_text_embeddings as get_text_embeddings_openai
from clip.custom_clip import get_text_embeddings as get_text_embeddings
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from data.cls_to_names import flower102_classes, food101_classes, dtd_classes, caltech101_classes, pets_classes, \
    sun397_classes, cars_classes, ucf101_classes, aircraft_classes, eurosat_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import set_random_seed
from utils.logger import setup_logger
import os

import torchattacks
import os
from torchvision import transforms

from helper_functions import print_args, plot_probability_bar
from torch.nn.functional import cosine_similarity
import torch
from PIL import Image

openai_model_dict = {
    "delta_clip_l14_224": "hf-hub:zw123/delta_clip_l14_224",
    "tecoa4": "hf-hub:chs20/tecoa4-clip",
    "tecoa2": "hf-hub:chs20/tecoa2-clip",
    "fare2": "hf-hub:chs20/fare2-clip",
    "fare4": "hf-hub:chs20/fare4-clip",
    # "RN50": "RN50",
    "vit_l_14_datacomp_1b": "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
}


import json

def get_zeroshot_templates(dset, template_path='zeroshot-templates.json'):
    """
    Load zeroshot templates based on dataset name.

    Args:
        dset (str): Dataset short name (e.g., 'I', 'cars', 'pets').
        template_path (str): Path to zeroshot-templates.json file.

    Returns:
        list of str: List of template strings.

    Raises:
        ValueError: If dataset name is unknown.
    """
    with open(template_path, 'r') as f:
        templates = json.load(f)

    dset = dset.lower()

    dataset_key_map = {
        'i': 'imagenet1k',
        'a': 'imagenet1k',
        'r': 'imagenet1k',
        'k': 'imagenet1k',
        'v': 'imagenet1k',
        'cars': 'cars',
        'aircraft': 'fgvc_aircraft',
        'pets': 'pets',
        'dtd': 'dtd',
        'caltech101': 'caltech101',
        'flower102': 'flowers',
        'eurosat': 'eurosat',
        'ucf101': 'dummy',
    }

    if dset not in dataset_key_map:
        raise ValueError(f"Unknown dataset: {dset}")

    key = dataset_key_map[dset]
    return templates[key]


def main():


    # Parse arguments and set random seed
    args = parser.parse_args()
    set_random_seed(args.seed)

    # Calculate alpha from epsilon if not provided
    args.alpha = args.eps / args.alpha_eps_ratio

    # Create output directory path with experiment parameters
    args.output_dir = os.path.join(args.output_dir, args.arch, args.test_sets)
    args.log_output_dir = os.path.join(args.log_output_dir, args.arch, args.test_sets)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_output_dir, exist_ok=True)

    # Set up logging

    # Create a log name that includes TTA variations
    # Format floating point values and ensure filename is valid
    log_name = f"ADV_Generation_eps_{args.eps}_steps_{args.steps}"

    log_name = f"ADV_Generation_eps_{args.eps}_steps_{args.steps}"

    if args.image_only_attack:
        log_name += f"_image_only_attack_{args.image_only_attack_type}"
    elif args.image_predicted_label_attack:
        log_name += "_image_predicted_label_attack"
    else:
        log_name = log_name

    logger, log_file = setup_logger(log_name, args.log_output_dir, level=logging.INFO)
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

    class_templates = get_zeroshot_templates(dset)

    # Initialize model with CoOp (Context Optimization)
    if args.arch in openai_model_dict:
        actual_model_name = openai_model_dict[args.arch]
        model = get_coop_openai(actual_model_name, classnames, args.gpu, args.n_ctx, args.ctx_init)
        class_text_embeddings, template_text_embeddings = get_text_embeddings_openai(actual_model_name, classnames, class_templates, args.gpu)

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
                                    augmix=len(dset)>1, only_base_image=True)

    batchsize = args.adv_bs # Process images one at a time for test-time adaptation

    # Create dataset and data loader
    val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
    logger.info(f"Number of test samples: {len(val_dataset)}")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                num_workers=args.workers, pin_memory=not args.no_pin_memory)

    logger.info(f"Evaluating dataset: {dset}")

    # Run evaluation with test-time adaptation
    test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform, logger)

    logger.info(f"Adversarial image generation completed. Results")



def get_adversarial_images(images, targets, attack, paths, index, output_dir, logger=None):
    """
    Generate or load cached adversarial images for a batch of samples.

    Args:
        images (torch.Tensor): Batch of original images (B, C, H, W).
        targets (torch.Tensor): Batch of target labels (B,).
        attack (torchattacks.Attack): Adversarial attack object.
        paths (list): List of paths to original image files (len = B).
        output_dir (str): Directory to save/load adversarial images.
        logger (logging.Logger, optional): Logger for logging information.

    Returns:
        torch.Tensor: Tensor of adversarial images.
    """
    batch_size = images.size(0)
    adv_images = []

    # Check if any adversarial image is missing and generate the attack
    generate_attack = False
    for i in range(batch_size):
        img_filename = os.path.basename(paths[i])
        img_filename = os.path.splitext(img_filename)[0] + ".pt"
        parent_folder_name = os.path.basename(os.path.dirname(paths[i]))
        adv_img_path = os.path.join(output_dir, f"{parent_folder_name}_{img_filename}")

        if not os.path.exists(adv_img_path):
            generate_attack = True  # If any image doesn't exist, attack for the whole batch
            break

    # Generate adversarial images for the entire batch if needed
    if generate_attack:
        adv_images = attack(images, targets)  # Perform the attack on the whole batch
        if logger:
            logger.info(f"Generated adversarial images for the entire batch.")

        for i in range(batch_size):
            # Move tensor to CPU before saving
            img_adv = adv_images[i].detach().cpu()
            img_filename = os.path.basename(paths[i])
            # change the extension to pt (PyTorch tensor)
            img_filename = os.path.splitext(img_filename)[0] + ".pt"
            parent_folder_name = os.path.basename(os.path.dirname(paths[i]))
            adv_img_path = os.path.join(output_dir, f"{parent_folder_name}_{img_filename}")

            # Save tensor directly
            torch.save(img_adv, adv_img_path)
            if logger:
                logger.info(f"Batch:[{index}] Image: [{i}] Saved adversarial tensor to {adv_img_path}")

        # Free memory after processing the batch
        torch.cuda.empty_cache()
        return adv_images
    else:
        logger.info(f"Batch:[{index}] Adversarial images for this batch already exist")
        adv_images = []
        for i in range(batch_size):
            img_filename = os.path.basename(paths[i])
            img_filename = os.path.splitext(img_filename)[0] + ".pt"
            parent_folder_name = os.path.basename(os.path.dirname(paths[i]))
            adv_img_path = os.path.join(output_dir, f"{parent_folder_name}_{img_filename}")

            # Load tensor directly
            img_adv = torch.load(adv_img_path)
            adv_images.append(img_adv)

        # Stack tensors and move to GPU
        adv_images = torch.stack(adv_images, dim=0).cuda()
        return adv_images


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
        torch.Tensor: Adversarial image tensor.
    """
    # Create a unique filename for the adversarial image
    if path is not None:
        # Extract filename from path and the preceding directory
        img_filename = os.path.basename(path[0])
        # Change the extension to pt (PyTorch tensor)
        img_filename = os.path.splitext(img_filename)[0] + ".pt"
        parent_folder_name = os.path.basename(os.path.dirname(path[0]))
        adv_img_path = os.path.join(output_dir, f"{parent_folder_name}_{img_filename}")
    else:
        # If path is not available, use index as identifier
        adv_img_path = os.path.join(output_dir, f"{index}.pt")

    # Check if adversarial image already exists
    if os.path.exists(adv_img_path):
        if logger:
            logger.info(f"Loading existing adversarial tensor from {adv_img_path}")
        # Load existing adversarial tensor
        img_adv = torch.load(adv_img_path)
    else:
        # Create adversarial image using attack
        adv_image = attack(image, target)
        if logger:
            logger.info(f"Generated adversarial image with shape: {adv_image.shape}")

        # Move tensor to CPU before saving
        img_adv = adv_image.squeeze(0).detach().cpu()
        # Save the adversarial tensor directly
        torch.save(img_adv, adv_img_path)
        if logger:
            logger.info(f"Saved adversarial tensor to {adv_img_path}")

        # Free memory for large datasets
        del adv_image
        torch.cuda.empty_cache()

    return img_adv


def purify_zi(img_emb, iter=10, step_size=30., temp_emb_all=None):
    step_size_u = step_size
    batch, device = img_emb.shape[0], img_emb.device
    if not img_emb.requires_grad:
        img_emb.requires_grad = True  # 确保图像嵌入需要梯度

    text_embed = temp_emb_all.mean(dim=1)
    text_embed = text_embed.repeat(batch, 1).to(device)

    momentum = torch.zeros_like(img_emb)
    norm = "L2"
    gamma = 0.
    for i in range(iter):
        r = torch.norm(img_emb, dim=1, keepdim=True)
        u = img_emb / r

        logits_uncond = cosine_similarity(img_emb, text_embed, dim=1)
        loss = - logits_uncond
        grad = torch.autograd.grad(loss, img_emb, torch.ones_like(loss), retain_graph=True)[0]

        grad_u = r * grad

        if norm == "Linf":
            momentum = gamma * momentum - (1 - gamma) * grad_u / torch.norm(grad_u, p=1)
            u = u + step_size_u * momentum.sign()
        elif norm == "L2":
            momentum = gamma * momentum - (1 - gamma) * grad_u / torch.norm(grad_u, p=2)
            u = u + step_size_u * momentum

        u = u / torch.norm(u, dim=1, keepdim=True)
        img_emb = r * u

    return img_emb

def get_memory_usage():
    """
    Get current memory usage of the process.
    
    Returns:
        tuple: (CPU memory in MB, GPU memory in MB if available)
    """
    # Get CPU memory
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    # Get GPU memory if available
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        
    return cpu_memory, gpu_memory

def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform, logger=None,  template_text_embeddings=None, class_text_embeddings=None):
    """
    Evaluate model performance with test-time adaptation.

    This function evaluates the model on a validation dataset, applying test-time adaptation
    to improve performance. It can also evaluate robustness against adversarial attacks
    if specified in the arguments.
    
    The function supports resuming from a previous run if interrupted. It saves checkpoints
    periodically during processing, which can be used to resume from the last saved state.
    This is useful for long-running computations on large datasets.

    Args:
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        model (torch.nn.Module): The model to evaluate.
        model_state (dict, optional): Model state dictionary for resetting.
        optimizer (torch.optim.Optimizer): Optimizer for test-time tuning.
        optim_state (dict): Optimizer state dictionary for resetting.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision.
        args (argparse.Namespace): Arguments containing evaluation parameters.
            args.resume (bool): Whether to resume from a previous run if checkpoint exists.
            args.checkpoint_freq (int): Frequency of saving checkpoints (in number of batches).
        data_transform (callable): Data transformation function.
        logger (logging.Logger, optional): Logger for logging information.
        template_text_embeddings (torch.Tensor, optional): Template text embeddings.
        class_text_embeddings (torch.Tensor, optional): Class text embeddings.

    Returns:
        list: [original_accuracy, test_time_adapted_accuracy]
    """

    # Set model to evaluation mode
    model.eval()
    
    # Initialize memory tracking
    initial_cpu_mem, initial_gpu_mem = get_memory_usage()
    if logger:
        logger.info(f"Initial memory usage - CPU: {initial_cpu_mem:.2f} MB, GPU: {initial_gpu_mem:.2f} MB")

    if logger:
        logger.info(f"Starting evaluation with batch size: {args.batch_size}, selection percentage: {args.selection_p}")
        #logger.info(f"Test-time adaptation steps: {args.tta_steps}, learning rate: {args.lr}")

    # Initialize adversarial attack if specified
    if args.eps > 0.0:
        assert args.steps > 0
        # Create PGD attack with specified parameters
        if args.image_only_attack:
            if args.image_only_attack_type=="prm":
                atk = torchattacks.PGD_PRM(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps)
            elif args.image_only_attack_type=="prm_adam":
                atk = torchattacks.PGD_PRM_ADAM(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps)
            else:
                raise ValueError(f"Unknown image only attack type: {args.image_only_attack_type}")


        else:
            atk = torchattacks.PGD(model, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps,
                                   image_only_attack=False,
                                   image_predicted_label_attack=args.image_predicted_label_attack)
        if logger:
            logger.info(f"Using PGD attack with epsilon: {args.eps/255:.6f}, alpha: {args.alpha/255:.6f}, steps: {args.steps} image only attack {args.image_only_attack} image predicted label attack {args.image_predicted_label_attack}")


    end = time.time()
    # Create directory for saving adversarial images if needed
    if args.eps > 0.0:
        if args.image_only_attack:
            adv_images_dir = os.path.join(args.output_dir, f"adv_images_eps_{args.eps}_alpha_{args.alpha}_steps_{args.steps}_image_only_attack_{args.image_only_attack_type}")
        elif args.image_predicted_label_attack:
            adv_images_dir = os.path.join(args.output_dir, f"adv_images_eps_{args.eps}_alpha_{args.alpha}_steps_{args.steps}_image_predicted_label_attack")
        else:
            adv_images_dir = os.path.join(args.output_dir, f"adv_images_eps_{args.eps}_alpha_{args.alpha}_steps_{args.steps}")
    else:
        adv_images_dir = os.path.join(args.output_dir, "Clean_images")

    # Create directory for saving statistics and checkpoints
    stats_dir = os.path.join(args.output_dir, "mean_statistics")
    os.makedirs(stats_dir, exist_ok=True)
    checkpoint_path = os.path.join(stats_dir, "checkpoint.pt")

    os.makedirs(adv_images_dir, exist_ok=True)
    if logger:
        logger.info(f"Using directory : {adv_images_dir}")
        logger.info(f"Statistics will be saved to: {stats_dir}")


    adv_correct_orig = 0
    clean_correct_orig = 0
    total = 0
    
    # Initialize accumulators for online computation of mean
    # This implementation uses tensor operations to efficiently compute means
    # in a single pass without storing all data points in memory.
    # This is memory-efficient for large datasets compared to storing all tensors.
    #
    # The algorithm works by:
    # 1. Computing the mean across the batch dimension using torch.mean()
    # 2. Updating the running mean using a weighted average formula:
    #    updated_mean = (prev_count * old_mean + batch_size * batch_mean) / new_count
    #
    # This vectorized approach is much faster than using loops over batch samples
    
    # ===== Resume functionality =====
    # Check if we should resume from a previous run by loading the checkpoint
    # This allows the script to continue from where it left off if it was interrupted
    start_batch_idx = 0
    if  os.path.exists(checkpoint_path):
        if logger:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Load checkpoint containing all the statistics and state information
        checkpoint = torch.load(checkpoint_path)
        
        # Restore all the statistics accumulators from the checkpoint
        # This ensures we continue with the exact same state as when the checkpoint was saved
        cls_token_count = checkpoint['cls_token_count']
        cls_token_mean = checkpoint['cls_token_mean']
        attention_counts = checkpoint['attention_counts']
        attention_means = checkpoint['attention_means']
        mlp_counts = checkpoint['mlp_counts']
        mlp_means = checkpoint['mlp_means']
        
        # Start from the next batch after the one that was checkpointed
        start_batch_idx = checkpoint['batch_idx'] + 1
        
        # Restore the total samples processed counter
        total = checkpoint['total']
        
        if logger:
            logger.info(f"Resuming from batch {start_batch_idx} with {total} samples processed so far")
    else:
        # If not resuming, initialize all accumulators from scratch
        
        # For cls_token
        cls_token_count = 0
        cls_token_mean = None
        
        # For features_after_attention and features_after_mlp
        # We'll initialize these after seeing the first batch to get the dimensions
        attention_counts = None
        attention_means = None
        
        mlp_counts = None
        mlp_means = None

    # Iterate through validation data
    for i, data in enumerate(val_loader):
        # ===== Resume functionality: Skip already processed batches =====
        # When resuming from a checkpoint, we need to skip batches that were already processed
        # This ensures we don't double-count samples and maintain deterministic results
        if i < start_batch_idx:
            continue
            
        # Handle different return formats (with or without path)
        if len(data) == 3:
            images, target, path = data
        else:
            images, target = data
            path = None

        assert args.gpu is not None
        target = target.cuda(args.gpu, non_blocking=True)
        images = images.cuda(args.gpu, non_blocking=True)

        if args.eps > 0.0:
            # Get adversarial image (either generate or load from cache)
            adv_images = get_adversarial_images(
                images, target, atk, path, i, adv_images_dir, logger=logger)

            # Pass adversarial images to the model
            adv_images = adv_images.cuda(args.gpu, non_blocking=True)


            # Adversarial examples
            # Compute original adversarial accuracy without purification
            adv_cls_token, adv_features_after_attention, adv_features_after_mlp = model(adv_images)



        # Clean Samples
        cls_token, features_after_attention, features_after_mlp = model(images)
        # cls_token shape : (B, D) Batch size, D is the dimension of the CLIP model
        # features_after_attention is a list of features for each layer: each tensor is of shape (N,B,D) N is the number of tokens, B is the batch size, D is the dimension of the CLIP model
        # features_after_mlp is a list of features for each layer: each tensor is of shape N,B,D) N is the number of tokens, B is the batch size, D is the dimension of the CLIP model

        # Process cls_token using tensor operations
        # Move tensor to CPU to save GPU memory and detach from computation graph
        cls_token_cpu = cls_token.detach().cpu()
        batch_size = cls_token_cpu.size(0)  # Get the current batch size
        
        # Initialize mean accumulator if this is the first batch
        if cls_token_mean is None:
            # Create a tensor with the same shape as a single cls_token
            cls_token_mean = torch.zeros_like(cls_token_cpu[0])
            cls_token_count = 0
        
        # Update running mean using batch mean
        # Compute mean across the batch dimension
        cls_token_batch_mean = torch.mean(cls_token_cpu, dim=0)
        
        # Update the running mean with the new batch
        # Formula: updated_mean = (count * old_mean + batch_size * batch_mean) / (count + batch_size)
        cls_token_mean = (cls_token_count * cls_token_mean + batch_size * cls_token_batch_mean) / (cls_token_count + batch_size)
        cls_token_count += batch_size
        
        # Initialize feature accumulators if this is the first batch
        if attention_counts is None:
            # Get the number of layers in the model
            num_layers = len(features_after_attention)
            
            # Create lists to store counters and means for each layer
            attention_counts = []
            attention_means = []
            mlp_counts = []
            mlp_means = []
            
            # Initialize accumulators for all layers at once
            for layer_idx in range(num_layers):
                # Get dimensions for this layer: num_tokens (e.g., 197 for ViT), batch_size, feature_dim
                num_tokens, _, feature_dim = features_after_attention[layer_idx].shape
                
                # Create counter and mean tensors using direct initialization
                attention_counts.append(torch.zeros(num_tokens, dtype=torch.int64))
                attention_means.append(torch.zeros(num_tokens, feature_dim))
                mlp_counts.append(torch.zeros(num_tokens, dtype=torch.int64))
                mlp_means.append(torch.zeros(num_tokens, feature_dim))
        
        # Process each layer to update running means using tensor operations
        # This vectorized approach is significantly faster than nested loops:
        # 1. It computes means across the batch dimension in a single operation
        # 2. It updates all feature dimensions for each token simultaneously
        # 3. It avoids Python loop overhead by leveraging optimized C++/CUDA implementations
        for layer_idx in range(len(features_after_attention)):
            # Move attention features to CPU to save GPU memory
            attention_features = features_after_attention[layer_idx].detach().cpu()
            num_tokens, batch_size, feature_dim = attention_features.shape
            
            # Compute mean across batch dimension for each token position
            # This replaces the nested loops with a single tensor operation
            batch_attention_means = torch.mean(attention_features, dim=1)  # Shape: [num_tokens, feature_dim]
            
            # Update counts for all token positions at once
            prev_counts = attention_counts[layer_idx].clone().unsqueeze(1)  # Shape: [num_tokens, 1]
            attention_counts[layer_idx] += batch_size
            new_counts = attention_counts[layer_idx].clone().unsqueeze(1)  # Shape: [num_tokens, 1]
            
            # Update running means for all token positions at once using vectorized operations
            # Formula: updated_mean = (prev_count * old_mean + batch_size * batch_mean) / new_count
            attention_means[layer_idx] = (prev_counts * attention_means[layer_idx] + 
                                         batch_size * batch_attention_means) / new_counts
            
            # Move MLP features to CPU to save GPU memory
            mlp_features = features_after_mlp[layer_idx].detach().cpu()
            
            # Compute mean across batch dimension for each token position
            # This replaces the nested loops with a single tensor operation
            batch_mlp_means = torch.mean(mlp_features, dim=1)  # Shape: [num_tokens, feature_dim]
            
            # Update counts for all token positions at once
            prev_counts = mlp_counts[layer_idx].clone().unsqueeze(1)  # Shape: [num_tokens, 1]
            mlp_counts[layer_idx] += batch_size
            new_counts = mlp_counts[layer_idx].clone().unsqueeze(1)  # Shape: [num_tokens, 1]
            
            # Update running means for all token positions at once using vectorized operations
            # Formula: updated_mean = (prev_count * old_mean + batch_size * batch_mean) / new_count
            mlp_means[layer_idx] = (prev_counts * mlp_means[layer_idx] + 
                                   batch_size * batch_mlp_means) / new_counts
        
        # Print progress information periodically
        if logger:
            # Track and log progress more frequently for smaller datasets
            log_freq = max(1, len(val_loader) // 20)  # Log approximately 20 times during processing
            
            if i % log_freq == 0 or i == len(val_loader) - 1:  # Log at regular intervals and at the end
                # Get current memory usage
                cpu_mem, gpu_mem = get_memory_usage()
                
                # Calculate progress percentage
                progress = (i + 1) / len(val_loader) * 100
                
                # Log detailed progress information
                logger.info(f"Progress: {progress:.1f}% - Processed batch {i+1}/{len(val_loader)} with {batch_size} samples")
                logger.info(f"Memory usage: CPU: {cpu_mem:.2f} MB, GPU: {gpu_mem:.2f} MB")
                logger.info(f"Total samples processed so far: {total + batch_size}")

        total += target.size(0)

        # ===== Checkpoint saving =====
        # Save checkpoint periodically based on the checkpoint frequency
        # This allows resuming the computation if the script is interrupted
        if args.checkpoint_freq > 0 and (i + 1) % args.checkpoint_freq == 0:
            # Create checkpoint dictionary with all necessary state information
            # This includes:
            # - Current batch index to know where to resume from
            # - Total samples processed so far
            # - All statistics accumulators (counts and means)
            checkpoint = {
                'batch_idx': i,
                'total': total,
                'cls_token_count': cls_token_count,
                # Move tensors to CPU before saving to avoid GPU memory issues
                'cls_token_mean': cls_token_mean.cpu() if cls_token_mean is not None else None,
                'attention_counts': [count.cpu() for count in attention_counts] if attention_counts is not None else None,
                'attention_means': [mean.cpu() for mean in attention_means] if attention_means is not None else None,
                'mlp_counts': [count.cpu() for count in mlp_counts] if mlp_counts is not None else None,
                'mlp_means': [mean.cpu() for mean in mlp_means] if mlp_means is not None else None
            }
            
            # Save checkpoint to disk
            torch.save(checkpoint, checkpoint_path)
            
            if logger:
                logger.info(f"Saved checkpoint at batch {i+1}/{len(val_loader)} ({(i+1)/len(val_loader)*100:.1f}%)")

        # Free memory
        del images
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        end = time.time()

    # ===== Finalize and save statistics =====
    if logger:
        logger.info("=" * 50)
        logger.info("Finalizing statistics computation and saving results...")
        logger.info(f"Total samples processed: {total}")
    
    # Create directory for saving statistics (already created earlier)
    if logger:
        logger.info(f"Saving final statistics to directory: {stats_dir}")
        
    # ===== Checkpoint cleanup =====
    # Clean up checkpoint file if processing completed successfully
    # This prevents accidentally resuming from an old checkpoint when a full run has completed
    # The checkpoint is only needed if the script is interrupted before completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        if logger:
            logger.info(f"Removed checkpoint file as processing completed successfully")
    
    # ----- Save CLS token mean -----
    cls_token_path = os.path.join(stats_dir, "cls_token_mean.pt")
    torch.save(cls_token_mean, cls_token_path)
    if logger:
        logger.info(f"Saved CLS token mean with shape {cls_token_mean.shape}")
        logger.info(f"File: {cls_token_path}")
    
    # ----- Save attention feature means for each layer -----
    if logger:
        logger.info(f"\nSaving attention feature means for {len(attention_means)} layers:")
    
    for layer_idx in range(len(attention_means)):
        # Create filename for this layer
        attention_path = os.path.join(stats_dir, f"features_after_attention_layer_{layer_idx}_mean.pt")
        # Save the mean statistics
        torch.save(attention_means[layer_idx], attention_path)
        if logger:
            shape_info = f"{attention_means[layer_idx].shape}"
            logger.info(f"Layer {layer_idx}: shape {shape_info}")
    
    # ----- Save MLP feature means for each layer -----
    if logger:
        logger.info(f"\nSaving MLP feature means for {len(mlp_means)} layers:")
    
    for layer_idx in range(len(mlp_means)):
        # Create filename for this layer
        mlp_path = os.path.join(stats_dir, f"features_after_mlp_layer_{layer_idx}_mean.pt")
        # Save the mean statistics
        torch.save(mlp_means[layer_idx], mlp_path)
        if logger:
            shape_info = f"{mlp_means[layer_idx].shape}"
            logger.info(f"Layer {layer_idx}: shape {shape_info}")
    
    # ----- Save a summary of the statistics in a text file -----
    summary_path = os.path.join(stats_dir, "statistics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("MEAN STATISTICS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset: {args.test_sets}\n")
        f.write(f"Model architecture: {args.arch}\n")
        f.write(f"Total samples processed: {total}\n")
        f.write(f"Total batches processed: {len(val_loader)}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Resume enabled: {args.resume}\n")
        f.write(f"Checkpoint frequency: Every {args.checkpoint_freq} batches\n\n")
        
        # Add timestamp for reproducibility
        from datetime import datetime
        f.write(f"Completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CLS token statistics:\n")
        f.write(f"  - Mean shape: {cls_token_mean.shape}\n\n")
        
        f.write("Features after attention statistics:\n")
        for layer_idx in range(len(attention_means)):
            f.write(f"  - Layer {layer_idx} shape: {attention_means[layer_idx].shape}\n")
        f.write("\n")
        
        f.write("Features after MLP statistics:\n")
        for layer_idx in range(len(mlp_means)):
            f.write(f"  - Layer {layer_idx} shape: {mlp_means[layer_idx].shape}\n")
    
    if logger:
        logger.info(f"\n[OK] Saved summary to {summary_path}")
        logger.info("=" * 50)
        logger.info("Statistics computation completed successfully!")
    
    # Final memory usage tracking
    final_cpu_mem, final_gpu_mem = get_memory_usage()
    if logger:
        logger.info("\n" + "=" * 50)
        logger.info("MEMORY USAGE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Initial memory - CPU: {initial_cpu_mem:.2f} MB, GPU: {initial_gpu_mem:.2f} MB")
        logger.info(f"Final memory   - CPU: {final_cpu_mem:.2f} MB, GPU: {final_gpu_mem:.2f} MB")
        logger.info(f"Memory change  - CPU: {final_cpu_mem - initial_cpu_mem:.2f} MB, GPU: {final_gpu_mem - initial_gpu_mem:.2f} MB")
        logger.info("=" * 50)
        logger.info(f"All mean statistics have been successfully computed and saved to {stats_dir}")


if __name__ == '__main__':
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')

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
    # pin memory, default is True
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Pin memory for data loading')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                        help='Mini-batch size for augmentation')
    parser.add_argument('--adv_bs', default=16, type=int, metavar='N',
                        help='Mini-batch size for augmentation')
    parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N',
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
    parser.add_argument('--log_output_dir', type=str, default='output_results/ckps/rtpt',
                        help='Directory to save results')

    # Adversarial attack parameters
    parser.add_argument('--image_only_attack', default=False, type=lambda x: (str(x).lower() == 'true') )
    parser.add_argument('--image_only_attack_type', default='prm', choices=["prm", "prm_adam"], type=str)
    parser.add_argument('--image_predicted_label_attack', default=False, type=lambda x: (str(x).lower() == 'true') )

    parser.add_argument('--eps', default=1.0, type=float,
                        help='Epsilon for adversarial attack (0.0 for clean evaluation)')
    parser.add_argument('--alpha', default=0.0, type=float,
                        help='Step size for adversarial attack (if not provided, calculated as eps/alpha_eps_ratio)')
    parser.add_argument('--alpha_eps_ratio', default=4.0, type=float,
                        help='Ratio of epsilon to alpha when alpha is not explicitly provided (default: 4.0)')
    parser.add_argument('--steps', type=int, default=7,
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
