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

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
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

from helper_functions import print_args
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
    if args.eps > 0.0:
        args.alpha = args.eps / args.alpha_eps_ratio

    # Set up logging
    log_name = f"ADV_Generation_eps_{args.eps}_steps_{args.steps}"

    # Create a log name that includes TTA variations
    # Format floating point values and ensure filename is valid
    if args.transferability:
        log_name = f"ADV_Generation_source_model_{args.source_model}_eps_{args.eps}_steps_{args.steps}"
    else:
        log_name = f"ADV_Generation_eps_{args.eps}_steps_{args.steps}"

    if args.image_only_attack:
        log_name += f"_image_only_attack_{args.image_only_attack_type}"
    elif args.image_predicted_label_attack:
        log_name += "_image_predicted_label_attack"
    else:
        log_name = log_name
    # NOTE: image feature purification logic removed from this script.

    # Update log name if noise-addition is enabled
    if args.add_noise:
        log_name = f"{log_name}_Added_Noise_{args.add_noise_init_noise}"
        if args.add_noise_init_noise == "uniform":
            if args.add_noise_tau == "normal":
                num_anchors = 1
            else:
                num_anchors = args.add_noise_noisy_tau_num_anchors

            log_name = f"{log_name}_Eps_{args.add_noise_eps}_Tau_Type_{args.add_noise_tau}_num_anchors_{num_anchors}"
        elif args.add_noise_init_noise == "gaussian":
            if args.add_noise_tau == "normal":
                num_anchors = 1
            else:
                num_anchors = args.add_noise_noisy_tau_num_anchors
            log_name = f"{log_name}_Sigma_{args.add_noise_gaussian_sigma}_Tau_Type_{args.add_noise_tau}_num_anchors_{num_anchors}"
        else:
            raise ValueError("Unknown init noise type")

        # Create output directory path with experiment parameters
    args.output_dir = os.path.join(args.output_dir, args.arch, args.test_sets)
    args.log_output_dir = os.path.join(args.log_output_dir, args.arch, args.test_sets, log_name)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_output_dir, exist_ok=True)

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
        class_text_embeddings, template_text_embeddings = get_text_embeddings_openai(actual_model_name, classnames,
                                                                                     class_templates, args.gpu)

    else:
        model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
        class_text_embeddings, template_text_embeddings = get_text_embeddings(args.arch, classnames, class_templates,
                                                                              args.gpu)

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
    data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1,
                                     augmix=len(dset) > 1, only_base_image=True)

    batchsize = args.adv_bs  # Process images one at a time for test-time adaptation

    # Create dataset and data loader
    val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
    logger.info(f"Number of test samples: {len(val_dataset)}")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                                             num_workers=args.workers, pin_memory=not args.no_pin_memory)

    logger.info(f"Evaluating dataset: {dset}")

    # Run evaluation with test-time adaptation
    test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform, logger,
                         template_text_embeddings, class_text_embeddings)

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


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform,
                         logger=None, template_text_embeddings=None, class_text_embeddings=None):
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
    # NOTE: Plot saving logic removed from this script.

    # Set model to evaluation mode
    model.eval()

    if logger:
        logger.info(f"Starting evaluation with batch size: {args.batch_size}, selection percentage: {args.selection_p}")
        # logger.info(f"Test-time adaptation steps: {args.tta_steps}, learning rate: {args.lr}")

    # Initialize adversarial attack if specified
    if args.eps > 0.0:
        assert args.steps > 0
        # Create PGD attack with specified parameters
        if args.image_only_attack:
            # NOTE: Only PRM image-only attack is supported in this script.
            if args.image_only_attack_type != "prm":
                raise ValueError(f"Unknown image only attack type: {args.image_only_attack_type}")
            atk = torchattacks.PGD_PRM(model, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps)


        else:
            atk = torchattacks.PGD(model, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps,
                                   image_only_attack=False,
                                   image_predicted_label_attack=args.image_predicted_label_attack)
    else:
        atk = torchattacks.PGD(model, eps=0.0, alpha=0, steps=0, random_start=False)
        if logger:
            logger.info(
                f"Using PGD attack with epsilon: {args.eps / 255:.6f}, alpha: {args.alpha / 255:.6f}, steps: {args.steps} image only attack {args.image_only_attack} image only attack type  {args.image_only_attack_type} image predicted label attack {args.image_predicted_label_attack}")

    if args.add_noise:
        noise_atk = torchattacks.AddNoise(
            model,
            eps=args.add_noise_eps / 255,
            init_noise=args.add_noise_init_noise,
            gaussian_sigma=args.add_noise_gaussian_sigma,
            tau_type=args.add_noise_tau,
            num_anchors=args.add_noise_noisy_tau_num_anchors,
        )
        if logger:
            logger.info(
                f"Using AddNoise with eps={args.add_noise_eps:.6f}, init_noise={args.add_noise_init_noise}, tau_type={args.add_noise_tau}"
            )

    end = time.time()
    # Create directory for saving adversarial images if needed
    if args.image_only_attack:
        adv_images_dir = os.path.join(args.output_dir,
                                      f"adv_images_eps_{args.eps}_alpha_{args.alpha}_steps_{args.steps}_image_only_attack_{args.image_only_attack_type}")
    elif args.image_predicted_label_attack:
        adv_images_dir = os.path.join(args.output_dir,
                                      f"adv_images_eps_{args.eps}_alpha_{args.alpha}_steps_{args.steps}_image_predicted_label_attack")
    else:
        adv_images_dir = os.path.join(args.output_dir,
                                      f"adv_images_eps_{args.eps}_alpha_{args.alpha}_steps_{args.steps}")

    if args.transferability:
        adv_images_dir = adv_images_dir.replace(args.arch, args.source_model)

    logger.info(f"Adversarial examples will be loaded from {adv_images_dir} and evaluated on {args.arch}")

    os.makedirs(adv_images_dir, exist_ok=True)
    if logger:
        logger.info(f"Using directory for adversarial images: {adv_images_dir}")

    adv_correct_counter = 0
    adv_correct_orig = 0
    clean_correct_orig = 0
    total = 0

    # Purification-related bookkeeping removed


    all_true_labels = []
    all_clean_preds = []
    all_adv_preds = []
    all_add_noise_preds = []

    # Vision feature saving
    n_samples = len(val_loader.dataset)
    feat_write_ptr = 0
    image_features_clean = []
    image_features_adv = []
    image_features_adv_counter = []

    if args.image_only_attack:
        features_dir = os.path.join(
            args.log_output_dir,

        )
    elif args.image_predicted_label_attack:
        features_dir = os.path.join(
            args.log_output_dir,

        )
    else:
        features_dir = os.path.join(
            args.log_output_dir,

        )

    if args.transferability:
        features_dir = features_dir.replace(args.arch, args.source_model)

    features_dir = os.path.join(features_dir, "image_features")

    os.makedirs(features_dir, exist_ok=True)
    if logger:
        logger.info(f"Vision features will be saved to {features_dir}")

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
        images = images.cuda(args.gpu, non_blocking=True)

        # Get adversarial image (either generate or load from cache)
        if args.eps > 0.0:
            adv_images = get_adversarial_images(
                images, target, atk, path, i, adv_images_dir, logger=logger)
        else:
            adv_images = images.clone()

        if args.add_noise:
            # If using noise addition, apply it to the generated image
            adv_images_counter, diff_ratio = noise_atk(adv_images, target)
            adv_images_counter = adv_images_counter.cuda(args.gpu, non_blocking=True)

        # Pass adversarial images to the model
        adv_images = adv_images.cuda(args.gpu, non_blocking=True)

        from contextlib import nullcontext

        context = nullcontext()
        # Model forward pass
        with context:


            # Adversarial examples (purification removed; always evaluate original)
            with torch.no_grad():
                adv_logits = model(adv_images)
                adv_probs = adv_logits.softmax(dim=-1)
                _, adv_pred = adv_probs.max(1)
            adv_correct_orig += adv_pred.eq(target).sum().item()

            # If using noise addition, pass the noise-added images to the model
            if args.add_noise:
                with torch.no_grad():
                    adv_logits_counter = model(adv_images_counter)
                    adv_probs_counter = adv_logits_counter.softmax(dim=-1)
                    _, adv_pred_counter = adv_probs_counter.max(1)
                adv_correct_counter += adv_pred_counter.eq(target).sum().item()

            # Clean samples (purification removed; always evaluate original)
            with torch.no_grad():
                clean_logits = model(images)
                clean_probs = clean_logits.softmax(dim=-1)
                _, clean_pred = clean_probs.max(1)
            clean_correct_orig += clean_pred.eq(target).sum().item()

            total += target.size(0)

            # Store labels and predictions
            all_true_labels.extend(target.cpu().numpy().tolist())
            all_clean_preds.extend(clean_pred.cpu().numpy().tolist())
            all_adv_preds.extend(adv_pred.cpu().numpy().tolist())
            if args.add_noise:
                all_add_noise_preds.extend(adv_pred_counter.cpu().numpy().tolist())


        images_vision_features = model(images, get_image_features=True)
        adv_images_vision_features = model(adv_images, get_image_features=True)
        if args.add_noise:
            adv_images_counter_vision_features = model(adv_images_counter, get_image_features=True)

        # Add vision features to the list sample-by-sample.
        # When `adv_bs` / dataloader batch size > 1, each forward returns a batch of features.
        images_vision_features_cpu = images_vision_features.detach().cpu().numpy()
        adv_images_vision_features_cpu = adv_images_vision_features.detach().cpu().numpy()
        for b in range(images_vision_features_cpu.shape[0]):
            image_features_clean.append(images_vision_features_cpu[b])
            image_features_adv.append(adv_images_vision_features_cpu[b])

        if args.add_noise:
            adv_images_counter_vision_features_cpu = adv_images_counter_vision_features.detach().cpu().numpy()
            for b in range(adv_images_counter_vision_features_cpu.shape[0]):
                image_features_adv_counter.append(adv_images_counter_vision_features_cpu[b])

        # Free memory
        del images, adv_images, target
        if args.add_noise:
            del adv_images_counter, adv_logits_counter, adv_probs_counter, adv_pred_counter

        # Clear additional temporary variables
        if 'path' in locals():
            del path

        del adv_logits, adv_probs, adv_pred
        del clean_logits, clean_probs, clean_pred

        # Force garbage collection and clear GPU cache more frequently
        torch.cuda.empty_cache()


        if logger:
            if args.add_noise:
                logger.info(
                    f"Batch {i + 1}/{len(val_loader)}: Clean accuracy {clean_correct_orig / total:.4f} | Adv accuracy: {adv_correct_orig / total:.4f} | AddNoise accuracy: {adv_correct_counter / total:.4f}")
            else:
                logger.info(
                    f"Batch {i + 1}/{len(val_loader)}: Clean accuracy {clean_correct_orig / total:.4f} | Adv accuracy: {adv_correct_orig / total:.4f} ")

        torch.cuda.empty_cache()
        end = time.time()


    # Calculate final accuracy
    original_accuracy_orig = clean_correct_orig / total
    adv_accuracy_orig = adv_correct_orig / total

    if args.add_noise:
        adv_accuracy_counter = adv_correct_counter / total

    # Verify the predictions list saved in the info give the correct accuracies
    all_true_labels_np = np.array(all_true_labels)
    all_clean_preds_np = np.array(all_clean_preds)
    all_adv_preds_np = np.array(all_adv_preds)

    clean_acc_calc = (all_clean_preds_np == all_true_labels_np).mean()
    adv_acc_calc = (all_adv_preds_np == all_true_labels_np).mean()

    if logger:
        logger.info(f"Verification - Clean Accuracy: {original_accuracy_orig:.4f} vs {clean_acc_calc:.4f}")
        logger.info(f"Verification - Adversarial Accuracy: {adv_accuracy_orig:.4f} vs {adv_acc_calc:.4f}")
    else:
        print(f"Verification - Clean Accuracy: {original_accuracy_orig:.4f} vs {clean_acc_calc:.4f}")
        print(f"Verification - Adversarial Accuracy: {adv_accuracy_orig:.4f} vs {adv_acc_calc:.4f}")

    assert abs(
        original_accuracy_orig - clean_acc_calc) < 1e-6, f"Clean accuracy mismatch: {original_accuracy_orig} vs {clean_acc_calc}"
    assert abs(
        adv_accuracy_orig - adv_acc_calc) < 1e-6, f"Adversarial accuracy mismatch: {adv_accuracy_orig} vs {adv_acc_calc}"

    if args.add_noise:
        all_add_noise_preds_np = np.array(all_add_noise_preds)
        counter_acc_calc = (all_add_noise_preds_np == all_true_labels_np).mean()
        if logger:
            logger.info(f"Verification - AddNoise Accuracy: {adv_accuracy_counter:.4f} vs {counter_acc_calc:.4f}")
        else:
            print(f"Verification - AddNoise Accuracy: {adv_accuracy_counter:.4f} vs {counter_acc_calc:.4f}")
        assert abs(
            adv_accuracy_counter - counter_acc_calc) < 1e-6, f"AddNoise accuracy mismatch: {adv_accuracy_counter} vs {counter_acc_calc}"

    if logger:
        if args.add_noise:
            logger.info(
                f"Final Clean accuracy: {original_accuracy_orig:.4f} | Adversarial accuracy: {adv_accuracy_orig:.4f} | AddNoise accuracy: {adv_accuracy_counter:.4f}")
        else:
            logger.info(f"Original accuracy: {original_accuracy_orig:.4f}")
            logger.info(f"Adversarial accuracy: {adv_accuracy_orig:.4f}")
    else:
        if args.add_noise:
            print(
                f"Original accuracy: {original_accuracy_orig:.4f} | Adversarial accuracy: {adv_accuracy_orig:.4f} | AddNoise accuracy: {adv_accuracy_counter:.4f}")
        else:
            print(f"Original accuracy: {original_accuracy_orig:.4f}")
            print(f"Adversarial accuracy: {adv_accuracy_orig:.4f}")

    # save the diff_ratio_after_add_noise and avg_diff_ratio_after_add_noise as a json in log directory
    info = {
        'original_clean_accuracy': original_accuracy_orig,
        'adversarial_accuracy': adv_accuracy_orig,
        "add_noise_accuracy": adv_accuracy_counter if args.add_noise else None,
        'true_labels': all_true_labels,
        'original_clean_predictions': all_clean_preds,
        'adversarial_predictions': all_adv_preds,
        'add_noise_predictions': all_add_noise_preds if args.add_noise else None,
    }
    with open(os.path.join(args.log_output_dir, f'results_after_add_noise.json'), 'w') as f:
        json.dump(info, f, indent=4)

    # save the image features as a npy file in log directory
    image_features_clean = np.array(image_features_clean)
    np.save(os.path.join(features_dir, f'image_features_clean.npy'), image_features_clean)
    if args.add_noise:
        image_features_adv_counter = np.array(image_features_adv_counter)
        np.save(os.path.join(features_dir, f'image_features_adv_counter.npy'), image_features_adv_counter)
    image_features_adv = np.array(image_features_adv)
    np.save(os.path.join(features_dir, f'image_features_adv.npy'), image_features_adv)


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
    parser.add_argument('-a', '--arch', metavar='ARCH', default='delta_clip_l14_224',
                        help='Model architecture (RN50, ViT-B/32, etc.)')
    parser.add_argument('--resolution', default=224, type=int,
                        help='CLIP image resolution')
    parser.add_argument('--transferability', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--source_model', default='fare4', type=str,
                        help="model on which adversarial examples will be generated")

    # Hardware and performance parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4)')
    # pin memory, default is True
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Pin memory for data loading')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
                        help='Mini-batch size for augmentation')
    parser.add_argument('--adv_bs', default=1, type=int, metavar='N',
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
    parser.add_argument('--image_only_attack', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--image_only_attack_type', default='prm', choices=["prm"], type=str)
    parser.add_argument('--image_predicted_label_attack', default=False, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--eps', default=1.0, type=float,
                        help='Epsilon for adversarial attack (0.0 for clean evaluation)')
    parser.add_argument('--alpha', default=0.0, type=float,
                        help='Step size for adversarial attack (if not provided, calculated as eps/alpha_eps_ratio)')
    parser.add_argument('--alpha_eps_ratio', default=4.0, type=float,
                        help='Ratio of epsilon to alpha when alpha is not explicitly provided (default: 4.0)')
    parser.add_argument('--steps', type=int, default=7,
                        help='Number of steps for adversarial attack')

    # Noise-addition (was previously named "counter_attack" in this script)
    parser.add_argument('--add_noise', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--add_noise_eps', default=4.0, type=float)
    parser.add_argument('--add_noise_init_noise', default='uniform', choices=["uniform", "gaussian"], type=str)
    parser.add_argument('--add_noise_gaussian_sigma', default=0.18, type=float)
    parser.add_argument('--add_noise_noisy_tau_num_anchors', default=10, type=int)
    parser.add_argument('--add_noise_tau', default='normal', choices=["normal", "noisy", "normal_anchors"],
                        type=str)

    # NOTE: image feature purification and plot-saving options were removed from this script.

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
