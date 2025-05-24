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
        'flowers102': 'flowers',
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

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging

    # Create a log name that includes TTA variations
    # Format floating point values and ensure filename is valid
    log_name = f"ADV_Generation_eps_{args.eps}_steps_{args.steps}"

    # Update log name if counter_attack is True
    if args.counter_attack:
        log_name = f"{log_name}_counter_attack_eps_{args.counter_attack_eps}_steps_{args.counter_attack_steps}_alpha_{args.counter_attack_alpha}_tau_thres_{args.counter_attack_tau_thres}_beta_{args.counter_attack_beta}_weighted_perturbations_{args.counter_attack_weighted_perturbations}"

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

    class_templates = get_zeroshot_templates(dset)

    # Initialize model with CoOp (Context Optimization)
    if args.arch in openai_model_dict:
        actual_model_name = openai_model_dict[args.arch]
        model = get_coop_openai(actual_model_name, classnames, args.gpu, args.n_ctx, args.ctx_init)
        class_text_embeddings, template_text_embeddings = get_text_embeddings_openai(actual_model_name, classnames, class_templates, args.gpu)

    else:
        model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
        class_text_embeddings, template_text_embeddings = get_text_embeddings(args.arch, classnames, class_templates, args.gpu)


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
    test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform, logger, template_text_embeddings, class_text_embeddings)

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

def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform, logger=None,  template_text_embeddings=None, class_text_embeddings=None):
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

    # Set model to evaluation mode
    model.eval()

    if logger:
        logger.info(f"Starting evaluation with batch size: {args.batch_size}, selection percentage: {args.selection_p}")
        #logger.info(f"Test-time adaptation steps: {args.tta_steps}, learning rate: {args.lr}")

    # Initialize adversarial attack if specified
    if args.eps > 0.0:
        assert args.steps > 0
        # Create PGD attack with specified parameters
        atk = torchattacks.PGD(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps)
        if logger:
            logger.info(f"Using PGD attack with epsilon: {args.eps/255:.6f}, alpha: {args.alpha/255:.6f}, steps: {args.steps}")

    if args.counter_attack:
        # Create counter-attack with specified parameters
        if args.counter_attack_type == "pgd":
            counter_atk = torchattacks.PGDCounter(model, eps=args.counter_attack_eps / 255,
                                                  alpha=args.counter_attack_alpha / 255,
                                                  steps=args.counter_attack_steps,
                                                  tau_thres=args.counter_attack_tau_thres,
                                                  beta=args.counter_attack_beta,
                                                  weighted_perturbation=args.counter_attack_weighted_perturbations)
        elif args.counter_attack_type == "pgd_clip_pure_i":
            if args.pgd_clip_pure_i_text_embeddings == "null":
                embeddings = template_text_embeddings
            elif args.pgd_clip_pure_i_text_embeddings == "class":
                embeddings = class_text_embeddings
            else:
                raise ValueError(f"Unknown text embedding type: {args.pgd_clip_pure_i_text_embeddings}")
            counter_atk = torchattacks.PGDClipPureImage(model, eps=args.counter_attack_eps / 255,
                                                        alpha=args.counter_attack_alpha / 255,
                                                        steps=args.counter_attack_steps, text_embeddings=embeddings
                                                        )
        elif args.counter_attack_type == "pgd_counter_and_clipure_i":
            if args.pgd_clip_pure_i_text_embeddings == "null":
                embeddings = template_text_embeddings
            elif args.pgd_clip_pure_i_text_embeddings == "class":
                embeddings = class_text_embeddings
            else:
                raise ValueError(f"Unknown text embedding type: {args.pgd_clip_pure_i_text_embeddings}")
            counter_atk = torchattacks.PGDCounterClipPureImage(model, eps=args.counter_attack_eps / 255,
                                                               alpha=args.counter_attack_alpha / 255,
                                                               steps=args.counter_attack_steps,
                                                               text_embeddings=embeddings,
                                                               tau_thres=args.counter_attack_tau_thres,
                                                               beta=args.counter_attack_beta,
                                                               weighted_perturbation=args.counter_attack_weighted_perturbations,
                                                               loss_lamda=args.pgd_counter_and_clipure_i_lamda)
        if logger:
            logger.info(f"Using counter-attack with epsilon: {args.counter_attack_eps:.6f}, alpha: {args.counter_attack_alpha:.6f}, steps: {args.counter_attack_steps}")


    end = time.time()
    # Create directory for saving adversarial images if needed
    adv_images_dir = os.path.join(args.output_dir, f"adv_images_eps_{args.eps}_alpha_{args.alpha}_steps_{args.steps}")


    if args.eps > 0.0:
        os.makedirs(adv_images_dir, exist_ok=True)
        if logger:
            logger.info(f"Using directory for adversarial images: {adv_images_dir}")

    adv_correct = 0
    clean_correct = 0
    adv_correct_counter = 0
    total = 0
    adv_emb_correct = 0
    purify_emb_correct = 0
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
        adv_images = get_adversarial_images(
            images, target, atk, path, i, adv_images_dir, logger=logger)

        if args.counter_attack:
            # If using counter-attack, apply it to the generated image
            adv_images_counter = counter_atk(adv_images, target)
            adv_images_counter = adv_images_counter.cuda(args.gpu, non_blocking=True)



        # Pass adversarial images to the model
        adv_images = adv_images.cuda(args.gpu, non_blocking=True)

        # Model forward pass
        with torch.no_grad():
            # Adv
            adv_logits = model(adv_images)
            adv_probs = adv_logits.softmax(dim=-1)
            _, adv_pred = adv_probs.max(1)
            adv_correct += adv_pred.eq(target).sum().item()

            # If using counter-attack, pass the counter-attacked images to the model
            if args.counter_attack:
                adv_logits_counter = model(adv_images_counter)
                adv_probs_counter = adv_logits_counter.softmax(dim=-1)
                _, adv_pred_counter = adv_probs_counter.max(1)
                adv_correct_counter += adv_pred_counter.eq(target).sum().item()

            # Clean
            clean_logits = model(images)
            clean_probs = clean_logits.softmax(dim=-1)
            _, clean_pred = clean_probs.max(1)
            clean_correct += clean_pred.eq(target).sum().item()


            total += target.size(0)

        # Free memory
        del images, adv_logits, adv_probs, adv_pred, clean_logits, clean_probs, clean_pred

        if args.counter_attack:
            del adv_images_counter, adv_logits_counter, adv_probs_counter, adv_pred_counter


        ############### Not working ###############################
        # with torch.enable_grad():
        #
        #     image_embeddings, text_embeddings, logit_scale = model(adv_images, get_image_text_features=True)
        #
        #     image_embeddings_purify = purify_zi(image_embeddings, 10, 30, template_text_embeddings)
        #
        # adv_emb_logits = logit_scale * image_embeddings @ text_embeddings.t()
        # adv_emb_probs = adv_emb_logits.softmax(dim=-1)
        # _, adv_emb_pred = adv_emb_probs.max(1)
        # adv_emb_correct += adv_emb_pred.eq(target).sum().item()
        #
        #
        # purify_emb_logits = logit_scale * image_embeddings_purify @ text_embeddings.t()
        # purify_emb_probs = purify_emb_logits.softmax(dim=-1)
        # _, purify_emb_pred = purify_emb_probs.max(1)
        # purify_emb_correct += purify_emb_pred.eq(target).sum().item()
        ############### Not working ###############################

        if logger and args.counter_attack:
            logger.info(
                f"Batch {i + 1}/{len(val_loader)}: Clean accuracy {clean_correct / total:.4f} | Adv accuracy: {adv_correct / total:.4f} | Counter-attack accuracy: {adv_correct_counter / total:.4f}")
        else:
            logger.info(
                f"Batch {i + 1}/{len(val_loader)}: Clean accuracy {clean_correct / total:.4f} | Adv accuracy: {adv_correct / total:.4f} ")



        torch.cuda.empty_cache()
        end = time.time()

    # Calculate final accuracy
    original_accuracy = clean_correct / total
    adv_accuracy = adv_correct / total


    if args.counter_attack:
        adv_accuracy_counter = adv_correct_counter / total
    if logger and args.counter_attack:
        logger.info(f"Final Clean accuracy: {original_accuracy:.4f} | Adversarial accuracy: {adv_accuracy:.4f} | Counter-attack accuracy: {adv_accuracy_counter:.4f} ")
    elif logger and not args.counter_attack:
        logger.info(f"Original accuracy: {original_accuracy:.4f}")
        logger.info(f"Adversarial accuracy: {adv_accuracy:.4f}")
    else:
        print(f"Original accuracy: {original_accuracy:.4f}")
        print(f"Adversarial accuracy: {adv_accuracy:.4f}")










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

    # Hardware and performance parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4)')
    # pin memory, default is True
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Pin memory for data loading')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                        help='Mini-batch size for augmentation')
    parser.add_argument('--adv_bs', default=48, type=int, metavar='N',
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

    # Adversarial attack parameters
    parser.add_argument('--eps', default=1.0, type=float,
                        help='Epsilon for adversarial attack (0.0 for clean evaluation)')
    parser.add_argument('--alpha', default=0.0, type=float,
                        help='Step size for adversarial attack (if not provided, calculated as eps/alpha_eps_ratio)')
    parser.add_argument('--alpha_eps_ratio', default=4.0, type=float,
                        help='Ratio of epsilon to alpha when alpha is not explicitly provided (default: 4.0)')
    parser.add_argument('--steps', type=int, default=7,
                        help='Number of steps for adversarial attack')

    parser.add_argument('--counter_attack', default=False, type=lambda x: (str(x).lower() == 'true') )
    parser.add_argument('--counter_attack_type', default='pgd', choices=["pgd", "pgd_clip_pure_i", "pgd_counter_and_clipure_i"], type=str)
    parser.add_argument('--counter_attack_steps', default=2, type=int)
    parser.add_argument('--counter_attack_eps', default=4.0, type=float)
    parser.add_argument('--counter_attack_alpha', default=1.0, type=float)
    parser.add_argument('--counter_attack_tau_thres', default=0.2, type=float)
    parser.add_argument('--counter_attack_beta', default=2.0, type=float)
    parser.add_argument('--counter_attack_weighted_perturbations', default=True, type=lambda x: (str(x).lower() == 'true') )

    parser.add_argument('--pgd_clip_pure_i_text_embeddings', default='null', choices=["null", "class"], type=str)
    parser.add_argument('--pgd_counter_and_clipure_i_lamda', default=1.0, type=float)


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
