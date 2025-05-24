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
import json

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
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
from utils.logger import setup_logger
import os

import torchattacks

import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
from helper_functions import plot_image, print_args, rtpt_entropy_avg, entropy_loss_ttl, entropy_of_each_sample, handle_long_windows_path


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


openai_model_dict = {
    "delta_clip_l14_224": "hf-hub:zw123/delta_clip_l14_224",
    "tecoa4": "hf-hub:chs20/tecoa4-clip",
    "tecoa2": "hf-hub:chs20/tecoa2-clip",
    "fare2": "hf-hub:chs20/fare2-clip",
    "fare4": "hf-hub:chs20/fare4-clip",
    # "RN50": "RN50",
}


def test_time_tuning(model, inputs, optimizer, scaler, args, logger=None):
    """
    Perform test-time tuning of the model using entropy minimization.

    This function adapts the model at test time by minimizing the entropy of predictions
    on the input batch. It selects confident samples based on their entropy and uses
    them for adaptation.

    Args:
        model (torch.nn.Module): The model to be tuned.
        inputs (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision training.
        args (argparse.Namespace): Arguments containing tuning parameters.
        logger (logging.Logger, optional): Logger for logging information.

    Returns:
        None
    """
    # Track indices of confident samples
    selected_idx = None

    if logger:
        logger.debug(f"Starting test-time tuning with {args.tta_steps} steps")

    selected_ids = []
    batch_entropies = []

    # Perform test-time adaptation for specified number of steps
    for j in range(args.tta_steps):
        # Forward pass
        output = model(inputs)

        # Use only confident samples for adaptation
        if selected_idx is not None:
            # Use previously selected confident samples
            output = output[selected_idx]
        else:
            # Select confident samples based on entropy
            output, selected_idx, batch_entropy = select_confident_samples(output, args.selection_p)
            if logger:
                logger.debug(f"Selected {len(selected_idx)}/{inputs.size(0)} samples for adaptation")

            # convert selected_idx to list
            selected_idx = selected_idx.tolist()
            selected_ids.append(selected_idx)
            batch_entropies.append(batch_entropy)

        # Calculate loss as average entropy (lower is better)
        if args.tpt_loss == "rtpt":
            loss = rtpt_entropy_avg(output)
        elif args.tpt_loss == "tpt":
            loss = entropy_loss_ttl(output)
        else:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

        if logger and (j == 0 or j == args.tta_steps - 1 or j % 5 == 0):
            logger.debug(f"Step {j+1}/{args.tta_steps}, Loss: {loss.item():.6f}")

        # Update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if logger:
        logger.debug(f"Completed test-time tuning with final loss: {loss.item():.6f}")

    return selected_ids, batch_entropies

def get_top_sim(sim_matrix, args):
    """
    Calculate the mean similarity of top-k most similar samples for each sample.

    Args:
        sim_matrix (torch.Tensor): Similarity matrix between samples.
        args (argparse.Namespace): Arguments containing the top_k parameter.

    Returns:
        torch.Tensor: Mean similarity scores of top-k neighbors for each sample.
    """
    # Exclude self-similarity (which is 1.0) by setting it to negative infinity
    sim_matrix[sim_matrix>=1.0] = float('-inf')
    # Get top-k similarity values for each sample
    top_k_values, _ = sim_matrix.topk(args.top_k, dim=-1) # default is 20 neighbors
    # Calculate mean similarity
    top_k_mean = top_k_values.mean(dim=-1)
    return top_k_mean

def select_confident_samples(logits, top):
    """
    Select the most confident samples based on entropy.

    Lower entropy indicates higher confidence in the prediction.

    Args:
        logits (torch.Tensor): Model output logits.
        top (float): Proportion of samples to select (0.0 to 1.0).

    Returns:
        tuple: (selected_logits, selected_indices)
    """
    # Calculate entropy for each sample in the batch
    batch_entropy = entropy_of_each_sample(logits)
    # Select indices of samples with lowest entropy (highest confidence)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    # return batch_entropy, first detach, get to cpu and then list
    batch_entropy_cpu = batch_entropy.detach().cpu().tolist()
    return logits[idx], idx, batch_entropy_cpu


def create_log_name(args):
    """Creates a standardized log name from experiment parameters"""
    # R-TPT hyperparameters for log name
    if args.tta_steps > 0:
        if args.counter_attack:
            log_name = f"ADV_eps_{args.eps}_steps_{args.steps}_counter_attack_eps_{args.counter_attack_eps}_steps_{args.counter_attack_steps}_alpha_{args.counter_attack_alpha}_tau_thres_{args.counter_attack_tau_thres}_beta_{args.counter_attack_beta}_weighted_perturbations_{args.counter_attack_weighted_perturbations}_TPT_loss_{args.tpt_loss}_lr_{args.lr}_step_{args.tta_steps}_selection_{args.selection_p}_weighted_ensemble_{args.ensemble_type}_topk_neighbours_{args.top_k}_sftemp_{args.softmax_temp}"
        else:
            log_name = f"ADV_eps_{args.eps}_steps_{args.steps}_TPT_loss_{args.tpt_loss}_lr_{args.lr}_step_{args.tta_steps}_selection_{args.selection_p}_weighted_ensemble_{args.ensemble_type}_topk_neighbours_{args.top_k}_sftemp_{args.softmax_temp}"
    else:
        if args.counter_attack:
            log_name = f"ADV_eps_{args.eps}_steps_{args.steps}_counter_attack_eps_{args.counter_attack_eps}_steps_{args.counter_attack_steps}_alpha_{args.counter_attack_alpha}_tau_thres_{args.counter_attack_tau_thres}_beta_{args.counter_attack_beta}_weighted_perturbations_{args.counter_attack_weighted_perturbations}_no_TPT_weighted_ensemble_{args.ensemble_type}_topk_neighbours_{args.top_k}_sftemp_{args.softmax_temp}"
        else:
            log_name = f"ADV_eps_{args.eps}_steps_{args.steps}_no_TPT_weighted_ensemble_{args.ensemble_type}_topk_neighbours_{args.top_k}_sftemp_{args.softmax_temp}"
    return log_name


def create_log_dir(args):
    """Creates a structured log path and filename from experiment parameters"""

    # Root: adversarial or clean
    data_type = "Adversarial" if args.eps > 0 else "Clean"

    # Counter-attack or not

    counter_attack_part = [f"Counter_Attack", f"Eps_{args.counter_attack_eps}_Steps_{args.counter_attack_steps}_Alpha_{args.counter_attack_alpha}",
        f"tau_{args.counter_attack_tau_thres}_beta_{args.counter_attack_beta}_weighted_pertrubation_{args.counter_attack_weighted_perturbations}"
    ]
    if args.counter_attack:
        if args.counter_attack_type == "pgd":
            counter_attack_part = [f"Counter_Attack",
                                   f"Eps_{args.counter_attack_eps}_Steps_{args.counter_attack_steps}_Alpha_{args.counter_attack_alpha}",
                                   f"tau_{args.counter_attack_tau_thres}_beta_{args.counter_attack_beta}_weighted_pertrubation_{args.counter_attack_weighted_perturbations}"
                                   ]
        elif args.counter_attack_type == "pgd_clip_pure_i":
            counter_attack_part = [f"Counter_Attack_PGDCLIPPureImage",
                                   f"Eps_{args.counter_attack_eps}_Steps_{args.counter_attack_steps}_Alpha_{args.counter_attack_alpha}_textembed_{args.pgd_clip_pure_i_text_embeddings}",
                                   ]

        elif args.counter_attack_type == "pgd_clip_pure_i":
            counter_attack_part = [f"Counter_Attack_PGDCounter_CLIPPureImage",
                                   f"Eps_{args.counter_attack_eps}_Steps_{args.counter_attack_steps}_Alpha_{args.counter_attack_alpha}_textembed_{args.pgd_clip_pure_i_text_embeddings}",
                                   f"tau_{args.counter_attack_tau_thres}_beta_{args.counter_attack_beta}_weighted_pertrubation_{args.counter_attack_weighted_perturbations}",
                                   f"loss_lamda_{args.pgd_counter_and_clipure_i_lamda}"

                                   ]

    else:
        counter_attack_part = ["No_Counter_Attack"]

    # TPT or no-TPT
    tpt_part = [f"TPT", f"Optimization_Loss_{args.tpt_loss}_LR_{args.lr}_Optimization_Steps_{args.tta_steps}_View_Selection_Fraction_{args.selection_p}"] if args.tta_steps > 0 else ["No_TPT"]


    # Ensemble details
    if args.ensemble_type == "weighted_rtpt":
        ensemble_part = [f"Inference_Ensemble_{args.ensemble_type}_topk_{args.top_k}_softmaxtemp_{args.softmax_temp}"]
    else:
        ensemble_part = [f"Inference_Ensemble_{args.ensemble_type}"]

    # Adversarial-specific part (include eps and attack steps if adversarial)
    if data_type == "Adversarial":
        data_type = f"{data_type}_Eps_{args.eps}_Steps_{args.steps}" if args.eps > 0 else ""

    # Combine folder structure
    # Create a list of path parts
    path_parts = []
    path_parts.append(data_type)
    for part in counter_attack_part:
        path_parts.append(part)

    for part in tpt_part:
        path_parts.append(part)
    for part in ensemble_part:
        path_parts.append(part)

    # Join the parts to create the final folder path
    folder_path = os.path.join(*path_parts)

    # replace all the "." with "_"
    folder_path = folder_path.replace(".", "_")

    # convert the file path to be compatible with the file system of windows

    return folder_path






def main():
    # Record start time for script execution
    start_time = time.time()

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
    log_dir = create_log_dir(args)
    if args.output_dir is None or args.output_dir.lower() == "none":
        log_dir = os.path.join(args.output_dir, log_dir)
    else:
        args.log_output_dir = os.path.join(args.log_output_dir, args.arch, args.test_sets)
        # Create a log directory if it doesn't exist

        #os.makedirs(args.log_output_dir, exist_ok=True)
        log_dir = os.path.join(args.log_output_dir, log_dir)

    log_dir = handle_long_windows_path(log_dir)
    args.log_dir = log_dir
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    logger, log_file = setup_logger("log", args.log_dir, level=logging.INFO)
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


    # Freeze all parameters except prompt learner
    logger.info("Freezing all parameters except prompt learner")
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
    results = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform, logger, template_text_embeddings, class_text_embeddings)

    # Clean up to free memory
    del val_dataset, val_loader

    # Format and save results
    if args.eps <= 0:
        # Clean accuracy (no adversarial attack)
        log_msg = f"=> Acc. on testset [{dset}]: Clean Acc @1 {results[0]}/ TTA Clean Acc @1 {results[2]}, Clean Acc @5 {results[1]}/ TTA Clean Acc @5 {results[3]}"
    else:
        # Adversarial accuracy
        log_msg = f"=> Acc. on testset [{dset}]: Adv Acc @1 {results[0]}/ TTA Adv Acc @1 {results[2]}, Adv Acc @5 {results[1]}/ TTA Adv Acc @5 {results[3]}"

    # Log results
    logger.info(log_msg)

    # Calculate and log total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Total script execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")



def get_adversarial_image(image, target, attack, path, index, output_dir, logger=None, counter_atk=None):
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
        """
        ***NOTE: This conversion to PIL causes precision loss, the adversarial image will not be exactly the same as the original tensor.***
        """
        if counter_atk:
            # If using counter-attack, apply it to the loaded tensor
            adv_tensor = counter_atk(adv_tensor.unsqueeze(0), target)
            if logger:
                logger.debug(f"Applied counter-attack to loaded adversarial image with shape: {adv_tensor.shape}")

        adv_tensor = adv_tensor.squeeze(0)
        img_adv = transforms.ToPILImage()(adv_tensor)


    else:
        # Create adversarial image using attack
        adv_image = attack(image, target)
        if logger:
            logger.debug(f"Generated adversarial image with shape: {adv_image.shape}")

        if counter_atk:
            # If using counter-attack, apply it to the generated image
            adv_image = counter_atk(adv_image, target)
            if logger:
                logger.debug(f"Applied counter-attack to generated adversarial image with shape: {adv_image.shape}")


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
        # raise an error if Adversarial image is not already generated
        #raise FileNotFoundError(f"Adversarial image not found at {adv_img_path}. Please generate it first.")


    return img_adv


def plot_average_weights(weighted_scores, output_dir, logger=None, filename='average_weights_plot.png'):
    """
    Create a plot showing the average weight for each view across all samples.

    Args:
        weighted_scores (dict): Dictionary mapping sample indices to lists of weights.
        output_dir (str): Directory to save the plot.
        logger (logging.Logger, optional): Logger for logging information.
    """
    if logger:
        logger.info("Creating plot of average weights for each view")

    # Convert the dictionary values to a numpy array
    all_weights = []
    for key, values in weighted_scores.items():
        # Ensure values is a 1D array
        if isinstance(values, (list, np.ndarray)):
            all_weights.append(np.asarray(values).flatten())

    # Convert to numpy array and calculate average across all samples
    all_weights = np.array(all_weights)
    avg_weights = np.mean(all_weights, axis=0)

    # Ensure avg_weights is 1D
    avg_weights = avg_weights.flatten()

    # Create the plot
    plt.figure(figsize=(12, 6))
    x_positions = np.arange(len(avg_weights))
    plt.bar(x_positions, avg_weights)
    plt.xlabel('View Index')
    plt.ylabel('Average Weight')
    plt.title('Average Weight for Each View')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a horizontal line at the average weight value
    plt.axhline(y=np.mean(avg_weights), color='r', linestyle='-', 
                label=f'Mean: {np.mean(avg_weights):.4f}')
    plt.legend()

    # Save the plot
    plot_path = os.path.join(output_dir, filename)
    plot_path = handle_long_windows_path(plot_path)
    plt.savefig(plot_path)
    plt.close()

    if logger:
        logger.info(f"Average weights plot saved to {plot_path}")


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform, logger=None, template_text_embeddings=None, class_text_embeddings=None):
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
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    tpt1 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)  # Test-time adapted accuracy
    tpt5 = AverageMeter('TTAcc@5', ':6.2f', Summary.AVERAGE)

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
            logger.info(f"Using PGD attack with epsilon: {args.eps}, alpha: {args.alpha}, steps: {args.steps}")

    if args.counter_attack:
        # Create counter-attack with specified parameters
        if args.counter_attack_type == "pgd":
            counter_atk = torchattacks.PGDCounter(model, eps=args.counter_attack_eps / 255,
                                              alpha=args.counter_attack_alpha / 255, steps=args.counter_attack_steps,
                                              tau_thres=args.counter_attack_tau_thres, beta=args.counter_attack_beta,
                                              weighted_perturbation=args.counter_attack_weighted_perturbations)
        elif args.counter_attack_type == "pgd_clip_pure_i":
            if args.pgd_clip_pure_i_text_embeddings=="null":
                embeddings = template_text_embeddings
            elif args.pgd_clip_pure_i_text_embeddings=="class":
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
            counter_atk = torchattacks.PGDCounterClipPureImage(model, eps=args.counter_attack_eps / 255,  alpha=args.counter_attack_alpha / 255,
                                                  steps=args.counter_attack_steps, text_embeddings=embeddings, tau_thres=args.counter_attack_tau_thres, beta=args.counter_attack_beta,
                                              weighted_perturbation=args.counter_attack_weighted_perturbations, loss_lamda=args.pgd_counter_and_clipure_i_lamda)
        if logger:
            logger.info(
                f"Using counter-attack with epsilon: {args.counter_attack_eps:.6f}, alpha: {args.alpha:.6f}, steps: {args.counter_attack_steps}, "
                f"tau_thres: {args.counter_attack_tau_thres}, beta: {args.counter_attack_beta}, weighted perturbation: {args.counter_attack_weighted_perturbations}")

    end = time.time()
    # Create directory for saving adversarial images if needed
    adv_images_dir = os.path.join(args.output_dir, f"adv_images_eps_{args.eps}_alpha_{args.alpha}_steps_{args.steps}")
    if args.eps > 0.0:
        os.makedirs(adv_images_dir, exist_ok=True)
        if logger:
            logger.info(f"Using directory for adversarial images: {adv_images_dir}")

    selected_ids_dic = {}
    weighted_scores = {}
    batch_entropies_dic = {}
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
                image, target, atk, path, i, adv_images_dir, logger=logger,  counter_atk=counter_atk if args.counter_attack else None
            )

            # Apply data transformations to adversarial image
            images = data_transform(img_adv)
            images = [_.unsqueeze(0) for _ in images]

            if logger:
                logger.debug(f"Created {len(images)} augmented views of the adversarial image")

        elif args.counter_attack:
            image = images[0].cuda(args.gpu, non_blocking=True)
            if logger and i == 0:
                logger.debug(f"Original image shape: {image.shape}, target: {target.item()}")
            # Get adversarial image (either generate or load from cache)
            img_adv = counter_atk(image, target)
            img_adv = img_adv.squeeze(0)
            img_adv = transforms.ToPILImage()(img_adv)
            # Apply data transformations to adversarial image
            images = data_transform(img_adv)
            images = [_.unsqueeze(0) for _ in images]
            if logger:
                logger.debug(f"Created {len(images)} augmented views of the adversarial image")

        else:
            logger.info(f"Evaluating clean images without adversarial attack or counter-attack")

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
        if args.tta_steps > 0:
            selected_ids, batch_entropies = test_time_tuning(model, images, optimizer, scaler, args, logger)
            selected_ids_dic[i] = selected_ids
            batch_entropies_dic[i] = batch_entropies



        # Get outputs after test-time adaptation
        with torch.no_grad():
            tuned_outputs = model(images)

        # Handle different types of ensembling
        if args.ensemble_type == 'none':
            # Use only the first output (no ensembling)
            tta_output = tuned_outputs[0].unsqueeze(0)
        elif args.ensemble_type == 'vanilla':
            # Use the average of all outputs
            tta_output = torch.mean(tuned_outputs, dim=0).unsqueeze(0)
        elif args.ensemble_type == 'weighted_rtpt':

            # Calculate similarity matrix between features
            sim_matrix_images = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
            # Get top similarity scores
            score = get_top_sim(sim_matrix_images, args)
            # Calculate weights based on similarity scores
            weight = torch.nn.functional.softmax(score/args.softmax_temp, dim=-1) # softmax temperature default is 0.01
            # Store weights in the weighted_scores dictionary as a list for each sample
            weighted_scores[i] = weight.detach().cpu().tolist()
            # Weighted average of tuned outputs
            tta_output = torch.bmm(weight.unsqueeze(-1).transpose(1, 2), tuned_outputs.unsqueeze(0)).squeeze(1)
        else:
            raise ValueError(f"Unknown ensemble type: {args.ensemble_type}")



        # Measure accuracy
        acc1, acc5 = accuracy(clip_output, target, topk=(1, 5))  # Original model accuracy
        tpt_acc1, tpt_acc5 = accuracy(tta_output, target, topk=(1, 5))  # Test-time adapted accuracy

        # Update accuracy metrics
        top1.update(acc1[0], images.size(0))
        tpt1.update(tpt_acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        tpt5.update(tpt_acc5[0], images.size(0))

        if logger and (i < 5 or i % 20 == 0):  # Log detailed info for first few samples and periodically
            if args.eps <= 0:
                logger.debug(f"Sample {i+1}: Original Model  Acc@1: {acc1[0].item():.2f}, Acc@5: {acc5[0].item():.2f}, TTA Acc@1: {tpt_acc1[0].item():.2f}, Acc@5: {tpt_acc5[0].item():.2f}")
            else:
                logger.debug(f"Sample {i+1}: Original Model Adversarial Acc@1: {acc1[0].item():.2f}, Acc@5: {acc5[0].item():.2f} TTA adversarial Acc@1: {tpt_acc1[0].item():.2f}, Acc@5: {tpt_acc5[0].item():.2f}")

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log progress
        if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
            if logger:
                if args.eps <= 0:
                    logger.info(f'iter:{i+1}/{len(val_loader)}, clip_acc1={top1.avg}, tta_acc1={tpt1.avg}')
                    logger.info(f'iter:{i+1}/{len(val_loader)}, Original Acc@1: {top1.avg:.2f}, Acc@5: {top5.avg:.2f}, TTA Acc@1: {tpt1.avg:.2f}, Acc@5: {tpt5.avg:.2f}')
                else:
                    logger.info(f'iter:{i+1}/{len(val_loader)}, Original Adv Acc@1: {top1.avg:.2f}, Acc@5: {top5.avg:.2f}, TTA Adv Acc@1: {tpt1.avg:.2f}, Acc@5: {tpt5.avg:.2f}')
            progress.display(i)

    # Display final summary
    progress.display_summary()

    if logger:
        if args.eps <= 0:
            logger.info(f"Final results - Original Acc@1: {top1.avg:.2f}, Acc@5: {top5.avg:.2f}, TTA Acc@1: {tpt1.avg:.2f}, Acc@5: {tpt5.avg:.2f}")
            logger.info(f"Improvement from TTA in Acc@1 {tpt1.avg - top1.avg:.2f}, and Acc@5 {tpt5.avg - top5.avg:.2f}")
        else:
            logger.info(f"Final results - Adversarial Acc@1: {top1.avg:.2f}, Acc@5: {top5.avg:.2f}, TTA Adversarial Acc@1: {tpt1.avg:.2f}, Acc@5: {tpt5.avg:.2f}")
            logger.info(f"Improvement from TTA in Adversarial Acc@1 {tpt1.avg - top1.avg:.2f}, and Acc@5 {tpt5.avg - top5.avg:.2f}")

    if args.tta_steps > 0:

        # Loop through selected IDs and check how many times entry 0 is in the values
        count = 0
        for key, values in selected_ids_dic.items():
            ids = values
            for value in values:
                if 0 in value:
                    count += 1
        logger.info(f"Number of selected samples with index 0: {count}")
        # log number of samples in the dataset, which is equal to number of keys
        logger.info(f"Number of samples in the dataset: {len(selected_ids_dic)}")

        # Save selected IDs to a file
        selected_ids_path = os.path.join(args.log_dir, args.selected_id_name)
        selected_ids_path = handle_long_windows_path(selected_ids_path)

        with open(selected_ids_path, 'w') as f:
            json.dump(selected_ids_dic, f, indent=4)
        logger.info(f"Selected IDs saved to {selected_ids_path}")

        # Save batch entropies to a file
        batch_entropies_path = os.path.join(args.log_dir, args.batch_entropy_name)
        # Handle long paths on Windows
        batch_entropies_path = handle_long_windows_path(batch_entropies_path)

        with open(batch_entropies_path, 'w') as f:
            json.dump(batch_entropies_dic, f, indent=4)
        logger.info(f"Batch entropies saved to {batch_entropies_path}")

    # Save weighted scores to a file if using weighted ensembling
    if args.ensemble_type == 'weighted_rtpt' and weighted_scores:
        weighted_scores_path = os.path.join(args.log_dir, f"{args.weighted_score_name}")
        # Handle long paths on Windows
        weighted_scores_path = handle_long_windows_path(weighted_scores_path)

        with open(weighted_scores_path, 'w') as f:
            json.dump(weighted_scores, f, indent=4)
        logger.info(f"Weighted scores saved to {weighted_scores_path}")

        # Create and save a plot of average weights for each view
        plot_average_weights(weighted_scores, args.log_dir, logger, filename=f"{args.weighted_score_name[:-5]}_plot.png")


    # Return original and test-time adapted accuracies
    return [top1.avg, top5.avg, tpt1.avg, tpt5.avg]


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
    parser.add_argument('--tpt_loss', type=str, default='rtpt', choices=['rtpt', 'tpt'])
    # Add this in the "Test-time adaptation parameters" section


    # Experiment parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='output_results/ckps/rtpt',
                        help='Directory to save adv images / results')
    parser.add_argument('--log_output_dir', type=str, default="none",
                        help='Directory to save log results')

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

    parser.add_argument('--ensemble_type', default='weighted_rtpt', type=str,
                        choices=['none', 'vanilla', 'weighted_rtpt'],
                        help='Type of ensembling to use (none, vanilla, or weighted)')
    parser.add_argument('--top_k', default=20, type=int,
                        help='Number of neighbors for similarity calculation')
    parser.add_argument('--softmax_temp', default=0.01, type=float,
                        help='Temperature parameter for softmax in similarity weighting')



    # Counter-attack parameters
    parser.add_argument('--counter_attack', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--counter_attack_type', default='pgd', choices=["pgd", "pgd_clip_pure_i", "pgd_counter_and_clipure_i"], type=str)
    parser.add_argument('--counter_attack_steps', default=2, type=int)
    parser.add_argument('--counter_attack_eps', default=4.0, type=float)
    parser.add_argument('--counter_attack_alpha', default=1.0, type=float)
    parser.add_argument('--counter_attack_tau_thres', default=0.2, type=float)
    parser.add_argument('--counter_attack_beta', default=2.0, type=float)
    parser.add_argument('--counter_attack_weighted_perturbations', default=True, type=lambda x: (str(x).lower() == 'true') )

    parser.add_argument('--pgd_clip_pure_i_text_embeddings', default='null', choices=["null", "class"], type=str)
    parser.add_argument('--pgd_counter_and_clipure_i_lamda', default=1.0, type=float)



    parser.add_argument('--selected_id_name', type=str, default='selected_topk.json',)
    parser.add_argument('--weighted_score_name', type=str, default='weighted_scores.json',)
    parser.add_argument('--batch_entropy_name', type=str, default='entropies.json',)


    # Run the main function
    main()
