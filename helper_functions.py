import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np


def handle_long_windows_path(file_path):
    """
    Handles long file paths in Windows by adding the extended path prefix if needed.

    Args:
        file_path (str): The original file path

    Returns:
        str: Modified file path with extended path prefix if needed for Windows
    """
    if os.name == 'nt' and len(file_path) > 260:
        # Use the \\?\ prefix to bypass the 260 character limitation
        if not file_path.startswith('\\\\?\\'):
            file_path = f"\\\\?\\{os.path.abspath(file_path)}"
    else:
        file_path = file_path

    return file_path

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
            output, selected_idx = select_confident_samples(output, args.selection_p)
            if logger:
                logger.debug(f"Selected {len(selected_idx)}/{inputs.size(0)} samples for adaptation")

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

    return


def plot_image(imgs, title=None, nrow=4, figsize=(12, 8)):
    """
    Plots a single image or a list of images in a grid.

    Args:
        imgs: Single torch.Tensor, PIL.Image, or numpy.ndarray, or a list of them.
        title: Optional string for the plot title.
        nrow: Number of images per row if plotting a list.
        figsize: Size of the figure for multiple images.
    """

    def prepare(img):
        """Helper to convert Tensor/PIL/numpy to numpy format for plotting."""
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            img = img.numpy()
        elif isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        return img

    if isinstance(imgs, list):
        imgs = [prepare(img) for img in imgs]
        n_imgs = len(imgs)
        ncols = nrow
        nrows = (n_imgs + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            if idx < n_imgs:
                ax.imshow(imgs[idx])
                ax.axis('off')
            else:
                ax.remove()  # Remove extra empty subplots

        if title:
            fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    else:
        img = prepare(imgs)
        plt.imshow(img)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.show()




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
        # Change the extension to png
        img_filename = os.path.splitext(img_filename)[0] + ".png"
        parent_folder_name = os.path.basename(os.path.dirname(path[0]))
        adv_img_path = os.path.join(output_dir, f"{parent_folder_name}_{img_filename}")
    else:
        # If path is not available, use index as identifier
        adv_img_path = os.path.join(output_dir, f"{index}.png")

    # Check if adversarial image already exists
    if os.path.exists(adv_img_path):
        if logger:
            logger.info(f"Loading existing adversarial image from {adv_img_path}")
        # Load existing adversarial image
        img_adv = Image.open(adv_img_path).convert('RGB')
    else:
        # Create adversarial image using attack
        adv_image = attack(image, target)
        if logger:
            logger.info(f"Generated adversarial image with shape: {adv_image.shape}")

        img_adv = transforms.ToPILImage()(adv_image.squeeze(0))
        # Save the adversarial image
        img_adv.save(adv_img_path)
        if logger:
            logger.info(f"Saved adversarial image to {adv_img_path}")

        # Free memory for large datasets
        del adv_image
        torch.cuda.empty_cache()

    return img_adv



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
        List[PIL.Image.Image]: List of adversarial images.
    """
    batch_size = images.size(0)
    adv_images = []

    # Check if any adversarial image is missing and generate the attack
    generate_attack = False
    for i in range(batch_size):
        img_filename = os.path.basename(paths[i])
        img_filename = os.path.splitext(img_filename)[0] + ".png"
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
            img_adv = transforms.ToPILImage()(adv_images[i].cpu())
            img_filename = os.path.basename(paths[i])
            # change the extension to png
            img_filename = os.path.splitext(img_filename)[0] + ".png"
            parent_folder_name = os.path.basename(os.path.dirname(paths[i]))
            adv_img_path = os.path.join(output_dir, f"{parent_folder_name}_{img_filename}")

            img_adv.save(adv_img_path)
            if logger:
                logger.info(f"Batch:[{index}] Image: [{i}] Saved adversarial image to {adv_img_path}")

        # Free memory after processing the batch
        del adv_images
        torch.cuda.empty_cache()
    else:
        logger.info(f"Batch:[{index}] Adversarial images for this batch already exist")


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

def print_args(args):
    """
    Format command line arguments for printing.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        str: Formatted string of all arguments.
    """
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += f"{arg}:{content}\n"
    return s




def entropy_of_each_sample(outputs):
    """
    Calculate entropy for each sample in the batch.

    Args:
        outputs (torch.Tensor): Model output logits.

    Returns:
        torch.Tensor: Entropy for each sample.
    """
    return -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)

def rtpt_entropy_avg(outputs):
    """
    Calculate the average entropy of model outputs.

    Args:
        outputs (torch.Tensor): Model output logits.

    Returns:
        torch.Tensor: Mean entropy across all samples.
    """
    # Calculate entropy for each sample and return mean
    return entropy_of_each_sample(outputs).mean()

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
    return logits[idx], idx


import torch

###########################################################################################################################
def compute_expected_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes Expected Entropy: E_i [H(P_i)].

    Args:
        logits (torch.Tensor): Logits tensor of shape (N, C).

    Returns:
        torch.Tensor: Scalar tensor, expected entropy (data uncertainty).
    """
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)  # [N, C]
    entropy_per_sample = -(log_probs.exp() * log_probs).sum(dim=-1)  # [N]
    expected_entropy = entropy_per_sample.mean()
    return expected_entropy

def compute_expected_kl_divergence(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes Expected KL Divergence: E_i [ KL(E[P_i] || P_i) ],
    which corresponds to Epistemic Uncertainty.

    Args:
        logits (torch.Tensor): Logits tensor of shape (N, C).

    Returns:
        torch.Tensor: Scalar tensor, expected KL divergence.
    """
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)  # [N, C]
    probs = log_probs.exp()  # [N, C]

    # Compute average probability across samples
    avg_probs = probs.mean(dim=0, keepdim=True)  # [1, C]
    avg_log_probs = avg_probs.log()  # [1, C]

    # KL divergence for each sample: KL(probs[i] || avg_probs)

    kl_div_per_sample = (probs * (probs / avg_probs).log()).sum(dim=-1)

    # Average over all samples
    expected_kl_divergence = kl_div_per_sample.mean()

    return expected_kl_divergence

def compute_uncertainty_decomposition(logits: torch.Tensor) -> dict:
    """
    Computes both uncertainty components separately:
    - Expected Entropy (Data Uncertainty)
    - Expected KL Divergence (Epistemic Uncertainty)

    Args:
        logits (torch.Tensor): Logits tensor of shape (N, C).

    Returns:
        dict: Dictionary with 'expected_entropy', 'expected_kl', and 'total_uncertainty'.
    """
    expected_entropy = compute_expected_entropy(logits)
    expected_kl = compute_expected_kl_divergence(logits)
    total_uncertainty = expected_entropy + expected_kl

    return {
        'expected_entropy': expected_entropy,
        'expected_kl': expected_kl,
        'total_uncertainty': total_uncertainty
    }


def entropy_loss_ttl(outputs: torch.Tensor) -> torch.Tensor:
    """
    Computes total uncertainty H[E(P_i)] directly.

    Args:
        outputs (torch.Tensor): Logits tensor of shape (N, C).

    Returns:
        torch.Tensor: Scalar tensor, total uncertainty.
    """
    # Convert to log-probabilities
    log_probs = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # [N, C]
    probs = log_probs.exp()  # [N, C]

    # Average probabilities
    avg_probs = probs.mean(dim=0, keepdim=True)  # [1, C]

    # Compute entropy of averaged probability
    avg_log_probs = avg_probs.log()  # [1, C]
    entropy = -(avg_probs * avg_log_probs).sum(dim=-1)  # [1]

    return entropy.squeeze(0)  # Make it scalar


############################################################################################################################



def test_uncertainty_functions():
    """
    Run test cases to evaluate different uncertainty calculation functions.

    This function creates various test tensors and runs them through the 
    uncertainty calculation functions to demonstrate their behavior.
    """
    print("\n===== TESTING UNCERTAINTY CALCULATION FUNCTIONS =====\n")

    # Create test tensors with different characteristics

    # Case 1: Highly confident predictions (low entropy)
    logits = torch.tensor([
        [10.0, 0.1, 0.1, 0.1],  # Very confident in first class
        [0.1, 9.0, 0.1, 0.1],   # Very confident in second class
        [0.1, 0.1, 8.0, 0.1]    # Very confident in third class
    ])
    print(f"Input logits shape:{logits.shape}")
    print(f"Input logits:\n {logits}")
    # Test calculate_entropy
    entropy = calculate_entropy(logits)
    print(f"\nEntropy per sample: {entropy}")

    # Test entropy_avg
    avg_entropy = entropy_avg(logits)
    print(f"Average entropy: {avg_entropy.item():.6f}")

    # Test compute_expected_kl_divergence
    expected_kl = compute_expected_kl_divergence(logits)
    print(f"Expected KL divergence (epistemic uncertainty): {expected_kl.item():.6f}")


    # Total loss
    print(f"Total Loss: {avg_entropy.item() + expected_kl.item():.6f}")

    entropy_ttl = entropy_loss_ttl(logits)
    print(f"Entropy TTL: {entropy_ttl.item():.6f}")





# Run the tests if this file is executed directly
if __name__ == "__main__":
    test_uncertainty_functions()
