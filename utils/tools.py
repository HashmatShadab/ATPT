"""
Utility functions and classes for model training, evaluation, and visualization.

This module provides common utilities used across the project, including:
- Metrics tracking and reporting
- Model checkpoint loading
- Evaluation functions
- Reproducibility utilities
"""

import os
import time
import random

import numpy as np

import shutil
from enum import Enum

import torch
import torchvision.transforms as transforms


def set_random_seed(seed):
    """
    Set random seeds for reproducibility across all random number generators.

    Args:
        seed (int): The random seed to use.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Summary(Enum):
    """
    Enumeration of summary types for metrics reporting.

    Attributes:
        NONE (int): No summary.
        AVERAGE (int): Report average of values.
        SUM (int): Report sum of values.
        COUNT (int): Report count of values.
    """
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.

    This class provides functionality to track metrics during training or evaluation,
    including current value, running average, sum, and count.

    Args:
        name (str): Name of the metric to display.
        fmt (str, optional): Display format string. Defaults to ':f'.
        summary_type (Summary, optional): Type of summary to use. Defaults to Summary.AVERAGE.
    """
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        """Reset all statistics to zero."""
        self.val = 0  # Current value
        self.avg = 0  # Running average
        self.sum = 0  # Running sum
        self.count = 0  # Number of updates

    def update(self, val, n=1):
        """
        Update statistics with new value.

        Args:
            val (float): Value to update with.
            n (int, optional): Number of items this value represents. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation showing current value and average."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        """
        Generate summary string based on the specified summary type.

        Returns:
            str: Formatted summary string.
        """
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Display progress of training or evaluation with metrics.

    This class provides functionality to display batch progress and metrics
    during training or evaluation loops.

    Args:
        num_batches (int): Total number of batches.
        meters (list): List of AverageMeter instances to display.
        prefix (str, optional): Prefix string for the output. Defaults to "".
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """
        Display current batch progress and metrics.

        Args:
            batch (int): Current batch index.
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        """Display summary of all metrics at the end of an epoch."""
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        """
        Create a format string for displaying batch progress.

        Args:
            num_batches (int): Total number of batches.

        Returns:
            str: Format string for batch display.
        """
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """
    Compute the top-k accuracy for the given predictions and targets.

    Args:
        output (torch.Tensor): Model predictions, shape [batch_size, num_classes].
        target (torch.Tensor): Ground truth labels, shape [batch_size].
        topk (tuple, optional): Tuple of k values for which to compute accuracy. Defaults to (1,).

    Returns:
        list: List of top-k accuracies as percentages, one for each k in topk.
    """
    with torch.no_grad():
        maxk = max(topk)  # Maximum k value
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # Transpose to shape [k, batch_size]

        # Compare predictions to targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # Calculate accuracies for each k
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))  # Convert to percentage
        return res


def load_model_weight(load_path, model, device, args):
    """
    Load model weights from a checkpoint file.

    This function loads model weights from a checkpoint, handling various edge cases
    such as token vectors and device mapping.

    Args:
        load_path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model to load weights into.
        device (str or torch.device): Device to load the weights onto.
        args (argparse.Namespace): Arguments containing start_epoch and other parameters.

    Returns:
        None
    """
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint['state_dict']

        # Ignore fixed token vectors (for prompt tuning models)
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        # Set starting epoch from checkpoint
        args.start_epoch = checkpoint['epoch']

        # Get best accuracy from checkpoint, or default to 0
        try:
            best_acc1 = checkpoint['best_acc1']
        except:
            best_acc1 = torch.tensor(0)

        # Move best accuracy to the correct device
        if device != 'cpu':
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(device)

        # Load state dict into model
        try:
            model.load_state_dict(state_dict)
        except:
            # Fallback for prompt generator models
            model.prompt_generator.load_state_dict(state_dict, strict=False)

        print("=> loaded checkpoint '{}' (epoch {})".format(load_path, checkpoint['epoch']))

        # Clean up to free memory
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> no checkpoint found at '{}'".format(load_path))


def validate(val_loader, model, criterion, args, output_mask=None):
    """
    Validate model performance on a validation dataset.

    This function evaluates a model on a validation dataset, computing loss and accuracy
    metrics. It supports optional output masking for specific class subsets.

    Args:
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        model (torch.nn.Module): Model to evaluate.
        criterion (callable): Loss function.
        args (argparse.Namespace): Arguments containing GPU and print frequency settings.
        output_mask (list or torch.Tensor, optional): Mask for output classes. Defaults to None.

    Returns:
        torch.Tensor: Top-1 accuracy on the validation set.
    """
    # Initialize metrics tracking
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    # Set up progress display
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # Switch to evaluation mode
    model.eval()

    # Evaluate without gradient computation
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # Move data to GPU if available
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # Compute model output and loss
            with torch.cuda.amp.autocast():
                output = model(images)
                # Apply output mask if provided (for subset of classes)
                if output_mask:
                    output = output[:, output_mask]
                loss = criterion(output, target)

            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Display progress at specified intervals
            if i % args.print_freq == 0:
                progress.display(i)

        # Display final summary
        progress.display_summary()

    # Return top-1 accuracy
    return top1.avg
