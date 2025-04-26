import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import fewshot_datasets, build_fewshot_dataset
import data.augmix_ops as augmentations

import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np


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


ID_to_DIRNAME={
    'I': 'imagenet/images',
    'A': 'imagenet-adversarial/imagenet-a',
    'K': 'imagenet-sketch/images',
    'R': 'imagenet-rendition/imagenet-r',
    'V': 'imagenetv2/imagenetv2-matched-frequency-format-val',
    'flower102': 'oxford_flowers',
    'dtd': 'dtd',
    'pets': 'oxford_pets',
    'cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}

class ImageFolder_path(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform,
    ):
        super().__init__(
            root=root,
            transform=transform
        )
        self.imgs = self.samples


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, torch.tensor(target).long(), path

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = ImageFolder_path(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = ImageFolder_path(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    else:
        raise NotImplementedError

    return testset

# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    """
    Perform AugMix-style augmentation on the input image.

    AugMix is a data augmentation technique designed to improve model robustness.
    It works by mixing the original image with several augmented versions using random
    augmentations from a provided list, blending them together with random weights.

    Process:
    1. Apply a fixed pre-augmentation (e.g., random resized crop, random flip) to the input image.
    2. Convert the pre-augmented image into a tensor using the given `preprocess` transform.
    3. If no augmentations are provided (aug_list is empty), return the processed original image directly.
    4. Otherwise:
       - Sample random weights `w` from a Dirichlet distribution to mix three augmentation paths.
        - For each of the 3 augmentation paths:
           a. Start from the pre-augmented image.
           b. Apply 1 to 3 random augmentations (randomly chosen from `aug_list`).
           c. Convert the augmented image to a tensor.
           d. Multiply the tensor by its respective weight `w[i]` and accumulate.

            - Each weight controls how much influence each augmentation branch gets when combining them.
            - Because the sampling is random every time:
            - Sometimes one augmentation dominates,
            - Sometimes two augmentations contribute equally,
            - Sometimes all three are mixed almost equally.

       - Sample a mixing factor `m` from a Beta distribution to mix the original image with the augmented mix.
            It controls the final blend between: the original (unaugmented) image and the mixture of augmentations.

    5. Finally, mix the original processed image and the augmented mixture using the mixing factor `m`.
    6. Return the final mixed image tensor.

    Args:
        image (PIL.Image): The input PIL image to be augmented.
        preprocess (callable): A preprocessing function (e.g., transforms like `ToTensor()`).
        aug_list (list): A list of augmentation functions, each taking (image, severity) as input.
        severity (int, optional): The strength of the augmentation to apply (default is 1).

    Returns:
        torch.Tensor: The final mixed tensor ready for model input.
    """

    # Step 1: Apply fixed pre-augmentations (e.g., crop and flip)
    preaugment = get_preaugment()
    x_orig = preaugment(image)

    # Step 2: Process pre-augmented image into a tensor
    x_processed = preprocess(x_orig)

    # Step 3: If no augmentations are provided, return the processed image
    if len(aug_list) == 0:
        return x_processed

    # Step 4: Sample mixing weights and mixing coefficient
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))  # Random weights for three augmentation paths
    m = np.float32(np.random.beta(1.0, 1.0))              # Random mixing factor between original and augmentations

    # Step 5: Initialize empty tensor for mixing
    mix = torch.zeros_like(x_processed)

    # Step 6: Build the mixed augmentation
    for i in range(3):   # For each of the three augmentation chains
        x_aug = x_orig.copy()   # Start from the pre-augmented image

        # Apply 1 to 3 random augmentations sequentially
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)   # Randomly select an augmentation function and  apply it with given severity

        # Process the augmented image into tensor and add weighted contribution
        mix += w[i] * preprocess(x_aug)

    # Step 7: Final mix between original and augmented versions
    mix = m * x_processed + (1 - m) * mix

    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1, only_base_image=False):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        self.only_base_image = only_base_image

    def __call__(self, x):

        """
         Transforms the input image and optionally generates augmented views.

         Args:
             x (PIL.Image or np.ndarray): Input image.

         Returns:
             List[torch.Tensor]: A list containing:
                 - Only the base preprocessed image if `only_base_image` is True.
                 - Otherwise, the base image followed by `n_views` AugMix-augmented versions.
        """

        image = self.preprocess(self.base_transform(x))

        if self.only_base_image:
            return image
        else:
            # Generate augmented views
            views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
            return [image] + views


class Post_AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity

    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views
