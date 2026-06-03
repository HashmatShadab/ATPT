import torch
import torch.nn as nn
from typing import Dict, List

from ..attack import Attack

import torchvision.transforms.functional as TF
from PIL import Image
import math
import io
import numpy as np
import torch.nn.functional as F

class PGDCounter(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True, tau_thres=0.20))
        >>> adv_images = attack(images, labels)

    """

    FACTOR_LEVELS: Dict[str, List[float]] = {
        # Additive noise in normalized image scale [0, 1].
        # 0.00 = identity; larger = more noise.
        "gaussian_noise": [0.00, 0.01, 0.03, 0.06, 0.12, 0.18, 0.24],
        "uniform_noise": [
            0.0000,
            0.0039,
            0.0078,
            0.0157,
            0.0314,
            0.0471,
            0.0627,
            0.0784,
            0.0941,
            0.1255,
            0.1569,
            0.1882,
        ],
        # Brightness direct factor. 1.0 = identity.
        # Split into two monotonic visual directions.
        "brightness_dark":   [1.00, 0.90, 0.75, 0.60, 0.45, 0.30, 0.20, 0.10],
        "brightness_bright": [1.00,  1.25, 1.50, 1.75, 2.00, 2.50, 3.0, 3.50],

        # Contrast direct factor. 1.0 = identity.
        "contrast_low":  [1.00, 0.90, 0.75, 0.60, 0.45, 0.30, 0.20, 0.10],
        "contrast_high": [1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.0, 3.50],

        # Saturation direct factor. 1.0 = identity; 0.0 = grayscale.
        "saturation_low":  [1.00, 0.90, 0.75, 0.50, 0.25, 0.00],
        "saturation_high": [1.00, 1.50, 2.00, 2.50, 3.00, 4.00],

        # Sharpness direct factor. 1.0 = identity; 0.0 is heavily softened/blurred.
        "sharpness_low":  [1.00, 0.75, 0.50, 0.25, 0.10, 0.00],
        "sharpness_high": [1.00, 1.25, 1.50, 2.00, 3.00, 4.00],

        # Gamma direct value. 1.0 = identity.
        # In torchvision/PIL convention: gamma < 1 brightens; gamma > 1 darkens.
        "gamma_bright": [1.00, 0.90, 0.75, 0.60, 0.45, 0.30, 0.20, 0.10],
        "gamma_dark":   [1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00],

        # Hue shift. 0.0 = identity. torchvision valid range is [-0.5, 0.5].
        "hue_negative": [0.00, -0.03, -0.06, -0.10, -0.20, -0.30, -0.40, -0.50],
        "hue_positive": [0.00,  0.03,  0.06,  0.10,  0.20,  0.30, 0.40, 0.50],

        # Spatial transforms.
        # Blur sigma: 0 = identity; larger = more blur.
        # Rotation angle: 0 = identity; larger = more clockwise rotation.
        # Translation: 0 = identity; larger = more right-shift in pixels.
        "gaussian_blur": [0.00, 0.25, 0.50, 1.00, 2.00, 4.00, 6.00],
        "rotation":      [0, 15, 30, 40, 60, 90, 120],
        "translation":   [0, 2, 4, 8, 16, 32, 48, 64],

        # Quantization / compression.
        # Posterize: dropped low-order bits; 0 = identity; larger = fewer color bits.
        # Solarize: strength; 0 = identity; larger = lower threshold, more inversion.
        # Downsample: factor; 1.0 = identity; larger = stronger downsample/upsample.
        # JPEG: compression drop from quality 100; 0 = identity, larger = lower quality.
        "posterize":  [0, 1, 2, 3, 4, 5, 6, 7],
        "solarize":   [0.00, 0.10, 0.25, 0.40, 0.60, 0.80, 0.90],
        "downsample": [1.00, 1.25, 1.50, 2.00, 4.00, 8.00, 12.00],
        "jpeg":       [0, 10, 25, 40, 60, 80, 90],
    }

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, tau_thres=None, beta=None, weighted_perturbation=True, init_noise="uniform", gaussian_sigma=0.18,
                 tau_type="normal", num_anchors=10, severity=1.0):
        super().__init__("PGDCounter", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.tau_thres = tau_thres
        self.beta = beta
        self.weighted_perturbation = weighted_perturbation
        self.init_noise = init_noise
        self.gaussian_sigma = gaussian_sigma
        self.tau_type = tau_type
        self.num_anchors = num_anchors
        self.severity = severity

        # Names that mean "perturb adv_images". Everything else => adv_images = images.
        self._noise_inits = {"uniform", "gaussian"}
        # Registry of new transforms (excluding noise, which keep their old paths).
        self._transform_inits = {
            "gaussian_noise", "uniform_noise",
            "brightness_dark", "brightness_bright",
            "contrast_low", "contrast_high",
            "saturation_low", "saturation_high",
            "sharpness_low", "sharpness_high",
            "gamma_bright", "gamma_dark",
            "hue_negative", "hue_positive",
            "gaussian_blur", "rotation", "translation",
            "posterize", "solarize", "downsample", "jpeg"
        }

    def _transform_anchors(self, images, transform_type, severity, num_anchors):
        """
        Build `num_anchors * B` randomly-transformed copies of `images`.

        Returns:
            Tensor of shape [num_anchors * B, C, H, W], values in [0, 1].
            Each (anchor, image) pair gets an INDEPENDENT random parameter
            draw, so anchors are genuinely diverse.

        Conventions:
            - `severity` (sev) is now used as a direct factor for most transforms.
            - Identity values: noise=0, blur=0, rotation=0, translation=0,
              brightness/contrast/saturation/sharpness/gamma/downsample=1.0,
              hue=0.0, posterize=0, solarize=0, jpeg=0.
        """
        B, C, H, W = images.shape
        NA = num_anchors
        N = NA * B
        device = images.device
        v = float(severity)

        # Mapping of split variants to base torchvision operations.
        BASE_TRANSFORMS = {
            "brightness_dark": "brightness", "brightness_bright": "brightness",
            "contrast_low": "contrast", "contrast_high": "contrast",
            "saturation_low": "saturation", "saturation_high": "saturation",
            "sharpness_low": "sharpness", "sharpness_high": "sharpness",
            "gamma_bright": "gamma", "gamma_dark": "gamma",
            "hue_negative": "hue", "hue_positive": "hue",
        }
        base = BASE_TRANSFORMS.get(transform_type, transform_type)

        # 1) ADDITIVE PIXEL NOISE (Remains unchanged per requirement)
        if base == "gaussian_noise":
            noise = torch.randn(NA, B, C, H, W, device=device) * v
            return (images.unsqueeze(0) + noise).view(N, C, H, W).clamp(0, 1)

        if base == "uniform_noise":
            noise = (2 * torch.rand(NA, B, C, H, W, device=device) - 1) * v
            return (images.unsqueeze(0) + noise).view(N, C, H, W).clamp(0, 1)

        expanded = images.unsqueeze(0).expand(NA, -1, -1, -1, -1).reshape(N, C, H, W)

        def _per_sample(fn, params):
            out = torch.empty_like(expanded)
            for i in range(N):
                out[i] = fn(expanded[i:i + 1], params[i])[0]
            return out

        # 2) PHOTOMETRIC - Using direct factor 'v'
        if base == "brightness":
            return _per_sample(TF.adjust_brightness, [max(0.0, v)] * N).clamp(0, 1)

        if base == "contrast":
            return _per_sample(TF.adjust_contrast, [max(0.0, v)] * N).clamp(0, 1)

        if base == "saturation":
            return _per_sample(TF.adjust_saturation, [max(0.0, v)] * N).clamp(0, 1)

        if base == "hue":
            hue_shift = max(-0.5, min(0.5, v))
            return _per_sample(TF.adjust_hue, [hue_shift] * N).clamp(0, 1)

        if base == "sharpness":
            return _per_sample(TF.adjust_sharpness, [max(0.0, v)] * N).clamp(0, 1)

        if base == "gamma":
            return _per_sample(TF.adjust_gamma, [max(1e-3, v)] * N).clamp(0, 1)

        # 3) SPATIAL
        if base == "gaussian_blur":
            if v <= 0:
                return expanded.clone()
            s = v
            ks = int(2 * math.ceil(3 * s) + 1)
            ks += (ks % 2 == 0)
            ks = max(ks, 3)
            return TF.gaussian_blur(expanded, kernel_size=[ks, ks], sigma=[s, s]).clamp(0, 1)

        if base == "rotation":
            out = torch.empty_like(expanded)
            for i in range(N):
                out[i] = TF.rotate(
                    expanded[i:i + 1],
                    angle=v,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=0,
                )[0]
            return out.clamp(0, 1)

        if base == "translation":
            tx = int(round(v))
            ty = 0
            out = torch.empty_like(expanded)
            for i in range(N):
                out[i] = TF.affine(
                    expanded[i:i + 1],
                    angle=0.0,
                    translate=[tx, ty],
                    scale=1.0,
                    shear=[0.0, 0.0],
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=0,
                )[0]
            return out.clamp(0, 1)

        # 4) QUANTIZATION / COMPRESSION
        if base == "posterize":
            drop = max(0, int(round(v)))
            bits = max(1, 8 - drop)
            u8 = (expanded.clamp(0, 1) * 255.0).to(torch.uint8)
            out = torch.empty_like(u8)
            for i in range(N):
                out[i] = TF.posterize(u8[i:i + 1], bits=bits)[0]
            return out.float() / 255.0

        if base == "solarize":
            strength = max(0.0, min(1.0, v))
            threshold = int((1.0 - strength) * 255)
            u8 = (expanded.clamp(0, 1) * 255.0).to(torch.uint8)
            out = torch.empty_like(u8)
            for i in range(N):
                out[i] = TF.solarize(u8[i:i + 1], threshold=threshold)[0]
            return out.float() / 255.0

        if base == "downsample":
            f = max(1.0, v)
            if f == 1.0:
                return expanded.clone()
            out = torch.empty_like(expanded)
            nh, nw = max(1, int(H / f)), max(1, int(W / f))
            for i in range(N):
                d = F.interpolate(expanded[i:i + 1], size=(nh, nw), mode="bilinear", align_corners=False)
                out[i] = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)[0]
            return out.clamp(0, 1)

        if base == "jpeg":
            quality = max(1, min(100, 100 - int(round(v))))
            return self._jpeg_roundtrip_per_sample(expanded, [quality] * N)

        raise ValueError(f"Unknown transform type: {transform_type}")

    def _jpeg_roundtrip_per_sample(self, images, qualities):
        """Per-sample JPEG roundtrip; `qualities` is a list of length images.size(0)."""
        device = images.device
        out = []
        imgs_u8 = (images.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8)
        for img, q in zip(imgs_u8, qualities):
            pil = Image.fromarray(img.permute(1, 2, 0).numpy())
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=int(q))
            buf.seek(0)
            dec = Image.open(buf).convert("RGB")
            arr = torch.from_numpy(np.array(dec)).permute(2, 0, 1).float() / 255.0
            out.append(arr)
        return torch.stack(out, dim=0).to(device)

    @torch.no_grad()
    def compute_tau_transform(self, images, transform_type, num_anchors):
        """
        Anchor-based tau via arbitrary transforms. Computes tau for all severity levels
        defined in FACTOR_LEVELS for the given transform_type.

        Returns:
            Dict[float, torch.Tensor]: A mapping from severity level to a Tensor of shape [B].
        """
        assert images.dim() == 4, "images must be [B,C,H,W]"
        B = images.size(0)

        orig_feat = self.model(images, get_image_features=True)  # [B, D]
        orig_feat = orig_feat / orig_feat.norm(dim=-1, keepdim=True)

        severity_levels = self.FACTOR_LEVELS.get(transform_type, [self.severity])
        tau_dict = {}

        for sev in severity_levels:
            transformed = self._transform_anchors(images, transform_type, sev, num_anchors)
            # transformed: [NA*B, C, H, W]

            f = self.model(transformed, get_image_features=True)  # [NA*B, D]
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.view(num_anchors, B, -1)  # [NA, B, D]

            drift = (f - orig_feat.unsqueeze(0)).norm(dim=-1)  # [NA, B]
            tau_dict[sev] = drift.mean(dim=0)  # [B]

        return tau_dict

    def compute_tau(self, images, delta):
        # Assume model(images) returns unnormalized image features
        """
        From CounterAttack Paper
        Wuetal. [52] show that adversarial images are more vulnerable to a small noise than clean images.
        In this study, we find that adversarial images are actually more robust to small random noises, and are only
        vulnerable to sufficiently large noises, based on our analysis of adversarial images obtained by iterative attack methods
        So the diff ratio value will be lower for adversarial images comapred to clean images under small noise perturbations.

        """

        with torch.no_grad():
            orig_feat = self.model(images, get_image_features=True)  # shape [bs, feat_dim]
            noisy_feat = self.model(images + delta, get_image_features=True)
            diff_ratio = (noisy_feat - orig_feat).norm(dim=-1) / orig_feat.norm(dim=-1)  # [bs]
        return diff_ratio

    @torch.no_grad()
    def compute_tau_noisy(self, images, sigma=0.18, num_anchors=10):
        """
        Estimate the local feature-space stability of images under random Gaussian noise.

        This function computes a Monte-Carlo estimate of how sensitive the model’s
        visual representation is to small, isotropic perturbations around each
        input image. For each image x, we sample multiple Gaussian-noisy versions
        x + δ, where δ ~ N(0, σ²I), and measure how much the normalized image
        embedding changes in the model’s feature space.

        Formally, it estimates:

            τ_noisy(x) = E_{δ ~ N(0, σ²I)} [ ||  f̂(x + δ) − f̂(x) ||₂  ]

        where f̂(·) denotes the L2-normalized image embedding produced by the model.

        This quantity characterizes the *local flatness* of the representation
        manifold around an image:
          - small τ_noisy indicates that the image lies in a stable, flat region
            of feature space (robust to random noise),
          - large τ_noisy indicates a sharp or fragile region where small noise
            causes large semantic drift.

        Unlike a single-perturbation metric, this provides an intrinsic measure
        of representation robustness that is independent of any particular
        adversarial direction.

        Args:
            images (torch.Tensor): Clean input images of shape [B, C, H, W] in [0, 1].
            sigma (float): Standard deviation of the Gaussian noise used to probe
                           local stability.
            num_anchors (int): Number of noisy samples drawn per image for Monte-Carlo
                               estimation.

        Returns:
            diff_ratio_mean (torch.Tensor): Tensor of shape [B], giving the average
                                            feature-space drift τ_noisy for each image.
        """
        assert images.dim() == 4, "images must be [B,C,H,W]"
        device = images.device

        # 1️ Get base (clean) feature representation
        orig_feat = self.model(images, get_image_features=True)  # [B, feat_dim]
        orig_feat_norm = orig_feat.norm(dim=-1, keepdim=True)
        orig_feat_normalized = orig_feat / orig_feat_norm

        # 2️ Generate Gaussian noisy versions in a single batch
        B = images.size(0)
        noise_batch = sigma * torch.randn(num_anchors, B, *images.shape[1:], device=device)
        noisy_images = images.unsqueeze(0) + noise_batch  # [n_anchors, batch_size, C, H, W]

        # Reshape to [n_anchors*batch_size, C, H, W] in order to pass through the network in a single batch
        noisy_images = noisy_images.view(num_anchors * B, *images.shape[1:])  # [n_anchors*batch_size, C, H, W]


        # 3️ Compute features for all noisy samples together
        f_noisy_all  = self.model(noisy_images, get_image_features=True)  # [num_anchors*B, feat_dim]
        # Reshape back to [n_anchors, batch_size, feature_dim]
        f_noisy_all = f_noisy_all.view(num_anchors, B, -1)

        # Calculate diff_ratio between f_source_normalized and normalized f_noisy_all
        f_noisy_normalized = f_noisy_all / f_noisy_all.norm(dim=-1,
                                                            keepdim=True)  # [n_anchors, batch_size, feature_dim]
        diff_ratio = (f_noisy_normalized - orig_feat_normalized.unsqueeze(0)).norm(dim=-1) / orig_feat_normalized.norm(
            dim=-1).unsqueeze(0)  # [n_anchors, batch_size]
        diff_ratio_mean = diff_ratio.mean(dim=0)

        return diff_ratio_mean

    @torch.no_grad()
    def compute_tau_noisy_uniform(self, images, eps=0.18, num_anchors=10):
        """
        Estimate the local feature-space stability of images under uniform random noise.

        This function mirrors compute_tau_noisy, but instead of sampling Gaussian
        perturbations, it uses isotropic uniform noise. For each clean image, multiple
        noisy anchors are generated by sampling perturbations uniformly from a bounded
        range, and the average feature-space drift is computed.

        This probes whether an image lies in a locally flat or fragile region of the
        model’s representation space under general (non-Gaussian) random noise.

        Small values indicate local robustness to bounded random perturbations, while
        large values indicate high sensitivity.

        Args:
            images (torch.Tensor): Clean input images of shape [B, C, H, W] in [0,1].
            eps (float): Half-width of the uniform noise range. Noise is sampled from
                         [-eps, eps] for each pixel.
            num_anchors (int): Number of noisy samples drawn per image.

        Returns:
            diff_ratio_mean (torch.Tensor): Tensor of shape [B], giving the average
                                            uniform-noise feature drift per image.
        """
        assert images.dim() == 4, "images must be [B,C,H,W]"
        device = images.device
        B = images.size(0)

        # 1) Clean image features (normalized)
        orig_feat = self.model(images, get_image_features=True)  # [B, feat_dim]
        orig_feat = orig_feat / orig_feat.norm(dim=-1, keepdim=True)

        # 2) Generate uniform noise anchors
        noise = (2 * torch.rand(num_anchors, B, *images.shape[1:], device=device) - 1.0) * eps
        noisy_images = images.unsqueeze(0) + noise  # [n_anchors, B, C, H, W]
        noisy_images = noisy_images.view(num_anchors * B, *images.shape[1:])

        # 3) Compute noisy features
        f_noisy = self.model(noisy_images, get_image_features=True)  # [n_anchors*B, feat_dim]
        f_noisy = f_noisy.view(num_anchors, B, -1)
        f_noisy = f_noisy / f_noisy.norm(dim=-1, keepdim=True)

        # 4) Compute feature-space drift
        diff_ratio = (f_noisy - orig_feat.unsqueeze(0)).norm(dim=-1)  # [n_anchors, B]
        diff_ratio_mean = diff_ratio.mean(dim=0)

        return diff_ratio_mean

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        adv_images = images.clone().detach()

        # Get original image features
        with torch.no_grad():
            original_features = self.get_logits(images, get_image_features=True)




        if self.random_start:
            # Starting at a uniformly random point
            if self.init_noise == "uniform":
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                    -self.eps, self.eps
                )
            elif self.init_noise == "gaussian":
                sigma = self.gaussian_sigma
                noise = torch.randn_like(adv_images) * sigma
                adv_images = adv_images + noise
            elif self.init_noise in self._transform_inits:
                # New transforms: don't perturb adv_images; tau probes clean `images`.
                adv_images = images.clone().detach()
            else:
                raise ValueError(f"Unknown init_noise type: {self.init_noise}")
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        # if self.tau_type == "normal":
        #     #################################################
        #     delta_initial = adv_images - images
        #     deltas_per_step = [delta_initial.clone().detach()]
        #     diff_ratio = self.compute_tau(images, delta_initial)
        #     ################################################
        # elif self.tau_type == "noisy":
        #     tau_sigma = self.gaussian_sigma
        #     number_of_anchors = self.num_anchors
        #     diff_ratio = self.compute_tau_noisy(images, tau_sigma, number_of_anchors)
        #
        # elif self.tau_type == "normal_anchors":
        #     tau_eps = self.eps
        #     number_of_anchors = self.num_anchors
        #     diff_ratio = self.compute_tau_noisy_uniform(images, tau_eps, number_of_anchors)

        # ----- compute tau -----
        tau_dict = {}
        if self.init_noise in self._noise_inits:
            # Existing behavior for noise inits, gated by tau_type.
            if self.tau_type == "normal":
                delta_initial = adv_images - images
                deltas_per_step = [delta_initial.clone().detach()]
                diff_ratio = self.compute_tau(images, delta_initial)
            elif self.tau_type == "noisy":
                diff_ratio = self.compute_tau_noisy(images, self.gaussian_sigma, self.num_anchors)
            elif self.tau_type == "normal_anchors":
                diff_ratio = self.compute_tau_noisy_uniform(images, self.eps, self.num_anchors)
            else:
                raise ValueError(f"Unknown tau_type: {self.tau_type}")
            tau_dict[self.severity] = diff_ratio
        else:
            # New transforms: anchor-based tau on clean images, severity-driven.
            tau_dict = self.compute_tau_transform(
                images, self.init_noise, self.num_anchors
            )
        # For backward compatibility and early exit logic below, use self.severity if present in tau_dict,
        # otherwise pick the first one.
        if self.init_noise in self._noise_inits:
            diff_ratio = tau_dict[self.severity]
        elif self.severity in tau_dict:
            diff_ratio = tau_dict[self.severity]
        else:
            diff_ratio = next(iter(tau_dict.values()))

        # Convert all tensors in tau_dict to CPU items if they are scalar tensors
        # or leave them as tensors if they are batches.
        # Based on the original code returning diff_ratio.item(), we should probably 
        # return items for batch size 1.
        if images.size(0) == 1:
            tau_dict_to_return = {k: v.item() for k, v in tau_dict.items()}
        else:
            tau_dict_to_return = tau_dict

        if self.steps == 0:
            return adv_images, tau_dict_to_return

        # ---- Algorithm 1 early exit (only when batch size == 1) ----
        if (
                self.tau_thres is not None
                and images.size(0) == 1
                and diff_ratio.item() >= self.tau_thres
        ):
            # Return delta^0 only (random start), no counterattack
            return adv_images, tau_dict_to_return

        for _ in range(self.steps):
            # Create a fresh copy for gradient computation
            adv_images_for_grad = adv_images.clone().detach().requires_grad_(True)
            outputs = self.get_logits(adv_images_for_grad, get_image_features=True)

            ###########################################
            scheme_sign = (self.tau_thres - diff_ratio).sign()
            ##############################################

            # Calculate L2 loss between original and adversarial features
            l2_loss = ((((outputs - original_features)**2).sum(1))).sum()

            # For targeted attacks, we want to maximize the L2 loss
            if self.targeted:
                cost = -l2_loss
            else:
                cost = l2_loss

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images_for_grad, retain_graph=False, create_graph=False
            )[0]

            # Update using the detached gradient
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # ####################################################################
            deltas_per_step.append(delta.clone().detach())
            # ###################################################################

            # Explicitly free memory
            del outputs, l2_loss, cost, grad, delta, adv_images_for_grad
            torch.cuda.empty_cache()

        if self.tau_thres is not None and self.beta is not None:

            if self.weighted_perturbation:
                weights = torch.arange(self.steps + 1, device=self.device).unsqueeze(0).expand(images.size(0), -1)
                weights = torch.exp(scheme_sign.view(-1, 1) * weights * self.beta)
                weights = weights / weights.sum(dim=1, keepdim=True)
            else:
                weights = torch.ones(self.steps + 1, device=self.device).unsqueeze(0).expand(images.size(0), -1)
                weights = weights / weights.sum(dim=1, keepdim=True)

            weights_hard = torch.zeros_like(weights)
            weights_hard[:, 0] = 1.0

            final_weights = torch.where(scheme_sign.unsqueeze(1) > 0, weights, weights_hard)
            final_weights = final_weights.view(images.size(0), self.steps + 1, 1, 1, 1)

            Delta_stack = torch.stack(deltas_per_step, dim=1)  # [bs, steps+1, C, H, W]
            final_delta = (final_weights * Delta_stack).sum(dim=1)

            adv_images = torch.clamp(images + final_delta, min=0, max=1).detach()

        # Clean up memory
        del original_features
        torch.cuda.empty_cache()

        return adv_images, tau_dict_to_return
