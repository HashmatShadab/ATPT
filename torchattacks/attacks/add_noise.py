import torch
import torch.nn as nn

from ..attack import Attack


class AddNoise(Attack):
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

    def __init__(self, model, eps=8 / 255,  random_start=True,  init_noise="uniform", gaussian_sigma=0.18,
                 tau_type="normal", num_anchors=10):
        super().__init__("AddNoise", model)
        self.eps = eps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.init_noise = init_noise
        self.gaussian_sigma = gaussian_sigma
        self.tau_type = tau_type
        self.num_anchors = num_anchors


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
            else:
                raise ValueError(f"Unknown init_noise type: {self.init_noise}")
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        if self.tau_type == "normal":
            #################################################
            delta_initial = adv_images - images
            deltas_per_step = [delta_initial.clone().detach()]
            diff_ratio = self.compute_tau(images, delta_initial)
            ################################################
        elif self.tau_type == "noisy":
            tau_sigma = self.gaussian_sigma
            number_of_anchors = self.num_anchors
            diff_ratio = self.compute_tau_noisy(images, tau_sigma, number_of_anchors)

        elif self.tau_type == "normal_anchors":
            tau_eps = self.eps
            number_of_anchors = self.num_anchors
            diff_ratio = self.compute_tau_noisy_uniform(images, tau_eps, number_of_anchors)


        return adv_images, diff_ratio.item()

