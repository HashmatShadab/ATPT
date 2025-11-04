import torch
import torch.nn as nn

from ..attack import Attack


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

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, tau_thres=None, beta=None, weighted_perturbation=True, init_noise="uniform", gaussian_sigma=0.18,
                 tau_type="normal"):
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


    def compute_tau(self, images, delta):
        # Assume model(images) returns unnormalized image features

        with torch.no_grad():
            orig_feat = self.model(images)  # shape [bs, feat_dim]
            noisy_feat = self.model(images + delta)
            diff_ratio = (noisy_feat - orig_feat).norm(dim=-1) / orig_feat.norm(dim=-1)  # [bs]
        return diff_ratio

    @torch.no_grad()
    def compute_tau_noisy(self, images, sigma=0.18, num_anchors=10):
        """
        Compute tau using multiple Gaussian-noisy anchors.
        Used when self.tau_type == 'noisy'.

        Args:
            images (torch.Tensor): Clean images [B, C, H, W], values in [0,1].
            sigma (float): Standard deviation for Gaussian noise.
            num_anchors (int): Number of noisy samples per image.

        Returns:
            diff_ratio_mean (torch.Tensor): [B] mean tau per image.
            diff_ratio_all (torch.Tensor): [num_anchors, B] per-anchor tau (optional, for analysis).
        """
        assert images.dim() == 4, "images must be [B,C,H,W]"
        device = images.device

        # 1️ Get base (clean) feature representation
        orig_feat = self.model(images)  # [B, feat_dim]
        orig_feat_norm = orig_feat.norm(dim=-1, keepdim=True)
        orig_feat_normalized = orig_feat / orig_feat_norm

        # 2️ Generate Gaussian noisy versions in a single batch
        B = images.size(0)
        noise_batch = sigma * torch.randn(num_anchors, B, *images.shape[1:], device=device)
        noisy_images = images.unsqueeze(0) + noise_batch  # [n_anchors, batch_size, C, H, W]

        # Reshape to [n_anchors*batch_size, C, H, W] in order to pass through the network in a single batch
        noisy_images = noisy_images.view(num_anchors * B, *images.shape[1:])  # [n_anchors*batch_size, C, H, W]


        # 3️ Compute features for all noisy samples together
        f_noisy_all  = self.model(noisy_images)  # [num_anchors*B, feat_dim]
        # Reshape back to [n_anchors, batch_size, feature_dim]
        f_noisy_all = f_noisy_all.view(num_anchors, B, -1)

        # Calculate diff_ratio between f_source_normalized and normalized f_noisy_all
        f_noisy_normalized = f_noisy_all / f_noisy_all.norm(dim=-1,
                                                            keepdim=True)  # [n_anchors, batch_size, feature_dim]
        diff_ratio = (f_noisy_normalized - orig_feat_normalized.unsqueeze(0)).norm(dim=-1) / orig_feat_normalized.norm(
            dim=-1).unsqueeze(0)  # [n_anchors, batch_size]
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
            number_of_anchors = 10
            diff_ratio = self.compute_tau_noisy(images, tau_sigma, number_of_anchors)


        if self.steps == 0:
            return adv_images

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

        return adv_images
