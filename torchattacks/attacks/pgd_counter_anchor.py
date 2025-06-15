import torch
import torch.nn as nn

from ..attack import Attack


class PGDCounterAnchor(Attack):
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
        tau_thres (float): threshold for tau computation. (Default: None)
        beta (float): parameter for weighted perturbation. (Default: None)
        weighted_perturbation (bool): whether to use weighted perturbation. (Default: True)
        noise_count (int): number of noisy samples to generate. (Default: 10)
        noise_sigma (float): standard deviation of Gaussian noise. (Default: 0.1)
        direction_weight (float): weight for directional loss component. Higher values emphasize 
                                 pushing features in the direction of original features. (Default: 1.0)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True, tau_thres=0.20, direction_weight=2.0))
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, tau_thres=None, beta=None, weighted_perturbation=True, noise_count=10, noise_sigma=0.1, direction_weight=1.0):
        super().__init__("PGDCounterAnchor", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.tau_thres = tau_thres
        self.beta = beta
        self.weighted_perturbation = weighted_perturbation
        self.noise_count = noise_count
        self.noise_sigma = noise_sigma
        self.direction_weight = direction_weight

    def compute_tau(self, images, delta):
        # Assume model(images) returns unnormalized image features
        with torch.no_grad():
            orig_feat = self.model(images)  # shape [bs, feat_dim]
            noisy_feat = self.model(images + delta)
            diff_ratio = (noisy_feat - orig_feat).norm(dim=-1) / orig_feat.norm(dim=-1)  # [bs]
        return diff_ratio

    def apply_gaussian_noise(self, images):
        # Apply Gaussian noise to the images
        noise = torch.randn_like(images) * self.noise_sigma
        return torch.clamp(images + noise, min=0, max=1)




    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        adv_images = images.clone().detach()

        # Get original image features with Gaussian noise
        with torch.no_grad():
            # Apply Gaussian noise n times and get features
            noisy_features_list = []
            for _ in range(self.noise_count):
                noisy_images = self.apply_gaussian_noise(images)
                noisy_features = self.get_logits(noisy_images, get_image_features=True)
                noisy_features_list.append(noisy_features)

            # Compute mean of features from noisy images
            original_features = torch.stack(noisy_features_list).mean(dim=0)

        # Initialize delta for tau computation
        delta_initial = torch.zeros_like(images)
        diff_ratio = self.compute_tau(images, delta_initial)




        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        #################################################
        delta_initial = adv_images - images
        deltas_per_step = [delta_initial.clone().detach()]
        diff_ratio = self.compute_tau(images, delta_initial)
        ################################################

        if self.steps == 0:
            return adv_images

        for _ in range(self.steps):
            # Create a fresh copy for gradient computation
            adv_images_for_grad = adv_images.clone().detach().requires_grad_(True)
            outputs = self.get_logits(adv_images_for_grad, get_image_features=True)

            ###########################################
            scheme_sign = (self.tau_thres - diff_ratio).sign()
            ##############################################

            # Calculate feature difference vector (direction from adversarial outputs to original features)
            # This vector points from the current adversarial features towards the original features
            feature_diff = original_features - outputs

            # Normalize the direction vector to get a unit vector
            # Adding a small epsilon (1e-8) to avoid division by zero
            feature_diff_norm = feature_diff / (feature_diff.norm(dim=1, keepdim=True) + 1e-8)

            # Calculate L2 loss between original and adversarial features
            # This is the standard feature distance minimization objective
            l2_loss = ((((outputs - original_features)**2).sum(1))).sum()

            # Calculate directional loss (dot product of feature difference and its normalized direction)
            # The negative sign encourages maximizing this dot product, which means
            # pushing the adversarial features in the direction of original features
            # This is different from just minimizing distance, as it explicitly considers direction
            directional_loss = -torch.sum(feature_diff_norm * feature_diff)

            # Combine both losses with weighting controlled by direction_weight parameter
            # Higher direction_weight values emphasize directional guidance over distance minimization
            combined_loss = l2_loss + self.direction_weight * directional_loss

            # For targeted attacks, we want to maximize the L2 loss
            if self.targeted:
                cost = -combined_loss
            else:
                cost = combined_loss

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images_for_grad, retain_graph=False, create_graph=False
            )[0]

            # Update using the detached gradient
            adv_images = adv_images.detach() - self.alpha * grad.sign()
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
