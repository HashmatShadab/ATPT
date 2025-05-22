import torch
import torch.nn as nn

from ..attack import Attack
from torch.nn.functional import cosine_similarity


class PGDCounterClipPureImage(Attack):
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

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True,
                 text_embeddings=None, tau_thres=None, beta=None, weighted_perturbation=True, loss_lamda=1.0):
        super().__init__("PGDCounterClipPureImage", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.text_embeddings = text_embeddings
        self.tau_thres = tau_thres
        self.beta = beta
        self.weighted_perturbation = weighted_perturbation
        self.loss_lamda = loss_lamda


    def compute_tau(self, images, delta):
        # Assume model(images) returns unnormalized image features

        with torch.no_grad():
            orig_feat = self.model(images)  # shape [bs, feat_dim]
            noisy_feat = self.model(images + delta)
            diff_ratio = (noisy_feat - orig_feat).norm(dim=-1) / orig_feat.norm(dim=-1)  # [bs]
        return diff_ratio


    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        adv_images = images.clone().detach()

        text_embed = self.text_embeddings.mean(dim=1)
        text_embed = text_embed.repeat(images.shape[0], 1).to(self.device)

        # Get original image features
        with torch.no_grad():
            original_features = self.get_logits(images, get_image_features=True)


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
            outputs = self.get_logits(adv_images_for_grad, get_image_features=True, normalize=True)

            ###########################################
            scheme_sign = (self.tau_thres - diff_ratio).sign()
            ##############################################

            # Calculate L2 loss between original and adversarial features
            l2_loss = ((((outputs - original_features) ** 2).sum(1))).sum()


            logits_uncond = cosine_similarity(outputs, text_embed, dim=1)

            loss = l2_loss + self.loss_lamda*logits_uncond
            # For targeted attacks, we want to maximize the L2 loss
            if self.targeted:
                cost = -loss
            else:
                cost = loss

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images_for_grad,  torch.ones_like(loss), retain_graph=False, create_graph=False
            )[0]

            # Update using the detached gradient
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # ####################################################################
            deltas_per_step.append(delta.clone().detach())
            # ###################################################################


            # Explicitly free memory
            del outputs, loss, cost, grad, delta, adv_images_for_grad
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


        del original_features
        torch.cuda.empty_cache()

        return adv_images
