import torch
import torch.nn as nn

from ..attack import Attack

import torch.nn.functional as F

class PGD_PRM_ADAM(Attack):
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
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True,
                 resize_rate=0.9,
                 diversity_prob=0.5,
                 decay=1.0,

                 ):
        super().__init__("PGD_PRM_ADAM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.decay = decay

        self.supported_mode = ["default", "targeted"]


    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(
            x, size=[rnd, rnd], mode="bilinear", align_corners=False
        )
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )

        return padded if torch.rand(1) < self.diversity_prob else x

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)


        adv_images = images.clone().detach()

        with torch.no_grad():
            original_output_features_list = self.get_logits(images, get_prm_layer_features=True, normalize=False)

        # if self.random_start:
        #     # Starting at a uniformly random point
        #     adv_images = adv_images + torch.empty_like(adv_images).uniform_(
        #         -self.eps, self.eps
        #     )
        #     adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        # Initialize perturbation
        perturbation = torch.zeros_like(images).float().to(self.device)
        if self.random_start:
            perturbation = torch.empty_like(images).uniform_(-self.eps, self.eps).to(self.device)

        perturbation.requires_grad_(True)
        optimizer = torch.optim.Adam([perturbation], lr=5e-1)

        cost_list = []
        for _ in range(self.steps):
            # Create a fresh copy for gradient computation
            # adv_images_for_grad = adv_images.clone().detach().requires_grad_(True)
            adv_images = images + perturbation
            adv_images = torch.clamp(adv_images, 0, 1)
            adv_outputs_features_list = self.get_logits(self.input_diversity(adv_images), get_prm_layer_features=True, normalize=False)

            losses = 0.0
            for layer, (item, clean_item) in enumerate(zip(adv_outputs_features_list, original_output_features_list)):
                L, B, D = item.shape
                item = F.normalize(item.reshape(-1, D))
                clean_item = F.normalize(clean_item.reshape(-1, D))
                losses += F.cosine_similarity(item, clean_item.detach()).mean()


            # cost = losses
            #
            cost_list.append(losses.item())
            # # Update adversarial images
            # grad = torch.autograd.grad(
            #     cost, adv_images_for_grad, retain_graph=False, create_graph=False
            # )[0]
            #
            # grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            # grad = grad + momentum * self.decay
            # momentum = grad
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # # Update using the detached gradient
            # adv_images = adv_images.detach() - self.alpha * grad.sign()
            # delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            # adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # # Explicitly free memory
            # del adv_outputs_features_list, cost, grad, delta, adv_images_for_grad
            # Clamp perturbation
            with torch.no_grad():
                perturbation.data = torch.clamp(perturbation, -self.eps, self.eps)
                adv_images = images + perturbation
                adv_images = torch.clamp(adv_images, 0, 1)
                perturbation.data = adv_images - images  # project to valid range
            torch.cuda.empty_cache()

        adv_images = torch.clamp(images + perturbation, 0, 1)

        return adv_images
