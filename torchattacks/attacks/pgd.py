import torch
import torch.nn as nn

from ..attack import Attack


class PGD(Attack):
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

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, image_only_attack=False):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.image_only_attack = image_only_attack
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)



        loss = nn.CrossEntropyLoss()
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        adv_images = images.clone().detach()

        if self.image_only_attack:
            with torch.no_grad():
                original_output_features = self.get_logits(images, get_image_features=True, normalize=True)

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        #cost_list = []
        for _ in range(self.steps):
            # Create a fresh copy for gradient computation
            adv_images_for_grad = adv_images.clone().detach().requires_grad_(True)

            if self.image_only_attack:
                outputs = self.get_logits(adv_images_for_grad, get_image_features=True, normalize=True)
                # COmpute cosine simialrity between original features and output features
                cost = -cosine_sim(original_output_features, outputs).mean()

            else:

                outputs = self.get_logits(adv_images_for_grad)

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)


            #cost_list.append(cost.item())
            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images_for_grad, retain_graph=False, create_graph=False
            )[0]

            # Update using the detached gradient
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # Explicitly free memory
            del outputs, cost, grad, delta, adv_images_for_grad
            torch.cuda.empty_cache()

        return adv_images
