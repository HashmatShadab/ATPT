import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .openai import load_openai_model
from .factory import create_model_and_transforms, get_tokenizer
from .tokenizer import tokenize
from .tokenizer import SimpleTokenizer as _Tokenizer
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import copy

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT = 'cache/open_clip'

mu = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
from torch.nn.functional import cosine_similarity

# OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
# OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def text_global_pool(
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        pool_type: str = 'argmax',
) -> torch.Tensor:
    if pool_type == 'first':
        pooled = x[:, 0]
    elif pool_type == 'last':
        pooled = x[:, -1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
    else:
        pooled = x

    return pooled

class ImageNormalizer(nn.Module):

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input):
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  # type: ignore


class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load_openai_model(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()

        self.cls_head = nn.Linear(embed_dim, n_class)

    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding[:40]
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.text_pool_type = clip_model.text_pool_type
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = self.transformer(x)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        x = text_global_pool(x, tokenized_prompts, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return  x



class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end',
                 learned_cls=False, tokenizer_openai=None, model_name=None):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            if tokenizer_openai is not None:
                if "delta" in model_name:
                    prompt = tokenizer_openai(ctx_init).to(self.device)
                else:
                    prompt = tokenizer_openai(ctx_init, context_length=40).to(self.device)
            else:
                prompt = tokenize(ctx_init, context_length=40).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None:
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  # (N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)  # to be optimized

        if tokenizer_openai is not None:
            if "delta" in model_name:
                tokenized_prompts = torch.cat([tokenizer_openai(p) for p in prompts]).to(self.device)
            else:
                tokenized_prompts = torch.cat([tokenizer_openai(p, context_length=40) for p in prompts]).to(self.device)
        else:
            tokenized_prompts = torch.cat([tokenize(p, context_length=40) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)  # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim,
                                      dtype=self.dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip,_,_ = create_model_and_transforms(arch, device=self.device, cache_dir=DOWNLOAD_ROOT, pretrained='openai')

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None:
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        cls,  # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx  # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                 n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, ):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _, tokenizer = get_open_clip(arch, device=device, cache_dir=DOWNLOAD_ROOT)
        self.device = device
        self.model = clip
        # self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls, tokenizer, model_name=arch)
        self.criterion = criterion

        self.normalize = ImageNormalizer(mu, std).cuda(device)


    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def encode_image(self, image, normalize: bool = False, get_all_layers=False):
        if not get_all_layers:
            features = self.model.encode_image(image)
            # F.normalize doesn't modify the input tensor in-place, so this is safe
            return F.normalize(features, dim=-1) if normalize else features
        else:
            image_features_all_layers_list = self.model.encode_image(image, get_all_layers=True)
            # if normalize, normalize all the features
            if normalize:
                normalized_layers = []
                for layer in image_features_all_layers_list:
                    normalized_layers.append(F.normalize(layer, dim=-1))
                return normalized_layers
            else:
                return image_features_all_layers_list


    def get_text_features(self, normalize=False):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.model.encode_text_embeddings(prompts, tokenized_prompts)
        if normalize:
            t_norm = t_features.norm(dim=-1, keepdim=True)
            t_features = t_features / t_norm

        return t_features

    def inference(self, image):

        image_features = self.encode_image(self.normalize(image.type(self.dtype)))
        # Use non-in-place normalization to avoid modifying tensors in the computation graph
        image_norm = image_features.norm(dim=-1, keepdim=True)
        image_features_normalized = image_features / image_norm

        text_features = self.get_text_features()
        # Use non-in-place normalization to avoid modifying tensors in the computation graph
        text_norm = text_features.norm(dim=-1, keepdim=True)
        text_features_normalized = text_features / text_norm

        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * image_features_normalized @ text_features_normalized.t())

        return logits

    def inference_text(self, image, text_features):

        image_features = self.encode_image(self.normalize(image.type(self.dtype)))
        # Use non-in-place normalization to avoid modifying tensors in the computation graph
        image_norm = image_features.norm(dim=-1, keepdim=True)
        image_features_normalized = image_features / image_norm

        # text_features = self.get_text_features()
        # # Use non-in-place normalization to avoid modifying tensors in the computation graph
        # text_norm = text_features.norm(dim=-1, keepdim=True)
        # text_features_normalized = text_features / text_norm

        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * image_features_normalized @ text_features.t())

        return logits

    def inference_move_image_features_noisy_anchor(self, image, sigma=0.18, n_anchors=10, alpha=1.0, diff_threshold=0.85,
                                                   noise_type='gaussian', normalize_embeddings=True, uniform_noise_eps=4.0):
        """
        image: Tensor of shape [B, C, H, W]
        sigma: standard deviation for Gaussian noise
        n_anchors: number of noisy samples to average for anchor
        alpha: interpolation factor
        """
        # Step 1: Encode source feature (adversarial or clean)
        image_input = self.normalize(image.type(self.dtype))
        f_source = self.encode_image(image_input)
        f_source_norm = f_source.norm(dim=-1, keepdim=True)
        f_source_normalized = f_source / f_source_norm

        # Step 2: Construct anchor by averaging n noisy features
        # Generate all noisy samples at once
        batch_size = image.size(0)
        # Create a batch of noisy images [n_anchors*batch_size, C, H, W]
        # number of noisy augmentations for each sample in the batch is equal to n_anchors
        if noise_type.lower() == "gaussian":
            noise_batch = sigma * torch.randn(n_anchors, batch_size, *image.shape[1:], device=image.device)
        elif noise_type.lower() == "uniform":
            # noise_batch = uniform_noise_eps * torch.rand(n_anchors, batch_size, *image.shape[1:], device=image.device) - uniform_noise_eps / 2
            eps = float(uniform_noise_eps/255.0)
            noise_batch = torch.empty(
                n_anchors, batch_size, *image.shape[1:], device=image.device
            ).uniform_(-eps, eps)
        else:
            raise ValueError("Noise type not supported: {}".format(noise_type))

        noisy_images = image.unsqueeze(0) + noise_batch  # [n_anchors, batch_size, C, H, W]
        # (Optional but often helpful) keep valid pixel range
        noisy_images = noisy_images.clamp(0.0, 1.0)

        # Reshape to [n_anchors*batch_size, C, H, W] in order to pass through the network in a single batch
        noisy_images = noisy_images.view(n_anchors * batch_size, *image.shape[1:])  # [n_anchors*batch_size, C, H, W]
        # Normalize and process all noisy images in one batch
        noisy_images = self.normalize(noisy_images.type(self.dtype))
        f_noisy_all = self.encode_image(noisy_images)  # [n_anchors*batch_size, feature_dim]

        # Reshape to [n_anchors, batch_size, feature_dim] and sum across anchors
        f_noisy_all = f_noisy_all.view(n_anchors, batch_size, -1)

        # Calculate diff_ratio between f_source_normalized and normalized f_noisy_all
        f_noisy_normalized = f_noisy_all / f_noisy_all.norm(dim=-1, keepdim=True)  # [n_anchors, batch_size, feature_dim]
        diff_ratio = (f_noisy_normalized - f_source_normalized.unsqueeze(0)).norm(dim=-1) / f_source_normalized.norm(dim=-1).unsqueeze(0)  # [n_anchors, batch_size]
        diff_ratio_mean = diff_ratio.mean(dim=0)
        # Initialize alpha as a vector (length = batch_size) with the same initial value
        alpha_vector = torch.full_like(diff_ratio_mean, fill_value=alpha)
        # Set alpha values to 0.0 where diff_ratio_mean < diff_threshold
        alpha_vector[diff_ratio_mean < diff_threshold] = 0.0

        f_anchor_sum = f_noisy_all.sum(dim=0)  # [batch_size, feature_dim]

        f_anchor = f_anchor_sum / n_anchors
        f_anchor_norm = f_anchor.norm(dim=-1, keepdim=True)
        f_anchor_normalized = f_anchor / f_anchor_norm

        # Step 3: One-step linear interpolation
        if normalize_embeddings:
            f_moved = (1 - alpha_vector.unsqueeze(1)) * f_source_normalized + alpha_vector.unsqueeze(1) * f_anchor_normalized
        else:
            f_moved = (1 - alpha_vector.unsqueeze(1)) * f_source + alpha_vector.unsqueeze(1) * f_anchor

        f_moved_norm = f_moved.norm(dim=-1, keepdim=True)
        f_moved = f_moved / f_moved_norm  # Final normalization

        # Step 4: Get text features and normalize
        text_features = self.get_text_features()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Step 5: Compute logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * f_moved @ text_features.t()

        return logits, diff_ratio_mean

    def inference_move_image_features_text_anchor(self, image, steps=10, step_size=10.0, null_text_features=None):
        """
        image: Tensor of shape [B, C, H, W]

        """
        # Step 1: Encode source feature (adversarial or clean)
        image_input = self.normalize(image.type(self.dtype))
        image_embedding = self.encode_image(image_input)
        image_embedding_norm = image_embedding.norm(dim=-1, keepdim=True)
        image_embedding = image_embedding / image_embedding_norm

        batch, device = image_embedding.shape[0], image_embedding.device

        # Step 2: Make sure requires gradient is True for image embeddings
        if not image_embedding.requires_grad:
            image_embedding.requires_grad = True

        # Step 3: Load Null Text features
        # null_text_features = null_text_features.mean(dim=1)
        null_text_features = null_text_features.repeat(batch, 1).to(device)

        # Step 4: Run image embedding optimization
        momentum = torch.zeros_like(image_embedding)
        norm = "L2"
        gamma = 0.

        for i in range(steps):
            r = torch.norm(image_embedding, dim=1, keepdim=True)
            u = image_embedding / r

            logits_uncond = cosine_similarity(image_embedding, null_text_features, dim=1)
            loss = - logits_uncond
            grad = torch.autograd.grad(loss, image_embedding, torch.ones_like(loss), retain_graph=True)[0]

            grad_u = r * grad

            if norm == "Linf":
                momentum = gamma * momentum - (1 - gamma) * grad_u / torch.norm(grad_u, p=1)
                u = u + step_size * momentum.sign()
            elif norm == "L2":
                momentum = gamma * momentum - (1 - gamma) * grad_u / torch.norm(grad_u, p=2)
                u = u + step_size * momentum

            u = u / torch.norm(u, dim=1, keepdim=True)
            image_embedding = r * u

        # Step 4: Get text features and normalize
        text_features = self.get_text_features()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Step 5: Compute logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_embedding @ text_features.t()

        return logits

    def forward(self, input, get_prm_layer_features=False, get_image_features=False, normalize=False, get_image_text_features=False, text_features=None, null_text_features=False,
                move_image_features_noisy_anchor=False, move_image_features_text_anchor=False, purify_params=None):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            if get_image_features:
                image_features = self.encode_image(self.normalize(input.type(self.dtype)), normalize=normalize)
                return image_features
            if get_prm_layer_features:
                image_features_all_layers = self.encode_image(self.normalize(input.type(self.dtype)), normalize=normalize, get_all_layers=True)
                return image_features_all_layers
            elif get_image_text_features:
                image_features, text_features, logit_scale = self.forward_features(input)
                return image_features, text_features, logit_scale
            elif move_image_features_noisy_anchor:
                # Set default values if purify_params is not provided
                if purify_params is None:
                    purify_params = {'sigma': 0.18, 'n_anchors': 10, 'alpha': 1.2, 'diff_threshold': 0.85}

                    # Extract parameters from the dictionary
                sigma = purify_params.get('sigma', 0.18)
                n_anchors = purify_params.get('n_anchors', 10)
                alpha = purify_params.get('alpha', 1.2)
                diff_threshold = purify_params.get('diff_threshold', 0.85)
                noise_type = purify_params.get('noise_type', 'gaussian')
                normalize_embeddings = purify_params.get('normalize_embeddings', False)
                uniform_noise_eps = purify_params.get('uniform_noise_eps', 4.0)

                logits, diff_ratio = self.inference_move_image_features_noisy_anchor(input, sigma=sigma, n_anchors=n_anchors,
                                                                                     alpha=alpha, diff_threshold=diff_threshold,
                                                                                     noise_type=noise_type, normalize_embeddings=normalize_embeddings, uniform_noise_eps=uniform_noise_eps)
                return logits, diff_ratio
            elif move_image_features_text_anchor:
                # Set default values if purify_params is not provided
                if purify_params is None:
                    purify_params = {'steps': 10, 'step_size': 10.0}

                # Extract parameters from the dictionary
                steps = purify_params.get('steps', 10)
                step_size = purify_params.get('step_size', 10.0)

                logits = self.inference_move_image_features_text_anchor(input, steps=steps, step_size=step_size, null_text_features=null_text_features)
                return logits
            else:
                if text_features is None:
                    return self.inference(input)
                else:
                    return self.inference_text(input, text_features)

    def forward_features(self, input):
        image_features = self.encode_image(self.normalize(input.type(self.dtype)))
        # Use non-in-place normalization to avoid modifying tensors in the computation graph
        image_norm = image_features.norm(dim=-1, keepdim=True)
        image_features_normalized = image_features / image_norm

        text_features = self.get_text_features()
        # Use non-in-place normalization to avoid modifying tensors in the computation graph
        text_norm = text_features.norm(dim=-1, keepdim=True)
        text_features_normalized = text_features / text_norm

        logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        return image_features_normalized, text_features_normalized, logit_scale


def get_coop(clip_arch, classnames, device, n_ctx, ctx_init, learned_cls=False):
    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                               n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model

def get_open_clip(clip_arch, device, cache_dir=DOWNLOAD_ROOT):
    model_names = {
        "hf-hub:zw123/delta_clip_l14_224",
        "hf-hub:chs20/tecoa4-clip",
        "hf-hub:chs20/tecoa2-clip",
        "hf-hub:chs20/fare2-clip",
        "hf-hub:chs20/fare4-clip",
    "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",

    }
    if clip_arch in model_names:
        clip, _, preprocess = create_model_and_transforms(clip_arch, device=device, cache_dir=DOWNLOAD_ROOT)  # precision='fp32'
    else:
        clip, _, preprocess = create_model_and_transforms(clip_arch, device=device, cache_dir=DOWNLOAD_ROOT, pretrained='openai')  # precision='fp32'
    print(f"Using CLIP model: {clip_arch} on device: {device}")
    tokenizer = get_tokenizer(clip_arch, cache_dir=cache_dir)
    # tokenizer = get_tokenizer("hf-hub:zw123/delta_clip_l14_224", cache_dir="cache/open_clip")
    # clip, _, preprocess = create_model_and_transforms("hf-hub:zw123/delta_clip_l14_224", device="cpu", cache_dir="cache/open_clip")


    return clip, _, preprocess, tokenizer

@torch.no_grad()
def get_text_embeddings(clip_arch, classnames, class_templates, device="cuda"):
    """
    Generates normalized text embeddings from class names and templates.

    Args:
        classnames (list of str): List of class names (e.g., ['dog', 'cat']).
        class_templates (list of str): List of templates (e.g., ["a photo of a {}", "a blurry photo of a {}"]).
        tokenizer: The tokenizer from OpenCLIP (e.g., from get_open_clip).
        clip_model: The CLIP model from OpenCLIP.
        device (str): Device to run on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Text embeddings of shape [num_classes, embed_dim].
    """

    clip, _, _, tokenizer = get_open_clip(clip_arch, device=device, cache_dir=DOWNLOAD_ROOT)

    all_text_features = []

    for classname in classnames:
        texts = [template.format(c=classname) for template in class_templates]  # prompt engineering
        if "delta" in clip_arch:
            tokenized = tokenizer(texts).to(device)  # [T, seq_len]
        else:
            tokenized = tokenizer(texts, context_length=40).to(device)                               # [T, seq_len]
        text_features = clip.encode_text(tokenized)                    # [T, D]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
        mean_text_feature = text_features.mean(dim=0)                        # Average templates
        mean_text_feature = mean_text_feature / mean_text_feature.norm()     # Normalize again
        all_text_features.append(mean_text_feature)

    all_text_features = torch.stack(all_text_features, dim=0)  # [num_classes, D]


    null_templates = [template.format(c="") for template in class_templates]
    all_null_text_features = []

    if "delta" in clip_arch:
        texts_null = tokenizer(null_templates).to(device)  # [T, seq_len]
    else:
        texts_null = tokenizer(null_templates, context_length=40).to(device)

    text_null_features = clip.encode_text(texts_null)
    text_null_features = text_null_features / text_null_features.norm(dim=-1, keepdim=True)  # Normalize
    mean_text_null_feature = text_null_features.mean(dim=0)  # Average templates
    mean_text_null_feature = mean_text_null_feature / mean_text_null_feature.norm()
    all_null_text_features.append(mean_text_null_feature)
    all_null_text_features = torch.stack(all_null_text_features, dim=0).to(device)



    # del clip and tokenizer
    del clip
    del tokenizer

    return all_text_features, all_null_text_features
