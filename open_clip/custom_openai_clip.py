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

# OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
# OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

CONFIDECES = {
        "ViT-L/14@336px": 10,
        "ViT-L/14": 5,
        "RN50x64": 3,
        "ViT-B/16": 1
    }

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
        self.model_name = model_name
        self.tokenizer_openai = tokenizer_openai

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
        if self.tokenizer_openai is not None:
            if "delta" in self.model_name:
                tokenized_prompts = torch.cat([self.tokenizer_openai(p) for p in prompts]).to(self.device)
            else:
                tokenized_prompts = torch.cat([self.tokenizer_openai(p, context_length=40) for p in prompts]).to(self.device)
        else:
            tokenized_prompts = torch.cat([tokenize(p, context_length=40) for p in prompts]).to(self.device)

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
        clip, embed_dim, _, tokenizer = get_open_clip(arch, device=device, cache_dir=DOWNLOAD_ROOT)
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

    def encode_image(self, image, normalize: bool = False):
        features = self.model.encode_image(image)
        # F.normalize doesn't modify the input tensor in-place, so this is safe
        return F.normalize(features, dim=-1) if normalize else features


    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.model.encode_text_embeddings(prompts, tokenized_prompts)

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

    def forward(self, input, get_image_features=False):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            if get_image_features:
                image_features = self.encode_image(self.normalize(input.type(self.dtype)))
                return image_features
            else:
                return self.inference(input)

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
    }
    if clip_arch in model_names:
        clip, embed_dim, preprocess = create_model_and_transforms(clip_arch, device=device, cache_dir=cache_dir)  # precision='fp32'
    else:
        clip, embed_dim, preprocess = create_model_and_transforms(clip_arch, device=device, cache_dir=DOWNLOAD_ROOT, pretrained='openai')  # precision='fp32'

    tokenizer = get_tokenizer(clip_arch)


    return clip, embed_dim, preprocess, tokenizer







def get_reward_model(device, args):
    openai_model_dict = {
        "delta_clip_l14_224": "hf-hub:zw123/delta_clip_l14_224",
        "tecoa4": "hf-hub:chs20/tecoa4-clip",
        "tecoa2": "hf-hub:chs20/tecoa2-clip",
        "fare2": "hf-hub:chs20/fare2-clip",
        "fare4": "hf-hub:chs20/fare4-clip",
        # "RN50": "RN50",
    }
    if args.reward_arch in openai_model_dict:
        model_name = openai_model_dict[args.reward_arch]
    else:
        model_name = args.reward_arch

    if args.multiple_reward_models:
        reward_model = CLIPRewardsMultiple(device, arch=["ViT-L/14@336px", "RN50x64", "ViT-L/14"], classification=True,
                            amplify_rewards=args.reward_amplify, sample_k=args.reward_sample_k,
                            reward_process=args.reward_process, process_batch=args.reward_process_batch,
                            weighted_scores=args.reward_weighted_scores,  clipscore_weight=args.reward_clipscore_weight)
    else:
        reward_model = CLIPRewards(device, arch=model_name, classification=True,
                                amplify_rewards=args.reward_amplify, sample_k=args.reward_sample_k,
                                reward_process=args.reward_process, process_batch=args.reward_process_batch,
                                   clipscore_weight=args.reward_clipscore_weight, classnames=args.classnames)

    return reward_model


class BaseRewards(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def extract_image_features(self, images):
        pass

    @torch.no_grad()
    def extract_text_features(self):
        pass

    @torch.no_grad()
    def set_class_features(self):
        self.class_features = self.extract_text_features()

    @torch.no_grad()
    def set_image_features(self, images):
        self.image_features = self.extract_image_features(images)

    @torch.no_grad()
    def confidence_gap(self, predictions):
        """
        Args:
            predictions: shape [bs, C]
        """
        value, index = torch.topk(predictions, 2, dim=-1)
        gap = value[:, 0] - value[:, 1]
        gap = gap - torch.mean(gap)

        return gap


class CLIPRewards(BaseRewards):
    def __init__(self, device, arch="ViT-B/16", clipscore_weight=2.5, classification=True,
                    amplify_rewards=False, sample_k=5, reward_process=True, process_batch=False,
                    default_resolutions=224, classnames=None) -> None:
        """
        calculating CLIP Reward
        Args:
            clipscore_weight: weight for calculating CLIPScore
            reward_process: If ture, post-process the rewards, e.g., subtract the reward mean or standardization
            amplify_rewards: If true, after subtracting the reward mean, also divide rewards by standard variance of rewards, i.e, standardization.
            sample_k: K for sampling.
            process_batch: If true, post-process the rewards within the {BatchSize x K} sampled text-image pairs.
                Others, post-process the rewards within the {1 x K} sampled text-image pairs.
                TPT augment the images, so we have a batch of augmented images from a single image.
        """
        super().__init__()
        self.default_resolutions = default_resolutions
        # self.clip_model, self.embed_dim, self.preprocess = clip.load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.clip_model, self.embed_dim, self.preprocess, self.tokenizer = get_open_clip(arch, device=device, cache_dir=DOWNLOAD_ROOT)
        self.classnames = classnames
        self.model_name = arch
        self.resolutions = self.clip_model.visual.image_size[0]
        self.clipscore_weight = clipscore_weight
        self.device = device
        self.classification = classification
        self.class_features = None
        self.image_features = None
        self.amplify_rewards = amplify_rewards
        self.sample_k = sample_k
        self.reward_process = reward_process
        self.process_batch = process_batch
        self.clip_model.eval()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = ["a photo of a" + " " + name + "." for name in classnames]

        if "delta" in self.model_name:
            tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts]).to(self.device)
        else:
            tokenized_prompts = torch.cat([self.tokenizer(p, context_length=40) for p in prompts]).to(self.device)

        self.tokenized_prompts = tokenized_prompts

        print("\n CLIPRewards model created: \n"
                "\t visual backbone: {}, resolutions: {}, amplify_rewards: {}, sample_k: {}, \n"
                "\t reward_process: {}, process_batch: {}\n".format(
                    arch, self.resolutions, amplify_rewards, sample_k, reward_process, process_batch))

    @torch.no_grad()
    def CLIPScore(self, class_index, images=None, image_features=None, captions=None, tokenized_cap=None, text_features=None,
                        pairwise=True):
        """
        class_index: sampled class index
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        text_features = self.class_features[class_index]
        image_features = torch.repeat_interleave(self.image_features, self.sample_k, dim=0)

        if pairwise:
            similarity = self.clipscore_weight * text_features @ image_features.t()
        else:
            similarity = self.clipscore_weight * torch.sum(text_features * image_features, dim=-1)

        scores = torch.maximum(similarity, torch.zeros_like(similarity)).squeeze()

        return scores

    @torch.no_grad()
    def extract_image_features(self, images):
        """extract image features without normalization"""
        if self.resolutions != self.default_resolutions:
            images = nn.functional.interpolate(images, size=self.resolutions, mode='bicubic', align_corners=True)
        image_features = self.clip_model.encode_image(images).float()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    @torch.no_grad()
    def extract_text_features(self):
        text_features = self.clip_model.encode_text(self.tokenized_prompts).float()

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    @torch.no_grad()
    def rewards_post_process(self, clip_score):
        """
        clip_score: shape [bs, K] or [bs * K]
        """
        if clip_score.shape[-1] > 1 and self.reward_process:
            mean = torch.mean(clip_score, dim=-1, keepdim=True)
            if self.amplify_rewards:
                std = torch.std(clip_score, dim=-1, keepdim=True) + 1e-5
            else:
                std = 1.0
            clip_score = (clip_score - mean) / std

        return clip_score.flatten()

    @torch.no_grad()
    def calulate_similarity(self):
        """
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * self.image_features @ self.class_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class CLIPRewardsMultiple(BaseRewards):
    def __init__(self, device, arch=["ViT-B/16", "RN50x64", "ViT-L/14"], clipscore_weight=2.5, classification=True,
                    amplify_rewards=False, sample_k=5, reward_process=True, process_batch=True, weighted_scores=True,
                    default_resolutions=224) -> None:
        """
        calculating CLIP Reward using multiple CLIP models
        Args:
            arch: a list of CLIP arches
            clipscore_weight: weight for calculating CLIPScore
            weighted_scores: if true, the final score is the weighted average of all scores
        """
        super().__init__()
        # load clip models
        clip_models = []
        self.preprocess = []
        self.resolutions = []
        weights = []
        self.default_resolutions = default_resolutions
        for ar in arch:
            # clip_model, embed_dim, preprocess = clip.load(ar, device=device, download_root=DOWNLOAD_ROOT)    ###########
            clip_model, embed_dim, preprocess, tokenizer = get_open_clip(ar, device=device, cache_dir=DOWNLOAD_ROOT)

            clip_models.append(clip_model)
            self.preprocess.append(preprocess)
            self.resolutions.append(clip_model.visual.input_resolution)
            weights.append(CONFIDECES[ar])
        self.clip_models = nn.ModuleList(clip_models)
        self.n_model = len(self.clip_models)
        self.weights = [round(x / sum(weights), 2) for x in weights]

        self.clipscore_weight = clipscore_weight
        self.device = device
        self.classification = classification
        self.class_features = None
        self.image_features = None
        self.amplify_rewards = amplify_rewards
        self.sample_k = sample_k
        self.reward_process = reward_process
        self.process_batch = process_batch
        self.weighted_scores = weighted_scores

        self.clip_models.eval()

        print("\n CLIPRewardsMultiple model created: \n"
                "\t visual backbone: {}, resolutions: {}, weighted_scores / weights: [ {} / {} ] \n"
                "\t amplify_rewards: {}, sample_k: {}, reward_process: {}, process_batch: {}\n".format(
                    arch, self.resolutions, weighted_scores, self.weights,
                    amplify_rewards, sample_k, reward_process, process_batch))

    @torch.no_grad()
    def CLIPScore(self, class_index, images=None, image_features=None, captions=None, tokenized_cap=None, text_features=None,
                        pairwise=True):
        """
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        all_scores = []
        for i in range(self.n_model):
            text_features = self.class_features[i][class_index]
            image_features = torch.repeat_interleave(self.image_features[i], self.sample_k, dim=0)

            if pairwise:
                similarity = self.clipscore_weight * text_features @ image_features.t()
                raise NotImplementedError
            else:
                similarity = self.clipscore_weight * torch.sum(text_features * image_features, dim=-1)

            # [n_samples]
            scores = torch.maximum(similarity, torch.zeros_like(similarity)).squeeze()
            all_scores.append(scores)

        scores = torch.stack(all_scores, dim=0)
        # [n_samples]
        if self.weighted_scores:
            # final_scores = torch.sum(scores / (torch.sum(scores, dim=0, keepdim=True) + 1e-5) * scores, dim=0)
            weights = torch.tensor(self.weights, device=scores.device, dtype=scores.dtype).unsqueeze(1)
            final_scores = torch.sum(weights * scores, dim=0)
        else:
            final_scores = torch.mean(scores, dim=0)

        return final_scores

    @torch.no_grad()
    def extract_image_features(self, images):
        """extract image features with normalization"""
        image_features = []
        for i in range(self.n_model):
            if self.resolutions[i] != self.default_resolutions:
                # different CLIP has different input sizes
                tmp_images = nn.functional.interpolate(images, size=self.resolutions[i], mode='bicubic', align_corners=True)
            else:
                tmp_images = images
            image_feat = self.clip_models[i].encode_image(tmp_images).float()
            image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
            image_features.append(image_feat)

        return image_features

    @torch.no_grad()
    def extract_text_features(self):
        """extract text features with normalization"""
        text_features = []
        for i in range(self.n_model):
            if captions is not None:
                caption_tokens = clip.tokenize(captions, truncate=True).to(self.device)  ###################
                text_feat = self.clip_models[i].encode_text(caption_tokens).float()

            if tokenized_cap is not None:
                text_feat = self.clip_models[i].encode_text(tokenized_cap).float()

            # normalized features
            text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
            text_features.append(text_feat)

        return text_features

    @torch.no_grad()
    def rewards_post_process(self, clip_score):
        """
        clip_score: shape [bs, K] or [bs * K]
        """
        # if (clip_score.ndim > 1 and clip_score.shape[-1] > 1) or (clip_score.ndim == 1 and clip_score.shape[-1] > 1):
        if clip_score.shape[-1] > 1 and self.reward_process:
            mean = torch.mean(clip_score, dim=-1, keepdim=True)
            if self.amplify_rewards:
                std = torch.std(clip_score, dim=-1, keepdim=True) + 1e-5
            else:
                std = 1.0
            clip_score = (clip_score - mean) / std

        return clip_score.flatten()


if __name__ == "__main__":
    # The Figure 1 (b) in the paper
    import torchvision.transforms as transforms
    from PIL import Image
    import os
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    path = "/home/shuzhao/Data/dataset/test_images"
    device = torch.device('cuda:{}'.format(0))
    images = os.listdir(path)
    print(images)
    resolution = 224
    arch = "ViT-B/16"
    arch = "ViT-L/14"
    clipscore_weight = 2.5
    captions = [
                "There are three sheeps standing together on the grass.",
                "A group of baseball players is crowded at the mound.",
                "Two girls bathe an elephant lying on its side"
                ]

    clip_model, embed_dim, preprocess = clip.load(arch, device=device, download_root=DOWNLOAD_ROOT)   ####
    clip_model = clip_model.float()
    clip_model.eval()

    all_iamges = []
    for file in images:
        image = Image.open(os.path.join(path, file))
        all_iamges.append(preprocess(image))
    images = torch.stack(all_iamges, dim=0).to(device)

    with torch.no_grad():
        # images
        image_features = clip_model.encode_image(images).float()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # captions
        caption_tokens = clip.tokenize(captions, truncate=True).to(device)
        text_features = clip_model.encode_text(caption_tokens).float()
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        similarity = clipscore_weight * text_features @ image_features.t()

    print(similarity)

    mean = torch.mean(similarity, dim=0, keepdim=True)
    print(similarity - mean)
    # ['COCO_val2014_000000001164.jpg', 'COCO_val2014_000000000772.jpg', 'COCO_val2014_000000000192.jpg']
    # tensor([[0.4146, 0.7624, 0.4753],
    #         [0.3114, 0.4829, 0.6724],
    #         [0.8394, 0.3277, 0.2738]], device='cuda:0')

    # CLIP-ViT-L/14
    # ['COCO_val2014_000000001164.jpg', 'COCO_val2014_000000000772.jpg', 'COCO_val2014_000000000192.jpg']
    # tensor([[0.0721, 0.6127, 0.2376],
    #         [0.0638, 0.2741, 0.3465],
    #         [0.7014, 0.2067, 0.0213]], device='cuda:0')
    # tensor([[-0.2070,  0.2482,  0.0358],
    #         [-0.2153, -0.0904,  0.1447],
    #         [ 0.4223, -0.1578, -0.1805]], device='cuda:0')
