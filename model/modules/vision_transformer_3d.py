""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from torch.nn import AvgPool3d
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        assert C % self.num_heads == 0
        qkv = (
            self.qkv(x)  # [B, N, 3*C]
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)  # [B, N, 3, num_heads, C//num_heads]
                .permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, C//num_heads]
        )
        q, k, v = (
            qkv[0],  # [B, num_heads, N, C//num_heads]
            qkv[1],  # [B, num_heads, N, C//num_heads]
            qkv[2],  # [B, num_heads, N, C//num_heads]
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)  # [B, num_heads, N, N]
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, num_heads, N, C//num_heads] -> [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PatchEmbed3D(nn.Module):
    """ Image to Patch Embedding for 3D"""

    def __init__(
            self,
            img_size,  # minimum of H or W ex. 384
            patch_size,  # p -> length of fixed patch ex. 32
            in_chans=1,
            embed_dim=768,
            no_patch_embed_bias=False,
    ):
        super().__init__()

        assert img_size % patch_size == 0
        num_patches = (img_size ** 3) // (patch_size ** 3)
        self.img_size = img_size  # default: 30
        self.patch_size = patch_size  # default: 5
        self.num_patches = num_patches

        self.proj = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)',
                      p1=patch_size, p2=patch_size, p3=patch_size),
            nn.Linear(patch_size * patch_size * patch_size * in_chans, embed_dim),
        )

    def forward(self, x):
        x = self.proj(x)  # [B, num_patches,
        return x  # [B, emb_dim, px, ph, pd]


class VisionTransformer3D(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=None,
            add_norm_before_transformer=False,
            mpp_ratio=0.15,
            config=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        self.in_chans = in_chans
        self.mpp_ratio = mpp_ratio

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.add_norm_before_transformer = add_norm_before_transformer

        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.patch_size = patch_size  # default = 32
        self.patch_dim = img_size // patch_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if add_norm_before_transformer:
            self.pre_norm = norm_layer(embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def mask_tokens(self, orig_image, feats, patch_size, mpp_ratio):
        """
        Prepare masked tokens inputs/labels for masked patch prediction: 80% MASK, 10% random, 10% original.
        :param orig_image = _x, Tensor [B, C, H, W, D]
        :param feats = x  Tensor [B, ph*pw*pd, emb_dim]

        :return feats [B, ph*pw, emb_dim], labels [B, ph*pw, C]

        """

        m = AvgPool3d(patch_size, patch_size)
        with torch.no_grad():
            img_patch = m(orig_image)

        labels = (
            (img_patch.long().flatten(start_dim=2, end_dim=4))  # [B, C, ph*pw*pd]
                .permute(0, 2, 1)
                .contiguous()
        )  # [B, ph*pw*pd, C]

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        # probability_matrix = torch.full(labels.shape[:-1], 0.15)  # [B, ph*pw*pd]
        probability_matrix = torch.full(labels.shape[:-1], mpp_ratio)  # [B, ph*pw*pd]
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens [B, ph*pw*pd, C]
        """
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
                torch.bernoulli(torch.full(labels.shape[:-1], 0.8)).bool() & masked_indices
        )  # [B, ph*pw*pd]

        feats[indices_replaced] = self.mask_token.to(feats)
        """
        return feats, labels

    def visual_embed(self, _x, max_image_len, mask_it=False):
        """

        :param _x: batch images, Tensor [B, C, H, W, D]
        :param max_image_len: Int (or -1)
        :param mask_it: Bool
        :return:
            x:  Tensor [B, max_image_len+1, hid_dim],
            x_mask: Tensor [B, max_image_len+1]],
            (patch_index, (H, W, D)): [[B, max_image_len+1, 3], [H, W, D]]
            label: [B, max_image_len+1, C]
        """

        B, _, _, _, _ = _x.shape
        x = self.patch_embed(_x)  # [B, ph*pw*pd, embed_dim]
        # x = x.flatten(2).transpose(1, 2)

        # mpp
        if mask_it:
            x, label = self.mask_tokens(_x, x, self.patch_size,
                                        self.mpp_ratio)  # [B, ph*pw*pd, emb_dim], [B, ph*pw*pd, C]
            label = torch.cat(
                [torch.full((label.shape[0], 1, self.in_chans), -100).to(label), label], dim=1,
            )  # [B, max_len+1, C]

        # cls tokens
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, ph*pw*pd, embed_dim]

        # positional embedding
        x += self.pos_embed
        x = self.pos_drop(x)

        if self.add_norm_before_transformer:
            x = self.pre_norm(x)

        x_mask = torch.ones(x.shape[:2]).to(x)  # [B, ph*pw*pd]

        if mask_it:
            return x, x_mask, label
        else:
            return x, x_mask, None
