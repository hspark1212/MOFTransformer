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
import torch.nn.functional as F

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
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = self.proj(x)
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
            embed_dim=768,
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
        drop_rate = drop_rate if config is None else config["drop_rate"]
        self.in_chans = in_chans
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def mask_tokens(self, orig_image, feats, unnormalize=True):
        """
        Prepare masked tokens inputs/labels for masked patch prediction: 80% MASK, 10% random, 10% original.
        :param orig_image = _x, Tensor [B, C, H, W, D]
        :param feats = x  Tensor [B, ph*pw*pd, emb_dim]

        :return feats [B, ph*pw, emb_dim], labels [B, ph*pw, C]

        """
        B, C, _, _, _ = orig_image.shape

        # unnormalize
        if unnormalize:
            mask_orig_image = orig_image == 0
            orig_image = (orig_image - 0.5) * (5000. * 2)  # [B, emb_dim, H, W, D]
            orig_image[mask_orig_image] = 0

        _, _, P, P, P = self.patch_embed.proj.weight.shape

        # [B, C, H, W, D] -> [B, C, ph, pw, pd] with average channel in patch
        with torch.no_grad():
            img_unnorm_patch = F.conv3d(
                orig_image,
                weight=torch.ones(C, 1, P, P, P).to(orig_image) / (P * P * P),
                bias=None,
                stride=(P, P, P),
                padding=0,
                groups=C,  # when channel=3 -> groups=3
            )  # [B, C, ph, pw, pd]

        labels = (
            (img_unnorm_patch.long().flatten(start_dim=2, end_dim=4))  # [B, C, ph*pw*pd]
                .permute(0, 2, 1)
                .contiguous()
        )  # [B, ph*pw*pd, C]

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape[:-1], 0.15)  # [B, ph*pw*pd]
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens [B, ph*pw*pd, C]

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
                torch.bernoulli(torch.full(labels.shape[:-1], 0.8)).bool() & masked_indices
        )  # [B, ph*pw*pd]

        feats[indices_replaced] = self.mask_token.to(feats)

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

        x = self.patch_embed(_x)  # [B, emb_dim, ph, pw, pd]
        x_mask = (_x.sum(dim=1) != 0).float()[:, None, :, :, :]  # [B, 1, H, W, D]

        x_mask = F.interpolate(x_mask, size=(x.shape[2], x.shape[3], x.shape[4])).long()  # [B, 1, ph, pw, pd]

        x_h = x_mask[:, 0].sum(dim=1)[:, 0, 0]  # [B], # of non-mask pixel
        x_w = x_mask[:, 0].sum(dim=2)[:, 0, 0]  # [B], # of non_mask pixel
        x_d = x_mask[:, 0].sum(dim=3)[:, 0, 0]  # [B], # of non_mask pixel

        B, emb_dim, ph, pw, pd = x.shape  # [B, emb_dim, ph, pw, pd]

        spatial_pos = (
            self.pos_embed[:, 1:, :]  # [1, num_patches, embed_dim]
                .transpose(1, 2)  # [1, embed_dim, num_patches]
                .view(1, emb_dim, self.patch_dim, self.patch_dim, self.patch_dim)
        )  # # [1, emb_dim, patch_dim, patch_dim, patch_dim] interpolate in the next line
        
        pos_embed = torch.cat(
            [
                F.pad(
                    F.interpolate(
                        spatial_pos, size=(h, w, d), mode="trilinear", align_corners=True,
                    ),
                    # spatial_pos : [1, emb_dim, patch_dim, patch_dim, patch_dim] -> [1, emb_dim, h(x_h), w(x_w), d(x_d)]
                    (0, pd - d, 0, pw - w, 0, ph - h),  # [1, emb_dim, ph, pw]
                )
                for h, w, d in zip(x_h, x_w, x_d)
            ],
            dim=0,
        )  # [B, emb_dim, ph, pw, pd]

        # flatten and get patch_index
        pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [B, ph*pw*pd, emb_dim]
        x = x.flatten(2).transpose(1, 2)  # [B, ph*pw*pd, emb_dim]

        patch_index = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(x_mask.shape[-3]),
                    torch.arange(x_mask.shape[-2]),
                    torch.arange(x_mask.shape[-1]),
                    indexing="ij",
                ),
                dim=-1,
            )[None, None, :, :, :, :]  # [ph, pw, pd, 3] -> [1, 1, ph, pw, pd, 3]
                .expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1, -1)  # [B, 1, ph, pw, pd, 3]
                .flatten(1, 4)
        )  # [B, ph*pw*pd, 3]

        x_mask = x_mask.flatten(1)  # [B, ph*pw*pd]

        if mask_it:
            x, label = self.mask_tokens(_x, x)  # [B, ph*pw*pd, emb_dim], [B, ph*pw*pd, C]

        if (
                max_image_len < 0
                or max_image_len is None
                or not isinstance(max_image_len, int)
        ):
            # suppose aug is 800 x 1333, then, maximum effective res is 800 x 1333 (if one side gets bigger, the other will be constrained and be shrinked)
            # (800 // self.patch_size) * (1333 // self.patch_size) is the maximum number of patches that single image can get.
            # if self.patch_size = 32, 25 * 41 = 1025
            # if res is 384 x 640, 12 * 20 = 240
            eff = x_h * x_w * x_d
            max_image_len = eff.max()
        else:
            eff = x_h * x_w * x_d
            max_image_len = min(eff.max(), max_image_len)

        # x_mask = [B, ph*pw*pd]
        valid_idx = x_mask.nonzero(as_tuple=False)
        non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
        non_valid_row_idx = [
            non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows
        ]

        valid_nums = [v.size(0) for v in valid_row_idx]
        non_valid_nums = [v.size(0) for v in non_valid_row_idx]
        pad_nums = [max_image_len - v for v in valid_nums]

        select = list()
        for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
            if p <= 0:  # no padding -> valid_nums = max_len
                valid_choice = torch.multinomial(torch.ones(v).float(), max_image_len)
                select.append(valid_row_idx[i][valid_choice])

            else:  # existing padding -> valid_nums < max_len
                # randomly choice pad
                pad_choice = torch.multinomial(
                    torch.ones(nv).float(), p, replacement=True
                )

                # valid_row_idx and (randomly choice non_valid_row_idx with # of pad)
                select.append(
                    torch.cat(
                        [valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0,
                    )
                )

        select = torch.cat(select, dim=0)  # [max_len * B, ]

        x = x[select[:, 0], select[:, 1]].view(B, -1, emb_dim)  # [B, max_len, emb_dim]
        x_mask = x_mask[select[:, 0], select[:, 1]].view(B, -1)  # [B, max_len]
        patch_index = patch_index[select[:, 0], select[:, 1]].view(B, -1, 3)  # [B, max_len, 2]
        pos_embed = pos_embed[select[:, 0], select[:, 1]].view(B, -1, emb_dim)  # [B, max_len, emb_dim]

        if mask_it:
            label = label[select[:, 0], select[:, 1]].view(B, -1, self.in_chans)  # [B, max_len, C]

            label[x_mask == 0] = -100  # [B, max_len, C]

            label = torch.cat(
                [torch.full((label.shape[0], 1, self.in_chans), -100).to(label), label, ], dim=1,
            )  # [B, max_len+1, C]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, max_len+1, C]
        pos_embed = torch.cat(
            (self.pos_embed[:, 0, :][:, None, :].expand(B, -1, -1), pos_embed), dim=1
        )  # [B, max_len+1, C]
        x = x + pos_embed
        x = self.pos_drop(x)

        if self.add_norm_before_transformer:
            x = self.pre_norm(x)
        # x_mask = [B, max_len] -> [B, max_len+1]
        x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1)

        if mask_it:
            return x, x_mask, (patch_index, (ph, pw)), label
        else:
            return x, x_mask, (patch_index, (ph, pw)), None
