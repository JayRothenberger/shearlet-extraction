"""
This code was originally obtained from:
https://github.com/facebookresearch/deit
and
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Callable, List, Optional, Tuple, Union

from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg
from timm.layers import DropPath, to_2tuple
from timm.models.registry import register_model

from .layers import SinGELU

from timm.layers import Format, nchw_to
from .shearlet_utils import trunc_normal_

import math
import logging

_logger = logging.getLogger(__name__)


def resample_patch_embed(
    patch_embed,
    new_size: List[int],
    interpolation: str = "bicubic",
    antialias: bool = True,
    verbose: bool = False,
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np

    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_size) == 2, "New shape should only be hw"
    old_size = patch_embed.shape[-2:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    if verbose:
        _logger.info(
            f"Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation."
        )

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias
        )[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.0
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.tensor(
        np.linalg.pinv(resize_mat.T), device=patch_embed.device
    )

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    orig_dtype = patch_embed.dtype
    patch_embed = patch_embed.float()
    patch_embed = v_resample_kernel(patch_embed)
    patch_embed = patch_embed.to(orig_dtype)
    return patch_embed


class Unraveling:
    def __init__(self, n, patch_size=1):
        self.levels = []
        self.patch_size = patch_size

        for i in range(0, n // 2):
            level = []
            for j in range(i, n - i):
                """
                if (j >= (n // 2)):
                    level.append((j, i))  # low, low-high
                    level.append((n - (i + 1), j)) # low, low-high
                    level.append((j, n - (i + 1)))
                elif j < (n // 2):
                    level.append((j, i))  # high, low
                    level.append((i, j))  # high, high
                    level.append((n - (i + 1), j))
                """
                level.append((j, i))
                level.append((i, j))
                level.append((n - (i + 1), j))
                level.append((j, n - (i + 1)))

            level = list(set(level))
            self.levels.append(
                (
                    torch.tensor([x for x, _ in level]),
                    torch.tensor([y for _, y in level]),
                )
            )

        levels = []

        for i in range(len(self.levels) // self.patch_size):
            # the elements here are tuples of tensors
            levels.append(
                (
                    torch.cat(
                        [
                            self.levels[j][0]
                            for j in range(i * patch_size, (i + 1) * patch_size)
                        ],
                        0,
                    ),
                    torch.cat(
                        [
                            self.levels[j][1]
                            for j in range(i * patch_size, (i + 1) * patch_size)
                        ],
                        0,
                    ),
                )
            )

        self.levels = levels

    def __call__(self, x):
        return [x[..., a, b] for a, b in self.levels]


class DirectionalUnraveling:
    """
    Note: expects FFT-shifted inputs
    """

    def __init__(self, n, channels, scales, patch_size=1):
        self.quadrants = []
        self.patch_size = patch_size

        print(channels // (3 * scales))

        groups = int(channels // 3)

        for quadrant in range(3):
            directions = []
            for c in range(0, channels // (3 * scales)):
                levels = []
                for i in range(0, n // 2):
                    level = []
                    for j in range(i, n - i):
                        if quadrant == 0:
                            level.append((j, i, c))  # low-high, low (R)
                            level.append((j, i, c + (1 * groups)))  # low-high, low (G)
                            level.append((j, i, c + (2 * groups)))  # low-high, low (B)

                        if quadrant == 1:
                            level.append((i, j, c))  # low, low-high (R)
                            level.append((i, j, c + (1 * groups)))  # low, low-high (G)
                            level.append((i, j, c + (2 * groups)))  # low, low-high (B)

                        # these are the high, high associated coordinates.
                        # ignore the high, high frequency coordinates and only keep the low, high
                        if quadrant == 2:
                            if (j > (n // 2)) or ((n - (i + 1)) > (n // 2)):
                                level.append((j, n - (i + 1), c))
                                level.append((n - (i + 1), j, c))

                                level.append((j, n - (i + 1), c + (1 * groups)))
                                level.append((n - (i + 1), j, c + (1 * groups)))

                                level.append((j, n - (i + 1), c + (2 * groups)))
                                level.append((n - (i + 1), j, c + (2 * groups)))

                    level = list(set(level))
                    levels.append(
                        (
                            torch.tensor([x for x, _, _ in level]).type(torch.int32),
                            torch.tensor([y for _, y, _ in level]).type(torch.int32),
                            torch.tensor([z for _, _, z in level]).type(torch.int32),
                        )
                    )
                directions.append(levels)
            self.quadrants.append(directions)

        levels = []

        print(len(self.quadrants))

        for quadrant in self.quadrants:
            for direction in quadrant:
                for i in range((len(direction) // self.patch_size)):
                    # the elements here are tuples of tensors
                    if (
                        i == 0
                    ):  # for the first patch (the smallest, least important section) we will do double the patch size (TODO)
                        levels.append(
                            (
                                torch.cat(
                                    [
                                        direction[j][0]
                                        for j in range(
                                            i * patch_size, (i + 1) * patch_size
                                        )
                                    ],
                                    0,
                                ),
                                torch.cat(
                                    [
                                        direction[j][1]
                                        for j in range(
                                            i * patch_size, (i + 1) * patch_size
                                        )
                                    ],
                                    0,
                                ),
                                torch.cat(
                                    [
                                        direction[j][2]
                                        for j in range(
                                            i * patch_size, (i + 1) * patch_size
                                        )
                                    ],
                                    0,
                                ),
                            )
                        )
                    else:
                        levels.append(
                            (
                                torch.cat(
                                    [
                                        direction[j][0]
                                        for j in range(
                                            i * patch_size, (i + 1) * patch_size
                                        )
                                    ],
                                    0,
                                ),
                                torch.cat(
                                    [
                                        direction[j][1]
                                        for j in range(
                                            i * patch_size, (i + 1) * patch_size
                                        )
                                    ],
                                    0,
                                ),
                                torch.cat(
                                    [
                                        direction[j][2]
                                        for j in range(
                                            i * patch_size, (i + 1) * patch_size
                                        )
                                    ],
                                    0,
                                ),
                            )
                        )

        self.levels = levels

    def __call__(self, x):
        return [x[..., c, a, b] for a, b, c in self.levels]


class LUnraveling:
    def __init__(self, n, slices, patch_size=1):
        self.levels_up = []
        self.levels_right = []
        self.patch_size = patch_size
        self.slices = slices

        print(n, slices)

        for i in range(0, slices):
            level_up = []
            level_right = []
            for j in range(0, n):
                level_up.append((j, i))
                level_right.append((((n - slices) + i), j))

            level_up = list(set(level_up))
            level_right = list(set(level_right))

            self.levels_up.append(
                (
                    torch.tensor([x for x, _ in level_up]),
                    torch.tensor([y for _, y in level_up]),
                )
            )
            self.levels_right.append(
                (
                    torch.tensor([x for x, _ in level_right]),
                    torch.tensor([y for _, y in level_right]),
                )
            )

        levels = []
        assert len(self.levels_up) == len(self.levels_right), (
            "Expected same number of spectral slices in both directions.  Is input square?"
        )
        for i in range(len(self.levels_up) // self.patch_size):
            # the elements here are tuples of tensors
            levels.append(
                (
                    torch.cat(
                        [
                            self.levels_up[j][0]
                            for j in range(i * patch_size, (i + 1) * patch_size)
                        ],
                        0,
                    ),
                    torch.cat(
                        [
                            self.levels_up[j][1]
                            for j in range(i * patch_size, (i + 1) * patch_size)
                        ],
                        0,
                    ),
                )
            )
            levels.append(
                (
                    torch.cat(
                        [
                            self.levels_right[j][0]
                            for j in range(i * patch_size, (i + 1) * patch_size)
                        ],
                        0,
                    ),
                    torch.cat(
                        [
                            self.levels_right[j][1]
                            for j in range(i * patch_size, (i + 1) * patch_size)
                        ],
                        0,
                    ),
                )
            )

        self.levels = levels

    def __call__(self, x):
        return [x[..., a, b] for a, b in self.levels]


class LFreqEmbed(nn.Module):
    """2D Shearlet Coefficients Image to Patch Embedding"""

    # TODO: support patch size here where we concat each pair of

    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        super().__init__()
        slices = img_size // 2
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)
        self.img_size = (img_size, img_size)
        self.grid_size = (img_size // 2, 1)
        self.num_patches = 2 * math.ceil(slices / patch_size)

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.unravel = LUnraveling(img_size, slices, patch_size)
        """
        self.layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(len(level[0]) * in_chans, embed_dim, bias=bias), 
                                                               CGELU(), 
                                                               torch.nn.Linear(embed_dim, embed_dim, bias=bias),
                                                               )
                                            for level in self.unravel.levels])
        """
        self.layer = torch.nn.Linear(
            len(self.unravel.levels[0][0]) * in_chans,
            embed_dim,
            bias=bias,
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def set_input_size(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_2tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            with torch.no_grad():
                new_proj = nn.Conv2d(
                    self.proj.in_channels,
                    self.proj.out_channels,
                    kernel_size=new_patch_size,
                    stride=new_patch_size,
                    bias=self.proj.bias is not None,
                )
                new_proj.weight.copy_(
                    resample_patch_embed(self.proj.weight, new_patch_size, verbose=True)
                )
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(
                img_size
            )

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(
                img_size[1] / self.patch_size[1]
            )
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape

        unraveled = self.unravel(x)
        tokens = self.layer(torch.stack([level.flatten(1) for level in unraveled], -2))
        tokens = self.norm(tokens)

        return tokens


class FreqEmbed(nn.Module):
    """2D Shearlet Coefficients Image to Patch Embedding"""

    # TODO: support patch size here where we concat each pair of

    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)
        self.img_size = (img_size, img_size)
        self.num_patches = math.ceil((img_size // 2) / patch_size)
        self.grid_size = (self.num_patches, 1)

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.unravel = Unraveling(img_size, patch_size)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    len(level[0]) * in_chans,
                    embed_dim,
                    bias=bias,
                )
                for level in self.unravel.levels
            ]
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def set_input_size(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_2tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            with torch.no_grad():
                new_proj = nn.Conv2d(
                    self.proj.in_channels,
                    self.proj.out_channels,
                    kernel_size=new_patch_size,
                    stride=new_patch_size,
                    bias=self.proj.bias is not None,
                )
                new_proj.weight.copy_(
                    resample_patch_embed(self.proj.weight, new_patch_size, verbose=True)
                )
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(
                img_size
            )

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(
                img_size[1] / self.patch_size[1]
            )
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape

        unraveled = self.unravel(x)
        tokens = torch.stack(
            [layer(level.flatten(1)) for layer, level in zip(self.layers, unraveled)],
            -2,
        )

        return tokens


class DirectionalFreqEmbed(nn.Module):
    """2D Shearlet Coefficients Image to Patch Embedding with Directional Tokens"""

    # TODO: support patch size here where we concat each pair of

    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
        scales: int = 1,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = (img_size, img_size)
        self.grid_size = (img_size // 2, 1)
        self.num_patches = (
            math.ceil((img_size // 2) / patch_size) * 3 * (in_chans // (scales * 3))
        )  # patches per quadrant * quadrants * directions

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.unravel = DirectionalUnraveling(img_size, in_chans, scales, patch_size)

        assert len(self.unravel.levels) == self.num_patches, (
            f"expected number of patches to match number of levels in the unraveling ({self.num_patches} and {len(self.unravel.levels)})"
        )

        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    len(level[0]) * (in_chans // (scales * 30)),
                    embed_dim,
                    bias=bias,
                )
                for level in self.unravel.levels
            ]
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        unraveled = self.unravel(x)
        tokens = torch.stack(
            [layer(level.flatten(1)) for layer, level in zip(self.layers, unraveled)],
            -2,
        )

        return tokens


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
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
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
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
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if Attention_block == torch.nn.MultiheadAttention:
            self.attn = Attention_block(dim, num_heads, drop, bias=qkv_bias)
        else:
            self.attn = Attention_block(
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
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
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
        Attention_block=torch.nn.MultiheadAttention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if Attention_block == torch.nn.MultiheadAttention:
            self.attn = Attention_block(dim, num_heads, drop, bias=qkv_bias)
        else:
            self.attn = Attention_block(
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
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.attn == torch.nn.MultiheadAttention:
            qkv = self.norm1(x)
            x = x + self.drop_path(
                self.gamma_1 * self.attn(qkv, qkv, qkv, need_weights=False)[0]
            )
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
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
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp1 = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_1_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True
        )
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True
        )

    def forward(self, x):
        x = (
            x
            + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            + self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        )
        x = (
            x
            + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            + self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        )
        return x


class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
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
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp1 = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = (
            x
            + self.drop_path(self.attn(self.norm1(x)))
            + self.drop_path(self.attn1(self.norm11(x)))
        )
        x = (
            x
            + self.drop_path(self.mlp(self.norm2(x)))
            + self.drop_path(self.mlp1(self.norm21(x)))
        )
        return x


class hMLP_stem(nn.Module):
    """hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=nn.SyncBatchNorm,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(
            *[
                nn.Conv2d(in_chans, embed_dim // 4, kernel_size=4, stride=4),
                norm_layer(embed_dim // 4),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2),
                norm_layer(embed_dim // 4),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
                norm_layer(embed_dim),
            ]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class vit_models(nn.Module):
    """Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        global_pool=None,
        block_layers=Block,
        Patch_layer=PatchEmbed,
        act_layer=nn.GELU,
        Attention_block=Attention,  # torch.nn.MultiheadAttention,
        Mlp_block=Mlp,
        dpr_constant=True,
        init_scale=1e-4,
        mlp_ratio_clstk=4.0,
        **kwargs,
    ):
        super().__init__()

        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block,
                    Mlp_block=Mlp_block,
                    init_values=init_scale,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

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

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = x + self.pos_embed

        x = torch.cat((cls_tokens, x), dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)

        return x


# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)


@register_model
def deit_tiny_patch1_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size,
        patch_size=1,
        embed_dim=192,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )

    return model


@register_model
def deit_tiny_patch2_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size,
        patch_size=2,
        embed_dim=192,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )

    return model


@register_model
def deit_tiny_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )

    return model


@register_model
def deit_small_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        name = "https://dl.fbaipublicfiles.com/deit/deit_3_small_" + str(img_size) + "_"
        if pretrained_21k:
            name += "21k.pth"
        else:
            name += "1k.pth"

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name, map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def deit_tiny_patch4_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size,
        patch_size=4,
        embed_dim=192,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        name = "https://dl.fbaipublicfiles.com/deit/deit_3_small_" + str(img_size) + "_"
        if pretrained_21k:
            name += "21k.pth"
        else:
            name += "1k.pth"

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name, map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def deit_medium_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        patch_size=16,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        name = (
            "https://dl.fbaipublicfiles.com/deit/deit_3_medium_" + str(img_size) + "_"
        )
        if pretrained_21k:
            name += "21k.pth"
        else:
            name += "1k.pth"

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name, map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )
    if pretrained:
        name = "https://dl.fbaipublicfiles.com/deit/deit_3_base_" + str(img_size) + "_"
        if pretrained_21k:
            name += "21k.pth"
        else:
            name += "1k.pth"

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name, map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_large_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )
    if pretrained:
        name = "https://dl.fbaipublicfiles.com/deit/deit_3_large_" + str(img_size) + "_"
        if pretrained_21k:
            name += "21k.pth"
        else:
            name += "1k.pth"

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name, map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_huge_patch14_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )
    if pretrained:
        name = "https://dl.fbaipublicfiles.com/deit/deit_3_huge_" + str(img_size) + "_"
        if pretrained_21k:
            name += "21k_v1.pth"
        else:
            name += "1k_v1.pth"

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name, map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_huge_patch14_52_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=14,
        embed_dim=1280,
        depth=52,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )

    return model


@register_model
def deit_huge_patch14_26x2_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=14,
        embed_dim=1280,
        depth=26,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block_paralx2,
        **kwargs,
    )

    return model


@register_model
def deit_Giant_48_patch14_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=14,
        embed_dim=1664,
        depth=48,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )
    return model


@register_model
def deit_giant_40_patch14_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )
    # model.default_cfg = _cfg()

    return model


# Models from Three things everyone should know about Vision Transformers (https://arxiv.org/pdf/2203.09795.pdf)


@register_model
def deit_small_patch16_36_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=36,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )

    return model


@register_model
def deit_small_patch16_36(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=36,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return model


@register_model
def deit_small_patch16_18x2_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=18,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block_paralx2,
        **kwargs,
    )

    return model


@register_model
def deit_small_patch16_18x2(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=18,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Block_paralx2,
        **kwargs,
    )

    return model


@register_model
def deit_base_patch16_18x2_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=18,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block_paralx2,
        **kwargs,
    )

    return model


@register_model
def deit_base_patch16_18x2(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=18,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Block_paralx2,
        **kwargs,
    )

    return model


@register_model
def deit_base_patch16_36x1_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=36,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **kwargs,
    )

    return model


@register_model
def deit_base_patch16_36x1(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=36,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return model


def init_random_2d_freqs(
    dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True
):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat(
            [mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1
        )
        fy = torch.cat(
            [mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def compute_mixed_cis(freqs, t_x, t_y, num_heads):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (
            (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
            .view(depth, N, num_heads, -1)
            .permute(0, 2, 1, 3)
        )
        freqs_y = (
            (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
            .view(depth, N, num_heads, -1)
            .permute(0, 2, 1, 3)
        )
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

    return freqs_cis


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]

    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_ = torch.complex(xq[..., 0], xq[..., 1])
    xk_ = torch.complex(xk[..., 0], xk[..., 1])

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    xq_out = torch.stack(((xq_ * freqs_cis).real, (xq_ * freqs_cis).imag), -1).flatten(
        3
    )
    xk_out = torch.stack(((xk_ * freqs_cis).real, (xk_ * freqs_cis).imag), -1).flatten(
        3
    )

    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class RoPEAttention(Attention):
    """Multi-head Attention block with relative position embeddings."""

    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(
            q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis
        )
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class RoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = RoPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(
            self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis)
        )
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


class rope_vit_models(vit_models):
    def __init__(self, rope_theta=100.0, rope_mixed=False, use_ape=False, **kwargs):
        super().__init__(**kwargs)

        img_size = kwargs["img_size"] if "img_size" in kwargs else 224
        patch_size = kwargs["patch_size"] if "patch_size" in kwargs else 16
        num_heads = kwargs["num_heads"] if "num_heads" in kwargs else 12
        embed_dim = kwargs["embed_dim"] if "embed_dim" in kwargs else 768
        mlp_ratio = kwargs["mlp_ratio"] if "mlp_ratio" in kwargs else 4.0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        self.use_ape = use_ape
        if not self.use_ape:
            self.pos_embed = None

        self.rope_mixed = rope_mixed
        self.num_heads = num_heads
        self.patch_size = patch_size

        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)

            freqs = []
            for i, _ in enumerate(self.blocks):
                freqs.append(
                    init_random_2d_freqs(
                        dim=embed_dim // num_heads,
                        num_heads=num_heads,
                        theta=rope_theta,
                    )
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(self.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)

            t_x, t_y = init_t_xy(
                end_x=img_size // patch_size, end_y=img_size // patch_size
            )
            self.register_buffer("freqs_t_x", t_x)
            self.register_buffer("freqs_t_y", t_y)
        else:
            self.compute_cis = partial(
                compute_axial_cis, dim=embed_dim // num_heads, theta=rope_theta
            )

            freqs_cis = self.compute_cis(
                end_x=img_size // patch_size, end_y=img_size // patch_size
            )
            self.freqs_cis = freqs_cis

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "freqs"}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_ape:
            pos_embed = self.pos_embed
            if pos_embed.shape[-2] != x.shape[-2]:
                img_size = self.patch_embed.img_size
                patch_size = self.patch_embed.patch_size
                pos_embed = pos_embed.view(
                    1,
                    (img_size[1] // patch_size[1]),
                    (img_size[0] // patch_size[0]),
                    self.embed_dim,
                ).permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed,
                    size=(H // patch_size[1], W // patch_size[0]),
                    mode="bicubic",
                    align_corners=False,
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            x = x + pos_embed

        x = torch.cat((cls_tokens, x), dim=1)

        if self.rope_mixed:
            if self.freqs_t_x.shape[0] != x.shape[1] - 1:
                t_x, t_y = init_t_xy(
                    end_x=W // self.patch_size, end_y=H // self.patch_size
                )
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)

            for i, blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis[i])
        else:
            if self.freqs_cis.shape[0] != x.shape[1] - 1:
                freqs_cis = self.compute_cis(
                    end_x=W // self.patch_size, end_y=H // self.patch_size
                )
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)

            for i, blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis)

        x = self.norm(x)
        x = x[:, 0]

        return x


# RoPE-Axial
@register_model
def rope_axial_deit_small_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=100.0,
        rope_mixed=False,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rope_axial_deit_base_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=100.0,
        rope_mixed=False,
        **kwargs,
    )
    return model


@register_model
def rope_axial_deit_large_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=100.0,
        rope_mixed=False,
        **kwargs,
    )
    return model


# RoPE-Mixed
@register_model
def rope_mixed_deit_small_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rope_mixed_deit_base_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        **kwargs,
    )
    return model


@register_model
def rope_mixed_deit_large_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        **kwargs,
    )
    return model


# RoPE-Axial + APE
@register_model
def rope_axial_ape_deit_small_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=100.0,
        rope_mixed=False,
        use_ape=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rope_axial_ape_deit_base_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=100.0,
        rope_mixed=False,
        use_ape=True,
        **kwargs,
    )
    return model


@register_model
def rope_axial_ape_deit_large_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=100.0,
        rope_mixed=False,
        use_ape=True,
        **kwargs,
    )
    return model


# RoPE-Mixed + APE
@register_model
def rope_mixed_ape_deit_small_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        use_ape=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def rope_mixed_ape_deit_base_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        use_ape=True,
        **kwargs,
    )
    return model


@register_model
def rope_mixed_ape_deit_large_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        use_ape=True,
        **kwargs,
    )
    return model


# my models
@register_model
def rope_mixed_ape_deit_small_patch1_LS(
    pretrained=False, img_size=224, pretrained_21k=False, **kwargs
):
    model = rope_vit_models(
        img_size=img_size,
        patch_size=1,
        embed_dim=192,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        rope_theta=10.0,
        rope_mixed=True,
        use_ape=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
