# -*- coding: utf-8 -*-
import gc
import os
import math
from abc import abstractmethod

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .nn import avg_pool_nd, conv_nd, linear, normalization, timestep_embedding, zero_module, get_activation, \
    AttentionPooling

from xformers.ops import memory_efficient_attention  # noqa


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, encoder_out=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, encoder_out)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining a convolution is applied.
    :param dims: determines the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, dtype=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.dtype = dtype
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1, dtype=self.dtype)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode='nearest')
        else:
            if self.dtype == torch.bfloat16:
                x = x.type(torch.float32 if x.device.type == 'cpu' else torch.float16)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if self.dtype == torch.bfloat16:
                x = x.type(torch.bfloat16)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining a convolution is applied.
    :param dims: determines the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, dtype=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.dtype = dtype
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1, dtype=self.dtype)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: specified, the number of out channels.
    :param use_conv: True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines the signal is 1D, 2D, or 3D.
    :param up: True, use this block for upsampling.
    :param down: True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            activation,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            up=False,
            down=False,
            dtype=None,
            efficient_activation=False,
            scale_skip_connection=False,
    ):
        super().__init__()
        self.dtype = dtype
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.efficient_activation = efficient_activation
        self.scale_skip_connection = scale_skip_connection

        self.in_layers = nn.Sequential(
            normalization(channels, dtype=self.dtype),
            get_activation(activation),
            conv_nd(dims, channels, self.out_channels, 3, padding=1, dtype=self.dtype),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=self.dtype)
            self.x_upd = Upsample(channels, False, dims, dtype=self.dtype)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=self.dtype)
            self.x_upd = Downsample(channels, False, dims, dtype=self.dtype)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.Identity() if self.efficient_activation else get_activation(activation),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                dtype=self.dtype
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, dtype=self.dtype),
            get_activation(activation),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, dtype=self.dtype)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1, dtype=self.dtype)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1, dtype=self.dtype)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        res = self.skip_connection(x) + h
        if self.scale_skip_connection:
            res *= 0.7071  # 1 / sqrt(2), https://arxiv.org/pdf/2104.07636.pdf
        return res


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            disable_self_attention=False,
            encoder_channels=None,
            dtype=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.channels = channels
        self.disable_self_attention = disable_self_attention
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f'q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}'
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels, dtype=self.dtype)
        self.qkv = conv_nd(1, channels, channels * 3, 1, dtype=self.dtype)
        if self.disable_self_attention:
            self.qkv = conv_nd(1, channels, channels, 1, dtype=self.dtype)
        else:
            self.qkv = conv_nd(1, channels, channels * 3, 1, dtype=self.dtype)
        self.attention = QKVAttention(self.num_heads, disable_self_attention=disable_self_attention)

        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1, dtype=self.dtype)
            self.norm_encoder = normalization(encoder_channels, dtype=self.dtype)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1, dtype=self.dtype))

    def forward(self, x, encoder_out=None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if encoder_out is not None:
            # from imagen article: https://arxiv.org/pdf/2205.11487.abs
            encoder_out = self.norm_encoder(encoder_out)
            # # #
            encoder_out = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads, disable_self_attention=False):
        super().__init__()
        self.n_heads = n_heads
        self.disable_self_attention = disable_self_attention

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        if self.disable_self_attention:
            ch = width // (1 * self.n_heads)
            q, = qkv.reshape(bs * self.n_heads, ch * 1, length).split(ch, dim=1)
        else:
            assert width % (3 * self.n_heads) == 0
            ch = width // (3 * self.n_heads)
            q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            if self.disable_self_attention:
                k, v = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            else:
                ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
                k = torch.cat([ek, k], dim=-1)
                v = torch.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        # if _FORCE_MEM_EFFICIENT_ATTN:
        q, k, v = map(lambda t: t.permute(0, 2, 1).contiguous(), (q, k, v))
        a = memory_efficient_attention(q, k, v)
        a = a.permute(0, 2, 1)
        # else:
        #     weight = torch.einsum(
        #         'bct,bcs->bts', q * scale, k * scale
        #     )  # More stable with f16 than dividing afterwards
        #     weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        #     a = torch.einsum('bts,bcs->bct', weight, v)
        return a.reshape(bs, -1, length)


class UNetSplitModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines the signal is 1D, 2D, or 3D.
    :param num_classes: specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            activation,
            encoder_dim,
            att_pool_heads,
            encoder_channels,
            image_size,
            disable_self_attentions=None,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            precision='32',
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            efficient_activation=False,
            scale_skip_connection=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.encoder_channels = encoder_channels
        self.encoder_dim = encoder_dim
        self.efficient_activation = efficient_activation
        self.scale_skip_connection = scale_skip_connection
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.secondary_device = torch.device("cpu")

        # adapt attention resolutions
        if isinstance(attention_resolutions, str):
            self.attention_resolutions = []
            for res in attention_resolutions.split(','):
                self.attention_resolutions.append(image_size // int(res))
        else:
            self.attention_resolutions = attention_resolutions
        self.attention_resolutions = tuple(self.attention_resolutions)
        #

        # adapt disable self attention resolutions
        if not disable_self_attentions:
            self.disable_self_attentions = []
        elif disable_self_attentions is True:
            self.disable_self_attentions = attention_resolutions
        elif isinstance(disable_self_attentions, str):
            self.disable_self_attentions = []
            for res in disable_self_attentions.split(','):
                self.disable_self_attentions.append(image_size // int(res))
        else:
            self.disable_self_attentions = disable_self_attentions
        self.disable_self_attentions = tuple(self.disable_self_attentions)
        #

        # adapt channel mult
        if isinstance(channel_mult, str):
            self.channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(','))
        else:
            self.channel_mult = tuple(channel_mult)
        #

        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dtype = torch.float32

        self.precision = str(precision)
        self.use_fp16 = precision == '16'
        if self.precision == '16':
            self.dtype = torch.float16
        elif self.precision == 'bf16':
            self.dtype = torch.bfloat16

        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.time_embed_dim = model_channels * max(self.channel_mult)
        self.time_embed = nn.Sequential(
            linear(model_channels, self.time_embed_dim, dtype=self.dtype),
            get_activation(activation),
            linear(self.time_embed_dim, self.time_embed_dim, dtype=self.dtype),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, self.time_embed_dim)

        ch = input_ch = int(self.channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1, dtype=self.dtype))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * len(self.channel_mult)
        self.num_res_blocks = num_res_blocks

        for level, mult in enumerate(self.channel_mult):
            for _ in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        activation=activation,
                        efficient_activation=self.efficient_activation,
                        scale_skip_connection=self.scale_skip_connection,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                            dtype=self.dtype,
                            disable_self_attention=ds in self.disable_self_attentions,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype,
                            activation=activation,
                            efficient_activation=self.efficient_activation,
                            scale_skip_connection=self.scale_skip_connection,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                activation=activation,
                efficient_activation=self.efficient_activation,
                scale_skip_connection=self.scale_skip_connection,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
                dtype=self.dtype,
                disable_self_attention=ds in self.disable_self_attentions,
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                activation=activation,
                efficient_activation=self.efficient_activation,
                scale_skip_connection=self.scale_skip_connection,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        activation=activation,
                        efficient_activation=self.efficient_activation,
                        scale_skip_connection=self.scale_skip_connection,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                            dtype=self.dtype,
                            disable_self_attention=ds in self.disable_self_attentions,
                        )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=self.dtype,
                            activation=activation,
                            efficient_activation=self.efficient_activation,
                            scale_skip_connection=self.scale_skip_connection,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch, dtype=self.dtype),
            get_activation(activation),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1, dtype=self.dtype)),
        )

        self.activation_layer = get_activation(activation) if self.efficient_activation else nn.Identity()

        self.encoder_pooling = nn.Sequential(
            nn.LayerNorm(encoder_dim, dtype=self.dtype),
            AttentionPooling(att_pool_heads, encoder_dim, dtype=self.dtype),
            nn.Linear(encoder_dim, self.time_embed_dim, dtype=self.dtype),
            nn.LayerNorm(self.time_embed_dim, dtype=self.dtype)
        )

        if encoder_dim != encoder_channels:
            self.encoder_proj = nn.Linear(encoder_dim, encoder_channels, dtype=self.dtype)
        else:
            self.encoder_proj = nn.Identity()

        self.cache = None

    def collect(self):
        gc.collect()
        torch.cuda.empty_cache()

    def to(self, x, stage=1):  # 0, 1, 2, 3
        if isinstance(x, torch.device):
            secondary_device = self.secondary_device
            if stage == 1:
                self.middle_block.to(secondary_device)
                self.output_blocks.to(secondary_device)
                self.out.to(secondary_device)
                self.collect()
                self.time_embed.to(x)
                self.encoder_proj.to(x)
                self.encoder_pooling.to(x)
                self.input_blocks.to(x)
            elif stage == 2:
                self.time_embed.to(secondary_device)
                self.encoder_proj.to(secondary_device)
                self.encoder_pooling.to(secondary_device)
                self.input_blocks.to(secondary_device)
                self.output_blocks.to(secondary_device)
                self.out.to(secondary_device)
                self.collect()
                self.middle_block.to(x)
            elif stage == 3:
                self.time_embed.to(secondary_device)
                self.encoder_proj.to(secondary_device)
                self.encoder_pooling.to(secondary_device)
                self.input_blocks.to(secondary_device)
                self.middle_block.to(secondary_device)
                self.collect()
                self.output_blocks.to(x)
                self.out.to(x)
        else:
            super().to(x)

    def forward(self, x, timesteps, text_emb, timestep_text_emb=None, aug_emb=None, use_cache=False, **kwargs):
        hs = []
        self.to(self.primary_device, stage=1)
        emb = self.time_embed(timestep_embedding(timesteps.to(torch.float32), self.model_channels,
                                                 dtype=torch.float32).to(self.primary_device).to(self.dtype))

        if use_cache and self.cache is not None:
            encoder_out, encoder_pool = self.cache
        else:
            text_emb = text_emb.type(self.dtype).to(self.primary_device)
            encoder_out = self.encoder_proj(text_emb)
            encoder_out = encoder_out.permute(0, 2, 1)  # NLC -> NCL
            if timestep_text_emb is None:
                timestep_text_emb = text_emb
            encoder_pool = self.encoder_pooling(timestep_text_emb)
            if use_cache:
                self.cache = (encoder_out, encoder_pool)

        emb = emb + encoder_pool.to(emb)

        if aug_emb is not None:
            emb = emb + aug_emb.to(emb)

        emb = self.activation_layer(emb)

        h = x.type(self.dtype).to(self.primary_device)

        for module in self.input_blocks:
            h = module(h, emb, encoder_out)
            hs.append(h)

        self.to(self.primary_device, stage=2)

        h = self.middle_block(h, emb, encoder_out)

        self.to(self.primary_device, stage=3)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, encoder_out)
        h = h.type(self.dtype)
        h = self.out(h)
        return h


class SuperResUNetModel(UNetSplitModel):
    """
    A text2im model that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, low_res_diffusion, interpolate_mode='bilinear', *args, **kwargs):
        self.low_res_diffusion = low_res_diffusion
        self.interpolate_mode = interpolate_mode
        super().__init__(*args, **kwargs)

        self.aug_proj = nn.Sequential(
            linear(self.model_channels, self.time_embed_dim, dtype=self.dtype),
            get_activation(kwargs['activation']),
            linear(self.time_embed_dim, self.time_embed_dim, dtype=self.dtype),
        )

    def forward(self, x, timesteps, low_res, aug_level=None, **kwargs):
        bs, _, new_height, new_width = x.shape

        align_corners = True
        if self.interpolate_mode == 'nearest':
            align_corners = None

        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode=self.interpolate_mode, align_corners=align_corners
        )

        if aug_level is None:
            aug_steps = (np.random.random(bs) * 1000).astype(np.int64)  # uniform [0, 1)
            aug_steps = torch.from_numpy(aug_steps).to(x.device, dtype=torch.long)
        else:
            aug_steps = torch.tensor([int(aug_level * 1000)]).repeat(bs).to(x.device, dtype=torch.long)

        upsampled = self.low_res_diffusion.q_sample(upsampled, aug_steps)
        x = torch.cat([x, upsampled], dim=1)

        aug_emb = self.aug_proj(
            timestep_embedding(aug_steps, self.model_channels, dtype=self.dtype)
        )
        return super().forward(x, timesteps, aug_emb=aug_emb, **kwargs)
