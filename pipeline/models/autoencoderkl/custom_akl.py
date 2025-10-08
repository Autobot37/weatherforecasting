from typing import Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from pipeline.models.autoencoderkl.distributions import DiagonalGaussianDistribution

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from pipeline.models.autoencoderkl.attention import AttentionBlock

class Upsample2D(nn.Module):
    """
    An upsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        use_conv_transpose:
        out_channels:
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states

class Downsample2D(nn.Module):
    """
    A downsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        out_channels:
        padding:
    """

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states
    
class Upsample4x2D(nn.Module):
    """
    A 4x upsampling layer composed of two sequential 2x Upsample2D layers.
    Mirrors the interface of Upsample2D.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: whether to apply a 3x3 conv after upsampling.
        use_conv_transpose: whether to use conv transpose for each 2x step.
        out_channels: output channels (if None, equal to channels).
        name: layer naming, passed to underlying Upsample2D.
    """
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        mid_channels = out_channels or channels
        # First 2x upsample
        self.up1 = Upsample2D(
            channels=channels,
            use_conv=use_conv,
            use_conv_transpose=use_conv_transpose,
            out_channels=mid_channels,
            name=name
        )
        # Second 2x upsample
        self.up2 = Upsample2D(
            channels=mid_channels,
            use_conv=use_conv,
            use_conv_transpose=use_conv_transpose,
            out_channels=out_channels or channels,
            name=name
        )

    def forward(self, hidden_states, output_size=None):
        # First pass: double size
        x = self.up1(hidden_states, output_size=None)
        # Second pass: quadruple size
        return self.up2(x, output_size=output_size)


class Downsample4x2D(nn.Module):
    """
    A 4x downsampling layer composed of two sequential 2x Downsample2D layers.
    Mirrors the interface of Downsample2D.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: whether to apply a 3x3 conv for downsampling.
        out_channels: output channels (if None, equal to channels).
        padding: padding for conv downsample.
        name: layer naming, passed to underlying Downsample2D.
    """
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        mid_channels = out_channels or channels
        # First 2x downsample
        self.down1 = Downsample2D(
            channels=channels,
            use_conv=use_conv,
            out_channels=mid_channels,
            padding=padding,
            name=name
        )
        # Second 2x downsample
        self.down2 = Downsample2D(
            channels=mid_channels,
            use_conv=use_conv,
            out_channels=out_channels or channels,
            padding=padding,
            name=name
        )

    def forward(self, hidden_states):
        # First pass: half size
        x = self.down1(hidden_states)
        # Second pass: quarter size
        return self.down2(x)

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


# unet_rl.py
def rearrange_dims(tensor):
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    if len(tensor.shape) == 3:
        return tensor[:, :, None, :]
    elif len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    else:
        raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.conv1d = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        self.mish = nn.Mish()

    def forward(self, x):
        x = self.conv1d(x)
        x = rearrange_dims(x)
        x = self.group_norm(x)
        x = rearrange_dims(x)
        x = self.mish(x)
        return x


# unet_rl.py
class ResidualTemporalBlock1D(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5):
        super().__init__()
        self.conv_in = Conv1dBlock(inp_channels, out_channels, kernel_size)
        self.conv_out = Conv1dBlock(out_channels, out_channels, kernel_size)

        self.time_emb_act = nn.Mish()
        self.time_emb = nn.Linear(embed_dim, out_channels)

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t):
        """
        Args:
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]

        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        t = self.time_emb_act(t)
        t = self.time_emb(t)
        out = self.conv_in(x) + rearrange_dims(t)
        out = self.conv_out(out)
        return out + self.residual_conv(x)


def upsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    r"""Upsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is
    a: multiple of the upsampling factor.

    Args:
        hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
        factor: Integer upsampling factor (default: 2).
        gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output: Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)

    kernel = kernel * (gain * (factor**2))
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        kernel.to(device=hidden_states.device),
        up=factor,
        pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
    )
    return output


def downsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    r"""Downsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and downsamples each image with the
    given filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the
    specified `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.

    Args:
        hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to average pooling.
        factor: Integer downsampling factor (default: 2).
        gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output: Tensor of the shape `[N, C, H // factor, W // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)

    kernel = kernel * gain
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states, kernel.to(device=hidden_states.device), down=factor, pad=((pad_value + 1) // 2, pad_value // 2)
    )
    return output


def upfirdn2d_native(tensor, kernel, up=1, down=1, pad=(0, 0)):
    up_x = up_y = up
    down_x = down_y = down
    pad_x0 = pad_y0 = pad[0]
    pad_x1 = pad_y1 = pad[1]

    _, channel, in_h, in_w = tensor.shape
    tensor = tensor.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = tensor.shape
    kernel_h, kernel_w = kernel.shape

    out = tensor.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out.to(tensor.device)  # Move back to mps if necessary
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)

def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    scale = 2,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownEncoderBlock2D":
        return DownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            scale = scale
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    scale = 2
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpDecoderBlock2D":
        return UpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            scale = scale
        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    AttentionBlock(
                        in_channels,
                        num_head_channels=attn_num_head_channels,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        scale = 2,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            if scale == 2:
                self.downsamplers = nn.ModuleList(
                    [
                        Downsample2D(
                            out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                        )
                    ]
                )
            elif scale == 4:
                self.downsamplers = nn.ModuleList(
                    [
                        Downsample4x2D(
                            out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                        )
                    ]
                )
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        scale = 2,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            if scale == 2:
                self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
            else:
                self.upsamplers = nn.ModuleList(
                    [Upsample4x2D(out_channels, use_conv=True, out_channels=out_channels)]
                )
        else:
            self.upsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down 128 -> 64 -> 16 -> 4 -> 1
        scales = [2, 2, 2, 2]

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
                scale=scales[i] if i < len(scales) else -1,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # up
        scales = [2, 2, 2, 2]

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
                scale=scales[i] if i < len(scales) else -1,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, z):
        sample = z
        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

import math

class SinusoidalPosEmb2D(nn.Module):
    def __init__(self, channels: int, height: int, width: int):
        """
        Precomputes and adds sinusoidal positional embeddings to input (B, C, H, W).
        Args:
            channels (int): number of channels (must be divisible by 4)
            height (int): fixed height
            width (int): fixed width
        """
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("Channels must be divisible by 4 for 2D sinusoidal embeddings.")

        self.channels = channels
        self.height = height
        self.width = width

        # Precompute positional embedding and save as buffer
        pe = self._build_embedding(channels, height, width)
        self.register_buffer("pe", pe, persistent=False)

    def _build_embedding(self, C, H, W):
        c_half = C // 2
        c_quarter = c_half // 2

        inv_freq_y = 1.0 / (10000 ** (torch.arange(0, c_quarter).float() / c_quarter))
        inv_freq_x = 1.0 / (10000 ** (torch.arange(0, c_quarter).float() / c_quarter))

        pos_y = torch.arange(H).unsqueeze(1).float()  # (H,1)
        pos_x = torch.arange(W).unsqueeze(1).float()  # (W,1)

        sinusoid_y = pos_y * inv_freq_y.unsqueeze(0)  # (H,Cq)
        sinusoid_x = pos_x * inv_freq_x.unsqueeze(0)  # (W,Cq)

        y_emb = torch.cat([torch.sin(sinusoid_y), torch.cos(sinusoid_y)], dim=1)  # (H, C/2)
        y_emb = y_emb.unsqueeze(1).repeat(1, W, 1)  # (H, W, C/2)

        x_emb = torch.cat([torch.sin(sinusoid_x), torch.cos(sinusoid_x)], dim=1)  # (W, C/2)
        x_emb = x_emb.unsqueeze(0).repeat(H, 1, 1)  # (H, W, C/2)

        emb = torch.cat([y_emb, x_emb], dim=2)  # (H, W, C)
        emb = emb.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return emb  # broadcast over batch

    def forward(self, x):
        """
        Adds precomputed positional embedding.
        x: (B, C, H, W)
        """
        return x + self.pe

class AutoencoderKL(nn.Module):
    r"""Variational Autoencoder (VAE) model with KL loss from the paper Auto-Encoding Variational Bayes by Diederik P. Kingma
    and Max Welling.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): TODO
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        timeseries_dim: int = 512,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.use_slicing = False

        self.latent_channels = latent_channels
        self.latent_height = sample_size // (2 ** len(down_block_types)) * 2
        self.latent_width = sample_size // (2 ** len(down_block_types)) * 2
        self.flattened_size = self.latent_channels * self.latent_height * self.latent_width
        self.pos_emb = SinusoidalPosEmb2D(
            channels=self.latent_channels,
            height=self.latent_height,
            width=self.latent_width)

        self.to_timeseries = nn.Linear(self.flattened_size, timeseries_dim)
        self.from_timeseries = nn.Linear(timeseries_dim, self.flattened_size)

    def encode(self, x: torch.FloatTensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)        
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def _decode(self, z: torch.FloatTensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously invoked, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def decode(self, z: torch.FloatTensor) -> torch.Tensor:
        z = z.reshape(z.shape[0], 64, 8, 8)
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)
        return decoded

    # def forward(
    #         self,
    #         sample: torch.FloatTensor,
    #         sample_posterior: bool = False,
    #         return_posterior: bool = False,
    #         generator: Optional[torch.Generator] = None,
    #     ) -> torch.FloatTensor:
    #     r"""
    #     Args:
    #         sample (`torch.FloatTensor`): Input sample.
    #         sample_posterior (`bool`, *optional*, defaults to `False`):
    #             Whether to sample from the posterior.
    #         return_posterior (`bool`, *optional*, defaults to `False`):
    #             Whether or not to return `posterior` along with `dec` for calculating the training loss.
    #     """
    #     x = sample
    #     posterior = self.encode(x)
    #     if sample_posterior:
    #         z = posterior.sample(generator=generator)
    #     else:
    #         z = posterior.mode()
    #     dec = self.decode(z)
    #     if return_posterior:
    #         return dec, posterior
    #     else:
    #         return dec
    def forward(self, sample: torch.FloatTensor, sample_posterior: bool = False):
        # 1. Encode to 2D latent grid
        posterior = self.encode(sample)
        z_2d = posterior.sample() if sample_posterior else posterior.mode()
        
        # 2. Add positional embedding
        z_2d_pos = self.pos_emb(z_2d)
        
        # 3. Flatten
        z_flat = z_2d_pos.view(z_2d_pos.shape[0], -1)
        
        # 4. Project to 1D "timeseries" bottleneck
        z_timeseries = self.to_timeseries(z_flat)
        
        # 5. Project back from bottleneck to flattened 2D grid size
        z_flat_reconstructed = self.from_timeseries(z_timeseries)
        
        # 6. Reshape back to 2D grid
        z_2d_reconstructed = z_flat_reconstructed.view(-1, self.latent_channels, self.latent_height, self.latent_width)
        
        # 7. Decode the reconstructed 2D grid
        reconstruction = self.decode(z_2d_reconstructed)
        
        return reconstruction, z_timeseries, posterior

if __name__ == "__main__":
    down_block_types = (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    )
    up_block_types = (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )
    model = AutoencoderKL(in_channels=1, out_channels=1, latent_channels=64, block_out_channels=(64, 128, 256, 512, 512), sample_size=128, down_block_types=down_block_types, up_block_types=up_block_types, timeseries_dim=2048)
    x = torch.randn(1, 1, 128, 128)
    out, z_t, z = model(x, sample_posterior=True)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = params / 1e6 
    print(f"Model parameters: {params_m:.2f}M")
    print(out.shape)  # Should be (1, 1, 128, 128)
    print(z.shape)  # Should be (1, 64, 8, 8)