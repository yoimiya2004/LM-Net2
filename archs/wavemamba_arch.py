import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
import time
from scipy.io import savemat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm import Mamba
from einops import rearrange, repeat
from functools import partial
from timm.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
from typing import Optional, Callable
import math
import numbers
from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.autograd

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.detail_enhance_net import DENet
from basicsr.archs.wavelet import DWT, IWT
from pytorch_wavelets import DWTForward


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True) if in_planes != groups else None
        self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True) if kernel_size > 1 else None
        self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True) if kernel_num > 1 else None

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = torch.sigmoid(self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size) / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = torch.softmax(self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1) / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.relu(x)

        if self.filter_fc is not None:
            filter_attention = self.get_filter_attention(x)
        else:
            filter_attention = self.skip(x)

        if self.kernel_fc is not None:
            kernel_attention = self.get_kernel_attention(x)
        else:
            kernel_attention = self.skip(x)

        if self.spatial_fc is not None:
            spatial_attention = self.get_spatial_attention(x)
        else:
            spatial_attention = self.skip(x)

        channel_attention = self.get_channel_attention(x)

        return channel_attention, filter_attention, spatial_attention, kernel_attention


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CAB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1, groups=channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
        )
        self.channel_attention = ChannelAttention(channels)

    def forward(self, x):
        return self.channel_attention(self.cab(x))


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)


class ffn(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_proj = (
            nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs),
            nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs),
            nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs),
            nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs),
        )
        self.dt_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_proj], dim=0))
        self.dt_proj_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_proj], dim=0))
        del self.dt_proj

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
               **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A = torch.log(A)
        if copies > 1:
            A = repeat(A, "d n -> r d n", r=copies)
            if merge:
                A = A.flatten(0, 1)
        A_log = nn.Parameter(A)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_proj_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_proj_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_corev0(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class LightSS2D(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
        **kwargs,   # swallow SS2D-specific args safely (dt_rank, dt_min/max, etc.)
    ):
        super().__init__()
        self.d_model = d_model
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout) if (dropout and dropout > 0) else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        if C != self.d_model:
            raise RuntimeError(f"LightSS2D expects last dim={self.d_model}, got {C}")

        # stability for mixed precision
        orig_dtype = x.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        # flatten to sequence and apply 1D Mamba
        y = x.view(B, H * W, C).contiguous()  # (B, HW, C)
        y = self.mamba(y)                     # (B, HW, C)
        y = self.dropout(y)
        y = y.view(B, H, W, C).contiguous()   # (B, H, W, C)

        if y.dtype != orig_dtype:
            y = y.to(orig_dtype)
        return y


class LFSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = LightSS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = ffn(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class MambaBlock(nn.Module):
    def __init__(self, dim, n_l_blocks=1, expand=2):
        super().__init__()
        self.l_blk = nn.Sequential(*[LFSSBlock(dim) for _ in range(n_l_blocks)])

    def forward(self, x_LL):
        b, c, h, w = x_LL.shape
        x_LL = rearrange(x_LL, "b c h w -> b (h w) c").contiguous()
        for l_layer in self.l_blk:
            x_LL = l_layer(x_LL, [h, w])
        x_LL = rearrange(x_LL, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        return x_LL


class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(Down_wt(in_chans, out_chans))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )

        self.conv_block = nn.Sequential(
            nn.Conv2d(out_chans * 2, out_chans, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, bridge):
        up = self.up(x)

        diff_h = bridge.size(2) - up.size(2)
        diff_w = bridge.size(3) - up.size(3)

        up = F.pad(up, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])

        out = torch.cat([up, bridge], dim=1)
        out = self.conv_block(out)
        return out


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2, up_mode='upconv', conv1=conv):
        super(UNet, self).__init__()

        assert up_mode in ('upconv', 'upsample')

        self.layer0 = conv1(in_chn, wf, kernel_size=3, stride=1)

        self.layer1 = UNetConvBlock(in_chans=16, out_chans=32)
        self.layer2 = UNetConvBlock(in_chans=32, out_chans=64)
        self.layer3 = UNetConvBlock(in_chans=64, out_chans=128)
        self.layer4 = UNetConvBlock(in_chans=128, out_chans=256)
        self.layer_0 = UNetUpBlock(in_chans=32, out_chans=16, up_mode=up_mode)
        self.layer_1 = UNetUpBlock(in_chans=64, out_chans=32, up_mode=up_mode)
        self.layer_2 = UNetUpBlock(in_chans=128, out_chans=64, up_mode=up_mode)
        self.layer_3 = UNetUpBlock(in_chans=256, out_chans=128, up_mode=up_mode)

        self.last = conv1(wf, in_chn, kernel_size=3, stride=1)

        self.down_group0 = MambaBlock(16, n_l_blocks=n_l_blocks[0], expand=ffn_scale)
        self.down_group1 = MambaBlock(32, n_l_blocks=n_l_blocks[0], expand=ffn_scale)
        self.down_group2 = MambaBlock(64, n_l_blocks=n_l_blocks[1], expand=ffn_scale)
        self.down_group3 = MambaBlock(128, n_l_blocks=n_l_blocks[2], expand=ffn_scale)
        self.down_group4 = MambaBlock(256, n_l_blocks=n_l_blocks[3], expand=ffn_scale)

        self.up_group4 = MambaBlock(128, n_l_blocks=n_l_blocks[3], expand=ffn_scale)
        self.up_group3 = MambaBlock(64, n_l_blocks=n_l_blocks[2], expand=ffn_scale)
        self.up_group2 = MambaBlock(32, n_l_blocks=n_l_blocks[1], expand=ffn_scale)
        self.up_group1 = MambaBlock(16, n_l_blocks=n_l_blocks[0], expand=ffn_scale)

        self.DE = DENet(3, 6)  # self.DE = DENet(3, 4) for real_haze

    def forward(self, x):

        dwt, idwt = DWT(), IWT()
        n, c, h, w = x.shape

        x_dwt = dwt(x)
        x_LL, x_high0 = x_dwt[:n, ...], x_dwt[n:, ...]

        blocks = []
        x0 = self.layer0(x_LL)
        x0 = self.down_group0(x0)
        blocks.append(x0)

        x1 = self.layer1(x0)
        x1 = self.down_group1(x1)
        blocks.append(x1)

        x2 = self.layer2(x1)
        x2 = self.down_group2(x2)
        blocks.append(x2)

        x3 = self.layer3(x2)
        x3 = self.down_group3(x3)
        blocks.append(x3)

        x4 = self.layer4(x3)
        x4 = self.down_group4(x4)

        x_3 = self.layer_3(x4, blocks[-1 - 1])
        x_3 = self.up_group4(x_3)

        x_2 = self.layer_2(x_3, blocks[-1 - 1])
        x_2 = self.up_group3(x_2)

        x_1 = self.layer_1(x_2, blocks[-2 - 1])
        x_1 = self.up_group2(x_1)

        x_0 = self.layer_0(x_1, blocks[-3 - 1])
        x_0 = self.up_group1(x_0)
        x_0 = self.last(x_0)

        output_LL = x_0 + x_LL

        x_stage1 = idwt(torch.cat((output_LL, x_high0), dim=0))

        x_final = self.DE(x_stage1)

        if self.training:
            return output_LL, x_stage1, x_final
        else:
            return x_final


@ARCH_REGISTRY.register()
class WaveMamba(nn.Module):
    def __init__(self,
                 in_chn=3,
                 wf=16,
                 n_l_blocks=[1, 2, 2, 4],
                 ffn_scale=2,
                 up_mode='upconv',
                 **kwargs):
        super(WaveMamba, self).__init__()
        self.unet = UNet(in_chn=in_chn, wf=wf, n_l_blocks=n_l_blocks, ffn_scale=ffn_scale, up_mode=up_mode)

    def forward(self, x):
        return self.unet(x)